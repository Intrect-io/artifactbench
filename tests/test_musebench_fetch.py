# Created: 2026-11-04
# Purpose: Cover scripts/musebench/fetch.py (byte-identical range fetch, ledger, resume,
#          free-space floor, unsafe names) and verify.py's acquire stage end to end on a
#          synthetic shard.
# Dependencies: pytest
# Test Status: passing
import hashlib
import io
import json
import tarfile

import pytest

from scripts.musebench import fetch as fx
from scripts.musebench import index_shards as ix
from scripts.musebench import select as sl
from scripts.musebench import verify as vf

CAPS = {"human_jamendo": 2, "human_fma_ood": 2, "clean_per_generator": 2, "co_created": 2,
        "transform_per_generator": 1}
TIERS = [{"magnet": 3}, {"human_jamendo": 1}]


def build_root(tmp_path):
    names = ["Caption/captions-jamendo.csv"]
    for i in range(6):
        names += [f"Jamendo/s{i} - a.mp3", f"OOD FMA/{i:05d}.mp3",
                  f"ai_human_co-created_music/Jamendo/s{i} - a_edited.wav",
                  f"MagNet_Generated_Music/Jamedo/s{i} - a.mp3.wav"]
        if i < 3:
            names.append(f"aac_generated_music/MagNet_Generated_Music/Jamedo/s{i} - a_aac128.m4a")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for n in names:
            data = hashlib.sha256(n.encode()).digest() * (5 + len(n) % 7)
            info = tarfile.TarInfo(n)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    blob = buf.getvalue()
    shard = {"part": 1, "repo": "r", "revision": "x", "file": "part1_001.tar",
             "size": len(blob), "lfs_sha256": None}
    (tmp_path / "index").mkdir(parents=True)
    ix.index_shard(shard, tmp_path / "index", lambda o, n: blob[o:o + n])
    (tmp_path / "index" / "shards.json").write_text(json.dumps([shard]))
    sel = sl.plan(sl.load_index(tmp_path), eval_caps=CAPS, train_tiers=TIERS,
                  train_budget_bytes=10 ** 6)
    sel["budget"] = {"budget_gb": 1.0, "free_floor_gb": 0.0}
    sl.atomic_write_json(tmp_path / "selection" / "selection.json", sel)
    return blob, {"part1_001.tar": shard}, sel


def reader_for(blob, calls=None):
    def factory(shard):
        def read(offset, length):
            if calls is not None:
                calls.append(offset)
            return blob[offset:offset + length]
        return read
    return factory


def test_fetch_writes_identical_bytes_and_ledger(tmp_path):
    blob, shards, sel = build_root(tmp_path)
    total, done, failed = fx.run(tmp_path, sel, shards, 3, 0, reader_factory=reader_for(blob))
    assert failed == [] and done == total == len(sel["rows"]) + len(sel["meta"])
    ledger = sl.load_ledger(tmp_path)
    for r in sel["rows"]:
        got = (tmp_path / "audio" / r["path"]).read_bytes()
        assert got == blob[r["data_offset"]:r["data_offset"] + r["size"]]
        assert ledger[r["path"]]["sha256"] == hashlib.sha256(got).hexdigest()
    assert (tmp_path / "meta" / "Caption" / "captions-jamendo.csv").is_file()
    assert not list(tmp_path.rglob("*.part"))


def test_fetch_resume_skips_finished_files(tmp_path):
    blob, shards, sel = build_root(tmp_path)
    fx.run(tmp_path, sel, shards, 2, 0, reader_factory=reader_for(blob))
    victim = sel["rows"][0]
    (tmp_path / "audio" / victim["path"]).write_bytes(b"truncated")
    calls = []
    total, done, failed = fx.run(tmp_path, sel, shards, 2, 0, reader_factory=reader_for(blob, calls))
    assert (total, done, failed) == (1, 1, [])
    assert calls == [victim["data_offset"]]


def test_free_space_floor_stops_the_run(tmp_path):
    blob, shards, sel = build_root(tmp_path)
    def disk_free(_):
        written = sum(f.stat().st_size for f in tmp_path.rglob("*") if f.is_file()
                      and ("audio" in f.parts or "meta" in f.parts))
        return 10_000 - written

    total, done, failed = fx.run(tmp_path, sel, shards, 1, 9_000, reader_factory=reader_for(blob),
                                 disk_free=disk_free)
    assert failed and all("floor" in f or "stopped" in f for f in failed)
    assert done < total


def test_short_read_is_an_error_and_leaves_no_file(tmp_path):
    blob, shards, sel = build_root(tmp_path)
    item = fx.work_items(sel)[1]
    f = fx.Fetcher(tmp_path, shards, 0, reader_factory=lambda s: (lambda o, n: blob[o:o + n - 1]))
    with pytest.raises(OSError, match="short read"):
        f.fetch(item)
    assert not fx.dest_for(tmp_path, item).exists()


def test_unsafe_member_name_is_refused(tmp_path):
    with pytest.raises(sl.UnsafePath):
        fx.dest_for(tmp_path, {"path": "../../etc/x.wav", "kind": "audio"})


# ----------------------------------------------------------------------- verify acquire

def fetched_root(tmp_path):
    blob, shards, sel = build_root(tmp_path)
    fx.run(tmp_path, sel, shards, 2, 0, reader_factory=reader_for(blob))
    man, missing = sl.build_manifest(sel, sl.load_ledger(tmp_path), tmp_path)
    assert missing == []
    sl.atomic_write_json(tmp_path / vf.MANIFEST_REL, man)
    return sel, man


def test_verify_acquire_passes_on_complete_fetch(tmp_path):
    sel, man = fetched_root(tmp_path)
    problems, stats = vf.verify_acquire(tmp_path, full_sha=True, free_floor_gb=0)
    assert problems == []
    assert stats["tracks"] == len(sel["rows"]) and stats["sha_checked"] == len(sel["rows"])


def test_verify_acquire_detects_corruption_and_gaps(tmp_path):
    sel, man = fetched_root(tmp_path)
    first = man["tracks"][0]
    p = tmp_path / "audio" / first["musebench_path"]
    p.write_bytes(bytes(b ^ 1 for b in p.read_bytes()))  # same size, different content
    man["tracks"] = man["tracks"][1:] + [dict(man["tracks"][1], track_id="mb_dup")]
    sl.atomic_write_json(tmp_path / vf.MANIFEST_REL, man)
    problems, _ = vf.verify_acquire(tmp_path, full_sha=True, free_floor_gb=0)
    text = "\n".join(problems)
    assert "not in the manifest" in text  # the dropped first row
    # Corrupt content is caught on rows still listed; restore row 0 to check sha path.
    man["tracks"] = [first] + man["tracks"][:-1]
    sl.atomic_write_json(tmp_path / vf.MANIFEST_REL, man)
    problems, _ = vf.verify_acquire(tmp_path, full_sha=True, free_floor_gb=0)
    assert any("sha256 mismatch" in x for x in problems)


def test_verify_acquire_detects_non_reproducible_selection(tmp_path):
    sel, _ = fetched_root(tmp_path)
    sel["seed"] = "tampered"
    sl.atomic_write_json(tmp_path / "selection" / "selection.json", sel)
    problems, _ = vf.verify_acquire(tmp_path, free_floor_gb=0)
    assert any("not reproducible" in x for x in problems)


def test_verify_acquire_enforces_free_space_floor(tmp_path):
    fetched_root(tmp_path)
    problems, _ = vf.verify_acquire(tmp_path, free_floor_gb=10 ** 9)
    assert any("below floor" in x for x in problems)

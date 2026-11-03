# Created: 2026-11-04
# Purpose: Cover scripts/musebench/verify.py index-stage checks on a synthetic shard index.
# Dependencies: pytest
# Test Status: passing
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from scripts.musebench import index_shards as ix
from scripts.musebench import verify as vf

PARTS = {1: 2, 2: 1}


def make_tar(names):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for name in names:
            data = name.encode() * 3
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def build_index(root: Path, finish=True):
    index = root / "index"
    index.mkdir(parents=True)
    contents = {
        "part1_001": ["Caption/captions-genius.csv", "Jamendo/a.mp3", "Jamendo/b.mp3"],
        "part1_002": ["Musicgen_Generated_Music/Jamedo/a.mp3.wav"],
        "part2_001": ["suno_generated_music/fake_00001_suno_0.mp3", "OOD FMA/000002.mp3"],
    }
    shards = []
    for stem, names in contents.items():
        blob = make_tar(names)
        shard = {"part": int(stem[4]), "repo": f"r{stem[4]}", "revision": f"sha{stem[4]}",
                 "file": f"{stem}.tar", "size": len(blob), "lfs_sha256": None}
        shards.append(shard)
        if finish or stem != "part2_001":
            ix.index_shard(shard, index, lambda o, n, b=blob: b[o:o + n])
    (index / "shards.json").write_text(json.dumps(shards))
    return index


def test_complete_index_passes(tmp_path):
    build_index(tmp_path)
    problems, stats = vf.verify_index(tmp_path, expected_parts=PARTS, min_audio_files=5)
    assert problems == []
    assert stats["done"] == 3 and stats["files"] == 6 and stats["audio_files"] == 5
    assert stats["top_level"]["Jamendo"] == 2 and stats["top_level"]["OOD FMA"] == 1


def test_unfinished_shard_fails(tmp_path):
    build_index(tmp_path, finish=False)
    problems, _ = vf.verify_index(tmp_path, expected_parts=PARTS, min_audio_files=1)
    assert any("part2_001: not finished" in p for p in problems)


def test_wrong_shard_count_and_too_few_files_fail(tmp_path):
    build_index(tmp_path)
    problems, _ = vf.verify_index(tmp_path, expected_parts={1: 15, 2: 15}, min_audio_files=99)
    assert any("shard list per part" in p for p in problems)
    assert any("audio files indexed" in p for p in problems)


def test_gap_and_member_count_mismatch_fail(tmp_path):
    index = build_index(tmp_path)
    jsonl = index / "part1_001.jsonl"
    lines = jsonl.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["offset"] += 512
    lines[1] = json.dumps(rec)
    jsonl.write_text("\n".join(lines[:2]) + "\n")
    problems, _ = vf.verify_index(tmp_path, expected_parts=PARTS, min_audio_files=1)
    assert any("gap/overlap" in p for p in problems)
    assert any("members 3 != 2" in p for p in problems)


def test_missing_index_fails(tmp_path):
    problems, _ = vf.verify_index(tmp_path)
    assert problems and "missing" in problems[0]


def test_cli_runs_isolated_and_reports_failure(tmp_path):
    script = Path(vf.__file__)
    out = subprocess.run([sys.executable, "-I", str(script), "--root", str(tmp_path), "index"],
                         capture_output=True, text=True)
    assert out.returncode == 1 and "[index] FAIL" in out.stdout
    out = subprocess.run([sys.executable, "-I", str(script), "--root", str(tmp_path), "run"],
                         capture_output=True, text=True)
    assert out.returncode == 1 and "no verifier yet" in out.stdout

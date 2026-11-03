# Created: 2026-11-04
# Purpose: Cover scripts/musebench/leak_map.py (identity keys, inheritance rules, report) and
#          verify.py's leak stage on synthetic reference corpora.
# Dependencies: pytest
# Test Status: passing
import json
from pathlib import Path

import pytest

from scripts.musebench import leak_map as lm
from scripts.musebench import verify as vf


def test_norm_and_jamendo_candidates_try_every_split():
    meta = {(lm.norm("Song - Part 2"), lm.norm("The Artist")): {7},
            (lm.norm("Black Peace & Der Pate"), lm.norm("DJ X")): {9}}
    assert lm.jamendo_candidates("Song - Part 2 - The Artist", meta) == {7}
    assert lm.jamendo_candidates("BLACK  Peace &amp; Der Pate - dj x", meta) == {9}
    assert lm.jamendo_candidates("Unknown - Nobody", meta) == set()


def test_fma_ids_from_names_and_lists(tmp_path):
    assert lm.fma_id("10000_Vena_Cava_50_Foot_Wave") == 10000
    assert lm.fma_id("000002.mp3") == 2
    assert lm.fma_id("abc.mp3") is None
    txt = tmp_path / "paths.txt"
    txt.write_text("/Volumes/x/fma_full/000/000002.mp3\n/Volumes/x/fma_full/010/010000.mp3\n\n")
    js = tmp_path / "files.json"
    js.write_text(json.dumps([{"filename": "012827.mp3"}, {"path": "/a/000148.mp3"}, "000005.mp3"]))
    assert lm.load_fma_ids(txt) == {2, 10000}
    assert lm.load_fma_ids(js) == {12827, 148, 5}


def test_load_mtg_meta(tmp_path):
    tsv = tmp_path / "raw.meta.tsv"
    tsv.write_text("TRACK_ID\tARTIST_ID\tALBUM_ID\tTRACK_NAME\tARTIST_NAME\tALBUM_NAME\n"
                   "track_0000214\tartist_1\talbum_1\tIntro\tDavid TMX\tPouce\n"
                   "track_0000215\tartist_1\talbum_1\tIntro\tDavid TMX\tOther\n", encoding="utf-8")
    assert lm.load_mtg_meta(tsv)[("intro", "david tmx")] == {214, 215}


def row(source, generator, variant, family, base, role="eval", size=10, tid=None, path="/x"):
    return {"track_id": tid or f"mb_{source}_{base}_{variant}", "source": source,
            "generator": generator, "variant": variant, "family": family, "base_key": base,
            "musebench_role": role, "size": size, "path": path, "musebench_path": f"{source}/{base}"}


def corpus_rows():
    return [
        row("musebench_suno", "suno", "clean", "suno/fake_00001_suno_0", "fake_00001_suno_0", size=111),
        row("musebench_suno__aac", "suno", "aac", "suno/fake_00001_suno_0", "fake_00001_suno_0"),
        row("musebench_udio", "udio", "clean", "udio/fake_00002_udio_1", "fake_00002_udio_1"),
        row("musebench_human_fma_ood", "human", "clean", "fma/10000_x", "10000_x", size=50),
        row("musebench_human_fma_ood", "human", "clean", "fma/20000_y", "20000_y"),
        row("musebench_human_jamendo", "human", "clean", "jamendo/Song - Art", "Song - Art"),
        row("musebench_co_created", "musicgen_style", "co_created", "jamendo/Song - Art", "Song - Art",
            role="train_pool"),
        row("musebench_magnet", "magnet", "clean", "jamendo/Song - Art", "Song - Art"),
        row("musebench_human_jamendo", "human", "clean", "jamendo/Other - Art", "Other - Art"),
    ]


def annotate(rows, duration=None):
    return lm.annotate(
        rows, sonics={"fake_00001_suno_0": 111}, fma_ids={10000},
        mtg_meta={("song", "art"): {5}, ("other", "art"): {6}},
        mtg_pool={5: Path("/mtg/00/5.mp3")}, mtg_all={5: Path("/mtg/00/5.mp3"), 6: Path("/mtg/50/6.mp3")},
        generators_in_training={"suno": True, "udio": True, "magnet": False},
        fma_files={10000: Path(__file__)}, duration=duration)


def test_annotate_identity_and_inheritance_rules():
    rows = corpus_rows()
    stats = annotate(rows)
    seen = {(r["source"], r["base_key"]): r["seen_in_training"] for r in rows}
    assert seen[("musebench_suno", "fake_00001_suno_0")] is True
    assert seen[("musebench_suno__aac", "fake_00001_suno_0")] is True  # transform inherits
    assert seen[("musebench_udio", "fake_00002_udio_1")] is False  # not in SONICS list
    assert seen[("musebench_human_fma_ood", "10000_x")] is True
    assert seen[("musebench_human_fma_ood", "20000_y")] is False
    assert seen[("musebench_human_jamendo", "Song - Art")] is True
    assert seen[("musebench_co_created", "Song - Art")] is True  # remix of a seen song
    assert seen[("musebench_magnet", "Song - Art")] is False  # caption-conditioned: new audio
    assert seen[("musebench_human_jamendo", "Other - Art")] is False  # MTG track outside pool
    other = [r for r in rows if r["base_key"] == "Other - Art"][0]
    assert other["in_local_mtg_jamendo"] is True and other["mtg_jamendo_ids"] == [6]
    assert stats["sonics_size_equal"] == 1 and stats["fma_id_match"] == 1
    assert stats["fma_size_differs"] == 1  # this test file is not 50 bytes
    gen = {r["generator"]: r["generator_in_training"] for r in rows}
    assert gen["suno"] is True and gen["magnet"] is False and gen["human"] is None
    assert all(r["seen_in_training"] == (r["seen_evidence"] is not None) for r in rows)


def test_duration_mismatch_is_not_counted_and_annotation_is_idempotent():
    rows = corpus_rows()
    annotate(rows, duration=lambda p: 100.0 if "mtg" in str(p) else 30.0)
    jam = [r for r in rows if r["source"] == "musebench_human_jamendo" and r["base_key"] == "Song - Art"][0]
    assert jam["seen_in_training"] is False and "not counted" in jam["leak_note"]
    annotate(rows, duration=lambda p: 30.0)
    assert jam["seen_in_training"] is True and "leak_note" not in jam


def test_per_source_counts_split_by_role():
    rows = corpus_rows()
    annotate(rows)
    c = lm.per_source_counts(rows)
    assert c["musebench_human_jamendo"] == {"total": 2, "seen": 1, "unseen": 1, "eval_seen": 1,
                                            "eval_unseen": 1, "train_pool_seen": 0,
                                            "train_pool_unseen": 0}
    assert c["musebench_co_created"]["train_pool_seen"] == 1


# ----------------------------------------------------------------------- end to end + verify

@pytest.fixture
def leak_root(tmp_path):
    root = tmp_path / "root"
    sonics = tmp_path / "sonics"
    sonics.mkdir()
    (sonics / "fake_00001_suno_0.mp3").write_bytes(b"s" * 111)
    fma_list = tmp_path / "fma.txt"
    fma_list.write_text("/v/fma_full/010/010000.mp3\n")
    mtg = tmp_path / "mtg"
    (mtg / "00").mkdir(parents=True)
    (mtg / "00" / "5.mp3").write_bytes(b"m")
    (mtg / "50").mkdir()
    (mtg / "50" / "6.mp3").write_bytes(b"m")
    tsv = tmp_path / "raw.meta.tsv"
    tsv.write_text("TRACK_ID\tARTIST_ID\tALBUM_ID\tTRACK_NAME\tARTIST_NAME\n"
                   "track_0000005\ta\tb\tSong\tArt\ntrack_0000006\ta\tb\tOther\tArt\n")
    refs = {"model": {"name": "Det v1"},
            "training_manifests": [{"name": "train set", "definition": "builder.py"}],
            "sonics_fake_dir": str(sonics), "fma_id_lists": [str(fma_list)],
            "mtg_meta_tsv": str(tsv), "mtg_audio_dir": str(mtg), "mtg_pool_folders": ["00"],
            "generators_in_training": {"suno": True}}
    (root / "leak").mkdir(parents=True)
    (root / "leak" / "references.json").write_text(json.dumps(refs))
    man = {"tracks": corpus_rows(), "metadata": {}}
    (root / "manifests").mkdir()
    (root / vf.MANIFEST_REL).write_text(json.dumps(man))
    assert lm.main(["--root", str(root), "--no-duration-check"]) == 0
    return root


def test_main_writes_report_and_verify_leak_passes(leak_root):
    report = json.loads((leak_root / lm.REPORT_REL).read_text())
    assert report["model"]["name"] == "Det v1"
    assert report["training_manifests"][0]["name"] == "train set"
    assert report["totals"] == {"seen": 5, "unseen": 4}
    assert report["reference_corpora"]["mtg_jamendo"]["pool_tracks"] == 1
    problems, stats = vf.verify_leak(leak_root)
    assert problems == [] and stats["seen"] == 5


def test_verify_leak_detects_missing_flag_and_stale_report(leak_root):
    path = leak_root / vf.MANIFEST_REL
    man = json.loads(path.read_text())
    del man["tracks"][0]["seen_in_training"]
    path.write_text(json.dumps(man))
    problems, _ = vf.verify_leak(leak_root)
    assert any("boolean seen_in_training" in p for p in problems)
    man["tracks"] = man["tracks"][1:]
    path.write_text(json.dumps(man))
    problems, _ = vf.verify_leak(leak_root)
    assert any("different manifest" in p for p in problems)


def test_verify_leak_requires_named_training_manifests(leak_root):
    rep = json.loads((leak_root / lm.REPORT_REL).read_text())
    rep["training_manifests"] = []
    (leak_root / lm.REPORT_REL).write_text(json.dumps(rep))
    problems, _ = vf.verify_leak(leak_root)
    assert any("training manifest" in p for p in problems)

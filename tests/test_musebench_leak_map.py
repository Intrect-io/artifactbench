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


# ----------------------------------------------------------------------- lineage + model key

def test_lineage_identity_kinds():
    assert lm.lineage_identity("/d/sonics/fake_songs/fake_21689_suno_1.mp3") == ("sonics", "fake_21689_suno_1")
    assert lm.lineage_identity("fake_songs/fake_00002_udio_1") == ("sonics", "fake_00002_udio_1")
    assert lm.lineage_identity("/a/fma_full/000/000708.mp3") == ("fma", 708)
    assert lm.lineage_identity("cell:mp3_128:fma_full_s1:10000") == ("fma", 10000)
    assert lm.lineage_identity("/m/mtg_jamendo/extracted/05/5.mp3") == ("mtg", 5)
    assert lm.lineage_identity("jamendo_full/00/6") == ("mtg", 6)
    assert lm.lineage_identity("/x/udio/123.mp3") is None  # a numeric stem outside FMA/Jamendo
    assert lm.lineage_identity("/x/fma_full/abc.mp3") is None
    assert lm.lineage_identity(None) is None


def test_load_lineage_reads_every_manifest_shape(tmp_path):
    sonics_file = tmp_path / "fake_00001_suno_0.mp3"
    sonics_file.write_bytes(b"s" * 111)
    split_manifest = tmp_path / "m1.json"
    split_manifest.write_text(json.dumps({
        "seed": 1, "epoch_caps": {"fma_full": 5},
        "train": [{"path": str(sonics_file), "label": "ai"},
                  {"path": "/a/fma_full/010/010000.mp3", "track_id": "fma_full/010/010000"}],
        "val": [{"path": "/m/mtg_jamendo/extracted/05/5.mp3"}],
        "codec_cells": {"mp3_128": [{"path": "/c/x.wav", "track_id": "cell:mp3_128:fma_full_s1:20000",
                                     "origin_path": "/a/fma_full/030/030000.mp3"}]}}))
    list_manifest = tmp_path / "m2.json"
    list_manifest.write_text(json.dumps(["/s/fake_00002_udio_1.mp3", {"source_path": "/j/jamendo/7.mp3"}]))
    sonics, fma, mtg, per = lm.load_lineage([split_manifest, list_manifest])
    assert sonics == {"fake_00001_suno_0": 111, "fake_00002_udio_1": lm.UNKNOWN_SIZE}
    assert fma == {10000, 20000, 30000}
    assert mtg == {5, 7}
    assert per == {str(split_manifest): 4, str(list_manifest): 2}
    empty = tmp_path / "m3.json"
    empty.write_text(json.dumps({"seed": 1}))
    with pytest.raises(ValueError):
        lm.load_lineage([empty])


def test_annotate_keyed_keeps_the_unkeyed_annotation():
    rows = corpus_rows()
    annotate(rows)  # first model: unkeyed fields
    before = [{k: r.get(k) for k in lm.LEAK_FIELDS} for r in rows]
    kw = dict(sonics={}, fma_ids={20000}, mtg_meta={("other", "art"): {6}}, mtg_pool={6: Path("/m/6.mp3")},
              mtg_all={6: Path("/m/6.mp3")}, generators_in_training={"magnet": True})
    lm.annotate_keyed(rows, "prod", **kw)
    assert [{k: r.get(k) for k in lm.LEAK_FIELDS} for r in rows] == before
    seen = {(r["source"], r["base_key"]): r["seen_in_training_prod"] for r in rows}
    assert seen[("musebench_suno", "fake_00001_suno_0")] is False
    assert seen[("musebench_human_fma_ood", "20000_y")] is True
    assert seen[("musebench_human_jamendo", "Other - Art")] is True
    gen = {r["generator"]: r["generator_in_training_prod"] for r in rows}
    assert gen["magnet"] is True and gen["suno"] is False
    lm.annotate_keyed(rows, "prod", **dict(kw, fma_ids=set()))  # idempotent: stale keyed fields replaced
    assert all(r["seen_in_training_prod"] is False or r["source"] != "musebench_human_fma_ood" for r in rows)
    with pytest.raises(ValueError):
        lm.annotate_keyed(rows, "Bad-Key", **kw)


def test_main_with_model_key_and_lineage_refs(leak_root, tmp_path):
    lineage = tmp_path / "gen1.json"
    lineage.write_text(json.dumps({"train": [{"path": "/a/fma_full/020/020000.mp3"},
                                             {"path": "/m/mtg_jamendo/extracted/50/6.mp3"}]}))
    base = json.loads((leak_root / "leak" / "references.json").read_text())
    refs = {"model": {"name": "Det v2"}, "training_manifests": [{"name": "gen1", "path": str(lineage)}],
            "lineage_manifests": [str(lineage)], "mtg_meta_tsv": base["mtg_meta_tsv"],
            "mtg_audio_dir": base["mtg_audio_dir"], "generators_in_training": {}}
    ref_path = leak_root / "leak" / "references_prod.json"
    ref_path.write_text(json.dumps(refs))
    v1_report = (leak_root / lm.REPORT_REL).read_text()
    assert lm.main(["--root", str(leak_root), "--references", str(ref_path), "--model-key", "prod",
                    "--no-duration-check"]) == 0
    rep = json.loads((leak_root / lm.report_rel("prod")).read_text())
    assert rep["seen_field"] == "seen_in_training_prod" and rep["model"]["name"] == "Det v2"
    assert rep["totals"] == {"seen": 2, "unseen": 7}  # FMA 20000 and Jamendo "Other - Art"
    assert rep["eval_seen"] == 2
    assert rep["reference_corpora"]["fma_ids"] == 1 and rep["reference_corpora"]["mtg_ids_local"] == 1
    assert (leak_root / lm.REPORT_REL).read_text() == v1_report  # the first model's report is untouched
    man = json.loads((leak_root / vf.MANIFEST_REL).read_text())
    assert man["metadata"]["leak_annotations"]["prod"]["seen_field"] == "seen_in_training_prod"
    assert vf.verify_leak(leak_root)[0] == []  # the unkeyed annotation still verifies

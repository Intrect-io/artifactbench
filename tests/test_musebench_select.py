# Created: 2026-11-04
# Purpose: Cover scripts/musebench/select.py: path classification, deterministic family-level
#          eval/train selection with a byte budget, and manifest construction.
# Dependencies: pytest
# Test Status: passing
import json

import pytest

from scripts.musebench import select as sl


@pytest.mark.parametrize("path, expect", [
    ("Jamendo/#charles - Romain ROMERO.mp3",
     dict(group="human_jamendo", label="real", variant="clean", caption_set="jamendo",
          base_key="#charles - Romain ROMERO")),
    ("OOD FMA/10000_Vena_Cava_50_Foot_Wave.mp3",
     dict(group="human_fma_ood", label="real", caption_set="fma")),
    ("ai_human_co-created_music/Jamendo/#charles - Romain ROMERO_edited.wav",
     dict(group="co_created", label="ai", variant="co_created", base_key="#charles - Romain ROMERO")),
    ("MagNet_Generated_Music/Genius-Lyrics/-_KILL.ZERO_4wUVupuI2Ko.mp3.wav",
     dict(group="magnet", generator="magnet", caption_set="genius", base_key="-_KILL.ZERO_4wUVupuI2Ko")),
    ("jasco_generated_music/Genius-lyrics/x.mp3.wav", dict(group="jasco", caption_set="genius")),
    ("magenta_generated_music/Jamedo/y - z.mp3.wav",
     dict(group="lyria_rt", generator="lyria_rt", caption_set="jamendo")),
    ("suno_generated_music/fake_00001_suno_0.mp3",
     dict(group="suno", caption_set="none", base_key="fake_00001_suno_0")),
    ("aac_generated_music/MagNet_Generated_Music/Genius-Lyrics/a_b_c_aac128.m4a",
     dict(group="magnet__aac", variant="aac", base_key="a_b_c", transform_param="aac128")),
    ("eq_generated_music/audioldm_generated_music/Jamedo/t - a_eq-cut_low.wav",
     dict(group="audioldm2__eq", base_key="t - a", transform_param="eq-cut_low")),
    ("pitch_time_generated_music/riffusion_generated_music/Jamedo/t_pt-ps+2st-ts0.9.wav",
     dict(group="riffusion__pitch_time", base_key="t", transform_param="pt-ps+2st-ts0.9")),
    ("loudness_generation_music/jasco_generated_music/Jamedo/t_lnLUFS-14.wav",
     dict(group="jasco__loudness", transform_param="lnLUFS-14")),
    ("gaussion_noised_music/suno_generated_music/root/fake_00025_suno_0_gn0.0025.mp3",
     dict(group="suno__gaussian", caption_set="none", base_key="fake_00025_suno_0")),
    ("pgd_generated_music/musicgen_generated_music/adv_t - a.mp3.wav",
     dict(group="musicgen__pgd", variant="pgd", caption_set="*", base_key="t - a")),
])
def test_classify(path, expect):
    row = sl.classify(path)
    for k, v in expect.items():
        assert row[k] == v, (k, row)


@pytest.mark.parametrize("path", ["Caption/captions-genius.csv", "Genius-Lyrics/metadata.csv",
                                  "Jamendo/segments.csv"])
def test_metadata_csvs_are_meta(path):
    assert sl.classify(path) == {"kind": "meta"}


@pytest.mark.parametrize("path", ["README.txt", "mystery_generated_music/a/b.wav",
                                  "aac_generated_music/MagNet_Generated_Music/Jamedo/no_suffix.m4a"])
def test_unknown_paths_are_not_classified(path):
    assert sl.classify(path) is None


@pytest.mark.parametrize("bad", ["/etc/passwd", "../x.wav", "a/../../x.wav", "a//b.wav",
                                 "a/./b.wav", "a\\b.wav", "", "a\0b"])
def test_safe_relpath_rejects_escapes(bad):
    with pytest.raises(sl.UnsafePath):
        sl.safe_relpath(bad)


def test_safe_relpath_accepts_odd_but_safe_names():
    assert sl.safe_relpath("OOD FMA/#x & y (1).mp3") == "OOD FMA/#x & y (1).mp3"


# ----------------------------------------------------------------------- planning

def synthetic_index(n_jam=40, n_gen=30):
    recs, off = [], 0

    def add(path, size=1000):
        nonlocal off
        recs.append({"path": path, "type": "file", "shard": "part1_001.tar",
                     "data_offset": off + 512, "size": size, "offset": off})
        off += 512 + (size + 511) // 512 * 512

    add("Caption/captions-jamendo.csv", 10)
    for i in range(n_jam):
        add(f"Jamendo/song{i:03d} - artist.mp3", 5000)
        add(f"ai_human_co-created_music/Jamendo/song{i:03d} - artist_edited.wav", 3000)
        add(f"OOD FMA/{i:05d}_x.mp3", 8000)
    for folder in ("MagNet_Generated_Music", "jasco_generated_music"):
        for i in range(n_gen):
            add(f"{folder}/Jamedo/song{i:03d} - artist.mp3.wav", 2000)
            if i % 2 == 0:  # "test split": only even songs have transforms
                add(f"aac_generated_music/{folder}/Jamedo/song{i:03d} - artist_aac128.m4a", 500)
                add(f"pgd_generated_music/{folder.lower()}/adv_song{i:03d} - artist.mp3.wav", 2000)
    for i in range(n_gen):
        add(f"suno_generated_music/fake_{i:05d}_suno_0.mp3", 900)
    return recs


CAPS = {"human_jamendo": 5, "human_fma_ood": 4, "clean_per_generator": 6, "co_created": 5,
        "transform_per_generator": 3}
TIERS = [{"magnet": 100, "jasco": 100}, {"co_created": 3}, {"human_jamendo": 100}]


def run_plan(budget=10 ** 9, seed=sl.SEED, recs=None):
    return sl.plan(recs or synthetic_index(), seed=seed, eval_caps=CAPS, train_tiers=TIERS,
                   train_budget_bytes=budget)


def test_plan_eval_caps_and_test_split_preference():
    sel = run_plan()
    assert sel["unclassified"] == []
    ev = [r for r in sel["rows"] if r["musebench_role"] == "eval"]
    counts = {g: sum(r["group"] == g for r in ev) for g in sel["expected_eval_counts"]}
    assert counts == sel["expected_eval_counts"]
    assert counts["human_jamendo"] == 5 and counts["magnet"] == 6 and counts["magnet__aac"] == 3
    # Clean eval tracks come from the families that have transforms (even song numbers).
    for r in ev:
        if r["generator"] in ("magnet", "jasco") and r["variant"] == "clean":
            assert int(r["base_key"][4:7]) % 2 == 0
    # PGD rows borrowed the caption set of their clean track.
    assert all(r["caption_set"] == "jamendo" for r in ev if r["variant"] == "pgd")


def test_plan_transforms_pair_with_clean_eval_and_roles_are_family_disjoint():
    sel = run_plan()
    ev = [r for r in sel["rows"] if r["musebench_role"] == "eval"]
    clean = {(r["generator"], r["family"]) for r in ev if r["variant"] == "clean"}
    transforms = [r for r in ev if r["variant"] in ("aac", "pgd")]
    assert transforms and all((r["generator"], r["family"]) in clean for r in transforms)
    eval_fams = {r["family"] for r in ev}
    train = [r for r in sel["rows"] if r["musebench_role"] == "train_pool"]
    assert train and not any(r["family"] in eval_fams for r in train)
    assert not any(r["variant"] in ("aac", "pgd") for r in train)  # transforms are eval-only
    assert sum(r["group"] == "co_created" for r in train) == 3


def test_plan_is_deterministic_and_seed_dependent():
    a, b = run_plan(), run_plan()
    assert sl.selection_fingerprint(a) == sl.selection_fingerprint(b)
    c = run_plan(seed="other-seed")
    assert sl.selection_fingerprint(a) != sl.selection_fingerprint(c)


def test_plan_respects_train_budget_round_robin():
    sel = run_plan(budget=9000)
    train = [r for r in sel["rows"] if r["musebench_role"] == "train_pool"]
    assert sum(r["size"] for r in train) <= 9000
    groups = [r["group"] for r in train]
    assert groups.count("magnet") == 2 and groups.count("jasco") == 2  # shared evenly
    assert sel["bytes"]["train_pool"] == sum(r["size"] for r in train)
    assert run_plan(budget=0)["bytes"]["train_pool"] == 0


# ----------------------------------------------------------------------- manifest

def test_build_manifest_uses_ledger_and_reports_missing(tmp_path):
    sel = run_plan()
    ledger = {r["path"]: {"path": r["path"], "size": r["size"], "sha256": "ab" * 32}
              for r in sel["rows"][1:]}
    man, missing = sl.build_manifest(sel, ledger, tmp_path)
    assert missing == [sel["rows"][0]["path"]]
    assert man["metadata"]["n_tracks"] == len(sel["rows"]) - 1
    t = man["tracks"][0]
    for k in ("track_id", "path", "label", "source", "generator", "variant", "caption_set",
              "musebench_role", "protocol_split", "sha256"):
        assert t[k] not in (None, "")
    assert t["path"].startswith(str(tmp_path / "audio"))
    assert t["source"].startswith("musebench_")
    assert len({x["track_id"] for x in man["tracks"]}) == len(man["tracks"])


def test_manifest_loads_with_runner_loader(tmp_path):
    from artifactbench.data.manifest import load_manifest

    sel = run_plan()
    ledger = {r["path"]: {"path": r["path"], "size": r["size"], "sha256": "0" * 64}
              for r in sel["rows"]}
    man, _ = sl.build_manifest(sel, ledger, tmp_path)
    path = tmp_path / "m.json"
    path.write_text(json.dumps(man))
    entries, by_source = load_manifest(str(path), split="all")
    assert len(entries) == len(sel["rows"])
    assert "musebench_magnet__aac" in by_source
    evals, _ = load_manifest(str(path), split="all", protocol_split="eval")
    assert len(evals) == sum(r["musebench_role"] == "eval" for r in sel["rows"])

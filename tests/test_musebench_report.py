# Created: 2026-11-04
# Purpose: Cover scripts/musebench/report.py: rates, Wilson CI, AUROC, seen/unseen splits,
#          failure accounting and Markdown rendering.
# Dependencies: pytest
# Test Status: passing
import json

import pytest

from scripts.musebench import report as rp


def test_wilson_interval_bounds():
    lo, hi = rp.wilson(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
    assert rp.wilson(0, 0) is None
    assert rp.wilson(0, 20)[0] == 0.0 and rp.wilson(20, 20)[1] == 1.0


def test_auroc_matches_definition_with_ties():
    assert rp.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert rp.auroc([0.1], [0.9]) == 0.0
    assert rp.auroc([0.5, 0.5], [0.5]) == 0.5
    pos, neg = [0.3, 0.7, 0.7, 0.9], [0.1, 0.7, 0.4]
    pairs = [(p > n) + 0.5 * (p == n) for p in pos for n in neg]
    assert rp.auroc(pos, neg) == pytest.approx(sum(pairs) / len(pairs))
    assert rp.auroc([], [0.1]) is None


def row(tid, source, label, generator, variant, seen=False):
    return {"track_id": tid, "source": source, "label": label, "generator": generator,
            "variant": variant, "seen_in_training": seen}


def cohort():
    rows, probs = [], {}

    def add(tid, prob, *a, **k):
        rows.append(row(tid, *a, **k))
        if prob is not None:
            probs[tid] = prob

    # Jamendo humans: 4 unseen (1 false positive), 1 seen (false positive)
    for i, p in enumerate([0.1, 0.2, 0.3, 0.9]):
        add(f"j{i}", p, "musebench_human_jamendo", "real", "human", "clean")
    add("js", 0.95, "musebench_human_jamendo", "real", "human", "clean", seen=True)
    # FMA: 2 called human, 1 called AI
    for i, p in enumerate([0.1, 0.4, 0.6]):
        add(f"f{i}", p, "musebench_human_fma_ood", "real", "human", "clean")
    # Clean MAGNeT (unseen): 1 of 4 missed; Suno (seen): 0 of 2 missed
    for i, p in enumerate([0.99, 0.8, 0.7, 0.2]):
        add(f"m{i}", p, "musebench_magnet", "ai", "magnet", "clean")
    for i, p in enumerate([0.9, 0.97]):
        add(f"s{i}", p, "musebench_suno", "ai", "suno", "clean", seen=True)
    # Transforms and co-created
    add("ma", 0.4, "musebench_magnet__aac", "ai", "magnet", "aac")
    add("sp", 0.6, "musebench_suno__pgd", "ai", "suno", "pgd", seen=True)
    add("cc", 0.5, "musebench_co_created", "ai", "musicgen_style", "co_created")
    # One failure, one silently missing row
    add("bad", None, "musebench_magnet", "ai", "magnet", "clean")
    add("gone", None, "musebench_jasco", "ai", "jasco", "clean")
    failures = [{"track_id": "bad", "source": "musebench_magnet", "stage": "decode"}]
    return rows, probs, failures


def test_build_report_rates_and_splits():
    rows, probs, failures = cohort()
    r = rp.build_report(rows, probs, failures)
    allx, uns, seen = (r["splits"][s] for s in ("all", "unseen", "seen"))
    assert allx["human_jamendo_fpr"]["k"] == 2 and allx["human_jamendo_fpr"]["n"] == 5
    assert uns["human_jamendo_fpr"]["rate"] == 0.25
    assert seen["human_jamendo_fpr"]["rate"] == 1.0
    assert allx["fma_ood_accuracy"]["rate"] == pytest.approx(2 / 3)
    mg = uns["per_generator_clean_fnr"]["magnet"]
    assert (mg["k"], mg["n"]) == (1, 4)
    assert mg["auroc_vs_jamendo"] == pytest.approx(rp.auroc([0.99, 0.8, 0.7, 0.2], [0.1, 0.2, 0.3, 0.9]))
    assert seen["per_generator_clean_fnr"]["suno"]["rate"] == 0.0
    assert uns["per_generator_clean_fnr"]["suno"]["n"] == 0
    assert allx["per_transform_fnr"]["aac"]["rate"] == 1.0
    assert allx["per_transform_fnr"]["pgd"]["rate"] == 0.0
    assert allx["per_transform_generator_fnr"]["suno__pgd"]["n"] == 1
    assert allx["co_created_fnr"]["rate"] == 0.0  # 0.5 >= threshold counts as AI
    assert allx["balanced_accuracy_clean"] == pytest.approx(1 - (2 / 5 + 1 / 6) / 2)
    cov = r["coverage"]
    assert (cov["eval_rows"], cov["scored"], cov["failures"], cov["missing"]) == (19, 17, 1, 1)
    assert cov["failures_by_source"] == {"musebench_magnet": 1}


def test_markdown_renders_all_sections():
    rows, probs, failures = cohort()
    md = rp.to_markdown(rp.build_report(rows, probs, failures), "T")
    for heading in ("## Headline", "## Clean AI FNR per generator", "## FNR per transform",
                    "## Paper context"):
        assert heading in md
    assert "| magnet | 25.0 [" in md or "| magnet | 20.0 [" in md


def write_run(run, probs, failures):
    (run / "artifactnet").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({"threshold": 0.5, "input_manifest_sha256": "x"}))
    (run / "artifactnet" / "track_probs.json").write_text(json.dumps(
        [{"track_id": k, "prob": v} for k, v in probs.items()]))
    (run / "artifactnet" / "inference_failures.json").write_text(json.dumps(failures))


def test_main_reads_run_dir_and_writes_outputs(tmp_path):
    rows, probs, failures = cohort()
    (tmp_path / "manifests").mkdir()
    (tmp_path / rp.EVAL_MANIFEST_REL).write_text(json.dumps({"tracks": rows}))
    run = tmp_path / "results" / "artifactnet_v94"
    write_run(run, probs, failures)
    assert rp.main(["--root", str(tmp_path)]) == 0
    rep = json.loads((run / "musebench_report.json").read_text())
    assert rep["primary"]["coverage"]["scored"] == 17 and rep["run"]["input_manifest_sha256"] == "x"
    assert rep["primary"] == rep["strict"]  # no finite-chunk result: primary falls back to strict
    assert (run / "musebench_report.md").read_text().startswith("# ArtifactNet v9.4 on MuseBench")


def test_finite_chunk_result_becomes_primary(tmp_path):
    rows, probs, failures = cohort()
    (tmp_path / "manifests").mkdir()
    (tmp_path / rp.EVAL_MANIFEST_REL).write_text(json.dumps({"tracks": rows}))
    run = tmp_path / "results" / "artifactnet_v94"
    write_run(run, probs, failures)
    recovered = dict(probs, bad=0.9)
    write_run(run / "finite_chunk", recovered, [])
    (run / "finite_chunk" / "artifactnet" / "finite_chunk_policy.json").write_text(json.dumps(
        {"track_aggregation": "median of finite", "minimum_valid_chunks": 4, "recovered": [{"track_id": "bad"}]}))
    assert rp.main(["--root", str(tmp_path)]) == 0
    rep = json.loads((run / "musebench_report.json").read_text())
    assert rep["primary"]["coverage"]["scored"] == 18 and rep["strict"]["coverage"]["scored"] == 17
    assert rep["primary_policy"]["minimum_valid_chunks"] == 4 and "recovered" not in rep["primary_policy"]
    md = (run / "musebench_report.md").read_text()
    assert "at least 4 finite chunks" in md and "## Strict sensitivity" in md

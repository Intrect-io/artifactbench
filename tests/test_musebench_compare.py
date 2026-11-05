# Created: 2026-11-05
# Purpose: Cover scripts/musebench/compare.py on synthetic reports built with report.build_report: a passing
#          gate, a failing gate, cohort and seen-field mismatches, missing metrics, malformed specs and the
#          Newcombe interval.
# Dependencies: pytest
# Test Status: passing
import json

import pytest

from scripts.musebench import compare as cp
from scripts.musebench import report as rp


def rows_and_probs(fpr_hits: int, lyria_misses: int, seen_every: int = 0):
    """20 Jamendo humans (fpr_hits called AI), 10 FMA, 20 clean lyria_rt (lyria_misses missed), 20 clean magnet."""
    rows, probs = [], {}

    def add(tid, source, label, generator, variant, prob, seen=False):
        rows.append({"track_id": tid, "source": source, "label": label, "generator": generator,
                     "variant": variant, "seen_in_training": seen})
        probs[tid] = prob

    for i in range(20):
        add(f"h{i}", rp.HUMAN_NEG, "real", "human", "clean", 0.9 if i < fpr_hits else 0.1)
    for i in range(10):
        add(f"f{i}", rp.FMA, "real", "human", "clean", 0.1)
    for i in range(20):
        add(f"l{i}", "musebench_lyria_rt", "ai", "lyria_rt", "clean", 0.1 if i < lyria_misses else 0.9,
            seen=bool(seen_every) and i % seen_every == 0)
    for i in range(20):
        add(f"m{i}", "musebench_magnet", "ai", "magnet", "clean", 0.8)
    return rows, probs


def full_report(fpr_hits=1, lyria_misses=10, run=None):
    rows, probs = rows_and_probs(fpr_hits, lyria_misses)
    body = rp.build_report(rows, probs, [], threshold=0.5)
    return {"primary": body, "strict": body,
            "run": run if run is not None else {"input_manifest_sha256": "a" * 64, "seen_field": "seen_in_training",
                                                "threshold": 0.5}}


def write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


SPEC = {"rules": [
    {"metric": "human_jamendo_fpr", "split": "all", "max": 0.10},
    {"metric": "human_jamendo_fpr", "split": "all", "max_increase": 0.02},
    {"metric": "clean_fnr", "generator": "lyria_rt", "split": "all", "min_decrease": 0.20},
    {"metric": "auroc_clean_ai_vs_jamendo", "split": "all", "min": 0.5},
]}


def run(tmp_path, base, cand, spec=None, extra=()):
    argv = ["--baseline", str(write(tmp_path, "base.json", base)),
            "--candidate", str(write(tmp_path, "cand.json", cand)), "--out", str(tmp_path / "out"), *extra]
    if spec is not None:
        argv += ["--gates", str(write(tmp_path, "gates.json", spec))]
    return cp.main(argv)


def test_passing_gates_exit_0_and_write_both_files(tmp_path):
    code = run(tmp_path, full_report(1, 10), full_report(1, 4), SPEC)
    assert code == 0
    doc = json.loads((tmp_path / "out" / "comparison.json").read_text())
    md = (tmp_path / "out" / "comparison.md").read_text()
    lyria = doc["splits"]["all"]["per_generator"]["lyria_rt"]["clean_fnr"]
    assert lyria["baseline"]["rate"] == 0.5 and lyria["candidate"]["rate"] == 0.2
    assert lyria["delta"] == pytest.approx(-0.3)
    assert lyria["baseline"]["ci95"] == list(rp.wilson(10, 20))
    lo, hi = lyria["delta_ci95"]
    assert lo < -0.3 < hi
    assert set(doc["splits"]) == {"all", "seen", "unseen"}
    assert doc["gates"]["passed"] is True and len(doc["gates"]["results"]) == 4
    assert doc["cohort"]["kind"] == "full_eval_cohort" and doc["seen_field"] == "seen_in_training"
    assert "## Gates" in md and "PASS" in md and "lyria_rt" in md


def test_failing_gate_exit_3_and_still_writes_the_comparison(tmp_path, capsys):
    code = run(tmp_path, full_report(1, 10), full_report(5, 9), SPEC)   # FPR 5 -> 25 %, lyria only -5 points
    assert code == 3
    doc = json.loads((tmp_path / "out" / "comparison.json").read_text())
    failed = {r["rule"] for r in doc["gates"]["results"] if not r["passed"]}
    assert failed == {"human_jamendo_fpr/all", "clean_fnr/all/lyria_rt"}
    assert "failed: clean_fnr/all/lyria_rt" in capsys.readouterr().err
    assert "FAIL" in (tmp_path / "out" / "comparison.md").read_text()


def test_cohort_mismatch_is_refused_with_exit_2(tmp_path, capsys):
    other = full_report(run={"input_manifest_sha256": "b" * 64, "seen_field": "seen_in_training"})
    assert run(tmp_path, full_report(), other, SPEC) == 2
    assert "cohort mismatch" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_sample_cohort_is_not_comparable_with_the_full_cohort(tmp_path, capsys):
    sample = full_report(run={"input_manifest_sha256": "a" * 64, "seen_field": "seen_in_training",
                              "cohort": {"sha256": "c" * 64, "size": 70}})
    assert run(tmp_path, full_report(), sample) == 2
    assert "cohort mismatch" in capsys.readouterr().err
    # two reports on the same sample cohort compare fine
    assert run(tmp_path, sample, sample) == 0
    assert json.loads((tmp_path / "out" / "comparison.json").read_text())["cohort"] == {
        "kind": "cohort_file", "sha256": "c" * 64, "size": 70}


def test_seen_field_mismatch_is_refused_with_exit_2(tmp_path, capsys):
    other = full_report(run={"input_manifest_sha256": "a" * 64, "seen_field": "seen_in_training_x"})
    assert run(tmp_path, full_report(), other) == 2
    assert "seen-field mismatch" in capsys.readouterr().err


def test_report_without_seen_field_key_means_the_default_field(tmp_path):
    legacy = full_report(run={"input_manifest_sha256": "a" * 64})
    assert run(tmp_path, legacy, full_report()) == 0


@pytest.mark.parametrize("rule, message", [
    ({"metric": "not_a_metric", "max": 0.1}, "metric must be one of"),
    ({"metric": "clean_fnr", "generator": "no_such_generator", "max": 0.1}, "is in neither report"),
    ({"metric": "clean_fnr", "generator": "suno", "max": 0.1}, "no rows"),          # in GENERATOR_ORDER, 0 rows
    ({"metric": "clean_fnr", "generator": "lyria_rt", "split": "seen", "max_increase": 0.0}, "no rows"),
    ({"metric": "human_jamendo_fpr"}, "needs one of"),
    ({"metric": "human_jamendo_fpr", "max": "5%"}, "must be a number"),
    ({"metric": "human_jamendo_fpr", "max": 0.1, "maximum": 1}, "unknown keys"),
    ({"metric": "human_jamendo_fpr", "split": "test", "max": 0.1}, "split must be one of"),
])
def test_missing_metric_or_malformed_rule_is_invalid_input(tmp_path, capsys, rule, message):
    assert run(tmp_path, full_report(), full_report(), {"rules": [rule]}) == 2
    assert message in capsys.readouterr().err


def test_empty_spec_and_missing_section_are_invalid(tmp_path, capsys):
    assert run(tmp_path, full_report(), full_report(), {"rules": []}) == 2
    no_primary = {"strict": full_report()["strict"], "run": full_report()["run"]}
    assert run(tmp_path, full_report(), no_primary) == 2
    assert "no 'primary' section" in capsys.readouterr().err
    # the strict section is selectable
    assert run(tmp_path, full_report(), no_primary, extra=("--section", "strict")) == 0


def test_newcombe_matches_the_published_example():
    # Newcombe (1998), method 10, example (a): 56/70 vs 48/80 -> 0.2000 [0.0524, 0.3339]
    cand, base = rp.rate(56, 70), rp.rate(48, 80)
    lo, hi = cp.newcombe(cand, base)
    assert lo == pytest.approx(0.0524, abs=5e-4) and hi == pytest.approx(0.3339, abs=5e-4)
    assert cp.newcombe(rp.rate(0, 0), base) is None


def test_example_gate_spec_is_valid_against_synthetic_reports(tmp_path):
    from pathlib import Path
    spec = json.loads((Path(__file__).resolve().parents[1] / "docs/examples/musebench_gates.example.json").read_text())
    for rule in spec["rules"]:
        if "generator" in rule:
            rule["generator"] = "lyria_rt"   # the synthetic cohort only has lyria_rt and magnet
        rule["split"] = "all"
    code = run(tmp_path, full_report(1, 10), full_report(1, 4), spec)
    assert code in (0, 3)

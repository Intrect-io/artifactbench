# Created: 2026-11-04
# Purpose: Cover scripts/musebench/import_scores.py and report.py's explicit-decision rule.
# Dependencies: pytest
# Test Status: passing
import json

import pytest

from scripts.musebench import import_scores as im
from scripts.musebench import report as rp
from test_musebench_report import cohort  # tests/ is on sys.path (rootdir mode)


def eval_rows():
    rows, _, _ = cohort()
    for r in rows:
        r["path"] = f"/a/{r['track_id']}.wav"
    return rows


def test_convert_maps_paths_decisions_and_failures():
    rows = eval_rows()
    lines = [{"path": "/a/j0.wav", "probability": 0.9, "verdict": "real"},   # rescued -> human
             {"path": "/a/m0.wav", "probability": 0.1, "verdict": "ai"},
             {"path": "/a/m1.wav", "probability": 0.5, "verdict": "unknown"},
             {"path": "/a/m2.wav", "probability": None, "verdict": "ai"},
             {"path": "/a/m3.wav", "probability": 0.7, "verdict": "ai", "error": "boom"},
             {"path": "/elsewhere.wav", "probability": 0.2, "verdict": "ai"}]
    probs, failures, unmatched = im.convert(rows, lines, decision_field="verdict")
    got = {p["track_id"]: p["decision_ai"] for p in probs}
    assert got == {"j0": False, "m0": True}
    stages = {f["track_id"]: f["stage"] for f in failures}
    assert stages["m1"] == stages["m2"] == stages["m3"] == "inference"
    assert stages["js"] == "missing" and len(failures) == len(rows) - 2
    assert unmatched == ["/elsewhere.wav"]
    with pytest.raises(ValueError, match="duplicate"):
        im.convert(rows, lines[:1] * 2, decision_field="verdict")


def test_report_uses_explicit_decisions_over_threshold():
    rows, probs, failures = cohort()
    decisions = {tid: False for tid in probs}  # call everything human
    r = rp.build_report(rows, probs, failures, threshold=0.5, decisions=decisions)
    assert r["splits"]["all"]["human_jamendo_fpr"]["k"] == 0
    assert r["splits"]["all"]["clean_ai_fnr"]["rate"] == 1.0
    assert r["decision_rule"] == "explicit per-track decision"
    # AUROC still comes from probabilities.
    assert r["splits"]["all"]["auroc_clean_ai_vs_jamendo"] == \
        rp.build_report(rows, probs, failures)["splits"]["all"]["auroc_clean_ai_vs_jamendo"]


def test_main_writes_run_dir_that_report_reads(tmp_path):
    rows = eval_rows()
    (tmp_path / "manifests").mkdir()
    (tmp_path / im.EVAL_MANIFEST_REL).write_text(json.dumps({"tracks": rows}))
    scores = tmp_path / "s.jsonl"
    scores.write_text("\n".join(json.dumps({"path": r["path"], "probability": 0.3, "verdict": "real"})
                                for r in rows) + "\n")
    run = tmp_path / "results" / "x"
    assert im.main(["--root", str(tmp_path), "--scores", str(scores), "--run-dir", str(run),
                    "--model-name", "Det", "--threshold", "0.2", "--decision-field", "verdict"]) == 0
    full = rp.build_full_report(rows, run)
    assert full["primary"]["coverage"]["scored"] == len(rows)
    assert full["primary"]["splits"]["all"]["human_jamendo_fpr"]["k"] == 0  # verdict, not 0.3 >= 0.2

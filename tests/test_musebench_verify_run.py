# Created: 2026-11-04
# Purpose: Cover verify.py's run stage: full-cohort selection, pinned model revision, every
#          track scored or failed, and a report that matches a rebuild from raw scores.
# Dependencies: pytest
# Test Status: passing
import json

import pytest

from artifactbench.models.artifactnet import DEFAULT_HF_REVISION
from scripts.musebench import report as rp
from scripts.musebench import verify as vf
from test_musebench_report import cohort  # tests/ is on sys.path (rootdir mode)


def write_result(d, probs, failures, extra=None):
    (d / "artifactnet").mkdir(parents=True)
    (d / "artifactnet" / "track_probs.json").write_text(json.dumps(
        [{"track_id": k, "prob": v} for k, v in probs.items()]))
    (d / "artifactnet" / "inference_failures.json").write_text(json.dumps(failures))
    for name, obj in (extra or {}).items():
        (d / "artifactnet" / name).write_text(json.dumps(obj))


@pytest.fixture
def run_root(tmp_path):
    rows, probs, failures = cohort()
    failures = failures + [{"track_id": "gone", "source": "musebench_jasco", "stage": "inference"}]
    (tmp_path / "manifests").mkdir()
    (tmp_path / rp.EVAL_MANIFEST_REL).write_text(json.dumps({"tracks": rows}))
    run = tmp_path / vf.RUN_REL
    write_result(run, probs, failures)
    (run / "run_manifest.json").write_text(json.dumps({
        "selected_track_ids": [r["track_id"] for r in rows], "models": ["artifactnet"],
        "n_per_source": 0, "threshold": 0.5}))
    (run / "summary.json").write_text(json.dumps(
        {"models": [{"name": "ArtifactNet v9.4", "provenance": {"hf_revision": DEFAULT_HF_REVISION}}]}))
    write_result(run / "finite_chunk", dict(probs, gone=0.8), failures[:1],
                 {"finite_chunk_policy.json": {"track_aggregation": "median", "minimum_valid_chunks": 4}})
    assert rp.main(["--root", str(tmp_path)]) == 0
    return tmp_path


def test_verify_run_passes(run_root):
    problems, stats = vf.verify_run(run_root)
    assert problems == []
    assert stats["scored_primary"] == 18 and stats["failures_strict"] == 2


def test_verify_run_rejects_partial_cohort_and_wrong_revision(run_root):
    run = run_root / vf.RUN_REL
    rm = json.loads((run / "run_manifest.json").read_text())
    rm["selected_track_ids"] = rm["selected_track_ids"][:-1]
    rm["n_per_source"] = 2
    (run / "run_manifest.json").write_text(json.dumps(rm))
    (run / "summary.json").write_text(json.dumps({"models": [{"provenance": {"hf_revision": "main"}}]}))
    text = "\n".join(vf.verify_run(run_root)[0])
    assert "exactly the eval cohort" in text and "n_per_source" in text and "pinned" in text


def test_verify_run_detects_unaccounted_track_and_stale_report(run_root):
    run = run_root / vf.RUN_REL
    probs = json.loads((run / "artifactnet" / "track_probs.json").read_text())
    (run / "artifactnet" / "track_probs.json").write_text(json.dumps(probs[1:]))
    text = "\n".join(vf.verify_run(run_root)[0])
    assert "strict: scored + failed" in text
    assert "differs from a rebuild" in text

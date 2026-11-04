# Created: 2026-11-04
# Purpose: Cover verify.py's production stage and report.py --seen-field on a synthetic data root:
#          raw score lines -> import_scores.py -> model-keyed leak annotation -> report split by it.
# Dependencies: pytest
# Test Status: passing
import json

import pytest

from scripts.musebench import import_scores as isc
from scripts.musebench import leak_map as lm
from scripts.musebench import report as rp
from scripts.musebench import verify as vf
from test_musebench_report import cohort  # tests/ is on sys.path (rootdir mode)

KEY = "prod"
FIELD = "seen_in_training_" + KEY


def build_root(tmp_path):
    rows, probs, _ = cohort()
    root = tmp_path / "root"
    (root / "manifests").mkdir(parents=True)
    eval_rows = []
    for r in rows:
        r = dict(r, path=f"/audio/{r['track_id']}.mp3", musebench_role="eval",
                 musebench_path=f"x/{r['track_id']}", seen_evidence="v1" if r["seen_in_training"] else None)
        eval_rows.append(r)
    (root / rp.EVAL_MANIFEST_REL).write_text(json.dumps({"tracks": eval_rows}))
    # Runtime manifest = eval rows + one train_pool row; the second model saw only the FMA tracks.
    runtime = [dict(r) for r in eval_rows] + [dict(eval_rows[0], track_id="tp", musebench_role="train_pool",
                                                   musebench_path="x/tp", path="/audio/tp.mp3")]
    for r in runtime:
        r[FIELD] = r["source"] == "musebench_human_fma_ood"
        r["seen_evidence_" + KEY] = "FMA id" if r[FIELD] else None
    (root / vf.MANIFEST_REL).write_text(json.dumps({"tracks": runtime, "metadata": {}}))
    report = {"model": {"name": "Prod"}, "seen_field": FIELD, "training_manifests": [{"name": "gen1"}],
              "per_source": lm.per_source_counts(runtime, FIELD),
              "eval_seen": sum(1 for r in runtime if r[FIELD] and r["musebench_role"] == "eval"),
              "manifest_fingerprint": lm.manifest_fingerprint(runtime)}
    (root / "leak").mkdir()
    (root / lm.report_rel(KEY)).write_text(json.dumps(report))
    # Raw scores from an external pipeline: one line per eval path, one 'unknown' verdict, the rest scored.
    run = root / "results" / "prod_run"
    run.mkdir(parents=True)
    lines = []
    for r in eval_rows:
        p = probs.get(r["track_id"], 0.5)
        verdict = "unknown" if r["track_id"] == "bad" else ("ai" if p >= 0.5 else "real")
        lines.append({"path": r["path"], "probability": p, "verdict": verdict, "error": None})
    (run / vf.PRODUCTION_SCORES).write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert isc.main(["--root", str(root), "--scores", str(run / vf.PRODUCTION_SCORES), "--run-dir", str(run),
                     "--model-name", "prod", "--threshold", "0.5", "--decision-field", "verdict"]) == 0
    (root / vf.PRODUCTION_SPEC_REL).write_text(json.dumps({"run_dir": "results/prod_run", "model_key": KEY}))
    assert rp.main(["--root", str(root), "--run-dir", str(run), "--seen-field", FIELD]) == 0
    return root, run


@pytest.fixture
def prod_root(tmp_path):
    return build_root(tmp_path)


def test_report_splits_by_the_model_specific_field(prod_root):
    _, run = prod_root
    saved = json.loads((run / "musebench_report.json").read_text())
    assert saved["run"]["seen_field"] == FIELD
    cov = saved["primary"]["coverage"]
    assert cov["seen_rows"] == 3  # the three FMA tracks, not the first model's seen rows
    assert saved["primary"]["splits"]["seen"]["fma_ood_accuracy"]["n"] == 3
    assert saved["primary"]["splits"]["seen"]["human_jamendo_fpr"]["n"] == 0


def test_attach_seen_field_joins_from_runtime_and_rejects_missing(prod_root):
    root, _ = prod_root
    eval_rows = json.loads((root / rp.EVAL_MANIFEST_REL).read_text())["tracks"]
    assert all(FIELD not in r for r in eval_rows)  # the consumed eval manifest is never rewritten
    joined = rp.attach_seen_field(eval_rows, root, FIELD)
    assert sum(r[FIELD] for r in joined) == 3
    with pytest.raises(ValueError):
        rp.attach_seen_field(eval_rows, root, "seen_in_training_other")


def test_verify_production_passes(prod_root):
    root, _ = prod_root
    problems, stats = vf.verify_production(root)
    assert problems == []
    assert stats["eval_seen"] == 3 and stats["failures"] == 1 and stats["scored"] == 18 and stats["seen_field"] == FIELD


def test_verify_production_detects_tampering(prod_root):
    root, run = prod_root
    lines = (run / vf.PRODUCTION_SCORES).read_text().splitlines()
    (run / vf.PRODUCTION_SCORES).write_text("\n".join(lines[1:]) + "\n")
    tp = json.loads((run / "artifactnet" / "track_probs.json").read_text())
    del tp[0]["decision_ai"]
    (run / "artifactnet" / "track_probs.json").write_text(json.dumps(tp))
    man = json.loads((root / vf.MANIFEST_REL).read_text())
    for t in man["tracks"]:
        if t["source"] == "musebench_human_jamendo":
            t[FIELD], t["seen_evidence_" + KEY] = True, "late"
    (root / vf.MANIFEST_REL).write_text(json.dumps(man))
    text = "\n".join(vf.verify_production(root)[0])
    assert "has 18 rows, eval cohort has 19" in text
    assert "scores_sha256" in text
    assert "decision_ai" in text
    assert "per-source counts differ" in text
    assert "differs from a rebuild" in text


def test_verify_production_requires_spec(tmp_path):
    problems, _ = vf.verify_production(tmp_path)
    assert problems and "production_run.json" in problems[0]

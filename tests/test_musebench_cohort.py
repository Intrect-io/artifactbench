# Created: 2026-11-05
# Purpose: Cover --cohort in scripts/musebench/import_scores.py and report.py (scripts/musebench/cohort.py).
# Dependencies: pytest
# Test Status: passing
import json

import pytest

from scripts.musebench import import_scores as im
from scripts.musebench import report as rp
from test_musebench_report import cohort as synthetic  # tests/ is on sys.path (rootdir mode)

COHORT = im.cohort


def eval_rows():
    rows, _, _ = synthetic()
    for r in rows:
        r["path"] = f"/a/{r['track_id']}.wav"
    return rows


def write_root(tmp_path, rows):
    (tmp_path / "manifests").mkdir()
    (tmp_path / im.EVAL_MANIFEST_REL).write_text(json.dumps({"tracks": rows}))
    scores = tmp_path / "s.jsonl"
    # the scorer ran the whole eval set; "gone" has no score line
    scores.write_text("".join(json.dumps({"path": r["path"], "probability": 0.3}) + "\n"
                              for r in rows if r["track_id"] != "gone"))
    return scores


def run_import(tmp_path, scores, run, *extra):
    return im.main(["--root", str(tmp_path), "--scores", str(scores), "--run-dir", str(run),
                    "--model-name", "Det", "--threshold", "0.5", *extra])


@pytest.mark.parametrize("form", ["list", "object"])
def test_cohort_file_forms_and_fingerprint(tmp_path, form):
    ids = ["m0", "j1", "s0"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(ids if form == "list" else {"track_ids": ids, "note": "pilot"}))
    assert COHORT.load(p) == ids
    info = COHORT.info(p, ids)
    assert info["size"] == 3 and info["sha256"] == COHORT.fingerprint(list(reversed(ids)))   # order-free


@pytest.mark.parametrize("content", [[], ["a", "a"], {"ids": ["a"]}, [1, 2], "x"])
def test_malformed_cohorts_are_rejected(tmp_path, content):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        COHORT.load(p)


def test_unknown_cohort_id_is_an_error(tmp_path):
    rows = eval_rows()
    with pytest.raises(ValueError, match="not in the eval manifest"):
        COHORT.restrict(rows, ["m0", "nope"])
    scores = write_root(tmp_path, rows)
    c = tmp_path / "c.json"
    c.write_text(json.dumps(["m0", "nope"]))
    with pytest.raises(ValueError):
        run_import(tmp_path, scores, tmp_path / "run", "--cohort", str(c))


def test_import_and_report_on_a_cohort(tmp_path):
    rows = eval_rows()
    scores = write_root(tmp_path, rows)
    ids = ["j0", "j3", "m0", "m3", "gone"]                       # "gone" has no score line -> missing failure
    c = tmp_path / "pilot.json"
    c.write_text(json.dumps({"track_ids": ids}))
    run = tmp_path / "run"
    assert run_import(tmp_path, scores, run, "--cohort", str(c)) == 0   # scores outside the cohort: not unmatched
    rm = json.loads((run / "run_manifest.json").read_text())
    assert rm["selected_track_ids"] == [r["track_id"] for r in rows if r["track_id"] in ids]
    assert rm["cohort"] == {"path": str(c), "size": 5, "sha256": COHORT.fingerprint(ids)}
    probs = json.loads((run / "artifactnet" / "track_probs.json").read_text())
    failures = json.loads((run / "artifactnet" / "inference_failures.json").read_text())
    assert len(probs) + len(failures) == len(ids)              # scored + failed == cohort size
    assert rp.main(["--root", str(tmp_path), "--run-dir", str(run), "--cohort", str(c)]) == 0
    rep = json.loads((run / "musebench_report.json").read_text())
    cov = rep["primary"]["coverage"]
    assert (cov["eval_rows"], cov["scored"], cov["failures"], cov["missing"]) == (5, 4, 1, 0)
    assert rep["run"]["cohort"]["sha256"] == COHORT.fingerprint(ids)
    assert rep["primary"]["splits"]["all"]["human_jamendo_fpr"]["n"] == 2


def test_report_refuses_a_cohort_mismatch_with_the_run(tmp_path, capsys):
    rows = eval_rows()
    scores = write_root(tmp_path, rows)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(["m0", "m1"]))
    b.write_text(json.dumps(["m0", "m2"]))
    run = tmp_path / "run"
    assert run_import(tmp_path, scores, run, "--cohort", str(a)) == 0
    assert rp.main(["--root", str(tmp_path), "--run-dir", str(run), "--cohort", str(b)]) == 2
    assert rp.main(["--root", str(tmp_path), "--run-dir", str(run)]) == 2          # cohort run needs --cohort
    assert "pass --cohort" in capsys.readouterr().err


def test_full_cohort_report_is_unchanged_without_cohort(tmp_path):
    rows = eval_rows()
    scores = write_root(tmp_path, rows)
    run = tmp_path / "run"
    assert run_import(tmp_path, scores, run) == 0
    rm = json.loads((run / "run_manifest.json").read_text())
    assert "cohort" not in rm
    assert rp.main(["--root", str(tmp_path), "--run-dir", str(run)]) == 0
    rep = json.loads((run / "musebench_report.json").read_text())
    assert "cohort" not in rep["run"]
    expect = json.loads(json.dumps(rp.build_full_report(rows, run)))   # pre-cohort code path, JSON round trip
    for key in ("strict", "primary", "primary_policy"):
        assert rep[key] == expect[key]
    assert rep["primary"]["coverage"]["eval_rows"] == len(rows)

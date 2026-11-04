#!/usr/bin/env python3
# Created: 2026-11-04
# Purpose: Turn per-file scores from an external scorer (JSONL: path, probability, verdict, error)
#          into the run-directory layout report.py reads, so detectors that do not run inside
#          artifactbench.bench can be reported on the same MuseBench eval cohort.
# Dependencies: standard library only.
# Test Status: tests/test_musebench_import_scores.py
"""Import external MuseBench scores.

    python scripts/musebench/import_scores.py --root ~/data/musebench \\
        --scores results.jsonl --run-dir ~/data/musebench/results/<name> \\
        --model-name "<detector>" --threshold 0.2 --decision-field verdict

Each JSONL line must carry ``path`` (as listed in the eval manifest) and ``probability``.
With ``--decision-field``, a value equal to ``--ai-value`` marks the track as called AI and
any value in ``--fail-values`` (or a missing probability / an ``error``) makes it a recorded
failure. Without it, report.py applies ``probability >= threshold``.
"""
from __future__ import annotations

import os as _os
import sys as _sys

# This directory holds select.py, which shadows the stdlib ``select`` module when the
# directory is on sys.path (``python scripts/musebench/x.py`` without -I). Drop it.
if _sys.path and _os.path.abspath(_sys.path[0] or ".") == _os.path.dirname(_os.path.abspath(__file__)):
    _sys.path.pop(0)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

DEFAULT_ROOT = Path(os.environ.get("MUSEBENCH_ROOT", "~/data/musebench")).expanduser()
EVAL_MANIFEST_REL = Path("manifests") / "musebench_eval_manifest.json"


def convert(eval_rows: list[dict], score_lines: list[dict], *, decision_field: str | None,
            ai_value: str = "ai", fail_values: tuple[str, ...] = ("unknown", "unreliable"),
            ) -> tuple[list[dict], list[dict], list[str]]:
    """Return (track_probs, failures, unmatched score paths)."""
    by_path = {r["path"]: r for r in eval_rows}
    seen: dict[str, dict] = {}
    unmatched = []
    for line in score_lines:
        row = by_path.get(line.get("path"))
        if row is None:
            unmatched.append(line.get("path"))
            continue
        if row["track_id"] in seen:
            raise ValueError(f"duplicate score for {line['path']!r}")
        seen[row["track_id"]] = line
    probs, failures = [], []
    for row in eval_rows:
        line = seen.get(row["track_id"])
        base = {"track_id": row["track_id"], "source": row["source"]}
        if line is None:
            failures.append({**base, "stage": "missing", "error": "no score line for this path"})
            continue
        p = line.get("probability")
        decision = line.get(decision_field) if decision_field else None
        if line.get("error") or p is None or (decision_field and decision in fail_values):
            failures.append({**base, "stage": "inference",
                             "error": line.get("error") or f"{decision_field}={decision}, probability={p}"})
            continue
        entry = {**base, "prob": float(p), "label": 1 if row["label"] == "ai" else 0}
        if decision_field:
            entry["decision_ai"] = decision == ai_value
        probs.append(entry)
    return probs, failures, unmatched


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Import external per-file scores for report.py.")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--scores", type=Path, required=True, help="JSONL with path/probability/...")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model-dir", default="artifactnet")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--decision-field", default=None)
    ap.add_argument("--ai-value", default="ai")
    ap.add_argument("--fail-values", nargs="*", default=["unknown", "unreliable"])
    args = ap.parse_args(argv)
    man_path = args.root / EVAL_MANIFEST_REL
    eval_rows = json.loads(man_path.read_text(encoding="utf-8"))["tracks"]
    with open(args.scores, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    probs, failures, unmatched = convert(eval_rows, lines, decision_field=args.decision_field,
                                         ai_value=args.ai_value, fail_values=tuple(args.fail_values))
    out = args.run_dir / args.model_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "track_probs.json").write_text(json.dumps(probs) + "\n", encoding="utf-8")
    (out / "inference_failures.json").write_text(json.dumps(failures, indent=1) + "\n", encoding="utf-8")
    run_manifest = {
        "imported_from": str(args.scores),
        "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
        "input_manifest": str(man_path),
        "input_manifest_sha256": hashlib.sha256(man_path.read_bytes()).hexdigest(),
        "models": [args.model_dir], "model_name": args.model_name,
        "threshold": args.threshold, "decision_field": args.decision_field,
        "n_per_source": 0, "selected_track_ids": [r["track_id"] for r in eval_rows],
    }
    (args.run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=1) + "\n",
                                                    encoding="utf-8")
    print(f"imported {len(probs)} scores, {len(failures)} failures, {len(unmatched)} unmatched paths")
    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())

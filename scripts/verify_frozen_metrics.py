#!/usr/bin/env python3
"""Independent point-estimate recalculation for the frozen result bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def calculate(rows, threshold):
    labels = np.array([row["label"] for row in rows], dtype=int)
    probs = np.array([row["prob"] for row in rows], dtype=float)
    pred = probs >= threshold
    tp = int(np.sum(pred & (labels == 1)))
    tn = int(np.sum(~pred & (labels == 0)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return {
        "n": len(rows),
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, pred)),
        "balanced_accuracy": (tpr + 1 - fpr) / 2,
        "tpr": tpr,
        "fpr": fpr,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def independently_choose_threshold(rows, max_fpr):
    """Recompute the calibration rule without importing the analysis module."""
    labels = np.array([row["label"] for row in rows], dtype=int)
    probs = np.array([row["prob"] for row in rows], dtype=float)
    candidates = list(np.unique(probs)) + [np.nextafter(np.max(probs), np.inf)]
    feasible = []
    for threshold in candidates:
        pred = probs >= threshold
        tp = int(np.sum(pred & (labels == 1)))
        fn = int(np.sum(~pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        tn = int(np.sum(~pred & (labels == 0)))
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        if fpr <= max_fpr + 1e-12:
            feasible.append((tpr, -float(threshold), float(threshold)))
    if not feasible:
        raise ValueError("No independently feasible calibration threshold")
    return max(feasible)[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--result", action="append", required=True, metavar="MODEL=RESULT_DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    manifest_rows = {row["track_id"]: row for row in manifest["tracks"]}
    test_ids = {
        row["track_id"] for row in manifest["tracks"] if row["protocol_split"] == "test"
    }
    metrics = json.loads(args.metrics.read_text())
    result_dirs = dict(spec.split("=", 1) for spec in args.result)
    all_scores = {}
    for model, directory in result_dirs.items():
        rows = json.loads((Path(directory) / model / "track_probs.json").read_text())
        all_scores[model] = {row["track_id"]: row for row in rows}
    scores = {
        model: {track_id: row for track_id, row in model_scores.items() if track_id in test_ids}
        for model, model_scores in all_scores.items()
    }
    common = set.intersection(*(set(model_scores) for model_scores in scores.values()))

    checks = {}
    passed = True
    for model, model_scores in scores.items():
        threshold = metrics["models"][model]["threshold_rule"]["selected_threshold"]
        calibration_rows = [
            row for track_id, row in all_scores[model].items()
            if manifest_rows[track_id]["protocol_split"] == "calibration"
        ]
        recalculated_threshold = independently_choose_threshold(
            calibration_rows, metrics["protocol"]["max_calibration_fpr"]
        )
        recalculated = calculate([model_scores[track_id] for track_id in sorted(common)], threshold)
        recorded = metrics["paired_common_test"]["per_model"][model]["metrics"]
        differences = {}
        differences["selected_threshold"] = abs(recalculated_threshold - threshold)
        for key in ("n", "auroc", "auprc", "f1", "balanced_accuracy", "tpr", "fpr"):
            differences[key] = abs(float(recalculated[key]) - float(recorded[key]))
        differences["confusion_match"] = recalculated["confusion"] == recorded["confusion"]
        model_pass = all(value <= 1e-12 for key, value in differences.items() if key != "confusion_match") and differences["confusion_match"]
        passed = passed and model_pass
        checks[model] = {
            "status": "pass" if model_pass else "fail",
            "recalculated": recalculated,
            "recalculated_threshold": recalculated_threshold,
            "recorded": recorded,
            "differences": differences,
        }
    report = {
        "status": "pass" if passed else "fail",
        "n_common_test_tracks": len(common),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "n_common_test_tracks": len(common)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

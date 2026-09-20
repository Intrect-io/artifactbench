#!/usr/bin/env python3
"""Analyze frozen ArtifactBench v2 results without tuning on sealed test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

SPLITS = ("calibration", "validation", "test")


def load_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.sum((y == 1) & (pred == 1))),
        "tn": int(np.sum((y == 0) & (pred == 0))),
        "fp": int(np.sum((y == 0) & (pred == 1))),
        "fn": int(np.sum((y == 1) & (pred == 0))),
    }


def point_metrics(rows: list[dict], threshold: float) -> dict[str, float | int | dict]:
    y = np.asarray([r["label"] for r in rows], dtype=int)
    p = np.asarray([r["prob"] for r in rows], dtype=float)
    pred = (p >= threshold).astype(int)
    cm = confusion(y, pred)
    pos = cm["tp"] + cm["fn"]
    neg = cm["tn"] + cm["fp"]
    return {
        "n": len(rows),
        "n_ai": int(np.sum(y == 1)),
        "n_real": int(np.sum(y == 0)),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, p)) if np.any(y == 1) else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": (
            float(((cm["tp"] / pos) + (cm["tn"] / neg)) / 2)
            if pos and neg
            else None
        ),
        "tpr": cm["tp"] / pos if pos else None,
        "fpr": cm["fp"] / neg if neg else None,
        "confusion": cm,
    }


def failure_as_error_metrics(
    scored_rows: list[dict], threshold: float, missing_rows: list[dict]
) -> dict[str, float | int | dict]:
    """Classification-only lower bound; do not fabricate scores for AUROC/AUPRC."""
    scored = point_metrics(scored_rows, threshold)
    cm = dict(scored["confusion"])
    for row in missing_rows:
        if row["label"] == "ai":
            cm["fn"] += 1
        else:
            cm["fp"] += 1
    pos = cm["tp"] + cm["fn"]
    neg = cm["tn"] + cm["fp"]
    precision_denominator = cm["tp"] + cm["fp"]
    precision = cm["tp"] / precision_denominator if precision_denominator else 0.0
    recall = cm["tp"] / pos if pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_attempted": len(scored_rows) + len(missing_rows),
        "n_scored": len(scored_rows),
        "n_failures_imputed_as_errors": len(missing_rows),
        "f1": f1,
        "balanced_accuracy": ((cm["tp"] / pos) + (cm["tn"] / neg)) / 2,
        "tpr": cm["tp"] / pos,
        "fpr": cm["fp"] / neg,
        "confusion": cm,
        "note": "Classification lower bound only; no synthetic score is created for AUROC or AUPRC.",
    }


def choose_threshold(calibration_rows: list[dict], max_fpr: float) -> tuple[float, dict]:
    """Use calibration only: maximize TPR subject to FPR, then lowest threshold."""
    probs = np.asarray([r["prob"] for r in calibration_rows], dtype=float)
    candidates = sorted(set(float(x) for x in probs))
    candidates.append(float(np.nextafter(np.max(probs), np.inf)))
    feasible: list[tuple[float, float, float, dict]] = []
    for threshold in candidates:
        metrics = point_metrics(calibration_rows, threshold)
        if metrics["fpr"] is not None and metrics["fpr"] <= max_fpr + 1e-12:
            feasible.append((float(metrics["tpr"]), -threshold, threshold, metrics))
    if not feasible:
        raise ValueError("No calibration threshold satisfies the FPR constraint")
    _, _, threshold, metrics = max(feasible, key=lambda x: (x[0], x[1]))
    return threshold, metrics


def bootstrap_ci(
    rows: list[dict], threshold: float, *, n_boot: int, seed: int
) -> dict[str, list[float] | int]:
    """Stratified lineage bootstrap; one draw represents one independent lineage."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["lineage_id"], []).append(row)
    by_label: dict[int, list[str]] = {0: [], 1: []}
    for lineage, members in groups.items():
        labels = {m["label"] for m in members}
        if len(labels) != 1:
            raise ValueError(f"Mixed-label lineage: {lineage}")
        by_label[labels.pop()].append(lineage)

    rng = np.random.default_rng(seed)
    names = ("auroc", "auprc", "f1", "balanced_accuracy", "tpr", "fpr")
    values = {name: [] for name in names}
    for _ in range(n_boot):
        sampled: list[dict] = []
        for label in (0, 1):
            lineages = by_label[label]
            picks = rng.choice(lineages, size=len(lineages), replace=True)
            for lineage in picks:
                sampled.extend(groups[str(lineage)])
        metrics = point_metrics(sampled, threshold)
        for name in names:
            value = metrics[name]
            if value is not None:
                values[name].append(float(value))
    return {
        "replicates": n_boot,
        **{
            name: [
                float(np.quantile(vals, 0.025)),
                float(np.quantile(vals, 0.975)),
            ]
            for name, vals in values.items()
        },
    }


def per_source(rows: list[dict], threshold: float) -> dict[str, dict]:
    sources: dict[str, list[dict]] = {}
    for row in rows:
        sources.setdefault(row["source"], []).append(row)
    return {source: point_metrics(items, threshold) for source, items in sorted(sources.items())}


def cohort_group(source: str) -> str:
    if source.startswith("aime_"):
        return "aime_ai"
    if source.startswith("suno_"):
        return "suno_supplementary_ai"
    if source.startswith("udio_"):
        return "udio_supplementary_ai"
    if source == "fma_hardneg":
        return "fma_real"
    if source == "youtube_hardneg":
        return "web_real"
    raise ValueError(f"Unmapped source group: {source}")


def per_cohort_group(rows: list[dict], threshold: float) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(cohort_group(row["source"]), []).append(row)
    return {group: point_metrics(items, threshold) for group, items in sorted(groups.items())}


def model_analysis(
    name: str,
    result_dir: Path,
    manifest_rows: dict[str, dict],
    *,
    max_fpr: float,
    n_boot: int,
    seed: int,
) -> tuple[dict, dict[str, dict]]:
    score_path = result_dir / name / "track_probs.json"
    failure_path = result_dir / name / "inference_failures.json"
    scores = load_json(score_path)
    failures = load_json(failure_path) if failure_path.exists() else []
    joined: dict[str, dict] = {}
    for score in scores:
        track_id = score["track_id"]
        meta = manifest_rows[track_id]
        joined[track_id] = {
            **score,
            "lineage_id": meta["lineage_id"],
            "protocol_split": meta["protocol_split"],
        }
    rows_by_split = {
        split: [r for r in joined.values() if r["protocol_split"] == split]
        for split in SPLITS
    }
    threshold, calibration = choose_threshold(rows_by_split["calibration"], max_fpr)
    observed = set(joined)
    missing_rows_by_split = {
        split: [
            row for track_id, row in manifest_rows.items()
            if track_id not in observed and row["protocol_split"] == split
        ]
        for split in SPLITS
    }
    split_metrics = {}
    for split in SPLITS:
        rows = rows_by_split[split]
        split_metrics[split] = {
            "metrics": point_metrics(rows, threshold),
            "native_threshold_0_5": point_metrics(rows, 0.5),
            "per_source": per_source(rows, threshold),
            "per_cohort_group": per_cohort_group(rows, threshold),
            "failure_as_error": failure_as_error_metrics(
                rows, threshold, missing_rows_by_split[split]
            ),
        }
        if split == "test":
            split_metrics[split]["bootstrap_95_ci"] = bootstrap_ci(
                rows, threshold, n_boot=n_boot, seed=seed
            )
    expected = set(manifest_rows)
    missing_by_split = {
        split: sorted(
            track_id
            for track_id in expected - observed
            if manifest_rows[track_id]["protocol_split"] == split
        )
        for split in SPLITS
    }
    return {
        "score_file": str(score_path.resolve()),
        "score_file_sha256": sha256(score_path),
        "threshold_rule": {
            "selection_split": "calibration",
            "constraint": f"FPR <= {max_fpr:.6g}",
            "objective": "maximize TPR; ties choose the lowest threshold",
            "selected_threshold": threshold,
            "calibration_metrics": calibration,
        },
        "splits": split_metrics,
        "inference_failures": failures,
        "missing_scored_track_ids_by_split": missing_by_split,
    }, joined


def paired_common_test(
    models: dict[str, dict[str, dict]],
    model_results: dict[str, dict],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    names = sorted(models)
    common = set.intersection(*(set(rows) for rows in models.values()))
    common = {
        track_id
        for track_id in common
        if next(iter(models.values()))[track_id]["protocol_split"] == "test"
    }
    paired_metrics = {}
    for name in names:
        rows = [models[name][track_id] for track_id in sorted(common)]
        threshold = model_results[name]["threshold_rule"]["selected_threshold"]
        paired_metrics[name] = {
            "metrics": point_metrics(rows, threshold),
            "native_threshold_0_5": point_metrics(rows, 0.5),
            "per_source": per_source(rows, threshold),
            "per_cohort_group": per_cohort_group(rows, threshold),
            "bootstrap_95_ci": bootstrap_ci(
                rows, threshold, n_boot=n_boot, seed=seed
            ),
        }
    return {
        "models": names,
        "n_common_test_tracks": len(common),
        "common_test_track_ids": sorted(common),
        "per_model": paired_metrics,
        "note": "Use this identical-ID intersection for all paired model comparisons.",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# ArtifactBench v2 frozen-protocol results",
        "",
        "Thresholds were selected on calibration only. Validation was diagnostic and did not alter thresholds.",
        "",
        "| Model | Threshold | Test n | AUROC | AUPRC | F1 | Bal. acc. | TPR | FPR | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in report["models"].items():
        metrics = report["paired_common_test"]["per_model"][name]["metrics"]
        own_n = model["splits"]["test"]["metrics"]["n"]
        lines.append(
            f"| {name} | {metrics['threshold']:.6f} | {own_n} "
            f"(paired {metrics['n']}) | "
            f"{metrics['auroc']:.3f} | {metrics['auprc']:.3f} | {metrics['f1']:.3f} | "
            f"{metrics['balanced_accuracy']:.3f} | {metrics['tpr']:.3f} | "
            f"{metrics['fpr']:.3f} | {len(model['inference_failures'])} |"
        )
    lines.extend(["", f"Common paired test tracks: {report['paired_common_test']['n_common_test_tracks']}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--result", action="append", required=True, metavar="MODEL=RESULT_DIR"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    manifest_rows = {row["track_id"]: row for row in manifest["tracks"]}
    model_results = {}
    joined = {}
    for spec in args.result:
        name, path = spec.split("=", 1)
        model_results[name], joined[name] = model_analysis(
            name,
            Path(path),
            manifest_rows,
            max_fpr=args.max_fpr,
            n_boot=args.bootstrap,
            seed=args.seed,
        )
    report = {
        "schema_version": 1,
        "protocol": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "threshold_selection": "calibration only",
            "validation_role": "diagnostic only; no retuning",
            "test_role": "sealed final evaluation",
            "max_calibration_fpr": args.max_fpr,
            "bootstrap": "label-stratified lineage resampling",
            "bootstrap_replicates": args.bootstrap,
            "seed": args.seed,
        },
        "models": model_results,
        "paired_common_test": paired_common_test(
            joined, model_results, n_boot=args.bootstrap, seed=args.seed
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "frozen_protocol_metrics.json"
    md_path = args.output / "frozen_protocol_metrics.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    md_path.write_text(render_markdown(report))
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()

"""Sanity suite — fail-loud check + regression detection."""
import json
from pathlib import Path

import numpy as np

from .thresholds import (
    AI_TPR_MIN_DEFAULT,
    AI_TPR_MIN_SOFT,
    CODEC_DELTA_MAX_MAX,
    CODEC_DELTA_MEAN_MAX,
    REAL_FPR_MAX,
    REGRESSION_FPR_MAX_DELTA,
    REGRESSION_TPR_MAX_DELTA,
)


def check_fail(per_source, codec_pairs):
    """per_source summary + codec_pairs → fails list."""
    fails = []
    for src, s in per_source.items():
        if s is None:
            continue
        if s["label"] == "real":
            if s["fpr"] > REAL_FPR_MAX:
                fails.append(f"[FAIL] {src}: real FPR {s['fpr']:.3f} > {REAL_FPR_MAX}")
        else:
            tpr_min = AI_TPR_MIN_SOFT.get(src, AI_TPR_MIN_DEFAULT)
            if s["tpr"] < tpr_min:
                fails.append(f"[FAIL] {src}: AI TPR {s['tpr']:.3f} < {tpr_min}")
    if codec_pairs:
        deltas = np.array([p["delta"] for p in codec_pairs])
        mean_d = float(deltas.mean())
        max_d = float(deltas.max())
        if mean_d > CODEC_DELTA_MEAN_MAX:
            fails.append(f"[FAIL] codec pair mean Δ {mean_d:.3f} > {CODEC_DELTA_MEAN_MAX}")
        if max_d > CODEC_DELTA_MAX_MAX:
            fails.append(f"[FAIL] codec pair max Δ {max_d:.3f} > {CODEC_DELTA_MAX_MAX}")
    return fails


def check_regression(per_source, baseline_path):
    """baseline JSON 대비 regression 감지.

    baseline_path: 이전 bench의 per_source.json 경로.
    Returns: list of regression warnings.
    """
    if not baseline_path or not Path(baseline_path).exists():
        return []

    with open(baseline_path) as f:
        baseline = json.load(f)

    regressions = []
    for src, s in per_source.items():
        if s is None:
            continue
        b = baseline.get(src)
        if b is None:
            continue

        if s["label"] == "real" and "fpr" in s and "fpr" in b:
            delta = s["fpr"] - b["fpr"]
            if delta > REGRESSION_FPR_MAX_DELTA:
                regressions.append(
                    f"[REGRESSION] {src}: FPR {b['fpr']:.3f} → {s['fpr']:.3f} "
                    f"(+{delta:.3f} > {REGRESSION_FPR_MAX_DELTA})"
                )
        elif s["label"] == "ai" and "tpr" in s and "tpr" in b:
            delta = b["tpr"] - s["tpr"]
            if delta > REGRESSION_TPR_MAX_DELTA:
                regressions.append(
                    f"[REGRESSION] {src}: TPR {b['tpr']:.3f} → {s['tpr']:.3f} "
                    f"(-{delta:.3f} > {REGRESSION_TPR_MAX_DELTA})"
                )

    return regressions

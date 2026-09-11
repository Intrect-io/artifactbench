#!/usr/bin/env python3
"""ArtifactBench v1 — 4모델 비교 차트 생성."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_all_results(base_dir):
    """4모델 per_source.json 로드."""
    models = {}

    paths = {
        "ArtifactNet\n(4.2M)": base_dir / "artifactbench_v94_260416" / "artifactnet-v94" / "per_source.json",
        "SpecTTTra\n(19M)": base_dir / "artifactbench_spectttra_clam_260416" / "spectttra" / "per_source.json",
        "CLAM\n(194M)": base_dir / "artifactbench_spectttra_clam_260416" / "clam" / "per_source.json",
    }

    for name, path in paths.items():
        if path.exists():
            with open(path) as f:
                models[name] = json.load(f)
    return models


def plot_aggregate_f1(models_data, output_path):
    """모델별 aggregate F1/Precision/Recall/FPR 바 차트."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    model_names = list(models_data.keys())
    metrics = {}
    for name, data in models_data.items():
        tp, fp, fn, tn = 0, 0, 0, 0
        for s in data.values():
            if s is None:
                continue
            n = s["n"]
            if s["label"] == "ai":
                t = int(round(n * s.get("tpr", 0)))
                tp += t; fn += n - t
            else:
                f = int(round(n * s.get("fpr", 0)))
                fp += f; tn += n - f
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        fpr = fp / max(fp + tn, 1)
        metrics[name] = {"F1": f1, "Precision": prec, "Recall": rec, "FPR": fpr}

    colors = ["#1565C0", "#FF9800", "#F44336"]
    for idx, metric_name in enumerate(["F1", "Precision", "Recall", "FPR"]):
        ax = axes[idx]
        vals = [metrics[m][metric_name] for m in model_names]
        bars = ax.bar(range(len(model_names)), vals, color=colors[:len(model_names)],
                      edgecolor="white", linewidth=1.5)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, fontsize=8)
        ax.set_title(metric_name, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if metric_name == "FPR":
            ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.7, label="5% threshold")
            ax.legend(fontsize=8)

    fig.suptitle("ArtifactBench v1 — Aggregate Performance", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_ai_tpr_heatmap(models_data, output_path):
    """AI source별 TPR 히트맵 (모델 × 소스)."""
    model_names = list(models_data.keys())

    # AI sources 수집
    ai_sources = set()
    for data in models_data.values():
        for src, s in data.items():
            if s and s["label"] == "ai":
                ai_sources.add(src)
    ai_sources = sorted(ai_sources)

    # 매트릭스 구성
    matrix = np.zeros((len(model_names), len(ai_sources)))
    for i, name in enumerate(model_names):
        for j, src in enumerate(ai_sources):
            s = models_data[name].get(src)
            if s and "tpr" in s:
                matrix[i, j] = s["tpr"]
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(20, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(ai_sources)))
    ax.set_xticklabels([s.replace("_", "\n") for s in ai_sources], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)

    # 수치 표시
    for i in range(len(model_names)):
        for j in range(len(ai_sources)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7,
                    color=color, fontweight="bold" if val < 0.9 else "normal")

    plt.colorbar(im, ax=ax, label="TPR", shrink=0.8)
    ax.set_title("AI Source TPR by Model — ArtifactBench v1", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_real_fpr_heatmap(models_data, output_path):
    """Real source별 FPR 히트맵."""
    model_names = list(models_data.keys())

    real_sources = set()
    for data in models_data.values():
        for src, s in data.items():
            if s and s["label"] == "real":
                real_sources.add(src)
    real_sources = sorted(real_sources)

    matrix = np.zeros((len(model_names), len(real_sources)))
    for i, name in enumerate(model_names):
        for j, src in enumerate(real_sources):
            s = models_data[name].get(src)
            if s and "fpr" in s:
                matrix[i, j] = s["fpr"]
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(real_sources)))
    ax.set_xticklabels([s.replace("_", "\n") for s in real_sources], fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)

    for i in range(len(model_names)):
        for j in range(len(real_sources)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold" if val > 0.05 else "normal")

    plt.colorbar(im, ax=ax, label="FPR (lower is better)", shrink=0.8)
    ax.set_title("Real Source FPR by Model — ArtifactBench v1", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_fail_count(models_data, output_path):
    """모델별 FAIL 수 + params 크기 대비."""
    from artifactbench.metrics.thresholds import (
        REAL_FPR_MAX, AI_TPR_MIN_DEFAULT, AI_TPR_MIN_SOFT,
    )

    model_names = list(models_data.keys())
    params = {
        "ArtifactNet\n(4.2M)": 4.2,
        "SpecTTTra\n(19M)": 19,
        "CLAM\n(194M)": 194.3,
    }

    fails = {}
    for name, data in models_data.items():
        n_fail = 0
        for src, s in data.items():
            if s is None:
                continue
            if s["label"] == "real" and s.get("fpr", 0) > REAL_FPR_MAX:
                n_fail += 1
            elif s["label"] == "ai":
                tpr_min = AI_TPR_MIN_SOFT.get(src, AI_TPR_MIN_DEFAULT)
                if s.get("tpr", 0) < tpr_min:
                    n_fail += 1
        fails[name] = n_fail

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = range(len(model_names))
    colors = ["#1565C0", "#FF9800", "#F44336"]

    bars = ax1.bar(x, [fails[m] for m in model_names], color=colors[:len(model_names)],
                   edgecolor="white", linewidth=2, alpha=0.8)
    ax1.set_ylabel("Sanity FAIL Count", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=10)
    ax1.set_ylim(0, 28)

    for bar, name in zip(bars, model_names):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{fails[name]}", ha="center", va="bottom", fontsize=14, fontweight="bold")

    # Params overlay
    ax2 = ax1.twinx()
    ax2.plot(x, [params.get(m, 0) for m in model_names], "ko-", markersize=10, linewidth=2, label="Params (M)")
    ax2.set_ylabel("Parameters (M)", fontsize=12)
    ax2.set_ylim(0, 220)
    ax2.legend(loc="upper left", fontsize=10)

    ax1.set_title("ArtifactBench v1 — FAIL Count vs Model Size", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    base_dir = Path("outputs")
    out_dir = base_dir / "artifactbench_charts_260416"
    out_dir.mkdir(exist_ok=True)

    print("Loading results...")
    models_data = load_all_results(base_dir)
    print(f"  Loaded {len(models_data)} models: {list(models_data.keys())}")

    print("\nGenerating charts...")
    plot_aggregate_f1(models_data, out_dir / "aggregate_metrics.png")
    plot_ai_tpr_heatmap(models_data, out_dir / "ai_tpr_heatmap.png")
    plot_real_fpr_heatmap(models_data, out_dir / "real_fpr_heatmap.png")
    plot_fail_count(models_data, out_dir / "fail_vs_params.png")

    print(f"\nAll charts saved to: {out_dir}")


if __name__ == "__main__":
    main()

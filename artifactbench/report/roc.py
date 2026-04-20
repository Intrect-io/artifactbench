"""ROC curve 생성 + threshold sweep 분석.

핵심 목적:
  1. 모든 모델의 ROC curve 비교 plot
  2. "어떤 threshold에서도 Real FPR ≤ 5%이면서 합리적 AI TPR 달성 불가" 입증
  3. MoM subset에서 CLAM F1 재현
"""
import json
from pathlib import Path

import numpy as np


def load_track_probs(path):
    """track_probs.json 로드 → (probs, labels) numpy."""
    with open(path) as f:
        data = json.load(f)
    probs = np.array([d["prob"] for d in data])
    labels = np.array([d["label"] for d in data])
    return probs, labels, data


def threshold_sweep(probs, labels, thresholds=None):
    """threshold sweep → TPR, FPR, F1 at each threshold."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 201)
    results = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tpr
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        results.append({
            "threshold": float(t),
            "tpr": float(tpr), "fpr": float(fpr),
            "precision": float(prec), "recall": float(rec),
            "f1": float(f1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })
    return results


def find_best_f1(sweep_results):
    """sweep에서 최적 F1 threshold."""
    best = max(sweep_results, key=lambda r: r["f1"])
    return best


def find_fpr_constrained(sweep_results, max_fpr=0.05):
    """FPR ≤ max_fpr 조건에서 최대 TPR 달성 threshold."""
    valid = [r for r in sweep_results if r["fpr"] <= max_fpr]
    if not valid:
        return None  # 어떤 threshold에서도 조건 미달
    return max(valid, key=lambda r: r["tpr"])


def mom_subset_f1(track_data):
    """MoM source만 필터 → F1 재현."""
    mom_sources = {
        "mom_real", "mom_real_wav", "mom_extra_real",
        "mom_diffrythm", "mom_riffusion", "mom_udio", "mom_yue",
    }
    filtered = [d for d in track_data if d["source"] in mom_sources]
    if not filtered:
        return None
    probs = np.array([d["prob"] for d in filtered])
    labels = np.array([d["label"] for d in filtered])
    pred = (probs >= 0.5).astype(int)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    tn = ((pred == 0) & (labels == 0)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "f1": float(f1), "precision": float(prec), "recall": float(rec),
        "fpr": float(fp / max(fp + tn, 1)),
        "n_ai": int(tp + fn), "n_real": int(fp + tn),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def generate_roc_plot(all_model_results, output_path):
    """다중 모델 ROC curve plot → PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skip ROC plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: ROC curve
    ax = axes[0]
    for model_result in all_model_results:
        name = model_result["model_info"]["name"]
        probs, labels, _ = load_track_probs_from_result(model_result)
        sweep = threshold_sweep(probs, labels)
        fprs = [r["fpr"] for r in sweep]
        tprs = [r["tpr"] for r in sweep]
        best = find_best_f1(sweep)
        ax.plot(fprs, tprs, label=f"{name} (best F1={best['f1']:.3f} @τ={best['threshold']:.2f})")
        ax.scatter([best["fpr"]], [best["tpr"]], marker="*", s=100, zorder=5)

    ax.axvline(x=0.05, color="red", linestyle="--", alpha=0.5, label="FPR=5% threshold")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate (Real → AI)")
    ax.set_ylabel("True Positive Rate (AI detected)")
    ax.set_title("ROC Curve — ArtifactBench v1")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # Right: F1 vs threshold
    ax2 = axes[1]
    for model_result in all_model_results:
        name = model_result["model_info"]["name"]
        probs, labels, _ = load_track_probs_from_result(model_result)
        sweep = threshold_sweep(probs, labels)
        thresholds = [r["threshold"] for r in sweep]
        f1s = [r["f1"] for r in sweep]
        ax2.plot(thresholds, f1s, label=name)

    ax2.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="τ=0.5 (default)")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 vs Threshold")
    ax2.legend(fontsize=8)
    ax2.set_xlim(-0.02, 1.02)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ROC plot: {output_path}")


def load_track_probs_from_result(model_result):
    """result dict에서 track_probs 추출."""
    data = model_result.get("track_probs", [])
    probs = np.array([d["prob"] for d in data])
    labels = np.array([d["label"] for d in data])
    return probs, labels, data


def generate_roc_report(all_model_results):
    """ROC 분석 markdown 리포트."""
    lines = []
    lines.append("# ROC Analysis — ArtifactBench v1")
    lines.append("")

    for model_result in all_model_results:
        name = model_result["model_info"]["name"]
        probs, labels, track_data = load_track_probs_from_result(model_result)
        if len(probs) == 0:
            continue

        sweep = threshold_sweep(probs, labels)
        best = find_best_f1(sweep)
        constrained = find_fpr_constrained(sweep, max_fpr=0.05)

        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"| Metric | @τ=0.5 | @Best F1 | @FPR≤5% |")
        lines.append(f"|---|---|---|---|")

        at_05 = next((r for r in sweep if abs(r["threshold"] - 0.5) < 0.003), None)
        if at_05:
            c_str = f"τ={constrained['threshold']:.2f} TPR={constrained['tpr']:.3f} FPR={constrained['fpr']:.3f} F1={constrained['f1']:.3f}" if constrained else "**impossible**"
            c_tpr = f"{constrained['tpr']:.4f}" if constrained else "N/A"
            c_fpr = f"{constrained['fpr']:.4f}" if constrained else "N/A"
            lines.append(f"| F1 | {at_05['f1']:.4f} | {best['f1']:.4f} (@τ={best['threshold']:.2f}) | {c_str} |")
            lines.append(f"| TPR | {at_05['tpr']:.4f} | {best['tpr']:.4f} | {c_tpr} |")
            lines.append(f"| FPR | {at_05['fpr']:.4f} | {best['fpr']:.4f} | {c_fpr} |")

        # MoM subset F1
        mom = mom_subset_f1(track_data)
        if mom:
            lines.append("")
            lines.append(f"**MoM subset**: F1={mom['f1']:.4f} (Prec={mom['precision']:.3f} Rec={mom['recall']:.3f} FPR={mom['fpr']:.3f}, n_ai={mom['n_ai']} n_real={mom['n_real']})")

        lines.append("")

    return "\n".join(lines)

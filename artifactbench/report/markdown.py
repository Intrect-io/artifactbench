"""Markdown 리포트 생성 — 단일 모델 + 다중 모델 비교."""
import numpy as np
from ..metrics.thresholds import (
    REAL_FPR_MAX, AI_TPR_MIN_DEFAULT, AI_TPR_MIN_SOFT,
    CODEC_DELTA_MEAN_MAX, CODEC_DELTA_MAX_MAX,
)


def single_model_report(model_info, per_source, codec_pairs, fails, elapsed):
    """단일 모델 sanity 리포트 생성."""
    lines = []
    lines.append(f"# ArtifactBench v1 — {model_info['name']}")
    lines.append(f"\nModel: {model_info['name']} | Params: {model_info['params']:,}")
    lines.append(f"Input: {model_info['input_sr']}Hz, {model_info['input_duration']}s")
    lines.append(f"Ref: {model_info['paper_ref']}")
    lines.append(f"Elapsed: {elapsed:.0f}s")
    lines.append("")

    # Fail summary
    lines.append("## Fail Summary")
    if fails:
        lines.append(f"**{len(fails)} failures**:")
        lines.append("")
        for f in fails:
            lines.append(f"- {f}")
    else:
        lines.append("All checks passed")
    lines.append("")

    # Source-level
    lines.append("## Source-level Results")
    lines.append("")
    lines.append("| source | n | label | prob mean | prob median | p10 | p90 | rate |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for src in sorted(per_source):
        s = per_source[src]
        if s is None:
            lines.append(f"| {src} | 0 | - | - | - | - | - | - |")
            continue
        rate = s.get("tpr", s.get("fpr", 0))
        rate_name = "tpr" if "tpr" in s else "fpr"
        lines.append(
            f"| {src} | {s['n']} | {s['label']} | "
            f"{s['prob_mean']:.3f} | {s['prob_median']:.3f} | "
            f"{s['prob_p10']:.3f} | {s['prob_p90']:.3f} | "
            f"{rate_name}={rate:.3f} |"
        )

    # Codec pair
    lines.append("")
    lines.append("## Codec Pair Invariance")
    if codec_pairs:
        deltas = np.array([p["delta"] for p in codec_pairs])
        lines.append(f"- n: {len(codec_pairs)}")
        lines.append(f"- Δ mean: {deltas.mean():.4f}")
        lines.append(f"- Δ median: {np.median(deltas):.4f}")
        lines.append(f"- Δ max: {deltas.max():.4f}")
        lines.append(f"- Δ p90: {np.percentile(deltas, 90):.4f}")
    else:
        lines.append("(not measured)")

    return "\n".join(lines)


def comparison_report(all_results):
    """다중 모델 비교 리포트 생성.

    all_results: list of {
        "model_info": dict, "per_source": dict, "codec_pairs": list,
        "fails": list, "elapsed": float
    }
    """
    models = [r["model_info"]["name"] for r in all_results]
    lines = []
    lines.append("# ArtifactBench v1 — Comparison Report")
    lines.append("")

    # Model specs table
    lines.append("## Model Specifications")
    lines.append("")
    header = "| | " + " | ".join(models) + " |"
    sep = "|---|" + "|".join(["---"] * len(models)) + "|"
    lines.append(header)
    lines.append(sep)
    for field, label in [("params", "Parameters"), ("input_sr", "Input SR"),
                          ("input_duration", "Duration (s)"), ("paper_ref", "Reference")]:
        vals = []
        for r in all_results:
            v = r["model_info"][field]
            if field == "params":
                v = f"{v:,}"
            vals.append(str(v))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    # Fail count
    lines.append("")
    lines.append("## Sanity FAIL Count")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    fail_vals = [str(len(r["fails"])) for r in all_results]
    lines.append("| **Total FAIL** | " + " | ".join(fail_vals) + " |")

    # Per-source comparison: AI TPR
    lines.append("")
    lines.append("## AI Source TPR Comparison")
    lines.append("")
    all_ai_sources = set()
    for r in all_results:
        for src, s in r["per_source"].items():
            if s and s["label"] == "ai":
                all_ai_sources.add(src)
    lines.append("| Source | " + " | ".join(models) + " |")
    lines.append("|---|" + "|".join(["---"] * len(models)) + "|")
    for src in sorted(all_ai_sources):
        vals = []
        for r in all_results:
            s = r["per_source"].get(src)
            if s and "tpr" in s:
                v = f"{s['tpr']:.1%}"
                tpr_min = AI_TPR_MIN_SOFT.get(src, AI_TPR_MIN_DEFAULT)
                if s["tpr"] < tpr_min:
                    v = f"**{v}**"
            else:
                v = "-"
            vals.append(v)
        lines.append(f"| {src} | " + " | ".join(vals) + " |")

    # Per-source comparison: Real FPR
    lines.append("")
    lines.append("## Real Source FPR Comparison")
    lines.append("")
    all_real_sources = set()
    for r in all_results:
        for src, s in r["per_source"].items():
            if s and s["label"] == "real":
                all_real_sources.add(src)
    lines.append("| Source | " + " | ".join(models) + " |")
    lines.append("|---|" + "|".join(["---"] * len(models)) + "|")
    for src in sorted(all_real_sources):
        vals = []
        for r in all_results:
            s = r["per_source"].get(src)
            if s and "fpr" in s:
                v = f"{s['fpr']:.1%}"
                if s["fpr"] > REAL_FPR_MAX:
                    v = f"**{v}**"
            else:
                v = "-"
            vals.append(v)
        lines.append(f"| {src} | " + " | ".join(vals) + " |")

    # Codec pair
    lines.append("")
    lines.append("## Codec Pair Δ Comparison")
    lines.append("")
    lines.append("| Metric | " + " | ".join(models) + " |")
    lines.append("|---|" + "|".join(["---"] * len(models)) + "|")
    for metric in ["Δ mean", "Δ max"]:
        vals = []
        for r in all_results:
            if r["codec_pairs"]:
                deltas = np.array([p["delta"] for p in r["codec_pairs"]])
                if metric == "Δ mean":
                    v = f"{deltas.mean():.4f}"
                else:
                    v = f"{deltas.max():.4f}"
            else:
                v = "-"
            vals.append(v)
        lines.append(f"| {metric} | " + " | ".join(vals) + " |")

    # Elapsed
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append("| Model | Elapsed (s) |")
    lines.append("|---|---|")
    for r in all_results:
        lines.append(f"| {r['model_info']['name']} | {r['elapsed']:.0f} |")

    return "\n".join(lines)

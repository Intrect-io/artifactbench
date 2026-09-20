#!/usr/bin/env python3
"""Render paper tables and figures from frozen_protocol_metrics.json only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DISPLAY = {
    "artifactnet": "ArtifactNet v9.4",
    "spectttra": r"SpecTTTra-$\alpha$",
    "deezer_ismir": "Deezer ISMIR",
    "clam": "CLAM",
}
ORDER = ["artifactnet", "spectttra", "deezer_ismir", "clam"]
SOURCE_DISPLAY = {
    "aime_musicgen_large": "MusicGen Large (AIME)",
    "aime_musicgen_medium": "MusicGen Medium (AIME)",
    "aime_musicgen_small": "MusicGen Small (AIME)",
    "aime_riffusion": "Riffusion (AIME)",
    "aime_stable_audio_v1": "Stable Audio v1 (AIME)",
    "aime_stable_audio_v2": "Stable Audio v2 (AIME)",
    "aime_suno_v3": "Suno v3 (AIME)",
    "aime_suno_v35": "Suno v3.5 (AIME)",
    "aime_udio": "Udio (AIME)",
    "suno_cdn_latest": "Suno v4 CDN",
    "suno_extra": "Suno supplementary",
    "udio_cdn_latest": "Udio CDN",
    "udio_extra": "Udio supplementary",
    "fma_hardneg": "FMA real",
    "youtube_hardneg": "Web real",
}
SOURCE_ORDER = list(SOURCE_DISPLAY)


def fmt(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def ci(report, model, metric):
    bounds = report["paired_common_test"]["per_model"][model]["bootstrap_95_ci"][metric]
    return f"{fmt(bounds[0])}--{fmt(bounds[1])}"


def aggregate_tex(report):
    rows = []
    for model in ORDER:
        if model not in report["models"]:
            continue
        result = report["models"][model]
        m = report["paired_common_test"]["per_model"][model]["metrics"]
        attempted = 579
        own_n = result["splits"]["test"]["metrics"]["n"]
        coverage = own_n / attempted
        rows.append(
            f"{DISPLAY[model]} & {own_n}/{attempted} & {m['threshold']:.4f} & "
            f"{fmt(m['auroc'])} & {fmt(m['auprc'])} & {fmt(m['f1'])} & "
            f"{fmt(m['balanced_accuracy'])} & {fmt(m['tpr'])} & {fmt(m['fpr'])} & "
            f"{coverage:.3f} \\\\"
        )
    return "\n".join([
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Model & scored & $\tau$ & AUROC & AUPRC & F1 & BAcc & TPR & FPR & coverage \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])


def interval_tex(report):
    rows = []
    for model in ORDER:
        if model not in report["models"]:
            continue
        rows.append(
            f"{DISPLAY[model]} & {ci(report, model, 'auroc')} & "
            f"{ci(report, model, 'auprc')} & {ci(report, model, 'tpr')} & "
            f"{ci(report, model, 'fpr')} \\\\"
        )
    return "\n".join([
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & AUROC & AUPRC & TPR & FPR \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])


def threshold_tex(report):
    rows = []
    ranking = {"calibrated_balanced_accuracy": [], "native_0_5_balanced_accuracy": [], "auroc": []}
    for model in ORDER:
        if model not in report["models"]:
            continue
        calibrated = report["paired_common_test"]["per_model"][model]["metrics"]
        native = report["paired_common_test"]["per_model"][model]["native_threshold_0_5"]
        rows.append(
            f"{DISPLAY[model]} & {fmt(calibrated['balanced_accuracy'])} & "
            f"{fmt(native['balanced_accuracy'])} & {fmt(calibrated['auroc'])} \\\\"
        )
        ranking["calibrated_balanced_accuracy"].append((model, calibrated["balanced_accuracy"]))
        ranking["native_0_5_balanced_accuracy"].append((model, native["balanced_accuracy"]))
        ranking["auroc"].append((model, calibrated["auroc"]))
    ranking = {
        key: [name for name, _ in sorted(values, key=lambda item: item[1], reverse=True)]
        for key, values in ranking.items()
    }
    table = "\n".join([
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & calibrated BAcc & native-0.5 BAcc & AUROC \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    return table, ranking


def cohort_group_tex(report):
    rows = []
    for model in ORDER:
        if model not in report["models"]:
            continue
        groups = report["paired_common_test"]["per_model"][model]["per_cohort_group"]
        rows.append(
            f"{DISPLAY[model]} & {fmt(groups['aime_ai']['tpr'])} & "
            f"{fmt(groups['suno_supplementary_ai']['tpr'])} & "
            f"{fmt(groups['udio_supplementary_ai']['tpr'])} & "
            f"{fmt(groups['fma_real']['fpr'])} & {fmt(groups['web_real']['fpr'])} \\\\"
        )
    return "\n".join([
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & AIME TPR & Suno TPR & Udio TPR & FMA FPR & Web FPR \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])


def coverage_failure_tex(report):
    rows = []
    for model in ORDER:
        if model not in report["models"]:
            continue
        result = report["models"][model]
        bound = result["splits"]["test"]["failure_as_error"]
        attempted = bound["n_attempted"]
        scored = bound["n_scored"]
        rows.append(
            f"{DISPLAY[model]} & {scored}/{attempted} & {attempted - scored} & "
            f"{scored / attempted:.3f} & {fmt(bound['f1'])} & "
            f"{fmt(bound['balanced_accuracy'])} & {fmt(bound['tpr'])} & "
            f"{fmt(bound['fpr'])} \\\\"
        )
    return "\n".join([
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & scored & failures & coverage & F1 & BAcc & TPR & FPR \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])


def source_heatmap(report, output):
    models = [model for model in ORDER if model in report["models"]]
    sources = sorted({
        source
        for model in models
        for source in report["models"][model]["splits"]["test"]["per_source"]
    }, key=lambda source: SOURCE_ORDER.index(source) if source in SOURCE_ORDER else len(SOURCE_ORDER))
    matrix = np.full((len(sources), len(models)), np.nan)
    annotations = np.empty_like(matrix, dtype=object)
    for col, model in enumerate(models):
        source_data = report["paired_common_test"]["per_model"][model]["per_source"]
        for row, source in enumerate(sources):
            m = source_data.get(source)
            if not m:
                annotations[row, col] = "--"
                continue
            if m["n_ai"]:
                matrix[row, col] = m["tpr"]
                annotations[row, col] = f"TPR\n{m['tpr']:.2f}"
            else:
                matrix[row, col] = 1.0 - m["fpr"]
                annotations[row, col] = f"FPR\n{m['fpr']:.2f}"
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(models)), [DISPLAY[m].replace(r"$\alpha$", "alpha") for m in models], rotation=20, ha="right")
    ax.set_yticks(range(len(sources)), [SOURCE_DISPLAY.get(source, source) for source in sources])
    for row in range(len(sources)):
        for col in range(len(models)):
            value = matrix[row, col]
            color = "white" if np.isfinite(value) and (value < 0.22 or value > 0.82) else "black"
            ax.text(col, row, annotations[row, col], ha="center", va="center", fontsize=7, color=color)
    ax.set_title("Sealed-test source behavior (color = class-correct rate)")
    fig.colorbar(image, ax=ax, label="TPR for AI sources; TNR for real sources")
    fig.tight_layout()
    fig.savefig(output / "source_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output / "source_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "aggregate_table.tex").write_text(aggregate_tex(report))
    (args.output / "bootstrap_intervals_table.tex").write_text(interval_tex(report))
    threshold_table, ranking = threshold_tex(report)
    (args.output / "threshold_sensitivity_table.tex").write_text(threshold_table)
    (args.output / "cohort_group_table.tex").write_text(cohort_group_tex(report))
    (args.output / "coverage_failure_table.tex").write_text(coverage_failure_tex(report))
    (args.output / "ranking_sensitivity.json").write_text(json.dumps(ranking, indent=2) + "\n")
    source_heatmap(report, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Created: 2026-11-04
# Purpose: Turn an ArtifactBench run on the MuseBench eval cohort into a MuseBench report:
#          human FPR, FMA-OOD accuracy, per-generator / co-created / per-transform FNR and
#          AUROC at a fixed threshold, each for all, seen and unseen rows.
# Dependencies: standard library only.
# Test Status: tests/test_musebench_report.py
"""Build ``musebench_report.json`` and ``musebench_report.md`` from a bench run.

Inputs: the eval manifest (with ``seen_in_training`` from leak_map.py) and the run directory
written by ``python -m artifactbench.bench`` (``run_manifest.json``, ``<model>/track_probs.json``,
``<model>/inference_failures.json``).

Conventions: a track is called AI when ``prob >= threshold`` (the runner's rule). FPR is the
share of human tracks called AI, FNR the share of AI tracks called human, FMA-OOD accuracy the
share of FMA tracks called human. AUROC compares an AI group with the Jamendo human negatives of
the same seen/unseen split. Rates carry a Wilson 95 % interval. Nothing is tuned on MuseBench.
"""
from __future__ import annotations

import os as _os
import sys as _sys

# This directory holds select.py, which shadows the stdlib ``select`` module when the
# directory is on sys.path (``python scripts/musebench/x.py`` without -I). Drop it.
if _sys.path and _os.path.abspath(_sys.path[0] or ".") == _os.path.dirname(_os.path.abspath(__file__)):
    _sys.path.pop(0)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

DEFAULT_ROOT = Path(os.environ.get("MUSEBENCH_ROOT", "~/data/musebench")).expanduser()
EVAL_MANIFEST_REL = Path("manifests") / "musebench_eval_manifest.json"
SPLITS = ("all", "seen", "unseen")
HUMAN_NEG = "musebench_human_jamendo"
FMA = "musebench_human_fma_ood"
GENERATOR_ORDER = ["musicgen", "audioldm2", "riffusion", "magnet", "jasco", "stable_audio_open",
                   "mustango", "lyria_rt", "suno", "udio"]
TRANSFORM_ORDER = ["mp3", "ogg", "aac", "eq", "pitch_time", "loudness", "gaussian", "pgd"]
# arXiv:2610.20310v1, Table 10 (FMA OOD accuracy) and Table 1 (FNR on co-created / transforms),
# in %. These detectors were trained on MuseBench Default; quoted for context only.
PAPER_TABLE10 = {"Q-SVM": 55.9, "Deezer": 94.4, "M5": 75.8, "RawNet2": 39.6, "ConvNeXt": 43.5,
                 "ResNet18+Spec": 28.4, "AST": 79.8, "ViT": 40.7, "SpecTTTra": 39.7}
PAPER_TABLE1_COLS = ["co_created", "mp3", "ogg", "aac", "eq", "pitch_time", "loudness",
                     "gaussian", "pgd"]
PAPER_TABLE1 = {
    "Deezer": [36.8, 12.3, 9.5, 7.7, 6.4, 93.6, 5.2, 19.6, 26.4],
    "M5": [20.1, 15.2, 1.9, 1.9, 2.0, 26.2, 1.3, 2.1, 92.4],
    "ConvNeXt": [12.8, 0.2, 0.3, 0.6, 0.2, 17.0, 0.2, 0.4, 11.7],
    "ResNet18+Spec": [0.0, 0.9, 3.7, 0.1, 3.1, 11.3, 2.0, 5.2, 11.1],
    "AST": [4.8, 24.9, 7.3, 3.5, 0.0, 4.2, 0.0, 0.0, 9.2],
    "ViT": [1.7, 0.4, 0.5, 0.5, 0.2, 19.9, 0.2, 0.8, 10.7],
    "SpecTTTra": [33.6, 0.5, 1.0, 1.0, 0.5, 16.7, 0.5, 0.9, 10.7],
}


# --------------------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> dict:
    return {"n": n, "k": k, "rate": (k / n) if n else None, "ci95": wilson(k, n)}


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney AUROC (ties count one half)."""
    if not pos or not neg:
        return None
    scored = sorted([(p, 1) for p in pos] + [(q, 0) for q in neg])
    rank_sum, i = 0.0, 0
    while i < len(scored):
        j = i
        while j < len(scored) and scored[j][0] == scored[i][0]:
            j += 1
        mid = (i + 1 + j) / 2  # average 1-based rank of the tie block
        rank_sum += mid * sum(1 for _, y in scored[i:j] if y == 1)
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# --------------------------------------------------------------------------- report

def split_of(row: dict) -> str:
    return "seen" if row["seen_in_training"] else "unseen"


def build_report(eval_rows: list[dict], probs: dict[str, float], failures: list[dict],
                 threshold: float = 0.5, decisions: dict[str, bool] | None = None) -> dict:
    """Pure function: eval manifest rows + {track_id: prob} + failures -> report dict."""
    failed = {f["track_id"] for f in failures}
    scored = [dict(r, prob=probs[r["track_id"]]) for r in eval_rows if r["track_id"] in probs]
    missing = [r["track_id"] for r in eval_rows
               if r["track_id"] not in probs and r["track_id"] not in failed]

    def rows(split: str, pred) -> list[dict]:
        return [r for r in scored if pred(r) and (split == "all" or split_of(r) == split)]

    def called_ai(r):
        # An explicit per-track decision (e.g. a pipeline with a rescue gate) overrides prob >= threshold.
        if decisions is not None and r["track_id"] in decisions:
            return decisions[r["track_id"]]
        return r["prob"] >= threshold

    def neg_probs(split):
        return [r["prob"] for r in rows(split, lambda r: r["source"] == HUMAN_NEG)]

    def ai_block(split, pred):
        rs = rows(split, pred)
        out = rate(sum(not called_ai(r) for r in rs), len(rs))  # FNR
        out["auroc_vs_jamendo"] = auroc([r["prob"] for r in rs], neg_probs(split))
        out["prob_median"] = sorted(r["prob"] for r in rs)[len(rs) // 2] if rs else None
        return out

    report: dict = {"threshold": threshold, "splits": {},
                    "decision_rule": "explicit per-track decision" if decisions else f"prob >= {threshold}"}
    for split in SPLITS:
        hum = rows(split, lambda r: r["source"] == HUMAN_NEG)
        fma = rows(split, lambda r: r["source"] == FMA)
        clean_ai = rows(split, lambda r: r["label"] == "ai" and r["variant"] == "clean")
        fpr = rate(sum(called_ai(r) for r in hum), len(hum))
        fnr_clean = rate(sum(not called_ai(r) for r in clean_ai), len(clean_ai))
        s = {
            "human_jamendo_fpr": fpr,
            "fma_ood_accuracy": rate(sum(not called_ai(r) for r in fma), len(fma)),
            "clean_ai_fnr": fnr_clean,
            "balanced_accuracy_clean": (
                1 - (fpr["rate"] + fnr_clean["rate"]) / 2
                if fpr["n"] and fnr_clean["n"] else None),
            "auroc_clean_ai_vs_jamendo": auroc([r["prob"] for r in clean_ai], neg_probs(split)),
            "per_generator_clean_fnr": {
                g: ai_block(split, lambda r, g=g: r["generator"] == g and r["variant"] == "clean")
                for g in GENERATOR_ORDER},
            "co_created_fnr": ai_block(split, lambda r: r["variant"] == "co_created"),
            "per_transform_fnr": {
                v: ai_block(split, lambda r, v=v: r["variant"] == v) for v in TRANSFORM_ORDER},
            "per_transform_generator_fnr": {
                f"{g}__{v}": ai_block(split, lambda r, g=g, v=v: r["generator"] == g and r["variant"] == v)
                for v in TRANSFORM_ORDER for g in GENERATOR_ORDER},
        }
        report["splits"][split] = s

    by_src_fail = Counter(f["source"] for f in failures)
    report["coverage"] = {
        "eval_rows": len(eval_rows), "scored": len(scored), "failures": len(failures),
        "missing": len(missing), "failure_rate": len(failures) / len(eval_rows) if eval_rows else None,
        "failures_by_source": dict(sorted(by_src_fail.items())),
        "failures_by_stage": dict(Counter(f.get("stage", "?") for f in failures)),
        "seen_rows": sum(1 for r in eval_rows if r["seen_in_training"]),
    }
    report["paper_context"] = {"table10_fma_ood_accuracy": PAPER_TABLE10,
                               "table1_columns": PAPER_TABLE1_COLS, "table1_fnr": PAPER_TABLE1,
                               "note": "paper detectors are trained on MuseBench Default; "
                                       "this detector is zero-shot. Not a head-to-head."}
    return report


def pct(block) -> str:
    if block is None:
        return "–"
    if isinstance(block, (int, float)):
        return f"{100 * block:.1f}"
    if not block.get("n"):
        return "–"
    lo, hi = block["ci95"]
    return f"{100 * block['rate']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}] (n={block['n']})"


def auc_s(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def to_markdown(report: dict, title: str) -> str:
    sp = report["splits"]
    L = [f"# {title}", "",
         f"Threshold {report['threshold']} (fixed, not tuned on MuseBench). Rates in % with Wilson "
         "95 % CI. `seen` = audio (or its direct ancestor) in the detector's training candidate "
         "pool; `unseen` = the zero-shot part.", ""]
    cov = report["coverage"]
    L += [f"Coverage: {cov['scored']} of {cov['eval_rows']} eval rows scored, {cov['failures']} "
          f"failures, {cov['missing']} missing; {cov['seen_rows']} rows are `seen`.", ""]
    L += ["## Headline", "", "| Metric | all | seen | unseen |", "|---|---|---|---|"]
    for key, name in [("human_jamendo_fpr", "Human (Jamendo) FPR"),
                      ("fma_ood_accuracy", "FMA-OOD accuracy"),
                      ("clean_ai_fnr", "Clean AI FNR (all generators)")]:
        L.append(f"| {name} | " + " | ".join(pct(sp[s][key]) for s in SPLITS) + " |")
    L.append("| Balanced accuracy, clean | " + " | ".join(pct(sp[s]["balanced_accuracy_clean"]) for s in SPLITS) + " |")
    L.append("| AUROC clean AI vs Jamendo | " + " | ".join(auc_s(sp[s]["auroc_clean_ai_vs_jamendo"]) for s in SPLITS) + " |")
    L += ["", "## Clean AI FNR per generator", "",
          "| Generator | FNR all | FNR unseen | FNR seen | AUROC all | AUROC unseen |",
          "|---|---|---|---|---|---|"]
    for g in GENERATOR_ORDER:
        a, u, s_ = (sp[x]["per_generator_clean_fnr"][g] for x in ("all", "unseen", "seen"))
        L.append(f"| {g} | {pct(a)} | {pct(u)} | {pct(s_)} | {auc_s(a['auroc_vs_jamendo'])} | "
                 f"{auc_s(u['auroc_vs_jamendo'])} |")
    a, u, s_ = (sp[x]["co_created_fnr"] for x in ("all", "unseen", "seen"))
    L.append(f"| co-created | {pct(a)} | {pct(u)} | {pct(s_)} | {auc_s(a['auroc_vs_jamendo'])} | "
             f"{auc_s(u['auroc_vs_jamendo'])} |")
    L += ["", "## FNR per transform (all generators)", "",
          "| Transform | FNR all | FNR unseen | FNR seen | AUROC all |", "|---|---|---|---|---|"]
    for v in TRANSFORM_ORDER:
        a, u, s_ = (sp[x]["per_transform_fnr"][v] for x in ("all", "unseen", "seen"))
        L.append(f"| {v} | {pct(a)} | {pct(u)} | {pct(s_)} | {auc_s(a['auroc_vs_jamendo'])} |")
    L += ["", "## FNR per transform × generator (all rows, %)", "",
          "| Generator | " + " | ".join(TRANSFORM_ORDER) + " |",
          "|---|" + "---|" * len(TRANSFORM_ORDER)]
    for g in GENERATOR_ORDER:
        cells = []
        for v in TRANSFORM_ORDER:
            b = sp["all"]["per_transform_generator_fnr"][f"{g}__{v}"]
            cells.append("–" if not b["n"] else f"{100 * b['rate']:.1f}")
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    ctx = report["paper_context"]
    L += ["", "## Paper context (arXiv:2610.20310v1; detectors trained on MuseBench)", "",
          "| Detector | FMA-OOD acc. (T10) | " + " | ".join(ctx["table1_columns"]) + " |",
          "|---|---|" + "---|" * len(ctx["table1_columns"])]
    for det, row in ctx["table1_fnr"].items():
        L.append(f"| {det} | {ctx['table10_fma_ood_accuracy'].get(det, '–')} | "
                 + " | ".join(f"{x:.1f}" for x in row) + " |")
    L += ["", "Failures by source: " + (json.dumps(cov["failures_by_source"]) or "{}"), ""]
    return "\n".join(L)


def load_run(run_dir: Path, model_dir: str = "artifactnet") -> tuple[dict[str, float], list[dict], dict]:
    probs = {t["track_id"]: float(t["prob"]) for t in
             json.loads((run_dir / model_dir / "track_probs.json").read_text(encoding="utf-8"))}
    failures = json.loads((run_dir / model_dir / "inference_failures.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    return probs, failures, run_manifest


def load_decisions(result_dir: Path, model_dir: str) -> dict[str, bool] | None:
    """Explicit per-track decisions (``decision_ai``) if the score file carries them."""
    rows = json.loads((result_dir / model_dir / "track_probs.json").read_text(encoding="utf-8"))
    if rows and all("decision_ai" in t for t in rows):
        return {t["track_id"]: bool(t["decision_ai"]) for t in rows}
    return None


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def build_full_report(eval_rows: list[dict], run_dir: Path, model_dir: str = "artifactnet",
                      finite_subdir: str = "finite_chunk") -> dict:
    """Primary = declared finite-chunk policy when present, else strict; strict kept as sensitivity."""
    probs, failures, run_manifest = load_run(run_dir, model_dir)
    threshold = run_manifest.get("threshold", 0.5)
    strict = build_report(eval_rows, probs, failures, threshold=threshold,
                          decisions=load_decisions(run_dir, model_dir))
    out = {"strict": strict}
    finite_dir = run_dir / finite_subdir
    if (finite_dir / model_dir / "track_probs.json").exists():
        f_probs, f_failures, _ = load_run_scores(finite_dir, model_dir)
        out["primary"] = build_report(eval_rows, f_probs, f_failures, threshold=threshold,
                                      decisions=load_decisions(finite_dir, model_dir))
        policy_path = finite_dir / model_dir / "finite_chunk_policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}
        out["primary_policy"] = {k: v for k, v in policy.items() if k != "recovered"}
        out["primary_policy"]["result_dir"] = str(finite_dir)
    else:
        out["primary"] = strict
        out["primary_policy"] = {"track_aggregation": "strict: any non-finite chunk fails the track"}
    out["run"] = {"run_dir": str(run_dir), "model_dir": model_dir,
                  "threshold": threshold,
                  "input_manifest_sha256": run_manifest.get("input_manifest_sha256"),
                  "runner_git_revision": run_manifest.get("runner_git_revision")}
    return out


def load_run_scores(result_dir: Path, model_dir: str) -> tuple[dict[str, float], list[dict], None]:
    probs = {t["track_id"]: float(t["prob"]) for t in
             json.loads((result_dir / model_dir / "track_probs.json").read_text(encoding="utf-8"))}
    failures = json.loads((result_dir / model_dir / "inference_failures.json").read_text(encoding="utf-8"))
    return probs, failures, None


def full_markdown(full: dict, title: str) -> str:
    md = to_markdown(full["primary"], title)
    pol = full["primary_policy"]
    head = [f"Primary track policy: {pol.get('track_aggregation')}"
            + (f" (at least {pol['minimum_valid_chunks']} finite chunks)" if "minimum_valid_chunks" in pol else ""),
            ""]
    st = full["strict"]["splits"]
    sens = ["", "## Strict sensitivity (any non-finite chunk fails the track)", "",
            f"Coverage: {full['strict']['coverage']['scored']} of {full['strict']['coverage']['eval_rows']} "
            f"scored, {full['strict']['coverage']['failures']} failures.", "",
            "| Metric | all | seen | unseen |", "|---|---|---|---|"]
    for key, name in [("human_jamendo_fpr", "Human (Jamendo) FPR"),
                      ("fma_ood_accuracy", "FMA-OOD accuracy"),
                      ("clean_ai_fnr", "Clean AI FNR")]:
        sens.append(f"| {name} | " + " | ".join(pct(st[x][key]) for x in SPLITS) + " |")
    lines = md.splitlines()
    return "\n".join(lines[:2] + head + lines[2:] + sens)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the MuseBench report from a bench run.")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="bench output directory (default: <root>/results/artifactnet_v94)")
    ap.add_argument("--model-dir", default="artifactnet")
    ap.add_argument("--title", default="ArtifactNet v9.4 on MuseBench (zero-shot)")
    args = ap.parse_args(argv)
    run_dir = args.run_dir or args.root / "results" / "artifactnet_v94"
    eval_rows = json.loads((args.root / EVAL_MANIFEST_REL).read_text(encoding="utf-8"))["tracks"]
    full = build_full_report(eval_rows, run_dir, args.model_dir)
    full["run"]["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write(run_dir / "musebench_report.json", json.dumps(full, indent=1) + "\n")
    atomic_write(run_dir / "musebench_report.md", full_markdown(full, args.title) + "\n")
    cov = full["primary"]["coverage"]
    print(f"report: {cov['scored']}/{cov['eval_rows']} scored, {cov['failures']} failures, "
          f"{cov['missing']} missing -> {run_dir / 'musebench_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""ArtifactBench — version-pinned AI-generated music detection runner.

Usage:
    # Single model
    python -m artifactbench.bench --model artifactnet \
        --manifest manifest.json --output out/

    # Three-way comparison (CLAM requires --clam-repo + --clam-ckpt)
    python -m artifactbench.bench \
        --model artifactnet --model spectttra --model clam \
        --clam-repo ~/dev/MoM-CLAM \
        --clam-ckpt ~/dev/MoM-CLAM/model_wts/best_model_triplet_loss_margin_0.2.pth \
        --manifest manifest.json --output out/

    # Smoke test with tiny sample
    python -m artifactbench.bench --model artifactnet \
        --manifest manifest.json --n-per-source 10 --n-codec-pair 5 --output out/
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .data.manifest import WAV_SOURCES, load_manifest
from .metrics.codec_pair import codec_pair_measure
from .metrics.failcheck import check_fail
from .metrics.source_level import measure_source, summarize
from .models import (
    MODEL_REGISTRY,
    ArtifactNetModel,
    BenchModel,
    CLAMModel,
    DeezerISMIRModel,
    SpecTTTraModel,
)
from .report.markdown import comparison_report, single_model_report
from .report.roc import generate_roc_plot, generate_roc_report


def build_model(model_name: str, args) -> "BenchModel":
    if model_name == "artifactnet":
        return ArtifactNetModel(
            onnx_path=args.artifactnet_onnx,
            hf_revision=args.artifactnet_revision,
        )
    if model_name == "clam":
        return CLAMModel(
            clam_repo=args.clam_repo,
            clam_ckpt=args.clam_ckpt,
            crop_policy=args.crop_policy,
        )
    if model_name == "spectttra":
        return SpecTTTraModel(crop_policy=args.crop_policy)
    if model_name == "deezer_ismir":
        return DeezerISMIRModel(
            onnx_path=args.deezer_onnx,
            hf_revision=args.deezer_revision,
        )
    cls = MODEL_REGISTRY[model_name]
    return cls()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_select(entries, limit: int, seed: int):
    """Select one stable cohort shared by every model in the run."""
    ranked = sorted(
        entries,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['track_id']}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked if limit == 0 else ranked[:limit]


def package_versions():
    result = {}
    for package in [
        "artifactbench", "numpy", "torch", "torchaudio", "soundfile",
        "onnxruntime", "huggingface-hub", "scikit-learn", "scipy", "soxr",
    ]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def runner_git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_single_model(model, entries, by_source, args):
    t0 = time.time()

    per_source_results = {}
    per_source_summary = {}
    inference_failures = []
    for src in sorted(by_source):
        print(f"\n  [{src}] measuring {len(by_source[src])} frozen tracks...")
        res, failures = measure_source(by_source[src], model)
        per_source_results[src] = res
        inference_failures.extend(failures)
        per_source_summary[src] = summarize(
            res, attempted=len(by_source[src]), threshold=args.threshold
        )
        s = per_source_summary[src]
        if s:
            rate = s.get("tpr", s.get("fpr", 0))
            rate_name = "tpr" if "tpr" in s else "fpr"
            print(f"    n={s['n']} prob_mean={s['prob_mean']:.3f} {rate_name}={rate:.3f}")

    if args.n_codec_pair > 0:
        print(f"\n  [codec pair] measuring up to {args.n_codec_pair} pairs...")
        all_wav_entries = [
            e for src in by_source for e in by_source[src]
            if e.get("source") in WAV_SOURCES
        ]
        codec_entries = deterministic_select(all_wav_entries, args.n_codec_pair, args.seed)
        codec_pairs = codec_pair_measure(codec_entries, model, n_pair=len(codec_entries))
    else:
        codec_pairs = []

    elapsed = time.time() - t0
    fails = check_fail(per_source_summary, codec_pairs)

    all_track_probs = []
    for src, results in per_source_results.items():
        for r in results:
            all_track_probs.append({
                "prob": r["prob"],
                "label": 1 if r["label"] == "ai" else 0,
                "source": r["source"],
                "track_id": r["track_id"],
            })

    return {
        "model_info": model.info(),
        "per_source": per_source_summary,
        "per_source_raw": per_source_results,
        "track_probs": all_track_probs,
        "codec_pairs": codec_pairs,
        "inference_failures": inference_failures,
        "fails": fails,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="ArtifactBench v2-compatible runner")
    parser.add_argument("--model", action="append", required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="model to benchmark (repeatable)")
    parser.add_argument("--manifest", required=True,
                        help="ArtifactBench manifest JSON")
    parser.add_argument("--split", default="bench",
                        choices=["test", "all", "bench"])
    parser.add_argument("--bench-origin", default=None, choices=["test", "train"],
                        help='subset filter; "test" = unseen by all models')
    parser.add_argument("--protocol-split", default=None,
                        choices=["calibration", "validation", "test"],
                        help="frozen ArtifactBench v2 protocol partition")
    parser.add_argument("--n-per-source", type=int, default=100,
                        help="max samples per source (0 = all)")
    parser.add_argument("--n-codec-pair", type=int, default=50,
                        help="number of codec invariance pairs (0 = skip)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="fixed P(AI) decision threshold recorded in every result")
    parser.add_argument("--crop-policy", choices=["center", "start"], default="center",
                        help="deterministic crop for fixed-duration baselines")
    parser.add_argument("--device", default=None,
                        help="force device (cuda/cpu/mps); auto-detects otherwise")

    # Model-specific options
    parser.add_argument("--artifactnet-onnx", default=None,
                        help="local ONNX path (default: auto-download from HF Hub)")
    parser.add_argument("--artifactnet-revision",
                        default="e915f0dc5962a57536bbe1f78b66adcb48dfae4c")
    parser.add_argument("--deezer-onnx", default=None)
    parser.add_argument("--deezer-revision",
                        default="d2180598fed79e3f917e8050a00439982466e5c6")
    parser.add_argument("--clam-repo", default=None,
                        help="path to cloned MoM-CLAM repository")
    parser.add_argument("--clam-ckpt", default=None,
                        help="path to CLAM checkpoint (.pth)")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 72)
    print("ArtifactBench")
    print("=" * 72)
    print(f"  Models:      {args.model}")
    print(f"  Manifest:    {args.manifest}")
    print(f"  Split:       {args.split}")
    print(f"  n/source:    {args.n_per_source}")
    print(f"  n codec:     {args.n_codec_pair}")
    print(f"  Device:      {device}")
    print(f"  Output:      {out_dir}")

    entries, by_source = load_manifest(args.manifest, split=args.split,
                                       bench_origin=args.bench_origin,
                                       protocol_split=args.protocol_split)
    selected_by_source = {
        source: deterministic_select(source_entries, args.n_per_source, args.seed)
        for source, source_entries in sorted(by_source.items())
    }
    selected_entries = [
        row for source in sorted(selected_by_source) for row in selected_by_source[source]
    ]
    print(f"  Sources:     {len(selected_by_source)} ({len(selected_entries)} frozen tracks)")

    run_manifest = {
        "schema_version": 2,
        "runner_git_revision": runner_git_revision(),
        "input_manifest": str(Path(args.manifest).resolve()),
        "input_manifest_sha256": sha256_file(args.manifest),
        "models": args.model,
        "seed": args.seed,
        "threshold": args.threshold,
        "protocol_split": args.protocol_split,
        "crop_policy": args.crop_policy,
        "n_per_source": args.n_per_source,
        "selected_track_ids": [row["track_id"] for row in selected_entries],
        "selected_per_source": {
            source: [row["track_id"] for row in rows]
            for source, rows in selected_by_source.items()
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
        },
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n"
    )

    all_results = []
    for model_name in args.model:
        print(f"\n{'='*72}")
        print(f"  Running: {model_name}")
        print(f"{'='*72}")

        model = build_model(model_name, args)
        model.load(device=device)
        print(f"  Loaded: {model.name} ({model.params:,} params)")

        result = run_single_model(model, selected_entries, selected_by_source, args)
        all_results.append(result)

        # Per-model artifacts
        report = single_model_report(
            result["model_info"], result["per_source"],
            result["codec_pairs"], result["fails"], result["elapsed"],
        )
        model_dir = out_dir / model_name
        model_dir.mkdir(exist_ok=True)
        (model_dir / "report.md").write_text(report)
        with open(model_dir / "per_source.json", "w") as f:
            json.dump(result["per_source"], f, indent=2, default=float)
        with open(model_dir / "track_probs.json", "w") as f:
            json.dump(result["track_probs"], f, default=float)
        with open(model_dir / "inference_failures.json", "w") as f:
            json.dump(result["inference_failures"], f, indent=2, ensure_ascii=False)
        if result["codec_pairs"]:
            with open(model_dir / "codec_pair.json", "w") as f:
                json.dump(result["codec_pairs"], f, indent=2, default=float)

        print(f"\n  {'='*40}")
        if result["fails"]:
            print(f"  {model.name}: {len(result['fails'])} FAIL")
            for f in result["fails"]:
                print(f"    {f}")
        else:
            print(f"  {model.name}: ALL PASS")
        print(f"  Elapsed: {result['elapsed']:.0f}s")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    # Comparison artifacts
    if len(all_results) > 1:
        comp = comparison_report(all_results)
        (out_dir / "comparison.md").write_text(comp)
        print(f"\nComparison report: {out_dir / 'comparison.md'}")

    if any(r.get("track_probs") for r in all_results):
        roc_report = generate_roc_report(all_results)
        (out_dir / "roc_analysis.md").write_text(roc_report)
        print(f"ROC analysis:      {out_dir / 'roc_analysis.md'}")
        try:
            generate_roc_plot(all_results, out_dir / "roc_curves.png")
            print(f"ROC curves:        {out_dir / 'roc_curves.png'}")
        except Exception as e:
            print(f"ROC plot skipped: {e}")

    summary = {
        "models": [r["model_info"] for r in all_results],
        "fail_counts": {r["model_info"]["name"]: len(r["fails"]) for r in all_results},
        "elapsed": {r["model_info"]["name"]: round(r["elapsed"], 1) for r in all_results},
        "inference_failure_counts": {
            r["model_info"]["name"]: len(r["inference_failures"])
            for r in all_results
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nArtifactBench complete. Results: {out_dir}")


if __name__ == "__main__":
    main()

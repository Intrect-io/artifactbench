#!/usr/bin/env python3
"""Diagnose ArtifactNet non-finite track outputs at the frozen-chunk level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from artifactbench.data.manifest import load_audio_mono
from artifactbench.models.artifactnet import (
    CHUNK_SAMPLES,
    N_CHUNKS_DEFAULT,
    ArtifactNetModel,
)


def dbfs(chunk: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float64)))))
    return float(20 * np.log10(max(rms, 1e-12)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    rows = {row["track_id"]: row for row in manifest["tracks"]}
    failures = json.loads(args.failures.read_text())
    target_ids = [f["track_id"] for f in failures if f.get("stage") == "inference"]

    model = ArtifactNetModel(n_chunks=N_CHUNKS_DEFAULT)
    model.load(device="cpu")
    audits = []
    for index, track_id in enumerate(target_ids, 1):
        row = rows[track_id]
        audio = load_audio_mono(row["runtime_path"])
        if len(audio) < CHUNK_SAMPLES:
            audio = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))
        max_start = len(audio) - CHUNK_SAMPLES
        starts = (
            [0] * N_CHUNKS_DEFAULT
            if max_start <= 0
            else np.linspace(0, max_start, N_CHUNKS_DEFAULT, dtype=np.int64)
        )
        chunks = []
        for start in starts:
            chunk = audio[start:start + CHUNK_SAMPLES]
            probability = model._forward_chunk(chunk)
            chunks.append({
                "start_sample": int(start),
                "rms_dbfs": dbfs(chunk),
                "peak": float(np.max(np.abs(chunk))),
                "prob": probability if np.isfinite(probability) else None,
                "finite": bool(np.isfinite(probability)),
            })
        finite = [chunk["prob"] for chunk in chunks if chunk["finite"]]
        audits.append({
            "track_id": track_id,
            "source": row["source"],
            "protocol_split": row["protocol_split"],
            "valid_chunks": len(finite),
            "total_chunks": len(chunks),
            "finite_chunk_median": float(np.median(finite)) if finite else None,
            "chunks": chunks,
        })
        print(f"{index}/{len(target_ids)} {track_id}: {len(finite)}/7 finite", flush=True)

    all_chunks = [chunk for audit in audits for chunk in audit["chunks"]]
    nonfinite = [chunk for chunk in all_chunks if not chunk["finite"]]
    report = {
        "tracks_audited": len(audits),
        "chunks_audited": len(all_chunks),
        "nonfinite_chunks": len(nonfinite),
        "tracks_with_at_least_four_finite_chunks": sum(a["valid_chunks"] >= 4 for a in audits),
        "nonfinite_rms_dbfs": {
            "min": min((x["rms_dbfs"] for x in nonfinite), default=None),
            "median": float(np.median([x["rms_dbfs"] for x in nonfinite])) if nonfinite else None,
            "max": max((x["rms_dbfs"] for x in nonfinite), default=None),
        },
        "tracks": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

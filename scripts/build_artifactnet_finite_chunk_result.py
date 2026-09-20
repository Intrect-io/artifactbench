#!/usr/bin/env python3
"""Create a declared finite-chunk ArtifactNet result without overwriting strict output."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--strict-result", type=Path, required=True)
    parser.add_argument("--chunk-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-valid-chunks", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    rows = {row["track_id"]: row for row in manifest["tracks"]}
    strict_model_dir = args.strict_result / "artifactnet"
    strict_scores = json.loads((strict_model_dir / "track_probs.json").read_text())
    strict_failures = json.loads((strict_model_dir / "inference_failures.json").read_text())
    audit = json.loads(args.chunk_audit.read_text())

    scores = {row["track_id"]: row for row in strict_scores}
    unresolved = []
    recovered = []
    for track in audit["tracks"]:
        track_id = track["track_id"]
        if track["valid_chunks"] >= args.min_valid_chunks:
            meta = rows[track_id]
            score = {
                "prob": track["finite_chunk_median"],
                "label": 1 if meta["label"] == "ai" else 0,
                "source": meta["source"],
                "track_id": track_id,
            }
            scores[track_id] = score
            recovered.append({
                "track_id": track_id,
                "valid_chunks": track["valid_chunks"],
                "total_chunks": track["total_chunks"],
            })
        else:
            original = next(f for f in strict_failures if f["track_id"] == track_id)
            unresolved.append({
                **original,
                "valid_chunks": track["valid_chunks"],
                "total_chunks": track["total_chunks"],
                "policy": f"requires at least {args.min_valid_chunks} finite chunks",
            })

    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "artifactnet"
    model_dir.mkdir(exist_ok=True)
    ordered_scores = [scores[row["track_id"]] for row in manifest["tracks"] if row["track_id"] in scores]
    (model_dir / "track_probs.json").write_text(json.dumps(ordered_scores, indent=2) + "\n")
    (model_dir / "inference_failures.json").write_text(
        json.dumps(unresolved, indent=2, ensure_ascii=False) + "\n"
    )
    (model_dir / "chunk_failures.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    policy = {
        "model": "artifactnet",
        "track_aggregation": "median of finite scores from seven evenly spaced 4-second chunks",
        "minimum_valid_chunks": args.min_valid_chunks,
        "strict_scored_tracks": len(strict_scores),
        "recovered_tracks": len(recovered),
        "final_scored_tracks": len(ordered_scores),
        "final_track_failures": len(unresolved),
        "recovered": recovered,
    }
    (model_dir / "finite_chunk_policy.json").write_text(
        json.dumps(policy, indent=2, ensure_ascii=False) + "\n"
    )
    for filename in ("run_manifest.json",):
        source_path = args.strict_result / filename
        if source_path.exists():
            data = json.loads(source_path.read_text())
            data["artifactnet_chunk_policy"] = policy["track_aggregation"]
            data["artifactnet_minimum_valid_chunks"] = args.min_valid_chunks
            (args.output / filename).write_text(json.dumps(data, indent=2) + "\n")
    for filename in ("summary.json",):
        source_path = args.strict_result / filename
        if source_path.exists():
            shutil.copy2(source_path, args.output / filename)
    print(json.dumps({k: v for k, v in policy.items() if k != "recovered"}, indent=2))


if __name__ == "__main__":
    main()

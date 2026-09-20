#!/usr/bin/env python3
"""Sanitize frozen result artifacts for a path- and title-free public release."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

FORBIDDEN_PUBLIC_KEYS = {
    "album",
    "api_key",
    "artist",
    "audio_path",
    "channel",
    "local_path",
    "password",
    "path",
    "runtime_path",
    "secret",
    "title",
    "token",
}
FORBIDDEN_PATH_FRAGMENTS = (
    "/Users/",
    "/Volumes/",
    "/home/",
    "/media/",
    "/mnt/",
    "/private/",
    "/tmp/",
    "\\Users\\",
)
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)


def assert_public_safe(value, trail: str = "root") -> None:
    """Reject common private fields and absolute local paths before writing."""
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_public_safe(item, f"{trail}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"Forbidden public key at {trail}: {key}")
            assert_public_safe(item, f"{trail}.{key}")
        return
    if isinstance(value, str):
        if any(
            fragment.lower() in value.lower() for fragment in FORBIDDEN_PATH_FRAGMENTS
        ):
            raise ValueError(f"Local path leaked at {trail}")
        if EMAIL_PATTERN.search(value):
            raise ValueError(f"Email address leaked at {trail}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def public_id(row: dict) -> str:
    return f"ab2-{row['audio_sha256'][:20]}"


def remap_failures(failures: list[dict], id_map: dict[str, str]) -> list[dict]:
    output = []
    for failure in failures:
        item = {k: v for k, v in failure.items() if k not in {"path", "runtime_path"}}
        item["track_id"] = id_map[item["track_id"]]
        output.append(item)
    return output


def remap_track_ids(value, id_map: dict[str, str]):
    """Recursively replace private track IDs in structured diagnostic files."""
    if isinstance(value, list):
        return [remap_track_ids(item, id_map) for item in value]
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if key == "track_id" and isinstance(item, str):
                output[key] = id_map[item]
            else:
                output[key] = remap_track_ids(item, id_map)
        return output
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--result", action="append", required=True, metavar="MODEL=RESULT_DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-revision")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    id_map = {row["track_id"]: public_id(row) for row in manifest["tracks"]}
    if len(id_map) != len(manifest["tracks"]):
        raise ValueError("Private manifest contains duplicate track IDs")
    if len(set(id_map.values())) != len(id_map):
        raise ValueError("Public track ID collision")
    result_dirs = dict(spec.split("=", 1) for spec in args.result)
    args.output.mkdir(parents=True, exist_ok=True)

    raw_index = {}
    for model, directory in result_dirs.items():
        directory = Path(directory)
        score_path = directory / model / "track_probs.json"
        failure_path = directory / model / "inference_failures.json"
        scores = json.loads(score_path.read_text())
        public_scores = [
            {
                "track_id": id_map[row["track_id"]],
                "source": row["source"],
                "label": row["label"],
                "prob": row["prob"],
            }
            for row in scores
        ]
        failures = json.loads(failure_path.read_text()) if failure_path.exists() else []
        public_failures = remap_failures(failures, id_map)
        assert_public_safe(public_scores, f"{model}.scores")
        assert_public_safe(public_failures, f"{model}.failures")
        model_dir = args.output / "raw" / model
        model_dir.mkdir(parents=True, exist_ok=True)
        public_score_path = model_dir / "track_probs.json"
        public_failure_path = model_dir / "inference_failures.json"
        public_score_path.write_text(json.dumps(public_scores, indent=2) + "\n")
        public_failure_path.write_text(
            json.dumps(public_failures, indent=2, ensure_ascii=False) + "\n"
        )
        raw_index[model] = {
            "scores": str(public_score_path.relative_to(args.output)),
            "scores_sha256": sha256(public_score_path),
            "failures": str(public_failure_path.relative_to(args.output)),
            "failures_sha256": sha256(public_failure_path),
        }
        run_manifest = json.loads((directory / "run_manifest.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        model_info = dict(zip(run_manifest["models"], summary["models"]))[model]
        provenance = {
            "runner_git_revision": args.runner_revision or run_manifest.get("runner_git_revision"),
            "input_manifest_sha256": run_manifest["input_manifest_sha256"],
            "seed": run_manifest["seed"],
            "threshold_during_raw_inference": run_manifest["threshold"],
            "crop_policy": run_manifest["crop_policy"],
            "n_per_source": run_manifest["n_per_source"],
            "environment": run_manifest["environment"],
            "model_info": model_info,
        }
        assert_public_safe(provenance, f"{model}.provenance")
        provenance_path = model_dir / "provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        raw_index[model]["provenance"] = str(provenance_path.relative_to(args.output))
        raw_index[model]["provenance_sha256"] = sha256(provenance_path)
        for diagnostic_name in ("chunk_failures.json", "finite_chunk_policy.json"):
            diagnostic_path = directory / model / diagnostic_name
            if not diagnostic_path.exists():
                continue
            public_diagnostic_path = model_dir / diagnostic_name
            diagnostic = remap_track_ids(json.loads(diagnostic_path.read_text()), id_map)
            assert_public_safe(diagnostic, f"{model}.{diagnostic_name}")
            public_diagnostic_path.write_text(
                json.dumps(diagnostic, indent=2, ensure_ascii=False) + "\n"
            )
            key = diagnostic_name.removesuffix(".json")
            raw_index[model][key] = str(public_diagnostic_path.relative_to(args.output))
            raw_index[model][f"{key}_sha256"] = sha256(public_diagnostic_path)

    metrics = copy.deepcopy(json.loads(args.metrics.read_text()))
    metrics["protocol"]["manifest"] = "artifactbench_v2_primary_manifest.json"
    metrics["protocol"]["runner_git_revision"] = args.runner_revision
    for model, result in metrics["models"].items():
        result["score_file"] = raw_index[model]["scores"]
        result["score_file_sha256"] = raw_index[model]["scores_sha256"]
        result["inference_failures"] = remap_failures(result["inference_failures"], id_map)
        result["missing_scored_track_ids_by_split"] = {
            split: [id_map[track_id] for track_id in track_ids]
            for split, track_ids in result["missing_scored_track_ids_by_split"].items()
        }
    metrics["paired_common_test"]["common_test_track_ids"] = [
        id_map[track_id] for track_id in metrics["paired_common_test"]["common_test_track_ids"]
    ]
    metrics["raw_artifacts"] = raw_index
    assert_public_safe(metrics, "frozen_protocol_metrics")
    metrics_path = args.output / "frozen_protocol_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(metrics_path)


if __name__ == "__main__":
    main()

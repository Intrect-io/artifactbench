#!/usr/bin/env python3
"""Verify that independent model runs used an identical frozen track order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def ordered_id_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    runs = []
    for path in args.run_manifest:
        payload = json.loads(path.read_text())
        ids = payload["selected_track_ids"]
        runs.append(
            {
                "path": str(path.resolve()),
                "models": payload["models"],
                "track_count": len(ids),
                "ordered_track_id_sha256": ordered_id_digest(ids),
                "input_manifest_sha256": payload["input_manifest_sha256"],
            }
        )
    cohort_digests = {run["ordered_track_id_sha256"] for run in runs}
    manifest_digests = {run["input_manifest_sha256"] for run in runs}
    report = {
        "runs": runs,
        "identical_ordered_track_cohort": len(cohort_digests) == 1,
        "identical_input_manifest": len(manifest_digests) == 1,
        "verified": len(cohort_digests) == 1 and len(manifest_digests) == 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

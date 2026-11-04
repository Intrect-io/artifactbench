#!/usr/bin/env python3
# Created: 2026-11-04
# Purpose: Split an inference_failures.json into N shards for parallel runs of
#          scripts/audit_artifactnet_nonfinite.py, and merge the shard audits back into one
#          report with the same schema (summary fields recomputed, original failure order).
# Dependencies: standard library only.
# Test Status: tests/test_musebench_merge_audits.py
"""Shard / merge helper for the ArtifactNet non-finite chunk audit.

    python scripts/musebench/merge_audits.py split --failures F.json --shards 4 --out-dir D
    python scripts/musebench/merge_audits.py merge --failures F.json --out A.json D/audit_*.json
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
import statistics  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402


def split(failures: list[dict], n: int) -> list[list[dict]]:
    """Round-robin over the inference failures only (the audit ignores other stages)."""
    inf = [f for f in failures if f.get("stage") == "inference"]
    return [inf[i::n] for i in range(n)]


def merge(failures: list[dict], audits: list[dict]) -> dict:
    by_id = {}
    for a in audits:
        for t in a["tracks"]:
            if t["track_id"] in by_id:
                raise ValueError(f"track audited twice: {t['track_id']}")
            by_id[t["track_id"]] = t
    wanted = [f["track_id"] for f in failures if f.get("stage") == "inference"]
    missing = [t for t in wanted if t not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} inference failures were not audited (e.g. {missing[0]})")
    extra = set(by_id) - set(wanted)
    if extra:
        raise ValueError(f"{len(extra)} audited tracks are not inference failures")
    tracks = [by_id[t] for t in wanted]
    chunks = [c for t in tracks for c in t["chunks"]]
    nonfinite = [c for c in chunks if not c["finite"]]
    dbfs = [c["rms_dbfs"] for c in nonfinite]
    return {
        "tracks_audited": len(tracks),
        "chunks_audited": len(chunks),
        "nonfinite_chunks": len(nonfinite),
        "tracks_with_at_least_four_finite_chunks": sum(t["valid_chunks"] >= 4 for t in tracks),
        "nonfinite_rms_dbfs": {
            "min": min(dbfs, default=None),
            "median": float(statistics.median(dbfs)) if dbfs else None,
            "max": max(dbfs, default=None),
        },
        "tracks": tracks,
        "merged_from_shards": len(audits),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shard / merge the ArtifactNet non-finite audit.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--failures", type=Path, required=True)
    s.add_argument("--shards", type=int, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--failures", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("audits", type=Path, nargs="+")
    args = ap.parse_args(argv)
    failures = json.loads(args.failures.read_text(encoding="utf-8"))
    if args.cmd == "split":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for i, part in enumerate(split(failures, args.shards)):
            (args.out_dir / f"failures_{i}.json").write_text(json.dumps(part, indent=1) + "\n")
            print(f"shard {i}: {len(part)} failures")
        return 0
    merged = merge(failures, [json.loads(p.read_text(encoding="utf-8")) for p in args.audits])
    args.out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print(f"merged {merged['tracks_audited']} tracks from {len(args.audits)} shards")
    return 0


if __name__ == "__main__":
    sys.exit(main())

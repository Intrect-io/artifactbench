#!/usr/bin/env python3
# Created: 2026-11-04
# Purpose: Assert the on-disk state of each MuseBench pipeline stage (index, acquire, leak, run)
#          under MUSEBENCH_ROOT; exit 0 only when the requested stage is complete and consistent.
# Dependencies: standard library only (later stages import artifactbench from this repo).
# Test Status: tests/test_musebench_verify.py
"""Verify MuseBench pipeline stages.

Usage: ``python -I scripts/musebench/verify.py [--root DIR] {index,acquire,leak,run,all}``

``-I`` keeps the script directory and user site-packages off ``sys.path``; this script
adds the repository root itself so it can import project code without trusting anything
under the data root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.musebench.index_shards import BLOCK, padded, record_end  # noqa: E402

DEFAULT_ROOT = Path(os.environ.get("MUSEBENCH_ROOT", "~/data/musebench")).expanduser()
EXPECTED_PARTS = {1: 15, 2: 15, 3: 15, 4: 15, 5: 15}
# Lower bound on audio files in a complete index: the measured count is 351,183 (paper total
# 361,166 minus the 9,983 Genius-Lyrics human tracks that are not shipped); see docs/MUSEBENCH.md.
MIN_AUDIO_FILES = 351_000
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_audio(path: str) -> bool:
    return path.lower().endswith(AUDIO_EXT)


def verify_index(root: Path, expected_parts=EXPECTED_PARTS,
                 min_audio_files: int = MIN_AUDIO_FILES) -> tuple[list[str], dict]:
    """Return (problems, stats) for the shard index under ``root/index``."""
    problems: list[str] = []
    index = root / "index"
    list_path = index / "shards.json"
    if not list_path.exists():
        return [f"missing {list_path}"], {}
    shards = json.loads(list_path.read_text(encoding="utf-8"))
    per_part = Counter(s["part"] for s in shards)
    if dict(per_part) != dict(expected_parts):
        problems.append(f"shard list per part {dict(per_part)} != expected {dict(expected_parts)}")
    if len({s["file"] for s in shards}) != len(shards):
        problems.append("duplicate shard names in shards.json")
    for part in per_part:
        if len({s["revision"] for s in shards if s["part"] == part}) != 1:
            problems.append(f"part {part}: shards not pinned to a single revision")

    stats = {"shards": len(shards), "done": 0, "members": 0, "files": 0, "audio_files": 0,
             "bytes": 0, "top_level": Counter()}
    seen_paths: set[str] = set()
    for shard in shards:
        stem = shard["file"][:-len(".tar")]
        done_path, jsonl = index / f"{stem}.done", index / f"{stem}.jsonl"
        if not done_path.exists():
            problems.append(f"{stem}: not finished (no .done)")
            continue
        done = json.loads(done_path.read_text(encoding="utf-8"))
        recs = load_jsonl(jsonl) if jsonl.exists() else []
        stats["done"] += 1
        if done.get("members") != len(recs):
            problems.append(f"{stem}: .done members {done.get('members')} != {len(recs)} records")
        if not recs:
            problems.append(f"{stem}: empty index")
            continue
        if recs[0]["offset"] != 0:
            problems.append(f"{stem}: first record at {recs[0]['offset']}, not 0")
        for prev, cur in zip(recs, recs[1:]):
            if cur["offset"] != record_end(prev):
                problems.append(f"{stem}: gap/overlap before {cur['path']!r} at {cur['offset']}")
                break
        end = done.get("end_offset")
        if end != record_end(recs[-1]):
            problems.append(f"{stem}: end_offset {end} != end of last record {record_end(recs[-1])}")
        elif end + 2 * BLOCK > shard["size"]:
            problems.append(f"{stem}: end-of-archive marker does not fit in {shard['size']} bytes")
        for rec in recs:
            if rec["shard"] != shard["file"]:
                problems.append(f"{stem}: record names shard {rec['shard']}")
                break
            if rec["data_offset"] + padded(rec["size"]) > shard["size"]:
                problems.append(f"{stem}: {rec['path']!r} extends past end of shard")
                break
            stats["members"] += 1
            if rec["type"] != "file":
                continue
            if rec["path"] in seen_paths:
                problems.append(f"duplicate member path across shards: {rec['path']!r}")
            seen_paths.add(rec["path"])
            stats["files"] += 1
            stats["bytes"] += rec["size"]
            if is_audio(rec["path"]):
                stats["audio_files"] += 1
            stats["top_level"][rec["path"].split("/", 1)[0]] += 1
    if stats["done"] == len(shards) and stats["audio_files"] < min_audio_files:
        problems.append(f"only {stats['audio_files']} audio files indexed (< {min_audio_files})")
    stats["top_level"] = dict(sorted(stats["top_level"].items()))
    return problems, stats


def not_implemented(stage: str):
    def check(root: Path) -> tuple[list[str], dict]:
        return [f"stage '{stage}' has no verifier yet (later milestone)"], {}
    return check


STAGES = {
    "index": verify_index,
    "acquire": not_implemented("acquire"),
    "leak": not_implemented("leak"),
    "run": not_implemented("run"),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify MuseBench pipeline stages.")
    ap.add_argument("stage", choices=[*STAGES, "all"])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--json", action="store_true", help="print stats as JSON")
    args = ap.parse_args(argv)
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    failed = False
    for stage in stages:
        problems, stats = STAGES[stage](args.root)
        if args.json:
            print(json.dumps({"stage": stage, "problems": problems, "stats": stats}, indent=1))
        else:
            brief = {k: v for k, v in stats.items() if k != "top_level"}
            print(f"[{stage}] {'OK' if not problems else 'FAIL'} {brief}")
            for p in problems[:50]:
                print(f"  - {p}")
            if len(problems) > 50:
                print(f"  ... {len(problems) - 50} more")
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

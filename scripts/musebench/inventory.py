#!/usr/bin/env python3
# Created: 2026-11-04
# Purpose: Summarize a MuseBench shard index by folder (file counts, bytes, extensions, shards).
# Dependencies: standard library only.
# Test Status: tests/test_musebench_inventory.py
"""Summarize the MuseBench shard index.

Groups regular files by their first ``--depth`` path components (default 2, e.g.
``Musicgen_Generated_Music/Jamedo``) and prints a Markdown table, or JSON with ``--json``.
"""
from __future__ import annotations

import os as _os
import sys as _sys

# This directory holds select.py, which shadows the stdlib ``select`` module when the
# directory is on sys.path (``python scripts/musebench/x.py`` without -I). Drop it.
if _sys.path and _os.path.abspath(_sys.path[0] or ".") == _os.path.dirname(_os.path.abspath(__file__)):
    _sys.path.pop(0)


import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("MUSEBENCH_ROOT", "~/data/musebench")).expanduser()


def iter_records(index_dir: Path):
    for jsonl in sorted(index_dir.glob("*.jsonl")):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def group_key(path: str, depth: int) -> str:
    parts = path.split("/")
    # Files directly under a folder at a shallower level keep their folder as the key.
    return "/".join(parts[:min(depth, len(parts) - 1)]) or "."


def summarize(records, depth: int = 2) -> dict[str, dict]:
    groups: dict[str, dict] = defaultdict(lambda: {"files": 0, "bytes": 0, "ext": Counter(),
                                                   "shards": set()})
    for rec in records:
        if rec.get("type") != "file":
            continue
        g = groups[group_key(rec["path"], depth)]
        g["files"] += 1
        g["bytes"] += rec["size"]
        name = rec["path"].rsplit("/", 1)[-1]
        g["ext"][name.rsplit(".", 1)[-1].lower() if "." in name else ""] += 1
        g["shards"].add(rec["shard"])
    out = {}
    for key in sorted(groups):
        g = groups[key]
        out[key] = {"files": g["files"], "bytes": g["bytes"], "ext": dict(g["ext"].most_common()),
                    "shards": sorted(g["shards"])}
    return out


def to_markdown(summary: dict[str, dict]) -> str:
    lines = ["| Folder | Files | GB | Extensions | Shards |", "|---|---:|---:|---|---:|"]
    total_files = total_bytes = 0
    for key, g in summary.items():
        ext = ", ".join(f"{e or '(none)'} {n}" for e, n in g["ext"].items())
        lines.append(f"| `{key}` | {g['files']:,} | {g['bytes'] / 1e9:.1f} | {ext} | {len(g['shards'])} |")
        total_files += g["files"]
        total_bytes += g["bytes"]
    lines.append(f"| **Total** | **{total_files:,}** | **{total_bytes / 1e9:.1f}** | | |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize the MuseBench shard index by folder.")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    summary = summarize(iter_records(args.root / "index"), args.depth)
    print(json.dumps(summary, indent=1) if args.json else to_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

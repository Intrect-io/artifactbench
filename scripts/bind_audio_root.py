#!/usr/bin/env python3
"""Bind a path-free ArtifactBench manifest to locally acquired audio by SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

AUDIO_SUFFIXES = {
    ".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    tracks = payload["tracks"]
    wanted = {row["audio_sha256"] for row in tracks}
    found: dict[str, list[str]] = defaultdict(list)
    scanned = 0
    for root in args.audio_root:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            scanned += 1
            digest = sha256(path)
            if digest in wanted:
                found[digest].append(str(path.resolve()))

    runtime_tracks = []
    missing = []
    duplicates = {}
    for row in tracks:
        item = dict(row)
        paths = sorted(found.get(row["audio_sha256"], []))
        if paths:
            item["runtime_path"] = paths[0]
            if len(paths) > 1:
                duplicates[row["track_id"]] = paths
        else:
            missing.append(row["track_id"])
        runtime_tracks.append(item)

    runtime = dict(payload)
    runtime["tracks"] = runtime_tracks
    runtime["runtime_binding"] = {
        "manifest_sha256": sha256(args.manifest),
        "audio_roots": [str(path.resolve()) for path in args.audio_root],
        "scanned_audio_files": scanned,
        "resolved_tracks": len(tracks) - len(missing),
        "missing_tracks": len(missing),
    }
    report = {
        **runtime["runtime_binding"],
        "missing_track_ids": missing,
        "duplicate_byte_locations": duplicates,
        "ready": not missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(runtime, indent=2, ensure_ascii=False) + "\n")
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("scanned_audio_files", "resolved_tracks", "missing_tracks", "ready")}, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

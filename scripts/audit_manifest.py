#!/usr/bin/env python3
"""Audit ArtifactBench manifest consistency without reading audio bytes.

The tool accepts any number of JSON manifests and writes a deterministic JSON
report. It supports the historical ``bench`` list, v1.1 ``tracks`` manifests,
and the metadata-first v2 public schema.
It intentionally does not infer that a local path proves public availability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("track_id", "label", "source", "bench_origin")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_rows(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    for key in ("bench", "tracks", "test"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return key, rows
    raise ValueError("manifest has no supported row list: bench, tracks, or test")


def duplicates(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        value = row.get(field)
        if value not in (None, ""):
            grouped[str(value)].append(index)
    return [
        {"value": value, "count": len(indices), "indices": indices}
        for value, indices in sorted(grouped.items())
        if len(indices) > 1
    ]


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "<missing>")) for row in rows).items()))


def audit(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    row_key, rows = get_rows(payload)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if not metadata and payload.get("schema_version") == 2:
        labels = payload.get("labels") if isinstance(payload.get("labels"), dict) else {}
        sources = payload.get("sources") if isinstance(payload.get("sources"), dict) else {}
        metadata = {
            "total": payload.get("total"),
            "ai": labels.get("ai"),
            "real": labels.get("real"),
            "n_sources": len(sources) if sources else None,
            "status": "public_release",
        }

    missing_fields = {
        field: [index for index, row in enumerate(rows) if row.get(field) in (None, "")]
        for field in REQUIRED_FIELDS
    }
    missing_fields = {field: indices for field, indices in missing_fields.items() if indices}

    actual = {
        "rows": len(rows),
        "labels": count_by(rows, "label"),
        "bench_origin": count_by(rows, "bench_origin"),
        "sources": count_by(rows, "source"),
        "generators": count_by(rows, "generator"),
    }
    expected = {
        key: metadata.get(key)
        for key in ("total", "ai", "real", "from_test", "from_train", "n_sources", "n_generators")
        if key in metadata
    }
    comparisons = {
        "total": (expected.get("total"), actual["rows"]),
        "ai": (expected.get("ai"), actual["labels"].get("ai", 0)),
        "real": (expected.get("real"), actual["labels"].get("real", 0)),
        "from_test": (expected.get("from_test"), actual["bench_origin"].get("test", 0)),
        "from_train": (expected.get("from_train"), actual["bench_origin"].get("train", 0)),
        "n_sources": (expected.get("n_sources"), len(actual["sources"])),
    }
    comparison_status = {
        key: "missing_metadata" if pair[0] is None else ("match" if pair[0] == pair[1] else "mismatch")
        for key, pair in comparisons.items()
    }

    absolute_paths = [
        index for index, row in enumerate(rows)
        if isinstance(row.get("path"), str) and row["path"].startswith("/")
    ]
    public_identity_fields = ("content_sha256", "audio_sha256", "lineage_id")
    identity_coverage = {
        field: sum(1 for row in rows if row.get(field) not in (None, ""))
        for field in public_identity_fields
    }

    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "row_key": row_key,
        "metadata": metadata,
        "actual": actual,
        "comparisons": {
            key: {"expected": pair[0], "actual": pair[1], "status": comparison_status[key]}
            for key, pair in comparisons.items()
        },
        "missing_required_fields": missing_fields,
        "duplicate_track_ids": duplicates(rows, "track_id"),
        "duplicate_paths": duplicates(rows, "path"),
        "absolute_path_rows": len(absolute_paths),
        "identity_coverage": identity_coverage,
        "release_ready": (
            all(status in ("match", "missing_metadata") for status in comparison_status.values())
            and not missing_fields
            and not duplicates(rows, "track_id")
            and not absolute_paths
            and "candidate" not in str(metadata.get("status", "")).lower()
            and not metadata.get("remaining_gates")
            and identity_coverage["lineage_id"] == len(rows)
            and (
                identity_coverage["content_sha256"] == len(rows)
                or identity_coverage["audio_sha256"] == len(rows)
            )
        ),
    }


def load_named_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open() as handle:
        payload = json.load(handle)
    _, rows = get_rows(payload)
    return {
        str(row["track_id"]): row
        for row in rows
        if row.get("track_id") not in (None, "")
    }


def compare_manifests(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = load_named_rows(left_path)
    right = load_named_rows(right_path)
    left_ids = set(left)
    right_ids = set(right)
    left_only = sorted(left_ids - right_ids)
    right_only = sorted(right_ids - left_ids)
    common = sorted(left_ids & right_ids)

    def source_counts(ids: list[str], rows: dict[str, dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(str(rows[item].get("source", "<missing>")) for item in ids).items()))

    fields = ("label", "source", "bench_origin", "generator")
    conflicts = {
        field: [
            track_id for track_id in common
            if left[track_id].get(field) != right[track_id].get(field)
        ]
        for field in fields
    }
    conflicts = {field: ids for field, ids in conflicts.items() if ids}

    return {
        "left": str(left_path.resolve()),
        "right": str(right_path.resolve()),
        "common_track_ids": len(common),
        "left_only": {
            "count": len(left_only),
            "by_source": source_counts(left_only, left),
            "track_ids": left_only,
        },
        "right_only": {
            "count": len(right_only),
            "by_source": source_counts(right_only, right),
            "track_ids": right_only,
        },
        "common_field_conflicts": conflicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "schema_version": 1,
        "manifests": [audit(path) for path in args.manifests],
        "pairwise_comparisons": [
            compare_manifests(args.manifests[left], args.manifests[right])
            for left in range(len(args.manifests))
            for right in range(left + 1, len(args.manifests))
        ],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

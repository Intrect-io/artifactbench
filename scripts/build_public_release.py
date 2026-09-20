#!/usr/bin/env python3
"""Build a path-free, title-free metadata release from the frozen protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SPLIT_SALT = "artifactbench-v2-primary-20260920"


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def public_id(row: dict) -> str:
    return f"ab2-{row['audio_sha256'][:20]}"


def public_lineage_id(row: dict) -> str:
    """Preserve lineage grouping without publishing the private lineage key."""
    lineage = row.get("lineage_id") or row["audio_sha256"]
    return f"ab2l-{digest_text(lineage)[:20]}"


def split_rank_sha256(row: dict) -> str:
    lineage = row.get("lineage_id") or row["audio_sha256"]
    return digest_text(f"{SPLIT_SALT}:{row['source']}:{lineage}")


def public_row(row: dict) -> dict:
    audio_sha = row["audio_sha256"]
    result = {
        "track_id": public_id(row),
        "label": row["label"],
        "source": row["source"],
        "generator": row.get("generator"),
        "generator_family": row.get("generator_family"),
        "generator_version": row.get("generator_version"),
        "bench_origin": row.get("bench_origin", "test"),
        "protocol_split": row["protocol_split"],
        "audio_sha256": audio_sha,
        "audio_bytes": row.get("audio_bytes"),
        "audio_format": row.get("audio_format"),
        "lineage_id": public_lineage_id(row),
        "split_rank_sha256": split_rank_sha256(row),
        "legacy_track_id_sha256": digest_text(row["track_id"]),
        "retrieval": {
            "mode": "digest_match_from_upstream",
            "upstream": "intrect/artifactbench-v1",
        },
    }
    source = row["source"]
    if source.startswith("aime_"):
        result["retrieval"] = {
            "mode": "digest_match_from_upstream",
            "upstream": "disco-eth/AIME",
        }
    elif source == "fma_hardneg":
        fma_id = row["track_id"].rsplit("_", 1)[-1]
        if not (len(fma_id) == 6 and fma_id.isdigit()):
            raise ValueError(f"Cannot recover FMA ID from {row['track_id']}")
        result["retrieval"] = {
            "mode": "official_archive_track_id",
            "upstream": "mdeff/fma:fma_large.zip",
            "fma_track_id": fma_id,
        }
    elif source == "youtube_hardneg":
        video_id = row["track_id"].split("_", 1)[-1]
        result["retrieval"] = {
            "mode": "source_reference_only_no_audio_redistribution",
            "upstream": "youtube",
            "video_id": video_id,
        }
    elif source.startswith(("suno_", "udio_")):
        result["retrieval"] = {
            "mode": "digest_reference_only_no_audio_redistribution",
            "upstream": source.split("_", 1)[0],
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fma-licenses", type=Path)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    rows = [public_row(row) for row in source["tracks"]]
    ids = [row["track_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Public ID collision")
    forbidden = ("runtime_path", "path", "title", "filename", "channel")
    serialized_rows = json.dumps(rows, ensure_ascii=False).lower()
    for key in forbidden:
        if f'"{key}"' in serialized_rows:
            raise ValueError(f"Forbidden public key: {key}")
    release = {
        "schema_version": 2,
        "name": "ArtifactBench v2 primary frozen protocol",
        "license": "CC-BY-NC-4.0",
        "license_scope": "benchmark-authored metadata and results only",
        "release_policy": "metadata_only_no_audio_bundle",
        "protocol": {
            key: source["metadata"]["protocol"][key]
            for key in (
                "schema_version",
                "salt",
                "unit",
                "stratification",
                "target_ratios",
                "split_counts",
                "label_split_counts",
            )
        },
        "total": len(rows),
        "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
        "splits": dict(sorted(Counter(row["protocol_split"] for row in rows).items())),
        "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        "tracks": sorted(rows, key=lambda row: row["track_id"]),
    }
    if release["protocol"]["salt"] != SPLIT_SALT:
        raise ValueError("Protocol salt differs from public split-rank salt")
    release["protocol"]["rank_field"] = "split_rank_sha256"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(release, indent=2, ensure_ascii=False) + "\n")
    if args.fma_licenses:
        license_evidence = json.loads(args.fma_licenses.read_text())
        legacy_to_public = {
            row["track_id"]: public_id(row) for row in source["tracks"]
        }
        public_licenses = {
            "metadata_source": license_evidence["metadata_source"],
            "metadata_zip_sha1": license_evidence["metadata_zip_sha1"],
            "tracks": [
                {
                    "track_id": legacy_to_public[row["artifactbench_track_id"]],
                    "fma_track_id": row["track_id"],
                    "license": row["license"],
                }
                for row in license_evidence["tracks"]
            ],
        }
        license_path = args.output.with_name("fma_track_licenses.json")
        license_path.write_text(
            json.dumps(public_licenses, indent=2, ensure_ascii=False) + "\n"
        )
        print(license_path)
    print(args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Created: 2026-11-05
# Purpose: Evaluation cohorts for MuseBench reports: a subset of the eval manifest named by track id (for example a
#          stratified pilot sample), shared by import_scores.py and report.py so both restrict to the same rows and
#          a report records which cohort it describes.
# Dependencies: standard library only.
# Test Status: tests/test_musebench_cohort.py
"""A cohort file is JSON: either a list of track ids, or an object with a ``track_ids`` list.

The cohort fingerprint is the sha256 of the sorted, newline-joined ids, so the same set of tracks has the same
fingerprint whichever file or order it came from; compare.py relies on it to refuse mismatched reports.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load(path: Path) -> list[str]:
    """Track ids of a cohort file, in file order. Raises ValueError on a malformed, empty or duplicated list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = data.get("track_ids") if isinstance(data, dict) else data
    if not isinstance(ids, list) or not all(isinstance(t, str) and t for t in ids):
        raise ValueError(f"{path}: a cohort is a JSON list of track ids or an object with a 'track_ids' list")
    if not ids:
        raise ValueError(f"{path}: the cohort is empty")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicated track ids in the cohort")
    return ids


def fingerprint(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def info(path: Path, ids: list[str]) -> dict:
    return {"path": str(path), "size": len(ids), "sha256": fingerprint(ids)}


def restrict(eval_rows: list[dict], ids: list[str]) -> list[dict]:
    """Eval rows of the cohort, in eval-manifest order. Every cohort id must be an eval row."""
    wanted = set(ids)
    rows = [r for r in eval_rows if r["track_id"] in wanted]
    if len(rows) != len(wanted):
        unknown = sorted(wanted - {r["track_id"] for r in rows})
        raise ValueError(f"{len(unknown)} cohort track ids are not in the eval manifest, e.g. {unknown[:5]}")
    return rows


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

import os as _os
import sys as _sys

# This directory holds select.py, which shadows the stdlib ``select`` module when the
# directory is on sys.path (``python scripts/musebench/x.py`` without -I). Drop it.
if _sys.path and _os.path.abspath(_sys.path[0] or ".") == _os.path.dirname(_os.path.abspath(__file__)):
    _sys.path.pop(0)


import argparse
import hashlib
import json
import os
import shutil
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


REQUIRED_ROW_FIELDS = ("track_id", "path", "label", "source", "generator", "variant",
                       "caption_set", "musebench_role")
MANIFEST_REL = Path("manifests") / "musebench_runtime_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_acquire(root: Path, sha_sample: int = 300, full_sha: bool = False,
                   free_floor_gb: float | None = None) -> tuple[list[str], dict]:
    """Check selection determinism, fetched files, and the runtime manifest."""
    from artifactbench.data.manifest import load_manifest
    from scripts.musebench import select as sel_mod

    problems: list[str] = []
    sel_path = root / "selection" / "selection.json"
    man_path = root / MANIFEST_REL
    for p in (sel_path, man_path):
        if not p.exists():
            problems.append(f"missing {p}")
    if problems:
        return problems, {}
    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    if selection.get("unclassified"):
        problems.append(f"{len(selection['unclassified'])} unclassified index members")

    # 1. The selection is reproducible from the index and its recorded parameters.
    replanned = sel_mod.plan(sel_mod.load_index(root), seed=selection["seed"],
                             eval_caps=selection["eval_caps"],
                             train_tiers=selection["train_tiers"],
                             train_budget_bytes=selection["train_budget_bytes"])
    if sel_mod.selection_fingerprint(replanned) != sel_mod.selection_fingerprint(selection):
        problems.append("selection.json is not reproducible from the index (fingerprint differs)")

    # 2. The manifest loads through the runner's loader and covers the selection exactly.
    entries, by_source = load_manifest(str(man_path), split="all")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    if manifest.get("metadata", {}).get("selection_fingerprint") != sel_mod.selection_fingerprint(selection):
        problems.append("manifest was built from a different selection")
    rows = {r["path"]: r for r in selection["rows"]}
    seen_ids: set[str] = set()
    man_paths = set()
    for e in entries:
        missing = [k for k in REQUIRED_ROW_FIELDS if e.get(k) in (None, "")]
        if missing:
            problems.append(f"{e.get('track_id')}: missing fields {missing}")
            continue
        if e["track_id"] in seen_ids:
            problems.append(f"duplicate track_id {e['track_id']}")
        seen_ids.add(e["track_id"])
        if e["label"] not in ("ai", "real") or e["musebench_role"] not in ("eval", "train_pool"):
            problems.append(f"{e['track_id']}: bad label/role {e['label']}/{e['musebench_role']}")
        man_paths.add(e["musebench_path"])
        r = rows.get(e["musebench_path"])
        if r is None:
            problems.append(f"manifest row not in selection: {e['musebench_path']!r}")
            continue
        if r["musebench_role"] != e["musebench_role"]:
            problems.append(f"{e['track_id']}: role differs from selection")
        p = Path(e["path"])
        if not p.is_file():
            problems.append(f"missing audio {p}")
        elif p.stat().st_size != r["size"]:
            problems.append(f"size mismatch {p}: {p.stat().st_size} != index {r['size']}")
    absent = set(rows) - man_paths
    if absent:
        problems.append(f"{len(absent)} selected members are not in the manifest")

    # 3. Content: SHA-256 of a deterministic sample (or all) equals the ledger/manifest value.
    ledger = sel_mod.load_ledger(root)
    ordered = sorted(entries, key=lambda e: e["track_id"])
    sample = ordered if full_sha else ordered[::max(1, len(ordered) // max(1, sha_sample))]
    for e in sample:
        p = Path(e["path"])
        if not p.is_file():
            continue
        digest = sha256_file(p)
        led = ledger.get(e["musebench_path"], {}).get("sha256")
        if digest != e.get("sha256") or digest != led:
            problems.append(f"sha256 mismatch {p}")
    for m in selection["meta"]:
        mp = root / "meta" / m["path"]
        if not mp.is_file() or mp.stat().st_size != m["size"]:
            problems.append(f"missing or wrong-size metadata file {mp}")

    # 4. Cohort shape: eval counts as planned, roles disjoint by family.
    eval_counts = Counter(r["group"] for r in selection["rows"] if r["musebench_role"] == "eval")
    expected = {k: v for k, v in selection["expected_eval_counts"].items() if v}
    if dict(eval_counts) != expected:
        problems.append("eval counts differ from expected_eval_counts")
    eval_fams = {r["family"] for r in selection["rows"] if r["musebench_role"] == "eval"}
    straddle = [r["path"] for r in selection["rows"]
                if r["musebench_role"] == "train_pool" and r["family"] in eval_fams]
    if straddle:
        problems.append(f"{len(straddle)} train_pool rows share a family with eval rows")

    # 5. Disk policy.
    budget = selection.get("budget", {})
    floor = budget.get("free_floor_gb", 80.0) if free_floor_gb is None else free_floor_gb
    free = shutil.disk_usage(root).free
    if free < floor * 1e9:
        problems.append(f"free space {free / 1e9:.1f} GB below floor {floor} GB")
    kept = sum(r["size"] for r in selection["rows"])
    if "budget_gb" in budget and kept > budget["budget_gb"] * 1e9:
        problems.append(f"kept audio {kept / 1e9:.1f} GB exceeds budget {budget['budget_gb']} GB")

    stats = {"tracks": len(entries), "sources": len(by_source),
             "roles": dict(Counter(e.get("musebench_role") for e in entries)),
             "kept_gb": round(kept / 1e9, 1), "free_gb": round(free / 1e9, 1),
             "sha_checked": len(sample)}
    return problems, stats


def verify_leak(root: Path) -> tuple[list[str], dict]:
    """Check the leak report and that every manifest row carries a boolean seen_in_training."""
    from scripts.musebench import leak_map as lm

    problems: list[str] = []
    rep_path, man_path = root / lm.REPORT_REL, root / MANIFEST_REL
    for p in (rep_path, man_path):
        if not p.exists():
            problems.append(f"missing {p}")
    if problems:
        return problems, {}
    report = json.loads(rep_path.read_text(encoding="utf-8"))
    tracks = json.loads(man_path.read_text(encoding="utf-8"))["tracks"]

    if not report.get("model", {}).get("name"):
        problems.append("leak report does not name the model")
    manifests = report.get("training_manifests") or []
    if not manifests or not all(m.get("name") for m in manifests):
        problems.append("leak report does not name the training manifest(s) it compared against")
    for key in ("suno_udio", "fma", "jamendo", "inheritance", "scope"):
        if not report.get("method", {}).get(key):
            problems.append(f"leak report method lacks '{key}'")

    bad = [t["track_id"] for t in tracks if not isinstance(t.get("seen_in_training"), bool)]
    if bad:
        problems.append(f"{len(bad)} manifest rows lack a boolean seen_in_training (e.g. {bad[0]})")
    incoherent = [t["track_id"] for t in tracks if isinstance(t.get("seen_in_training"), bool)
                  and t["seen_in_training"] != (t.get("seen_evidence") is not None)]
    if incoherent:
        problems.append(f"{len(incoherent)} rows: seen_in_training disagrees with seen_evidence")
    if report.get("manifest_fingerprint") != lm.manifest_fingerprint(tracks):
        problems.append("leak report was built from a different manifest")
    if not bad:
        recount = lm.per_source_counts(tracks)
        if recount != report.get("per_source"):
            problems.append("per-source seen/unseen counts differ from the manifest")
        if sum(report.get("totals", {}).values()) != len(tracks):
            problems.append("report totals do not add up to the manifest size")
    seen = sum(1 for t in tracks if t.get("seen_in_training") is True)
    stats = {"tracks": len(tracks), "seen": seen, "unseen": len(tracks) - seen,
             "eval_seen": sum(1 for t in tracks if t.get("seen_in_training") is True
                              and t.get("musebench_role") == "eval"),
             "training_manifests": len(manifests)}
    return problems, stats


def not_implemented(stage: str):
    def check(root: Path) -> tuple[list[str], dict]:
        return [f"stage '{stage}' has no verifier yet (later milestone)"], {}
    return check


STAGES = {
    "index": verify_index,
    "acquire": verify_acquire,
    "leak": verify_leak,
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

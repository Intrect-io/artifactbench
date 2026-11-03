#!/usr/bin/env python3
# Created: 2026-11-04
# Purpose: Classify every MuseBench shard member, choose a deterministic sealed eval cohort and a
#          disk-budgeted train pool from the shard index, and (after fetch.py) write the runtime
#          manifest that artifactbench.data.manifest.load_manifest accepts.
# Dependencies: standard library only.
# Test Status: tests/test_musebench_select.py
"""Select MuseBench members and build the runtime manifest.

Subcommands::

    python -I scripts/musebench/select.py plan      # index -> selection/selection.json
    python -I scripts/musebench/select.py manifest  # selection + fetch ledger -> manifest

Determinism: every choice is ordered by ``sha256(seed:family)``, where *family* is the source
track a member derives from (``caption_set/base name``). A Jamendo song, its co-created remix,
every AI generation conditioned on its caption and every transform of those generations share
one family, so

* the eval cohort prefers the same families across generators and transforms (paired
  clean/transform and human/co-created comparisons), and
* the train pool excludes every family that has any eval member (no family straddles roles).

Clean AI eval tracks are drawn from the tracks that have transformed variants: MuseBench
applied transforms to its test split only, so this approximates the paper's test split.
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
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

DEFAULT_ROOT = Path(os.environ.get("MUSEBENCH_ROOT", "~/data/musebench")).expanduser()
SEED = "musebench-20261104"
GB = 10 ** 9

GENERATORS = {  # folder name (lower case) -> canonical generator id
    "magnet_generated_music": "magnet",
    "musicgen_generated_music": "musicgen",
    "audioldm_generated_music": "audioldm2",
    "jasco_generated_music": "jasco",
    # The card calls it "Magenta"; its 19,903 files match the paper's Lyria RealTime row.
    "magenta_generated_music": "lyria_rt",
    "mustango_generated_music": "mustango",
    "riffusion_generated_music": "riffusion",
    "stable_audio_generated_music": "stable_audio_open",
    "suno_generated_music": "suno",
    "udio_generated_music": "udio",
}
TRANSFORMS = {
    "aac_generated_music": "aac",
    "mp3_noised_music": "mp3",
    "ogg_noised_music": "ogg",
    "eq_generated_music": "eq",
    "pitch_time_generated_music": "pitch_time",
    "loudness_generation_music": "loudness",
    "gaussion_noised_music": "gaussian",
    "pgd_generated_music": "pgd",
}
TRANSFORM_SUFFIX = {
    "aac": re.compile(r"_(aac\d+)$"),
    "mp3": re.compile(r"_(mp3\d+)$"),
    "ogg": re.compile(r"_(ogg\d+)$"),
    "eq": re.compile(r"_(eq-(?:cut|boost)_(?:low|mid|high))$"),
    "pitch_time": re.compile(r"_(pt-ps[+-]\d+st-ts[\d.]+)$"),
    "loudness": re.compile(r"_(lnLUFS-?\d+)$"),
    "gaussian": re.compile(r"_(gn[\d.]+)$"),
}
CAPTION_SETS = {"genius-lyrics": "genius", "jamedo": "jamendo", "jamendo": "jamendo"}
META_PREFIXES = ("Caption/", "Genius-Lyrics/")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

# Eval cohort caps (per source group). See docs/MUSEBENCH.md for the rationale.
EVAL_CAPS = {"human_jamendo": 3000, "human_fma_ood": 3000, "clean_per_generator": 1000,
             "co_created": 1000, "transform_per_generator": 200}
# Train pool: tiers filled in order, each source up to its cap, until the byte budget binds.
TRAIN_TIERS = [
    {"audioldm2": 5000, "magnet": 5000, "jasco": 5000, "mustango": 5000, "lyria_rt": 5000},
    {"co_created": 5000},
    {"human_jamendo": 3000, "human_fma_ood": 2000},
    {"musicgen": 2000, "riffusion": 2000, "stable_audio_open": 2000},
    {"suno": 0, "udio": 0},  # already on local disk as SONICS
]


class UnsafePath(ValueError):
    pass


# --------------------------------------------------------------------------- classification

def strip_audio_ext(name: str) -> str:
    for ext in (".mp3.wav", ".wav", ".mp3", ".m4a", ".ogg", ".flac"):
        if name.lower().endswith(ext):
            return name[:-len(ext)]
    return name


def classify(path: str) -> dict | None:
    """Return classification fields for a member path, or None if it is not recognised."""
    parts = path.split("/")
    name = parts[-1]
    top = parts[0]
    if path.endswith(".csv") and (path.startswith(META_PREFIXES) or top == "Jamendo"):
        return {"kind": "meta"}
    if not name.lower().endswith(AUDIO_EXT):
        return None
    if top == "Jamendo" and len(parts) == 2:
        base = strip_audio_ext(name)
        return _row("human", "human_jamendo", "real", "human", "clean", "jamendo", base)
    if top == "OOD FMA" and len(parts) == 2:
        base = strip_audio_ext(name)
        return _row("human", "human_fma_ood", "real", "human", "clean", "fma", base)
    if top == "ai_human_co-created_music" and len(parts) == 3:
        base = strip_audio_ext(name)
        base = base[:-len("_edited")] if base.endswith("_edited") else base
        cset = CAPTION_SETS.get(parts[1].lower())
        if cset is None:
            return None
        return _row("ai", "co_created", "ai", "musicgen_style", "co_created", cset, base)
    gen = GENERATORS.get(top.lower())
    if gen is not None:
        if gen in ("suno", "udio") and len(parts) == 2:
            return _row("ai", gen, "ai", gen, "clean", "none", strip_audio_ext(name))
        if len(parts) == 3 and parts[1].lower() in CAPTION_SETS:
            return _row("ai", gen, "ai", gen, "clean", CAPTION_SETS[parts[1].lower()],
                        strip_audio_ext(name))
        return None
    variant = TRANSFORMS.get(top)
    if variant is None or len(parts) < 3:
        return None
    gen = GENERATORS.get(parts[1].lower())
    if gen is None:
        return None
    if len(parts) == 4 and parts[2].lower() in CAPTION_SETS:
        cset = CAPTION_SETS[parts[2].lower()]
    elif len(parts) == 4 and parts[2] == "root" and gen in ("suno", "udio"):
        cset = "none"  # standalone generators: transforms sit under a literal "root/" folder
    elif len(parts) == 3:
        cset = "none" if gen in ("suno", "udio") else "*"  # pgd has no caption-set level
    else:
        return None
    stem = strip_audio_ext(name)
    if variant == "pgd":
        param = "pgd"
        base = stem[len("adv_"):] if stem.startswith("adv_") else stem
    else:
        m = TRANSFORM_SUFFIX[variant].search(stem)
        if m is None:
            return None
        param, base = m.group(1), stem[:m.start()]
    row = _row("ai", f"{gen}__{variant}", "ai", gen, variant, cset, base)
    row["transform_param"] = param
    return row


def _row(kind, group, label, generator, variant, caption_set, base):
    return {"kind": kind, "group": group, "label": label, "generator": generator,
            "variant": variant, "caption_set": caption_set, "base_key": base}


def source_name(group: str) -> str:
    return f"musebench_{group}"


def rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8", "surrogateescape")).hexdigest()


def family_of(row: dict) -> str:
    """Source-track identity shared by a song, its remix, generations and their transforms."""
    cset = row["caption_set"]
    if row["generator"] in ("suno", "udio"):
        return f"{row['generator']}/{row['base_key']}"  # standalone tracks: own family
    if cset == "fma":
        return f"fma/{row['base_key']}"
    return f"{cset}/{row['base_key']}"


def safe_relpath(path: str) -> str:
    """Reject member names that could escape the data root."""
    if not path or "\0" in path or path.startswith("/") or "\\" in path:
        raise UnsafePath(path)
    parts = PurePosixPath(path).parts
    if any(p in ("", ".", "..") for p in parts) or path.split("/") != list(parts):
        raise UnsafePath(path)
    return path


# --------------------------------------------------------------------------- planning

def load_index(root: Path) -> list[dict]:
    recs = []
    for jsonl in sorted((root / "index").glob("*.jsonl")):
        with open(jsonl, encoding="utf-8") as f:
            recs.extend(json.loads(line) for line in f if line.strip())
    return recs


def classify_index(records: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """Split index records into (audio rows, meta records, unclassified paths)."""
    audio, meta, unknown = [], [], []
    for rec in records:
        if rec.get("type") != "file":
            continue
        cls = classify(rec["path"])
        if cls is None:
            unknown.append(rec["path"])
            continue
        if cls["kind"] == "meta":
            meta.append(rec)
            continue
        row = {**cls, "path": rec["path"], "shard": rec["shard"],
               "data_offset": rec["data_offset"], "size": rec["size"]}
        row["family"] = family_of(row)
        audio.append(row)
    return audio, meta, unknown


def _resolve_pgd_caption_sets(rows: list[dict]) -> None:
    """PGD files lack the caption-set folder; borrow it from the clean track with that base."""
    clean_sets = defaultdict(set)
    for r in rows:
        if r["variant"] == "clean" and r["kind"] == "ai":
            clean_sets[(r["generator"], r["base_key"])].add(r["caption_set"])
    for r in rows:
        if r["caption_set"] == "*":
            sets = clean_sets.get((r["generator"], r["base_key"]), set())
            r["caption_set"] = sets.pop() if len(sets) == 1 else "unknown"
            r["family"] = family_of(r)


def plan(records: list[dict], seed: str = SEED, eval_caps: dict = EVAL_CAPS,
         train_tiers: list[dict] = TRAIN_TIERS, train_budget_bytes: int = 85 * GB) -> dict:
    audio, meta, unknown = classify_index(records)
    _resolve_pgd_caption_sets(audio)
    fam_rank = {r["family"]: rank(seed, r["family"]) for r in audio}

    def by_rank(rows):
        return sorted(rows, key=lambda r: (fam_rank[r["family"]], r["path"]))

    groups = defaultdict(list)
    for r in audio:
        groups[r["group"]].append(r)

    eval_rows: list[dict] = []
    expected: dict[str, int] = {}

    def take(group: str, rows: list[dict], cap: int) -> list[dict]:
        chosen = by_rank(rows)[:cap]
        expected[group] = len(chosen)
        eval_rows.extend(chosen)
        return chosen

    take("human_jamendo", groups["human_jamendo"], eval_caps["human_jamendo"])
    take("human_fma_ood", groups["human_fma_ood"], eval_caps["human_fma_ood"])
    take("co_created", groups["co_created"], eval_caps["co_created"])

    # Families that have transformed variants per generator ~ the paper's test split.
    transformed = defaultdict(set)
    for r in audio:
        if r["variant"] not in ("clean", "co_created"):
            transformed[r["generator"]].add(r["family"])
    clean_eval_families: dict[str, set] = {}
    for gen in sorted(set(GENERATORS.values())):
        rows = groups.get(gen, [])
        test_like = [r for r in rows if r["family"] in transformed[gen]]
        rest = [r for r in rows if r["family"] not in transformed[gen]]
        cap = eval_caps["clean_per_generator"]
        chosen = by_rank(test_like)[:cap]
        if len(chosen) < cap:
            chosen += by_rank(rest)[:cap - len(chosen)]
        expected[gen] = len(chosen)
        eval_rows.extend(chosen)
        clean_eval_families[gen] = {r["family"] for r in chosen}

    for group in sorted(g for g in groups if "__" in g):
        gen = group.split("__")[0]
        rows = groups[group]
        paired = [r for r in rows if r["family"] in clean_eval_families.get(gen, set())]
        unpaired = [r for r in rows if r["family"] not in clean_eval_families.get(gen, set())]
        cap = eval_caps["transform_per_generator"]
        chosen = by_rank(paired)[:cap]
        if len(chosen) < cap:
            chosen += by_rank(unpaired)[:cap - len(chosen)]
        expected[group] = len(chosen)
        eval_rows.extend(chosen)

    eval_families = {r["family"] for r in eval_rows}
    eval_paths = {r["path"] for r in eval_rows}

    train_rows: list[dict] = []
    budget_left = train_budget_bytes
    for tier in train_tiers:
        queues = {}
        for group, cap in tier.items():
            pool = [r for r in groups.get(group, [])
                    if r["family"] not in eval_families and r["path"] not in eval_paths]
            queues[group] = by_rank(pool)[:cap]
        # Round-robin across the tier so a binding budget is shared evenly.
        pos = {g: 0 for g in queues}
        active = [g for g in sorted(queues) if queues[g]]
        while active:
            nxt = []
            for g in active:
                r = queues[g][pos[g]]
                pos[g] += 1
                if r["size"] <= budget_left:
                    train_rows.append(r)
                    budget_left -= r["size"]
                if pos[g] < len(queues[g]):
                    nxt.append(g)
            active = nxt

    for r in eval_rows:
        r["musebench_role"] = "eval"
    for r in train_rows:
        r["musebench_role"] = "train_pool"
    return {
        "seed": seed,
        "eval_caps": eval_caps,
        "train_tiers": train_tiers,
        "train_budget_bytes": train_budget_bytes,
        "expected_eval_counts": dict(sorted(expected.items())),
        "unclassified": unknown,
        "meta": [{"path": m["path"], "shard": m["shard"], "data_offset": m["data_offset"],
                  "size": m["size"]} for m in meta],
        "rows": eval_rows + train_rows,
        "bytes": {"eval": sum(r["size"] for r in eval_rows),
                  "train_pool": sum(r["size"] for r in train_rows),
                  "meta": sum(m["size"] for m in meta)},
    }


def selection_fingerprint(selection: dict) -> str:
    h = hashlib.sha256()
    for r in sorted(selection["rows"], key=lambda r: r["path"]):
        h.update(f"{r['path']}\t{r['musebench_role']}\n".encode("utf-8", "surrogateescape"))
    return h.hexdigest()


# --------------------------------------------------------------------------- manifest

def track_id(path: str) -> str:
    return "mb_" + hashlib.sha1(path.encode("utf-8", "surrogateescape")).hexdigest()[:16]


def load_ledger(root: Path) -> dict[str, dict]:
    ledger = {}
    path = root / "selection" / "fetch_ledger.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    ledger[entry["path"]] = entry  # last entry wins
    return ledger


def build_manifest(selection: dict, ledger: dict[str, dict], root: Path) -> dict:
    tracks, missing = [], []
    for r in selection["rows"]:
        entry = ledger.get(r["path"])
        if entry is None or entry.get("size") != r["size"]:
            missing.append(r["path"])
            continue
        tracks.append({
            "track_id": track_id(r["path"]),
            "path": str(root / "audio" / r["path"]),
            "label": r["label"],
            "source": source_name(r["group"]),
            "generator": r["generator"],
            "variant": r["variant"],
            "caption_set": r["caption_set"],
            "musebench_role": r["musebench_role"],
            "protocol_split": r["musebench_role"],
            "musebench_path": r["path"],
            "family": r["family"],
            "base_key": r["base_key"],
            "transform_param": r.get("transform_param"),
            "size": r["size"],
            "sha256": entry["sha256"],
            "shard": r["shard"],
            "data_offset": r["data_offset"],
            # The shipped CSVs carry captions only; per-track Jamendo licenses are not included.
            "license_ccurl": None,
            "license_source": "not shipped in MuseBench; see JamendoMaxCaps / FMA metadata",
        })
    return {
        "tracks": tracks,
        "metadata": {
            "dataset": "MuseBench (arXiv:2610.20310)",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": selection["seed"],
            "selection_fingerprint": selection_fingerprint(selection),
            "n_tracks": len(tracks),
            "n_missing": len(missing),
            "roles": dict(Counter(t["musebench_role"] for t in tracks)),
        },
    }, missing


def atomic_write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Select MuseBench members / build the manifest.")
    ap.add_argument("command", choices=["plan", "manifest"])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--budget-gb", type=float, default=170.0,
                    help="cap on all kept audio (eval + train pool), in GB")
    ap.add_argument("--free-floor-gb", type=float, default=80.0,
                    help="free space that must remain on the data root's filesystem")
    args = ap.parse_args(argv)
    sel_path = args.root / "selection" / "selection.json"

    if args.command == "plan":
        records = load_index(args.root)
        probe = plan(records, train_budget_bytes=0)
        eval_bytes = probe["bytes"]["eval"] + probe["bytes"]["meta"]
        have = sum(e["size"] for e in load_ledger(args.root).values())
        free_now = shutil.disk_usage(args.root).free
        cap_total = int(args.budget_gb * GB)
        cap_free = free_now + have - int(args.free_floor_gb * GB)
        train_budget = max(0, min(cap_total, cap_free) - eval_bytes)
        selection = plan(records, train_budget_bytes=train_budget)
        selection["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        selection["budget"] = {"budget_gb": args.budget_gb, "free_floor_gb": args.free_floor_gb,
                               "free_at_plan": free_now}
        atomic_write_json(sel_path, selection)
        counts = Counter((r["musebench_role"], r["group"]) for r in selection["rows"])
        print(f"eval {selection['bytes']['eval'] / GB:.1f} GB, train_pool "
              f"{selection['bytes']['train_pool'] / GB:.1f} GB, unclassified "
              f"{len(selection['unclassified'])}")
        for (role, group), n in sorted(counts.items()):
            print(f"  {role:10s} {group:28s} {n}")
        return 1 if selection["unclassified"] else 0

    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    manifest, missing = build_manifest(selection, load_ledger(args.root), args.root)
    atomic_write_json(args.root / "manifests" / "musebench_runtime_manifest.json", manifest)
    print(f"manifest: {manifest['metadata']['n_tracks']} tracks, {len(missing)} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

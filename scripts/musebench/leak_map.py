#!/usr/bin/env python3
# Created: 2026-11-04
# Purpose: Annotate every MuseBench runtime-manifest row with whether its audio (or its direct
#          audio ancestor) is in the training candidate pool of the detector under test, and
#          write a leak report with per-source seen/unseen counts and the evidence used.
# Dependencies: standard library; ffprobe (optional, duration confirmation of Jamendo matches).
# Test Status: tests/test_musebench_leak_map.py
"""Leak map for MuseBench against a detector's training history.

MuseBench overlaps three corpora that detectors are commonly trained on:

* **Suno / Udio** tracks are SONICS tracks with SONICS file names
  (``fake_<id>_<generator>_<seed>.mp3``) -> matched by name, confirmed by file size.
* **OOD FMA** files are named ``<fma track id>_<title>.mp3`` -> matched by FMA track id
  against the FMA track lists used for training.
* **Jamendo** files are named ``<title> - <artist>.mp3`` (JamendoMaxCaps) -> matched by
  normalised (title, artist) against MTG-Jamendo ``raw.meta.tsv``, restricted to the local
  MTG-Jamendo folders that were part of the training pool, confirmed by duration.

A row is ``seen_in_training`` when its own audio, or its direct audio ancestor, is in that pool:
a transform inherits from its clean track, a co-created remix from its human source track.
Generations conditioned on a seen track's caption are different audio and are not "seen";
``generator_in_training`` records generator-level exposure separately.

The match is against the training *candidate pool* (what the dataset builder could draw from),
which is an upper bound on what the checkpoint actually saw.
"""
from __future__ import annotations

import os as _os
import sys as _sys

# This directory holds select.py, which shadows the stdlib ``select`` module when the
# directory is on sys.path (``python scripts/musebench/x.py`` without -I). Drop it.
if _sys.path and _os.path.abspath(_sys.path[0] or ".") == _os.path.dirname(_os.path.abspath(__file__)):
    _sys.path.pop(0)

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

DEFAULT_ROOT = Path(os.environ.get("MUSEBENCH_ROOT", "~/data/musebench")).expanduser()
MANIFEST_REL = Path("manifests") / "musebench_runtime_manifest.json"
REPORT_REL = Path("leak") / "leak_report.json"
DURATION_TOLERANCE_S = 2.0
LEAK_FIELDS = ("seen_in_training", "seen_evidence", "generator_in_training",
               "in_local_mtg_jamendo", "mtg_jamendo_ids", "leak_note")


# --------------------------------------------------------------------------- identity keys

def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("&amp;", "&").casefold()
    return re.sub(r"\s+", " ", text).strip()


def load_mtg_meta(tsv: Path) -> dict[tuple[str, str], set[int]]:
    """(normalised title, normalised artist) -> MTG-Jamendo track ids."""
    index: dict[tuple[str, str], set[int]] = defaultdict(set)
    with open(tsv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        for row in reader:
            if len(row) <= max(col["TRACK_ID"], col["TRACK_NAME"], col["ARTIST_NAME"]):
                continue
            tid = int(row[col["TRACK_ID"]].split("_")[-1])
            index[(norm(row[col["TRACK_NAME"]]), norm(row[col["ARTIST_NAME"]]))].add(tid)
    return index


def jamendo_candidates(base: str, meta: dict[tuple[str, str], set[int]]) -> set[int]:
    """MTG ids whose (title, artist) matches ``"<title> - <artist>"`` at any ' - ' split."""
    parts = base.split(" - ")
    out: set[int] = set()
    for i in range(1, len(parts)):
        out |= meta.get((norm(" - ".join(parts[:i])), norm(" - ".join(parts[i:]))), set())
    return out


def scan_mtg_audio(audio_dir: Path, folders: set[str] | None) -> dict[int, Path]:
    """MTG track id -> local file, for the given two-digit folders (all if None)."""
    out: dict[int, Path] = {}
    for sub in sorted(audio_dir.iterdir()):
        if not sub.is_dir() or (folders is not None and sub.name not in folders):
            continue
        for f in sub.iterdir():
            stem = f.name.split(".")[0]
            if stem.isdigit():
                out[int(stem)] = f
    return out


FMA_ID = re.compile(r"^(\d+)")


def fma_id(name: str) -> int | None:
    m = FMA_ID.match(Path(name).name)
    return int(m.group(1)) if m else None


def load_fma_ids(path: Path) -> set[int]:
    """FMA track ids from a text list of paths or a JSON list of {filename|path} records."""
    text = path.read_text(encoding="utf-8")
    names: list[str] = []
    if path.suffix == ".json":
        data = json.loads(text)
        for item in data:
            names.append(item if isinstance(item, str) else
                         item.get("filename") or item.get("filepath") or item.get("path", ""))
    else:
        names = [line.strip() for line in text.splitlines() if line.strip()]
    return {i for i in (fma_id(n) for n in names) if i is not None}


def scan_fma_audio(dirs: list[Path]) -> dict[int, Path]:
    """FMA track id -> local file (flat or fma_full/<000>/ layout)."""
    out: dict[int, Path] = {}
    for d in dirs:
        for f in d.rglob("*.mp3"):
            fid = fma_id(f.name)
            if fid is not None:
                out.setdefault(fid, f)
    return out


def scan_sonics(fake_dir: Path) -> dict[str, int]:
    """SONICS fake file stem -> size in bytes."""
    return {f.name.rsplit(".", 1)[0]: f.stat().st_size for f in fake_dir.iterdir() if f.is_file()}


# --------------------------------------------------------------------------- lineage manifests
# A detector's real training history is a chain of training manifests (one per generation), not a
# candidate-pool directory. Their rows name files by path and sometimes by a derived track id
# (codec cells: "cell:<cell>:<source>:<id>"), and derived rows point back with origin_* fields.
LINEAGE_SPLITS = ("train", "val", "test", "hardneg", "codec_cells", "rows", "tracks", "items")
LINEAGE_ROW_KEYS = ("path", "track_id", "origin_path", "origin_track_id", "source_path")
SONICS_STEM = re.compile(r"fake_\d+_(suno|udio)")
UNKNOWN_SIZE = -1  # a SONICS name taken from a manifest whose file is not on this host


def lineage_rows(data: object) -> list:
    """Rows of one training manifest: a list, or a dict of split lists (a dict of lists is flattened)."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    out: list = []
    for split in LINEAGE_SPLITS:
        v = data.get(split)
        if isinstance(v, dict):
            v = [r for sub in v.values() if isinstance(sub, list) for r in sub]
        if isinstance(v, list):
            out.extend(v)
    return out


def lineage_identity(value: object) -> tuple[str, object] | None:
    """('sonics', stem) | ('fma', id) | ('mtg', id) | None for one path or track id.

    The identity is the last path component (after the last ':' for cell ids), up to the first '.'.
    FMA and MTG-Jamendo are told apart by the corpus name appearing anywhere in the string; both
    name their files by a bare numeric track id."""
    if not value:
        return None
    s = str(value)
    stem = s.rsplit("/", 1)[-1].rsplit(":", 1)[-1].split(".")[0]
    low = s.lower()
    if SONICS_STEM.match(stem):
        return "sonics", stem
    if "fma" in low and stem.isdigit():
        return "fma", int(stem)
    if "jamendo" in low and stem.isdigit():
        return "mtg", int(stem)
    return None


def load_lineage(paths: list[Path]) -> tuple[dict[str, int], set[int], set[int], dict[str, int]]:
    """(SONICS stem -> size, FMA ids, MTG-Jamendo ids, rows per manifest) over a lineage of manifests."""
    sonics: dict[str, int] = {}
    fma: set[int] = set()
    mtg: set[int] = set()
    per_manifest: dict[str, int] = {}
    for mp in paths:
        rows = lineage_rows(json.loads(Path(mp).read_text(encoding="utf-8")))
        if not rows:
            raise ValueError(f"{mp}: no rows (unsupported manifest shape)")
        for r in rows:
            values = [r] if isinstance(r, str) else [r.get(k) for k in LINEAGE_ROW_KEYS] if isinstance(r, dict) else []
            for v in values:
                ident = lineage_identity(v)
                if ident is None:
                    continue
                kind, key = ident
                if kind == "sonics":
                    if key not in sonics:
                        p = Path(str(v))
                        sonics[key] = p.stat().st_size if p.is_absolute() and p.is_file() else UNKNOWN_SIZE
                elif kind == "fma":
                    fma.add(key)
                else:
                    mtg.add(key)
        per_manifest[str(mp)] = len(rows)
    return sonics, fma, mtg, per_manifest


def ffprobe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", str(path)],
                             capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------- annotation

def annotate(tracks: list[dict], *, sonics: dict[str, int], fma_ids: set[int],
             mtg_meta: dict[tuple[str, str], set[int]], mtg_pool: dict[int, Path],
             mtg_all: dict[int, Path], generators_in_training: dict[str, bool],
             fma_files: dict[int, Path] | None = None,
             duration: Callable[[Path], float | None] | None = None,
             tolerance_s: float = DURATION_TOLERANCE_S) -> dict:
    """Add ``seen_in_training``/``seen_evidence``/... to each row; return match statistics."""
    stats = Counter()
    for t in tracks:  # idempotent: drop fields from an earlier annotation
        for key in LEAK_FIELDS:
            t.pop(key, None)
    # 1. Ancestor identities: clean Suno/Udio, FMA and Jamendo human rows.
    seen_family: dict[str, str] = {}
    for t in tracks:
        gen, var, base = t["generator"], t["variant"], t["base_key"]
        if gen in ("suno", "udio") and var == "clean":
            size = sonics.get(base)
            if size is not None:
                stats["sonics_name_match"] += 1
                if size == UNKNOWN_SIZE:  # named by a training manifest, file not on this host
                    stats["sonics_size_unknown"] += 1
                    note = "size not checked: training file not on this host"
                else:
                    same = size == t["size"]
                    stats["sonics_size_equal" if same else "sonics_size_differs"] += 1
                    note = "same size" if same else "size differs"
                seen_family[t["family"]] = f"SONICS file {base} ({note})"
        elif t["source"] == "musebench_human_fma_ood":
            fid = fma_id(base)
            if fid is not None and fid in fma_ids:
                stats["fma_id_match"] += 1
                note = f"FMA track {fid} in training FMA list"
                ref = (fma_files or {}).get(fid)
                if ref is not None:
                    same = ref.stat().st_size == t["size"]
                    stats["fma_size_equal" if same else "fma_size_differs"] += 1
                    note += f" ({'same size as' if same else 'size differs from'} local copy)"
                seen_family[t["family"]] = note
        elif t["source"] == "musebench_human_jamendo":
            cands = jamendo_candidates(base, mtg_meta)
            if not cands:
                continue
            stats["jamendo_meta_match"] += 1
            local = [c for c in sorted(cands) if c in mtg_all]
            if local:
                t["mtg_jamendo_ids"] = local
            pooled = [c for c in local if c in mtg_pool]
            if not pooled:
                continue
            note = f"MTG-Jamendo track {pooled[0]} (title/artist match)"
            if duration is not None:
                d_ref = duration(mtg_pool[pooled[0]])
                d_mb = duration(Path(t["path"]))
                if d_ref is not None and d_mb is not None:
                    if abs(d_ref - d_mb) > tolerance_s:
                        stats["jamendo_duration_mismatch"] += 1
                        t["leak_note"] = f"{note}; duration {d_mb:.1f}s vs {d_ref:.1f}s: not counted"
                        continue
                    note += f", duration {d_mb:.1f}s vs {d_ref:.1f}s"
                    stats["jamendo_duration_confirmed"] += 1
            stats["jamendo_pool_match"] += 1
            seen_family[t["family"]] = note

    # 2. Every row: own identity or direct ancestor (same family, and the row derives from it).
    for t in tracks:
        evidence = None
        fam = t["family"]
        if fam in seen_family:
            if t["variant"] == "clean" and t["generator"] in ("human", "suno", "udio"):
                evidence = seen_family[fam]
            elif t["generator"] in ("suno", "udio"):
                evidence = "transform of " + seen_family[fam]  # transforms of a seen SONICS track
            elif t["variant"] == "co_created":
                evidence = "co-created remix of " + seen_family[fam]
        t["seen_in_training"] = evidence is not None
        t["seen_evidence"] = evidence
        g = t["generator"]
        t["generator_in_training"] = None if g == "human" else bool(generators_in_training.get(g, False))
        if t["source"] == "musebench_human_jamendo":
            t["in_local_mtg_jamendo"] = bool(t.get("mtg_jamendo_ids"))
    return dict(stats)


def keyed(field: str, model_key: str | None) -> str:
    """Field name for a model-specific annotation: seen_in_training -> seen_in_training_<key>."""
    return f"{field}_{model_key}" if model_key else field


def annotate_keyed(tracks: list[dict], model_key: str, **kwargs) -> dict:
    """Run annotate() on copies and store its fields under model-specific names, so the rows keep any
    earlier (unkeyed) annotation untouched. Idempotent: stale keyed fields are dropped first."""
    if not model_key or not re.fullmatch(r"[a-z0-9_]+", model_key):
        raise ValueError(f"model key must be [a-z0-9_]+, got {model_key!r}")
    copies = [dict(t) for t in tracks]
    stats = annotate(copies, **kwargs)
    for t, c in zip(tracks, copies):
        for f in LEAK_FIELDS:
            t.pop(keyed(f, model_key), None)
            if f in c:
                t[keyed(f, model_key)] = c[f]
    return stats


def per_source_counts(tracks: list[dict], seen_field: str = "seen_in_training") -> dict[str, dict]:
    out: dict[str, Counter] = defaultdict(Counter)
    for t in tracks:
        c = out[t["source"]]
        c["total"] += 1
        c["seen" if t[seen_field] else "unseen"] += 1
        c[f"{t['musebench_role']}_{'seen' if t[seen_field] else 'unseen'}"] += 1
    return {k: {"total": v["total"], "seen": v["seen"], "unseen": v["unseen"],
                "eval_seen": v["eval_seen"], "eval_unseen": v["eval_unseen"],
                "train_pool_seen": v["train_pool_seen"], "train_pool_unseen": v["train_pool_unseen"]}
            for k, v in sorted(out.items())}


def manifest_fingerprint(tracks: list[dict]) -> str:
    h = hashlib.sha256()
    for t in sorted(tracks, key=lambda t: t["track_id"]):
        h.update(f"{t['track_id']}\t{t['musebench_path']}\n".encode("utf-8", "surrogateescape"))
    return h.hexdigest()


def atomic_write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=True, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def report_rel(model_key: str | None) -> Path:
    """leak/leak_report.json for the unkeyed (first) annotation, leak/leak_report_<key>.json otherwise."""
    return REPORT_REL if not model_key else REPORT_REL.with_name(f"leak_report_{model_key}.json")


def load_references(refs: dict) -> dict:
    """Reference identities from a references JSON.

    Two shapes: a *candidate pool* (``sonics_fake_dir`` + ``fma_id_lists`` + ``mtg_pool_folders``)
    or a *lineage* (``lineage_manifests``: the training manifests of every generation in the model's
    weight lineage, read row by row). Both need ``mtg_meta_tsv`` and ``mtg_audio_dir``."""
    mtg_meta = load_mtg_meta(Path(refs["mtg_meta_tsv"]))
    mtg_all = scan_mtg_audio(Path(refs["mtg_audio_dir"]), None)
    if refs.get("lineage_manifests"):
        sonics, fma_ids, mtg_ids, per_manifest = load_lineage([Path(p) for p in refs["lineage_manifests"]])
        return dict(kind="lineage", sonics=sonics, fma_ids=fma_ids, mtg_meta=mtg_meta, mtg_all=mtg_all,
                    mtg_pool={i: p for i, p in mtg_all.items() if i in mtg_ids}, fma_files={},
                    corpora={"lineage_manifests": per_manifest, "sonics_stems": len(sonics),
                             "fma_ids": len(fma_ids), "mtg_ids": len(mtg_ids),
                             "mtg_ids_local": sum(1 for i in mtg_ids if i in mtg_all),
                             "mtg_meta_tsv": refs["mtg_meta_tsv"], "mtg_audio_dir": refs["mtg_audio_dir"]})
    sonics = scan_sonics(Path(refs["sonics_fake_dir"]))
    fma_ids: set[int] = set()
    for p in refs["fma_id_lists"]:
        fma_ids |= load_fma_ids(Path(p))
    fma_files = scan_fma_audio([Path(p) for p in refs.get("fma_audio_dirs", [])])
    mtg_pool = scan_mtg_audio(Path(refs["mtg_audio_dir"]), set(refs["mtg_pool_folders"]))
    return dict(kind="pool", sonics=sonics, fma_ids=fma_ids, mtg_meta=mtg_meta, mtg_all=mtg_all,
                mtg_pool=mtg_pool, fma_files=fma_files,
                corpora={
                    "sonics_fake_dir": {"path": refs["sonics_fake_dir"], "files": len(sonics)},
                    "fma_id_lists": {"paths": refs["fma_id_lists"], "track_ids": len(fma_ids),
                                     "audio_dirs": refs.get("fma_audio_dirs", []),
                                     "local_files": len(fma_files)},
                    "mtg_jamendo": {"meta_tsv": refs["mtg_meta_tsv"], "audio_dir": refs["mtg_audio_dir"],
                                    "pool_folders": refs["mtg_pool_folders"],
                                    "pool_tracks": len(mtg_pool), "local_tracks": len(mtg_all)}})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Annotate the MuseBench manifest with training overlap.")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--references", type=Path, default=None,
                    help="JSON describing the detector and its training pool or lineage "
                         "(default: <root>/leak/references.json)")
    ap.add_argument("--model-key", default=None,
                    help="write seen_in_training_<key> etc. and leak/leak_report_<key>.json, keeping the "
                         "unkeyed annotation of another model intact")
    ap.add_argument("--no-duration-check", action="store_true")
    args = ap.parse_args(argv)
    ref_path = args.references or args.root / "leak" / "references.json"
    refs = json.loads(ref_path.read_text(encoding="utf-8"))
    ref = load_references(refs)
    key = args.model_key
    seen_field = keyed("seen_in_training", key)

    man_path = args.root / MANIFEST_REL
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    tracks = manifest["tracks"]
    kwargs = dict(sonics=ref["sonics"], fma_ids=ref["fma_ids"], mtg_meta=ref["mtg_meta"],
                  mtg_pool=ref["mtg_pool"], mtg_all=ref["mtg_all"],
                  generators_in_training=refs["generators_in_training"], fma_files=ref["fma_files"],
                  duration=None if args.no_duration_check else ffprobe_duration)
    stats = annotate_keyed(tracks, key, **kwargs) if key else annotate(tracks, **kwargs)
    rep_rel = report_rel(key)
    note = {"model": refs["model"]["name"], "report": str(args.root / rep_rel), "seen_field": seen_field,
            "annotated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if key:
        manifest["metadata"].setdefault("leak_annotations", {})[key] = note
    else:
        manifest["metadata"]["leak_annotation"] = {k: v for k, v in note.items() if k != "seen_field"}
    atomic_write_json(man_path, manifest)

    lineage = ref["kind"] == "lineage"
    report = {
        "model": refs["model"],
        "model_key": key,
        "seen_field": seen_field,
        "training_manifests": refs["training_manifests"],
        "reference_corpora": ref["corpora"],
        "method": {
            "suno_udio": "SONICS file-name match (fake_<id>_<gen>_<seed>), size compared"
                         + (" when the training file is on this host" if lineage else ""),
            "fma": ("FMA track id (numeric file stem of a training row whose path or id names FMA)"
                    if lineage else
                    "FMA track id (leading digits of the file name) in the training FMA id lists; "
                    "size compared with a local FMA copy where one exists"),
            "jamendo": "normalised (title, artist) vs MTG-Jamendo raw.meta.tsv, restricted to the "
                       + ("MTG ids named by training rows" if lineage else "pool folders")
                       + f"; duration within {DURATION_TOLERANCE_S}s"
                       + (" (not checked)" if args.no_duration_check else ""),
            "inheritance": "transforms inherit from their clean track; co-created remixes from "
                           "their human source; caption-conditioned generations do not",
            "scope": ("union of the train/val/test rows of every manifest in the weight lineage: an upper "
                      "bound on exposure for the manifests listed, a lower bound if one is missing"
                      if lineage else
                      "candidate pool of the training dataset builder: an upper bound on exposure"),
        },
        "generators_in_training": refs["generators_in_training"],
        "caveats": refs.get("caveats", []),
        "match_stats": stats,
        "per_source": per_source_counts(tracks, seen_field),
        "totals": dict(Counter("seen" if t[seen_field] else "unseen" for t in tracks)),
        "eval_seen": sum(1 for t in tracks if t[seen_field] and t.get("musebench_role") == "eval"),
        "manifest": str(man_path),
        "manifest_fingerprint": manifest_fingerprint(tracks),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(args.root / rep_rel, report)
    print(json.dumps({"match_stats": stats, "totals": report["totals"], "eval_seen": report["eval_seen"],
                      "report": str(args.root / rep_rel)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

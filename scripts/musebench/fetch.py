#!/usr/bin/env python3
# Created: 2026-11-04
# Purpose: Download the members chosen by select.py from the MuseBench shards with HTTP range
#          requests, byte-identical, SHA-256 recorded, atomic, resumable, with a free-space floor.
# Dependencies: standard library only.
# Test Status: tests/test_musebench_fetch.py
"""Fetch selected MuseBench members.

Reads ``<root>/selection/selection.json`` and writes

* audio to ``<root>/audio/<member path>`` and metadata CSVs to ``<root>/meta/<member path>``,
* one ledger line per finished file to ``<root>/selection/fetch_ledger.jsonl``
  (``path, kind, size, sha256, fetched_at``).

Each member is read with one range request ``[data_offset, data_offset + size)``, written to a
``.part`` file, checked for length, hashed, and renamed into place. A file already in the
ledger with the right size on disk is skipped, so the command can be re-run after any
interruption. Before each download the free space of the data root's filesystem must stay at
or above ``--free-floor-gb`` after the write, otherwise the run stops.

Member names are untrusted: each is validated by ``select.safe_relpath`` before use.
"""
from __future__ import annotations

import os as _os
import sys as _sys

# This directory holds select.py, which shadows the stdlib ``select`` module when the
# directory is on sys.path (``python scripts/musebench/x.py`` without -I). Drop it.
if _sys.path and _os.path.abspath(_sys.path[0] or ".") == _os.path.dirname(_os.path.abspath(__file__)):
    _sys.path.pop(0)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.musebench.index_shards import RangeFetcher, resolve_url  # noqa: E402
from scripts.musebench.select import load_ledger, safe_relpath  # noqa: E402

DEFAULT_ROOT = Path(os.environ.get("MUSEBENCH_ROOT", "~/data/musebench")).expanduser()
GB = 10 ** 9
ROLE_ORDER = {"meta": 0, "eval": 1, "train_pool": 2}


class DiskFloorReached(RuntimeError):
    pass


def work_items(selection: dict) -> list[dict]:
    items = [{**m, "kind": "meta", "role": "meta"} for m in selection["meta"]]
    items += [{"path": r["path"], "shard": r["shard"], "data_offset": r["data_offset"],
               "size": r["size"], "kind": "audio", "role": r["musebench_role"]}
              for r in selection["rows"]]
    items.sort(key=lambda it: (ROLE_ORDER[it["role"]], it["shard"], it["data_offset"]))
    return items


def dest_for(root: Path, item: dict) -> Path:
    rel = safe_relpath(item["path"])
    base = root / ("meta" if item["kind"] == "meta" else "audio")
    dest = (base / rel).resolve()
    if base.resolve() not in dest.parents:
        raise ValueError(f"destination escapes data root: {item['path']!r}")
    return dest


class Fetcher:
    """Thread-safe member downloader; one persistent RangeFetcher per (thread, shard)."""

    def __init__(self, root: Path, shards: dict[str, dict], free_floor_bytes: int,
                 reader_factory: Callable[[dict], Callable[[int, int], bytes]] | None = None,
                 disk_free: Callable[[Path], int] | None = None):
        self.root = root
        self.shards = shards
        self.floor = free_floor_bytes
        self.reader_factory = reader_factory or (
            lambda s: RangeFetcher(resolve_url(s["repo"], s["revision"], s["file"])))
        self.disk_free = disk_free or (lambda p: shutil.disk_usage(p).free)
        self.local = threading.local()
        self.ledger_lock = threading.Lock()
        self.space_lock = threading.Lock()
        self.reserved = 0
        self.ledger_path = root / "selection" / "fetch_ledger.jsonl"
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()

    def _reader(self, shard_file: str):
        # Work items are sorted by shard, so one open connection per thread is enough.
        current = getattr(self.local, "current", None)
        if current is None or current[0] != shard_file:
            if current is not None and hasattr(current[1], "_close"):
                current[1]._close()
            current = self.local.current = (shard_file, self.reader_factory(self.shards[shard_file]))
        return current[1]

    def _reserve(self, size: int) -> None:
        with self.space_lock:
            free = self.disk_free(self.root) - self.reserved
            if free - size < self.floor:
                self.stop.set()
                raise DiskFloorReached(
                    f"free space {free / GB:.1f} GB - {size / GB:.3f} GB < floor {self.floor / GB:.0f} GB")
            self.reserved += size

    def _release(self, size: int) -> None:
        with self.space_lock:
            self.reserved -= size

    def fetch(self, item: dict, attempts: int = 3) -> dict:
        if self.stop.is_set():
            raise DiskFloorReached("stopped")
        dest = dest_for(self.root, item)
        self._reserve(item["size"])
        try:
            for attempt in range(attempts):
                data = self._reader(item["shard"])(item["data_offset"], item["size"]) \
                    if item["size"] else b""
                if len(data) == item["size"]:
                    break
                if attempt == attempts - 1:
                    raise OSError(f"short read for {item['path']!r}: {len(data)} of {item['size']}")
            digest = hashlib.sha256(data).hexdigest()
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            with open(part, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(part, dest)
        finally:
            self._release(item["size"])
        entry = {"path": item["path"], "kind": item["kind"], "size": item["size"],
                 "sha256": digest, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with self.ledger_lock:
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")
        return entry


def pending(root: Path, items: list[dict]) -> list[dict]:
    """Items not yet in the ledger with a correctly sized file on disk."""
    ledger = load_ledger(root)
    out = []
    for it in items:
        entry = ledger.get(it["path"])
        dest = dest_for(root, it)
        if entry and entry["size"] == it["size"] and dest.exists() \
                and dest.stat().st_size == it["size"]:
            continue
        out.append(it)
    return out


def run(root: Path, selection: dict, shards: dict[str, dict], workers: int,
        free_floor_bytes: int, **fetcher_kw) -> tuple[int, int, list[str]]:
    items = pending(root, work_items(selection))
    fetcher = Fetcher(root, shards, free_floor_bytes, **fetcher_kw)
    done, failed = 0, []
    t0, last = time.time(), 0.0
    nbytes = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetcher.fetch, it): it for it in items}
        for fut in as_completed(futures):
            it = futures[fut]
            try:
                fut.result()
                done += 1
                nbytes += it["size"]
            except DiskFloorReached as exc:
                failed.append(f"{it['path']}: {exc}")
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                failed.append(f"{it['path']}: {exc!r}")
            now = time.time()
            if now - last >= 60:
                last = now
                rate = nbytes / max(now - t0, 1e-9) / 1e6
                print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(items)} files, "
                      f"{nbytes / GB:.1f} GB, {rate:.0f} MB/s, {len(failed)} failed",
                      file=sys.stderr, flush=True)
    return len(items), done, failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch selected MuseBench members by range.")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--free-floor-gb", type=float, default=80.0)
    args = ap.parse_args(argv)
    selection = json.loads((args.root / "selection" / "selection.json").read_text(encoding="utf-8"))
    shards = {s["file"]: s for s in
              json.loads((args.root / "index" / "shards.json").read_text(encoding="utf-8"))}
    total, done, failed = run(args.root, selection, shards, args.workers,
                              int(args.free_floor_gb * GB))
    print(f"fetched {done}/{total} pending files; {len(failed)} failed", file=sys.stderr)
    for line in failed[:20]:
        print(f"  {line}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

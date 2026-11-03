#!/usr/bin/env python3
# Created: 2026-11-04
# Purpose: Build a byte-range index of every MuseBench tar shard on Hugging Face by walking
#          tar headers with HTTP range requests (no full download), so any member can later be
#          fetched by {shard, data_offset, size}.
# Dependencies: standard library only.
# Test Status: tests/test_musebench_index.py
"""Index the MuseBench tar shards without downloading them.

MuseBench (arXiv:2610.20310) is published as five Hugging Face dataset repositories
(``Anonymousv22222/MuseBench-part1`` .. ``part5``), each holding 15 standalone GNU tar
shards (~800 GB in total). Every shard starts with a tar header at offset 0, so a shard can
be walked header by header: read 512 bytes, parse name and size, jump to the next header.

Output, under ``<root>/index/``:

* ``shards.json`` -- the pinned shard list: repo, revision (commit sha), file, size, LFS sha256.
* ``<shard>.jsonl`` -- one record per member: ``offset`` (first header byte, including any
  GNU long-name / pax extension headers), ``data_offset``, ``size``, ``path``, ``type``.
* ``<shard>.done`` -- written atomically when the end-of-archive marker is reached.

A shard without ``.done`` is resumed from the end of its last complete record.

Shard bytes are untrusted input. This script only parses header blocks and never writes
member content; run it with ``python -I``.
"""
from __future__ import annotations

import os as _os
import sys as _sys

# This directory holds select.py, which shadows the stdlib ``select`` module when the
# directory is on sys.path (``python scripts/musebench/x.py`` without -I). Drop it.
if _sys.path and _os.path.abspath(_sys.path[0] or ".") == _os.path.dirname(_os.path.abspath(__file__)):
    _sys.path.pop(0)


import argparse
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterator

BLOCK = 512
HF = "https://huggingface.co"
REPO_FMT = "Anonymousv22222/MuseBench-part{part}"
PARTS = (1, 2, 3, 4, 5)
DEFAULT_ROOT = Path(os.environ.get("MUSEBENCH_ROOT", "~/data/musebench")).expanduser()
USER_AGENT = "artifactbench-musebench-indexer/1.0"

Fetch = Callable[[int, int], bytes]


class TarFormatError(ValueError):
    """Raised when a header block is not a valid tar header."""


# --------------------------------------------------------------------------- tar parsing

def parse_number(field: bytes) -> int:
    """Parse a tar numeric field: NUL/space-terminated octal, or GNU base-256."""
    if field and field[0] & 0x80:
        # GNU base-256: the high bit of the first byte flags the encoding.
        raw = bytes([field[0] & 0x7F]) + field[1:]
        return int.from_bytes(raw, "big")
    text = field.split(b"\0", 1)[0].strip(b" ")
    if not text:
        return 0
    try:
        return int(text, 8)
    except ValueError as exc:
        raise TarFormatError(f"bad numeric field {field!r}") from exc


def header_checksum_ok(header: bytes) -> bool:
    stored = parse_number(header[148:156])
    computed = sum(header[:148]) + 8 * 0x20 + sum(header[156:BLOCK])
    return stored == computed


def padded(size: int) -> int:
    return (size + BLOCK - 1) // BLOCK * BLOCK


def parse_pax(data: bytes) -> dict[str, bytes]:
    """Parse pax extended-header records ``"<len> <key>=<value>\\n"``."""
    out: dict[str, bytes] = {}
    pos = 0
    while pos < len(data):
        if data[pos:pos + 1] == b"\0":
            break
        space = data.find(b" ", pos)
        digits = data[pos:space] if space > pos else b""
        if not digits.isdigit():
            raise TarFormatError("bad pax record length")
        length = int(digits)
        if length <= space - pos or pos + length > len(data):
            raise TarFormatError("bad pax record length")
        record = data[space + 1:pos + length]
        if not record.endswith(b"\n") or b"=" not in record:
            raise TarFormatError("bad pax record")
        key, value = record[:-1].split(b"=", 1)
        out[key.decode("utf-8", "replace")] = value
        pos += length
    return out


def decode_name(raw: bytes) -> str:
    # surrogateescape keeps non-UTF-8 names round-trippable (json escapes the surrogates).
    return raw.decode("utf-8", "surrogateescape")


def walk_tar(fetch: Fetch, start: int = 0, prefetch: int = 4 * BLOCK) -> Iterator[dict]:
    """Yield member records of a tar archive reachable through ``fetch(offset, length)``.

    ``fetch`` may return fewer bytes than asked only at end of file. The generator's
    return value (``StopIteration.value``) is the end-of-archive offset: the first byte of
    the zero block that terminates the archive.
    """
    off = start
    ext_start: int | None = None
    long_name: bytes | None = None
    pax_path: bytes | None = None
    while True:
        buf = fetch(off, prefetch)
        header = buf[:BLOCK]
        if len(header) < BLOCK:
            raise TarFormatError(f"truncated archive at offset {off}")
        if header == b"\0" * BLOCK:
            if ext_start is not None:
                raise TarFormatError(f"extension header without member before {off}")
            return off
        if not header_checksum_ok(header):
            raise TarFormatError(f"header checksum mismatch at offset {off}")
        typ = header[156:157]
        size = parse_number(header[124:136])
        data_offset = off + BLOCK
        if ext_start is None:
            ext_start = off

        if typ in (b"L", b"K", b"x", b"g"):
            have = buf[BLOCK:BLOCK + size]
            data = have if len(have) == size else fetch(data_offset, size)
            if len(data) < size:
                raise TarFormatError(f"truncated extension header at offset {off}")
            if typ == b"L":
                long_name = data.split(b"\0", 1)[0]
            elif typ == b"x":
                pax_path = parse_pax(data).get("path", pax_path)
            elif typ == b"g":
                pass  # global pax header: no per-member path
            off = data_offset + padded(size)
            if typ == b"g":
                ext_start = None
            continue

        name = header[0:100].split(b"\0", 1)[0]
        magic = header[257:263]
        if magic == b"ustar\0":
            # POSIX ustar: bytes 345..500 are the name prefix. (In GNU "ustar  " headers
            # the same bytes hold atime/ctime, so they must not be read as a prefix.)
            prefix = header[345:500].split(b"\0", 1)[0]
            if prefix:
                name = prefix + b"/" + name
        if long_name is not None:
            name = long_name
        if pax_path is not None:
            name = pax_path

        kind = {b"0": "file", b"\0": "file", b"7": "file", b"5": "dir",
                b"1": "hardlink", b"2": "symlink"}.get(typ, "other:" + typ.decode("latin-1"))
        # Hard links, symlinks and directories carry no data even if a size is recorded.
        if kind in ("dir", "hardlink", "symlink"):
            size = 0
        yield {
            "offset": ext_start,
            "data_offset": data_offset,
            "size": size,
            "path": decode_name(name),
            "type": kind,
            "mtime": parse_number(header[136:148]),
        }
        off = data_offset + padded(size)
        ext_start, long_name, pax_path = None, None, None


def record_end(rec: dict) -> int:
    return rec["data_offset"] + padded(rec["size"])


# --------------------------------------------------------------------------- HTTP fetch

class RangeFetcher:
    """Persistent-connection HTTP range reader for one Hugging Face file.

    The ``resolve`` URL redirects to a signed CDN URL; it is resolved once and re-resolved
    when the CDN answers 401/403/404/410 (signature expired) or the connection breaks.
    """

    def __init__(self, url: str, timeout: float = 60.0, max_retries: int = 12):
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.signed: urllib.parse.SplitResult | None = None
        self.conn: http.client.HTTPSConnection | None = None
        self.requests = 0

    def _resolve(self) -> None:
        target = self.url
        for _ in range(8):
            parts = urllib.parse.urlsplit(target)
            conn = http.client.HTTPSConnection(parts.netloc, timeout=self.timeout)
            path = parts.path + ("?" + parts.query if parts.query else "")
            conn.request("GET", path, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT})
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status in (301, 302, 303, 307, 308):
                target = urllib.parse.urljoin(target, resp.getheader("Location"))
                continue
            if resp.status in (200, 206):
                self.signed = urllib.parse.urlsplit(target)
                self._close()
                return
            raise OSError(f"resolve {self.url}: HTTP {resp.status}")
        raise OSError(f"resolve {self.url}: too many redirects")

    def _close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
        self.conn = None

    def __call__(self, offset: int, length: int) -> bytes:
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                if self.signed is None:
                    self._resolve()
                if self.conn is None:
                    self.conn = http.client.HTTPSConnection(self.signed.netloc, timeout=self.timeout)
                path = self.signed.path + ("?" + self.signed.query if self.signed.query else "")
                self.conn.request("GET", path, headers={
                    "Range": f"bytes={offset}-{offset + length - 1}", "User-Agent": USER_AGENT})
                resp = self.conn.getresponse()
                body = resp.read()
                self.requests += 1
                if resp.status == 206:
                    return body
                if resp.status == 416:
                    return b""
                if resp.status in (401, 403, 404, 410):
                    self.signed = None
                    self._close()
                elif resp.status == 429 or resp.status >= 500:
                    retry_after = resp.getheader("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = max(delay, float(retry_after))
                    self._close()
                else:
                    raise OSError(f"range {offset}+{length}: HTTP {resp.status}")
            except (OSError, http.client.HTTPException):
                self._close()
                if attempt == self.max_retries - 1:
                    raise
            time.sleep(min(delay, 120.0))
            delay *= 2
        raise OSError(f"range {offset}+{length}: retries exhausted")


class StreamFetcher:
    """Forward-only reader: one streaming GET from the resume offset, data skipped locally.

    For shards with many small members, one range request per header is latency-bound
    (~0.7 s each); reading the shard sequentially and discarding member data is faster.
    ``walk_tar`` only moves forward, so a small look-back buffer is enough. A broken stream
    is reopened with ``Range: bytes=<offset>-``.
    """

    CHUNK = 1 << 20

    def __init__(self, url: str, opener: Callable[[str, int], object] | None = None,
                 max_retries: int = 12):
        self.url = url
        self.opener = opener or self._open_http
        self.max_retries = max_retries
        self.stream = None
        self.buf = b""
        self.buf_start = 0  # file offset of buf[0]
        self.bytes_read = 0

    @staticmethod
    def _open_http(url: str, offset: int):
        req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-",
                                                   "User-Agent": USER_AGENT})
        resp = urllib.request.urlopen(req, timeout=120)
        if resp.status != 206 and not (resp.status == 200 and offset == 0):
            raise OSError(f"stream {url}: HTTP {resp.status}")
        return resp

    def _reopen(self, offset: int) -> None:
        if self.stream is not None:
            try:
                self.stream.close()
            except OSError:
                pass
        self.stream = self.opener(self.url, offset)
        self.buf, self.buf_start = b"", offset

    def _read_more(self) -> bool:
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                if self.stream is None:
                    self._reopen(self.buf_start + len(self.buf))
                chunk = self.stream.read(self.CHUNK)
                if chunk:
                    self.buf += chunk
                    self.bytes_read += len(chunk)
                    return True
                return False  # EOF
            except (OSError, http.client.HTTPException):
                if attempt == self.max_retries - 1:
                    raise
                keep_start, keep = self.buf_start, self.buf
                self.stream = None
                time.sleep(min(delay, 120.0))
                delay *= 2
                try:
                    self._reopen(keep_start + len(keep))
                except (OSError, http.client.HTTPException):
                    self.stream = None
                self.buf, self.buf_start = keep, keep_start
        raise OSError(f"stream {self.url}: retries exhausted")

    def __call__(self, offset: int, length: int) -> bytes:
        if self.stream is None and not self.buf:
            self.buf_start = offset
        if offset < self.buf_start:
            raise ValueError(f"stream reader cannot seek back to {offset} (< {self.buf_start})")
        end_have = self.buf_start + len(self.buf)
        if offset > end_have + 64 * self.CHUNK:
            # Far jump (resume or huge member): cheaper to reopen than to read through.
            self._reopen(offset)
        # Drop bytes before ``offset`` and read until the request is covered or EOF.
        while True:
            if offset > self.buf_start:
                drop = min(offset - self.buf_start, len(self.buf))
                self.buf, self.buf_start = self.buf[drop:], self.buf_start + drop
            if offset == self.buf_start and len(self.buf) >= length:
                break
            if not self._read_more():
                break
        rel = offset - self.buf_start
        return self.buf[rel:rel + length] if rel >= 0 else b""


# --------------------------------------------------------------------------- shard index

def resolve_url(repo: str, revision: str, filename: str) -> str:
    return f"{HF}/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(filename)}"


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def list_shards(parts=PARTS) -> list[dict]:
    """Return the pinned shard list from the Hugging Face API."""
    shards = []
    for part in parts:
        repo = REPO_FMT.format(part=part)
        revision = _get_json(f"{HF}/api/datasets/{repo}")["sha"]
        tree = _get_json(f"{HF}/api/datasets/{repo}/tree/{revision}")
        for entry in tree:
            if entry.get("type") == "file" and entry["path"].endswith(".tar"):
                shards.append({
                    "part": part, "repo": repo, "revision": revision, "file": entry["path"],
                    "size": entry["size"], "lfs_sha256": (entry.get("lfs") or {}).get("oid"),
                })
    return sorted(shards, key=lambda s: (s["part"], s["file"]))


def shard_stem(shard: dict) -> str:
    return shard["file"][:-len(".tar")]


def atomic_write_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def resume_offset(jsonl: Path) -> tuple[int, int]:
    """Return (next header offset, complete record count); drop a torn trailing line."""
    if not jsonl.exists():
        return 0, 0
    raw = jsonl.read_bytes()
    keep = raw.rfind(b"\n") + 1
    if keep != len(raw):
        with open(jsonl, "r+b") as f:
            f.truncate(keep)
        raw = raw[:keep]
    lines = raw.splitlines()
    if not lines:
        return 0, 0
    return record_end(json.loads(lines[-1])), len(lines)


def index_shard(shard: dict, out_dir: Path, fetch: Fetch,
                progress: Callable[[str, int], None] | None = None) -> dict:
    """Walk one shard into ``<stem>.jsonl`` and mark it ``<stem>.done``. Idempotent."""
    stem = shard_stem(shard)
    jsonl, done = out_dir / f"{stem}.jsonl", out_dir / f"{stem}.done"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    start, count = resume_offset(jsonl)
    gen = walk_tar(fetch, start)
    with open(jsonl, "a", encoding="utf-8") as f:
        while True:
            try:
                rec = next(gen)
            except StopIteration as stop:
                end_offset = stop.value
                break
            rec["shard"] = shard["file"]
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            count += 1
            if progress is not None:
                progress(stem, count)
    if end_offset > shard["size"]:
        raise TarFormatError(f"{stem}: end offset {end_offset} beyond file size {shard['size']}")
    info = {
        **shard,
        "members": count,
        "end_offset": end_offset,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(done, info)
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="data root (default: $MUSEBENCH_ROOT or ~/data/musebench)")
    ap.add_argument("--parts", type=int, nargs="+", default=list(PARTS))
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these shard stems")
    ap.add_argument("--workers", type=int, default=75, help="shards walked concurrently")
    ap.add_argument("--mode", choices=["range", "stream"], default="range",
                    help="range: one range request per header (cheap on bandwidth); "
                         "stream: one sequential GET per shard, member data discarded "
                         "(faster for shards with many small members)")
    ap.add_argument("--refresh-list", action="store_true",
                    help="re-query the shard list even if index/shards.json exists")
    args = ap.parse_args(argv)

    out_dir = args.root / "index"
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / "shards.json"
    if list_path.exists() and not args.refresh_list:
        shards = json.loads(list_path.read_text(encoding="utf-8"))
    else:
        shards = list_shards(PARTS)
        atomic_write_json(list_path, shards)
    todo = [s for s in shards if s["part"] in args.parts
            and (args.only is None or shard_stem(s) in args.only)]

    counts: dict[str, int] = {}
    lock = threading.Lock()
    last_report = [0.0]

    def progress(stem: str, n: int) -> None:
        with lock:
            counts[stem] = n
            now = time.time()
            if now - last_report[0] >= 60:
                last_report[0] = now
                print(f"[{time.strftime('%H:%M:%S')}] members indexed: {sum(counts.values())} "
                      f"across {len(counts)} shards", file=sys.stderr, flush=True)

    def run(shard: dict) -> dict:
        url = resolve_url(shard["repo"], shard["revision"], shard["file"])
        fetcher = StreamFetcher(url) if args.mode == "stream" else RangeFetcher(url)
        return index_shard(shard, out_dir, fetcher, progress)

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, s): s for s in todo}
        for fut in as_completed(futures):
            stem = shard_stem(futures[fut])
            try:
                info = fut.result()
                print(f"done {stem}: {info['members']} members", file=sys.stderr, flush=True)
            except Exception as exc:  # noqa: BLE001 - report and continue with other shards
                failures += 1
                print(f"FAILED {stem}: {exc!r}", file=sys.stderr, flush=True)
    print(f"{len(todo) - failures}/{len(todo)} shards indexed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

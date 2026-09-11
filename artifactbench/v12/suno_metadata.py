"""공개 Suno 곡의 정확한 ID에 연결된 버전 근거를 캐시한다. 인증 API는 사용하지 않는다."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid

from .common import digest, write_once


class ScriptParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.current = []

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.current is not None:
            self.scripts.append("".join(self.current))
            self.current = None


def walk_objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_objects(child)


def page_objects(page):
    """JS 실행 없이 JSON과 Next flight의 JSON 레코드만 해석한다."""
    parser = ScriptParser()
    parser.feed(page)
    decoder = json.JSONDecoder()
    flight = []
    for script in parser.scripts:
        if script.lstrip().startswith(("{", "[")):
            try:
                yield from walk_objects(json.loads(script))
            except json.JSONDecodeError:
                # JS 배열과 순수 JSON script를 구분한다.
                continue
        for match in re.finditer(r"self\.__next_f\.push\(", script):
            try:
                record, _ = decoder.raw_decode(script[match.end():].lstrip())
            except json.JSONDecodeError as exc:
                raise ValueError("Malformed Next flight envelope") from exc
            if isinstance(record, list) and len(record) > 1 and record[0] == 1 and isinstance(record[1], str):
                flight.append(record[1])
    # 여러 script에 나뉜 레코드도 순서대로 결합한 후 해석한다.
    stream = "".join(flight).encode("utf-8")
    offset = 0
    while offset < len(stream):
        match = re.match(rb"[0-9a-f]*:", stream[offset:])
        if match is None:
            raise ValueError(f"Malformed Next flight row at byte {offset}")
        offset += match.end()
        # ReactFlightClient의 ROW_CHUNK_BY_LENGTH: T 본문은 줄바꿈이 아닌
        # UTF-8 바이트 길이로 끝난다. 가사 속 JSON 모양 텍스트는 해석하지 않는다.
        if stream[offset:offset + 1] in [bytes([c]) for c in b"TAOobUSsLlGgMmV"]:
            length = re.match(rb"[A-Za-z]([0-9a-f]+),", stream[offset:])
            if length is None:
                raise ValueError("Malformed length-delimited flight row")
            offset += length.end() + int(length[1], 16)
            if offset > len(stream):
                raise ValueError("Truncated length-delimited flight row")
            continue
        end = stream.find(b"\n", offset)
        if end < 0:
            raise ValueError("Truncated newline-delimited flight row")
        body = stream[offset:end]
        offset = end + 1
        if body.startswith((b"[", b"{")):
            try:
                record = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ValueError("Malformed Next flight JSON record") from exc
            yield from walk_objects(record)


def extract_suno(page, media_id):
    media_id = str(uuid.UUID(media_id))
    clips = [obj for obj in page_objects(page)
             if obj.get("id") == media_id and
             any(key in obj for key in ("major_model_version", "model_version", "model_name"))]
    if not clips:
        raise ValueError(f"No model-bearing clip object for requested ID {media_id}")
    results = []
    for clip in clips:
        metadata = clip.get("metadata") or {}
        version = clip.get("major_model_version") or clip.get("model_version")
        badge = metadata.get("model_badges", {}).get("songcard", {}).get("display_name")
        if badge and version and badge != version:
            raise ValueError(f"Conflicting version evidence: {version!r} versus {badge!r}")
        results.append({
            "media_id": media_id,
            "generator_version": version,
            "version_field": "major_model_version" if clip.get("major_model_version") else "model_version",
            "model_name": clip.get("model_name"), "version_badge": badge,
            "created_at": clip.get("created_at"), "creator_handle": clip.get("handle"),
            "creator_id": clip.get("user_id"), "status": clip.get("status"),
            "is_public": clip.get("is_public"), "video_url": clip.get("video_url"),
            "generation_task": metadata.get("task"), "generation_type": metadata.get("type"),
            "is_remix": metadata.get("is_remix"),
            "duration_seconds": metadata.get("duration"),
            "lineage": {k: metadata[k] for k in ("edited_clip_id", "upsample_clip_id", "history") if k in metadata},
            "label_scope": "platform-generated-or-edited; fully-synthetic status not established",
        })
    if any(item != results[0] for item in results[1:]):
        raise ValueError(f"Conflicting clip objects for {media_id}")
    return results[0]


def collect_one(candidate, output, delay=0.2, reuse_pages_from=None):
    media_id = str(uuid.UUID(candidate["media_id"]))
    result_path = output / "metadata" / f"{media_id}.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if cached["track_id"] != candidate["track_id"]:
            raise ValueError(f"Cache identity mismatch: {media_id}")
        if cached.get("page_sha256") and digest(output / cached["page_file"]) != cached["page_sha256"]:
            raise ValueError(f"Cache digest mismatch: {media_id}")
        return cached
    url = "https://suno.com/song/" + media_id
    row = {"track_id": candidate["track_id"], "source_url": url,
           "retrieved_at": datetime.now(timezone.utc).isoformat(),
           "extractor_sha256": digest(__file__)}
    try:
        parent_path = reuse_pages_from / "metadata" / f"{media_id}.json" if reuse_pages_from else None
        parent = json.loads(parent_path.read_text()) if parent_path and parent_path.exists() else {}
        if parent.get("page_sha256"):
            if parent["track_id"] != candidate["track_id"] or digest(reuse_pages_from / parent["page_file"]) != parent["page_sha256"]:
                raise ValueError(f"Parent cache mismatch: {media_id}")
            payload = (reuse_pages_from / parent["page_file"]).read_bytes()
            row.update({key: parent[key] for key in ("http_status", "final_url", "content_type", "retrieved_at")})
            row["parent_metadata_sha256"] = digest(parent_path)
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ArtifactBench/1.2 research)"})
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = response.read(4 * 1024 * 1024 + 1)
                if len(payload) > 4 * 1024 * 1024:
                    raise ValueError("Page exceeds 4 MiB bound")
                row.update(http_status=response.status, final_url=response.url,
                           content_type=response.headers.get("Content-Type"))
        page_sha = hashlib.sha256(payload).hexdigest()
        page_path = output / "pages" / f"{media_id}-{page_sha}.html"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        if not page_path.exists():
            with page_path.open("xb") as stream:
                stream.write(payload)
        row.update(page_sha256=page_sha, page_file=str(page_path.relative_to(output)))
        row.update(extract_suno(payload.decode("utf-8"), media_id))
        row["outcome"] = "verified" if row["generator_version"] else "unknown_version"
    except (urllib.error.URLError, TimeoutError, ValueError, UnicodeError) as exc:
        row.update(outcome="error", error_type=type(exc).__name__, error=str(exc))
    finally:
        time.sleep(delay)
    write_once(result_path, row)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--reuse-pages-from", type=Path)
    args = ap.parse_args()
    if not 1 <= args.workers <= 2:
        ap.error("Use one or two workers for public page collection")
    candidates = [r for r in json.loads(args.candidates.read_text()) if r["provider"] == "suno"]
    write_once(args.output / "collection_inputs.json", {
        "candidates_sha256": digest(args.candidates), "candidate_tracks": len(candidates),
        "extractor_sha256": digest(__file__), "workers": args.workers,
        "reuse_pages_from": str(args.reuse_pages_from) if args.reuse_pages_from else None,
    })
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(collect_one, row, args.output, reuse_pages_from=args.reuse_pages_from) for row in candidates]
        for future in as_completed(pending):
            rows.append(future.result())
            if len(rows) % 25 == 0 or len(rows) == len(candidates):
                print(json.dumps({"done": len(rows), "total": len(candidates),
                                  "outcomes": dict(Counter(r["outcome"] for r in rows)),
                                  "versions": dict(Counter(r.get("generator_version") or "unknown" for r in rows))}), flush=True)
    write_once(args.output / "summary.json", {
        "attempted": len(rows), "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "versions": dict(Counter(r.get("generator_version") or "unknown" for r in rows)),
        "tasks": dict(Counter(r.get("generation_task") or "unknown" for r in rows)),
    })


if __name__ == "__main__":
    main()

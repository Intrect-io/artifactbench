"""Udio의 공개 곡 페이지를 CDN 미디어 ID에 결합하여 제작자와 생성일을 확인한다."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
from urllib.parse import urlparse
import urllib.request

from .common import digest, write_once
from .suno_metadata import page_objects


def has_media_id(url, media_id):
    return isinstance(url, str) and media_id in urlparse(url).path.split("/")


def extract_udio(page, media_id):
    if not re.fullmatch(r"[0-9a-f]{32}", media_id):
        raise ValueError(f"Invalid Udio media ID: {media_id}")
    songs = [obj for obj in page_objects(page)
             if any(has_media_id(obj.get(key), media_id) for key in ("song_path", "video_path", "original_song_path"))
             and "generation_id" in obj]
    if not songs:
        raise ValueError(f"No song object linked to requested media ID {media_id}")
    results = [{
        "media_id": media_id, "song_id": song.get("id"),
        "creator_id": song.get("user_id"), "artist": song.get("artist"),
        "created_at": song.get("created_at"), "published_at": song.get("published_at"),
        "generator_version": song.get("model_version") or song.get("modelVersion"),
        "version_evidence": "page_model_field" if song.get("model_version") or song.get("modelVersion") else "not_disclosed_in_song_object",
        "generation_id": song.get("generation_id"), "duration_seconds": song.get("duration"),
        "finished": song.get("finished"), "audio_conditioning_type": song.get("audio_conditioning_type"),
        "lineage": {key: song.get(key) for key in ("parent_id", "style_source_song_id", "style_source_type", "original_song_path")},
        "song_path": song.get("song_path"), "video_path": song.get("video_path"),
        "label_scope": "platform-generated-or-edited; fully-synthetic status not established",
    } for song in songs]
    if any(row != results[0] for row in results[1:]):
        raise ValueError(f"Conflicting song objects for media ID {media_id}")
    return results[0]


def collect_one(candidate, output):
    mid = candidate["media_id"]
    if not re.fullmatch(r"[0-9a-f]{32}", mid):
        raise ValueError(f"Invalid Udio media ID: {mid}")
    result_path = output / "metadata" / f"{mid}.json"
    if result_path.exists():
        row = json.loads(result_path.read_text())
        if row["track_id"] != candidate["track_id"]:
            raise ValueError("Cached track ID changed")
        if row.get("page_sha256") and digest(output / row["page_file"]) != row["page_sha256"]:
            raise ValueError("Cached page changed")
        return row
    urls = sorted({r["source_page"] for r in candidate["variants"]
                   if urlparse(r["source_page"]).hostname in ("udio.com", "www.udio.com")
                   and urlparse(r["source_page"]).path.startswith("/songs/")})
    row = {"track_id": candidate["track_id"], "retrieved_at": datetime.now(timezone.utc).isoformat()}
    try:
        if not urls:
            raise ValueError("No public song-page URL in pinned candidate")
        row["source_url"] = urls[0]
        request = urllib.request.Request(urls[0], headers={"User-Agent": "Mozilla/5.0 (compatible; ArtifactBench/1.2 research)"})
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = response.read(4 * 1024 * 1024 + 1)
            if len(payload) > 4 * 1024 * 1024:
                raise ValueError("Page exceeds 4 MiB bound")
            row.update(http_status=response.status, final_url=response.url,
                       content_type=response.headers.get("Content-Type"))
        sha = hashlib.sha256(payload).hexdigest()
        page_file = f"pages/{mid}-{sha}.html"
        path = output / page_file
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(payload)
        row.update(page_sha256=sha, page_file=page_file)
        row.update(extract_udio(payload.decode("utf-8"), mid))
        row["outcome"] = "verified" if row["generator_version"] else "unknown_version"
    except (urllib.error.URLError, TimeoutError, ValueError, UnicodeError) as exc:
        row.update(outcome="error", error_type=type(exc).__name__, error=str(exc))
    finally:
        time.sleep(0.2)
    write_once(result_path, row)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    candidates = [r for r in json.loads(args.candidates.read_text()) if r["provider"] == "udio"]
    write_once(args.output / "collection_inputs.json", {
        "candidates_sha256": digest(args.candidates), "extractor_sha256": digest(__file__),
        "flight_parser_sha256": digest(Path(__file__).with_name("suno_metadata.py")),
        "candidate_tracks": len(candidates), "workers": 2,
    })
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(collect_one, row, args.output) for row in candidates]
        for future in as_completed(pending):
            rows.append(future.result())
            if len(rows) % 25 == 0 or len(rows) == len(candidates):
                print(json.dumps({"done": len(rows), "total": len(candidates),
                                  "outcomes": dict(Counter(r["outcome"] for r in rows))}), flush=True)
    write_once(args.output / "summary.json", {"attempted": len(rows),
               "outcomes": dict(Counter(r["outcome"] for r in rows)),
               "versions": dict(Counter(r.get("generator_version") or "unknown" for r in rows))})


if __name__ == "__main__":
    main()

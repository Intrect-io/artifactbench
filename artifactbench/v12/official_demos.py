"""버전이 명시된 공식 데모만 별도 challenge용으로 보존한다. 대표표본으로 취급하지 않는다."""
import argparse
from datetime import datetime, timezone
import hashlib
from html import unescape
import json
from pathlib import Path
import re
from urllib.parse import urldefrag, urljoin, urlparse
import urllib.request

from .common import digest, write_once


PAGES = {"lyria": "https://deepmind.google/models/lyria/",
         "stable_audio": "https://stability.ai/stable-audio"}


def demo_urls(provider, page):
    if provider == "lyria":
        start = re.search(r'\bid=["\']?lyria-35["\']?[\s>]', page)
        if start is None:
            raise ValueError("Missing explicitly labelled Lyria 3.5 section")
        end = page.find("<hr", start.end())
        if end < 0:
            raise ValueError("Cannot delimit Lyria 3.5 section")
        section = page[start.start():end]
        if "Introducing Lyria 3.5" not in section:
            raise ValueError("Section version label changed")
        urls = []
        for article in re.findall(r"<article\b.*?</article>", section, re.S):
            candidates = {urldefrag(unescape(u))[0] for u in re.findall(
                r'(?:data-src|src)=["\']([^"\']+\.(?:webm|mp4)[^"\']*)', article)
                if "lyria-3-5__track__" in u}
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one recording per carousel article, got {len(candidates)}")
            urls.extend(candidates)
    elif provider == "stable_audio":
        headings = [unescape(re.sub(r"<[^>]+>", "", h)).strip()
                    for h in re.findall(r"<h1\b.*?</h1>", page, re.S)]
        if "Stable Audio 3.0" not in headings:
            raise ValueError("Stable Audio page version heading changed")
        urls = [urljoin(PAGES[provider], unescape(u)) for u in re.findall(r'data-media-src=["\']([^"\']+\.mp3)["\']', page)]
    else:
        raise ValueError(f"Unknown provider: {provider}")
    if not urls or len(urls) != len(set(urls)):
        raise ValueError("Empty or duplicate official demo list")
    return urls


def download(url, path, max_bytes):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ArtifactBench/1.2 research)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError(f"Download exceeds {max_bytes} bytes: {url}")
        if not payload:
            raise ValueError(f"Empty download: {url}")
        info = {"source_url": url, "final_url": response.url, "content_type": response.headers.get("Content-Type"),
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "http_status": response.status,
                "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return info


def collect(output):
    all_rows = []
    for provider, url in PAGES.items():
        page_path = output / "pages" / f"{provider}.html"
        evidence_path = page_path.with_suffix(".json")
        if evidence_path.exists():
            evidence = json.loads(evidence_path.read_text())
            if digest(page_path) != evidence["sha256"]:
                raise ValueError(f"Page changed: {page_path}")
        else:
            evidence = download(url, page_path, 4 * 1024 * 1024)
            write_once(evidence_path, evidence)
        urls = demo_urls(provider, page_path.read_text())
        selection = {
            "page_sha256": evidence["sha256"], "urls": urls,
            "sampling": "all distinct recordings in explicitly version-labelled official demo section",
            "tool_sha256": digest(__file__),
        }
        selection_path = output / f"{provider}_selection.json"
        if selection_path.exists():
            previous = json.loads(selection_path.read_text())
            if any(previous[k] != selection[k] for k in ("page_sha256", "urls", "sampling")):
                raise ValueError(f"Official demo selection changed: {provider}")
        else:
            write_once(selection_path, selection)
        for media_url in urls:
            name = Path(urlparse(media_url).path).name
            media_path = output / "audio" / provider / name
            record_path = output / "records" / f"{provider}-{name}.json"
            if record_path.exists():
                row = json.loads(record_path.read_text())
                if digest(media_path) != row["sha256"]:
                    raise ValueError(f"Media changed: {media_path}")
            else:
                row = download(media_url, media_path, 256 * 1024 * 1024)
                if row["content_type"] and "text/html" in row["content_type"]:
                    raise ValueError(f"HTML response at media URL: {media_url}")
                row.update(
                    track_id=provider + ":official:" + name.rsplit(".", 1)[0],
                    path=str(media_path.resolve()), provider=provider, label="ai",
                    generator_version="3.5" if provider == "lyria" else "3.0",
                    model_size="unknown", source=f"{provider}_official_demo_260905",
                    partition="official_demo", source_page=url, page_sha256=evidence["sha256"],
                    version_evidence="official_version_labelled_demo_section",
                    creator_group=f"{provider}:official_demo_selection", transport="native_official_demo",
                    rights="metadata_only; audio redistribution permission not established",
                    label_scope="official generated-music demonstration; production workflow not independently audited",
                )
                write_once(record_path, row)
            all_rows.append(row)
            print(json.dumps({"provider": provider, "file": name, "bytes": row["bytes"]}), flush=True)
    write_once(output / "manifest.local.json", all_rows)
    print(json.dumps({"downloaded": len(all_rows), "validation": "pending full audio validation"}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    collect(args.output)


if __name__ == "__main__":
    main()

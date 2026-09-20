"""확장자가 아니라 전체 디코딩과 해시로 실제 오디오를 확인한다."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .common import digest, write_once


def inspect_audio(path):
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(256).lstrip().lower()
    if header.startswith((b"<!doctype html", b"<html", b"<?xml")):
        raise ValueError("Markup document masquerading as audio")
    probe_cmd = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    result = subprocess.run(probe_cmd, capture_output=True, check=True, timeout=30)
    probe = json.loads(result.stdout)
    streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    if not streams:
        raise ValueError("No audio stream")
    info = streams[0]
    duration = float(info.get("duration") or probe["format"].get("duration") or 0)
    if duration > 1800:
        raise ValueError("Audio exceeds 30-minute validation bound")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-nostdin",
           "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "22050",
           "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"]
    decoded = subprocess.run(cmd, capture_output=True, check=True, timeout=180)
    audio = np.frombuffer(decoded.stdout, dtype="<f4")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Empty or non-finite decoded PCM")
    fp_cmd = ["fpcalc", "-json", "-raw", "-length", "120", str(path)]
    fp = subprocess.run(fp_cmd, capture_output=True, timeout=60)
    fingerprint = None
    if fp.stdout:
        try:
            parsed = json.loads(fp.stdout)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed.get("fingerprint"), list) and parsed["fingerprint"]:
            fingerprint = parsed["fingerprint"]
    return {
        "sha256": digest(path), "bytes": path.stat().st_size,
        "audio_codec": info["codec_name"], "container": probe["format"].get("format_name"),
        "sample_rate": int(info["sample_rate"]), "channels": info["channels"],
        "sample_format": info.get("sample_fmt"), "bits_per_raw_sample": info.get("bits_per_raw_sample"),
        "container_duration_seconds": duration, "decoded_duration_seconds": len(audio) / 22050,
        "decoded_samples_22050_mono": len(audio),
        "decoded_pcm_sha256_22050_mono_f32le": hashlib.sha256(decoded.stdout).hexdigest(),
        "peak": float(np.max(np.abs(audio))),
        "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        "decode_command": cmd, "decode_stderr": decoded.stderr.decode(errors="replace"),
        "chromaprint_raw_120s": fingerprint, "fpcalc_returncode": fp.returncode,
        "fpcalc_stderr": fp.stderr.decode(errors="replace").strip(),
        "fingerprint_status": "available" if fingerprint else "unavailable",
    }


def validate_entry(entry, output):
    path = Path(entry["path"])
    key = hashlib.sha256(str(path).encode()).hexdigest()
    result_path = output / "records" / f"{key}.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if cached["path"] != str(path):
            raise ValueError("Cached validation path changed")
        if cached.get("sha256") and digest(path) != cached["sha256"]:
            raise ValueError(f"Audio changed after validation: {path}")
        return cached
    row = {"path": str(path), "track_id": entry["track_id"]}
    try:
        row.update(inspect_audio(path))
        if entry.get("sha256") and row["sha256"] != entry["sha256"]:
            raise ValueError("Acquisition hash differs from validated audio hash")
        row["outcome"] = "valid"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        row.update(outcome="error", error_type=type(exc).__name__, error=str(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            row["stderr"] = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
    write_once(result_path, row)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    for tool in ("ffmpeg", "ffprobe", "fpcalc"):
        if not shutil.which(tool):
            raise RuntimeError(f"Required validation tool missing: {tool}")
    doc = json.loads(args.manifest.read_text())
    entries = doc["bench"] if isinstance(doc, dict) else doc
    write_once(args.output / "inputs.json", {
        "manifest_sha256": digest(args.manifest), "tool_sha256": digest(__file__),
        "expected": len(entries),
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        "fpcalc": subprocess.check_output(["fpcalc", "-version"], text=True).strip(),
    })
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(validate_entry, e, args.output) for e in entries]):
            rows.append(future.result())
            if len(rows) % 25 == 0 or len(rows) == len(entries):
                print(json.dumps({"done": len(rows), "expected": len(entries), "outcomes": dict(Counter(r["outcome"] for r in rows))}), flush=True)
    write_once(args.output / "summary.json", {"expected": len(entries), "attempted": len(rows),
               "outcomes": dict(Counter(r["outcome"] for r in rows)),
               "fingerprints": sum(r.get("fingerprint_status") == "available" for r in rows)})
    if any(r["outcome"] != "valid" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

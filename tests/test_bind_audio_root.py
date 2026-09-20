import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_bind_audio_root_matches_bytes_without_names(tmp_path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    audio = audio_root / "unrelated-name.wav"
    audio.write_bytes(b"RIFF-test-audio-bytes")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "tracks": [{
            "track_id": "ab2-test",
            "audio_sha256": digest,
            "label": "ai",
            "source": "test",
            "bench_origin": "test",
        }]
    }))
    output = tmp_path / "runtime.json"
    report = tmp_path / "report.json"
    script = Path(__file__).parents[1] / "scripts" / "bind_audio_root.py"
    result = subprocess.run(
        [
            sys.executable, str(script), "--manifest", str(manifest),
            "--audio-root", str(audio_root), "--output", str(output),
            "--report", str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    runtime = json.loads(output.read_text())
    assert runtime["tracks"][0]["runtime_path"] == str(audio.resolve())
    assert json.loads(report.read_text())["ready"] is True

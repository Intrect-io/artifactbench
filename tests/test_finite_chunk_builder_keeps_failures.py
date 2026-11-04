# Created: 2026-11-04
# Purpose: build_artifactnet_finite_chunk_result.py must keep strict failures the chunk audit
#          does not cover (e.g. decode failures) instead of dropping those tracks.
# Dependencies: pytest
# Test Status: passing
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_artifactnet_finite_chunk_result.py"


def test_decode_failures_survive_and_recovery_applies(tmp_path):
    rows = [{"track_id": t, "label": "ai", "source": "s"} for t in ("ok", "nan_rec", "nan_bad", "dec")]
    (tmp_path / "m.json").write_text(json.dumps({"tracks": rows}))
    strict = tmp_path / "strict" / "artifactnet"
    strict.mkdir(parents=True)
    (strict / "track_probs.json").write_text(json.dumps([{"track_id": "ok", "prob": 0.9, "label": 1, "source": "s"}]))
    (strict / "inference_failures.json").write_text(json.dumps([
        {"track_id": "nan_rec", "source": "s", "stage": "inference"},
        {"track_id": "nan_bad", "source": "s", "stage": "inference"},
        {"track_id": "dec", "source": "s", "stage": "decode"}]))
    audit = {"tracks": [{"track_id": "nan_rec", "valid_chunks": 5, "total_chunks": 7, "finite_chunk_median": 0.7},
                        {"track_id": "nan_bad", "valid_chunks": 2, "total_chunks": 7, "finite_chunk_median": 0.1}]}
    (tmp_path / "audit.json").write_text(json.dumps(audit))
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(tmp_path / "m.json"),
                    "--strict-result", str(tmp_path / "strict"), "--chunk-audit", str(tmp_path / "audit.json"),
                    "--output", str(out)], check=True, capture_output=True)
    probs = {p["track_id"] for p in json.loads((out / "artifactnet" / "track_probs.json").read_text())}
    fails = {f["track_id"]: f["stage"] for f in json.loads((out / "artifactnet" / "inference_failures.json").read_text())}
    assert probs == {"ok", "nan_rec"}
    assert fails == {"nan_bad": "inference", "dec": "decode"}

"""Manifest loader + audio utilities (public runner)."""
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as TAF

# WAV-quality sources eligible for codec invariance tests
WAV_SOURCES = {
    "sonics_real", "fma_hardneg", "youtube_hardneg", "mom_real_wav", "mom_extra_real",
    "aime_musicgen_small", "aime_musicgen_medium", "aime_musicgen_large",
    "aime_riffusion", "aime_stable_audio_v1", "aime_stable_audio_v2",
    "aime_suno_v3", "aime_suno_v35", "aime_udio",
}


def load_manifest(path, split="bench", bench_origin=None, protocol_split=None):
    """Load ArtifactBench public or private runtime manifest JSON.

    Supports:
      1. ArtifactBench v1 format: {"bench": [...], "metadata": {...}}
      2. ArtifactBench v2 format: {"tracks": [...], ...}
      3. Legacy training format: {"train": [...], "test": [...]}

    Args:
        split: "test" | "train" | "all" | "bench".
        bench_origin: "test" | "train" | None — optional historical-origin filter.
            This field alone is not evidence that a row is unseen by every model.
    """
    with open(path) as f:
        m = json.load(f)

    if "bench" in m:
        entries = m["bench"]
        if bench_origin:
            entries = [e for e in entries if e.get("bench_origin") == bench_origin]
    elif "tracks" in m:
        entries = m["tracks"]
        if bench_origin:
            entries = [e for e in entries if e.get("bench_origin") == bench_origin]
    elif split == "all":
        entries = m.get("train", []) + m.get("test", [])
    else:
        entries = m.get(split, [])

    by_source = defaultdict(list)
    if protocol_split:
        entries = [e for e in entries if e.get("protocol_split") == protocol_split]
    for e in entries:
        if "track_id" not in e:
            raise ValueError("every manifest row must contain track_id")
        by_source[e["source"]].append(e)
    return entries, by_source


def load_audio_mono(path, target_sr=44100):
    """Load audio → mono float32 @ target_sr. Falls back to ffmpeg for exotic formats."""
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        try:
            with tempfile.TemporaryDirectory() as d:
                wav_path = Path(d) / "decoded.wav"
                subprocess.run(
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                     "-i", str(path), "-ar", str(target_sr), "-ac", "1",
                     "-f", "wav", str(wav_path)],
                    capture_output=True, timeout=30,
                )
                if not wav_path.exists():
                    return None
                audio, sr = sf.read(str(wav_path), dtype="float32")
        except Exception:
            return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = TAF.resample(torch.from_numpy(audio), sr, target_sr).numpy()
    # Clip intersample distortion — matches ArtifactNet production preprocessing
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


def encode_variant(audio_44k, codec):
    """Round-trip audio through a codec via ffmpeg.

    codec: "wav" | "mp3_128" | "aac_128" | "opus_128"
    """
    if codec == "wav":
        return audio_44k
    SR = 44100
    codec_map = {
        "mp3_128": (["-acodec", "libmp3lame", "-b:a", "128k"], "out.mp3"),
        "aac_128": (["-acodec", "aac", "-b:a", "128k"], "out.m4a"),
        "opus_128": (["-acodec", "libopus", "-b:a", "128k"], "out.opus"),
    }
    if codec not in codec_map:
        return None
    args, fname = codec_map[codec]
    with tempfile.TemporaryDirectory() as d:
        wav_in = Path(d) / "in.wav"
        sf.write(str(wav_in), audio_44k, SR)
        enc = Path(d) / fname
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(wav_in)] + args + [str(enc)]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
            decoded = load_audio_mono(str(enc), target_sr=SR)
            if decoded is None:
                return None
            if len(decoded) < len(audio_44k):
                decoded = np.pad(decoded, (0, len(audio_44k) - len(decoded)))
            else:
                decoded = decoded[:len(audio_44k)]
            return decoded
        except Exception:
            return None

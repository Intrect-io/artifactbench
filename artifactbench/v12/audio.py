"""v1.2용 float 디코딩과 양자화/손실코덱 분리 실험. 과거 helper는 변경하지 않는다."""
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as TAF


CODECS = {
    "mp3_128": (["-c:a", "libmp3lame", "-b:a", "128k"], ".mp3"),
    "aac_128": (["-c:a", "aac", "-b:a", "128k"], ".m4a"),
    "opus_128": (["-c:a", "libopus", "-b:a", "128k"], ".opus"),
}
TRANSFORMS = ("float_identity", "pcm16_only", *CODECS,
              *("pcm16_" + codec for codec in CODECS))


def load_audio_float(path, target_sr=44100):
    """표준 포맷은 기존 soundfile 경로, 대체 디코더는 명시적 float PCM을 사용한다."""
    try:
        audio, sr = sf.read(path, dtype="float32")
    except (sf.LibsndfileError, RuntimeError):
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(target_sr),
                   "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"]
        result = subprocess.run(command, capture_output=True, check=True, timeout=120)
        audio, sr = np.frombuffer(result.stdout, dtype="<f4").copy(), target_sr
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    if sr != target_sr:
        audio = TAF.resample(torch.from_numpy(audio), sr, target_sr).numpy()
    return np.clip(audio, -1, 1).astype(np.float32)


def transform_audio(audio, transform):
    """변환 PCM과 적용한 subtype/명령/길이 근거를 함께 반환한다."""
    if transform not in TRANSFORMS:
        raise ValueError(f"Unsupported transform: {transform}")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Expected finite nonempty mono float audio")
    if np.max(np.abs(audio)) > 1:
        raise ValueError("Input must follow the benchmark [-1, 1] clipping policy")
    pcm16 = transform == "pcm16_only" or transform.startswith("pcm16_")
    subtype = "PCM_16" if pcm16 else "FLOAT"
    info = {"transform": transform, "staging_subtype": subtype, "input_frames": len(audio),
            "sample_rate": 44100, "alignment": "retain decoded start; trim or zero-pad tail",
            "output_clipped_to_unit_interval": True}
    with tempfile.TemporaryDirectory(prefix="artifactbench-codec-") as directory:
        source = Path(directory) / "input.wav"
        sf.write(source, audio, 44100, subtype=subtype)
        if transform in ("float_identity", "pcm16_only"):
            decoded, _ = sf.read(source, dtype="float32")
        else:
            codec = transform.removeprefix("pcm16_")
            arguments, suffix = CODECS[codec]
            encoded = Path(directory) / ("encoded" + suffix)
            encode_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                          "-i", str(source), *arguments, str(encoded)]
            subprocess.run(encode_cmd, capture_output=True, check=True, timeout=60)
            decode_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                          "-i", str(encoded), "-map", "0:a:0", "-ac", "1", "-ar", "44100",
                          "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"]
            output = subprocess.run(decode_cmd, capture_output=True, check=True, timeout=60)
            decoded = np.frombuffer(output.stdout, dtype="<f4").copy()
            info.update(encode_arguments=arguments, decode_format="pcm_f32le", encoded_bytes=encoded.stat().st_size)
    if not len(decoded) or not np.isfinite(decoded).all():
        raise ValueError(f"Invalid decoded transform: {transform}")
    info["decoded_frames_before_alignment"] = len(decoded)
    info["tail_adjustment_frames"] = len(audio) - len(decoded)
    if len(decoded) < len(audio):
        decoded = np.pad(decoded, (0, len(audio) - len(decoded)))
    else:
        decoded = decoded[:len(audio)]
    return np.clip(decoded, -1, 1).astype(np.float32), info

"""DeepFense 원본 float64/native-rate 경로와 명시적인 AAC/Opus 디코더 확장."""
import hashlib
import io
import json
from pathlib import Path
import subprocess

import librosa
import numpy as np
import soundfile as sf
import torch

CONDITION = 'deepfense-native-soxr/2'
CONTRACT = {
    'name': CONDITION, 'supported_decoder': 'SoundFile float64, always_2d=False',
    'unsupported_decoder': 'Only AAC or Opus: FFmpeg native-rate/channel pcm_f64le WAV, then SoundFile float64',
    'mono': 'arithmetic mean in float64', 'resampling': 'native sample rate directly to 16000; librosa soxr_hq, fix=True, scale=False',
    'padding': 'first 64000 samples, repeat if short, using upstream pad_combined(random_pad=False)',
    'cast': 'float32 after resampling and padding', 'clipping': 'none',
    'normalization': 'checkpoint EAT frontend owns DC removal and Fbank normalization',
    'score': 'softmax(two-class logits)[spoof=0]',
    'decoded_frames_scope': 'native decoded frames before mono, resampling and padding',
    'extension_limit': 'AAC/Opus fallback is a benchmark decoder extension; not claimed identical to an upstream unsupported-format path',
}


def native_decode(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError('Native input must be a regular local audio file')
    decoder, source_error = 'soundfile_native_f64', None
    try:
        audio, sample_rate = sf.read(path, dtype='float64', always_2d=False)
    except sf.LibsndfileError as exc:
        source_error = type(exc).__name__
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate,channels,codec_name', '-of', 'json', str(path)],
            check=True, capture_output=True, text=True, timeout=30)
        streams = json.loads(probe.stdout).get('streams', [])
        if len(streams) != 1 or streams[0].get('codec_name') not in ('aac', 'opus'):
            raise ValueError('The declared native decoder extension supports AAC or Opus only') from exc
        stream = streams[0]
        native_rate, channels = int(stream['sample_rate']), int(stream['channels'])
        if native_rate <= 0 or channels <= 0:
            raise ValueError('Invalid native stream sample rate or channel count')
        # -ar/-ac/-af 없음: 여기서는 resample/downmix하지 않는다. WAV header도 대조한다.
        decoded = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-threads', '1', '-i', str(path), '-map', '0:a:0', '-vn', '-c:a', 'pcm_f64le',
            '-f', 'wav', 'pipe:1'], check=True, capture_output=True, timeout=120)
        audio, sample_rate = sf.read(io.BytesIO(decoded.stdout), dtype='float64', always_2d=False)
        actual_channels = 1 if audio.ndim == 1 else audio.shape[1]
        if sample_rate != native_rate or actual_channels != channels:
            raise ValueError('FFmpeg changed native sample rate or channel count')
        decoder = 'ffmpeg_'+stream['codec_name']+'_native_f64'
    if (audio.dtype != np.float64 or audio.ndim not in (1, 2) or not audio.size
            or sample_rate <= 0 or not np.isfinite(audio).all()):
        raise ValueError('Expected nonempty finite native float64 audio')
    return audio, sample_rate, {'decoder': decoder, 'upstream_decoder_error_type': source_error}


def prepare_waveform(audio, sample_rate):
    """원본 float64 mean/resample/pad 순서를 유지한다. 전송 float32 입력은 호출자가 승격한다."""
    audio = np.asarray(audio)
    if (audio.dtype != np.float64 or audio.ndim not in (1, 2) or not audio.size
            or not np.isfinite(audio).all() or type(sample_rate) is not int or sample_rate <= 0):
        raise ValueError('Expected finite float64 native audio and an integer sample rate')
    native_frames = len(audio)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000,
                                 res_type='soxr_hq', fix=True, scale=False)
    resampled_frames = len(audio)
    if not resampled_frames or not np.isfinite(audio).all():
        raise ValueError('Invalid resampled waveform')
    from deepfense.data.transforms.transforms import pad_combined
    audio = pad_combined(audio, max_len=64000, random_pad=False, pad_type='repeat')
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.shape != (64000,) or not np.isfinite(audio).all():
        raise ValueError('Invalid float32 model input after padding')
    metadata = {'condition': CONDITION, 'native_sample_rate': sample_rate, 'native_channels': channels,
        'native_frames': native_frames, 'resampled_frames': resampled_frames,
        'model_sample_rate': 16000, 'model_frames': 64000,
        'model_input_sha256': hashlib.sha256(audio.astype('<f4', copy=False).tobytes()).hexdigest(),
        'peak': float(np.abs(audio).max()), 'rms': float(np.sqrt(np.mean(audio.astype(np.float64)**2)))}
    return audio, metadata


def load_native(path):
    audio, sample_rate, decoding = native_decode(path)
    prepared, metadata = prepare_waveform(audio, sample_rate)
    return prepared, dict(metadata, **decoding)


@torch.no_grad()
def score_prepared(model, audio):
    if (model.max_len != 64000 or model.spoof_idx != 0 or model.input_sr != 16000
            or audio.dtype != np.float32 or audio.shape != (64000,) or not np.isfinite(audio).all()):
        raise ValueError('Unexpected pinned DeepFense model/input contract')
    output = model.model(torch.from_numpy(audio).unsqueeze(0).to(model.device))
    logits = output.get('logits')
    if not isinstance(logits, torch.Tensor) or logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise ValueError('Expected finite labelled two-class DeepFense logits')
    return float(torch.softmax(logits, -1)[0, 0]), logits.detach().cpu().tolist()[0]

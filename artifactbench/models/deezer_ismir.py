"""Deezer ISMIR 2025 LR detector adapter.

Logistic regression over a hand-crafted "fakeprint" spectral fingerprint.
Based on Afchar et al., "A Fourier Explanation of AI-music Artifacts"
(ISMIR 2025, arXiv:2506.19108).

Pipeline:
  1. Audio → 16 kHz mono
  2. Power STFT (n_fft=8192) → 10·log10
  3. Mean spectrum over channels + time
  4. Quadratic lower-hull subtraction (10-bin sliding minima)
  5. [1000, 8000] Hz bandwidth → 3,585 features
  6. Clip to [0, 5] dB, normalise by max → fakeprint
  7. ONNX logistic regression → P(AI)

Weights: HF Hub lofcz/ai-music-detector (~3.6K parameters).
"""
from pathlib import Path

import numpy as np

from .base import BenchModel


DEFAULT_HF_REPO = "lofcz/ai-music-detector"
DEFAULT_ONNX_FILENAME = "ai_music_detector.onnx"

# HF model config (preprocessing_config.json) — n_fft=8192, sr=16000.
# NOTE: Official ISMIR repo uses n_fft=16384; HF weights trained with 8192.
N_FFT = 8192
SR_MODEL = 16000
FMIN = 1000
FMAX = 8000
HULL_AREA = 10
MAX_DB = 5
MIN_DB = -45
MAX_DURATION = 180  # seconds


def _lower_hull(x, area=HULL_AREA):
    idx, hull = [], []
    for i in range(len(x) - area + 1):
        patch = x[i:i + area]
        rel = int(np.argmin(patch))
        abs_i = rel + i
        if abs_i not in idx:
            idx.append(abs_i)
            hull.append(patch[rel])
    if idx[0] != 0:
        idx.insert(0, 0); hull.insert(0, x[0])
    if idx[-1] != len(x) - 1:
        idx.append(len(x) - 1); hull.append(x[-1])
    return np.array(idx), np.array(hull)


def _curve_profile(x, c, f_range=(FMIN, FMAX), min_db=MIN_DB):
    from scipy import interpolate
    # HF model expects exactly 3585 bins → inclusive boundary.
    cutoff = np.where((f_range[0] <= x) & (x <= f_range[1]))
    x_, c_ = x[cutoff], c[cutoff]
    lower_x, lower_c = _lower_hull(c_, area=HULL_AREA)
    hull_curve = interpolate.interp1d(x_[lower_x], lower_c,
                                      kind="quadratic")(x_)
    hull_curve = np.clip(hull_curve, min_db, None)
    return x_, np.clip(c_ - hull_curve, 0, None)


def _fakeprint(stft_db, sr=SR_MODEL):
    """STFT (C, F, T) dB → fakeprint (3585,)."""
    fp = np.mean(stft_db, axis=(0, 2))
    x_real = np.linspace(0, sr / 2, num=len(fp))
    _, fp_curve = _curve_profile(x_real, fp, (FMIN, FMAX))
    fp_curve = np.clip(fp_curve, 0, MAX_DB)
    fp_curve = fp_curve / (1e-6 + np.max(fp_curve))
    return fp_curve.astype(np.float32)


class DeezerISMIRModel(BenchModel):
    """Deezer ISMIR 2025 fakeprint + LR detector."""

    name = "Deezer ISMIR LR"
    params = 3_586        # 3585 weights + 1 bias
    input_sr = 44100      # bench feeds 44.1k; we resample internally
    input_duration = 0.0  # variable; uses up to 180 s
    paper_ref = "arXiv:2506.19108"

    def __init__(self, onnx_path: str | None = None,
                 hf_repo: str = DEFAULT_HF_REPO,
                 hf_filename: str = DEFAULT_ONNX_FILENAME):
        self.onnx_path = onnx_path
        self.hf_repo = hf_repo
        self.hf_filename = hf_filename
        self.sess = None
        self.device = "cpu"
        self._stft_transformer = None

    def load(self, device: str = "cuda") -> None:
        import onnxruntime as ort
        import torch
        import torchaudio

        if self.onnx_path is None:
            from huggingface_hub import hf_hub_download
            self.onnx_path = hf_hub_download(repo_id=self.hf_repo,
                                             filename=self.hf_filename)

        # LR is tiny; CPU is fastest (avoids cuDNN version pinning).
        providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.device = device  # logged only; actually runs on CPU

        self._input_name = self.sess.get_inputs()[0].name
        self._output_name = self.sess.get_outputs()[0].name
        self._stft_transformer = torchaudio.transforms.Spectrogram(
            n_fft=N_FFT, power=2)
        self._torch = torch
        self._torchaudio = torchaudio

    def _spectrogram(self, audio_44k: np.ndarray) -> np.ndarray:
        """44.1 kHz mono → (C=1, F, T) dB power-spectrogram at 16 kHz."""
        import soxr
        # Cap duration for memory + consistency with ISMIR repo (max 180 s).
        max_samples_44k = int(MAX_DURATION * self.input_sr)
        if len(audio_44k) > max_samples_44k:
            audio_44k = audio_44k[:max_samples_44k]

        # Resample to 16 kHz
        audio_16k = soxr.resample(audio_44k, self.input_sr, SR_MODEL)
        # Shape (T, C=1) to match ISMIR repo convention (channels_first=False).
        p = audio_16k.reshape(-1, 1).astype(np.float32)
        stft = self._stft_transformer(self._torch.Tensor(p.T)).numpy()
        stft_db = 10 * np.log10(np.clip(stft, 1e-10, 1e6))
        return stft_db  # (C=1, F=4097, T)

    def forward(self, audio_44k: np.ndarray) -> float:
        if audio_44k.ndim > 1:
            audio_44k = audio_44k.mean(axis=-1)
        stft_db = self._spectrogram(audio_44k)
        fp = _fakeprint(stft_db).reshape(1, -1)
        out = self.sess.run([self._output_name], {self._input_name: fp})[0]
        return float(np.asarray(out).reshape(-1)[0])

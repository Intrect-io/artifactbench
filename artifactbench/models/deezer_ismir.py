"""Adapter for the public Deezer ISMIR 2025 fakeprint detector."""

from __future__ import annotations

import numpy as np

from .assets import materialize_hf_onnx_bundle
from .base import BenchModel

DEFAULT_HF_REPO = "lofcz/ai-music-detector"
DEFAULT_ONNX_FILENAME = "ai_music_detector.onnx"
DEFAULT_HF_REVISION = "d2180598fed79e3f917e8050a00439982466e5c6"
N_FFT = 8192
SR_MODEL = 16000
FMIN = 1000
FMAX = 8000
HULL_AREA = 10
MAX_DB = 5
MIN_DB = -45
MAX_DURATION = 180


def _lower_hull(values: np.ndarray, area: int = HULL_AREA) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = []
    hull: list[float] = []
    for offset in range(len(values) - area + 1):
        patch = values[offset:offset + area]
        absolute = int(np.argmin(patch)) + offset
        if absolute not in indices:
            indices.append(absolute)
            hull.append(float(values[absolute]))
    if not indices:
        return np.array([0, len(values) - 1]), values[[0, -1]]
    if indices[0] != 0:
        indices.insert(0, 0)
        hull.insert(0, float(values[0]))
    if indices[-1] != len(values) - 1:
        indices.append(len(values) - 1)
        hull.append(float(values[-1]))
    return np.asarray(indices), np.asarray(hull)


def _curve_profile(
    frequencies: np.ndarray,
    curve: np.ndarray,
    frequency_range: tuple[int, int] = (FMIN, FMAX),
    min_db: int = MIN_DB,
) -> np.ndarray:
    from scipy import interpolate

    selected = np.where(
        (frequency_range[0] <= frequencies) & (frequencies <= frequency_range[1])
    )
    x_values, curve_values = frequencies[selected], curve[selected]
    lower_x, lower_curve = _lower_hull(curve_values)
    hull_curve = interpolate.interp1d(
        x_values[lower_x], lower_curve, kind="quadratic"
    )(x_values)
    hull_curve = np.clip(hull_curve, min_db, None)
    return np.clip(curve_values - hull_curve, 0, None)


def _fakeprint(stft_db: np.ndarray, sr: int = SR_MODEL) -> np.ndarray:
    profile = np.mean(stft_db, axis=(0, 2))
    frequencies = np.linspace(0, sr / 2, num=len(profile))
    fingerprint = _curve_profile(frequencies, profile)
    fingerprint = np.clip(fingerprint, 0, MAX_DB)
    fingerprint /= 1e-6 + np.max(fingerprint)
    return fingerprint.astype(np.float32)


class DeezerISMIRModel(BenchModel):
    """Fakeprint and logistic-regression detector from Afchar et al."""

    name = "Deezer ISMIR LR"
    params = 3_586
    input_sr = 44100
    input_duration = 0.0
    paper_ref = "arXiv:2506.19108"

    def __init__(
        self,
        onnx_path: str | None = None,
        hf_repo: str = DEFAULT_HF_REPO,
        hf_filename: str = DEFAULT_ONNX_FILENAME,
        hf_revision: str = DEFAULT_HF_REVISION,
    ):
        self.onnx_path = onnx_path
        self.hf_repo = hf_repo
        self.hf_filename = hf_filename
        self.hf_revision = hf_revision
        self.provenance = {
            "hf_repo": hf_repo,
            "hf_filename": hf_filename,
            "hf_revision": hf_revision,
        }
        self.sess = None
        self.device = "cpu"
        self._stft_transformer = None

    def load(self, device: str = "cuda") -> None:
        import onnxruntime as ort
        import torch
        import torchaudio

        if self.onnx_path is None:
            self.onnx_path = materialize_hf_onnx_bundle(
                self.hf_repo, self.hf_filename, self.hf_revision
            )
        self.sess = ort.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )
        self.device = device
        self._input_name = self.sess.get_inputs()[0].name
        self._output_name = self.sess.get_outputs()[0].name
        self._stft_transformer = torchaudio.transforms.Spectrogram(
            n_fft=N_FFT, power=2
        )
        self._torch = torch

    def _spectrogram(self, audio_44k: np.ndarray) -> np.ndarray:
        import soxr

        audio_44k = audio_44k[: int(MAX_DURATION * self.input_sr)]
        audio_16k = soxr.resample(audio_44k, self.input_sr, SR_MODEL)
        audio = audio_16k.reshape(1, -1).astype(np.float32)
        stft = self._stft_transformer(self._torch.from_numpy(audio)).numpy()
        return 10 * np.log10(np.clip(stft, 1e-10, 1e6))

    def forward(self, audio_44k: np.ndarray) -> float:
        if audio_44k.ndim > 1:
            audio_44k = audio_44k.mean(axis=-1)
        fingerprint = _fakeprint(self._spectrogram(audio_44k)).reshape(1, -1)
        output = self.sess.run(
            [self._output_name], {self._input_name: fingerprint}
        )[0]
        return float(np.asarray(output).reshape(-1)[0])

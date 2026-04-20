"""ArtifactNet ONNX adapter — loads pre-compiled inference build from HF Hub.

The public ONNX release encapsulates the full pipeline (STFT → UNet → HPSS →
7-channel CNN → sigmoid). No PyTorch model code is required — only onnxruntime.

HF Hub: https://huggingface.co/intrect/artifactnet (CC BY-NC 4.0)
"""
from pathlib import Path

import numpy as np

from .base import BenchModel


DEFAULT_HF_REPO = "intrect/artifactnet"
DEFAULT_ONNX_FILENAME = "artifactnet_v94_full.onnx"

CHUNK_SEC = 4.0
SR = 44100
CHUNK_SAMPLES = int(CHUNK_SEC * SR)
N_CHUNKS_DEFAULT = 7


class ArtifactNetModel(BenchModel):
    """ArtifactNet v9.4 full end-to-end ONNX inference build."""

    name = "ArtifactNet v9.4"
    params = 4_200_000
    input_sr = 44100
    input_duration = 4.0
    paper_ref = "arXiv:2604.16254"

    def __init__(self, onnx_path: str | None = None,
                 hf_repo: str = DEFAULT_HF_REPO,
                 hf_filename: str = DEFAULT_ONNX_FILENAME,
                 n_chunks: int = N_CHUNKS_DEFAULT):
        self.onnx_path = onnx_path
        self.hf_repo = hf_repo
        self.hf_filename = hf_filename
        self.n_chunks = n_chunks
        self.sess = None
        self.device = "cpu"

    def load(self, device: str = "cuda") -> None:
        import onnxruntime as ort

        if self.onnx_path is None:
            # Auto-download from HF Hub
            from huggingface_hub import hf_hub_download
            self.onnx_path = hf_hub_download(repo_id=self.hf_repo, filename=self.hf_filename)

        providers = []
        if device == "cuda":
            providers.append("CUDAExecutionProvider")
        elif device == "mps":
            # onnxruntime doesn't support MPS natively yet; fall back to CPU
            pass
        providers.append("CPUExecutionProvider")

        self.sess = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.device = device

        # Introspect input name (older builds may differ)
        self._input_name = self.sess.get_inputs()[0].name
        self._output_name = self.sess.get_outputs()[0].name

    def _forward_chunk(self, chunk: np.ndarray) -> float:
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        else:
            chunk = chunk[:CHUNK_SAMPLES]
        inp = chunk.astype(np.float32).reshape(1, -1)
        out = self.sess.run([self._output_name], {self._input_name: inp})[0]
        out = np.asarray(out).reshape(-1)
        return float(out[0])

    def forward(self, audio_44k: np.ndarray) -> float:
        """44.1 kHz mono → median P(AI) across multiple chunks."""
        if len(audio_44k) < CHUNK_SAMPLES:
            audio_44k = np.pad(audio_44k, (0, CHUNK_SAMPLES - len(audio_44k)))

        n_chunks = max(self.n_chunks, len(audio_44k) // CHUNK_SAMPLES)
        max_start = len(audio_44k) - CHUNK_SAMPLES
        if max_start <= 0:
            starts = [0] * n_chunks
        else:
            starts = np.linspace(0, max_start, n_chunks, dtype=np.int64)

        probs = [self._forward_chunk(audio_44k[s:s + CHUNK_SAMPLES]) for s in starts]
        return float(np.median(probs))

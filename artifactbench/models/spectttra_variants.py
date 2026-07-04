"""SpecTTTra beta/gamma-5s variant adapter (SONICS).

Same architecture/loader as spectttra.py's alpha-120s, parameterized by
HF repo id + segment duration. 5s variant lets ArtifactBench compare under
a segment condition closer to ArtifactNet's 4s chunking.

Source: https://huggingface.co/awsaf49/sonics-spectttra-{beta,gamma}-{5s,120s} (MIT)
Requires: pip install git+https://github.com/awsaf49/sonics.git
"""
import random

import numpy as np
import torch
import torchaudio.functional as TAF

from .base import BenchModel

SPECTTTRA_SR = 16000


class SpecTTTraVariantModel(BenchModel):
    """Generic SpecTTTra variant — set hf_repo/duration per instance."""

    paper_ref = "SONICS (ICLR 2025)"
    input_sr = 16000

    def __init__(self, hf_repo: str = "awsaf49/sonics-spectttra-beta-5s",
                 duration: float = 5.0, params: int = 18_679_693,
                 display_name: str = "SpecTTTra beta-5s"):
        self.hf_repo = hf_repo
        self.input_duration = duration
        self.params = params
        self.name = display_name
        self.n_samples = int(SPECTTTRA_SR * duration)
        self.device = "cpu"
        self.model = None

    def load(self, device: str = "cuda") -> None:
        try:
            from sonics import HFAudioClassifier
        except ImportError as e:
            raise ImportError(
                "SpecTTTra requires the `sonics` package:\n"
                "  pip install git+https://github.com/awsaf49/sonics.git"
            ) from e

        self.device = device
        self.model = HFAudioClassifier.from_pretrained(self.hf_repo)
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, audio_44k: np.ndarray) -> float:
        audio_t = torch.from_numpy(audio_44k).float()
        audio_16k = TAF.resample(audio_t, 44100, SPECTTTRA_SR).numpy()

        if len(audio_16k) >= self.n_samples:
            max_start = len(audio_16k) - self.n_samples
            start = random.randint(0, max_start) if max_start > 0 else 0
            audio_16k = audio_16k[start:start + self.n_samples]
        else:
            audio_16k = np.pad(audio_16k, (0, self.n_samples - len(audio_16k)))

        inp = torch.from_numpy(audio_16k).float().unsqueeze(0).to(self.device)
        logit = self.model(inp)
        return torch.sigmoid(logit).item()

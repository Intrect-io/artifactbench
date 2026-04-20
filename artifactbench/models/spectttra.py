"""SpecTTTra alpha-120s adapter (SONICS).

Source: https://huggingface.co/awsaf49/sonics-spectttra-alpha-120s (MIT)
Requires: pip install git+https://github.com/awsaf49/sonics.git
"""
import random

import numpy as np
import torch
import torchaudio.functional as TAF

from .base import BenchModel


SPECTTTRA_SR = 16000
SPECTTTRA_DURATION = 120
SPECTTTRA_SAMPLES = SPECTTTRA_SR * SPECTTTRA_DURATION


class SpecTTTraModel(BenchModel):
    name = "SpecTTTra alpha-120s"
    params = 18_679_693
    input_sr = 16000
    input_duration = 120.0
    paper_ref = "SONICS (ICLR 2025)"

    def __init__(self, hf_repo: str = "awsaf49/sonics-spectttra-alpha-120s"):
        self.hf_repo = hf_repo
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
        # Resample 44.1k → 16k
        audio_t = torch.from_numpy(audio_44k).float()
        audio_16k = TAF.resample(audio_t, 44100, SPECTTTRA_SR).numpy()

        if len(audio_16k) >= SPECTTTRA_SAMPLES:
            max_start = len(audio_16k) - SPECTTTRA_SAMPLES
            start = random.randint(0, max_start) if max_start > 0 else 0
            audio_16k = audio_16k[start:start + SPECTTTRA_SAMPLES]
        else:
            audio_16k = np.pad(audio_16k, (0, SPECTTTRA_SAMPLES - len(audio_16k)))

        inp = torch.from_numpy(audio_16k).float().unsqueeze(0).to(self.device)
        logit = self.model(inp)
        return torch.sigmoid(logit).item()

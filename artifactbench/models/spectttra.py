"""SpecTTTra alpha-120s adapter (SONICS).

Source: https://huggingface.co/awsaf49/sonics-spectttra-alpha-120s (MIT)
Requires: pip install git+https://github.com/awsaf49/sonics.git
"""
import numpy as np
import torch
import torchaudio.functional as TAF

from .base import BenchModel

SPECTTTRA_SR = 16000
SPECTTTRA_DURATION = 120
SPECTTTRA_SAMPLES = SPECTTTRA_SR * SPECTTTRA_DURATION
DEFAULT_HF_REPO = "awsaf49/sonics-spectttra-alpha-120s"
DEFAULT_HF_REVISION = "094b32a5545098a71c113f6ae9d5c55310564268"


class SpecTTTraModel(BenchModel):
    name = "SpecTTTra alpha-120s"
    params = 18_679_693
    input_sr = 16000
    input_duration = 120.0
    paper_ref = "SONICS (ICLR 2025)"

    def __init__(self, hf_repo: str = DEFAULT_HF_REPO,
                 hf_revision: str = DEFAULT_HF_REVISION,
                 crop_policy: str = "center"):
        if crop_policy not in {"center", "start"}:
            raise ValueError("crop_policy must be 'center' or 'start'")
        self.hf_repo = hf_repo
        self.hf_revision = hf_revision
        self.crop_policy = crop_policy
        self.provenance = {
            "hf_repo": hf_repo,
            "hf_revision": hf_revision,
            "crop_policy": crop_policy,
        }
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
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(
            repo_id=self.hf_repo,
            revision=self.hf_revision,
            allow_patterns=["config.json", "pytorch_model.bin"],
        )
        self.model = HFAudioClassifier.from_pretrained(model_dir)
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
            start = max_start // 2 if self.crop_policy == "center" else 0
            audio_16k = audio_16k[start:start + SPECTTTRA_SAMPLES]
        else:
            audio_16k = np.pad(audio_16k, (0, SPECTTTRA_SAMPLES - len(audio_16k)))

        inp = torch.from_numpy(audio_16k).float().unsqueeze(0).to(self.device)
        logit = self.model(inp)
        return torch.sigmoid(logit).item()

"""DeepFense adapter (Kheir et al., arXiv:2604.08450).

Loads a DeepFense StandardDetector checkpoint (frontend+backend fused,
e.g. EAT + Nes2Net trained on FakeMusicCaps). label_map: bonafide=1 (real),
spoof=0 (AI) -- P(AI) = probs[:, spoof_idx].

Source: https://huggingface.co/DeepFense/FakeMusicCaps_EAT_Nes2Net_NoAug_Seed42
Requires: pip install deepfense
"""
import numpy as np
import torch
import torchaudio.functional as TAF
from omegaconf import OmegaConf

from .base import BenchModel

try:
    import deepfense.models as _deepfense_models  # noqa: F401 — registry 등록 트리거
except ImportError:
    _deepfense_models = None

DEFAULT_REPO = "FakeMusicCaps_EAT_Nes2Net_NoAug_Seed42"


class DeepFenseModel(BenchModel):
    name = "DeepFense EAT_Nes2Net (FakeMusicCaps)"
    params = 0  # config에 미기재 (EAT frontend ~90M + Nes2Net head 추정)
    input_sr = 16000
    input_duration = 4.0  # max_len 64000 samples @ 16kHz (config)
    paper_ref = "DeepFense (arXiv:2604.08450)"

    def __init__(self, hf_repo: str = DEFAULT_REPO):
        self.hf_repo = hf_repo
        self.device = "cpu"
        self.model = None
        self.spoof_idx = 0  # label_map: bonafide=1, spoof=0 (config.yaml)
        self.max_len = 64000

    def load(self, device: str = "cuda") -> None:
        from deepfense.hub import download_model
        from deepfense.utils.registry import build_detector

        paths = download_model(self.hf_repo)
        cfg = OmegaConf.load(paths["config"])
        self.max_len = cfg.data.train.base_transform[0].max_len
        label_map = dict(cfg.data.label_map)
        self.spoof_idx = int(label_map.get("spoof", 0))

        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        self.device = device
        self.model = build_detector(cfg.model.type, model_cfg)
        state = torch.load(paths["checkpoint"], map_location=device)
        self.model.load_state_dict(state["model_state"])
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, audio_44k: np.ndarray) -> float:
        audio_t = torch.from_numpy(audio_44k).float()
        audio_16k = TAF.resample(audio_t, 44100, self.input_sr).numpy()

        n = self.max_len
        if len(audio_16k) >= n:
            audio_16k = audio_16k[:n]
        else:
            # config base_transform: pad_type=repeat
            reps = int(np.ceil(n / len(audio_16k)))
            audio_16k = np.tile(audio_16k, reps)[:n]

        x = torch.from_numpy(audio_16k).float().unsqueeze(0).to(self.device)
        out = self.model(x)
        scores = out["scores"]
        if scores.ndim > 1 and scores.shape[-1] > 1:
            probs = torch.softmax(scores, dim=-1)[0]
            return probs[self.spoof_idx].item()
        # 1-dim/scalar scores: 단일 로짓 -> sigmoid(logit) = P(bonafide) 가정, P(AI) = 1 - 그것
        p_bonafide = torch.sigmoid(scores).flatten()[0].item()
        return 1.0 - p_bonafide

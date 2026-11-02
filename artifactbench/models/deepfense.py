"""DeepFense adapter (Kheir et al., arXiv:2604.08450).

Loads a DeepFense StandardDetector checkpoint (frontend+backend fused,
e.g. EAT + Nes2Net trained on FakeMusicCaps). label_map: bonafide=1 (real),
spoof=0 (AI) -- P(AI) = softmax(logits)[0, spoof_idx].

Scoring uses only the labelled two-class ``logits`` output. Single-value
``scores`` outputs are rejected instead of being reinterpreted (an earlier
version treated sigmoid(score) as P(bonafide), which is not what the detector
emits; see docs/DEEPFENSE_INPUT_PATH_AUDIT_v1.2.md).

Source: https://huggingface.co/DeepFense/FakeMusicCaps_EAT_Nes2Net_NoAug_Seed42
Requires: pip install deepfense omegaconf
"""
import numpy as np
import torch
import torchaudio.functional as TAF

from .base import BenchModel

try:
    import deepfense.models as _deepfense_models  # noqa: F401 — registry 등록 트리거
except ImportError:
    _deepfense_models = None

DEFAULT_REPO = "FakeMusicCaps_EAT_Nes2Net_NoAug_Seed42"
DEFAULT_FRONTEND_REPO = "worstchan/EAT-large_epoch20_pretrain"


def two_class_probability(output, spoof_idx: int) -> float:
    """P(AI) from the detector's labelled two-class logits; anything else is an error."""
    logits = output.get("logits") if isinstance(output, dict) else None
    if (spoof_idx not in (0, 1) or not isinstance(logits, torch.Tensor)
            or tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all()):
        raise ValueError("Expected finite labelled two-class logits from DeepFense")
    return torch.softmax(logits, dim=-1)[0, spoof_idx].item()


def _replace_strings(node, target: str, replacement: str):
    """Recursively replace string values equal to ``target`` (used to pin the frontend path)."""
    if isinstance(node, dict):
        return {k: _replace_strings(v, target, replacement) for k, v in node.items()}
    if isinstance(node, list):
        return [_replace_strings(v, target, replacement) for v in node]
    if isinstance(node, str) and node == target:
        return replacement
    return node


class DeepFenseModel(BenchModel):
    name = "DeepFense EAT_Nes2Net (FakeMusicCaps)"
    params = 0  # config에 미기재 (EAT frontend ~90M + Nes2Net head 추정)
    input_sr = 16000
    input_duration = 4.0  # max_len 64000 samples @ 16kHz (config)
    paper_ref = "DeepFense (arXiv:2604.08450)"

    def __init__(self, hf_repo: str = DEFAULT_REPO, config_path: str | None = None,
                 checkpoint_path: str | None = None, frontend_path: str | None = None):
        if (config_path is None) != (checkpoint_path is None):
            raise ValueError("config_path and checkpoint_path must be given together "
                             "(a local DeepFense snapshot needs both files)")
        self.hf_repo = hf_repo
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        # Local copy of the EAT frontend; None keeps the id written in the config.
        self.frontend_path = frontend_path
        self.device = "cpu"
        self.model = None
        self.spoof_idx = 0  # label_map: bonafide=1, spoof=0 (config.yaml)
        self.max_len = 64000

    def load(self, device: str = "cuda") -> None:
        from omegaconf import OmegaConf
        from deepfense.utils.registry import build_detector

        if self.config_path is not None:
            paths = {"config": self.config_path, "checkpoint": self.checkpoint_path}
        else:
            from deepfense.hub import download_model
            paths = download_model(self.hf_repo)
        cfg = OmegaConf.load(paths["config"])
        self.max_len = cfg.data.train.base_transform[0].max_len
        label_map = dict(cfg.data.label_map)
        self.spoof_idx = int(label_map.get("spoof", 0))

        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        if self.frontend_path is not None:
            model_cfg = _replace_strings(model_cfg, DEFAULT_FRONTEND_REPO, str(self.frontend_path))
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
        return two_class_probability(self.model(x), self.spoof_idx)

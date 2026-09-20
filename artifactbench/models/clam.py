"""CLAM (MoM) adapter.

Source: https://github.com/StarkVision-AI/MoM-CLAM
Paper:  Melody or Machine (arXiv:2512.00621)

CLAM has no pip package and no HF weight distribution. You must:
  1. git clone https://github.com/StarkVision-AI/MoM-CLAM
  2. Obtain `best_model_triplet_loss_margin_0.2.pth` per that repo's instructions
  3. Pass --clam-repo <path> --clam-ckpt <path> to the benchmark CLI
     (or construct CLAMModel(clam_repo=..., clam_ckpt=...))
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as TAF

from .base import BenchModel

MERT_REPO = "m-a-p/MERT-v1-95M"
MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
W2V_REPO = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
W2V_REVISION = "caf978c8328a2cec4229b3eb0b41b162e379caa1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CLAMModel(BenchModel):
    name = "CLAM (MoM)"
    params = 194_300_000
    input_sr = 24000
    input_duration = 90.0
    paper_ref = "arXiv:2512.00621"

    def __init__(self, clam_repo: str | None = None, clam_ckpt: str | None = None,
                 use_fp16: bool = True, crop_policy: str = "center"):
        if crop_policy not in {"center", "start"}:
            raise ValueError("crop_policy must be 'center' or 'start'")
        self.clam_repo = clam_repo
        self.clam_ckpt = clam_ckpt
        self.use_fp16 = use_fp16
        self.crop_policy = crop_policy
        self.device = "cpu"
        self.provenance = {
            "clam_repo_revision": "74e3a3277e1dfe9ae9ed433b6e8c51d74e9e1d9b",
            "mert_repo": MERT_REPO,
            "mert_revision": MERT_REVISION,
            "wav2vec2_repo": W2V_REPO,
            "wav2vec2_revision": W2V_REVISION,
            "crop_policy": crop_policy,
        }

    def load(self, device: str = "cuda") -> None:
        if self.clam_repo is None or self.clam_ckpt is None:
            raise ValueError(
                "CLAM requires both --clam-repo and --clam-ckpt. "
                "Clone https://github.com/StarkVision-AI/MoM-CLAM and download its weights."
            )
        clam_repo = Path(self.clam_repo).expanduser().resolve()
        if not clam_repo.exists():
            raise FileNotFoundError(f"CLAM repo not found: {clam_repo}")
        ckpt = Path(self.clam_ckpt).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"CLAM checkpoint not found: {ckpt}")
        self.provenance["checkpoint_sha256"] = _sha256(ckpt)

        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.device = device

        self.mert_model = AutoModel.from_pretrained(
            MERT_REPO, revision=MERT_REVISION, trust_remote_code=True
        )
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MERT_REPO, revision=MERT_REVISION, trust_remote_code=True)
        self.mert_model = self.mert_model.to(device).eval()

        self.w2v_model = AutoModel.from_pretrained(
            W2V_REPO, revision=W2V_REVISION, trust_remote_code=True)
        self.w2v_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            W2V_REPO, revision=W2V_REVISION, trust_remote_code=True)
        self.w2v_model = self.w2v_model.to(device).eval()

        # Import CLAM model class from cloned repo
        sys.path.insert(0, str(clam_repo))
        try:
            from models.clam import CLAM
        except ImportError as e:
            raise ImportError(
                f"Could not import CLAM model class from {clam_repo}/models/clam.py. "
                "Make sure --clam-repo points to the root of MoM-CLAM."
            ) from e

        self.clam = CLAM(in_channel1=13, in_channel2=13, embed_dim1=768, embed_dim2=768)
        self.clam.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=False))
        self.clam = self.clam.to(device).eval()

        if self.use_fp16 and device == "cuda":
            self.mert_model = self.mert_model.half()
            self.w2v_model = self.w2v_model.half()
            self.clam = self.clam.half()

        for m in [self.mert_model, self.w2v_model, self.clam]:
            for p in m.parameters():
                p.requires_grad = False

    def _extract_embedding(self, audio_44k, model, processor, target_sr, duration=90):
        if target_sr != 44100:
            audio_t = torch.from_numpy(audio_44k).float()
            audio = TAF.resample(audio_t, 44100, target_sr).numpy()
        else:
            audio = audio_44k

        target_len = target_sr * duration
        if len(audio) > target_len:
            max_start = len(audio) - target_len
            start = max_start // 2 if self.crop_policy == "center" else 0
            audio = audio[start:start + target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))

        inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt")
        if self.use_fp16 and self.device == "cuda":
            inputs = {
                k: v.to(self.device).half() if v.is_floating_point() else v.to(self.device)
                for k, v in inputs.items()
            }
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        layer_embeddings = torch.stack(outputs.hidden_states).squeeze(1)
        return layer_embeddings.mean(dim=1).unsqueeze(0).float()

    @torch.no_grad()
    def forward(self, audio_44k: np.ndarray) -> float:
        mert_sr = self.mert_processor.sampling_rate
        w2v_sr = self.w2v_processor.sampling_rate
        emb_mert = self._extract_embedding(audio_44k, self.mert_model, self.mert_processor, mert_sr)
        emb_w2v = self._extract_embedding(audio_44k, self.w2v_model, self.w2v_processor, w2v_sr)
        if self.use_fp16 and self.device == "cuda":
            emb_mert = emb_mert.half()
            emb_w2v = emb_w2v.half()
        logit = self.clam(emb_mert, emb_w2v)
        return torch.sigmoid(logit.float()).item()

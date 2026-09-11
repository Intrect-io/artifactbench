"""AI-Music-Detection AST-60s adapter.

Audio Spectrogram Transformer fine-tuned for binary AI-vs-human music
classification (id2label: 0=human, 1=ai_generated).

Source: https://huggingface.co/AI-Music-Detection/ai_music_detection_large_60s
Requires: transformers
"""
import numpy as np
import torch
import torchaudio.functional as TAF

from .base import BenchModel

AST_SR = 16000
AST_DURATION = 60.0
AST_SAMPLES = int(AST_SR * AST_DURATION)


class ASTMusicDetectionModel(BenchModel):
    name = "AI-Music-Detection AST-60s"
    params = 90_800_000
    input_sr = 16000
    input_duration = 60.0
    paper_ref = "AI-Music-Detection/ai_music_detection_large_60s (HF)"

    def __init__(self, hf_repo: str = "AI-Music-Detection/ai_music_detection_large_60s"):
        self.hf_repo = hf_repo
        self.device = "cpu"
        self.model = None
        self.extractor = None
        self.ai_idx = 1

    def load(self, device: str = "cuda") -> None:
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.device = device
        self.model = ASTForAudioClassification.from_pretrained(self.hf_repo)
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # repo에 preprocessor_config.json 없음 — base AST extractor를
        # 이 fine-tune의 max_length(config.json, 60s=6000 frames)로 재구성
        max_length = getattr(self.model.config, "max_length", 6000)
        self.extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", max_length=max_length)
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.ai_idx = next(i for i, v in id2label.items() if "ai" in v.lower())

    @torch.no_grad()
    def forward(self, audio_44k: np.ndarray) -> float:
        audio_t = torch.from_numpy(audio_44k).float()
        audio_16k = TAF.resample(audio_t, 44100, AST_SR).numpy()

        if len(audio_16k) >= AST_SAMPLES:
            audio_16k = audio_16k[:AST_SAMPLES]
        else:
            audio_16k = np.pad(audio_16k, (0, AST_SAMPLES - len(audio_16k)))

        inputs = self.extractor(audio_16k, sampling_rate=AST_SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[0, self.ai_idx].item()

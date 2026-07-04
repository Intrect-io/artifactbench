"""FST (Fusion Segment Transformer) adapter — Mippia 2025.

Two-stage AI music detector:
  Stage-1: MERT_AudioCAT (cross-attention transformer over MERT embeddings)
  Stage-2: Fusion Segment Transformer (Gated Fusion Layer for content+structure)

Pipeline:
  1. 44.1 kHz mono → resample to 24 kHz
  2. Beat detection (beat_this/File2Beats) → downbeats
  3. Segment into 10-second windows aligned to downbeats (up to 48 segments)
  4. Stage-1: MERT extracts 768-d embedding per segment
  5. Stage-2: FusionSegmentTransformer aggregates → P(AI)

Reference: https://github.com/Mippia/FST-AI-music-detection
Paper:     arXiv:2601.13647
"""
import gc
import sys
from pathlib import Path

import numpy as np

from .base import BenchModel


FST_REPO_DEFAULT = "/home/unohee/dev/FST-AI-music-detection"
STAGE1_CKPT_DEFAULT = "checkpoints/backbone_stage1.ckpt"
STAGE2_CKPT_DEFAULT = "checkpoints/classifier_stage2.ckpt"

SR_MODEL = 24000
SEGMENT_SAMPLES = 240000  # 10 s @ 24 kHz
MAX_SEGMENTS = 48
MERT_BATCH = 8           # MERT 소배치로 GPU 메모리 절감
FILE2BEATS_RELOAD = 20   # File2Beats 재생성 주기 (메모리 누수 방지)
# 긴 트랙 처리 시간 폭발 방지. FST는 여러 10초 segment를 집계하므로
# 벤치에서는 앞 90초만 사용해 File2Beats 지연을 제한한다.
MAX_DURATION_SEC = 90


class FSTModel(BenchModel):
    """FST (Mippia 2025) — two-stage MERT + Fusion Segment Transformer."""

    name = "FST (Mippia 2025)"
    params = 174_396_803  # Stage-1 (170.2M) + Stage-2 (4.2M)
    input_sr = 44100
    input_duration = 0.0  # variable (up to 48 × 10 s)
    paper_ref = "arXiv:2601.13647"

    def __init__(self,
                 fst_repo: str = FST_REPO_DEFAULT,
                 stage1_ckpt: str | None = None,
                 stage2_ckpt: str | None = None):
        self.fst_repo = Path(fst_repo)
        self.stage1_ckpt = stage1_ckpt or str(self.fst_repo / STAGE1_CKPT_DEFAULT)
        self.stage2_ckpt = stage2_ckpt or str(self.fst_repo / STAGE2_CKPT_DEFAULT)
        self.backbone = None
        self.classifier = None
        self.file2beats = None
        self.device = "cpu"
        self._n_calls = 0  # File2Beats 재생성 카운터

    def load(self, device: str = "cuda") -> None:
        import torch

        # FST 레포를 import path에 추가
        if str(self.fst_repo) not in sys.path:
            sys.path.insert(0, str(self.fst_repo))

        from model import MERT_AudioCAT, MusicAudioClassifier
        from beat_this.inference import File2Beats

        self.device = device
        self._torch = torch

        # Stage-1: MERT_AudioCAT (FP32 — MERT weights expect FP32)
        self.backbone = MERT_AudioCAT.load_from_checkpoint(
            self.stage1_ckpt
        ).to(device).eval()

        # Stage-2: FusionSegmentTransformer (FP16 — forward() does x.half() internally)
        self.classifier = MusicAudioClassifier.load_from_checkpoint(
            self.stage2_ckpt,
            input_dim=768,
            backbone="fusion_segment_transformer",
            is_emb=True,
        ).to(device).half().eval()

        # Beat detection model (재사용)
        self.file2beats = File2Beats(checkpoint_path="final0", device=device, dbn=False)

        # preprocess.find_optimal_segment_length 함수 캐싱
        from preprocess import find_optimal_segment_length
        self._find_optimal = find_optimal_segment_length

    def _reload_file2beats_if_needed(self):
        """File2Beats GPU 캐시 누적 방지 — 주기적으로 재생성."""
        self._n_calls += 1
        if self._n_calls % FILE2BEATS_RELOAD == 0:
            from beat_this.inference import File2Beats
            del self.file2beats
            self._torch.cuda.empty_cache()
            gc.collect()
            self.file2beats = File2Beats(checkpoint_path="final0", device=self.device, dbn=False)

    def _audio_to_segments(self, audio_44k: np.ndarray) -> tuple:
        """44.1 kHz mono → (segments[48,1,240000], padding_mask[48])."""
        import soxr

        # 24 kHz 리샘플링
        audio_24k = soxr.resample(audio_44k.astype(np.float32), self.input_sr, SR_MODEL)
        wav = self._torch.tensor(audio_24k, dtype=self._torch.float32).unsqueeze(0)  # (1, T)

        # Beat detection — 일시적으로 wav 파일 거치지 않고 raw audio 사용
        # File2Beats는 path를 받기 때문에 임시 파일 사용
        import tempfile
        import soundfile as sf
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio_24k, SR_MODEL)
            _, downbeats = self.file2beats(tmp_path)
            _, cleaned = self._find_optimal(downbeats)
        except Exception:
            # fallback: 10초 슬라이딩 윈도우
            total_dur = wav.shape[1] / SR_MODEL
            cleaned = np.arange(0, max(total_dur - 10.0, 0), 10.0)[:MAX_SEGMENTS]
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

        # 세그먼트 추출
        if wav.shape[1] < SEGMENT_SAMPLES:
            pad = self._torch.zeros(1, SEGMENT_SAMPLES - wav.shape[1], dtype=self._torch.float32)
            wav = self._torch.cat([wav, pad], dim=1)

        segments = []
        for start_time in cleaned:
            s = int(start_time * SR_MODEL)
            e = s + SEGMENT_SAMPLES
            if e > wav.shape[1]:
                continue
            segments.append(wav[:, s:e])
            if len(segments) >= MAX_SEGMENTS:
                break

        if not segments:
            # 앞쪽 10초라도 사용
            if wav.shape[1] >= SEGMENT_SAMPLES:
                segments.append(wav[:, :SEGMENT_SAMPLES])
            else:
                return None, None

        stacked = self._torch.stack(segments)  # (N, 1, 240000)
        n = stacked.shape[0]
        padding_mask = self._torch.zeros(MAX_SEGMENTS, dtype=self._torch.bool)
        if n < MAX_SEGMENTS:
            # 원본 inference.py와 동일하게 zero-pad → 항상 (48, 1, 240000)
            pad = self._torch.zeros(
                MAX_SEGMENTS - n, 1, SEGMENT_SAMPLES, dtype=self._torch.float32
            )
            stacked = self._torch.cat([stacked, pad], dim=0)
            padding_mask[n:] = True

        return stacked, padding_mask

    @property
    def _no_grad(self):
        return self._torch.no_grad()

    def forward(self, audio_44k: np.ndarray) -> float:
        """44.1 kHz mono → P(AI) ∈ [0, 1]."""
        if audio_44k.ndim > 1:
            audio_44k = audio_44k.mean(axis=-1)

        # 긴 트랙 cap — File2Beats가 5분+ 파일에서 분 단위 지연 발생
        max_samples = MAX_DURATION_SEC * self.input_sr
        if len(audio_44k) > max_samples:
            audio_44k = audio_44k[:max_samples]

        self._reload_file2beats_if_needed()

        with self._torch.no_grad():
            segments, padding_mask = self._audio_to_segments(audio_44k)
            if segments is None:
                return 0.5  # 처리 불가 → 중립값

            # Stage-1: MERT 소배치 처리 (GPU 메모리 절감)
            audio_all = segments.squeeze(1).float()  # (N, 240000) CPU
            n_segs = audio_all.shape[0]
            emb_chunks = []
            for i in range(0, n_segs, MERT_BATCH):
                batch = audio_all[i:i + MERT_BATCH].to(self.device)
                _, emb_chunk = self.backbone(batch)  # (B, 768)
                emb_chunks.append(emb_chunk.cpu())
                del batch, emb_chunk
                self._torch.cuda.empty_cache()

            embedding = self._torch.cat(emb_chunks, dim=0).to(self.device)  # (N, 768)

            # Stage-2: FusionSegmentTransformer
            emb = embedding.unsqueeze(0).float()  # (1, N, 768)
            mask = padding_mask.unsqueeze(0).to(self.device)  # (1, 48)
            logit = self.classifier(emb, mask).squeeze()  # scalar
            prob = self._torch.sigmoid(logit).item()

            del embedding, emb, mask, logit

        return float(prob)

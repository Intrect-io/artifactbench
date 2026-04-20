"""Sanity suite thresholds — ArtifactBench v1."""

# Per-source pass/fail
REAL_FPR_MAX = 0.05
AI_TPR_MIN_DEFAULT = 0.90
AI_TPR_MIN_SOFT = {
    "aime_stable_audio_v1": 0.60,
    "aime_stable_audio_v2": 0.60,
}

# Codec invariance
CODEC_DELTA_MEAN_MAX = 0.15
CODEC_DELTA_MAX_MAX = 0.35

# UNet health (constant mask bug detection)
MASK_STD_MIN = 0.01

# Regression detection (vs baseline)
REGRESSION_FPR_MAX_DELTA = 0.03  # FPR 악화 3pp 이상 → regression
REGRESSION_TPR_MAX_DELTA = 0.05  # TPR 하락 5pp 이상 → regression

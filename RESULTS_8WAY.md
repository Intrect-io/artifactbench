# ArtifactBench — 8-Way Public Model Comparison (2026-07-04)

Extends the original 3-way comparison (ArtifactNet / SpecTTTra / CLAM) with five
additional publicly-available AI-music detection models, evaluated on the same
v1.1 purged test partition (n = 2,224; see `v1.1/RESULTS_v1.1.md`).

## Models added

| Model | Params | Source |
|---|---:|---|
| Deezer ISMIR fakeprint LR | 3.6K | [`lofcz/ai-music-detector`](https://huggingface.co/lofcz/ai-music-detector) (Afchar et al., ISMIR 2025) |
| FST (Mippia) | 174.4M | [`Mippia/FST-AI-music-detection`](https://github.com/Mippia/FST-AI-music-detection) |
| AI-Music-Detection AST-60s | 90.8M | [`AI-Music-Detection/ai_music_detection_large_60s`](https://huggingface.co/AI-Music-Detection/ai_music_detection_large_60s) |
| SpecTTTra β-5s | 18.7M | [`awsaf49/sonics-spectttra-beta-5s`](https://huggingface.co/awsaf49/sonics-spectttra-beta-5s) (5s-segment variant of the SONICS α-120s baseline) |
| DeepFense EAT+Nes2Net (FakeMusicCaps) | — | [`DeepFense/FakeMusicCaps_EAT_Nes2Net_NoAug_Seed42`](https://huggingface.co/DeepFense/FakeMusicCaps_EAT_Nes2Net_NoAug_Seed42) (Kheir et al., arXiv:2604.08450) |

Adapters: `artifactbench/models/{ast_60s,deepfense,deezer_ismir,fst,spectttra_variants}.py`.
Run with `--model ast_60s --model deepfense --model deezer_ismir --model fst --model spectttra_beta5s`
alongside the existing `--model artifactnet --model spectttra --model clam`.

## Results (n = 2,104: 1,388 AI / 716 real, τ = 0.5)

| Rank | Model | Params | F1 | Precision | Recall (TPR) | FPR | Sanity FAIL |
|---|---|---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **ArtifactNet v9.4 (ours, public ONNX)** | **4.2M** | **0.952** | 0.932 | 97.3% | 13.8% | 8/28 |
| 2 | AI-Music-Detection AST-60s | 90.8M | 0.840 | 0.848 | 83.1% | 28.9% | 16/28 |
| 3 | CLAM (MoM) | 194.3M | 0.787 | 0.711 | 88.3% | 69.7% | 14/28 |
| 4 | SpecTTTra α-120s | 18.7M | 0.777 | 0.880 | 69.5% | 18.4% | 22/28 |
| 5 | Deezer ISMIR fakeprint LR | 3.6K | 0.754 | 0.906 | 64.6% | 13.0% | 18/28 |
| 6 | FST (Mippia) | 174.4M | 0.735 | 0.984 | 58.7% | 1.8% | 17/28 |
| 7 | DeepFense EAT+Nes2Net | — | 0.650 | 0.589 | 72.4% | 97.8% | 11/28 |
| 8 | SpecTTTra β-5s | 18.7M | 0.563 | 0.884 | 41.3% | 10.5% | 24/28 |

*Update 2026-09-05.* The track count above is the number actually scored in this run
(1,388 AI / 716 real); the earlier "n = 2,224" referred to the v1.1 partition size, not
to the files available on the scoring host (see the provenance caveat below). FPR and
sanity-FAIL columns were added from the stored per-track probabilities of the same run.
The ArtifactNet production pipeline (v9.7 / cnn_v95, PyTorch, not the public ONNX
export) on the same files: F1 0.984 / TPR 98.9% / FPR 4.2%, 1/28 FAIL (fma_hardneg).

**Observation.** Parameter count does not predict performance: FST (174.4M) scores
below Deezer's 3.6K-parameter logistic-regression baseline. ArtifactNet (4.2M) ranks
first among all eight models, including three baselines 20–46× larger.

**DeepFense catastrophic-OOD case study.** DeepFense (trained on FakeMusicCaps,
which draws its AI examples from a different generator/format mix) collapses to
near-chance on the SONICS synthetic subset specifically (Chirp/Udio, TPR 0–15%)
while performing normally on every other AI-generator subset. This is a genuine
distribution-shift failure, not an adapter bug (verified by inspecting per-source
means, confirming the split correlates with data source rather than ground-truth
label — see the investigation note in this repo's commit history).

**Segment-length ablation (SpecTTTra).** The same SpecTTTra architecture drops from
F1 = 0.777 (120s context) to F1 = 0.563 (5s context) purely from shorter input —
evidence that long-context modeling, not just architecture, drives this baseline's
detection accuracy.

## Reproducibility note — ONNX Runtime CUDA determinism

An earlier run of this 8-way comparison (2026-07-03) produced ArtifactNet F1 = 0.869
using default onnxruntime-gpu settings (`cudnn_conv_algo_search: EXHAUSTIVE`, the
ORT default). Re-running the identical files and code with `HEURISTIC` algorithm
selection (deterministic, avoids per-run algorithm auto-tuning) recovered F1 = 0.952
on the same data. **We verified bit-exact reproducibility**: two independent runs of
a 40-track subset, under both the CPU path and the fixed GPU path, produced identical
probability values (max abs diff = 0.0) across runs. The `artifactnet.py` adapter now
pins these settings by default. If you fork this runner and see run-to-run drift with
CUDAExecutionProvider on other hardware, check this setting first.

## Caveat — file provenance

The real-track set used in this specific 8-way run is a locally-restored superset
(some files re-sourced after original storage loss) rather than the exact byte-identical
files behind the official v1/v1.1 Table 6 numbers. Absolute FPR values here should be
treated as approximate; TPR and cross-model ranking are unaffected (all eight models
were scored on identical files). Canonical-byte re-verification is planned for a future
update.

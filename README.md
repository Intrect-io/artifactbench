# ArtifactBench

> A public runner for evaluating AI-generated music detectors on the **ArtifactBench v1**
> benchmark, with baseline adapters for **ArtifactNet**, **CLAM**, and **SpecTTTra**.

[![paper](https://img.shields.io/badge/arXiv-2604.16254-b31b1b.svg)](https://arxiv.org/abs/2604.16254)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97-dataset-yellow)](https://huggingface.co/datasets/intrect/artifactbench)
[![model](https://img.shields.io/badge/%F0%9F%A4%97-model-yellow)](https://huggingface.co/intrect/artifactnet)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Why

AI music detectors need evaluation across source distributions, generator
versions, and audio transports. Existing resources such as SONICS and MoM
already study synthetic-song detection; MoM explicitly includes OOD evaluation.
ArtifactBench adds a source-level and transport-focused audit protocol.

The legacy inventory uses 22 AI source/version cells and six real source cells.
These are not 22 independent generator families. Manifest revisions and leak
purges have changed the counts; use the exact manifest hash, not the original
nominal 6,200-track target. The corrected September 5 test selection has 2,164
evaluation entries across 28 cells, including three duplicate/excerpt pairs
identified in the v1.2 construction audit.

**Version 1.2 is a locally verified rc2 candidate, not yet publicly released.**
The first frozen candidate was invalidated by a Udio file-identity bug; its
incomplete run is preserved.
See [the correction audit](docs/UDIO_IDENTITY_CORRECTION_v1.2.md). A corrected
rc2 has passed validation and its eight-model evaluation has restarted. The current plan is in
[PLAN_v1.2.md](PLAN_v1.2.md), with the score-blind sampling and evaluation rules
in [docs/PROTOCOL_v1.2.md](docs/PROTOCOL_v1.2.md). The primary selection contains
2,579 entries / 2,576 identified recording groups, including 200 Suno v5.5,
200 Udio recordings created in 2026, and 15 version-labelled official demos.
See [the draft dataset card](docs/DATASET_CARD_v1.2.md). Eight-model evaluation
and the paper are complete as a reproducible draft. Historical results below are
not v1.2 results.
The verified artifact inventory and remaining publication gates are tracked in
[release readiness](docs/RELEASE_READINESS_v1.2.md).
The [metadata-only notebook](notebooks/v1.2_metadata_quickstart.ipynb) is executable
without audio, model weights, or a GPU; setup is in the dataset card. Passing it
verifies metadata consistency, not semantic source identity or eight-model
results. All six metadata cells have been executed against the corrected rc2.
The metadata-only ZIP also passed checks after extraction outside the workspace,
including an actual six-cell notebook run. See the
[bundle scope and commands](docs/REPRODUCING_v1.2.md#independent-metadata-bundle).
This does not establish full detector reproduction or a public release.

The first completed rc2 ArtifactNet run scored 2,578/2,579 entries. At raw 0.5,
legacy F1 is 0.9887 but Suno v5.5 detection is 10/200 (5.0%; 7.5% with the
existing operating stack), versus 200/200 Udio detections. All native Suno files
are AAC and Udio files are MP3: this is an observed source/transport contrast,
not a generator-version-only causal estimate. The [LaTeX draft](paper/main.tex)
reports this first checkpoint separately; the full eight-checkpoint comparison
and transport results are included in the rc2 results bundle. Do not substitute
the historical SONICS table below.

The second completed rc2 run, SpecTTTra alpha, scored all 2,579 entries. At raw
0.5, legacy F1 is 0.7716 with FPR 18.78%, while native Suno detection is
169/200 (84.5%) and Udio detection is 108/200 (54.0%). These source-specific
differences are not a universal ranking: ArtifactNet's higher legacy F1 does
not imply higher native Suno recall.

AST scored all 2,579 entries, detecting
199/200 in each native cohort, but with 25.98% false positives on legacy real
recordings (legacy F1 0.8363). These are separate recall and false-positive
measurements, not a universal ranking.

SpecTTTra beta 5s is the third completed rc2 run, also scoring all 2,579 entries:
legacy F1 0.5564, FPR 11.71%; native Suno 84/200 (42.0%) and Udio 36/200 (18.0%).
Each completed run has its own 2,000-replicate report. These are checkpoint-specific observations, not a controlled
ablation of input duration. See the current evidence in the plan above.

All eight original runs are complete. CLAM scored all 2,579 entries:
legacy F1 0.7607, FPR 69.76%; native Suno 147/200 (73.5%) and Udio 166/200 (83.0%).
The original DeepFense run remains under an
[input-path interpretation hold](docs/DEEPFENSE_INPUT_PATH_AUDIT_v1.2.md).
Its reference-consistent native-input condition, `deepfense-native-soxr/2`,
passed a separate 32-source CPU smoke and scored all 2,579 entries in the
native-input condition.
It preserves float64 native decoding/direct 16-kHz upstream preprocessing and
explicitly adds AAC/Opus fallback; original scores remain unchanged.
The full comparison and transport results are included in the rc2 public bundle.

For locally acquired audio, the [path-binding workflow](docs/LOCAL_AUDIO_BINDING_v1.2.md)
checks encoded-file hashes against the public metadata and writes a separate
private runnable manifest only when every required file matches. It performs
no download or inference and does not establish redistribution rights.

The [external baseline asset guide](docs/EXTERNAL_ASSETS_v1.2.md) supplies pinned
source revisions, official checkpoint links and a model-free hash checker.
Freshly acquired CLAM/FST/BeatThis files match the evaluated asset identities;
this is asset-preparation evidence, not fresh detector inference.

A [raw P(AI)-only ONNX candidate](docs/ARTIFACTNET_ONNX_v1.2.md) now embeds the
evaluated UNet/CNN and preserves song-wide normalization and RMS aggregation.
The source/boundary check completed on 34 recordings: 33 comparable scores,
maximum difference 0.000509, zero raw-0.5 flips, one original failure retained.
A deterministic source-smoke-labelled ZIP also reproduced one actual waveform
after fresh extraction into a Torch-free runtime outside the checkout.
Full rc2 ONNX parity is not yet verified; an earlier raw candidate failed the
fixed 1e-3 tolerance (first failed recording difference 0.0010404). Diagnostic
and alternative-probe results are documented without adopting a new candidate
or relaxing the policy. This remains a local engineering candidate,
not a newly published model or a replacement for the original benchmark scores.

A [public-score reproduction ZIP](docs/RESULTS_REPRODUCING_v1.2.md#build-and-independently-execute-the-public-score-zip)
now passes actual notebook execution after fresh extraction and reproduces all
nine rc2 statistical files byte-for-byte. This is saved-score reproduction
evidence, not fresh detector inference or a published release.

## Baseline results — SONICS full-test (23,288 tracks)

| Model | Params | F1 | Precision | Recall | FPR | AUC |
|------|-------:|:---:|:---:|:---:|:---:|:---:|
| **ArtifactNet v9.5** | **4.2M** | **0.9993** | **0.9993** | **99.93%** | **0.09%** | **0.99999865** |
| SpecTTTra α-120s | 18.7M | 0.8874 | 0.8610 | 91.55% | 17.97% | 0.9303 |
| CLAM | 194M | 0.7652 | 0.6351 | 96.24% | 67.16% | 0.8222 |

Reproduce via the **Three-way comparison** recipe below.

## Install

```bash
git clone https://github.com/Intrect-io/artifactbench
cd artifactbench
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
# For CLAM (MERT + Wav2Vec2 feature extractors via transformers)
pip install -e .[clam]

# For SpecTTTra (SONICS)
pip install git+https://github.com/awsaf49/sonics.git
```

System requirement: `ffmpeg` on PATH (required for codec invariance tests).

## Quickstart — ArtifactNet only

By default, ArtifactNet loads the current production CNN and UNet specified in
the local ArtifactNet repository's `src/pipeline/model_config.py`. Set
`ARTIFACTNET_REPO` if the repository is not at `/home/unohee/dev/ArtifactNet`.
The adapter reports the raw RMS-weighted mean probability; it does not apply
the deployment rescue gate. Use `--artifactnet-onnx PATH` for legacy ONNX weights.

```bash
# Grab the ArtifactBench v1 manifest
wget https://huggingface.co/datasets/intrect/artifactbench/resolve/main/artifactbench_v1_manifest.json

# Run ArtifactNet on a small sample
python -m artifactbench.bench \
    --model artifactnet \
    --manifest artifactbench_v1_manifest.json \
    --n-per-source 20 \
    --n-codec-pair 10 \
    --output results/smoke
```

You'll get `results/smoke/artifactnet/{report.md, per_source.json, track_probs.json, codec_pair.json}`.

### Local production rerun (2026-09-05)

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/unohee/RTX_ENV/bin/python -u \
    -m artifactbench.rerun_current --output out/current_production_260905_recovered \
    --codec-pairs 50 --recover-fma-from out/current_production_260905_full
```

This local recipe evaluates all 28 sources in the recovered test partition,
intersected with the August leak-purged manifest (2,164 tracks). It checks for
remaining recording-stem collisions against the bigset train/val/test pool,
records checkpoint and source hashes, and checkpoints every scored track.
It writes raw-detector metrics at 0.5 and separately measures the current CNN
threshold plus generation-matched rescue gate. AcoustID, the file integrity
gate, and codec TTA are outside that operating-point measurement.
The codec test uses 50 tracks with WAV/MP3/AAC/Opus variants.
The recovered run resolves 120 FMA files that contained HTML despite their MP3
extensions to the same track IDs in the local `fma_full` dataset. It preserves
the original files, records replacement paths and hashes, and inherits only
successful rows from the verified earlier run. The earlier runner source is
preserved as `runner_snapshot.py` alongside its provenance.
`summary.json` is written only after every selected track has been attempted and
unchanged input hashes are verified. Inference failures are retained in
`errors.json` and excluded from scored-only metrics, with their count reported
explicitly. GPU OOM stops the run so resource contention is not scored as a model
failure. A rerun with changed inputs requires a new output directory.

## Three-way comparison

```bash
# Install SpecTTTra first
pip install git+https://github.com/awsaf49/sonics.git

# Clone CLAM and fetch its weights (per upstream instructions)
git clone https://github.com/StarkVision-AI/MoM-CLAM ~/dev/MoM-CLAM
# ... follow upstream README to obtain best_model_triplet_loss_margin_0.2.pth ...

# Three-way run
python -m artifactbench.bench \
    --model artifactnet --model spectttra --model clam \
    --clam-repo ~/dev/MoM-CLAM \
    --clam-ckpt ~/dev/MoM-CLAM/model_wts/best_model_triplet_loss_margin_0.2.pth \
    --manifest artifactbench_v1_manifest.json \
    --bench-origin test \
    --output results/3way
```

Output layout:

```
results/3way/
├── artifactnet/    # per-model artifacts
├── spectttra/
├── clam/
├── comparison.md   # side-by-side per-source table
├── roc_analysis.md
├── roc_curves.png
└── summary.json
```

## Using your own detector

Implement the `BenchModel` interface:

```python
from artifactbench.models import BenchModel
import numpy as np

class MyDetector(BenchModel):
    name = "MyDetector"
    params = 1_000_000
    input_sr = 44100
    paper_ref = "mine, 2026"

    def load(self, device="cuda"):
        self.model = ...  # your code

    def forward(self, audio_44k: np.ndarray) -> float:
        return float(self.model(audio_44k))  # P(AI) in [0, 1]
```

Register it and run:

```python
from artifactbench.models import MODEL_REGISTRY
MODEL_REGISTRY["mine"] = MyDetector
# then: python -m artifactbench.bench --model mine ...
```

## Evaluation protocol — what's actually measured

Beyond overall F1, ArtifactBench reports:

1. **Per-source TPR / FPR** — catches models that average-out per-source failures.
2. **Codec invariance** Δ across `wav ↔ mp3 ↔ aac ↔ opus` round-trips.
3. **Sanity FAIL count** — real FPR ≤ 5%, AI TPR ≥ 90% (Stable Audio: ≥ 60%),
   mean codec Δ ≤ 0.15, max Δ ≤ 0.35.
4. **Exposure-aware subsets** — `bench_origin=test` is historical split metadata,
   not proof of no training or model-selection overlap for every detector.
   Use model-specific exposure evidence and the exact corrected manifest.
5. **ROC / AUC** — plotted across all compared models.

Thresholds are in [`artifactbench/metrics/thresholds.py`](artifactbench/metrics/thresholds.py).

## License

The runner code is MIT licensed. **Model weights and datasets are subject to
their own licenses** — see [NOTICE.md](NOTICE.md) for a full breakdown.

## Citation

If you use ArtifactBench or the ArtifactNet baseline in your research, please cite:

```bibtex
@article{oh2026artifactnet,
  title        = {ArtifactNet: Detecting AI-Generated Music via Forensic Residual Physics},
  author       = {Oh, Heewon},
  journal      = {arXiv preprint arXiv:2604.16254},
  year         = {2026},
  eprint       = {2604.16254},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD},
  doi          = {10.48550/arXiv.2604.16254},
  url          = {https://arxiv.org/abs/2604.16254}
}
```

**Contact**: contact@intrect.io

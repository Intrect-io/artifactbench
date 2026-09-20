# ArtifactBench v2

ArtifactBench v2 is a lineage-aware, metadata-first evaluation suite for
AI-generated music detectors. The frozen primary protocol contains 828 tracks
(605 AI and 223 real), partitioned into calibration, validation, and sealed test
sets before final model comparison. The public release does not add a new audio
bundle.

[![paper](https://img.shields.io/badge/paper-arXiv_pending-b31b1b.svg)](#citation)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97-dataset-yellow)](https://huggingface.co/datasets/intrect/artifactbench)
[![model](https://img.shields.io/badge/%F0%9F%A4%97-model-yellow)](https://huggingface.co/intrect/artifactnet)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Why

Aggregate detector scores are not comparable when models silently evaluate
different files, tune thresholds on test data, or inherit upstream training
overlap. v2 freezes one ordered cohort, records decoding and inference failures,
selects thresholds on calibration only, and reports paired metrics on the common
successfully scored test-ID intersection.

## Install

```bash
git clone https://github.com/Intrect-io/artifactbench
cd artifactbench
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The project dependency pins the SONICS framework to commit
`9156ffad151f797c71556923c4a02fa01fa8fc91`. Install CLAM's feature-extractor
dependencies with `pip install -e .[clam]`.

System requirement: `ffmpeg` on PATH (required for codec invariance tests).

## Frozen protocol quickstart

The ArtifactNet ONNX build is auto-downloaded from HuggingFace:

```bash
# Obtain artifactbench_v2_primary_manifest.json and independently acquire the
# upstream audio identified by its retrieval fields and SHA-256 digests.
python scripts/bind_audio_root.py \
    --manifest artifactbench_v2_primary_manifest.json \
    --audio-root /path/to/acquired/audio \
    --output artifactbench_v2_runtime_manifest.json \
    --report artifactbench_v2_binding_report.json

# Run ArtifactNet on a small sample
python -m artifactbench.bench \
    --model artifactnet \
    --manifest artifactbench_v2_runtime_manifest.json \
    --split all --n-per-source 0 --n-codec-pair 0 \
    --output results/smoke
```

You'll get per-source summaries, raw track probabilities, structured inference
failures, and a run manifest containing the exact cohort and environment.

## Four-model comparison

```bash
# Clone the pinned CLAM source and obtain its checkpoint under the upstream terms.
git clone https://github.com/StarkVision-AI/MoM-CLAM ./MoM-CLAM
git -C ./MoM-CLAM checkout 74e3a3277e1dfe9ae9ed433b6e8c51d74e9e1d9b
# ... follow upstream README to obtain best_model_triplet_loss_margin_0.2.pth ...

# Four-model run
python -m artifactbench.bench \
    --model artifactnet --model spectttra --model deezer_ismir --model clam \
    --clam-repo ./MoM-CLAM \
    --clam-ckpt ./MoM-CLAM/model_wts/best_model_triplet_loss_margin_0.2.pth \
    --manifest artifactbench_v2_runtime_manifest.json \
    --split all --n-per-source 0 --n-codec-pair 0 --crop-policy center \
    --output results/four_model
```

Output layout:

```
results/four_model/
├── artifactnet/    # per-model artifacts
├── spectttra/
├── deezer_ismir/
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

## Evaluation protocol

The v2 publication protocol adds analysis scripts to the versioned release:

1. thresholds are selected on calibration only by maximizing TPR subject to
   FPR no greater than 5%;
2. validation is diagnostic only and cannot change the threshold;
3. the sealed test reports AUROC, AUPRC, F1, balanced accuracy, TPR/FPR,
   confusion matrices, per-source rates, and 2,000 lineage-bootstrap intervals;
4. inference failures remain outside classification metrics and are reported as
   coverage; and
5. paired comparisons use the identical successfully scored test-ID intersection.

Codec-pair evaluation remains available as a secondary experiment, but it is not
mixed into the frozen 579-track primary test.

## License

The runner code is MIT licensed. **Model weights and datasets are subject to
their own licenses** — see [NOTICE.md](NOTICE.md) for a full breakdown.

## Citation

The ArtifactBench v2 citation will be added when its arXiv identifier is issued.
If you use the ArtifactNet baseline, cite:

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

# ArtifactBench

> A public runner for evaluating AI-generated music detectors on the **ArtifactBench v1**
> benchmark, with baseline adapters for **ArtifactNet**, **CLAM**, and **SpecTTTra**.

[![paper](https://img.shields.io/badge/arXiv-2604.16254-b31b1b.svg)](https://arxiv.org/abs/2604.16254)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97-dataset-yellow)](https://huggingface.co/datasets/intrect/artifactbench)
[![model](https://img.shields.io/badge/%F0%9F%A4%97-model-yellow)](https://huggingface.co/intrect/artifactnet)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Why

Existing AI music detection benchmarks (SONICS: 5 generators; MoM: 6) don't
measure out-of-distribution generalization. Models reporting F1 > 0.97 on
SONICS collapse on diverse generators.

**ArtifactBench v1** evaluates 22 AI generators × 6 real source families
(6,200 tracks, 28 sources) under a 6-dimensional **sanity protocol** that
catches models that hide behind mean F1.

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

The ArtifactNet ONNX build is auto-downloaded from HuggingFace:

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
4. **`bench_origin=test`** subset (2,280 tracks) unseen by all compared models
   for leak-free evaluation.
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

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

## Try the maintained detector for free

Run ArtifactNet in your browser, no signup: **[demo.intrect.io](https://demo.intrect.io/)**. For the batch/API product and its limits: **[intrect.io/artifactnet](https://intrect.io/artifactnet/)**. Methodology, per-source results and the public comparison: the [ArtifactBench field report](https://intrect.io/research/artifact-observatory/ai-music-detector-benchmark-2026/).

## Why

Aggregate detector scores are not comparable when models silently evaluate
different files, tune thresholds on test data, or inherit upstream training
overlap. v2 freezes one ordered cohort, records decoding and inference failures,
selects thresholds on calibration only, and reports paired metrics on the common
successfully scored test-ID intersection.

The earlier v1.1 and v1.2 development materials remain available in
[`v1.1/`](v1.1/), [`PLAN_v1.2.md`](PLAN_v1.2.md),
[`RESULTS_8WAY.md`](RESULTS_8WAY.md), [`docs/`](docs/), and [`paper/`](paper/).
They are retained as historical evidence and are separate from the frozen v2
publication protocol documented below.

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

### Reproduce the frozen analysis

Run every adapter over the full bound manifest before analysis. The `--result`
arguments point to runner output directories; when two models were executed in
one run, the same directory is supplied for both model names.

```bash
python scripts/analyze_frozen_protocol.py \
  --manifest artifactbench_v2_runtime_manifest.json \
  --result artifactnet=results/artifactnet \
  --result spectttra=results/spectttra-deezer \
  --result deezer_ismir=results/spectttra-deezer \
  --result clam=results/clam \
  --output results/frozen-analysis \
  --bootstrap 2000

python scripts/verify_frozen_metrics.py \
  --manifest artifactbench_v2_runtime_manifest.json \
  --metrics results/frozen-analysis/frozen_protocol_metrics.json \
  --result artifactnet=results/artifactnet \
  --result spectttra=results/spectttra-deezer \
  --result deezer_ismir=results/spectttra-deezer \
  --result clam=results/clam \
  --output results/frozen-analysis/independent-verification.json

python scripts/render_frozen_artifacts.py \
  --metrics results/frozen-analysis/frozen_protocol_metrics.json \
  --output results/frozen-analysis/rendered
```

`analyze_frozen_protocol.py` selects thresholds from calibration rows only and
then evaluates validation and test without retuning. `verify_frozen_metrics.py`
independently repeats threshold selection and every paired point calculation.
`render_frozen_artifacts.py` consumes only the frozen metrics JSON, preventing
manual transcription of paper values.

### Build shareable metadata and result artifacts

```bash
python scripts/build_public_results.py \
  --manifest artifactbench_v2_runtime_manifest.json \
  --metrics results/frozen-analysis/frozen_protocol_metrics.json \
  --result artifactnet=results/artifactnet \
  --result spectttra=results/spectttra-deezer \
  --result deezer_ismir=results/spectttra-deezer \
  --result clam=results/clam \
  --runner-revision "$(git rev-parse HEAD)" \
  --output release/results

python scripts/audit_manifest.py \
  release/artifactbench_v2_primary_manifest.json \
  --output release/manifest_audit.json
```

The public-results builder remaps private track IDs, rejects private path and
identity fields, and retains raw probabilities and structured failures. A final
release review must still inspect free-form error text and upstream metadata.
`build_public_release.py` is a maintainer utility for producing the
path-free manifest from a verified private runtime manifest; its optional
`--fma-licenses` input is the private FMA evidence file containing
`artifactbench_track_id`, not the already-sanitized public license map.

ArtifactNet's non-finite chunk investigation is reproducible with
`audit_artifactnet_nonfinite.py` and
`build_artifactnet_finite_chunk_result.py`. The declared primary policy requires
at least four finite scores among seven fixed chunks; the strict any-non-finite
result remains a separate sensitivity analysis.

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

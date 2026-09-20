# Third-Party Licenses

This runner is licensed under the MIT License (see `LICENSE`). It integrates with
several third-party models, each under its own license. You must comply with the
respective license when downloading and running these models.

## Models

### ArtifactNet ONNX build
- **Repo**: https://huggingface.co/intrect/artifactnet
- **License**: CC BY-NC 4.0 (non-commercial research only)
- **Patent notice**: The ArtifactNet model card contains a pending-patent notice.
  The CC BY-NC license does not convey patent rights.
- **Weights redistribution**: this runner downloads the ONNX build via HF Hub at
  runtime; it does not redistribute the weights.

### SpecTTTra (SONICS)
- **Model weights**: https://huggingface.co/awsaf49/sonics-spectttra-alpha-120s (MIT)
- **Framework package**: https://github.com/awsaf49/sonics
- Pinned install: `pip install git+https://github.com/awsaf49/sonics.git@9156ffad151f797c71556923c4a02fa01fa8fc91`

### Deezer ISMIR 2025 research detector
- **Method and research code**: https://github.com/deezer/ismir25-ai-music-detector
- **License**: CC BY-NC 4.0; non-commercial research use only.
- **Patent notice**: the upstream repository includes its own patent notice.
- **ONNX artifact**: downloaded at runtime from
  `lofcz/ai-music-detector` at pinned revision
  `d2180598fed79e3f917e8050a00439982466e5c6`; not redistributed here.

### CLAM (MoM — Melody or Machine)
- **Repo**: https://github.com/StarkVision-AI/MoM-CLAM
- **License**: see upstream repo (no SPDX identifier declared at time of writing;
  assume "all rights reserved" unless the upstream clarifies).
- **Weights redistribution**: this runner does **not** redistribute CLAM weights.
  You must obtain them by following the instructions in the upstream repo.
- Pass `--clam-repo` and `--clam-ckpt` at runtime.

## Datasets

### ArtifactBench v2
- **Repo**: https://huggingface.co/datasets/intrect/artifactbench
- **Benchmark-authored metadata and results**: CC BY-NC 4.0
- **Release policy**: metadata only; v2 does not add a new audio bundle.
- **Upstream material**: acquire independently and comply with each source's
  license and terms. The benchmark license does not grant audio rights.

### MERT / Wav2Vec2 feature extractors (used by CLAM)
- `m-a-p/MERT-v1-95M` — see HF card for license terms.
- `m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres` — see HF card for license terms.
- Downloaded automatically by `transformers` at runtime.

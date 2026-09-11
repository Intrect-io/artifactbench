# Third-Party Licenses

This runner is licensed under the MIT License (see `LICENSE`). It integrates with
several third-party models, each under its own license. You must comply with the
respective license when downloading and running these models.

## Models

### ArtifactNet ONNX build
- **Repo**: https://huggingface.co/intrect/artifactnet
- **License**: CC BY-NC 4.0 (non-commercial research only)
- **Patent notice**: Methods are covered by pending patent applications (KR + PCT).
  The CC BY-NC license does not convey patent rights.
- **Weights redistribution**: this runner downloads the ONNX build via HF Hub at
  runtime; it does not redistribute the weights.

### SpecTTTra (SONICS)
- **Model weights**: https://huggingface.co/awsaf49/sonics-spectttra-alpha-120s (MIT)
- **Framework package**: https://github.com/awsaf49/sonics (install via pip)
- Install: `pip install git+https://github.com/awsaf49/sonics.git`

### CLAM (MoM — Melody or Machine)
- **Repo**: https://github.com/StarkVision-AI/MoM-CLAM
- **License**: see upstream repo (no SPDX identifier declared at time of writing;
  assume "all rights reserved" unless the upstream clarifies).
- **Weights redistribution**: this runner does **not** redistribute CLAM weights.
  You must obtain them by following the instructions in the upstream repo.
- Pass `--clam-repo` and `--clam-ckpt` at runtime.

## Datasets

### ArtifactBench v1
- **Repo**: https://huggingface.co/datasets/intrect/artifactbench
- **License**: CC BY-NC 4.0
- **AI audio**: distributed as Parquet with anonymized embedded audio bytes.
- **Real audio**: distributed as CSV metadata with YouTube IDs. You download audio
  yourself under fair-use research-use exemptions.

### MERT / Wav2Vec2 feature extractors (used by CLAM)
- `m-a-p/MERT-v1-95M` — see HF card for license terms.
- `m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres` — see HF card for license terms.
- Downloaded automatically by `transformers` at runtime.

# ArtifactNet raw — ArtifactBench 1.2 rc2 candidate

**Local packaging candidate. Not uploaded or approved for publication.**
Read `PARITY_REPORT.json` for the measured scope: source-smoke is not full
benchmark parity. Do not remove this notice merely because the archive builds.

This candidate exports the exact UNet and bigset CNN used by the September 5,
2026 ArtifactBench rc2 reference run. It changes the inference format, not the
training checkpoint or decision threshold. It produces one raw `P(AI)` score
and excludes the LGBM rescue model, codec TTA, AcoustID, and attribution labels.

## Interface

Input to the Python wrapper is already decoded 44.1 kHz mono floating-point
audio in [-1, 1]. The wrapper retains the first 60 seconds, pads short inputs
to four seconds, takes non-overlapping four-second chunks, and discards a final
chunk containing less than two seconds of audio. Other retained tails are padded.

The ONNX input is float32 `audio_chunks[chunks, 176400]`, with 1–15 chunks from
**one recording**. Its single output is float64 `p_ai[1]`. Normalization shares
one top-dB maximum across every chunk and residual/harmonic/percussive view,
followed by the seven-channel CNN and RMS-weighted aggregation. Do not split a
song into independent ONNX calls or combine chunks from unrelated songs.

The network and RMS are FP32; final probability aggregation is FP64. This is
an uncalibrated detector score, not proof of authorship or a calibrated
probability suitable for automatically penalizing musicians.

## Run without PyTorch

Python 3.12 and a CPU are sufficient for the included runtime:

```bash
python -m venv ../artifactnet-runtime-env
../artifactnet-runtime-env/bin/python -m pip install -r requirements.txt
../artifactnet-runtime-env/bin/python -B -m artifactbench.v12.verify_onnx_bundle --bundle .
../artifactnet-runtime-env/bin/python -B -m artifactbench.v12.artifactnet_onnx \
  --model artifactnet_raw.onnx --waveform-npy ../private/decoded-44100-mono.npy
```

Keep the environment and private inputs outside the bundle; `-B` prevents Python
from adding bytecode caches to its strict file inventory. The supplied waveform
is user-owned input, not included sample audio. Exact
benchmark comparisons must retain the benchmark's decoder/resampler identity;
changing those components is a separate replication condition. CPU inference
is tested. CUDA, CoreML, NNAPI, mobile providers, quantization, and training are
not validated by this candidate.

## Evidence and limits

`PARITY_REPORT.json` binds the original run, frozen public manifest, ONNX bytes,
and actual paired score observations. The acceptance rule was declared before
the comparison: absolute score difference at most `1e-3` and no raw-0.5 decision
flips on originally scored entries. Original inference failures remain explicit
non-comparable observations, even if the probability-only graph emits a score.

Source-smoke covers the hash-first entry per source plus duration boundaries.
Only a full report covering all 2,579 rc2 entries can establish full rc2 parity.
Neither scope proves accuracy outside this benchmark or independent public
availability of its audio. Native Suno and Udio cohorts have completely
confounded source/codec identities; results are not generator-version-only
causal effects. Training exposure is incompletely known, and provider-selected
demos are not representative population samples. The final benchmark paper
remains separate from this model-format candidate.

The ONNX file contains the weights as ordinary initializers. A single output
limits the interface; it does **not** make the weights opaque or prevent their
inspection. No raw `.pt` files, training corpus, restricted audio, private paths,
or internal diagnostic traces are included in this archive.

## Licensing and publication status

`RUNTIME_LICENSE.txt` applies only to the included ArtifactBench runtime code.
It does not relicense model weights or audio. The existing public ArtifactNet
model at [revision 7c9b753](https://huggingface.co/intrect/artifactnet/blob/7c9b753a9d006b48e4bfaf85bf0157e135f4aad4/README.md)
declares CC BY-NC 4.0. The owner has been asked whether the new ONNX release
should retain that license; this draft does not assume the answer or authorize
publication. Patent and other licensing assertions are not established by
ONNX conversion or benchmark execution.

## Related work

Heewon Oh, *ArtifactNet: Detecting AI-Generated Music via Forensic Residual
Physics*, [arXiv:2604.16254](https://arxiv.org/abs/2604.16254).
The new ArtifactBench 1.2 manuscript is a separate LaTeX draft and has not been
assigned a new arXiv identifier here.

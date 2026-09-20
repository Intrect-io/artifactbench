# ArtifactNet raw P(AI) ONNX candidate

Status: **local candidate, not a published model and not full rc2 parity yet**.
The running eight-model benchmark is unchanged. No checkpoint is retrained,
no decision threshold is tuned, and no earlier score or failed row is replaced.

**2026-09-05 16:05 KST: full-parity acceptance failure observed.** At the
205-comparison checkpoint, one recording exceeded the predeclared `1e-3`
absolute-score tolerance; no raw-0.5 decisions had flipped. Recording
`fe2d32060d3efaca5d838660` (`mom_riffusion`, 15 chunks) scored
`0.7292302923969539` in the original GPU run and `0.7281898997184076` in
ONNX CPU, an absolute difference of `0.0010403926785462758`. Its saved record
SHA-256 is `e9280757300cbb76c1c3657e080a94ea5c891a2f5bf928277a26b9e9b5a17d80`.
The model hash and all pinned source/input file hashes still match. This is
an observed end-to-end numerical discrepancy, not an established exporter
root cause. A same-input fresh CPU reference was not yet available at that
checkpoint; the subsequent diagnostic is reported below. The full run
continues unchanged to measure its remaining scope.
Keep the original tolerance, scores and candidate; do not promote the successful
source-smoke bundle to a full-parity release. Evidence is under
`out/v1.2_artifactnet_onnx_full_v2_260905/records/`.

## One-recording numerical decomposition

The outcome-selected tolerance failure was replayed using the exact frozen
audio, decoder, original factory and weights, on two CPU threads. The original
factory sets `GPU_BATCH_MAX = 1`; constructing the adapter directly leaves
the upstream default of eight. The diagnostic uses the factory and separately
controls the graph's batched execution. Neither source file is edited.

| Condition | Raw P(AI) |
| --- | ---: |
| Original saved GPU | 0.7292302923969539 |
| Original factory, fresh CPU, microbatch 1 | 0.7282716435500012 |
| Export graph in PyTorch, DFT convolutions | 0.7281895723724483 |
| Same graph, native PyTorch FFT STFT | 0.7282716435500010 |
| Same FFT graph, STFT/UNet microbatch 1 | 0.7282716435500010 |
| Actual v2 ONNX CPU replay | 0.7281898997184076 |

The saved-GPU/fresh-CPU difference is `0.0009586488469527`; v2 ONNX adds a
same-direction difference of `0.0000817438315936` relative to that CPU score.
Actual ONNX replay is identical to its earlier score. The FFT/microbatch-1
graph matches the CPU original's observed magnitude, UNet mask, seven-channel
CNN input and logits **bit for bit**, with only a `2.22e-16` aggregation
difference. Batched versus microbatch-1 FFT graphs have identical logits here,
despite tiny intermediate differences. Replacing only the FFT with the DFT
convolutions changes maximum magnitude by `0.0002288818359375`, mask by
`0.0011826157569885`, features by `0.33447265625`, and logits by
`0.0051114559173584`. This establishes an additional STFT-path contribution
for this recording; it does not locate the saved-GPU/fresh-CPU discrepancy
within a particular kernel or imply that these bounds hold on all recordings.

This is a targeted numerical diagnostic, not a representative accuracy sample
or a reason to relax the original acceptance rule. All input/source hashes
were rechecked after execution. Evidence:
`out/v1.2_artifactnet_onnx_diagnostic_260905/summary.json`, SHA-256
`68ce44b9f8f4b8d8c01ff08d5b577d64ce63b39e565261770058e484cc80e00b`.
The script is `artifactbench/v12/diagnose_artifactnet_onnx.py`; it saves
tensor identities, per-chunk logits, scores and paired differences, removes
all observation hooks even on failure, and leaves every original run intact.
The [PyTorch numerical-accuracy notes](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
describe the distinction between mathematical and bitwise equivalence across
devices and batching. The actual measurements here use installed PyTorch 2.13.0,
not a runtime upgrade to the documentation's 2.14 version.

### Standard ONNX STFT intervention: measured, not adopted

A separately named diagnostic copy replaces only the DFT-convolution prefix
with [standard ONNX STFT-17](https://onnx.ai/onnx/operators/onnx__STFT.html),
explicit reflect centering and the same FP32 Hann window. All 73 remaining
parent initializers are byte-identical; only the cosine/sine bases are removed.
The graph uses 1,015 standard nodes and is 16,728,276 bytes, SHA-256
`3d18241f221a8e25cb1f1372bbb41761a7671764c1255d051a81a4806ef8fcc7`.
It is a CPU-only probe, not a published model or replacement full-parity candidate.

On the same outcome-selected recording, actual ORT output is
`0.7288501596423501`: its GPU-reference error falls to `0.0003801327546038`,
but its CPU-reference error **increases** to `0.0005785160923489`. Both are
within the fixed 1e-3 rule on this recording. STFT maximum magnitude difference
from native PyTorch FFT is `3.0517578125e-5`, smaller than the DFT-convolution
error, but downstream score agreement is not monotonic in this bound. Partial
error cancellation relative to one reference must not select a release.
No threshold or acceptance tolerance changed, and the original full v2
validator continues. Before adopting a replacement, use the existing metadata-
selected source/boundary sample and then all originally scored recordings;
keep both CPU and saved-GPU comparisons and every failed candidate.

Evidence: `out/v1.2_artifactnet_onnx_stft_probe_260905/summary.json`, SHA-256
`3df8bf28dd0d8dc67c85853d15501c3e2a30743aacb9a089c6a176965e4ffdf1`.
`probe_onnx_stft.py` creates a new output directory, verifies the parent and
diagnostic hashes, checks learned-initializer preservation, and never writes
a release-export summary. No CUDA/mobile provider or throughput claim is made.

## Output and preprocessing contract

The graph uses the exact rc2 UNet and bigset CNN checkpoint hashes. It embeds
their weights and the DFT-convolution STFT basis in one ONNX file. Its single
output is `p_ai`, shape `[1]`, an uncalibrated raw sigmoid score aggregated by
chunk RMS. It does not contain LGBM rescue decisions, AcoustID, or codec TTA.
Single-output ONNX is an inference interface, not encryption of the weights.

The NumPy-only `song_chunks` wrapper accepts **already decoded, 44.1 kHz mono
floating-point audio clipped to [-1, 1]**. It keeps the first 60 seconds, pads
inputs shorter than four seconds, extracts non-overlapping four-second chunks,
discards a final chunk containing less than two seconds of actual audio, and
zero-pads any retained partial chunk. The graph input is float32
`audio_chunks[chunks, 176400]`, for 1–15 chunks from **one song**.

The chunk axis is not a batch of independent songs. In the evaluated pipeline,
`DifferentiableMel` uses one global maximum for its top-80-dB clamp, and
`extract_features` jointly transforms residual/harmonic/percussive magnitudes
across all song chunks. Splitting the song into independent calls or running
three separate Mel transforms changes this normalization. A regression test
specifically distinguishes these cases. The new graph preserves the joint
normalization, seven-channel features, CNN, and final RMS aggregation.

The network and RMS reductions are FP32; the final normalization and weighted
probability sum are FP64, matching the intended rc2 aggregation. The scalar
output is therefore float64. CPU ONNX Runtime uses two intra-op threads, one
inter-op thread, sequential execution, deterministic compute requested, and
graph optimization disabled. GPU/mobile providers are not yet validated.

Audio decoding/resampling is **outside** the graph. Exact benchmark reproduction
uses the unchanged `v12.audio.load_audio_float` path and its recorded runtime.
Replacing that resampler or decoder is a different replication condition; an
arbitrary FFmpeg conversion is not asserted to be sample-identical. The small
ONNX runtime itself needs NumPy and ONNX Runtime, not PyTorch or the ArtifactNet
training source.

## Build and verify

The measured export overlay uses torch 2.13.0 and ORT 1.29.0 from the existing
RTX environment, with ONNX 1.22.0 / ONNX Script 0.7.1 and their locked helper
packages installed into a separate export environment. It does not modify the
active evaluation environment. `requirements-onnx-export.lock` records the
overlay; `export_inputs.local.json` records the full visible package identity.
This is an explicitly shared read-only base, not a fully independent install.

The [current PyTorch exporter documentation](https://docs.pytorch.org/docs/2.14/onnx.html)
and the installed 2.13.0 signature both distinguish `dynamo=True` from the
retained TorchScript path. This implementation explicitly uses `dynamo=False`,
opset 18 and a dynamic chunk axis. The legacy-path deprecation is recorded;
spatial-padding tracer warnings concern fixed 1025-by-345 STFT dimensions,
not permission to assume dynamic chunk correctness without measurement.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv-onnx-export/bin/python -m artifactbench.v12.export_artifactnet_onnx \
  --reference-run out/v1.2_full_rc2_260905/artifactnet \
  --output out/my-artifactnet-onnx

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv-onnx-export/bin/python -m artifactbench.v12.validate_artifactnet_onnx \
  --model out/my-artifactnet-onnx/artifactnet_raw.onnx \
  --reference-run out/v1.2_full_rc2_260905/artifactnet \
  --release out/v1.2_frozen_rc2_260905 --source-smoke \
  --output out/my-artifactnet-onnx-smoke
```

Use new output directories; no in-place candidate or benchmark replacement.
Source-smoke chooses the hash-first entry per source plus shortest/longest
duration boundaries, without consulting scores: 34 rc2 recordings. It compares
actual chunk arrays and freshly executed CPU PyTorch scores as well as the saved
original GPU results. Full mode omits `--source-smoke` and requires every rc2
entry, comparing the ONNX output with the saved original scores.

The predeclared acceptance policy is maximum absolute score difference `1e-3`
and **zero raw-0.5 decision flips** on all originally scored entries. The known
original single-chunk inference failure remains an explicit non-comparable row;
a newly computable probability-only score cannot silently repair its benchmark
coverage. Partial smoke or engineering vectors cannot authorize a full-parity
or published-model claim.

## Measured export defect and correction

Candidate 1 exported and passed a seeded engineering vector, then failed on the
fourth real recording (`781bf0f587e4ff683c459105`): output
`1.0000000119162602`, outside [0, 1]. The original GPU score was
`0.9999999992197944`. The exported graph showed its FP64 RMS sum cast to FP32
before addition of the Python scalar epsilon and then cast back to FP64.
This was not the intended FP64 denominator. The old candidate, three completed
records, interruption note, and original source snapshots are preserved.

The correction keeps the sum one-dimensional and adds an explicitly FP64
tensor epsilon. The actual ONNX regression checks both Add-input types and
numeric probability bounds. It passes on the corrected formula and was also
executed against the preserved old aggregation, where it fails as intended.
No output clamp or relaxed tolerance was used. The same real recording now
produces `0.9999999936970877`, within the declared tolerance of its reference.

Current candidate:

- `out/v1.2_artifactnet_onnx_candidate_v2_260905/artifactnet_raw.onnx`
- 33,514,792 bytes; SHA-256
  `590d992b0839ac49900f6eed820cd2c9e68af76304af932dcff37e5bfd7b4621`.
- Single-file embedded weights; ONNX full checker and CPU load succeeded.
- DFT/STFT engineering-vector max absolute magnitude error `9.536743e-6`.
- Torch-graph/ORT engineering-vector score difference `5.109293e-8`.
- Corrected full-graph source-smoke completed: 34 attempted, 33 comparable,
  one original failure retained; maximum absolute error `0.0005089868448281409`
  versus the original GPU score and `0.0002830396000408353` versus fresh CPU
  reference scores. Zero raw-0.5 flips. Summary SHA-256
  `a60b6f459b6fe1827581d6c7f2923b55d3327ac19a9bbc1337de49ce37a9bfa8`.
- Full rc2 parity has started separately under
  `out/v1.2_artifactnet_onnx_full_v2_260905`; it is not complete.

The standalone runtime can be installed from `requirements-onnx-runtime.lock`:

```bash
python -m venv .venv-onnx-runtime
.venv-onnx-runtime/bin/python -m pip install -r requirements-onnx-runtime.lock
.venv-onnx-runtime/bin/python -m artifactbench.v12.artifactnet_onnx \
  --model out/my-artifactnet-onnx/artifactnet_raw.onnx \
  --waveform-npy private/decoded-44100-mono.npy
```

The `.npy` input is actual decoded audio and must remain private unless its
own source terms permit redistribution. The [ORT API documentation](https://onnxruntime.ai/docs/api/python/api_summary.html)
describes explicit provider selection and session options. This candidate
validates only the CPU provider; it makes no CUDA, mobile, CoreML, or NNAPI
compatibility claim. Publication and licensing review remain separate steps.

## Torch-free execution evidence

A fresh `.venv-onnx-runtime` was actually installed with only the five locked
runtime packages; `pip check` passed and Python reported that `torch` was not
available. The four minimal source files, v2 graph/metadata, and one real
private decoded waveform were copied into a new temporary directory outside
the checkout. From that directory, with `PYTHONPATH` and `PYTHONHOME` removed,
the runtime produced `0.9755876587251223`, exactly matching the corresponding
v2 smoke record. Both the absence of a Torch installation/import and the
temporary import root were checked during execution.

Evidence: `out/v1.2_artifactnet_onnx_runtime_portability_260905/summary.local.json`,
SHA-256 `7091529be0297d977889891b0df237cf93ef6e1754487448596da038e3706c49`.
This is a genuine one-recording runtime test, not all-recording parity or a
publicly redistributable audio fixture. The separately preserved initial
invocation used the wrong working directory and failed before inference;
only the corrected fresh-directory execution supports this result.

The export environment passed all **223 unit tests**, including actual ONNX
serialization of the FP64 aggregation regression. Both original evaluation
environments passed 220 tests, skipping three ONNX-only tests because ONNX
was deliberately not installed there. The pre-existing real-weight slow test
is separately measured and was deselected in these unit runs.

## Scope-labelled candidate bundle

### Fixed-source qualification of the standard-STFT alternative

`qualify_onnx_stft.py` tests the preserved standard-STFT probe without changing
the original v2 export or creating an `export_summary.json`. The selection is
the same 34 metadata/source/duration-boundary entries in the same order as the
earlier v2 smoke. It compares saved GPU scores and a fresh `build_pinned`
factory CPU reference separately; the latter sets `GPU_BATCH_MAX=1`, unlike
the older direct-adapter smoke's default eight. The original failed reference
remains noncomparable. Tolerance stays 0.001 with zero raw-0.5 flips permitted.

Actual input-only preflight passed: 34 exact selected audio hashes, matching
factory assets and 72 pinned sources/inputs. Evidence
`out/v1.2_artifactnet_stft_preflight_260905/preflight.local.json`, SHA-256
`66f7c18863b35e6d0e8771658c26c2959fbe2aab3cba60d52e234678fa02391c`;
qualifier source SHA-256
`8c469f05bff1185a66c1868f3b8c7e3d5ff06507a49fcbc975eb955c0af46ca9`.
The 34-entry inference was subsequently run in a frozen-source Bubblewrap
namespace with no competing heavy process. It completed 34/34 entries. Of the
33 entries with scored primary references, GPU comparison had zero failures and
zero raw-0.5 flips (maximum absolute error 0.0005109871); factory-CPU comparison
also had zero failures and zero flips (maximum 0.0002899710). One original
primary model-execution error was retained as non-comparable coverage. Evidence:
`out/v1.2_artifactnet_stft_qualification_frozen_260905/summary.json`, SHA-256
`5f156a773734f0565806543be0a4b34af51a46f6b7b3807d8cd0d56f31978105`.

This is still a 34-entry qualification, not full scored-reference parity. A
full run must be scheduled alone under OOM-safe service limits; do not run it
concurrently with another memory-heavy CPU launch:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
nice -n 10 .venv-onnx-export/bin/python -m artifactbench.v12.qualify_onnx_stft \
  --probe out/v1.2_artifactnet_onnx_stft_probe_260905 \
  --parent out/v1.2_artifactnet_onnx_candidate_v2_260905 \
  --reference-run out/v1.2_full_rc2_260905/artifactnet \
  --release out/v1.2_frozen_rc2_260905 \
  --snapshots out/v1.2_baseline_assets_v2_260905/snapshots.json \
  --output out/my-stft-source-qualification
```

An earlier attempt against the live tree was correctly rejected because two
pinned source files had changed; the successful run used the complete frozen
source directory. No hashes or the 0.001 tolerance were relaxed.

`package_artifactnet_onnx` requires a completed successful parity summary,
unchanged original scores, the exact score-blind recording selection, and runtime
source hashes matching the tested implementation. It inspects the actual ONNX
graph: 1,033 standard-domain nodes, 75 embedded initializers, one input and one
output, no external tensor files, and no machine-local paths in the bytes.
The public comparison report omits private inputs and diagnostic tracebacks.

```bash
.venv-onnx-export/bin/python -m artifactbench.v12.package_artifactnet_onnx \
  --model out/v1.2_artifactnet_onnx_candidate_v2_260905/artifactnet_raw.onnx \
  --parity out/v1.2_artifactnet_onnx_smoke_v2_260905 \
  --release out/v1.2_frozen_rc2_260905 \
  --reference-run out/v1.2_full_rc2_260905/artifactnet \
  --output out/my-onnx-candidate-bundle
```

The actual `artifactnet-raw-SOURCE-SMOKE-CANDIDATE.zip` is 29,934,792 bytes,
SHA-256 `95722b47f554c540a7541616ea42ee6fd2cdb2cb9eefa98e7e9ebbee7350297a`,
under `out/v1.2_artifactnet_onnx_bundle_smoke_260905`. An independent repeat
is byte-identical. It includes 11 payload files and a hash inventory; it does
not include audio or PyTorch checkpoints. The model card distinguishes runtime
MIT from the existing model's CC BY-NC 4.0 declaration; the owner's new-weight
license decision and publication remain pending. Source-smoke is never renamed
to full parity merely because the ZIP builds.

The actual ZIP was then extracted into a new directory outside the checkout.
Its verifier passed with site packages disabled (`python -B -S`), and the
Torch-free environment executed one real private waveform from the extracted
runtime. The result was exactly `0.9755876587251223`, matching the source-smoke
candidate record. Post-inference inventory verification also passed. Evidence:
`out/v1.2_artifactnet_onnx_bundle_portability_260905/summary.local.json`, SHA-256
`5784a9933b6860b576371be9b9082a28b64c32620be869dd28cbb5406a3a839c`.
This reuses the separately installed runtime environment; it is not another
fresh dependency installation or full-dataset inference replication.

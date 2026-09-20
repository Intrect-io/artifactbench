# DeepFense input-path audit: final resampling is not neutral

Measured 2026-09-05. The completed original DeepFense run is preserved under
`out/v1.2_full_rc2_260905/deepfense`, with an `INTERPRETATION_HOLD.md` note.
Its 797/820 legacy false positives describe the frozen adapter condition, not
intrinsic framework/checkpoint quality. No original prediction, class label,
threshold, checkpoint or frozen benchmark input was overwritten.

## Selection and controls

Before reading reference scores, the diagnostic saves the first entry in
`SHA256(260905:<id>)` order from each of six legacy real-source cells, one from
pooled legacy AI entries, and one from each contemporary native source. Demos
are excluded. These nine entries are not a new population performance estimate.
The first direct-path diagnostic and follow-up control have byte-identical
`selection.json`, SHA-256
`32517b16669fc76b4eaee3e7cf543b457be3efe8c87e74b41185d2e0d06cdb68`.

Reference row hashes, selected audio bytes, assets, package versions and adapter
code are verified. Both checks use the same environment, two CPU threads,
no gradients and empty `CUDA_VISIBLE_DEVICES`; CUDA remains uninitialized.

| Condition | What changes |
| --- | --- |
| Original GPU versus fresh CPU adapter | Device only; checkpoint and input pipeline retained. |
| Captured model-input replay | Nothing: exact 64,000-sample float32 tensor on the same CPU model. A forward hook observes, not replaces, the real adapter input. |
| Same common input, librosa final resampling | Only final 44.1-to-16-kHz resampling changes from TorchAudio's default to librosa `soxr_hq`. Decoder, float32 mono waveform, clipping, prefix/repeat padding, checkpoint and labels stay fixed. |
| Direct upstream input | Actual installed DeepFense SoundFile/librosa loader and validation padding. Original-rate handling and precision also differ; no unsupported-format fallback is added. |

EAT normalizes Fbank features internally. Both model calls receive raw waveform
and use class-0 softmax from the same two-class logits, not the framework's
single-score sigmoid or inverted classes.

## Actual result

Fresh CPU P(AI), rounded below. Original GPU scores differ by at most
`1.1920928955078125e-7`, with no threshold flips.

| Selected source | Native Hz | Original adapter | Same common input, librosa final stage | Direct upstream |
| --- | ---: | ---: | ---: | ---: |
| Legacy AI: sonics_udio-30s | 16,000 | 0.000000251 | 0.000000413 | 0.000000732 |
| Real: fma_hardneg | 44,100 | 0.996226 | 0.000025478 | 0.000025478 |
| Real: mom_extra_real | 44,100 | 0.998937 | 0.000017831 | 0.000017831 |
| Real: mom_real | 44,100 | 0.996376 | 0.000018359 | 0.000018359 |
| Real: mom_real_wav | 44,100 | 0.998379 | 0.000007789 | 0.000007789 |
| Real: sonics_real | 48,000 | 0.998973 | 0.000000357 | 0.000000237 |
| Real: youtube_hardneg | 44,100 | 0.990599 | 0.000025313 | 0.000025313 |
| Native Suno v5.5 | 48,000 | 0.998955 | 0.000012414 | AAC input error |
| Native Udio 2026 | 48,000 | 0.999021 | 0.000004196 | 0.000004555 |

Captured-input replay is **bit-exact in score on all nine cases**. Changing only
the final resampler flips **8/9** raw-0.5 decisions, maximum score difference
`0.9990168569770503`. All six real examples change to real, but **both native AI
examples also change to real**: this is not uniformly improved accuracy.
Direct upstream input flips 7/8 comparable cases. The selected Suno AAC file
raises `LibsndfileError`, retained without a probability or replacement.

The final-resampler intervention is sufficient to reverse these eight decisions.
It does not identify the model's learned spectral cue, establish a corrected
full-cohort FPR/TPR, or prove full equivalence to the authors' original pipeline.
The common 44.1-kHz intermediate stage is still present for other source rates;
replacing its final filter is not necessarily direct native-rate preprocessing.
These nine cases cannot select a best resampler.

## Evidence and reproduction

- Initial diagnostic: `out/v1.2_deepfense_inputpaths_260905/summary.json`, SHA-256
  `a139f35ad8d918ed8a930ef60ec3222a375d1fdbd32d3e5497ca7cd33d597531`.
  Its pre-control script version remains in its source snapshots.
- Same-input controls: `out/v1.2_deepfense_resampler_controls_260905/summary.json`,
  SHA-256 `bd9c7c30b5d318f1b04f26463c306a993fb2945fd0c61cfffbe0c38086bd5d38`.
  Identity SHA-256
  `396ce0e1c0ef33a5f346b5bbadac661c5cdf4abf53e3efe4b1774ace7cb779e4`.
  Records retain waveform hashes, logits, errors and comparison arithmetic;
  local artifacts may contain private paths.

From the repository root, using a new output directory:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
HF_MODULES_CACHE="$PWD/.tools/huggingface/diagnostic-modules" \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 \
.venv-eval/bin/python -m artifactbench.v12.diagnose_deepfense_inputs \
  --release out/v1.2_frozen_rc2_260905 \
  --snapshots out/v1.2_baseline_assets_v2_260905/snapshots.json \
  --reference-run out/v1.2_full_rc2_260905/deepfense \
  --output out/my-deepfense-input-audit
```

This uses pinned local assets and a separate module cache, without downloading
weights or altering the main queue. API behavior was checked against official
[librosa 0.11 resampling](https://librosa.org/doc/0.11.0/generated/librosa.resample.html),
[TorchAudio 2.8 resampling](https://docs.pytorch.org/audio/2.8.0/generated/torchaudio.functional.resample.html),
and [PyTorch forward hooks](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook).

## Required follow-up

### Pinned reference-consistent condition: deepfense-native-soxr/2

Version 1 was actually executed on the fixed 32-source smoke selection:
31 scores and one `input_execution_error`. The failed Lyria demo
`88b1f8e36b2cf97959aa081c` is WebM/Opus, 48 kHz stereo (actual ffprobe), outside
version 1's AAC-only fallback. Its failure, all scores, identity and source
snapshots remain in `out/v1.2_deepfense_native_smoke_260905`. Version 2 extends
the fallback allowlist to AAC and Opus, chosen from the measured container/
codec gap, not a detector score or label. It requires fresh output; existing
version-1 public rows remain supported without accepting Opus under version 1.

The separate condition is now implemented in `deepfense_native.py` and
`run_deepfense_native.py`. The original adapter/run is not changed. On local
files supported by SoundFile, decoding uses float64 and the original sample
rate, followed by float64 channel mean, direct librosa `soxr_hq` resampling
to 16 kHz (`fix=True`, `scale=False`), the actual upstream validation
`pad_combined` function (first 64,000 samples/repeat, no random crop), and
only then float32 casting. There is no common 44.1-kHz stage and no extra
clipping. EAT owns its original feature normalization; the same two-class
checkpoint head supplies class-0 softmax. No weight or threshold is selected
using these diagnostic recordings.

The declared unsupported-codec extension is **AAC or Opus only**. After an actual
SoundFile `LibsndfileError`, ffprobe must identify AAC or Opus. FFmpeg decodes its first
audio stream to float64 PCM WAV without `-ar`, `-ac`, or an audio filter; the
decoded WAV sample rate and channels must equal the probed stream. Mono and
resampling then use the same float64/librosa path as above. This extends the
benchmark's supported inputs; it does not pretend that upstream SoundFile
can decode the unsupported container/codec. No other decoder fallback is silently added. Failed inputs
remain `input_execution_error` rows without probabilities, separate from
model execution errors. The original full release membership stays fixed.

Each row records the decoder route, native rate/channels/frame count, resampled
frame count, final input hash, RMS and peak. Published predictions retain these
safe fields but exclude audio arrays, paths and exception text. The identity
contains the complete input contract and the installed DeepFense source hashes.
The code snapshot and final hash checks bind the new condition; model assets
and environment are compared with the original run before the additional
source/protocol metadata is attached. Device is explicit and may differ from
the original CUDA run. New runs require a fresh output directory.

Actual unit checks cover 16/44.1/48/96-kHz, mono/stereo, short/long inputs and
values above unity, with bit-exact prepared arrays against upstream load/pad.
Actual stereo AAC and WebM/Opus encode/decode tests compare independent raw
float64 FFmpeg output while retaining 48 kHz and two channels. These are processing fixtures,
not detector accuracy estimates.

The actual version-2 CPU smoke has now scored all 32 fixed sources: 30 native
SoundFile, one AAC fallback and one Opus fallback. Its summary SHA-256 is
`dd74659741879c28a2bfa954f8b8b39a1b42164c41d4ad17eae941086e5f99cb`, identity
`45ca686c7fd0924575fb6aadc3b44ed0c25d313a0a476acb7add7eae07b0b7d5`, under
`out/v1.2_deepfense_native_smoke_v2_260905`. Version-1 summary SHA-256
`2ae06b910552884768f813786e55e594423f2bc7dd1b7fd5b516b6cc1fe1d9d8` is preserved.
All 31 previously scored prepared waveforms, logits and probabilities match
bit-for-bit across the two versions; all eight comparable direct-upstream
waveform hashes and probabilities match the earlier diagnostic. Both runs'
116 source snapshots (including 104 installed DeepFense Python files) were
verified against their identities. Both public identity projections and all
64 projected rows passed JSON roundtrip and privacy checks. This checks the
new projection path, not an eight-model public-results ZIP.

The fixed version-2 full evaluation completed under
`out/v1.2_deepfense_native_full_v2_260905`, CPU/two threads, while the original
FST used the GPU. Mean measured version-2 smoke inference was 0.809 seconds
per entry; this is a source-smoke timing under concurrent work, not a dedicated
throughput benchmark or a CUDA timing claim. All 2,579 entries were attempted:
2,578 scored and one input failure. After reboot, the saved identity, all record
hashes and complete manifest membership were revalidated. Full summary SHA-256:
`84e4d5c8ee552bf0b8f52d6333192132a121e66a589efcb7f481f3c254032060`.
The 2,000-replicate checkpoint report is under
`out/v1.2_checkpoint_deepfense_native_v2_rc2_260905`, checkpoint SHA-256
`adeb644d69819c8e9e3343f225389392690025103828b814bb60c07dc3419cca`.
At raw 0.5, legacy TP/FN/FP/TN are 129/1215/18/801 (one unscored real input):
TPR 9.60%, FPR 2.20%, F1 0.1730, AUROC 0.36846. Native Suno is 0/200 and
Udio 6/200. Lower false positives therefore do not establish improved accuracy;
the corrected input condition also sharply reduces AI detections. Neither the
poor original adapter results nor these results justify a general claim about
the architecture outside the measured checkpoint and input condition.
Targeted native/public tests: 31 passed. Full non-slow suites: 285 passed/four
reported skips in the original evaluation environment, 287/four in the ONNX
export environment and 281/eight in RTX_ENV. DeepFense upstream fixtures skip
when that package is absent; ONNX-only checks skip outside its export environment.
The unchanged slow real-model test remains separately measured.

During the full run, the FMA recording `37e60a38e8b30b949551f7e4` raised an
input error (340 saved rows at the inspection: 339 scored, one input failure).
The actual upstream `load_audio` independently reproduces SoundFile's
`LibsndfileError: Unspecified internal error.` on the same frozen bytes.
ffprobe identifies 44.1-kHz stereo MP3, outside the AAC/Opus fallback allowlist.
A separate, non-scoring FFmpeg diagnostic decodes finite float64 stereo PCM
at 44.1 kHz (6,823,296 frames), with empty stderr. This is a decoder-dependent
failure, not evidence that the benchmark bytes are undecodable by every loader.
The fixed /2 policy was not broadened: preserve the `input_execution_error`
without a replacement probability. Saved failure SHA-256
`aba4d6f7085d5ce6bf29c4433d2711730fa178a4bc88ffedba9972828cffb054`.
All-attempt coverage and any paired complete-case analysis must retain this gap.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
HF_MODULES_CACHE="$PWD/.tools/huggingface/native-modules" \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 \
.venv-eval/bin/python -m artifactbench.v12.run_deepfense_native \
  --release out/v1.2_frozen_rc2_260905 \
  --snapshots out/v1.2_baseline_assets_v2_260905/snapshots.json \
  --reference-run out/v1.2_full_rc2_260905/deepfense \
  --output out/my-deepfense-native-smoke --device cpu --smoke-per-source 1
```

The full run omits `--smoke-per-source`. Device/compute conditions must be
reported as measured; do not claim CUDA timing for a CPU execution. Transport
alignment is implemented via `run_transports --deepfense-native`, with an
actual 14-view path smoke, but the full 406-view execution/report remain
pending. See [commands and evidence](REPRODUCING_v1.2.md#native-deepfense-transport-condition).
Controlled caches begin at the
declared common 44.1-kHz master, whereas native pairs must use the same native
file path as the corrected primary run. Do not pass a 16-kHz prepared input
through the old adapter's 44.1-to-16-kHz resampler again.

Preserve the original run as an adapter-condition audit. Define and pin a
separately named reference-consistent input path, including unsupported-codec
handling, before full re-evaluation. Evaluate all fixed entries with coverage,
not only favorable real examples, and retain both conditions. Reconcile the
new condition with controlled-transport and reporting tools. Check other
adapters' upstream preprocessing contracts rather than blanket-replacing their
resamplers based on this model. The unqualified final eight-checkpoint
comparison remains incomplete until this issue and full re-evaluation are resolved.

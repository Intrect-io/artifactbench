# Reproducing the rc2 evaluation

This describes the implemented workflow. Full eight-model and transport runs
are still in progress; do not read these commands as a completed reproduction
claim. The metadata notebook has actually passed in a fresh Python 3.12
environment; the eight adapters have separately passed 32 real-input smoke
attempts each. Final results must come from completed rc2 runs.

## Inputs and environments

Use `out/v1.2_frozen_rc2_260905`, public manifest SHA-256
`feab7c4c3d037919fd784dc40e46199470088abde532c5181e34573f20dceefd`.
Do not substitute rc1, which was invalidated by the Udio file-identity audit.
The public manifest is not an audio download: reconstruct available source
files under their own terms and create a separate local path map with exactly
the public IDs and byte hashes. The runner refuses mismatching inputs. No
audio redistribution permission follows from this repository's MIT license.

The [local audio binding command](LOCAL_AUDIO_BINDING_v1.2.md) turns a private
ID-to-path map into a separate runnable release directory after checking every
encoded-file hash. Missing or mismatching files produce an audit and nonzero exit,
not incomplete runnable manifests. Use the resulting directory as `--release`
with new inference outputs; do not edit the frozen release or resume an old run
with a different private manifest hash.

The measured Linux run uses two interpreters: ArtifactNet's existing RTX_ENV
(Python 3.12, torch 2.13 CUDA 13.0) and the isolated `.venv-eval` for the seven
external checkpoints (Python 3.12, torch 2.8 CUDA 12.8). The latter's packages
are pinned in `requirements-v1.2.lock`. Exact package, GPU, decoder, source,
checkpoint, and RNG identities are saved for each run. Different hardware or
runtime is a new replication condition, not automatically a bitwise match.
Do not install or upgrade dependencies inside a running evaluation environment.

`ffmpeg`, `ffprobe`, and `fpcalc` are needed for data construction; the observed
FFmpeg version is 6.1.1. No competing GPU inference is started during the
sequential benchmark on the 12 GiB RTX 3060.

## OOM and reboot safety on the author host

The 2026-09-05 previous-boot kernel journal records OOM at 17:54:22;
the next boot began at 17:55:06. Do not infer a completed run from files merely
existing: FST had seven zero-byte records, ONNX one, and the new native transport
run nineteen. Completed primary runs passed their full record-hash checks.
Preserve incomplete evidence before retrying. Do not delete successful records
or change frozen manifests to work around missing storage.

Before starting any heavy job, inspect `free -h`, `nvidia-smi`, other inference
processes and required mounts. Run only one heavy benchmark at a time. Host RAM
and VRAM are separate budgets; a cgroup RAM limit does not constrain CUDA memory
or protect against unrelated processes consuming the rest of the machine.

The FST recovery used a systemd user service with `MemoryHigh=6G`,
`MemoryMax=8G`, `MemorySwapMax=0`, `OOMPolicy=stop`, and `Restart=no`.
A separate probe read back the actual cgroup-v2 values (6442450944,
8589934592, and 0 bytes); do not rely only on accepted command-line options.
These are measured FST recovery settings, not universal limits for other models.
Retain the original interpreter, working directory, arguments and environment
so the runner's exact saved identity comparison remains effective.

Inspect the service's `ActiveState`, `SubState`, `MainPID`, `MemoryCurrent`,
`MemoryPeak` and journal after launch. A running unit's `Result=success` is not
a completed benchmark. Completion requires the runner summary and record hashes.
Do not automatically restart after an OOM or an input error.

The first limited FST recovery scored one additional entry (1116 total) and
then stopped with `FileNotFoundError`, at a measured 3.6GiB memory peak: the
Archive exFAT volume was unmounted after reboot. A read-only udisks mount was
denied by polkit. Restore the authorized original mount, verify frozen inputs,
then resume; do not treat this infrastructure failure as a detector prediction.

## Model assets and location overrides

`python -m artifactbench.v12.fetch_baselines --cache <cache> --output <assets>`
fetches nine explicitly pinned public HF revisions; it never substitutes
`main`. Its `snapshots.json` binds actual cache locations and content hashes.
Use the resulting file in `--snapshots`. A copied snapshots file with stale
absolute paths is not portable by itself; acquire the same revisions into the
new machine's cache and retain its new location manifest.

Three source checkouts and the BeatThis checkpoint are separate assets. Their
locations can be set with these environment variables before model creation:

| Variable | Required content |
|---|---|
| `ARTIFACTBENCH_ARTIFACTNET_REPO` | Matching ArtifactNet checkout, `src`, and checkpoint files selected by its pinned `model_config` |
| `ARTIFACTBENCH_CLAM_REPO` | MoM-CLAM checkout and `model_wts/best_model_triplet_loss_margin_0.2.pth` |
| `ARTIFACTBENCH_FST_REPO` | FST checkout and `checkpoints/{backbone_stage1,classifier_stage2}.ckpt` |
| `ARTIFACTBENCH_BEAT_CHECKPOINT` | BeatThis `final0.ckpt` |

Overrides change paths only: the factory still hashes all used files and source.
The [external-asset acquisition and verification workflow](EXTERNAL_ASSETS_v1.2.md)
provides pinned CLAM/FST source revisions, official checkpoint locations, and a
stdlib-only byte/code check against the public prediction identity. Run that
check before loading external code or deserializing a checkpoint; downloading
an HTML confirmation page is not successful model acquisition.
The default asset dictionaries for all eight models were verified unchanged
when this location support was introduced. Match checkpoint/source hashes to
the recorded identity; a successful load of another checkpoint is not a
reproduction. In particular, the current local ArtifactNet checkpoint stack
must not be silently replaced by the older public ONNX build. Availability of
every author-local ArtifactNet asset on a public service has not been established.
An unauthenticated metadata audit on 2026-09-05 checked the declared public
`intrect/artifactnet` repository at revision
`7c9b753a9d006b48e4bfaf85bf0157e135f4aad4`. It contained only two v9.4 ONNX
assets plus README/attributes. None matched the SHA-256 of `cnn_bigset_best.pt`,
`unet_codec4_best.pt`, or `lgbm_bigset_28dim.txt`; neither small non-LFS file had
the same byte size as a required checkpoint. Thus these exact three files are
absent from **that repository revision**, not proven absent from every possible
host. Evidence: `out/v1.2_artifactnet_public_asset_audit_260905/audit.json`.
Acquiring this evaluated stack remains an explicit public-inference replication
gap. No weights were uploaded, substituted or changed by this audit.

The three-file audit does **not** imply that all three original checkpoints must
be distributed to reproduce the raw detector. A newly exported UNet+CNN ONNX
with identical preprocessing and first-60-second, 4-second sliding-chunk RMS
aggregation can reproduce raw `P(AI)` without the LGBM rescue model. The
existing single-output exporter uses an older CNN, and the legacy ONNX adapter
uses seven chunks and a median, so neither currently proves rc2 parity. New
export is now implemented as a [local candidate](ARTIFACTNET_ONNX_v1.2.md), and
its 34-recording source/boundary check passed on 33 comparable original scores
with one original failure retained. Full 2,579-entry parity is running and
publication remains pending. Reproducing the additional rescue-stack results is a separate
requirement; changing format is not an accuracy improvement.

## Actual inference and report commands

From the repository root, using the interpreter appropriate to the model:

```bash
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_MODULES_CACHE="$PWD/.tools/huggingface/modules"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m artifactbench.v12.run --model artifactnet \
  --snapshots out/v1.2_baseline_assets_v2_260905/snapshots.json \
  --release out/v1.2_frozen_rc2_260905 --output out/my-rc2/artifactnet
```

Repeat sequentially for `spectttra`, `spectttra_beta5s`, `ast_60s`, `deepfense`,
`deezer_ismir`, `clam`, and `fst`. `--smoke-per-source 1` is a wiring check,
never a leaderboard. A resumed output directory requires exactly matching
identity; use a new directory for changed code, assets, or data. OOM and device
failures stop execution. Intrinsic per-file errors remain recorded, without
invented probabilities or silent denominator changes.

Only after all eight full runs and the corrected native DeepFense primary have
completion summaries (see the native input-condition instructions below):

```bash
python -m artifactbench.v12.report --release out/v1.2_frozen_rc2_260905 \
  --runs out/my-rc2 \
  --run-override deepfense=out/v1.2_deepfense_native_full_v2_260905 \
  --output out/my-rc2-statistics
```

## Paired transport evaluation

`prepare_transports` prepares a score-blind 50-recording legacy sample and eight
variants per recording. `verify_transports` checks hashes, exact float identity,
and native waveform alignment. The measured rc2 prepared directory is
`out/v1.2_transports_rc2_260905`: 400 controlled + 6 native waveforms. Its
unchanged legacy arrays were explicitly rebound from verified earlier caches,
not re-encoded. It contains only the three verified native MP3/AAC pairs.

After the corresponding primary model run is complete, with the **same**
interpreter, assets, source, environment, and device:

```bash
python -m artifactbench.v12.run_transports --model artifactnet \
  --snapshots out/v1.2_baseline_assets_v2_260905/snapshots.json \
  --release out/v1.2_frozen_rc2_260905 \
  --prepared out/v1.2_transports_rc2_260905 \
  --reference-run out/my-rc2/artifactnet --output out/my-transports/artifactnet
python -m artifactbench.v12.transport_statistics \
  --prepared out/v1.2_transports_rc2_260905 \
  --run out/my-transports/artifactnet --output out/my-transport-statistics/artifactnet
```

Do not run this alongside the main GPU queue. Resetting the same per-recording
seed for every variant controls random crop selection. Float identity versus
the primary run is a repeatability diagnostic, not a codec effect. Pair failure
coverage is reported. Three native pairs support only a very small descriptive
comparison; PCM/FLAC input storage does not establish never-lossy source history.
The statistics preserve total shifts from float identity and separately emit
`controlled_contrasts`: PCM16-only change, each codec's increment from its
matching float/PCM16 baseline, staging change with codec fixed, and the signed
four-view interaction. See [the exact formulas](STATISTICS_v1.2.md#matched-staging-transport-contrasts).
Do not call a PCM16-plus-codec total shift a codec-only increment.

### Native DeepFense transport condition

After the separately versioned native primary evaluation is complete, use
`--deepfense-native` and its CPU reference directory. This is an explicit
alternative input condition, not a change to other detectors or old results:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
HF_MODULES_CACHE="$PWD/.tools/huggingface/native-modules" \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 \
.venv-eval/bin/python -m artifactbench.v12.run_transports \
  --model deepfense --device cpu --deepfense-native \
  --snapshots out/v1.2_baseline_assets_v2_260905/snapshots.json \
  --release out/v1.2_frozen_rc2_260905 \
  --prepared out/v1.2_transports_rc2_260905 \
  --reference-run out/v1.2_deepfense_native_full_v2_260905 \
  --output out/my-transports/deepfense-native
```

The same `transport_statistics` command processes the completed output.
The runner reconstructs current package/source/input identities and requires
the same native condition, weights, environment and device as the completed
primary. It does not accept a still-running primary as completed evidence.
Controlled views use the hash-verified 44.1-kHz float32 master, promoted to
float64 before the pinned native preparation. This does not restore precision
already lost in the common master. The three native pairs instead use their
actual encoded files, joined exactly to both frozen public and local pair
metadata. Their old 44.1-kHz native caches are not model inputs. Each native
file's encoded hash is checked before and after inference; model-input hashes
and explicit no-score failures are retained. No input re-enters the old adapter.

The controlled float-identity versus native primary comparison is an
**input-path difference**, not a repeatability measurement. The report puts
it in `common_master_input_path_difference`, separate from native-file
`primary_repeatability`. Neither is itself a codec-only effect.

Actual preflight: 400 controlled views plus six native views bind to 53 unique
primary entries; runtime matches the native primary identity. All six native
files decoded, and the three MP3 primaries matched actual upstream prepared
arrays exactly. A fixed path smoke (first selected controlled recording's
eight views plus all three native pairs) scored 14/14 on CPU. Evidence:
`out/v1.2_deepfense_transport_smoke_260905`, summary SHA-256
`98c1f47be8aabed58258f08353cd878f2ebbdee06c18e5cffcf5ce72ce3c343d`.
This does not establish full 406-view execution or transport population effects.

## Independent metadata bundle

The metadata-only bundle contains an explicit allowlist of 20 payload files,
plus their hash inventory. The public JSON is checked structurally; raw pages,
private paths, lyrics, prompts, audio, weights, and predictions are excluded.
The ZIP has fixed member timestamps and permissions for deterministic packaging.

```bash
python -m artifactbench.v12.package_metadata \
  --release out/v1.2_frozen_rc2_260905 --output out/my-metadata-bundle
python -m artifactbench.v12.check_metadata_portability \
  --archive out/my-metadata-bundle/artifactbench-1.2-rc2-metadata.zip \
  --notebook-python .venv-notebook/bin/python --output out/my-portability-check
```

The portability check actually extracts into a fresh temporary directory,
removes `PYTHONPATH`/`PYTHONHOME`, and runs the verifier with site imports
disabled. It then executes the six-cell notebook from that extracted source
root, checking the actual kernel executable and environment prefix. It reuses
the already isolated notebook dependency environment; this is not a second
fresh dependency-installation claim. Temporary inputs and execution outputs
are preserved for inspection. No upload is performed.

See the dataset card for measured metadata execution, and `STATISTICS_v1.2.md`
for the 2,000-replicate clustered/stratified bootstrap. A final results notebook,
full-inference reproducibility package, and arXiv source archive remain release
gates. Passing the metadata bundle cannot satisfy these remaining checks.

The public-prediction export and saved-score notebook are described in
[RESULTS_REPRODUCING_v1.2.md](RESULTS_REPRODUCING_v1.2.md). Their full-result
gate is still pending; the historical smoke path has actually passed.

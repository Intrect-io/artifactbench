# External baseline assets: acquisition and byte verification

This is an asset-preparation workflow for the seven external checkpoints, not
an inference or benchmark-completion claim. ArtifactNet's raw ONNX candidate
has a [separate workflow](ARTIFACTNET_ONNX_v1.2.md). Checkpoint availability does
not transfer ownership or change source-specific licenses. No third-party
checkpoint is redistributed by the instructions or the verifier.

## Hugging Face dependencies

Use `artifactbench.v12.fetch_baselines` with a new cache/output location as
described in [REPRODUCING_v1.2.md](REPRODUCING_v1.2.md). It requests nine pinned
repository revisions. Preserve the resulting `snapshots.json`: it identifies
your cache locations, not the author's paths. Do not rewrite a revision to
`main` or replace a missing model with a similarly named checkpoint.

The prediction JSON in a verified results bundle contains the expected
revision and SHA-256 of every selected file. The checker below reads that
public identity and hashes the actual local bytes; it does not trust the hashes
claimed by your local snapshots file alone. Its output is private because it
contains local paths. First verify the results bundle itself: this checker is
not an independent signature or authenticity check for a reference JSON.

## Source checkouts and additional checkpoints

The official source revisions used by this evaluation are:

- [MoM-CLAM at 74e3a327](https://github.com/StarkVision-AI/MoM-CLAM/tree/74e3a3277e1dfe9ae9ed433b6e8c51d74e9e1d9b).
  The checkpoint is already tracked at
  `model_wts/best_model_triplet_loss_margin_0.2.pth`; training a new head is not
  necessary to acquire this evaluated file.
- [FST at b564f8be](https://github.com/Mippia/FST-AI-music-detection/tree/b564f8be8b3db6b7810c2aab61f0b4f86f889579).
  Its README links separate Google Drive downloads for
  [Stage 1](https://drive.google.com/file/d/1frT4Mn0l6rso407Sy3eWCKbZmgwuVceN/view)
  and [Stage 2](https://drive.google.com/file/d/1E_xPsosYWI4UjKT8XQCbZW4ILvsWnmda/view).
  These two files are not present in the pinned Git tree. Save them as
  `checkpoints/backbone_stage1.ckpt` and `checkpoints/classifier_stage2.ckpt`.
  The README declares GPL for its code/demo; this document does not infer a
  broader model-weight license from that sentence.
- FST also uses the official
  [BeatThis final0 checkpoint](https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt).

Create new directories; do not check out revisions inside an active evaluation
checkout. Example source acquisition from the repository root:

```bash
mkdir -p external
git clone https://github.com/StarkVision-AI/MoM-CLAM.git external/MoM-CLAM
git -C external/MoM-CLAM checkout --detach 74e3a3277e1dfe9ae9ed433b6e8c51d74e9e1d9b
git clone https://github.com/Mippia/FST-AI-music-detection.git external/FST
git -C external/FST checkout --detach b564f8be8b3db6b7810c2aab61f0b4f86f889579
```

Download the two FST checkpoints through the linked provider pages and follow
their normal download confirmation flow. A successful HTTP status is not a
checkpoint: the observed initial Drive response was a virus-scan-warning HTML
page. Do not pass that HTML to a model loader. Keep failed/partial downloads
separate; compare complete files to `EXTERNAL_CHECKPOINTS_SHA256SUMS` or use the
identity checker below. The original provider filenames differ from the
adapter's required filenames.

## Verify before executing model code

The verifier uses only the Python standard library plus the `git` executable.
It does not import external Python modules, deserialize weights, download
anything, or run a detector. It checks actual checkpoint bytes, every selected
HF file, the source Git revision, and the complete expected Python-file set.
Tracked bytecode caches are excluded just as they are in the evaluated source
identity. Modified, missing, extra Python files and HTML masquerading as a
checkpoint are explicit non-passing outcomes.

After acquiring the additional files, set paths for the existing runner:

```bash
export ARTIFACTBENCH_CLAM_REPO="$PWD/external/MoM-CLAM"
export ARTIFACTBENCH_FST_REPO="$PWD/external/FST"
export ARTIFACTBENCH_BEAT_CHECKPOINT="$PWD/external/beat-this/final0.ckpt"

python -B -S -m artifactbench.v12.verify_external_assets \
  --reference out/my-public-results/fst.json \
  --snapshots out/my-baseline-assets/snapshots.json \
  --output out/my-fst-asset-audit
```

Repeat with the JSON for each external model (`spectttra`, `spectttra_beta5s`,
`ast_60s`, `deepfense`, `deezer_ismir`, `clam`, `fst`) and a new audit output.
The source paths are unnecessary for a model that uses only HF files.
Missing/mismatched assets produce `audit.local.json` and exit status 1. Invalid
input schemas or unsafe paths stop the check rather than producing a success
report. Input changes during inspection also fail. A passed check is a
point-in-time asset check: do not modify the assets afterward.

The reference's smoke/full scope is preserved in the audit. A historical smoke
identity can verify assets used by that run, but cannot establish complete rc2
results. Fresh inference still requires matching audio, runtime, adapter code,
seeds, device settings, and a new output directory. Run it only after the
existing GPU queue has finished.

## Measured preparation check (2026-09-05)

All seven external-model identities passed. The nine HF revisions were read
from the existing cache and rehashed, not downloaded again. CLAM and FST were
freshly cloned into a separate temporary tree; every expected Python source
file matched. CLAM's tracked 22,060,722-byte head and the separately downloaded
FST Stage 1 (1,287,502,845 bytes), Stage 2 (47,513,137 bytes), and BeatThis
(81,058,141 bytes) all matched the reference SHA-256 values.

The actual 2,428-byte Drive confirmation HTML was also deliberately passed as
a checkpoint: the verifier returned `hash_mismatch` and exit status 1, with
the other assets still passing. A fresh six-file copy of the checker outside
the checkout verified the FST files using `python -B -S`, with custom Python
search paths unset. No model was loaded in either check.

Evidence: `out/v1.2_external_asset_check_260905/summary.json`; individual audits
are private. Unit tests: 239 passed in the ONNX export environment; 236 passed
in each original environment, with three ONNX-only tests skipped and the
separately measured slow test deselected. These checks use the published-form
identities of completed historical smoke runs; they do not certify full rc2
measurements or fresh inference with the downloaded assets.

# ArtifactBench 1.2-rc2: metadata-only reproduction bundle

This is a locally prepared release candidate, not a published dataset or a
completed detector benchmark. It contains 2,579 primary metadata entries,
2,576 identified recording groups, and three verified native transport pairs.
The manifest SHA-256 is
`feab7c4c3d037919fd784dc40e46199470088abde532c5181e34573f20dceefd`.

No audio, model weights, predictions, raw song pages, lyrics, full prompts,
private path maps, or inference environments are included. This bundle proves
only that the metadata checks and notebook can run independently of the
author's data directories. It does not establish detector accuracy, access to
every source recording, or a portable full-model reproduction.

## Verify without installing any package

Extract the ZIP into a new directory and run from its top-level directory:

```bash
python3 -m artifactbench.v12.verify_bundle --bundle .
python3 -m artifactbench.v12.verify_release --release out/v1.2_frozen_rc2_260905
```

Python 3.10 or newer is required for these standard-library checks. The file
inventory checks relative names, SHA-256 hashes, and release identity. Hashes
detect changes against the included manifest; they are not a digital signature
or an independent claim about the publisher's identity.

## Execute the source notebook

Use a dedicated Python 3.12 environment. The package installation step needs
access to its package sources; notebook execution needs no audio, GPU, weights,
or network downloads.

```bash
python3.12 -m venv .venv-notebook
.venv-notebook/bin/python -m pip install -r requirements-notebook.lock
.venv-notebook/bin/python -m artifactbench.v12.execute_notebook \
  --notebook notebooks/v1.2_metadata_quickstart.ipynb \
  --release out/v1.2_frozen_rc2_260905 --output out/my-notebook-execution
```

The source notebook has no prefilled outputs. Six code cells verify counts,
recording groups, source/version/date evidence, paired views, and exposure
limitations. The executor preserves actual kernel outputs and fails on errors;
use a new output directory for each attempt.

## Scope and interpretation

Read `docs/DATASET_CARD_v1.2.md` first. All 820 real-labelled entries are legacy;
the contemporary native set has no independent real-audio FPR estimate. Udio
generator versions are unknown. Platform-generated-or-edited is not a verified
fully synthetic label. Three native transport pairs support only descriptive
comparisons. The native pool is retrospective, not a newly sealed blind holdout.

The protocol, duplicate audit, statistics specification, transport protocol,
and rc1-to-rc2 identity correction are included under `docs/`. They refer to
additional local construction or inference artifacts that are deliberately
not included here. Metadata verification does not re-decode the source audio
or independently repeat the full waveform-identity audit.

`LICENSE` applies to the supplied project code. It does not grant rights to
third-party recordings, song pages, model weights, or other source content.
Source access and redistribution require their own terms and permissions.

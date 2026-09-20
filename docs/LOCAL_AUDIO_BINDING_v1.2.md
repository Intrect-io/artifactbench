# Binding acquired audio to the public rc2 metadata

The public metadata does not contain audio or machine-local paths. Obtain each
file under its own terms, then bind its **exact encoded bytes**, not merely a
matching song title or a newly transcoded copy, to the public entry ID. This
step neither downloads audio nor grants redistribution rights.

## Path-map contract

Create a private UTF-8 JSON file with exactly three top-level fields:

- `schema`: `artifactbench-local-paths/1`.
- `primary`: an object mapping each `manifest.public.json` entry's `id` to its
  local file path. All 2,579 rc2 primary IDs are required for a complete binding.
- `native`: an object mapping `primary_id + "-" + variant_id` from
  `native_pairs.public.json` to the corresponding variant file path. All three
  verified rc2 pairs are required for a complete binding.

Relative paths are relative to the **path-map file's directory**, not the shell's
working directory. Absolute paths and resolvable local symlinks are supported;
the saved manifests contain resolved absolute paths. URLs are not accepted.
Omitted IDs or `null` values mean unavailable files; the audit records them and
the command fails. Unknown IDs, duplicate JSON keys, and malformed schemas fail
before an output directory is created. The map supplies paths only: labels,
groups, source identities, checksums, and other metadata come from the verified
public release, never from the path map.

## Bind and inspect

Run from this source checkout with Python 3.12. This command uses only the
standard library and does not import a detector, decode audio, or use a GPU:

```bash
python -m artifactbench.v12.bind_local_audio \
  --release out/v1.2_frozen_rc2_260905 \
  --path-map private/paths.local.json \
  --output out/my-rc2-local
```

Use a new output directory for each attempt. Output inside the source release
is rejected, including through symlinks. Existing output directories are never
overwritten. The original frozen release is not edited.

The command first verifies the public metadata, then checks each supplied file's
size when declared and its SHA-256. It detects stat changes while hashing and
checks that neither the public inputs nor the path map changed during binding.
On missing, inaccessible, changed, or mismatching audio it writes an explicit
private audit and incomplete summary, exits nonzero, and **does not write runnable
local manifests**. A native variant is a complete pair only when its primary
also passed. It never fills a missing input or invents a detector probability.

Successful output contains the four public JSON files copied byte-for-byte,
`manifest.local.json`, `native_pairs.local.json`, and these private records:

- `binding_inputs.local.json`: input, map, and tool hashes plus Python version.
- `binding_audit.local.json`: per-file outcome and measured byte identity.
- `binding_summary.local.json`: verified counts and output hashes.

Keep the **whole output directory private**: the local manifests and diagnostics
contain local filesystem paths. A successful binding proves byte identity at
check time, not successful decoding, provider identity, model execution,
availability on another machine, or audio redistribution permission. The
inference runner rechecks each file's hash before using it.

Use `--release out/my-rc2-local` for subsequent inference and transport
preparation, with **new result directories**. Its private manifest hash may differ
from an author's original local manifest; an existing run identity must not be
reused. The public manifest hash stays fixed. Model acquisition, runtime identity,
and full evaluation remain separate steps in [the reproduction guide](REPRODUCING_v1.2.md).

## Verification evidence

`tests/test_v12_local_binding.py` covers 31 path/schema/hash/output-boundary
cases with explicitly labelled byte-only unit fixtures, not benchmark audio.
These include same-size wrong content, missing primaries and variants,
symlink loops, permission errors, mutation during hashing or map reading,
immutable public copies, and refusal to overwrite existing output.

The actual rc2 command completed on 2026-09-05 in the separate standard-library
verification environment. It rehashed **2,579 primary files (25,668,835,924 bytes)
and three native variants (27,259,100 bytes)**; all 2,582 matched. The input map
was derived from the author's original local manifests, so this is an actual
local availability/byte-identity check, not independent public reacquisition.

Evidence is `out/v1.2_local_audio_binding_rc2_260905`. Its private summary has
SHA-256 `9155b50ff5761ba236f86074deb8f1043ed78493e856a75c346010f1fa29650d`,
and audit SHA-256
`73271b444692f0c067afafbd2f47dc4e34205a47deab09fdc3fe1738b065c4f8`.
The four public JSON copies are byte-identical to frozen rc2. The complete
primary ID/path/hash/label/source/group projection consumed by the runner
matches the original local release; the new private manifest also carries
public metadata fields and therefore has its own hash. The copied release
passed the metadata verifier with Python site imports disabled. No decoding,
detector execution, or changes to the live inference queue occurred here.

All 167 unit tests passed in both evaluation environments, with the separately
measured real-weight smoke test deselected. Unit tests and real-file binding
are distinct evidence; neither is the pending full-inference reproduction.

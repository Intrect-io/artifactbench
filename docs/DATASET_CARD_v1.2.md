# ArtifactBench 1.2 release-candidate dataset card

**Correction, 2026-09-05 13:20 KST:** rc1 is not a valid final benchmark.
Complete-URL and waveform checks found five misselected Udio primaries and
seven sibling outputs incorrectly called native transports. The original run
was stopped. rc2 has passed revalidation and is now frozen. See
[the evidence and correction policy](UDIO_IDENTITY_CORRECTION_v1.2.md).
Counts/hashes below describe rc2; the original rc1 files remain preserved.

Date: 2026-09-05. Status: **data frozen as 1.2-rc2; eight-checkpoint detector
and controlled transport evaluations complete; public release not complete**.
Local canonical release: `out/v1.2_frozen_rc2_260905/`.

## Scope and composition

This is a retrospective, source-stratified AI-music detection evaluation, not a
probability sample of music uploads or a universally unseen test set.

| Partition | Entries | Interpretation |
|---|---:|---|
| Legacy | 2,164 | Corrected September 5 historical selection, 28 source cells |
| Suno v5.5 native | 200 | Exact recording version evidence, 122 creator IDs |
| Udio 2026 native | 200 | Creation in 2026, 172 creator IDs; model version unknown |
| Lyria 3.5 official demos | 11 | Provider-selected supplementary examples |
| Stable Audio 3.0 official demos | 4 | Provider-selected; model size unknown |
| Total | 2,579 | 1,759 AI-labelled / 820 real-labelled entries |

The 2,579 entries form 2,576 **identified** recording groups and 2,456 dependence
clusters under the recorded evidence. These are not claims of exhaustive
deduplication or independence. Three additional native transport files are paired
views, not extra primary entries. The 820 real-labelled entries are all legacy;
the native cohort does not independently estimate a contemporary real-audio FPR.
The 22 legacy AI source/version cells are not 22 independent generator families.

## Collection and labels

The pre-score protocol is `PROTOCOL_v1.2.md` revision 2. A fixed native inventory
contains 30,213 file variants / 28,488 recording IDs. Metadata survey order is
SHA-256 of `260905:<recording-id>`, capped at two recordings per provisional
creator. The survey read 1,200 Suno and 271 Udio public pages, then reapplied
identity/exposure checks and the creator cap using stronger public user IDs.
No new detector scores were read for selection, replacements, or quotas.

Suno supplied 1,006 version-labelled records, 186 version-unknown records, and
eight explicit errors; 426 were v5.5. Its exact `major_model_version` field is
used, not the crawl date. Udio's CDN media ID must be joined to its separate
song-page UUID. All 271 parent-ID joins succeeded, but none disclosed a model version.
Sixty-nine Udio recordings were created before 2026, motivating the documented
metadata-only date restriction before scoring. Unknown versions remain unknown.

Parent-ID joining alone was insufficient: rc2 checks the complete page-declared
URL for every selected native entry. All 200 Suno files match their exact clip
object's audio/video URL. Of 200 Udio files, 199 match the current song URL and
one matches a declared untrimmed original URL. The inventory `track_id` is a
sampling-slot key; `provider_recording_id`, source-link evidence, complete URL,
and file hash disambiguate the evaluated recording. Seven sibling-output
candidates are excluded from native transport analysis; three MP3/AAC pairs
passed independent full-waveform matching (correlation 0.99890–0.99978).

Platform generation, extension, cover, remix, or upsampling may involve human
input. The native positive label means **platform-generated-or-edited**, not
necessarily fully synthetic vocals and accompaniment. Official-demo production
workflows are not independently audited. Legacy labels inherit their sources.
No per-song detector score can establish authorship or justify punitive action.

## Validation and dependence

All 2,579 primary and three paired-view files have successful signature, full
FFmpeg decoding, nonempty/finite PCM, and SHA-256 evidence. For rc2, byte hashes
were rechecked and the same-file complete-decode records were explicitly rebound;
this is cached evidence reuse, not a claim that FFmpeg was rerun on every file.
The duplicate search and full native-pair waveform checks were rerun. Original
sample rate, channel count, codec, format, and duration are retained. The 120
legacy HTML-disguised FMA files were mapped to genuine same-ID originals; bad
downloads were preserved, not silently overwritten.

`DUPLICATE_AUDIT_v1.2.md` specifies the candidate search. Every primary file has
a first-120-second Chromaprint fingerprint. Nine candidate pairs received
independent waveform verification and full-length follow-up:

- Three same-recording/excerpt pairs: one byte-identical Yue pair, two MoM
  real-audio transport/excerpt pairs with matching source recording IDs.
- Three Udio pairs with shared audio but different extensions/edits.
- Three candidates not confirmed under the declared waveform criteria. One
  Udio pair has aligned full-waveform correlation 0.934 below the 0.985 rule;
  it is already in the same creator dependence cluster and remains documented.

The audit does not establish absence of shared samples, covers, time stretching,
or matches outside fingerprinted excerpts. Legacy duplicates remain in the
historical entry view. The `recording_representative` field defines the
deterministic recording-balanced sensitivity view. Creator/lineage clusters
must be preserved when estimating uncertainty, including across source cells.

## Exposure and evaluation

Five pinned local ArtifactNet train/validation/checkpoint-selection manifests
had zero unresolved path-stem candidates against these entries, after the
explicit different-generator seed exception. Metadata-level creator and lineage
checks also exclude documented native overlaps. This is not a full audio-level
search against all training data, and the native source pool was previously
mined and observed during ArtifactNet development. It is not prospectively blind.

External checkpoint membership is **unknown** without per-recording training
manifests. SONICS, MoM, FakeMusicCaps and FMA overlap risks remain checkpoint-
specific. The fakeprint LR model is the `lofcz` third-party implementation,
not the official Deezer detector. `exposure.public.json` preserves these limits.

Primary decisions use raw P(AI) >= 0.5; ArtifactNet's existing CNN 0.225 plus
matched rescue 0.02 is separate. Report failures, scored-only and all-attempt
metrics, common-success paired comparisons, micro/macro source metrics, and
clustered uncertainty. Do not pool tiny provider-demo cells into a headline
population metric. No new threshold is calibrated on these evaluation labels.

## Files, identity, and access

- `manifest.public.json`: metadata-only primary entries; no local paths, lyrics,
  full prompts, credentials, or raw page snapshots.
- `native_pairs.public.json`: three verified transport views tied to primary IDs,
  with full-waveform identity evidence.
- `exposure.public.json`: model-specific exposure limitations and checked inputs.
- `release.json`: authoritative counts and public-file hashes.
- `manifest.local.json`, `native_pairs.local.json`, `provenance.local.json`:
  local reconstruction/evidence files, **not public distribution artifacts**.

Primary public manifest SHA-256:
`feab7c4c3d037919fd784dc40e46199470088abde532c5181e34573f20dceefd`.
URLs and recording IDs enable source-specific reconstruction, subject to source
availability and permissions. Legacy reconstruction still requires the original
datasets and their identifiers; no universal public audio archive is supplied.
Changes require a new manifest identity, not in-place replacement.

Metadata verification (Python standard library only; tested in a clean venv):

```bash
python -m artifactbench.v12.verify_release --release out/v1.2_frozen_rc2_260905
```

This checks hashes, counts, group representatives, paired-view references, and
accidental local/raw-content fields. It does not assert current audio availability
or replace the portable model-execution quickstart; model-weight portability is
tracked separately.

The [metadata quickstart notebook](../notebooks/v1.2_metadata_quickstart.ipynb)
also passed all six code cells in a new, GPU-free Python 3.12 environment. It
recomputes counts, source composition, native version/date evidence, the three
verified MP3/AAC pairs, and exposure limitations. It does not run
the detectors. To reproduce from the repository root:

```bash
python3.12 -m venv .venv-notebook
.venv-notebook/bin/python -m pip install -r requirements-notebook.lock
.venv-notebook/bin/python -m artifactbench.v12.execute_notebook \
  --notebook notebooks/v1.2_metadata_quickstart.ipynb \
  --release out/v1.2_frozen_rc2_260905 --output out/metadata-notebook-new-run
```

Use a new output directory for each execution; failed executions are preserved.
The source notebook contains no prefilled result cells. The executor saves only
actual kernel outputs and refuses errors or the wrong kernel environment.

A metadata-only ZIP was also extracted into a fresh temporary directory and
verified with Python's site imports disabled and `PYTHONPATH` removed. All 20
allowlisted payload files passed SHA-256 checks; all six notebook code cells
then passed using the separate notebook environment. The notebook verifies its
actual interpreter path, environment prefix, and local verifier import root.
This uses an already isolated dependency environment, not a second claim of a
fresh dependency installation. Audio, weights, predictions, private path maps,
and raw pages are excluded. See `METADATA_BUNDLE_README.md` in the source repo
(installed as the ZIP's top-level `README.md`). Full-model reproduction remains
a separate, unfinished gate.

Public playback does not confer audio redistribution rights. This release starts
with metadata and reconstruction guidance only; source licenses/terms remain
separate. No Stable Audio model license was newly accepted and no paid generation
was initiated. A larger controlled-generation cohort requires a separate plan.
The runner's MIT license does not license third-party audio or weights.

## Remaining release gates

Finish final model-release verification (including ONNX full parity) and the
arXiv source bundle. Metadata, saved-score reconstruction, and result-notebook
portability checks have passed. Actual public release or arXiv submission is a
separate action, not implied by this draft.

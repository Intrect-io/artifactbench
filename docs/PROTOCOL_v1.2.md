# ArtifactBench 1.2 evaluation protocol

Protocol revision: 2026-09-05 / 2. Status: **pre-score protocol**, not a released
dataset or a completed evaluation. The new selection has not been scored by
this benchmark run. Historical mining scores exist for the native corpus and
are explicitly outside the sampling read-set.

## Questions and units

1. How do fixed detector checkpoints behave across legacy source cells and
   contemporary, version-evidenced platform outputs?
2. How much does the same recording's score change under native transport and
   controlled PCM quantization / lossy encoding?
3. How do coverage, source imbalance, and known exposure change interpretation?

The independent recording is the count unit. A creator or known recording
lineage is a resampling cluster. A source cell is not necessarily an independent
generator family. Transport variants and excerpts do not increase the number
of recordings. Scores estimate performance on the stated sampling frame, not
on all online music or every possible synthesis architecture.

## Partitions and score-blind selection

- **Legacy comparison:** retain all 2,164 entries of the September 5 corrected
  test selection, including the 120 same-ID FMA original-audio replacements.
  Keep the historical one-chunk inference failure in the denominator and log.
  This partition is retrospective, not an untouched final test.
- **Contemporary native challenge:** target up to 200 Suno v5.5 recordings and
  up to 200 Udio recordings created on or after 2026-01-01. Udio's generator version remains unknown unless
  per-recording evidence establishes it. The Udio cell measures a collected
  platform cohort, not a claimed latest-version cohort. Report creation-date
  distribution separately from collection date.
- **Official demonstration challenge:** admit all distinct music demonstrations
  in the explicitly version-labelled Lyria 3.5 carousel and Stable Audio 3.0
  demonstration list. Keep these in separate tables, not pooled into a
  population-level contemporary score. Demo selection is by the provider;
  model-size labels are unknown unless the page establishes them. Repeated
  light/dark media and shortened copies are not independent recordings.

Native metadata survey: rank provider-prefixed recording IDs by SHA-256 of
`260905:<recording-id>`, cap each provisional creator at two recordings, inspect
the first 1,200 Suno and at most 300 Udio candidates. Actual Udio survey size is
271 after this cap. Preserve all candidate IDs, input hashes, and exclusions.
Enrich creator identity before final selection, then apply the same hash order
and a two-recording creator cap again. Do not read detector scores for ordering,
replacement, version assignment, eligibility, or sample count decisions.

Exclude candidates with documented training-track, training-creator, or
model-selection creator overlap against the pinned local exposure manifests.
Recheck enriched handles/artist identifiers against the same index. Where a
creator's old and new identity cannot be joined, mark that limitation; absence
of an exact string match is not proof of universal independence. Unknown
creator candidates are excluded from the creator-disjoint native challenge.
If fewer than a target quota qualify, report the shortfall; do not duplicate
tracks or relax the rule after seeing scores. Save byte hashes and successfully
decode the entire admitted file before the final manifest is frozen.

Use a single deterministic primary native view: audio-only MP3 when available,
otherwise the recorded native MP4. Retain other native views only for paired
transport analysis. Provenance labels distinguish generated-or-edited platform
output from demonstrated text-to-music output; remix, upsample, conditioning,
and lineage evidence remain separate fields. An unknown field is not false.
No copyrighted prompt/lyric text is needed in the public native manifest.

## Exposure reporting

Pre-score amendment (revision 2): page metadata showed that 69 of the 271 Udio
survey recordings were created in 2024/2025 despite the 2026 crawl. Restrict the
contemporary Udio frame to calendar-year 2026 (202 candidates before subsequent
identity/exposure/decode checks). This uses creation metadata only, with no new
detector scores available, and makes the temporal cohort label meaningful.

Known train, validation, checkpoint-selection test, hard-negative mining, and
researcher-observed evaluation are distinct exposures. The native corpus was
used in ArtifactNet development before this benchmark and must not be described
as a prospective blind holdout. A model-specific exposure matrix accompanies
results. A missing external baseline training manifest is `unknown`, not
`unseen`. Preserve legacy SONICS/MoM/AIME overlap limitations for their associated
baselines instead of claiming all-model leak freedom from one split name.

## Detector comparisons

Evaluate the eight existing adapters: ArtifactNet, SpecTTTra alpha-120s,
SpecTTTra beta-5s, CLAM, Deezer ISMIR, FST, AST-60s, and DeepFense. Freeze local
weights, adapter hashes, upstream revision, environment, duration limit,
resampling, channel policy, and aggregation per detector. These are evaluated
checkpoints, not retrained architecture comparisons. Report native input budgets
and compute separately; short-context and long-context systems are not assumed
to consume equivalent evidence. Do not replace unavailable models silently.

2026-09-05 input-path correction: the original DeepFense adapter condition
remains an audit result, not the final unqualified checkpoint comparison.
The separately versioned `deepfense-native-soxr/2` condition restores native-
rate float64 loading and upstream librosa/validation padding, with an explicit
AAC/Opus-only native-rate FFmpeg decoder extension. Version 1's 32-source smoke
is preserved with 31 scores and one unsupported WebM/Opus input error; version 2
adds that measured format without changing the scores or policy of version 1.
The fixed checkpoint, class
direction, threshold and 2,579-entry membership do not change. Input failures
remain counted separately and no favorable subset substitutes for the full
re-evaluation. Exact processing/device/transport contracts and evidence are in
`DEEPFENSE_INPUT_PATH_AUDIT_v1.2.md`; both original and corrected conditions
must remain distinguishable in the final report.

Primary hard decision: each raw AI score >= 0.5. Report threshold-free AUROC
alongside this fixed operating point. The pre-existing ArtifactNet deployment
CNN threshold 0.225 plus matched rescue threshold 0.02 is a separate operating
stack, not a substitute score for raw-detector ranking. No threshold is selected
on the new test partition. Threshold sweeps, if included, are descriptive and
cannot be labelled an independently calibrated deployment result.

Report TP/FP/FN/TN, precision, recall, F1, balanced accuracy, specificity, AUROC,
per-cell TPR/FPR, unweighted AI-cell macro TPR, real-cell macro FPR, and the
worst observed cell with its sample size. Micro results are conditional on this
benchmark's class/source mixture. Do not compare raw precision across cohorts
with different class priors as if they were interchangeable.

Use 2,000 deterministic percentile bootstrap replicates, seed 260905, clustered
by creator/lineage where established and otherwise by recording. Stratify by
label and source cell for within-mixture intervals. Paired model differences
use the same resampled clusters and common successful recordings, with their
coverage shown. Tiny provider-demo cells receive counts and appropriately wide
descriptive intervals; they cannot support broad model-family rankings.

## Failure handling and validation

Before inference: verify file signatures, byte hashes, complete decode, duration,
sample rate, channels, nonempty finite PCM, and exact duplicates. Search for
perceptually duplicated recordings and verify candidate matches; fingerprint
similarity alone is not a duplicate verdict. Preserve original files and all
exclusions. Do not infer file health from an extension or `exists()`.

Infrastructure failures (missing dependency, OOM, device contention) stop or
retry the affected run with explicit provenance. Intrinsic per-file model
failures remain in coverage. Report scored-only metrics plus an all-attempt
correctness metric treating abstentions as incorrect; this is an error policy,
not an invented model prediction. Also report success-only metrics on common
coverage for paired comparisons. No fake 0.5 scores or empty successful tables.

## Transport experiment

Deterministically sample up to 50 lossless-source recordings from the legacy
partition by the same seeded hash order, balanced across eligible source cells.
Record original subtype: PCM-float staging cannot undo prior quantization.
From exactly the same decoded float32 mono signal create:

1. identity / float32 WAV round-trip (numerical control);
2. PCM16 WAV round-trip (quantization only);
3. MP3 128 kbit/s from float32 WAV;
4. AAC 128 kbit/s from float32 WAV;
5. Opus 128 kbit/s from float32 WAV;
6. the same three lossy encoders from PCM16 staging (historical-path control).

Use explicit `pcm_f32le` when decoding and record ffmpeg version/commands. Treat
codec delay/length handling consistently and record input/output frame counts.
Report absolute score shifts, within-recording range, decision-flip counts and
paired uncertainty. Separate native transport pairs from generated transcodes.
Do not attribute the entire old PCM16-staged codec delta to compression alone.

## Release and paper gates

Release metadata, hashes, access/reconstruction instructions, and local path map
separately. No audio rights are inferred from public availability. Raw page
snapshots are local audit evidence, not an automatically redistributable corpus.
Unavailable/licence-gated generation is recorded; no paid generation or new
license acceptance is required by this protocol. A larger controlled-generation
cohort requires a separately documented acquisition plan and access authority.

All tables and figures derive from saved manifest/prediction hashes. The paper,
dataset card, and executable quickstart share the same release identity.
Completion requires actual baseline runs, count/hash audit, a fresh quickstart,
compiled and visually inspected PDF, bibliography checks, and an arXiv source
bundle. Draft status stays explicit until those gates pass. Actual publication
or arXiv submission is not implied by preparation of the bundle.

# ArtifactBench 1.2 + arXiv benchmark paper

Started: 2026-09-05. Status: active; no release or paper completion claim yet.

**20:50 transport 6/8 completed:** native DeepFense, SpecTTTra alpha/beta,
AST, original-source ArtifactNet and Fakeprint each406/406 scores and verified
2,000-bootstrap transport statistics. ArtifactNet used read-only bwrap source
overlay with /dev and /proc; all53 original-view probabilities exactly matched
primary. Current unit `artifactbench-clam-transports-260905.service` PID833070;
FST transport still pending. All primary8 results/public ZIP notebook reproduction
are complete. LaTeX v13 (10pages) includes primary results; both source ZIP builds
match. Transport synthesis, ONNX parity/release and final artifact updates remain.

**20:24 ArtifactNet transport preflight stopped safely:** RTX_ENV packages and
benchmark code pins match primary, but live ArtifactNet source inventory changed:
`src/kernels/hpss_fallback.py`, `src/pipeline/infer.py` hashes differ and six
new Python files exist. Strict model_assets guard rejected before inference;
unit `artifactbench-artifactnet-transports-260905.service` failed, not OOM.
Do not revert user source or weaken identity checks. Next investigate preserved
source snapshots for an isolated exact-source replay; other model transports
can proceed independently. Native DeepFense transport already completed406/406
and full public-score ZIP portability passed all6 notebook cells.

**20:13 full public statistics reproduced:** all nine JSON statistical files
byte-identical in `.venv-results`, including 28 paired comparisons and intervals.
Public predictions: `out/v1.2_public_results_rc2_260905`; generated paper
tables/figures: `out/v1.2_paper_results_rc2_260905` (not yet inserted into main.tex).
Native DeepFense transport fresh retry started as
`artifactbench-deepfense-native-transports-retry-260905.service`, CPU2threads,
MemoryHigh8GiB/Max10GiB/Swap0, output `out/v1.2_deepfense_native_transports_retry_260905`.
Old interrupted output preserved. No other heavy inference started.

**20:06 original FST full complete:** 2579/2579 scored, unit exit0,
last resume peak4.4G/swap0. All row hashes, identity and membership verified;
2000-bootstrap checkpoint completed in `out/v1.2_checkpoint_fst_rc2_260905`.
Legacy F1 .74341193, TPR .59821429, FPR .01829268; Suno108/200,
Udio151/200, official demos0/15. Eight-model primary inference now complete.
Started `out/v1.2_statistics_rc2_260905` with native DeepFense override.
Full transports, ONNX qualification and final public/paper artifacts remain pending.

**19:17 correction: alleged one-hour FST stall was a timing interpretation error.**
Journal proves regular scored rows through 19:11:30 (entry1947), followed by
assistant-requested SIGINT at 19:11:34. Elapsed service lifetime was confused
with time since last prediction. There is no evidence for a stuck input.
Stopped the unnecessary isolated full unit and preserved its experimental output.
Verified original1947 rows readable and source pins unchanged, then resumed the
original service (PID389426), with MemoryHigh6GiB/Max8GiB/Swap0 intact.
Use original FST completion for the primary comparison. Experimental worker
code/results remain unadopted; do not change the timeout policy or rerun all
entries on the basis of the retracted stall diagnosis.

**Unadopted FST isolation experiment:** `run_fst_isolated.py` is a separate condition,
not a mutation/resume of the preserved partial run. Every fixed entry receives
a fresh Python/CUDA worker with a 600s ceiling; timeout creates a hash-bound
no-score error. Actual one-entry cgroup smoke (6/8GiB, swap0) matched a prior
FST P(AI) and diagnostics bit-for-bit; 14.451s CPU/2.2GiB peak. Targeted tests
22 passed. The experimental full run was stopped following the correction above.

The 19:12 alleged stall diagnosis is retracted. Neither the 130.896s Udio
nor the 275.2s Suno input was shown to be defective; the adapter caps at90s.

**Post-reboot transport/ONNX preflight:** native transport `model_info.json`
is also zero-byte invalid JSON. Preserve the interrupted output and use a fresh
directory for its eventual rerun. All118 snapshot mappings (117 unique content
files) hash-match; empty __init__.py snapshot is legitimately empty, not damage.
ONNX full's51 pinned input/source files currently hash-match. No inference
started; FST remains the sole benchmark. Previous turn was a verified wait.

**LaTeX v12 checkpoint:** completed native DeepFense full results now reflected
in main.tex. Both source-package builds pass9-page text/raster equivalence;
repeat ZIP identical (29475bytes, SHA99fb41240a3b4bc09d53f4813b85549b5bbbac93527bb6f9f3c6ad446018452f).
Changed pages6–9 visually inspected; five original underfull warnings only.
Source-reproduction docs updated. FST unit72689 verified active through1134;
no parallel heavy inference. Previous turn made progress through mount recovery.

**18:08 mount restored:** Archive now mounted at the original path (external
state change). All2579 input paths exist; previously missing FMA file hash
matches freeze. FST unit restarted with unchanged 6/8GiB RAM limits and swap0;
PID72689 active, actual new scored rows through1120, cgroup RAM1.70GiB.
No other benchmark launched. Reproduction docs now describe
OOM/reboot recovery and the canonical DeepFense audit records the full result
instead of stale running status. Manuscript v11 still needs that narrative update.

**18:06 correction:** FST unit exited after one new scored row (1116 total), not OOM (peak3.6GiB). Next FMA file missing because Archive `/dev/sdc2` exFAT is unmounted after reboot. Read-only `udisksctl mount --options ro --no-user-interaction` denied by polkit. No benchmark now running; user disk-mount action required before resume. No filesystem repair attempted.

**2026-09-05 18:04: OOM/reboot recovery; FST only resumed under cgroup limits.**
Previous turn verified the reboot and preserved results (progress). Prior-boot
kernel journal records OOM at 17:54:22; boot began 17:55:06. All former jobs died.
Completed original seven models and native DeepFense pass identity, all row hashes
and full membership checks. FST had 1115 valid scored rows and seven zero-byte
files; only those seven were moved to `fst/reboot_empty_records_260905` (recoverable).
Cached row metadata/probability and current source pins verified before resume.
`artifactbench-fst-resume-260905.service` runs the original command/environment,
with actual cgroup MemoryHigh=6GiB, MemoryMax=8GiB, MemorySwapMax=0,
OOMPolicy=stop and Restart=no. Unit active; no other benchmark launched.
These limits bound this job's host RAM, not VRAM or unrelated processes.
ONNX remains stopped: 611 parseable rows plus one empty file. Native transport
has 19 empty files, no usable rows or completion summary; preserve before retry.
Do not run concurrent heavy benchmarks. Next: verify resumed FST new rows and
completion, then schedule remaining jobs one at a time with measured limits.

Native DeepFense full completed: 2578 scored/input failure1; summary SHA
`84e4d5c8ee552bf0b8f52d6333192132a121e66a589efcb7f481f3c254032060`.
2000-bootstrap checkpoint SHA
`adeb644d69819c8e9e3343f225389392690025103828b814bb60c07dc3419cca`.
Legacy TPR9.5982%/FPR2.1978%; native Suno0/200 and Udio6/200. This is not an
accuracy improvement. Three native-primary smoke probabilities/preprocessing
match the full primary exactly. Original adapter results remain interpretation-held.

Prior uncheckpointed renderer work is complete: final `--predictions` admission
binds eight model statistics plus paired differences through the public loader.
Historical rc1 smoke's 10 outputs byte-match prior outputs; wrong release and
smoke-as-final rejected. Suites export306/eval304/RTX300 passed (4/4/8 skips).
LaTeX v11 ZIP 29225 bytes, SHA
`888421db61a95c5e2fde819fa5541cd5bc23702be5f7fc57a57ac3223a1c62cb`;
repeat archive identical, all nine rebuilt pages match, changed pages7–9 inspected.
Reproduction docs updated. Paper remains interim; native completion must be
reflected in the next manuscript revision. No upload or release completion.

**2026-09-05 17:17: STFT qualification prepared; final public input guard added.**
Previous goal turn made progress through transport implementation/actual smoke.
This turn revalidated all three live processes. New `qualify_onnx_stft.py`
retains the probe and fixed policy, compares the original saved GPU against
factory CPU/batch1 separately, and never creates a release export summary.
Actual preflight matched the original source-smoke's exact 34 IDs/order/audio,
factory assets and 72 pinned inputs. Preflight SHA
`66f7c18863b35e6d0e8771658c26c2959fbe2aab3cba60d52e234678fa02391c`,
under `out/v1.2_artifactnet_stft_preflight_260905`. Inference NOT started:
available memory14GiB, existing ONNX RSS12GiB/native5.5GiB at scheduling check.
Wait for headroom before the additional large CPU run; do not overlap it with
native transport by default. Exact command: docs/ARTIFACTNET_ONNX_v1.2.md.

Public full export/reload now require corrected native DeepFense identity and
prepared-row condition; actual old2,579-row run rejected and native32-row
smoke passes the condition validator only. Historical smoke mode is retained.
Generic report/renderer admission still needs review before final artifacts;
`report.py` is pinned by the running native primary AND required by the native
transport identity, so do not edit it before that dependent inference ends.
No active source or original prediction changed. Five new tests; full suites
export303/skip4, eval301/skip4, RTX297/skip8, unchanged slow test separately
measured. At this check native1663 attempted/1662 scored/input failure1,
FST629 scored, ONNX476 comparisons/failure1/flip0; all three still live and
pinned source/input hashes unchanged. No new STFT inference, model adoption,
paper version, upload, license decision or completion claim.

**2026-09-05 17:03: native transport integration and actual path smoke verified.**
The previous goal turn made progress (native input implementation/full launch
and LaTeX v10); this turn revalidated all three live processes and implemented
the aligned transport path. `run_transports --deepfense-native` reconstructs
current native package/source/runtime identity before strict comparison to a
completed primary. Controlled views promote the verified common float32/44.1k
master to float64 before native preparation; native pairs bind exact original
encoded files against frozen public/local metadata, never the old 44.1k cache.
Encoded input hashes, no-score failures and model-input hashes remain explicit.
`transport_statistics` separates common-master input-path differences from
actual primary repeatability. The former must not be called codec effects or
runtime repeatability. Other model input paths and the active native/main/ONNX
inference sources are unchanged.

Actual runtime/input preflight matches the running native primary: 400+6 views,
53 unique entries, all six native files decoded and the three MP3 primary
prepared arrays bit-exact with actual upstream. A fixed 14-view CPU path smoke
(the first frozen controlled recording's eight views plus all three native
pairs) scored 14/14. Output `out/v1.2_deepfense_transport_smoke_260905`, summary
SHA `98c1f47be8aabed58258f08353cd878f2ebbdee06c18e5cffcf5ce72ce3c343d`,
identity `ae8fa37f136cbe0e785063a72885a10a66f0c620e3f04ae2ea762b1dea62068f`.
All 14 row hashes and 118 source snapshots/current sources verified. None of
the three native primaries was yet scored in the full run, so primary score
repeatability is still unmeasured. No partial primary bypass was added.
Eleven new tests; full non-slow suites eval296/skip4, export298/skip4,
RTX_ENV292/skip8. The unchanged slow real-model test remains separately measured.

At 17:02 native full972 attempted/971 scored/input failure1, FST455 scored,
ONNX426 comparisons/failure1/flip0. PIDs3205903/2255036/2778940 remain live;
current native/FST code and all ONNX51 pinned inputs verified unchanged.
Next: finish the native primary, run all406 native transports through the full
CLI and generate its report; finish other main/transport runs, ONNX replacement
qualification, full public artifacts and manuscript. Canonical commands and
scope are in docs/REPRODUCING_v1.2.md and docs/STATISTICS_v1.2.md. Paper v10
is unchanged this turn; no upload, license decision or completion claim.

**Full native coverage update after launch:** at 340 saved rows, 339 scored and
one FMA input failure (`37e60a38e8b30b949551f7e4`). Actual upstream loading
reproduces the same `LibsndfileError`; ffprobe identifies 44.1k stereo MP3.
Separate FFmpeg native-f64 decoding succeeds (6,823,296 frames, finite, empty
stderr), but MP3 is outside the pinned AAC/Opus fallback policy. Keep the
failure without scoring replacement or changing the running /2 condition.
Record SHA `aba4d6f7085d5ce6bf29c4433d2711730fa178a4bc88ffedba9972828cffb054`.
FST302/ONNX382 at this later check, both still live; ONNX failure1/flip0.
Native112/FST8 source hashes plus ONNX51 pinned inputs verified unchanged.
After the manuscript edit, all 27 paper packaging/rendering tests pass again.

**2026-09-05 16:46: native DeepFense compatibility verified; full CPU run live.**
The separately versioned `deepfense-native-soxr/2` preserves native float64
decode/mono, direct upstream librosa soxr_hq 16k/prefix-repeat, then float32;
no common 44.1k stage or extra clip. The initial AAC-only version-1 smoke is
preserved: 31 scores plus one unsupported Lyria WebM/Opus input error. Actual
ffprobe identified 48k stereo Opus; version 2 explicitly adds that codec,
without selecting weights, thresholds or a favorable score. New actual smoke:
32/32 scored (SoundFile 30, AAC 1, Opus 1), all prior 31 waveform/logit/scores
bit-exact, all eight prior comparable upstream waveform/scores bit-exact.
Both 116-file source snapshots and all 64 public row roundtrips verified.
Summary SHA `dd74659741879c28a2bfa954f8b8b39a1b42164c41d4ad17eae941086e5f99cb`;
canonical `docs/DEEPFENSE_INPUT_PATH_AUDIT_v1.2.md`. Targeted tests 31 passed;
full suites export 287/four skips, eval 285/four, RTX_ENV 281/eight. The unchanged
slow real-model test remains separately measured.

Full native evaluation is live in `out/v1.2_deepfense_native_full_v2_260905`,
CPU/two threads, PID 3205903/session 96727, identity SHA
`11f8b7d7165579e949228445a7eef90991c5f46eb2edd97578ddae45650492cf`.
81 scored/no failures at this checkpoint. Device difference is explicit;
smoke mean 0.809 s inference is not dedicated throughput or CUDA timing.
Do not edit its pinned runner/helper/report or original code while it runs.
Use an explicit DeepFense run override for final reporting, align native and
controlled transport processing, and preserve the original interpretation hold.

**Seven original checkpoints complete; FST remains live.** CLAM scored all
2,579, no errors; summary SHA
`b5ccc1b0cfed0852cff4588bb0321a2f7cd374b724121699a39eb896be2b31e1`,
identity `e57501836e349a3b82955ffa2a0bbf452f6ac7563f0671d1034d3745ec265638`.
Full hash/membership-verified 2,000-bootstrap report:
`out/v1.2_checkpoint_clam_rc2_260905`, checkpoint SHA
`8ab5f2adcb62a0d0e2dadd83abb9090b421fe78ec73f4949b9ea0f9fecffca48`.
Legacy TP1176/FN168/FP572/TN248; F1 .760673 [.749917,.771429],
FPR 69.756% [66.585,72.983], AUROC .707546. Suno147/200=73.5%
[66.839,79.798], Udio166/200=83.0% [77.169,88.351]; non-demo F1 .782654.
Official demos separate: Lyria7/11, Stable4/4. FST239 scored/no errors at this
check, PID2255036/session96096. All eight original identities' 40 code hashes
still match current files. Original DeepFense is complete only as an adapter
audit, not an unqualified final comparison result.

Full ONNX v2 remains live at 364 comparisons, one tolerance failure, zero
raw-.5 flips; all 51 pinned source/input hashes unchanged. This candidate
does not qualify for full-parity publication. No new ONNX graph was adopted.
LaTeX v10 is still nine pages, now including the native-input implementation
and measured smoke limits. ZIP29,101 bytes, SHA
`bf7d9d8e49cc32b9598e436f974d63be129c3a55e16358b6bae642e6cded7251`;
two builds byte-identical, both fresh compilations match all page text/raster
hashes. Changed pages6–9 visually checked; PDF SHA
`3062db468abf4a649009556acae4b5fe0be4059ada298dad2fd52ff4d3ffbbbf`.
Five underfull warnings remain; no overfull/unresolved references.
Next: finish remaining inference, implement aligned native transport scoring,
verify any ONNX alternative on the fixed source/boundary selection and full
reference, then final statistics/public-source bundle/manuscript. No upload,
license decision or arXiv submission occurred.

**2026-09-05 16:16: ONNX numerical decomposition and bounded STFT probe complete.**
For the first failed recording, original factory CPU/batch1 gives
0.7282716435500012 versus saved GPU 0.7292302923969539. The FFT/batch1
export graph matches all four observed stage tensors bit-for-bit and its final
score within 2.22e-16. DFT substitution contributes an additional ~8.2e-5;
actual v2 ORT replay is exactly the prior saved result. Diagnostic summary
SHA `68ce44b9f8f4b8d8c01ff08d5b577d64ce63b39e565261770058e484cc80e00b`.
A separate standard-STFT copy preserves all 73 learned/other parent initializers
and 1,007 downstream nodes, with the same interface. On this one recording it
scores 0.7288501596423501: closer to GPU (error .0003801), farther from CPU
(error .0005785). It is not adopted; error cancellation on an outcome-selected
recording cannot choose a release. Probe summary SHA
`3df8bf28dd0d8dc67c85853d15501c3e2a30743aacb9a089c6a176965e4ffdf1`.
Canonical details and limits: `docs/ARTIFACTNET_ONNX_v1.2.md`.

The original full v2 validator is still live (246 comparisons at this check,
one tolerance failure, zero raw-0.5 flips), as is the original main queue
(six complete; CLAM 1,607 scored without errors; FST next). No active source,
original score, frozen input, weight or acceptance policy changed. Next priorities:
define and verify reference-consistent DeepFense native-rate input/codec handling
and full re-evaluation; validate any ONNX replacement against the existing
metadata-selected source/boundary sample and then the full scored reference,
with CPU/GPU conditions explicit. Remaining main/transport/final-result gates
are unchanged.

Eleven new diagnostic/STFT tests: export environment 271 passed; both original
environments 265 passed with four reported skips (three individual tests plus
the three-test ONNX-only module). Slow real-model test remains separately
measured. LaTeX v9 retains nine pages and now reports the tolerance failure
and decomposition. ZIP 28,702 bytes, SHA
`9852e18424a05d38cc403299b1a70b684e47c797ad3b2179937bb33c3df91bb7`;
two builds identical, both fresh compilations match every text/raster page.
Changed pages 7–9 visually inspected; PDF SHA
`64e138ed5a324bc6ff8dd898416adf04ddfcee30a5a3b9135a3078209489e508`.
No upload or license decision occurred.

**2026-09-05 16:05: ONNX v2 full-parity acceptance failure observed.**
At 205 saved comparisons, `mom_riffusion` recording
`fe2d32060d3efaca5d838660` exceeds the fixed 1e-3 tolerance:
original GPU 0.7292302923969539, ONNX CPU 0.7281898997184076,
absolute difference 0.0010403926785462758; zero raw-0.5 flips so far.
Model and all pinned source/input hashes match. Root cause is not yet
established; same-input fresh CPU reference for this recording remains to
be measured. The original full validator continues unchanged; neither its
threshold nor the original benchmark is modified. No full-parity release
claim is valid for this candidate under the existing policy. Canonical:
`docs/ARTIFACTNET_ONNX_v1.2.md`. P(AI)-only export/standalone execution works,
but that does not by itself establish numerical equivalence or better accuracy.

**2026-09-05 16:00 priority finding: DeepFense input-path interpretation hold.**
The original run completed, but is not an unqualified checkpoint-quality result.
A fixed metadata-first nine-file CPU diagnostic reproduced original GPU scores
within 1.20e-7. Direct upstream input flipped 7/8 comparable cases (AAC input
error retained). A second run on the byte-identical selection held the common
float32/44.1-kHz waveform fixed and changed only the final resampler: **8/9
flips**, maximum score shift 0.9990168569770503; exact model-input replay had
zero difference on all nine. All six real examples and both native AI examples
changed to real, so this is not universally improved accuracy. Control summary
SHA-256 `bd9c7c30b5d318f1b04f26463c306a993fb2945fd0c61cfffbe0c38086bd5d38`.
Canonical: `docs/DEEPFENSE_INPUT_PATH_AUDIT_v1.2.md`; original run has a separate
`INTERPRETATION_HOLD.md` without modifying its scores, identity or sources.
**Before final comparison:** define/pin a reference-consistent input condition
and explicit unsupported-codec policy, fully re-evaluate all 2,579 entries,
reconcile transport/reporting, and retain original and corrected conditions.
Do not blanket-swap other models' resamplers or tune against these nine labels.
The current common 44.1-kHz stage is not always direct upstream preprocessing.

**Six original runs complete; CLAM running.** Third-party fakeprint LR
(`deezer_ismir`) scored all 2,579 entries, no errors. Summary SHA-256
`50f10b29b709c0d3550138f0f02529353c8bde85fcac26bbe17cc71bfd8a584c`,
identity `2b0208c6ca7497ad1373956c4771a135dd9629df0eeb0ef42b1845ca4ede777e`.
Full row/membership-validated 2,000-bootstrap checkpoint report:
`out/v1.2_checkpoint_deezer_ismir_rc2_260905`, checkpoint SHA-256
`dd815c42e77e31098781ca311dfba765c3327d77164f97632f309799a3a12f32`.
Raw .5 legacy TP856/FN488/FP103/TN717, F1 0.743378 [0.730501,0.755615],
FPR 12.561% [10.487,14.915], AUROC 0.779803; Suno 200/200,
Udio 133/200=66.5% [59.701,72.947], non-demo F1 0.783267.
This is the third-party checkpoint, not the official Deezer service. The
original queue is live on CLAM (PID 2984698), with FST next. Two original runs
plus the DeepFense input-path re-evaluation, transport, full ONNX parity and
final statistical/publication artifacts remain. No upload/license decision.

**Diagnostic and paper verification:** eight new diagnostic tests; export
environment 260 passed, original environments 257 passed + ONNX-only 3 skips,
slow real-model test separately measured. The real nine-file control's record
hashes, comparison arithmetic, replay waveform identity and all 111 source
hashes were independently rechecked. LaTeX v8 now describes the finding and
hold: nine pages, source ZIP 28,419 bytes, SHA-256
`956f5915988b1cc3702814044c305f1ab11a30259eacfef88609d5a80744b352`.
Repeat ZIP identical; both fresh compilations match all nine text/raster pages.
Pages 1 and 6–9 visually inspected; main PDF SHA-256
`e529e3ab466e5a56096921e199d4e5a018c239c8737044bde75ecdf1311ea1f9`.
Five underfull warnings retained; no overfull or undefined citations.

**2026-09-05 15:44 latest: five full checkpoints complete.** DeepFense scored
all 2,579 rc2 entries with no execution failures. Summary SHA-256
`d0f40b5057ec3efb0f872efadca363ce4d856ca003243b4671305ddba9d152e6`,
identity `ed26b09136300e62d6be8c10c27f60aed2f316df27ccd8cfed1b0881ec12b9d3`.
The full-membership/hash-validated 2,000-replicate checkpoint report is under
`out/v1.2_checkpoint_deepfense_rc2_260905`, checkpoint SHA-256
`3be33787ef88396d2004cae14f3e2da869b601dfcc41fd4df0f6aaf3e7054c24`.
At raw 0.5: legacy TP948/FN396/FP797/TN23, F1 0.613791
(CI 0.608441–0.619002), FPR 97.195% (96.093–98.291%), AUROC 0.412003.
Suno 200/200, Udio 199/200 (98.454–100%); non-demo F1 0.692901.
Separate demos Lyria 11/11 and Stable 4/4. High native recall is not a useful
detector by itself. Read-only source checks confirm configured spoof class 0,
the two-class CE logits, frontend-owned normalization and four-second prefix/
repeat policy. The high false-positive rate's cause is not established; direct
upstream file loading uses librosa resampling, unlike the common benchmark
44.1-kHz/TorchAudio route. No score inversion, threshold change or new weight.
The original sequential queue has moved to third-party fakeprint LR
(`deezer_ismir`, PID 2910675); three full runs, transport and final artifacts remain.

**Transport reporting completed before transport detector scoring.**
`transport_statistics.py` now retains total float-relative shifts while adding
same-staging codec increments, quantization-only change, fixed-codec staging
change, and a within-recording four-view signed interaction. Comparing
`codec(PCM16)` only to float identity would include quantization in the codec
effect. Comparisons retain failed coverage, return null when all fail, and
report flip counts; interaction CIs are not differences of independent CIs.
Exact formulas are in `docs/STATISTICS_v1.2.md`. No cached waveform, selection,
inference runner, adapter, weight or completed prediction was changed.
Thirteen new arithmetic/coverage/CLI tests; export environment 252 passed,
both original environments 249 passed + three ONNX-only skips; the slow real
model test stays separately measured. Active main code (30 checks) and ONNX
source/input (51 checks) hashes still match. Full ONNX parity is live, not done.
LaTeX v7 incorporates the contrast definitions: eight pages, source ZIP 27,739
bytes, SHA-256 `2a20231f0aef2ff8d46b7fd2eb61b8bcbfbfd6f64448a476c31ce5f0a67d5ad9`;
repeat ZIP identical, both fresh compilations match every text/raster page,
changed pages 4–8 visually inspected. Main PDF SHA-256
`457f5835351dea7ef3e620933fc97afaa3e64618346aef8ab2dd2915fdb8b642`.
The paper remains a first-checkpoint draft. No publication/license decision.

**2026-09-05 15:37 manuscript source audit complete for this draft.**
AIME is now cited as ICASSP 2025 with its DOI. Original AST architecture and
the evaluated 60-second fine-tune have separate citations. DeepFense is named
as `FakeMusicCaps_EAT_Nes2Net_NoAug_Seed42`, with its pinned configuration and
actual class-0 score direction distinguished from AST's class 1. AST/lofcz
model-card URLs now pin evaluated revisions. The bounded claim/source map is
`docs/PAPER_SOURCE_AUDIT_v1.2.md`; upstream training declarations are not
independently verified training histories. Inference sources remain unchanged.
LaTeX v6 is still eight pages and a first-checkpoint draft. Source ZIP 27,541
bytes, SHA-256 `0e5f39c0d890312b37b8a0c8f1039a873feb2ca3ea8ce52fb4752f67112f988d`;
independent repeat is byte-identical, both fresh extractions compile and match
all page text/raster hashes, and changed pages 2/4/7/8 were visually inspected.
All 27 paper-source/render tests pass; five underfull warnings remain disclosed.
Full evaluation remains 4/8 complete with DeepFense running. Full CPU ONNX
parity is also live (87 saved comparisons at this checkpoint, maximum error
0.0006421210567838509, zero threshold flips or tolerance failures); this is
not a full-parity result. No publication or license decision occurred.

**2026-09-05 15:27 latest state: four full checkpoints complete.** AST scored
all 2,579 rc2 entries with no failures; summary SHA-256
`f2454382bd596f5bfda2a3a0602db8c11ab1ad400a4fc174ca72dff1c316b484`,
identity `92d7a4bbcdce53b76d6657fb540d93872e115029e488c9af742570280faf0d00`.
The 2,000-replicate report at `out/v1.2_checkpoint_ast_60s_rc2_260905` validates
all original row hashes and release membership. Raw 0.5: legacy F1 0.836323
(CI 0.824757–0.848488), TPR 83.259%, FPR 25.976% (23.169–28.851%);
Suno 199/200=99.5% (98.437–100%), Udio 199/200=99.5% (98.446–100%).
Non-demo F1 0.873345. These native cohorts lack real controls; high recall
with a high legacy false-positive rate is not universal superiority.
The original sequential queue has advanced to DeepFense; four full runs,
transport, complete result reproduction and the final paper remain.

External asset preparation is now executable without importing model code.
`verify_external_assets.py` checks one public prediction identity against actual
HF files, source revision/Python inventory and local checkpoint bytes. All seven
external models passed: HF files reused the existing cache; CLAM/FST source,
CLAM head, two FST stages and BeatThis were freshly acquired from official
locations and matched reference hashes. Actual Drive confirmation HTML was
rejected with exit 1; a fresh six-source checker copy passed with `python -B -S`.
Evidence `out/v1.2_external_asset_check_260905/summary.json`, SHA-256
`2f0f766acb4b8f07f35c6bc10eb928012619e2601288b222d0f01376d223c078`.
This is asset preparation, not fresh model inference or a finished source bundle.
Tests: export environment 239 passed; original environments 236 passed,
three ONNX-only skips, one separately measured slow test deselected.
Full ONNX parity remains live and incomplete. No upload or license decision.
Older progress entries below describe earlier states, not current completion.

**ONNX source-smoke and candidate packaging completed, 2026-09-05.** All 34
score-blind source/boundary recordings were attempted: 33 comparable original
scores, maximum absolute error 0.0005089868448281409, zero raw-0.5 flips, and
one original failure retained as non-comparable. Fresh CPU reference comparisons
also passed. Summary SHA-256
`a60b6f459b6fe1827581d6c7f2923b55d3327ac19a9bbc1337de49ce37a9bfa8`.
Full 2,579-entry CPU parity has now started under
`out/v1.2_artifactnet_onnx_full_v2_260905` and remains incomplete.
The source-smoke-labelled candidate ZIP is 29,934,792 bytes, SHA-256
`95722b47f554c540a7541616ea42ee6fd2cdb2cb9eefa98e7e9ebbee7350297a`,
with an identical independent repeat. The builder pins the tested runtime,
original scores and selection; the standalone verifier checks its exact public
inventory and comparison arithmetic without pretending to execute the model.
Tests: export environment 223 passed; both original environments 220 passed,
three ONNX-only skips, one separately measured slow test deselected. Owner
decision on the new weights' license remains pending. Nothing was uploaded.
Fresh ZIP extraction passed a site-disabled verifier and actual one-recording
Torch-free inference with an exact candidate-score match; private proof SHA-256
`5784a9933b6860b576371be9b9082a28b64c32620be869dd28cbb5406a3a839c`.
LaTeX v5 now records these evidence limits: eight pages, source ZIP 26,609 bytes,
SHA-256 `31cbadd031c47970c6ac36c508c56294b73d06e4cae4522e839a1cd2695a8ebb`.
The repeat ZIP is identical, both fresh compilations match all eight text/raster
pages, and the changed page 6 was visually checked. Main PDF SHA-256
`9635df183922e714f4d06368ef67350e4481dddb1787f22d1dd24acefc511112`.

**Latest: P(AI)-only ONNX candidate implemented, 2026-09-05.** The current
UNet/bigset CNN are exported without retraining or changing the main benchmark.
The single-file v2 candidate is 33,514,792 bytes, SHA-256
`590d992b0839ac49900f6eed820cd2c9e68af76304af932dcff37e5bfd7b4621`,
under `out/v1.2_artifactnet_onnx_candidate_v2_260905`. Its one output includes
song-wide RMS aggregation; the dynamic input axis is 1–15 chunks of ONE song,
not independent songs. Joint residual/H/P Mel normalization uses a global max
over all chunks, so independent chunk or component calls are not equivalent.
Decoder/resampler and LGBM rescue remain outside the raw graph. See
`docs/ARTIFACTNET_ONNX_v1.2.md` for commands, contract and evidence limits.

Candidate 1 is preserved as failed: the TorchScript exporter cast the FP64 RMS
sum to FP32 before adding epsilon, producing `1.0000000119162602` on one real
recording. Keeping a one-dimensional FP64 sum and typed tensor epsilon fixes
the exported denominator. The actual-ONNX regression passes on the fix and
rejects the preserved old aggregation; the formerly failing real recording
now returns `0.9999999936970877` against original `0.9999999992197944`.
No output clamp, relaxed tolerance, threshold adjustment or data change was used.

A fresh Torch-free environment and fresh minimal-source/model/input copy
outside the checkout actually reproduced one real recording's v2 score exactly
(`0.9755876587251223`). Runtime proof SHA-256
`7091529be0297d977889891b0df237cf93ef6e1754487448596da038e3706c49`,
`out/v1.2_artifactnet_onnx_runtime_portability_260905/summary.local.json`.
All 194 unit tests pass in the export environment; both original environments
pass 193 with the ONNX-only test skipped, plus the separately measured slow test.
The 34-recording score-blind source/boundary smoke is running under its own CPU
process in `out/v1.2_artifactnet_onnx_smoke_v2_260905`. Inspect its live process
and summary before claiming completion; **full rc2 parity remains required**
under the predeclared 1e-3 absolute-score/zero raw-0.5-flip policy. The original
failed benchmark row stays non-comparable, even if this raw graph can score it.
No ONNX or paper upload occurred. Main full evaluation remains 3/8 completed,
with AST still running; five full runs, transport and final artifacts remain.

**Third full checkpoint completed, 2026-09-05.** SpecTTTra beta 5s scored all
2,579 entries with no failures. Summary SHA-256
`5074de19a607a90f91be8dcb0795f428ed7c586bd06a035d8025ae1a8b8f6680`;
identity `90e3cd8241fcedb4c789ee487373f6ca047e6551c8008cd0abcbcee8a1151211`.
The hash-bound 2,000-replicate report at
`out/v1.2_checkpoint_spectttra_beta5s_rc2_260905` has checkpoint SHA-256
`d09bb3d03e05bb8c6ebd1fc36d7e57dc9a8162db93108780431b34cb32bc8d62`.
Raw 0.5: legacy F1 0.556391 (CI 0.538926–0.575744), TPR 41.2946%,
FPR 11.7073%; Suno 84/200=42% (35.32–48.50%), Udio 36/200=18%
(12.70–23.30%). Non-demo F1 0.536779; separate demos Lyria 0/11, Stable 1/4.
The original sequential process has started AST; five full runs plus transport
and final reports remain. This is not a duration-only ablation or final ranking.

**Local audio binding implementation:** `bind_local_audio.py` now validates a
private ID-to-path map against public primary/native metadata and requires every
encoded-file hash to match before writing a separate runnable release. No
downloads, audio decoding, model execution, or edits to the frozen release.
Missing/mismatching inputs retain explicit private audits without runnable
manifests. All 31 new boundary tests passed; **167 tests passed in both**
`.venv-eval` and RTX_ENV, with the real-weight test separately measured.
Actual rc2 byte-binding completed: all 2,579 primary files (25,668,835,924 bytes)
and three variants (27,259,100 bytes) match; the four public JSON copies are
byte-identical, and the runner's full primary identity/path projection matches
the original local release. This used the author's existing local files, not
independent public reacquisition. Evidence `out/v1.2_local_audio_binding_rc2_260905`,
summary SHA-256 `9155b50ff5761ba236f86074deb8f1043ed78493e856a75c346010f1fa29650d`.
The copied release also passed site-disabled metadata verification. See
`docs/LOCAL_AUDIO_BINDING_v1.2.md` for the private map contract and limits.

The LaTeX draft now describes local byte binding and separates raw ONNX from
rescue-stack reproducibility. The v4 source ZIP is 26,442 bytes, SHA-256
`5b4ef6cb766a4c2a920baaffb7e69897e35728933ea18df94c484d2408a4fa3d`,
under `out/v1.2_latex_source_draft_v4_260905`; a second build is byte-identical.
Both fresh extractions actually compile and match the reference text and
raster hashes for all eight pages. Current main.pdf SHA-256
`32ecf2acdfae7188d9d3f4b4591ba0556d79baa73d0b128eeab0a0392a766bbc`.
This remains a draft with the first-checkpoint table, not a full leaderboard.

**User-proposed raw ONNX release:** raw `P(AI)` can be reproduced by an equivalent
UNet+CNN export plus exact preprocessing/sliding/RMS aggregation; it does not
require distributing the LGBM rescue model. Existing single-output export code
uses an older CNN, and the legacy adapter uses a different seven-chunk median.
New latest-checkpoint export/parity/publication is unperformed. The bounded
three-file asset absence audit below is not a mandate to expose three original
checkpoints. Operating rescue-stack reproduction remains separate. No active
inference code, weights, thresholds, or queue were changed for this proposal.

**LaTeX source portability and asset-access audit, 2026-09-05.** The manuscript
now has an explicit eight-file source inventory and a deterministic ZIP builder.
The latest draft ZIP (`out/v1.2_latex_source_draft_v3_260905`) is 26,163 bytes,
SHA-256 `09a519c16895d1d8603531836e1143d1cace5feb7d07b06071114bf479f852bc`.
It was freshly extracted and actually compiled using Tectonic 0.17.0's existing
cache, with custom TeX search paths removed. All eight pages have identical
layout text and raster hashes to the manuscript PDF; the emitted dependency
trace matches all declared inputs. `.bib` and `.bbl` are included, while the
main PDF, logs and evidence stay outside the source ZIP. This is local compiler
portability, not arXiv-server or scientific-completion evidence. See
`docs/LATEX_SOURCE_REPRODUCING_v1.2.md`. Tests: **136 passed in both environments**,
one actual-weight test remains separately measured. Active inference is unchanged.

The declared public `intrect/artifactnet` repository was inspected without a
token at revision `7c9b753a9d006b48e4bfaf85bf0157e135f4aad4`. Its four files
contain only two v9.4 ONNX assets plus text metadata; none matches the three
exact CNN/UNet/LGBM checkpoints used by rc2. This is a bounded repository-revision
finding, not proof of absence from every host. Audit SHA-256:
`6ee97e9681a0040d573734f79635f01012637b46d49d6bafd1f51a086d03f144`,
`out/v1.2_artifactnet_public_asset_audit_260905/audit.json`. The paper now cites
this public revision and explicitly states the remaining asset-acquisition gap.
No upload or substitution was made. Current PDF SHA-256:
`e7bf3d13bdaf1c3d44a95840b651323deb370df840984b1c1791b320d461c821`.
The inference-source package and final all-model/transport results are still
required; draft ZIP success does not replace them.

**Second full checkpoint completed, 2026-09-05 14:20 KST.** SpecTTTra alpha
scored all 2,579 rc2 entries with no execution failures. Summary SHA-256:
`ee55e75fe48c18fed0dad618d08c64f18178b9be41fca73ccf6c6347fc988cac`.
The same sequential queue has moved to SpecTTTra beta 5s; two of eight full
runs are complete. The 2,000-replicate single-model report is
`out/v1.2_checkpoint_spectttra_rc2_260905`, checkpoint SHA-256
`0d8967a76fa51d9a34df2352ff1f72572f99b462c3d1ed28e05dc836530fefbb`.
At raw 0.5: legacy F1 0.7716277, TPR 70.0149%, FPR 18.7805%; native Suno
169/200 = 84.5% (cluster CI 79.40–89.40%), Udio 108/200 = 54.0%
(47.42–60.87%). Non-demo mixture F1 0.7817715; provider demos separately
Lyria 5/11 and Stable Audio 1/4. This source-specific inversion relative to
ArtifactNet precludes claiming universal superiority from legacy F1. No paired
all-model ranking, causal codec attribution, or completed release is claimed.
The LaTeX draft still labels its table as the first ArtifactNet checkpoint;
the full comparison must replace that interim section once all models finish.

**Public-score archive verification, 2026-09-05.** A 40-payload-file results
bundle now includes the actual public scores, nine reference statistical files,
cleared notebook, minimum reporting source and dependency locks. It requires
completed runs and successful hash-bound reproduction; full mode pins rc2,
while explicit historical smoke cannot become full output. ZIP verification
rejects unsafe/duplicate paths, non-regular files, unlisted payloads and changed
hashes. Two independent historical smoke builds are byte-identical (618,202
bytes, SHA-256 `dd0e680262c97fbc8b4daab12c0e574811f31dff4bb9f2bf507081ede16a763c`).
The ZIP was freshly extracted and its packaged verifier and six-cell notebook
actually ran in the isolated CPU results environment, reproducing all nine JSON
files exactly. Evidence: `out/v1.2_results_zip_portability_smoke_260905`; commands
and limits: `docs/RESULTS_REPRODUCING_v1.2.md`. This is historical software wiring,
not the final full rc2 export or an inference archive. The original main-model
queue remains live; no active inference code/assets or frozen rc2 data changed.
All **112 unit tests passed in both .venv-eval and RTX_ENV**, with one separately
measured real-weight test deselected. The prior rc2 metadata bundle still verifies.
The LaTeX reproducibility section now documents actual ZIP extraction/saved-score
checks, with smoke versus full-inference scope explicit. The eight-page draft
compiled with resolved references and no overfull boxes; table/prose layout on
page 6 was inspected. PDF SHA-256:
`bfa5412852a139ed59f225fd0f551dcbdc75b8f0b2edb445bfc9ccb53364adcf`.

**Latest checkpoint: first full model completed, 2026-09-05 14:04 KST.**
ArtifactNet attempted all 2,579 rc2 entries; 2,578 scored and the known short-input
error remained explicit. All 2,579 prediction hashes/membership checks passed.
`out/v1.2_full_rc2_260905/artifactnet/summary.json` SHA-256 is
`db4a1c20e3c80a9b9e8bf14d6bef760cd1a2390fde87b383d26ef06329cfea09`.
The same sequential queue moved on to SpecTTTra alpha; no restart or source/asset
change was made to active inference. Seven models remain; inspect live status.

- `out/v1.2_checkpoint_artifactnet_rc2_260905` contains the separate 2,000-replicate
  single-model report. Raw 0.5: legacy F1 0.9887218, TPR 97.9151%, FPR 0.2439%;
  Suno v5.5 10/200 = 5.0% (cluster CI 1.94–8.91%), Udio 200/200; demos 0/11
  Lyria and 0/4 Stable Audio. Existing CNN .225/rescue .02: Suno 15/200 = 7.5%.
  Non-demo mixture F1 is 0.9327217 raw / 0.9389614 stack. These are not all-model
  leaderboard results. Suno200 are all AAC and Udio200 all MP3: complete
  source/transport association, not a generator-version-only causal contrast.
- User explicitly reconfirmed **LaTeX**. `main.tex` now includes a labelled first
  checkpoint section with generated `checkpoint_{numbers,table,evidence}` files,
  the source/transport limitation and degenerate Udio bootstrap qualification.
  Seven-page PDF compiled with resolved references; full comparison stays pending.
- Public prediction exporter strips absolute paths and private traceback text,
  preserves probabilities/failures and checkpoint/row hashes, and binds the
  exact completed runs/statistics. `report.generate_reports` is shared by local
  and public-score reporting; every original smoke output and summary stayed
  byte-identical after refactoring. Original report source is preserved beside
  the v2 smoke artifacts. Active inference sources were not refactored.
- The new six-cell `v1.2_results_reproduction.ipynb` executed in fresh `.venv-results`
  (NumPy 2.1.3, notebook dependencies only, pip check passed). It requires all nine
  statistical JSON files to match, including 28 model comparisons. Historical
  smoke export/reproduction and 23-file fresh-copy portability all passed;
  final rc2 reproduction still awaits all eight complete runs. Evidence:
  `out/v1.2_public_results_reproduction_smoke_260905`,
  `out/v1.2_results_notebook_smoke_260905`, `out/v1.2_results_portability_smoke_260905`.
- **95 unit tests passed in both RTX_ENV and .venv-eval**, one real-weight test
  deselected (actual eight-model smoke remains separate evidence). Full transport
  inference/statistics, final comparison/notebook, public-safe result/inference
  archive, and submission-ready LaTeX/arXiv package are still required.

**Latest verification checkpoint: 13:49 KST, 2026-09-05.** Full rc2 inference is
still live in the same sequential queue; ArtifactNet has attempted 1,473/2,579
entries (one preserved intrinsic short-input error), and no full-model summary
exists yet. Do not infer accuracy from this progress count or historical smoke.

- `paper/render_results.py` requires all eight completed hash-bound reports and
  2,000 bootstrap replicates for final output. Explicit historical smoke wiring
  is watermarked and cannot write under `paper/`. The v4 wiring outputs and a
  separate LaTeX table-layout compile passed; external legends no longer cover
  points, and metadata-defined source rows remain present even if all models
  fail. The recompiled methods draft is now seven pages, with exact file-URL
  identity and the actual unequal adapter input budgets documented.
- `out/v1.2_metadata_bundle_rc2_v3_260905/artifactbench-1.2-rc2-metadata.zip`
  is a local metadata-only candidate, 491,177 bytes, SHA-256
  `ec26169cb3c46d18083b10b46bf4fcd0c05dd196d4abf6da8bf44a13d7270bfe`.
  Two independent builds are byte-identical. It includes 20 allowlisted payload
  files plus their inventory, with no audio/weights/predictions/private mappings.
- `out/v1.2_metadata_portability_rc2_v3_260905` proves actual fresh extraction,
  stdlib verification without site imports/PYTHONPATH, and six notebook cells
  executing from the extracted source root. It reuses the isolated notebook
  dependency environment; this is not a second fresh-install claim. The kernel
  check now compares executable paths without resolving venv symlinks, checks
  `sys.prefix`, and verifies that the metadata verifier imports from that root.
- All **84 unit tests passed in both RTX_ENV and .venv-eval**, one real-weight
  test deselected (the eight-model actual smoke is separate evidence). Active
  inference code/assets and frozen rc2 public hashes are unchanged. Final full
  statistics, detector transport results, results notebook, full-model portable
  package, and submission-ready paper/arXiv archive remain pending.

**Current: rc2 frozen and eight-model full run restarted, 13:32 KST.**
`out/v1.2_frozen_rc2_260905` has public manifest
`feab7c4c3d037919fd784dc40e46199470088abde532c5181e34573f20dceefd`.
Current byte hashes and previously successful same-file decode evidence cover
2,579 primaries + 3 verified native MP3/AAC views. Full duplicate search/review
rerun: same 9 candidates and 3/3/3 verdicts; 2,576 recording groups / 2,456
dependence clusters; no label/known lineage exposure conflicts. All 400 native
files have exact clip/song/original URL evidence, with provider recording IDs
now separate from inventory sampling-slot IDs. `out/v1.2_full_rc2_260905` is the
only current full evaluation; rc1's 1,477 partial attempts are preserved and
must not be completed or reused. Do not mutate active inference source/assets.

Controlled PCM caches are hash-verified and rebound without re-encoding in
`out/v1.2_transports_rc2_260905`: 50×8 + 3×2 = 406 waveforms. Source history is
unknown before the supplied PCM/FLAC input; do not call it never-lossy audio.
`run_transports.py` and `transport_statistics.py` are implemented, but real
transport detector evaluation is pending completed primary runs. Metadata
notebook passed six cells in an isolated new environment for rc2, and paper
construction macros/PDF were regenerated. Current unit suite: 68 passed,
one real-weight test deselected (eight-model actual smoke done separately).
Model factory now permits explicit repository/BeatThis location environment
variables. All eight default asset dictionaries were compared to prior runs
and are exactly unchanged; only location configurability/source hash changed.

**13:20 KST correction — rc1 is superseded, full run stopped and preserved.**
Native waveform checks found the Udio parent media ID conflates numbered
outputs. Five primary files differ from the page-declared song path; seven of
ten candidate native pairs are not verified same-recording transports. See
`docs/UDIO_IDENTITY_CORRECTION_v1.2.md`. Build rc2 by exact complete-URL matching,
rerun validation/grouping, then restart all eight models. No detector scores
were used for this correction. The checkpoints below describe historical rc1
work, not current release validity. Newly implemented metadata notebook passed
six actual kernel cells; controlled waveform preparation passed 50 exact float
identities and all 400 variants. These need an explicit rc2 identity update.

## Objective and deliverables

Build ArtifactBench 1.2 including current generator data, run reproducible
comparisons, and produce an English benchmark paper suitable for arXiv with
LaTeX source, compiled PDF, bibliography, figures, and a submission source bundle.
Dataset card, notebook, paper tables, and release manifest must share one
version/checksum/count definition. Actual arXiv submission is a separate action.

Related existing issues: AUD-1208 (manifest/provenance), AUD-1206 (runner and
baselines), AUD-1205 (dataset card/notebook/paper), AUD-1207 (audio distribution).
Their broader publication/CI checklists are not automatically completion claims.

## Verified construction checkpoint (2026-09-05)

Latest checkpoint (13:00 KST): evaluation data are frozen as **1.2-rc1** in
`out/v1.2_frozen_260905`. All 2,579 primary and ten additional native files
passed full decode/finite-PCM/hash validation. Nine fingerprint candidates were
reviewed against complete waveforms: three identical/excerpt pairs, three shared
Udio edit/extension pairs, three not confirmed under the declared criterion.
Final units: 2,579 evaluation entries / 2,576 identified recording groups /
2,456 dependence clusters. The preliminary duplicate report's 2,574 includes
extensions; **do not use it as the final recording count**.

Public manifest SHA-256:
`f5d3a85e2145f1b377604302916e2da8e2ef4e33b8904f30f500a59b31002099`.
Five pinned local train/val/selection manifests had zero unresolved path-stem
candidates. This is not audio-level all-training independence or external
baseline unseen status. `docs/DATASET_CARD_v1.2.md` now records the limitations.
Metadata hash/count/privacy checks passed in dependency-free `.venv-verify`:
`python -m artifactbench.v12.verify_release --release out/v1.2_frozen_260905`.

All eight models loaded their real fixed weights and completed **32/32 real
source-balanced smoke inferences each**, after fixing the DeepFense adapter.
Seven successful smoke directories are `out/v1.2_smoke_260905/<model>`;
DeepFense's successful rerun is `out/v1.2_smoke_v2_260905/deepfense`.
Its initial five execution errors are preserved, with an adapter snapshot that
matches the failed run's recorded SHA. These are smoke measurements, not the
full leaderboard. **Full sequential eight-model runs started** in
`out/v1.2_full_260905`; each model has a log and per-entry records. Check actual
process/summary status before resuming. Do not mutate `v12/run.py`, its shared
helpers, the active adapter, its external source, or weights during a run.

DeepFense 0.2.2's CE `get_score()` returns only the bonafide logit despite its
LLR docstring; `out['logits']` plus two-class softmax and config spoof=0 is the
verified probability mapping. Shared-logit-shift regression guards it. Its
local EAT snapshot also requires transitive relative-import cache registration
(`eat_model -> model_core`). Actual EAT+Nes2Net size is 309,177,378 parameters.
FST now exposes execution failures instead of swallowing them / returning 0.5;
normal no-downbeat fallback and PCM16 beat staging are explicitly recorded.
Each model's HF assets are loaded offline from fixed local snapshots. Run
identity now snapshots actual Python source and records loaded parameter counts.

HF asset collection completed in `out/v1.2_baseline_assets_v2_260905`; the first
attempt failed from concurrent `snapshot_download` calls racing over tqdm's
global lock. Repository-level sequential download, still with two file workers,
completed all nine repos. Original output and source snapshots are preserved.
The paper remains a six-page methods draft, updated with generated rc1 counts;
compile and visual checks passed on pages 1/3. Codex and Claude memory copies
were synchronized and verified. Remaining: full predictions, bootstrap/statistics,
controlled/native codec scores, portable model quickstart/notebook and arXiv bundle.

The earlier construction notes below are historical checkpoints, not current
run-status claims:

Additional verified implementation checkpoint: **54 tests passed** in both
RTX_ENV and `.venv-eval` (one real-weight test deselected; actual eight-model
smoke was run separately). `statistics.py` now keeps cross-source clusters
together using connected label/source strata; exact weighted AUROC agrees with
sklearn, failed rows stay in all-attempt denominators, and identical models
produce zero paired differences. See `docs/STATISTICS_v1.2.md` for the disclosed
implementation clarification. `v12/report.py` successfully processed all eight
real smoke runs with 20 test-only replicates in `out/v1.2_report_smoke_260905`;
this is **report wiring evidence, not full-result uncertainty**. Full reports
require completed summaries and 2,000 replicates. Current production ArtifactNet
loaded parameter count is 4,027,490 (UNet 3,603,937 + CNN 423,553), replacing the
old adapter's approximate 4.2M when preparing the final model table.

- `docs/PROTOCOL_v1.2.md` revision 2 fixes selection and reporting before new
  detector scoring. Udio's 2026 creation-date restriction is a documented
  metadata-only amendment, not a score-based adjustment.
- Native corpus inventory: 30,213 variants / 28,488 recordings. Known local
  training/selection creator exclusions leave 6,688 Suno and 413 Udio candidates.
- Public-page survey completed: Suno 1,200 (1,006 version-labelled, 186 unknown,
  eight explicit errors); 426 are v5.5. Udio 271 exact CDN/page joins, all version
  unknown; 202 created in 2026. Collection date is demonstrably not model version
  or creation date.
- Creator-enriched selection: 200 Suno v5.5 recordings / 122 creator IDs, 200 Udio
  recordings / 172 creator IDs. Add legacy 2,164 and official demos 15 for a
  **2,579-recording validation target**, not yet the final released manifest.
  Canonical selection: `out/v1.2_selection_260905/selection.local.json` and its
  input hashes / exclusion audit. No contemporary detector results exist yet.
- Official demos: Lyria 3.5 eleven tracks and Stable Audio 3.0 four tracks.
  All 15 passed full decode/hash/finite-PCM validation. Keep separately reported
  provider-selected examples; SA3 model size is unknown. Raw audio is local only.
- Full 2,579-recording validation is running in `out/v1.2_validation_260905`.
  Complete it, audit exact/perceptual duplication and lineages, then finalize
  public metadata + local mapping. Do not call this release complete early.
- `artifactbench/v12/audio.py` implements float identity, PCM16-only, and three
  lossy encoders with both float and PCM16 staging; it preserves the old helper.
  Signal-level tests pass. Detector-level controlled-codec results remain pending.
- `paper/main.tex` and `paper/refs.bib` produce a six-page **methods draft** PDF.
  Saved historical rows generate the historical table/CDF. All references resolve;
  pages 1/3/4/5 were visually inspected. There are minor underfull boxes but no
  overfull boxes. It is not submission-ready and does not claim v1.2 model scores.
- Tectonic 0.17.0 installed under `.tools/` after matching the official asset
  SHA-256. `.venv-eval` is a separate Python3.12 / torch2.8 CUDA128 environment,
  dependencies in `requirements-v1.2.{in,lock}`, SONICS pinned to revision
  `9156ffad151f797c71556923c4a02fa01fa8fc91`. Both RTX_ENV and .venv-eval pass
  **34 tests**, with one real-weight slow test deselected. CUDA/imports verified.
- HF baseline assets are being cached at fixed revisions via
  `artifactbench.v12.fetch_baselines`; asset hashes go in
  `out/v1.2_baseline_assets_260905`. Loading/scoring all eight checkpoints remains
  to be verified. The fakeprint LR baseline is the third-party `lofcz` checkpoint,
  not the official Deezer service; its display name and the draft say so.
- The first Suno parser missed records after length-delimited lyric text.
  Original pages and the faulty extractor snapshot are preserved; the corrected
  UTF-8 byte-length parser reprocessed them in a separate output directory.
  Six final metadata errors are conflicting duplicated clip objects; two are
  HTTP404. None was filled with an inferred version.

Next: finish data audit; freeze release; implement/run failure-aware per-track
evaluation with pinned assets and per-track RNG seeds; controlled codec experiment;
bootstrap/statistical reports; regenerate final paper/card/notebook; fresh-env
quickstart and arXiv bundle. Actual submission remains separate.

## Execution plan

1. **Inventory and freeze the protocol.** Inspect local corpora, actual generator
   version evidence, collection dates, source/creator groups, transport variants,
   and all known training/model-selection exposure. Survey primary sources for
   current generator releases and related benchmarks. Pin a sampling policy
   before inspecting detector scores on the new selection.
2. **Build and validate 1.2.** Retain a legacy comparison partition and add current
   generator/transport cells. Enrich missing version evidence or acquire verified
   samples. Validate actual decode, content signatures, exact/perceptual duplicates,
   track and creator overlap, byte hashes, and model-specific exposure masks.
   Preserve exclusions and unavailable files explicitly. Produce public metadata
   and a local path mapping separately.
3. **Run the benchmark.** Evaluate the existing eight baseline adapters on the
   final selected data, with checkpoint/runtime/preprocessing identities and
   explicit failures. Report micro/macro metrics, per-source and per-version
   results, raw versus deployed operating points, fixed-threshold results,
   uncertainty, and paired transport effects. Add a PCM-float codec path so
   quantization and lossy encoding can be measured separately.
4. **Write from verified evidence.** Build tables/figures from saved predictions.
   Draft the benchmark paper around dataset provenance, temporal/generator shift,
   codec sensitivity, exposure limits, and reproducibility. Include related work,
   limitations, sampling bias, rights/access constraints, and prior ArtifactNet
   paper relationship. No unmeasured result or claimed universal generalization.
5. **Audit and package.** Execute the quickstart in an isolated environment,
   validate counts/checksums and paper numbers, compile LaTeX without missing
   references, inspect the PDF, and assemble the arXiv source bundle. Verify
   any requested publication separately. Finish with dual-memory synchronization.

## Initial measured findings

- Legacy corrected test: 2,164 selected tracks, 28 sources; 2,163 scored and one
  single-chunk ArtifactNet failure. FMA recovery fixed 120 HTML files masquerading
  as MP3 via an explicit original-audio mapping. Canonical evidence:
  `out/current_production_260905_recovered/EVALUATION.md`.
- Native CDN snapshot: 12,215 files / 10,490 tracks / 892 creator groups.
  Additional Suno delta: 17,998 files. Native training additions already exist
  in multiple ArtifactNet training manifests; the entire corpus is not unseen.
- Grabber databases contain 44,819 Suno and 68,262 Udio media records. All inspected
  version fields are missing (Suno model_version/model_name all null; Udio has
  neither field). Collection in August/September does not establish generator
  version. Do not label all these tracks as the latest model.
- Current official candidate releases verified on 2026-09-05: Suno v5.5,
  Stable Audio 3, Lyria 3.5. Each admitted recording still needs its own provenance.
- Local shell has no `grabber`, `pdflatex`, `latexmk`, or `tectonic` on PATH.
  Grabber source exists at `/home/unohee/dev/grabber`; inspect its own environment
  before claiming the collector is unavailable. A TeX build environment remains
  to be prepared.

## Protocol decisions to preserve

- Separate provider, model/version, recording, creator, and transport identities.
  Count generator families separately from historical source/version cells.
- Unknown versions remain unknown. A later crawl date is not a release label.
- Native MP3/AAC views of one recording are paired observations, not independent
  tracks. Bootstrap and data splits must respect recording/creator groups.
- Existing selection-exposed groups can support retrospective analysis, but do
  not become a newly sealed test set by renaming their split.
- Keep fixed operating points distinct from thresholds tuned on calibration data.
  No threshold search on the headline test results.
- Keep raw-detector metrics separate from rescue/AcoustID/codec-routing stacks.
- Model failures remain in coverage and all-attempt metrics; infrastructure errors
  are retried, not silently counted as model performance.
- Distribution starts with auditable metadata, hashes, and reconstruction tools;
  audio redistribution requires source-specific permission evidence.

## Primary-source starting points

- Suno v5.5: https://suno.com/blog/v5-5 (2026-03-26).
- Stable Audio 3: https://stability.ai/research/stable-audio-3
- Lyria 3.5: https://deepmind.google/models/lyria/
- SONICS: https://arxiv.org/abs/2408.14080
- arXiv TeX guidance: https://info.arxiv.org/help/submit_tex.html

The final paper needs full source verification and bibliographic details;
this list is an investigation starting point, not a finished bibliography.
- 2026-09-05 21:23 KST: FST transport benchmark finished 406/406, exit 0. Peak memory ~6.44 GiB with high-limit throttling but no max/OOM events. Transport statistics written to out/v1.2_fst_transport_statistics_260905 (statistics SHA 3920c725359f6a6ffbe08f9dae9c90430463d642efddedda45730d99a710900e). All 8 model transport runs complete; paper transport table/update remains.
- 2026-09-05: LaTeX v14 now includes completed 8-model transport table; independent source-package compile passed (10 pages, ZIP SHA de361af7fbb020d253e8241b97da8330fbfc284019ffd29a3d1cb52194e0736b). ONNX STFT qualifier was attempted against frozen source via Bubblewrap but failed asset identity due primary git/path binding; preserve as failure and do not promote ONNX.
- 2026-09-05: Frozen-source STFT ONNX qualifier completed 34/34; 33 comparable references had max GPU error 0.0005109871, max factory-CPU error 0.0002899710, zero tolerance failures and zero flips. One primary execution error retained. Full ONNX parity and package promotion still pending.
- 2026-09-05: Public results ZIP portability passed on fresh extraction (40 files, 6 notebook cells, all 2,579 attempts/model). Remaining gates are ONNX full parity/model release and arXiv source bundle; metadata and saved-score portability are complete.
- 2026-09-05: Full standard-STFT ONNX parity launch was safely stopped before first record at ~6.7 GiB cgroup memory (high throttling, no OOM/max). Competing host processes included ~5.8 GiB FMA precompute. No result inferred; schedule only after workload isolation.
- 2026-09-05: Full STFT parity launches stopped before first record under memory pressure; experimental lazy-import optimization was reverted to preserve primary source identity (audio.py hash matches identity). No ONNX result inferred.
- 2026-09-05: FMA precompute still active (~43k cached, ~6.1 GiB RSS); defer ONNX full parity until host workload isolation. Exact 48 kHz torchaudio resampling prevents a Torch-free parity shortcut.
- 2026-09-05: Added RELEASE_READINESS_v1.2.md as single handoff checklist; verified artifacts and remaining ONNX full-parity/licensing/arXiv gates are now explicit.
- 2026-09-05: README status corrected to current rc2 evidence (all 8 runs and portability complete); stale “FST running/evaluation pending” claims removed. ONNX full parity/publication remain pending.
- 2026-09-05: README and dataset card wording synchronized with completed transport/results portability; remaining gates are model-weight/ONNX/arXiv.
- 2026-09-05: Results reproduction documentation synchronized with verified rc2 archive/notebook portability and completed transport evaluation.
- 2026-09-05: README links the release-readiness handoff checklist.
- 2026-09-05: ONNX full STFT parity queued behind FMA PID 266771 in an OOM-safe service; light runner uses subprocess torchaudio resampling and frozen source. Await completion; no score inferred yet.

# Statistical implementation clarification

2026-09-05. This implements the existing 2,000-replicate, seed-260905 clustered
percentile-bootstrap protocol. Written after source-balanced execution smoke
tests and before full-result reporting. It changes no selection, checkpoint,
threshold, or detector result. Strata are derived from metadata, never scores.

An identified recording can occur in multiple legacy source cells. Independently
resampling each cell would split its dependence cluster and violate the stated
bootstrap unit. Therefore connect label/source cells when a dependence cluster
spans them, and bootstrap whole clusters within the resulting connected strata.
Ordinary cells remain individually stratified; connected cells move jointly.
Save the actual strata and cluster membership with every report. Individual cell
counts may vary within a connected stratum; all cluster members receive the same
replicate weight. Do not split cross-cell duplicates merely to fix cell counts.

Report legacy, contemporary-native, and the non-demo combination separately.
The combination is conditional on the benchmark mixture. Native-only has no real
labels, so FPR, binary AUROC, and two-class F1 are undefined there. Official demos
are separate counts, with descriptive Wilson intervals that do not remove
provider-selection bias or imply a probability sample of generator output.

Each model keeps all attempted rows for coverage and all-attempt correctness;
scored-only confusion matrices use successful rows. Percentile intervals use
2.5%/97.5% quantiles. Undefined metrics remain null, with the number of finite
bootstrap replicates shown. No failed row receives a fake probability.

Pairwise model comparisons are on each pair's common successful recordings,
with coverage and identical resampled clusters for both models. Differences are
model A minus model B, not independently bootstrapped confidence intervals
subtracted after the fact. These are descriptive multiple comparisons, not
unadjusted claims of statistical superiority. Repeat the non-demo analysis on
the manifest's deterministic representative-only recording view.

Bootstrap dependence correction does not correct creator/genre/popularity bias,
undisclosed external training overlap, unknown lineage, or researcher exposure.
No interval is evidence of universal generalization.

## Matched-staging transport contrasts

Implementation clarification, 2026-09-05, before any rc2 transport detector
inference. The eight already prepared views and score-blind 50-recording
selection do not change. Comparing every view only to float identity cannot
separate the combined PCM16-plus-codec path into its two increments.

For one recording, let `F` be the float identity score, `S` the PCM16-only score,
`C(F)` a codec score from float staging, and `C(S)` the same codec from PCM16.
`transport_statistics.py` keeps the original total shifts and additionally
reports these fixed contrasts for MP3, AAC and Opus at the existing bitrate:

| Quantity | Signed score contrast | Required successful views |
| --- | --- | --- |
| PCM16-only change | `S - F` | 2 |
| Codec increment from float | `C(F) - F` | 2 |
| Codec increment from PCM16 | `C(S) - S` | 2 |
| Staging change with codec fixed | `C(S) - C(F)` | 2 |
| Signed interaction | `[C(S) - S] - [C(F) - F]` | 4 |

The new JSON section is `controlled_contrasts`; `controlled` retains the total
shift of each view from float identity and the eight-view range. Every two-view
contrast has signed/absolute shifts and raw-0.5 flip rate, plus the observed
flip count. Counts are null when no pair is comparable, not misleading zeros.
Each contrast retains all attempted recordings for coverage. A failed view
only removes comparisons requiring that view; four-view interactions require
all four scores. No model failure receives a substitute probability.

The interaction is formed **within each recording before bootstrapping** using
the existing 2,000-replicate creator/lineage clustered scheme. Subtracting two
separately estimated confidence intervals is not this paired calculation.
Absolute changes and decision flips are not additively decomposed; the
interaction is explicitly a signed score contrast, not a new accuracy metric.
The conditioned increment includes the declared encode/decode/alignment chain,
not only codec mathematics. None of this can undo a source's prior encoding.

Tests use explicit arithmetic fixtures, not fabricated benchmark predictions:
they distinguish the PCM16-conditioned increment from the total path shift,
verify four-view coverage and within-recording cancellation, retain all-failure
nulls, reject invalid scored values, and exercise the hash-bound CLI output.
NumPy [finite-value checks](https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html)
and [default linear quantiles](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)
follow the official API documentation; no alternate estimator was selected.

### Native-primary versus common-master comparisons

The native DeepFense condition reads original files directly, whereas its
controlled transports start with the declared float32/44.1-kHz master.
Consequently, even its `float_identity` comparison to the primary includes an
input-path change. Such rows explicitly set `repeatability_eligible=false` and
are summarized in `common_master_input_path_difference`, not mixed into
`primary_repeatability`. Native pair primaries retain the actual original
input path and qualify for the latter. Missing comparisons stay unscored;
these two diagnostic summaries are not codec effects or accuracy estimates.
The report retains `input_condition` so the separate conditions are identifiable.

# ArtifactBench v1.1 — Official Results

**v1.1 = v1 test partition after integrity audit** (2026-07-02): 34 real tracks found to
overlap the ArtifactNet v9.4 training manifest (filename-stem/content-hash audit) and
5 real tracks whose source videos were deleted were purged: **n = 2,263 → 2,224**
(1,388 AI / 836 real). All purged tracks are real; the AI side is unchanged.

## Results on v1.1

Because the purge removes only real tracks, each model's recall is unchanged and its
post-purge FPR lies between two exact extremes (all purged tracks were negatives /
all were false positives). No re-inference is required; the v1 official runs bound the
v1.1 numbers exactly.

| Model | Params | Recall (unchanged) | FPR on v1.1 (exact bounds) | F1 on v1.1 (exact bounds) |
|---|---:|:---:|:---:|:---:|
| ArtifactNet v9.4 | 4.0M | 0.9755 | [0.0%, 1.6%] | [0.9829, 0.9876] |
| SpecTTTra α-120s | 18.7M | 0.7046 | [15.7%, 20.3%] | — |
| CLAM (MoM) | 194M | 0.8761 | [67.8%, 72.5%] | — |

The ranking and all qualitative conclusions of the v1 comparison are unchanged under
either extreme. An independent re-measurement of ArtifactNet on the purged partition
(fixed-7-chunk median protocol) yields **F1 = 0.9830**, consistent with the bound.

## Reference — v1 official numbers (n = 2,263, unchanged, for reproducibility)

| Model | F1 | Precision | Recall | FPR |
|---|:---:|:---:|:---:|:---:|
| ArtifactNet v9.4 | 0.9829 | 0.9905 | 0.9755 | 1.49% |
| CLAM (MoM) | 0.7576 | 0.6674 | 0.8761 | 69.26% |
| SpecTTTra α-120s | 0.7713 | 0.8519 | 0.7046 | 19.43% |

## Files

- `artifactbench_v1.1_test_purged.json` — purged test partition manifest (2,224 tracks,
  purged track_ids and unrecoverable list preserved in `metadata`)
- `real_tracks_v1.1.csv` — per-track v1.1 status column
  (`test_ok` 836 / `purged_train_overlap` 34 / `train_partition_or_unrecoverable` 930)
- v1 files are unchanged — results computed on v1 remain reproducible.

*Maintained by Intrect. License: CC BY-NC 4.0 (same as v1).*

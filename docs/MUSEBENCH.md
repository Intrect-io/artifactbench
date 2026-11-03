# MuseBench — research note and zero-shot protocol

MuseBench is a large human-vs-AI music detection benchmark:

> Yi Chen Liu, Syed Irfan Ali Meerza, Jiawei Yu, Xi Guo, Samuel Bright, Jian Liu.
> *MuseBench: A Benchmark for AI-Generated Music Detection and Artifact Analysis.*
> arXiv:2610.20310v1, 14 Oct 2026. <https://arxiv.org/abs/2610.20310>

Data: Hugging Face `Anonymousv22222/MuseBench-part1` … `MuseBench-part5` (full release) and
`Anonymousv22222/MuseBench-Example` (one file per folder). This page summarizes the paper and the
dataset card, lists where they disagree, and defines how ArtifactBench evaluates a detector that
was **not** trained on MuseBench. It contains no ArtifactNet scores: results stay with the
maintainer's data root.

## Composition (paper, Table 4)

| Source | Tracks | Hours | Type |
|---|---:|---:|---|
| Genius-Lyrics (human, **not redistributed**) | 9,983 | 609 | vocal |
| JamendoMaxCaps (human) | 9,920 | 648 | instrumental |
| MusicGen, AudioLDM2, MAGNeT, Stable Audio Open, Lyria RealTime | 19,903 each | 165 each | instr. |
| Riffusion | 19,903 | 169 | vocal |
| JASCO | 19,903 | 55 | instr. |
| Mustango | 19,903 | 56 | vocal |
| Suno (from SONICS: v2, v3, v3.5) | 10,000 | 485 | vocal |
| Udio (from SONICS: Udio-32, Udio-130) | 10,000 | 307 | vocal |
| Human–AI co-created (MusicGen-Style remixes of human tracks) | 9,887 | 82 | mixed |
| Lossy transforms (MP3 / OGG / AAC) | 53,805 | 571 | — |
| Acoustic post-processing (EQ / pitch-time / loudness) | 53,805 | 572 | — |
| Additive (Gaussian / PGD) | 35,870 | 239 | — |
| FMA, pre-2010 (human, out-of-domain) | 8,672 | 678 | mixed |
| **Total** | **361,166** | **5,296** | |

The paired generators were conditioned on captions of the Genius-Lyrics and Jamendo tracks (one
generation per source track, capped at 30 s; JASCO and Mustango produce ~10 s). Each transform
family has 17,935 tracks per transform (53,805 = 3 × 17,935), about 10 % of the 179,224 AI tracks,
consistent with transforms being applied to the test split only.

## Distribution on Hugging Face

Five dataset repositories, each with 15 standalone GNU tar shards (`partN_001.tar` …
`partN_015.tar`): 75 shards, 798.4 GB. Every shard starts with a tar header at offset 0, so a
shard can be indexed by walking its headers with HTTP range requests, without downloading it.
The shard list is pinned to these commits:

| Repository | Revision |
|---|---|
| `Anonymousv22222/MuseBench-part1` | `7c7131dba082dcee0858327d1beac17b1bb540f8` |
| `Anonymousv22222/MuseBench-part2` | `906fccf9d3f7608fb1078e0d38973ea086b70eb8` |
| `Anonymousv22222/MuseBench-part3` | `f15d3bc904d5d76b883c840c6963e86802528152` |
| `Anonymousv22222/MuseBench-part4` | `251fd5aeab23b9ec5bb6b7c2f1c8847c1d0bda57` |
| `Anonymousv22222/MuseBench-part5` | `95dbba8b8e3b6e917c1119af8046d3ab8ded7189` |

## Measured inventory (shard index, 2026-11-04)

Built with `scripts/musebench/index_shards.py` and summarized with `scripts/musebench/inventory.py`
(`python scripts/musebench/inventory.py --depth 1`). The 75 shards hold 351,188 regular files,
798.0 GB: 351,183 audio files plus 5 CSV files (`Caption/captions-genius.csv`,
`Caption/captions-jamendo.csv`, two files under `Genius-Lyrics/`, and one CSV in `Jamendo/`).
351,183 is exactly the paper's 361,166 tracks minus the 9,983 Genius-Lyrics human tracks that
are not redistributed.

| Folder | Files | GB | Extensions | Shards |
|---|---:|---:|---|---:|
| `Caption` | 2 | 0.0 | csv 2 | 2 |
| `Genius-Lyrics` | 2 | 0.1 | csv 2 | 2 |
| `Jamendo` | 9,921 | 54.9 | mp3 9920, csv 1 | 10 |
| `MagNet_Generated_Music` | 19,903 | 38.2 | wav 19903 | 5 |
| `Musicgen_Generated_Music` | 19,903 | 38.2 | wav 19903 | 10 |
| `OOD FMA` | 8,672 | 70.9 | mp3 8672 | 10 |
| `aac_generated_music` | 17,935 | 9.1 | m4a 17935 | 6 |
| `ai_human_co-created_music` | 9,887 | 26.2 | wav 9887 | 9 |
| `audioldm_generated_music` | 19,903 | 38.2 | wav 19903 | 10 |
| `eq_generated_music` | 17,935 | 48.0 | wav 17935 | 8 |
| `gaussion_noised_music` | 17,935 | 40.2 | wav 15935, mp3 2000 | 7 |
| `jasco_generated_music` | 19,903 | 12.7 | wav 19903 | 8 |
| `loudness_generation_music` | 17,935 | 48.0 | wav 17935 | 7 |
| `magenta_generated_music` | 19,903 | 114.6 | wav 19903 | 20 |
| `mp3_noised_music` | 17,935 | 11.0 | mp3 17935 | 5 |
| `mustango_generated_music` | 19,903 | 6.5 | wav 19903 | 5 |
| `ogg_noised_music` | 17,935 | 4.7 | ogg 17935 | 5 |
| `pgd_generated_music` | 17,935 | 15.5 | wav 15935, mp3 2000 | 5 |
| `pitch_time_generated_music` | 17,935 | 48.5 | wav 17935 | 10 |
| `riffusion_generated_music` | 19,903 | 53.9 | wav 19903 | 10 |
| `stable_audio_generated_music` | 19,903 | 105.3 | wav 19903 | 15 |
| `suno_generated_music` | 10,000 | 8.1 | mp3 10000 | 5 |
| `udio_generated_music` | 10,000 | 5.1 | mp3 10000 | 5 |
| **Total** | **351,188** | **798.0** | | |

Observations that matter for acquisition:

- Every AI generator folder holds 19,903 files (9,983 `Genius-Lyrics/` + 9,920 `Jamedo/`), except
  Suno and Udio (10,000 each, `fake_<id>_<generator>_<seed>.mp3`).
- The clean AI audio is large: `magenta` 114.6 GB, `stable_audio` 105.3 GB, `riffusion` 53.9 GB,
  `MagNet`/`Musicgen`/`audioldm` 38.2 GB each, `jasco` 12.7 GB, `mustango` 6.5 GB, Suno 8.1 GB,
  Udio 5.1 GB. Human audio: Jamendo 54.9 GB, OOD FMA 70.9 GB. Co-created: 26.2 GB.
- Each of the eight transform folders holds 17,935 files spread over the ten generator
  sub-folders (about 2,000 per open generator, 1,000 for each of Suno and Udio, i.e. a test-split
  sample). Their formats follow the transform: `aac_*` is `.m4a`, `ogg_*` is `.ogg`, `mp3_*` is
  `.mp3`. In `gaussion_noised_music` and `pgd_generated_music` the Suno and Udio files are `.mp3`
  (2,000 each) and the rest `.wav`.
- Member paths are unique across all 75 shards (checked by `verify.py index`); a path therefore
  identifies a track without the shard name.


## Dataset card vs paper

| Topic | Dataset card | Paper |
|---|---|---|
| Eighth paired generator | `magenta_generated_music/` ("Magenta", Apache 2.0) | Lyria RealTime via API; the folder holds 19,903 files, the same count and hours as the paper's Lyria row, so it is inferred to be that generator (neither source says so) |
| Paired generators | 8: MusicGen, MAGNeT, AudioLDM, JASCO, Magenta, Mustango, Riffusion, Stable Audio | 8: seven open-source (MusicGen, AudioLDM**2**, Riffusion, MAGNeT, JASCO, Mustango, Stable Audio Open) plus Lyria RealTime; with Suno and Udio, 10 generators in both |
| Jamendo human count | ~9,945 | 9,920 (the shards hold exactly 9,920 mp3 files) |
| OOD FMA window | "2008–2010" in one place, "pre-2008" in the tree comment | "uploaded before 2010" |
| Genius-Lyrics human audio | not shipped (copyrighted) | used in every split |
| Suno / Udio origin | "closed-source commercial generators" | reused from SONICS |
| Splits | none published | Default 70/20/10, LOMO; no split file, no seeds |
| Transforms | "every AI track" | test-set AI tracks only (Appendix A.3); the shards hold 17,935 files per transform |

Consequences: the Default and LOMO human negatives cannot be rebuilt (half of them, the vocal
Genius-Lyrics side, is missing), and the paper's test split is unknown, so paper numbers for
Default/LOMO cannot be reproduced on the released files. Folder spellings are preserved as
shipped: `Jamedo/` under each generator, `Genius-lyrics/` under JASCO,
`gaussion_noised_music/`, `loudness_generation_music/`.

## Paper protocol

- **Input:** one random 10 s mono excerpt at 48 kHz per track (30 s in an appendix ablation);
  minority class oversampled by drawing fresh excerpts.
- **Settings:** *Default* (human negative vs AI positive, 70/20/10 train/val/test); *LOMO*
  (leave one generator out, remaining AI 80/20 train/val); *OOD* (FMA accuracy with Default
  checkpoints); *robustness* (co-created and transformed AI tracks, FNR, Default checkpoints).
- **Metrics:** balanced accuracy, precision, recall, FPR (human flagged as AI), FNR (AI passed
  as human).
- **Detectors:** Q-SVM, Deezer, M5, RawNet2, ConvNeXt, ResNet18+Spec, ViT, AST, SpecTTTra, and
  the Ircam Amplify and Resemble AI APIs (on a 1,990 human / 4,500 AI subset). Except for the
  two APIs, every detector is **trained on MuseBench Default**.
- **Transforms (Appendix A.3):** MP3 and AAC at 128 kbps; OGG Vorbis q3; EQ ±6 dB boost/cut in
  one of three bands (25–250 Hz, 250 Hz–4 kHz, 4–20 kHz; scaled to the track bandwidth);
  pitch shift −2/−1/+1/+2 semitones; speed 0.9×/1.1×; loudness normalized to −14 LUFS; Gaussian
  noise σ = 0.025; PGD (ℓ∞, α = 0.001, ε = 0.005, T = 10) with M5 / RawNet2 gradients.

Paper results that a zero-shot detector can be placed next to:

| Detector | FMA OOD acc. (Table 10) | Co-created FNR | MP3 | OGG | AAC | EQ | Pitch/time | Loudness | Gaussian | PGD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deezer | 94.4 | 36.8 | 12.3 | 9.5 | 7.7 | 6.4 | 93.6 | 5.2 | 19.6 | 26.4 |
| M5 | 75.8 | 20.1 | 15.2 | 1.9 | 1.9 | 2.0 | 26.2 | 1.3 | 2.1 | 92.4 |
| ConvNeXt | 43.5 | 12.8 | 0.2 | 0.3 | 0.6 | 0.2 | 17.0 | 0.2 | 0.4 | 11.7 |
| ResNet18+Spec | 28.4 | 0.0 | 0.9 | 3.7 | 0.1 | 3.1 | 11.3 | 2.0 | 5.2 | 11.1 |
| AST | 79.8 | 4.8 | 24.9 | 7.3 | 3.5 | 0.0 | 4.2 | 0.0 | 0.0 | 9.2 |
| SpecTTTra | 39.7 | 33.6 | 0.5 | 1.0 | 1.0 | 0.5 | 16.7 | 0.5 | 0.9 | 10.7 |

(All values in %, from Tables 1 and 10 of arXiv:2610.20310v1; transform columns are FNR.)

## ArtifactBench zero-shot protocol for MuseBench

ArtifactBench evaluates detectors that were not trained on MuseBench. The rules:

1. **No tuning on MuseBench.** The detector's own fixed operating threshold is used (0.5 for
   ArtifactNet), and AUROC is reported as the threshold-free view. Nothing (threshold, crop,
   calibration) is selected on MuseBench data.
2. **Native input.** Each detector receives the audio through its own adapter (for ArtifactNet:
   44.1 kHz, seven 4 s chunks, centre crop), not the paper's 10 s / 48 kHz excerpt.
3. **Eval cohort** (deterministic, sealed), chosen by `scripts/musebench/select.py`:
   3,000 Jamendo human tracks, 3,000 OOD-FMA tracks, 1,000 clean tracks per generator (all
   ten, including Suno and Udio), 1,000 co-created tracks, and 200 tracks per transform ×
   generator (8 × 10 = 80 groups). Clean AI tracks are drawn from tracks that have transformed
   variants (the paper's test split), and the transform samples are the transforms of those
   same tracks, so clean/transform results are paired. Rows in the cohort are marked
   `musebench_role = eval` (also `protocol_split = eval`) and must never be used for training.
4. **Leak annotation.** Suno and Udio are SONICS tracks; FMA and Jamendo audio may also appear in
   a detector's training history. Every row carries `seen_in_training` for the detector under
   test, and every metric is reported for seen and unseen rows separately.
5. **What is reported:** human FPR (Jamendo), OOD-FMA accuracy, per-generator FNR on clean
   tracks, co-created FNR, per-transform FNR, AUROC (human vs each AI group), decode/inference
   failures. Comparisons are drawn against Tables 1 and 10 only, with the caveat that the
   paper's detectors were trained in-distribution; there is no valid comparison with Table 8.

## Acquisition

The full release (~800 GB) is larger than the space available, so it is indexed in full and
acquired selectively:

```bash
# Byte-range index of all 75 shards (no audio is downloaded); resumable.
python scripts/musebench/index_shards.py --root ~/data/musebench --workers 75
# Assert the index is complete and internally consistent.
python scripts/musebench/verify.py --root ~/data/musebench index
# Choose the eval cohort and the budgeted train pool (deterministic).
python scripts/musebench/select.py plan --root ~/data/musebench --budget-gb 170 --free-floor-gb 80
# Download the chosen members by byte range; resumable, stops at the free-space floor.
python scripts/musebench/fetch.py --root ~/data/musebench --workers 16 --free-floor-gb 80
# Write the runtime manifest, then check files, hashes, cohort shape and disk policy.
python scripts/musebench/select.py manifest --root ~/data/musebench
python scripts/musebench/verify.py --root ~/data/musebench acquire
```

### Families and the train pool

Every member is assigned a *family*: the source track it derives from (`caption_set/base
name`; Suno and Udio tracks are their own family). A Jamendo song, its co-created remix, every
generation conditioned on its caption and every transform of those generations share one
family. Selection is ordered by `sha256(seed:family)`, so the same families are preferred
across generators, and the **train pool excludes every family that has an eval member**.

The train pool fills the remaining disk budget in priority tiers, each source up to a cap,
round-robin within a tier:

1. generators that are new to most detectors' training data: AudioLDM2, MAGNeT, JASCO,
   Mustango, Lyria RealTime (5,000 each);
2. co-created (5,000);
3. human Jamendo (3,000) and OOD-FMA (2,000);
4. MusicGen, Riffusion, Stable Audio Open (2,000 each);
5. Suno and Udio: none (they are SONICS tracks, usually available locally already).

Transforms are never put in the train pool; all of them except PGD can be regenerated from
clean audio with the Appendix A.3 parameters.

### Acquired corpus (2026-11-04)

With the defaults above (`--budget-gb 170 --free-floor-gb 80`): 64,756 audio files, 169.9 GB —
33,000 eval rows in 93 sources (91.3 GB) and 31,756 train-pool rows (78.6 GB: 5,000 each of
AudioLDM2, MAGNeT, JASCO, Mustango, Lyria RealTime and co-created, 877 Jamendo, 879
OOD-FMA; the budget ran out inside tier 3). Every file's SHA-256 was recomputed and matches the
fetch ledger. The index allows fetching more of the train pool later with a larger budget.

### Observations from the shipped files

- Suno and Udio transforms sit under a literal `root/` folder
  (`aac_generated_music/suno_generated_music/root/…`); PGD files have no caption-set folder and
  are named `adv_<clean name>`; the PGD folders for MAGNeT and MusicGen are lower case.
- Transform file names end with the transform parameters (`_aac128`, `_mp3128`, `_ogg3`,
  `_eq-cut_low`, `_pt-ps+2st-ts0.9`, `_lnLUFS-14`, `_gn0.0025`). The Gaussian files say
  `gn0.0025`, while Appendix A.3 states σ = 0.025.
- The shipped CSVs (`Caption/captions-*.csv`, `Genius-Lyrics/metadata.csv`,
  `Genius-Lyrics/segments.csv`, `Jamendo/segments.csv`) carry captions, segment descriptions and
  YouTube ids only. The per-track Jamendo `license_ccurl` that the card mentions is **not**
  shipped; manifest rows carry `license_ccurl: null` until it is joined from JamendoMaxCaps.

Rules for the data root (`$MUSEBENCH_ROOT`, default `~/data/musebench`):

- Shard bytes and extracted audio are untrusted: they live only under the data root, scripts live
  in this repository, and Python that parses them runs with `-I`.
- Audio is kept lossless and byte-identical to the shard member (verified by size and SHA-256);
  it is never re-encoded, because re-encoding changes the artifacts detectors look for.
- Non-PGD transforms can be regenerated from clean audio with the parameters above, so only the
  scored transform sample is kept.
- Nothing derived from MuseBench audio (manifests with local paths, scores, reports) is committed
  to this repository.

## Leak map

`scripts/musebench/leak_map.py` marks every manifest row `seen_in_training` (boolean) for the
detector under test and writes `$MUSEBENCH_ROOT/leak/leak_report.json`. The detector and its
training pool are described in `$MUSEBENCH_ROOT/leak/references.json` (machine paths stay out of
this repository): the model, the training manifest(s) or dataset builder it came from, a SONICS
`fake_songs` directory, FMA track-id lists, MTG-Jamendo `raw.meta.tsv` with the audio folders
that were in the pool, and generator-level exposure.

```bash
python scripts/musebench/leak_map.py --root ~/data/musebench --references ~/data/musebench/leak/references.json
python scripts/musebench/verify.py --root ~/data/musebench leak
```

Identity rules:

| MuseBench rows | Key | Confirmation |
|---|---|---|
| Suno, Udio | SONICS file name `fake_<id>_<generator>_<seed>` | file size |
| OOD FMA | FMA track id (leading digits of the file name) | file size vs a local FMA copy, where available |
| Jamendo | normalised (title, artist) vs MTG-Jamendo `raw.meta.tsv`, any ` - ` split | duration within 2 s |

A transform inherits the flag of its clean track and a co-created remix that of its human
source. Generations conditioned on a seen track's caption are different audio and are not
"seen"; exposure to a *generator* is recorded separately in `generator_in_training`. The match
is against the training **candidate pool**, an upper bound on what a checkpoint saw.

Dataset facts found on the way: all MuseBench Suno and Udio files checked so far (2,000 of
2,000 in the acquired corpus) have the name **and byte size** of a SONICS `fake_songs` file, so
any detector trained on SONICS has seen them. JamendoMaxCaps names match MTG-Jamendo metadata
for only a minority of tracks (266 of 3,877 acquired Jamendo rows), so most MuseBench Jamendo
audio is outside MTG-Jamendo.

## Running a detector

The runner's `--protocol-split` has no `eval` choice, so the eval rows are written to their own
manifest first. For ArtifactNet the declared primary policy of the frozen v2 protocol applies:
median of the finite scores of seven 4 s chunks, at least four finite; the strict result (any
non-finite chunk fails the track) is kept as a sensitivity analysis.

```bash
python scripts/musebench/select.py eval-manifest --root ~/data/musebench
python -m artifactbench.bench --model artifactnet \
    --manifest ~/data/musebench/manifests/musebench_eval_manifest.json \
    --split all --n-per-source 0 --n-codec-pair 0 --device cuda \
    --output ~/data/musebench/results/artifactnet_v94
python scripts/audit_artifactnet_nonfinite.py --device cuda \
    --manifest ~/data/musebench/manifests/musebench_eval_manifest.json \
    --failures ~/data/musebench/results/artifactnet_v94/artifactnet/inference_failures.json \
    --output ~/data/musebench/results/artifactnet_v94/nonfinite_audit.json
python scripts/build_artifactnet_finite_chunk_result.py --min-valid-chunks 4 \
    --manifest ~/data/musebench/manifests/musebench_eval_manifest.json \
    --strict-result ~/data/musebench/results/artifactnet_v94 \
    --chunk-audit ~/data/musebench/results/artifactnet_v94/nonfinite_audit.json \
    --output ~/data/musebench/results/artifactnet_v94/finite_chunk
python scripts/musebench/report.py --root ~/data/musebench
python scripts/musebench/verify.py --root ~/data/musebench all
```

`report.py` writes `musebench_report.json` and `.md` next to the run: human (Jamendo) FPR,
FMA-OOD accuracy, clean-AI FNR per generator, co-created FNR, FNR per transform and per
transform × generator, AUROC against the Jamendo negatives, Wilson 95 % intervals, all for
`all`, `seen` and `unseen` rows, plus failure counts. The paper's Table 1 and Table 10 values are
included for context only. Reports stay in the data root; they are not committed here.

With onnxruntime-gpu installed from pip, the CUDA provider needs the `nvidia/*/lib` folders of
that environment on `LD_LIBRARY_PATH`; otherwise onnxruntime silently falls back to the CPU
(about 7× slower for ArtifactNet). CUDA and CPU scores differed by at most 0.016 on a 30-track
probe, with no decision changes at 0.5.

## Licenses

- Collection: CC BY-SA 3.0 (inherited from JamendoMaxCaps).
- Jamendo audio: per-track Creative Commons licenses (CC BY, BY-SA, BY-NC, BY-NC-ND), recorded
  in JamendoMaxCaps' `license_ccurl` metadata (not shipped with MuseBench); NC and ND tracks
  restrict commercial use and derivatives.
- FMA audio: per-track Creative Commons licenses; FMA metadata CC BY 4.0.
- AI audio: bound by each generator's terms; most open weights are non-commercial
  (MusicGen / MAGNeT / JASCO weights CC BY-NC 4.0, AudioLDM and Mustango CC BY-NC-SA 4.0,
  Stable Audio Open community license).
- Suno / Udio outputs: research use only under each platform's terms.

Benchmarking is research use. Whether any part of MuseBench may be used to train a commercial
detector is a separate legal decision; per-row license fields are kept so the corpus can be
filtered.

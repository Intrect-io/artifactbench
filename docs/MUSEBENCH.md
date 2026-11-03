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

<!-- INVENTORY -->

## Dataset card vs paper

| Topic | Dataset card | Paper |
|---|---|---|
| Tenth generator | `magenta_generated_music/` ("Magenta", Apache 2.0) | Lyria RealTime via API |
| Paired generators | 8 (MusicGen, MAGNeT, AudioLDM, JASCO, Magenta, Mustango, Riffusion, Stable Audio) | 8 open + Lyria; AudioLDM is **AudioLDM2** |
| Jamendo human count | ~9,945 | 9,920 |
| OOD FMA window | "2008–2010" in one place, "pre-2008" in the tree comment | "uploaded before 2010" |
| Genius-Lyrics human audio | not shipped (copyrighted) | used in every split |
| Suno / Udio origin | "closed-source commercial generators" | reused from SONICS |
| Splits | none published | Default 70/20/10, LOMO; no split file, no seeds |
| Transforms | "every AI track" | test-set AI tracks only (Appendix A.3) |

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
3. **Eval cohort** (deterministic, sealed): all Jamendo human tracks, all OOD-FMA tracks,
   up to 2,000 clean tracks per generator (including Suno and Udio), up to 2,000 co-created
   tracks, and up to 200 tracks per transform × generator. Rows in the cohort are marked
   `musebench_role = eval` and must never be used for training.
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
```

Rules for the data root (`$MUSEBENCH_ROOT`, default `~/data/musebench`):

- Shard bytes and extracted audio are untrusted: they live only under the data root, scripts live
  in this repository, and Python that parses them runs with `-I`.
- Audio is kept lossless and byte-identical to the shard member (verified by size and SHA-256);
  it is never re-encoded, because re-encoding changes the artifacts detectors look for.
- Non-PGD transforms can be regenerated from clean audio with the parameters above, so only the
  scored transform sample is kept.
- Nothing derived from MuseBench audio (manifests with local paths, scores, reports) is committed
  to this repository.

## Licenses

- Collection: CC BY-SA 3.0 (inherited from JamendoMaxCaps).
- Jamendo audio: per-track Creative Commons licenses (CC BY, BY-SA, BY-NC, BY-NC-ND), recorded
  in the `license_ccurl` metadata; NC and ND tracks restrict commercial use and derivatives.
- FMA audio: per-track Creative Commons licenses; FMA metadata CC BY 4.0.
- AI audio: bound by each generator's terms; most open weights are non-commercial
  (MusicGen / MAGNeT / JASCO weights CC BY-NC 4.0, AudioLDM and Mustango CC BY-NC-SA 4.0,
  Stable Audio Open community license).
- Suno / Udio outputs: research use only under each platform's terms.

Benchmarking is research use. Whether any part of MuseBench may be used to train a commercial
detector is a separate legal decision; per-row license fields are kept so the corpus can be
filtered.

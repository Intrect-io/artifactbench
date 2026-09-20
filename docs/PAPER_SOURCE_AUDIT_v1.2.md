# Manuscript source and checkpoint audit

Inspected 2026-09-05. Scope: the current LaTeX draft's bibliography and bounded
dataset/model descriptions. This is not a systematic literature review or a
verification of every claim in the cited papers. Reported training recipes are
upstream declarations, not independently audited training histories.

## Corrections applied

- AIME is an ICASSP 2025 proceedings paper, not only an arXiv preprint. The
  [author-deposited record](https://arxiv.org/abs/2506.19085) identifies the venue
  and DOI `10.1109/ICASSP49660.2025.10887745`. Its 6,000 generated recordings
  and 12 generation-model labels are distinguished from the 500 MTG-Jamendo
  recordings documented by the [dataset card](https://huggingface.co/datasets/disco-eth/AIME).
  This citation does not make the mixed-source dataset uniformly licensed.
- The [original AST paper](https://arxiv.org/abs/2104.01778) is an Interspeech
  2021 architecture reference. The evaluated 60-second music-detection
  [fine-tune card](https://huggingface.co/AI-Music-Detection/ai_music_detection_large_60s/blob/5e29461bbfc41dc5fe4ed5f7fadc961aa84a7e3e/README.md)
  separately declares its AudioSet AST base and `SleepyJesse/ai_music_large`
  training source. Neither the base model's original benchmark numbers nor
  the card's training logs are ArtifactBench results.
- [DeepFense](https://arxiv.org/abs/2604.08450) is a framework. The evaluated
  checkpoint is `FakeMusicCaps_EAT_Nes2Net_NoAug_Seed42`, revision
  `c8ec432e71e94babaa83e0ef681c24bfa540ada7`. Its
  [configuration](https://huggingface.co/DeepFense/FakeMusicCaps_EAT_Nes2Net_NoAug_Seed42/blob/c8ec432e71e94babaa83e0ef681c24bfa540ada7/config.yaml)
  declares the EAT frontend, Nes2Net backend, FakeMusicCaps train/dev recipes,
  seed 42, empty augmentation list and `spoof: 0`, `bonafide: 1` mapping.
  There is no `README.md` in the selected local snapshot; the evidence is the
  configuration, not an assumed model card. Private upstream training paths
  are not reproduced here.
- The [fakeprint paper](https://arxiv.org/abs/2506.19108) is an ISMIR 2025
  methodological reference. The evaluated logistic-regression checkpoint is
  the third-party [lofcz release](https://huggingface.co/lofcz/ai-music-detector/blob/d2180598fed79e3f917e8050a00439982466e5c6/README.md),
  not an official Deezer production detector. The citation now pins the actual
  revision, as does the AST fine-tune citation.

AST and DeepFense adapter sources were inspected read-only alongside these
files. AST uses a 60-second, zero-padded prefix and class-1 softmax probability;
DeepFense uses a four-second prefix, repeat padding, and class-0 softmax
probability. These are **adapter observations**, not claims inferred from the
framework paper. No inference code, checkpoint or active run was changed.

## Retained claims and their boundaries

| Draft claim | Primary source | Boundary |
| --- | --- | --- |
| SONICS: over 97,000 songs, over 49,000 synthetic; SpecTTTra long-range modeling | [SONICS, ICLR 2025](https://arxiv.org/abs/2408.14080) | Counts describe the upstream dataset, not this benchmark's selected subset. |
| MoM: over 130,000 songs, curated OOD evaluation, CLAM | [Author-deposited paper](https://arxiv.org/abs/2512.00621) | Record says accepted at TMLR; the draft retains the traceable arXiv citation without inventing journal volume/pages. |
| FakeMusicCaps: text-to-music detection and attribution | [Dataset paper](https://arxiv.org/abs/2409.10684) | Does not establish non-overlap with any evaluated model's training set. |
| FST: contextual and multi-encoder detection | [FST paper](https://arxiv.org/abs/2601.13647) | The two-stage checkpoint and actual 90-second/beat-segment policy require code and asset evidence. |
| FMA artist/recording metadata supports recovery and grouping | [FMA, ISMIR 2017](https://arxiv.org/abs/1612.01840) | A same-ID recovery is measured locally; the paper does not certify recovered bytes. |
| Suno v5.5 exists by March 2026 | [Official announcement](https://suno.com/blog/v5-5) | Existence does not identify the generator or workflow of an individual song. |
| Stable Audio 3 is a latent-diffusion family with a semantic-acoustic autoencoder | [Technical paper](https://arxiv.org/abs/2605.17991) | Does not identify the model size of the four [official page demos](https://stability.ai/stable-audio). |
| Lyria's official page labels a 3.5 demonstration carousel | [Provider page](https://deepmind.google/models/lyria/) | Eleven admitted recordings come from frozen acquisition evidence, not an invariant website count. |
| ArtifactBench was introduced with ArtifactNet | [Original detector paper](https://arxiv.org/abs/2604.16254) | Its earlier counts, exposure claims and detector results are not substituted for rc2 measurements. |
| The inspected public ArtifactNet revision differs from the evaluated local stack | [Pinned public repository](https://huggingface.co/intrect/artifactnet/tree/7c9b753a9d006b48e4bfaf85bf0157e135f4aad4) | Bounded asset inventory, not proof of absence from all public hosts or permission to publish new weights. |

Provider pages are mutable. Frozen page hashes, recording identities, local
asset hashes and statistical report hashes remain the measurement evidence;
citations provide attribution and context. External model-card metrics are not
copied into the performance table. Final results, transport experiments,
full ONNX parity, access/rights decisions and arXiv compilation remain separate
completion gates.

## DeepFense post-run interpretation check

After this checkpoint completed all 2,579 entries, its raw-0.5 legacy false-
positive count was 797/820 despite 399/400 native positives being detected.
Evidence: `out/v1.2_checkpoint_deepfense_rc2_260905/checkpoint.json`, SHA-256
`3be33787ef88396d2004cae14f3e2da869b601dfcc41fd4df0f6aaf3e7054c24`.
These are selected-checkpoint/adapter results, not framework-wide performance.
Do not invert the class direction or tune a threshold from these labels.

A read-only check of installed DeepFense 0.2.2 confirms that the dataset maps
labels through the declared `label_map`; its CE training head and returned
`logits` use that same two-class projection. The adapter applies softmax and
selects class 0. The framework's `scores`/`probs` are not substituted: CE
`get_score()` returns a single bonafide logit despite its LLR docstring, and
the detector applies sigmoid to that scalar. EAT's wrapper owns its Fbank
extraction and normalization; the adapter supplies raw 16-kHz waveform.
Its four-second prefix/repeat rule matches the declared validation padding.

This does **not** establish the cause of the high false-positive rate or prove
exact equivalence to the authors' original evaluation pipeline. In particular,
upstream `data/transforms/transforms.py` reads with SoundFile then resamples
directly using librosa; ArtifactBench uses its declared common float32/44.1-kHz
loader followed by model-specific TorchAudio resampling. A subsequent fixed
nine-file CPU diagnostic confirms seven flips among eight directly comparable
inputs. Its same-common-waveform control isolates the final resampler and
flips eight of nine decisions; exact-input replay is bit-exact. The original
run has an interpretation hold. See [the full audit](DEEPFENSE_INPUT_PATH_AUDIT_v1.2.md).
Original predictions, model and adapter sources remain unchanged; corrected
full-cohort performance is still unavailable.

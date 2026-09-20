# Transport implementation and input-provenance clarification

2026-09-05, before any v1.2 controlled-transport inference. Primary benchmark
inference has started; transport sampling reads only the frozen metadata.
No detector score is read to choose inputs, replacements, or transformations.

The protocol's "lossless-source" criterion is implemented as a **lossless input
container/codec** (PCM or FLAC) in the legacy partition. This does not establish
that the audio has never been compressed: AIME files are stored as float PCM,
and several web-sourced cells are FLAC/PCM exports. Earlier encoding history is
not fully established. The experiment estimates the incremental effect of each
specified transform on the supplied decoded signal, not degradation from a
provably pristine original. The paper must retain this qualification.

Select at most 50 recording groups. Within eligible entries, choose one entry
per identified recording group by SHA256(`260905:<entry-id>`). Visit eligible
source cells in SHA256(`260905:codec-cell:<source>`) order, taking their remaining
entries in seeded hash order, one per cell per round until 50. Save all eligible
IDs and final selections. No short input is dropped for an anticipated model
failure. Source sample formats, sample rates, and unknown prior encoding history
are preserved. This is not the same cohort as the historical 50-pair audit.

Every controlled variant starts from **the exact same cached float32, mono,
44.1 kHz waveform**, using the fixed loader. The eight transforms are float WAV
identity, PCM16 WAV only, and MP3/AAC/Opus 128 kbit/s from each of float and PCM16
WAV staging. Decode explicitly as f32le, retain the decoded start, trim/pad the
tail to the original frame count, and record the length adjustment. Preserve
float waveform hashes, staging subtype, encoder arguments, and FFmpeg version.
Identity must be numerically exact. Random-crop seeds are reset to the same
recording seed for every variant so crop selection is not a new intervention.

The initial ten native candidates were not all paired transports. Complete-URL
and waveform checks on 2026-09-05, before transport detector scoring, confirmed
only the three MP3/AAC pairs. The seven MP3/MP3 candidates are sibling outputs
under a shared CDN parent ID, not verified same-recording transports. They are
excluded under the identity requirement, with all original evidence retained.
See [the rc2 correction](UDIO_IDENTITY_CORRECTION_v1.2.md), which also corrects
five primary Udio file selections. The paired experiment uses the three
verified MP3/AAC pairs and reports its very small sample size. Record duration,
hash, and waveform-alignment evidence; the provider's transport path may change
more than the codec. No sample-alignment edit is applied to native inputs.

Model runtime, checkpoint, adapter, seed, and probability mapping must match the
corresponding primary run. Keep each attempt and failure; never fill a failed
variant with 0.5 or drop it without coverage. Report score shifts, whole-pair
coverage, ranges, and fixed-threshold flips, with paired recording/creator
bootstrap intervals. Official-demo files are not added to this controlled
legacy subset. The final metadata release remains immutable; transport evidence
is a separate, hash-bound supplement.

The statistics retain total float-relative differences and additionally
separate codec increments against their matching staging baselines. The
same-recording four-view interaction, failure coverage and exact directions
are specified in [the statistical clarification](STATISTICS_v1.2.md#matched-staging-transport-contrasts).
This calculation was implemented before rc2 transport detector scoring; the
prepared inputs and transformations are unchanged.

Implementation references: [SoundFile read/write and subtype documentation](https://python-soundfile.readthedocs.io/en/latest/),
[FFmpeg codec documentation](https://ffmpeg.org/ffmpeg-codecs.html).

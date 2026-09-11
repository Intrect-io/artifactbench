# Udio file-identity correction before transport scoring

2026-09-05 13:20 KST. This supersedes the rc1 claim that all ten native
candidate pairs are two transports of the same recording.

## Evidence and impact

The inventory grouped URLs by the first 32-hex directory under `samples/`.
That is insufficient: `/samples/<id>/1/...` and `/samples/<id>/2/...` can be
different audio. `extract_udio()` joined a song page to this parent media ID;
`select_native()` then chose an MP3 by variant hash, without matching the
complete page-declared `song_path`. This incorrectly transferred recording
metadata to five sibling primary files.

The full-waveform transport audit read no detector scores. MP3/AAC pairs pass
the previously declared correlation criterion (three pairs, zero estimated lag,
correlation 0.99890–0.99978). All seven MP3/MP3 candidates fail it. One has a
shared suffix but a different beginning; six lack a confirmed full-waveform
match. Parent-directory identity is therefore not transport identity.

All 200 primary Udio URLs were checked against the saved public song objects:
194 match `song_path`, one matches explicitly declared `original_song_path`,
and five select the other numbered file. The five affected media IDs are:

- `7a8811fdc2fd4361a4cbebca99e045ea`
- `3bcc652eafe641c2a8bed814659782aa`
- `ef763f0cd2b8460ab5952db893d431ae`
- `b279dab20ce040eb92d3526aa066e17d`
- `ca758df4f1184ec9b3bb4cf65c6bc181`

## Correction policy

Stop the incomplete rc1 sequential run and preserve every prediction, metadata
file, original waveform, and failed assumption. Do not edit the frozen rc1
manifest in place, reuse its partial scores as final results, or call the
interrupted run complete. Rebuild as rc2 with a new manifest identity.

Use the same selected recording/page/creator cohort and quota. Select only a
complete URL explicitly declared by `song_path`, `video_path`, or
`original_song_path`; prefer the current song/video path over an original-path
fallback, then use the existing MP3/hash preference. Percent-encoding
normalization is allowed, but numbered path components and query semantics
must not be discarded. No detector score, predicted label, or anticipated
model success enters the decision. Unavailable matching files are errors,
not silently accepted siblings or new score-selected recordings.

Keep only page-linked additional views and independently verify their full
waveforms. The seven rejected sibling candidates remain in the rc1 audit, not
in the rc2 transport denominator. A declared untrimmed original remains
explicitly identified as such. Recheck every changed primary's audio hashes,
the full duplicate candidate search, creator/lineage grouping, and exposure
checks before freezing rc2 and restarting all eight models.

The intended primary cohort remains 2,579 entries (including 200 Udio), but
final recording-group and dependence-cluster counts must come from the rc2
audit, not be assumed unchanged. The controlled 50-recording legacy sample is
unaffected in content; its prepared waveforms may be reused only with verified
byte hashes and a separately recorded rc2 binding.

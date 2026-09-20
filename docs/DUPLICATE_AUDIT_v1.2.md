# Pre-score duplicate audit specification

Fixed on 2026-09-05 before running the duplicate matcher or new detector scores.
This supplements protocol revision 2; it does not change score-based eligibility.

- Join every selected entry to its completed full-decode validation by path.
  Require all expected reports and preserve their hashes. Group identical file
  bytes and identical full decoded mono PCM separately.
- Search the available first-120-second Chromaprint sequences with an inverted
  index of consecutive landmark pairs, three fingerprint frames apart. Use both
  low and high 16-bit halves. Discard nondiscriminative tokens with over 200
  occurrences or over 20 recording IDs; report their count. Require eight
  consistent offset votes, at least 80 overlapping fingerprint frames and half
  the shorter available fingerprint, and at most 20% mean bit disagreement.
- Fingerprints only propose candidates. Decode each candidate's first 120
  seconds to 8 kHz mono float PCM and independently cross-correlate the waveforms
  within 0.5 seconds of the fingerprint offset. A confirmed shared excerpt needs
  at least ten seconds / 80% of the shorter decoded excerpt, absolute normalized
  correlation >= 0.985 and median five-second-window absolute correlation >=
  0.985. Silent inputs are not positive evidence. Log all candidates and scores;
  borderline/unconfirmed candidates remain explicitly reviewable.
- The search is a bounded near-identical-recording audit, not proof that there
  are no musical covers, common samples, time-stretched remixes, or matches
  outside the first 120 seconds. Short unavailable fingerprints are disclosed.
  Chromaprint's own scope is near-identical audio, not arbitrary similarity:
  https://github.com/acoustid/chromaprint.
- Keep legacy evaluation entries for historical comparison even if duplicate
  evidence is found. Report evaluation-entry counts and unique recording groups
  separately, use shared groups for uncertainty, and add a deterministic
  representative-only sensitivity view. Do not turn transport variants into
  extra independent observations. Any conflicting labels block release pending
  investigation; no majority-vote relabelling.
- Creator and known lineage form dependence clusters, separate from duplicate
  groups. Ignore null, `$undefined`, and all-zero UUID lineage sentinels. A shared
  style parent can justify dependence without proving identical audio. Preserve
  unresolved ancestry and compare resolvable IDs to known local exposure IDs.

These thresholds are conservative audit rules, not a validated music-retrieval
benchmark. Preserve the review table and limitations with the release.

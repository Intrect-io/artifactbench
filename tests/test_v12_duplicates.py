import numpy as np

from artifactbench.v12.duplicates import Groups, compare_waveforms, fingerprint_candidates, lineage_ids
from artifactbench.v12.review_duplicates import shared_window_spans


def test_shifted_noisy_fingerprint_is_candidate_not_verdict():
    rng = np.random.default_rng(7)
    a = rng.integers(0, 2**32, 500, dtype=np.uint32)
    b = np.r_[rng.integers(0, 2**32, 10, dtype=np.uint32), a ^ 1]
    c = rng.integers(0, 2**32, 500, dtype=np.uint32)
    candidates, _ = fingerprint_candidates([a, b, c, None])
    assert len(candidates) == 1
    assert candidates[0]['offset_fp_frames'] == -10
    assert candidates[0]['fingerprint_bit_error'] == 1/32
    assert 'confirmed_shared_excerpt' not in candidates[0]


def test_waveform_confirms_offset_gain_and_polarity_not_unrelated_noise():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.1, 20*8000).astype(np.float32)
    b = np.r_[np.zeros(230), -0.7*a].astype(np.float32)
    result = compare_waveforms(a, b, -230/8000)
    assert result['confirmed_shared_excerpt']
    assert result['offset_samples_8000'] == -230
    assert not compare_waveforms(a, rng.normal(size=len(a)).astype(np.float32), 0)['confirmed_shared_excerpt']
    assert not compare_waveforms(np.zeros_like(a), np.zeros_like(a), 0)['confirmed_shared_excerpt']


def test_short_overlap_cannot_verify_duplicate():
    a = np.ones(4*8000, dtype=np.float32)
    assert compare_waveforms(a, a, 0)['reason'] == 'insufficient_waveform_overlap'


def test_lineage_ignores_sentinels_and_normalizes_media_and_song_uuid():
    row = {'provider': 'udio', 'media_id': 'a' * 32, 'song_id': '12345678-abcd-abcd-abcd-123456789012',
           'lineage': {'parent_id': '$undefined', 'edited_clip_id': '00000000-0000-0000-0000-000000000000',
                       'original_song_path': 'https://host/samples/' + 'b'*32 + '/song.mp3'}}
    assert lineage_ids(row) == {'udio:'+'a'*32, 'udio:'+'b'*32, 'udio:12345678abcdabcdabcd123456789012'}


def test_group_labels_are_order_independent_and_shared_parent_joins():
    first, second = Groups(['a', 'b', 'c']), Groups(['c', 'b', 'a'])
    for g in (first, second):
        g.join('a', 'parent')
        g.join('b', 'parent')
    assert first.labels(['a', 'b', 'c'], 'x:') == second.labels(['c', 'b', 'a'], 'x:')
    assert first.root('a') == first.root('b') != first.root('c')


def test_shared_prefix_is_dependence_evidence_not_whole_track_identity():
    assert shared_window_spans([1., 1., .2, 1., .2, 1., 1.]) == [
        {'start_seconds': 0, 'end_seconds': 10}, {'start_seconds': 25, 'end_seconds': 35}]

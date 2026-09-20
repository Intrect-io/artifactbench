import numpy as np
import pytest

from artifactbench.v12.prepare_transports import sample_controlled, save_array


def test_controlled_sample_is_metadata_only_balanced_and_recording_disjoint():
    rows = [{'id': f'{source}:{i}', 'source': source, 'partition': 'legacy', 'audio_codec': codec,
             'recording_group': f'{source}:{i}'} for source, codec in [('a', 'flac'), ('b', 'pcm_s16le'), ('c', 'mp3')] for i in range(10)]
    rows.append(dict(rows[0], id='other-view'))
    selected, eligible = sample_controlled(rows, limit=8)
    assert len(eligible) == 21 and len(selected) == 8
    assert sum(r['source'] == 'a' for r in selected) == 4
    assert len({r['recording_group'] for r in selected}) == 8
    assert selected == sample_controlled(list(reversed(rows)), limit=8)[0]


def test_waveform_cache_roundtrip_is_exact_and_refuses_changes(tmp_path):
    wave = np.linspace(-.7, .8, 128, dtype=np.float32)
    path = tmp_path/'audio.npy'
    first = save_array(path, wave)
    assert save_array(path, wave) == first
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), wave)
    with pytest.raises(ValueError, match='differs'):
        save_array(path, wave/2)

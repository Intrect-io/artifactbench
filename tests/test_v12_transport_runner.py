import json

import numpy as np
import pytest
import torch

from artifactbench.v12.audio import TRANSFORMS
from artifactbench.v12.prepare_transports import save_array
from artifactbench.v12.run_transports import infer_view, load_cached, transport_cases


def test_seed_is_identical_across_views_and_different_for_other_recording():
    class CropLikeModel:
        def forward(self, audio):
            return np.random.random()
    model, audio = CropLikeModel(), np.zeros(10, dtype=np.float32)
    a, b = infer_view(model, audio, 'one'), infer_view(model, audio, 'one')
    assert a['seed'] == b['seed'] and a['prob'] == b['prob']
    assert a['prob'] != infer_view(model, audio, 'two')['prob']


def test_execution_error_has_no_fabricated_probability_and_oom_is_not_prediction():
    class BadModel:
        def forward(self, audio):
            raise ValueError('intrinsic input error')
    row = infer_view(BadModel(), np.zeros(10, dtype=np.float32), 'one')
    assert row['outcome'] == 'model_execution_error' and 'prob' not in row
    class OomModel:
        def forward(self, audio):
            raise torch.cuda.OutOfMemoryError('infrastructure')
    with pytest.raises(torch.cuda.OutOfMemoryError):
        infer_view(OomModel(), np.zeros(10, dtype=np.float32), 'one')


def test_cached_waveform_hash_and_format_are_checked(tmp_path):
    cache = save_array(tmp_path/'wave.npy', np.zeros(20, dtype=np.float32))
    assert load_cached(cache).shape == (20,)
    with pytest.raises(ValueError, match='sample rate'):
        load_cached(dict(cache, sample_rate=16000))
    with pytest.raises(ValueError, match='hash'):
        load_cached(dict(cache, npy_sha256='0'*64))


def test_transport_membership_requires_all_variants_and_exact_identity():
    entry = {'id': 'one'}
    base = {'frames': 20, 'pcm_f32le_sha256': 'test-hash'}
    manifest = {'controlled': [{'entry': entry, 'base': base,
                'variants': {name: dict(base) for name in TRANSFORMS}}], 'native': []}
    assert len(transport_cases(manifest, {'one': entry})) == 8
    changed = json.loads(json.dumps(manifest))
    changed['controlled'][0]['variants'].pop('pcm16_only')
    with pytest.raises(ValueError, match='membership'):
        transport_cases(changed, {'one': entry})
    changed = json.loads(json.dumps(manifest))
    changed['controlled'][0]['variants']['float_identity']['pcm_f32le_sha256'] = 'changed'
    with pytest.raises(ValueError, match='Float identity'):
        transport_cases(changed, {'one': entry})

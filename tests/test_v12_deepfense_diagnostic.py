import copy

import numpy as np
import pytest
import torch

from artifactbench.models.deepfense import DeepFenseModel
from artifactbench.v12.common import rank
from artifactbench.v12.diagnose_deepfense_inputs import (
    capture_adapter, check_reference_runtime, comparison, control_waveforms, describe_waveform,
    score_output, select_cases, upstream_waveform,
)


def test_selection_is_metadata_only_hash_order_and_excludes_demos():
    rows = [{'id': str(i), 'partition': p, 'source': s, 'label': label}
            for i, (p, s, label) in enumerate([
                ('legacy', 'real_a', 'real'), ('legacy', 'real_a', 'real'),
                ('legacy', 'real_b', 'real'), ('legacy', 'ai_a', 'ai'),
                ('legacy', 'ai_b', 'ai'), ('contemporary_native', 'native_a', 'ai'),
                ('official_demo', 'demo', 'ai')])]
    chosen = select_cases(rows)
    assert chosen == select_cases(list(reversed(rows)))
    assert {c['group'] for c in chosen} == {'legacy_real:real_a', 'legacy_real:real_b', 'legacy_ai', 'native:native_a'}
    assert next(c['entry']['id'] for c in chosen if c['group'] == 'legacy_ai') == min(('3', '4'), key=rank)
    with pytest.raises(ValueError, match='Duplicate'):
        select_cases(rows+[rows[0]])


def test_runtime_only_allows_device_and_subset_differences():
    prior = {'device': 'cuda', 'gpu': 'original GPU', 'selected_ids': ['a', 'b'], 'expected': 2,
             'model': 'deepfense', 'packages': [['torch', '2.8.0']], 'code_hashes': {'a': 'original'}}
    current = dict(prior, device='cpu', gpu=None, selected_ids=['a'], expected=1)
    check_reference_runtime(current, prior)
    changed = copy.deepcopy(current)
    changed['code_hashes']['a'] = 'changed'
    with pytest.raises(ValueError, match='code_hashes'):
        check_reference_runtime(changed, prior)
    with pytest.raises(ValueError, match='CPU diagnostic'):
        check_reference_runtime(prior, prior)


def test_output_uses_labelled_logits_and_rejects_invalid_output():
    logits = torch.tensor([[1., 2.]])
    assert score_output({'logits': logits, 'probs': torch.tensor([.999])}, 0)['p_ai'] == torch.softmax(logits, -1)[0, 0].item()
    with pytest.raises(ValueError, match='two-class'):
        score_output({'probs': logits}, 0)
    with pytest.raises(ValueError, match='two-class'):
        score_output({'logits': torch.tensor([[float('nan'), 2.]])}, 0)


def test_capture_observes_actual_adapter_input_and_removes_hook_on_error():
    class UnitDetector(torch.nn.Module):
        def forward(self, audio):
            return {'logits': torch.tensor([[1., 2.]])}
    model = DeepFenseModel()
    model.model = UnitDetector()
    audio = np.ones(44100, dtype=np.float32)
    waveform, result = capture_adapter(model, audio, 'unit')
    assert waveform.shape == (64000,) and result['waveform'] == describe_waveform(waveform)
    assert result['p_ai'] == model.forward(audio)
    assert not model.model._forward_hooks
    class FailingDetector(torch.nn.Module):
        def forward(self, audio):
            raise RuntimeError('unit failure')
    model.model = FailingDetector()
    with pytest.raises(RuntimeError, match='unit failure'):
        capture_adapter(model, audio, 'unit')
    assert not model.model._forward_hooks


def test_upstream_input_error_is_preserved_without_fallback_or_score():
    calls = []
    def loader(path, **kwargs):
        calls.append((path, kwargs))
        raise RuntimeError('unsupported unit format')
    def padder(*args, **kwargs):
        raise AssertionError('Must not pad a failed input')
    waveform, result = upstream_waveform('unit.m4a', loader, padder)
    assert waveform is None and result['outcome'] == 'upstream_input_error'
    assert 'p_ai' not in result and len(calls) == 1
    assert calls[0][1] == {'target_sr': 16000, 'mono': True}


def test_upstream_uses_validation_padding_and_casts_without_extra_clipping():
    def loader(path, **kwargs):
        return np.ones(4, dtype=np.float64)*1.01
    def padder(audio, **kwargs):
        assert kwargs == {'max_len': 64000, 'random_pad': False, 'pad_type': 'repeat'}
        return np.tile(audio, 16000)
    waveform, result = upstream_waveform('unit.wav', loader, padder)
    assert waveform.dtype == np.float32 and result['waveform']['peak'] > 1


def test_comparison_direction_and_boundary():
    result = comparison(.49, .5)
    assert result['raw_05_flip'] and result['signed_shift'] == pytest.approx(.01)
    with pytest.raises(ValueError, match='Invalid'):
        comparison(.5, 1.001)


def test_resampler_control_changes_only_final_resampler_and_keeps_replay_exact(monkeypatch):
    common = np.linspace(-.8, .8, 44100, dtype=np.float32)
    before = common.copy()
    captured = np.full(64000, .3, dtype=np.float32)
    def resample(audio, **kwargs):
        assert audio is common
        assert kwargs == {'orig_sr': 44100, 'target_sr': 16000, 'res_type': 'soxr_hq', 'fix': True, 'scale': False}
        return np.full(16000, 1.01, dtype=np.float32)
    def padder(audio, **kwargs):
        assert kwargs == {'max_len': 64000, 'random_pad': False, 'pad_type': 'repeat'}
        return np.tile(audio, 4)
    monkeypatch.setattr('artifactbench.v12.diagnose_deepfense_inputs.librosa.resample', resample)
    result = control_waveforms(common, captured, padder)
    np.testing.assert_array_equal(common, before)
    np.testing.assert_array_equal(result['benchmark_input_replay'], captured)
    assert result['common44_librosa'].dtype == np.float32
    assert result['common44_librosa'].max() > 1  # 추가 clip을 넣지 않는다.
    result['benchmark_input_replay'][0] = -1
    assert captured[0] == np.float32(.3)

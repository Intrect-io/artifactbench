"""입력 경로/identity 경계용 unit fixture. 실제 detector 성능 값이 아니다."""
import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from artifactbench.v12 import deepfense_transport as native
from artifactbench.v12.deepfense_native import CONTRACT
from artifactbench.v12.transport_statistics import primary_comparisons


def binding_fixture():
    entry = {'id': 'p', 'sha256': 'primary'}
    local = [dict(entry, path='/unit/primary.mp3')]
    pair = {'primary_id': 'p', 'variant_id': 'v', 'primary_sha256': 'primary',
            'sha256': 'variant', 'primary_path': '/unit/primary.mp3', 'path': '/unit/variant.mp4'}
    published = {k: v for k, v in pair.items() if k not in ('path', 'primary_path')}
    manifest = {'native': [{'id': 'p-v', 'entry': entry, 'native_pair': pair}]}
    cases = [{'kind': 'cross_codec_native', 'case_id': 'p-v', 'view': view, 'entry': entry,
              'cache': {'path': '/unit/do-not-read.npy'}} for view in ('primary', 'variant')]
    return cases, manifest, local, [pair], [published]


def test_native_views_bind_encoded_files_not_common_rate_cache():
    cases = native.bind_native_cases(*binding_fixture())
    assert cases[0]['encoded_input'] == {'path': '/unit/primary.mp3', 'sha256': 'primary'}
    assert cases[1]['encoded_input'] == {'path': '/unit/variant.mp4', 'sha256': 'variant'}


@pytest.mark.parametrize('change', ('primary_path', 'public_hash', 'missing_pair', 'duplicate_pair', 'case_id'))
def test_native_binding_rejects_mismatched_or_incomplete_inventory(change):
    cases, manifest, local, pairs, published = copy.deepcopy(binding_fixture())
    if change == 'primary_path':
        local[0]['path'] = '/unit/other.mp3'
    elif change == 'public_hash':
        published[0]['sha256'] = 'wrong'
    elif change == 'missing_pair':
        manifest['native'] = []
    elif change == 'duplicate_pair':
        pairs.append(pairs[0])
    else:
        manifest['native'][0]['id'] = 'unbound'
    with pytest.raises(ValueError):
        native.bind_native_cases(cases, manifest, local, pairs, published)


def test_native_inference_uses_primary_entry_seed_and_preserves_input_failure(monkeypatch):
    case = native.bind_native_cases(*binding_fixture())[1]
    def entry(model, entry):
        assert entry['id'] == 'p' and entry['path'] == '/unit/variant.mp4' and entry['sha256'] == 'variant'
        return {'id': 'p', 'audio_sha256': 'variant', 'seed': 7,
                'outcome': 'input_execution_error', 'error_type': 'ValueError'}
    monkeypatch.setattr(native, 'infer_entry', entry)
    monkeypatch.setattr(native, 'digest', lambda p: 'variant')
    got = native.infer_native_view(None, case)
    assert got == {'seed': 7, 'outcome': 'input_execution_error', 'error_type': 'ValueError'}
    monkeypatch.setattr(native, 'digest', lambda p: 'changed')
    with pytest.raises(ValueError, match='changed'):
        native.infer_native_view(None, case)


def test_controlled_master_promoted_before_preparing_without_old_adapter(monkeypatch):
    wave = np.arange(12, dtype=np.float32)/20
    model = SimpleNamespace(forward=lambda _: pytest.fail('Old adapter must not execute'))
    def prepare(audio, sr):
        assert audio.dtype == np.float64 and sr == 44100
        np.testing.assert_array_equal(audio, wave)
        return np.zeros(64000, np.float32), {'condition': CONTRACT['name']}
    monkeypatch.setattr(native, 'prepare_waveform', prepare)
    monkeypatch.setattr(native, 'score_prepared', lambda model, audio: (.25, [-1., 1.]))
    case = {'kind': 'controlled', 'entry': {'id': 'unit'}}
    first, second = native.infer_native_view(model, case, wave), native.infer_native_view(model, case, wave)
    assert first['prob'] == second['prob'] == .25 and first['seed'] == second['seed']
    assert first['input_preprocessing']['decoder'] == 'controlled_float32_master_promoted_f64'
    def fail(model, audio):
        raise torch.cuda.OutOfMemoryError('unit infrastructure')
    monkeypatch.setattr(native, 'score_prepared', fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        native.infer_native_view(model, case, wave)


def test_wrong_native_reference_condition_rejected_before_loading_package():
    with pytest.raises(ValueError, match='condition'):
        native.native_reference_identity({'model': 'deepfense'}, {'input_condition': {'name': 'wrong'}})


def test_native_reference_reconstructs_current_package_hashes_without_copying_reference(tmp_path, monkeypatch):
    from artifactbench.v12.common import digest
    source = tmp_path/'__init__.py'
    source.write_text('# unit package source\n')
    monkeypatch.setitem(sys.modules, 'deepfense', SimpleNamespace(__file__=str(source)))
    base = {'model': 'deepfense', 'model_assets': {'source_repos': {}}, 'code_hashes': {}}
    result = native.native_reference_identity(base, {'input_condition': CONTRACT})
    assert result['model_assets']['source_repos']['DeepFense-package']['python_files'] == {str(source): digest(source)}
    assert base['model_assets']['source_repos'] == {} and base['code_hashes'] == {}
    assert len(result['code_hashes']) == 3 and result['input_condition'] == CONTRACT
    source.write_text('# changed unit package source\n')
    updated = native.native_reference_identity(base, {'input_condition': CONTRACT})
    assert result['model_assets'] != updated['model_assets']


def test_common_master_shift_is_not_primary_repeatability():
    result = primary_comparisons([
        {'primary_run_comparison': {'score_difference': .8, 'repeatability_eligible': False}},
        {'primary_run_comparison': {'score_difference': 0., 'repeatability_eligible': True}},
        {'primary_run_comparison': {'score_difference': None, 'repeatability_eligible': True}},
    ])
    assert result['primary_repeatability'] == {'comparable_cases': 1, 'max_absolute_difference': 0.}
    assert result['common_master_input_path_difference'] == {'comparable_cases': 1, 'max_absolute_difference': .8}
    assert primary_comparisons([{'primary_run_comparison': {'score_difference': .01}}])['primary_repeatability']['comparable_cases'] == 1

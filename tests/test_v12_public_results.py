"""출력 경계 unit fixture. 아래 값은 실제 detector/benchmark 결과가 아니다."""
import hashlib
import json

import pytest

from artifactbench.v12.public_results import probability, public_record, public_preprocessing, relative_hashes
from artifactbench.v12 import public_results


def test_original_deepfense_audit_cannot_be_promoted_to_final_public_comparison():
    with pytest.raises(ValueError, match='original adapter remains an audit'):
        public_results.require_final_input_condition({'model': 'deepfense'}, {'unit': row()})
    public_results.require_final_input_condition({'model': 'artifactnet'}, {'unit': row()})


def test_corrected_condition_keeps_failed_input_coverage_without_inventing_metadata():
    identity = {'model': 'deepfense', 'input_condition': {'name': 'deepfense-native-soxr/2'}}
    public_results.require_final_input_condition(identity, {'unit': {'outcome': 'input_execution_error'}})
    with pytest.raises(ValueError, match='different native input condition'):
        public_results.require_final_input_condition(identity, {'unit': row()})
    with pytest.raises(ValueError, match='different native input condition'):
        public_results.require_final_input_condition(identity, {'unit': row() | {
            'input_preprocessing': {'condition': 'deepfense-native-soxr/1'}}})


def row():
    identifier = 'unit-record'
    return {'id': identifier, 'label': 'ai', 'source': 'unit-source', 'track_id': 'unit-track',
            'partition': 'legacy', 'recording_group': 'unit-group', 'recording_representative': True,
            'dependence_cluster': 'unit-cluster', 'audio_sha256': '0'*64,
            'seed': int(hashlib.sha256(('260905:'+identifier).encode()).hexdigest()[:8], 16),
            'outcome': 'scored', 'prob': .8}


def test_projection_preserves_values_without_private_fields():
    original = row() | {'error': '/home/private/trace', 'traceback': 'private unit marker',
                        'model_diagnostics': {'path': '/home/private/audio'}, 'decoded_frames': 10}
    result = public_record(original)
    assert result['prob'] == original['prob'] and result['decoded_frames'] == 10
    assert 'private' not in json.dumps(result)
    assert not {'error', 'traceback', 'model_diagnostics'}.intersection(result)


@pytest.mark.parametrize('value', (-.1, 1.1, float('nan'), float('inf'), True))
def test_invalid_probability_rejected(value):
    with pytest.raises(ValueError, match='probability'):
        probability(value)


def test_failed_entry_is_explicit_and_never_gets_a_probability():
    original = row()
    original.pop('prob')
    original.update(outcome='model_execution_error', error_type='RuntimeError',
                    error='private unit marker', traceback='/home/private/code.py')
    result = public_record(original)
    assert result['outcome'] == 'model_execution_error' and 'prob' not in result
    original['prob'] = .5
    with pytest.raises(ValueError, match='Failed prediction'):
        public_record(original)


def test_wrong_seed_and_stack_decision_are_rejected():
    with pytest.raises(ValueError, match='seed'):
        public_record(row() | {'seed': 1})
    with pytest.raises(ValueError, match='decision'):
        public_record(row() | {'rescue_prob': .8, 'production_ai': False, 'n_chunks': 1})


def test_relative_hashes_preserve_subdirectories_and_reject_escape():
    result = relative_hashes({'/unit/repo/a/model.py': 'a', '/unit/repo/b/model.py': 'b'}, '/unit/repo')
    assert result == {'a/model.py': 'a', 'b/model.py': 'b'}
    with pytest.raises(ValueError):
        relative_hashes({'/outside/model.py': 'c'}, '/unit/repo')


def test_smoke_export_is_not_a_full_result(tmp_path, monkeypatch):
    monkeypatch.setattr(public_results, 'verify', lambda _: {'unit_fixture': True})
    (tmp_path/'summary.json').write_text(json.dumps({'status': 'SMOKE EXPORT WIRING ONLY',
        'reference_statistics': {'replicates': 20}}))
    with pytest.raises(ValueError, match='scope mismatch'):
        public_results.load_public_results(tmp_path, tmp_path)


def test_empty_reference_inventory_cannot_claim_reproduction(tmp_path, monkeypatch):
    monkeypatch.setattr(public_results, 'verify', lambda _: {'unit_fixture': True})
    (tmp_path/'summary.json').write_text(json.dumps({'status': 'SMOKE EXPORT WIRING ONLY',
        'reference_statistics': {'replicates': 20, 'file_sha256': {}}}))
    with pytest.raises(ValueError, match='nine reference'):
        public_results.load_public_results(tmp_path, tmp_path, allow_smoke=True)


def test_native_input_failure_remains_an_input_failure_without_private_text():
    original = row()
    original.pop('prob')
    original.update(outcome='input_execution_error', error_type='ValueError', error='/private/path')
    projected = public_record(original)
    assert projected['outcome'] == 'input_execution_error' and projected['error_type'] == 'ValueError'
    assert 'error' not in projected and 'prob' not in projected


@pytest.mark.parametrize('condition,decoder', [('deepfense-native-soxr/1', 'ffmpeg_aac_native_f64'),
    ('deepfense-native-soxr/2', 'ffmpeg_aac_native_f64'), ('deepfense-native-soxr/2', 'ffmpeg_opus_native_f64')])
def test_native_preprocessing_projection_keeps_route_and_hash_without_paths(condition, decoder):
    value = {'condition': condition, 'decoder': decoder,
             'upstream_decoder_error_type': 'LibsndfileError', 'native_sample_rate': 48000,
             'native_channels': 2, 'native_frames': 1000, 'resampled_frames': 334,
             'model_sample_rate': 16000, 'model_frames': 64000, 'model_input_sha256': 'a'*64,
             'peak': 1.05, 'rms': .1, 'path': '/private/audio', 'waveform': [1., 2.]}
    projected = public_preprocessing(value)
    assert projected['peak'] > 1 and 'path' not in projected and 'waveform' not in projected
    assert public_record(row() | {'input_preprocessing': value})['input_preprocessing'] == projected
    with pytest.raises(ValueError, match='decoder route'):
        public_preprocessing(value | {'decoder': 'unannounced_decoder'})
    with pytest.raises(ValueError, match='size/rate'):
        public_preprocessing(value | {'native_sample_rate': True})
    with pytest.raises(ValueError, match='contract'):
        public_preprocessing(value | {'condition': 'deepfense-native-soxr/3'})
    with pytest.raises(ValueError, match='decoder route'):
        public_preprocessing(value | {'condition': 'deepfense-native-soxr/1', 'decoder': 'ffmpeg_opus_native_f64'})

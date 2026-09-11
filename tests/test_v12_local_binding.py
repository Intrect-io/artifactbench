"""경로·바이트 연결용 unit fixture. 오디오나 실제 벤치마크 점수를 생성하지 않는다."""
import json
from pathlib import Path

import pytest

from artifactbench.v12 import bind_local_audio as binding
from artifactbench.v12.common import digest, write_once
from artifactbench.v12.verify_release import verify


@pytest.fixture
def local_release(tmp_path):
    release = tmp_path/'public-release'
    release.mkdir()
    primary, native = tmp_path/'primary.bytes', tmp_path/'variant.bytes'
    primary.write_bytes(b'unit primary byte identity, not audio')
    native.write_bytes(b'unit variant byte identity, not audio')
    entry = {'id': 'unit-primary', 'sha256': digest(primary), 'bytes': primary.stat().st_size,
             'partition': 'legacy', 'source': 'unit-source', 'label': 'real',
             'recording_group': 'unit-group', 'recording_representative': True,
             'dependence_cluster': 'unit-cluster', 'duration_seconds': 1, 'sample_rate': 44100}
    pair = {'primary_id': entry['id'], 'variant_id': 'unit-variant',
            'primary_sha256': entry['sha256'], 'sha256': digest(native)}
    write_once(release/'manifest.public.json', {'schema': 'unit-schema', 'bench': [entry]})
    write_once(release/'native_pairs.public.json', [pair])
    write_once(release/'exposure.public.json', [])
    write_once(release/'release.json', {'evaluation_entries': 1, 'partitions': {'legacy': 1},
        'source_counts': {'unit-source': 1}, 'labels': {'real': 1}, 'identified_recording_groups': 1,
        'dependence_clusters': 1, 'native_transport_pairs': 1,
        'public_file_sha256': {name: digest(release/name) for name in binding.PUBLIC_FILES if name != 'release.json'}})
    path_map = tmp_path/'paths.local.json'
    write_once(path_map, {'schema': 'artifactbench-local-paths/1', 'primary': {'unit-primary': primary.name},
                         'native': {'unit-primary-unit-variant': str(native)}})
    return release, path_map, tmp_path/'bound-release', primary, native


def change_map(path_map, change):
    document = json.loads(path_map.read_text())
    change(document)
    path_map.write_text(json.dumps(document))


def test_complete_binding_preserves_public_bytes_and_resolves_from_map_directory(local_release, monkeypatch):
    release, path_map, output, primary, native = local_release
    before = {name: digest(release/name) for name in binding.PUBLIC_FILES}
    monkeypatch.chdir(release)
    result = binding.bind_audio(release, path_map, output)
    assert result['status'] == 'local audio byte binding complete'
    assert result['primary_bound'] == result['native_pairs_bound'] == 1
    assert result['outcomes'] == {'bound': 2}
    assert before == {name: digest(output/name) for name in binding.PUBLIC_FILES}
    assert before == {name: digest(release/name) for name in binding.PUBLIC_FILES}
    rows = json.loads((output/'manifest.local.json').read_text())['bench']
    public = json.loads((release/'manifest.public.json').read_text())['bench']
    assert rows == [dict(public[0], path=str(primary))]
    pairs = json.loads((output/'native_pairs.local.json').read_text())
    assert pairs[0]['path'] == str(native) and pairs[0]['primary_path'] == str(primary)
    assert verify(output)['manifest_sha256'] == before['manifest.public.json']


@pytest.mark.parametrize('kind', ('primary', 'native'))
def test_missing_path_never_produces_runnable_manifest(local_release, kind):
    release, path_map, output, _, _ = local_release
    change_map(path_map, lambda document: document[kind].clear())
    with pytest.raises(ValueError, match='Incomplete local audio binding'):
        binding.bind_audio(release, path_map, output)
    assert not (output/'manifest.local.json').exists()
    assert not (output/'native_pairs.local.json').exists()
    summary = json.loads((output/'binding_summary.local.json').read_text())
    assert summary['outcomes'] == {'bound': 1, 'not_provided': 1}
    assert summary['native_pairs_bound'] == 0


@pytest.mark.parametrize('kind', ('primary', 'native'))
def test_same_size_wrong_bytes_are_not_bound(local_release, kind):
    release, path_map, output, primary, native = local_release
    wrong = primary if kind == 'primary' else native
    wrong.write_bytes(b'x'*wrong.stat().st_size)
    with pytest.raises(ValueError, match='Incomplete'):
        binding.bind_audio(release, path_map, output)
    audit = json.loads((output/'binding_audit.local.json').read_text())
    assert next(row for row in audit if row['kind'] == kind)['outcome'] == 'hash_mismatch'
    assert not (output/'manifest.local.json').exists()


def test_wrong_size_is_reported_without_hashing_file(tmp_path, monkeypatch):
    path = tmp_path/'unit.bytes'
    path.write_bytes(b'unit')
    def unexpected_hash(_):
        pytest.fail('Size mismatch must not read the entire file')
    monkeypatch.setattr(binding, 'digest', unexpected_hash)
    assert binding.inspect_mapping(path.name, tmp_path, '0'*64, 100)['outcome'] == 'size_mismatch'


@pytest.mark.parametrize('value', ('', ' ', 17, [], 'https://example.org/audio.wav', 'a\0b'))
def test_invalid_path_is_explicit(tmp_path, value):
    assert binding.inspect_mapping(value, tmp_path, '0'*64)['outcome'] == 'invalid_local_path'


def test_missing_file_directory_and_null_are_explicit(tmp_path):
    assert binding.inspect_mapping(None, tmp_path, '0'*64)['outcome'] == 'not_provided'
    assert binding.inspect_mapping('missing.bytes', tmp_path, '0'*64)['outcome'] == 'not_a_file'
    assert binding.inspect_mapping(str(tmp_path), tmp_path, '0'*64)['outcome'] == 'not_a_file'


@pytest.mark.parametrize('document', ([], None, 17, {'schema': 'wrong', 'primary': {}, 'native': {}},
                                     {'schema': 'artifactbench-local-paths/1', 'primary': [], 'native': {}}))
def test_invalid_mapping_schema_rejected(tmp_path, document):
    path = tmp_path/'map.json'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='schema'):
        binding.read_mapping(path, {}, {})


def test_duplicate_map_keys_are_rejected(tmp_path):
    path = tmp_path/'map.json'
    path.write_text('{"schema":"artifactbench-local-paths/1","primary":{"id":"a","id":"b"},"native":{}}')
    with pytest.raises(ValueError, match='Duplicate key'):
        binding.read_mapping(path, {'id'}, {})


@pytest.mark.parametrize('kind', ('primary', 'native'))
def test_unknown_ids_fail_before_creating_output(local_release, kind):
    release, path_map, output, _, _ = local_release
    change_map(path_map, lambda document: document[kind].update({'unknown-id': None}))
    with pytest.raises(ValueError, match='Unknown '+kind):
        binding.bind_audio(release, path_map, output)
    assert not output.exists()


def test_existing_output_is_never_overwritten(local_release):
    release, path_map, output, _, _ = local_release
    output.mkdir()
    (output/'keep.txt').write_text('existing user artifact')
    with pytest.raises(FileExistsError):
        binding.bind_audio(release, path_map, output)
    assert (output/'keep.txt').read_text() == 'existing user artifact'
    assert list(output.iterdir()) == [output/'keep.txt']


def test_output_cannot_be_in_frozen_release_even_via_symlink(local_release):
    release, path_map, output, _, _ = local_release
    output.symlink_to(release, target_is_directory=True)
    with pytest.raises(ValueError, match='separate'):
        binding.bind_audio(release, path_map, output/'child')
    assert not (release/'child').exists()


def test_audio_symlinks_are_resolved_to_verified_target(tmp_path):
    source, link = tmp_path/'unit.bytes', tmp_path/'link.bytes'
    source.write_bytes(b'unit byte identity')
    link.symlink_to(source)
    result = binding.inspect_mapping(link.name, tmp_path, digest(source))
    assert result['outcome'] == 'bound' and result['path'] == str(source)


def test_symlink_loop_is_reported_as_access_error(tmp_path):
    loop = tmp_path/'loop'
    loop.symlink_to(loop)
    assert binding.inspect_mapping(loop.name, tmp_path, '0'*64)['outcome'] == 'file_access_error'


def test_permission_failure_is_explicit(tmp_path, monkeypatch):
    source = tmp_path/'unit.bytes'
    source.write_bytes(b'unit byte identity')
    def denied(_):
        raise PermissionError('unit permission error')
    monkeypatch.setattr(binding, 'digest', denied)
    result = binding.inspect_mapping(source.name, tmp_path, '0'*64)
    assert result['outcome'] == 'file_access_error' and result['error_type'] == 'PermissionError'


def test_change_during_file_hash_is_rejected(tmp_path, monkeypatch):
    source = tmp_path/'unit.bytes'
    source.write_bytes(b'unit byte identity')
    checksum = digest(source)
    def changed(path):
        actual = digest(path)
        path.write_bytes(b'changed during unit hash')
        return actual
    monkeypatch.setattr(binding, 'digest', changed)
    assert binding.inspect_mapping(source.name, tmp_path, checksum)['outcome'] == 'changed_during_hash'


@pytest.mark.parametrize('target', ('map', 'public'))
def test_metadata_or_map_mutation_prevents_manifest_publication(local_release, monkeypatch, target):
    release, path_map, output, _, _ = local_release
    original = binding.inspect_mapping
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path = path_map if target == 'map' else release/'exposure.public.json'
        path.write_text(path.read_text()+'\n')
        return result
    monkeypatch.setattr(binding, 'inspect_mapping', changed)
    with pytest.raises(ValueError, match='changed during binding'):
        binding.bind_audio(release, path_map, output)
    assert not (output/'manifest.local.json').exists()
    assert not (output/'binding_summary.local.json').exists()


def test_map_change_while_reading_is_detected(local_release, monkeypatch):
    release, path_map, output, _, _ = local_release
    original = binding.read_mapping
    def changed(*args):
        result = original(*args)
        path_map.write_text(path_map.read_text()+'\n')
        return result
    monkeypatch.setattr(binding, 'read_mapping', changed)
    with pytest.raises(ValueError, match='changed during binding'):
        binding.bind_audio(release, path_map, output)
    assert not (output/'manifest.local.json').exists()


def test_invalid_public_hash_fails_before_creating_output(local_release):
    release, path_map, output, _, _ = local_release
    path = release/'manifest.public.json'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='Public file hash mismatch'):
        binding.bind_audio(release, path_map, output)
    assert not output.exists()

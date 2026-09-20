"""아카이브 경계만 공격하는 unit fixture. 실제 결과 재현은 별도 실측한다."""
import json
import stat
import zipfile

import pytest

from artifactbench.v12.package_results import (
    SCOPE, bundle_name, expected_files, extract_results_bundle, verify_results_bundle,
)


def unit_manifest(directory, smoke=True, files=None):
    document = {'schema': 'artifactbench-results-bundle/1', 'scope': SCOPE, 'smoke_only': smoke,
                'file_sha256': {name: '0'*64 for name in expected_files()} if files is None else files}
    (directory/'BUNDLE_MANIFEST.json').write_text(json.dumps(document))


def unit_zip(path, members):
    with zipfile.ZipFile(path, 'x') as archive:
        for name, mode in members:
            member = zipfile.ZipInfo(name)
            member.create_system = 3
            member.external_attr = mode << 16
            archive.writestr(member, b'unit boundary fixture, not benchmark data')


def test_smoke_bundle_cannot_be_verified_as_full(tmp_path):
    unit_manifest(tmp_path)
    with pytest.raises(ValueError, match='scope'):
        verify_results_bundle(tmp_path)


def test_empty_results_inventory_rejected(tmp_path):
    unit_manifest(tmp_path, files={})
    with pytest.raises(ValueError, match='inventory'):
        verify_results_bundle(tmp_path, True)


def test_unlisted_payload_rejected_before_data_loading(tmp_path):
    unit_manifest(tmp_path)
    (tmp_path/'private.json').write_text('{}')
    with pytest.raises(ValueError, match='Unlisted or missing'):
        verify_results_bundle(tmp_path, True)


def test_symlink_manifest_rejected(tmp_path):
    marker = tmp_path/'unit.json'
    marker.write_text('{}')
    (tmp_path/'BUNDLE_MANIFEST.json').symlink_to(marker)
    with pytest.raises(ValueError, match='Symbolic'):
        verify_results_bundle(tmp_path, True)


def test_symlink_payload_rejected(tmp_path):
    unit_manifest(tmp_path)
    (tmp_path/'linked.json').symlink_to(tmp_path/'BUNDLE_MANIFEST.json')
    with pytest.raises(ValueError, match='Symbolic'):
        verify_results_bundle(tmp_path, True)


def test_payload_hash_checked_before_scores(tmp_path):
    unit_manifest(tmp_path)
    for name in expected_files():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('unit hash fixture')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_results_bundle(tmp_path, True)


@pytest.mark.parametrize('name', ('../escape', '/absolute', 'a/../escape', 'a\\escape'))
def test_unsafe_zip_path_rejected_before_extraction(tmp_path, name):
    archive = tmp_path/'unit.zip'
    unit_zip(archive, [(name, stat.S_IFREG | 0o644)])
    with pytest.raises(ValueError, match='Unsafe'):
        extract_results_bundle(archive, tmp_path/'extracted', True)
    assert not (tmp_path/'extracted').exists()


@pytest.mark.parametrize('mode', (stat.S_IFLNK | 0o777, stat.S_IFDIR | 0o755, stat.S_IFIFO | 0o600))
def test_nonregular_zip_member_rejected(tmp_path, mode):
    archive = tmp_path/'unit.zip'
    unit_zip(archive, [('unit', mode)])
    with pytest.raises(ValueError, match='Non-regular'):
        extract_results_bundle(archive, tmp_path/'extracted', True)


def test_duplicate_zip_member_rejected(tmp_path):
    archive = tmp_path/'unit.zip'
    with pytest.warns(UserWarning, match='Duplicate'):
        unit_zip(archive, [('unit', stat.S_IFREG | 0o644)]*2)
    with pytest.raises(ValueError, match='Duplicate'):
        extract_results_bundle(archive, tmp_path/'extracted', True)


def test_unlisted_zip_member_rejected_before_extraction(tmp_path):
    archive = tmp_path/'unit.zip'
    unit_zip(archive, [('private.json', stat.S_IFREG | 0o644)])
    with pytest.raises(ValueError, match='inventory'):
        extract_results_bundle(archive, tmp_path/'extracted', True)
    assert not (tmp_path/'extracted').exists()


def test_extraction_cannot_overwrite_existing_directory(tmp_path):
    (tmp_path/'existing.txt').write_text('preserve')
    with pytest.raises(ValueError, match='must be empty'):
        extract_results_bundle(tmp_path/'not-even-opened.zip', tmp_path, True)
    assert (tmp_path/'existing.txt').read_text() == 'preserve'


def test_fixed_inventory_and_smoke_name_are_explicit():
    names = expected_files()
    assert 'predictions/artifactnet.json' in names
    assert 'statistics/paired_differences.json' in names
    assert 'notebooks/v1.2_results_reproduction.ipynb' in names
    assert not any(name.endswith(('.wav', '.mp3', '.pt', '.html')) for name in names)
    assert bundle_name(True) == 'artifactbench-results-SMOKE-ONLY'
    assert bundle_name(False) == 'artifactbench-1.2-rc2-results'

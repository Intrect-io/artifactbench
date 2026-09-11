"""단위 fixture로 경계만 검사한다. 가중치를 로드하거나 benchmark 결과를 만들지 않는다."""
import json
from types import SimpleNamespace

import pytest

from artifactbench.v12 import verify_external_assets as module
from artifactbench.v12.common import digest


def unit_inputs(tmp_path, monkeypatch):
    revision = 'a'*40
    snapshot = tmp_path/revision
    snapshot.mkdir()
    (snapshot/'unit.bin').write_bytes(b'unit fixture; not model weights')
    root = tmp_path/'source'
    root.mkdir()
    (root/'unit.py').write_text('raise RuntimeError("This fixture must never be imported")\n')
    (root/'head.pth').write_bytes(b'unit checkpoint fixture; not a model')
    expected = {'repo': 'unit/source', 'revision': revision, 'files': {'unit.bin': digest(snapshot/'unit.bin')}}
    assets = {'snapshots': {'unit/source': expected}, 'source_repos': {'MoM-CLAM': {
        'git_head': revision, 'python_files': {'unit.py': digest(root/'unit.py')}}},
        'files': {'MoM-CLAM/head.pth': digest(root/'head.pth')}}
    reference = tmp_path/'reference.json'
    reference.write_text(json.dumps({'model': 'clam', 'identity': {
        'model': 'clam', 'expected': 32, 'smoke_per_source': 1, 'model_assets': assets}}))
    locations = tmp_path/'snapshots.local.json'
    locations.write_text(json.dumps([dict(expected, snapshot=str(snapshot))]))
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout=revision+'\n'))
    return reference, locations, {'MoM-CLAM': root}


def test_asset_match_does_not_claim_fresh_inference(tmp_path, monkeypatch):
    reference, locations, roots = unit_inputs(tmp_path, monkeypatch)
    result = module.verify(reference, locations, roots)
    assert result['passed']
    assert len(result['checks']) == 5
    assert result['reference_smoke_per_source'] == 1
    assert 'no model execution' in result['scope']


@pytest.mark.parametrize('change', ['weight', 'python', 'extra_python', 'missing_root', 'revision', 'hf_bytes', 'missing_snapshot', 'git'])
def test_differences_are_explicit_nonpasses(tmp_path, monkeypatch, change):
    reference, locations, roots = unit_inputs(tmp_path, monkeypatch)
    if change == 'weight':
        (roots['MoM-CLAM']/'head.pth').write_text('<html>download confirmation</html>')
    elif change == 'python':
        (roots['MoM-CLAM']/'unit.py').write_text('# changed source fixture\n')
    elif change == 'extra_python':
        (roots['MoM-CLAM']/'extra.py').write_text('# extra source fixture\n')
    elif change == 'missing_root':
        roots = {}
    elif change == 'revision':
        rows = json.loads(locations.read_text())
        rows[0]['revision'] = 'b'*40
        locations.write_text(json.dumps(rows))
    elif change == 'hf_bytes':
        rows = json.loads(locations.read_text())
        (tmp_path/rows[0]['revision']/'unit.bin').write_bytes(b'changed fixture')
    elif change == 'missing_snapshot':
        locations.write_text('[]')
    elif change == 'git':
        monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='b'*40))
    result = module.verify(reference, locations, roots)
    assert not result['passed']
    assert any(row['status'] != 'match' for row in result['checks'].values())


def test_duplicate_snapshot_rejected(tmp_path, monkeypatch):
    reference, locations, roots = unit_inputs(tmp_path, monkeypatch)
    locations.write_text(json.dumps(json.loads(locations.read_text())*2))
    with pytest.raises(ValueError, match='Duplicate'):
        module.verify(reference, locations, roots)


@pytest.mark.parametrize('name', ['../private', '/absolute', 'a/../private', 'a\\private'])
def test_unsafe_relative_asset_rejected(tmp_path, monkeypatch, name):
    reference, locations, roots = unit_inputs(tmp_path, monkeypatch)
    document = json.loads(reference.read_text())
    document['identity']['model_assets']['source_repos']['MoM-CLAM']['python_files'] = {name: '0'*64}
    reference.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='Unsafe'):
        module.verify(reference, locations, roots)


def test_file_changing_during_hash_is_not_match(tmp_path, monkeypatch):
    path = tmp_path/'fixture.bin'
    path.write_bytes(b'unit-before')
    checksum = digest(path)
    def mutate(target):
        path.write_bytes(b'unit-after-different-size')
        return checksum
    monkeypatch.setattr(module, 'digest', mutate)
    assert module.checked_file(path, checksum)['status'] == 'changed_during_read'


def test_failure_cli_preserves_private_audit_and_nonzero_exit(tmp_path, monkeypatch):
    reference, locations, roots = unit_inputs(tmp_path, monkeypatch)
    output = tmp_path/'audit'
    monkeypatch.setattr('sys.argv', ['verify', '--reference', str(reference), '--snapshots', str(locations), '--output', str(output)])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 1
    assert not json.loads((output/'audit.local.json').read_text())['passed']

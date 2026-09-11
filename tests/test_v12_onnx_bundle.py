"""번들 경계만 검사하는 단위 fixture. 실제 모델/벤치마크 결과가 아니다."""
import json
from pathlib import Path

import pytest

from artifactbench.v12.common import digest
from artifactbench.v12.package_artifactnet_onnx import check_bound_sources, inspect_graph
from artifactbench.v12.verify_onnx_bundle import FILES, FROZEN_MANIFEST, score, verify, verify_parity


def unit_parity():
    rows = [{'id': f'{i:024x}', 'audio_sha256': '0'*64, 'source': 'unit-fixture-only',
             'chunks': 1, 'onnx_p_ai': .2, 'reference_p_ai': .2, 'reference_outcome': 'scored'}
            for i in range(33)]
    rows.append({'id': 'cfae4e3c1d5317fe58fe6422', 'audio_sha256': '0'*64,
                 'source': 'unit-fixture-only', 'chunks': 1, 'onnx_p_ai': .2,
                 'reference_outcome': 'model_execution_error', 'reference_error_type': 'RuntimeError'})
    return {'schema': 'artifactnet-public-parity/1', 'scope': 'source-smoke',
            'public_manifest_sha256': FROZEN_MANIFEST, 'onnx_sha256': '0'*64,
            'records': rows, 'maximum_absolute_error': 0., 'raw_05_decision_flips': 0,
            'original_failed_rows_retained': 1}


def unit_bundle(directory):
    for name in FILES:
        path = directory/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('UNIT FIXTURE: not an ONNX model or actual benchmark result\n')
    checksum = digest(directory/'artifactnet_raw.onnx')
    report = unit_parity()
    report['onnx_sha256'] = checksum
    (directory/'PARITY_REPORT.json').write_text(json.dumps(report))
    (directory/'artifactnet_raw.json').write_text(json.dumps({
        'schema': 'artifactnet-raw-onnx/1', 'onnx_sha256': checksum}))
    manifest = {'schema': 'artifactnet-onnx-candidate-bundle/1',
                'publication_status': 'local candidate; not uploaded', 'parity_scope': 'source-smoke',
                'file_sha256': {name: digest(directory/name) for name in FILES}}
    (directory/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize('value', [True, False, -.1, 1.00000001, float('inf'), float('nan'), '0.5', None])
def test_invalid_probability_rejected(value):
    with pytest.raises(ValueError, match='probability'):
        score(value)


def test_scope_is_not_upgraded_from_smoke():
    report = unit_parity()
    assert verify_parity(report)['comparable'] == 33
    report['scope'] = 'full-rc2'
    with pytest.raises(ValueError, match='membership'):
        verify_parity(report)


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'error', 'flip', 'lie', 'failure', 'chunks', 'id'])
def test_bad_parity_evidence_rejected(change):
    report = unit_parity()
    if change == 'duplicate':
        report['records'][0] = report['records'][1]
    elif change == 'missing':
        report['records'].pop()
    elif change == 'error':
        report['records'][0]['onnx_p_ai'] += .01
    elif change == 'flip':
        report['records'][0].update(onnx_p_ai=.5001, reference_p_ai=.4999)
        report['maximum_absolute_error'] = .5001-.4999
        report['raw_05_decision_flips'] = 1
    elif change == 'lie':
        report['maximum_absolute_error'] = .0001
    elif change == 'failure':
        report['records'][-1]['reference_p_ai'] = .2
    elif change == 'chunks':
        report['records'][0]['chunks'] = True
    elif change == 'id':
        report['records'][-1]['id'] = 'f'*24
    with pytest.raises(ValueError):
        verify_parity(report)


def test_stdlib_bundle_check_explicitly_does_not_claim_inference(tmp_path):
    unit_bundle(tmp_path)
    result = verify(tmp_path)
    assert result['files'] == 11
    assert result['parity']['scope'] == 'source-smoke'
    assert 'No detector execution' in result['scope']


@pytest.mark.parametrize('change', ['extra', 'missing', 'changed', 'metadata', 'scope', 'publication', 'symlink'])
def test_bundle_mutations_rejected(tmp_path, change):
    manifest = unit_bundle(tmp_path)
    if change == 'extra':
        (tmp_path/'private.wav').write_text('unit fixture')
    elif change == 'missing':
        (tmp_path/'README.md').rename(tmp_path/'wrong-name.md')
    elif change == 'changed':
        (tmp_path/'README.md').write_text('changed unit fixture')
    elif change == 'metadata':
        (tmp_path/'artifactnet_raw.json').write_text(json.dumps({
            'schema': 'artifactnet-raw-onnx/1', 'onnx_sha256': 'f'*64}))
        manifest['file_sha256']['artifactnet_raw.json'] = digest(tmp_path/'artifactnet_raw.json')
    elif change == 'scope':
        manifest['parity_scope'] = 'full-rc2'
    elif change == 'publication':
        manifest['publication_status'] = 'published'
    elif change == 'symlink':
        (tmp_path/'linked').symlink_to(tmp_path/'README.md')
    (tmp_path/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(tmp_path)


def test_bound_runtime_accepts_relative_absolute_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'runtime.py').write_text('# unit runtime fixture\n')
    inputs = {'source_file_sha256': {'runtime.py': digest(tmp_path/'runtime.py')}}
    check_bound_sources(inputs, [tmp_path/'runtime.py'])
    (tmp_path/'runtime.py').write_text('# changed unit runtime fixture\n')
    with pytest.raises(ValueError, match='binding mismatch'):
        check_bound_sources(inputs, [Path('runtime.py')])


def test_duplicate_source_bindings_cannot_hide_old_hash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'runtime.py').write_text('# unit fixture\n')
    inputs = {'source_file_sha256': {'runtime.py': '0'*64,
                                     str(tmp_path/'runtime.py'): digest(tmp_path/'runtime.py')}}
    with pytest.raises(ValueError, match='Conflicting'):
        check_bound_sources(inputs, [tmp_path/'runtime.py'])


@pytest.mark.parametrize('output_name', ['p_ai', 'private_diagnostics'])
def test_actual_onnx_contract_inspection(tmp_path, output_name):
    onnx = pytest.importorskip('onnx')
    helper = onnx.helper
    # 실제 ONNX 직렬화/검사만 확인하는 상수 graph이며 검출기 대역이 아니다.
    value = helper.make_tensor('unit', onnx.TensorProto.DOUBLE, [1], [.2])
    graph = helper.make_graph([helper.make_node('Constant', [], [output_name], value=value)], 'unit-only',
        [helper.make_tensor_value_info('audio_chunks', onnx.TensorProto.FLOAT, ['chunks', 176400])],
        [helper.make_tensor_value_info(output_name, onnx.TensorProto.DOUBLE, [1])])
    path = tmp_path/'unit.onnx'
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)]), str(path))
    if output_name == 'p_ai':
        assert inspect_graph(path)['external_initializers'] == 0
        graph.input[0].type.tensor_type.shape.dim[1].dim_value = 1
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)]), str(path))
        with pytest.raises(ValueError, match='type or shape'):
            inspect_graph(path)
    else:
        with pytest.raises(ValueError, match='single-output'):
            inspect_graph(path)

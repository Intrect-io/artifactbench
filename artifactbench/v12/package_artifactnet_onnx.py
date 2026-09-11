"""완료된 실제 parity에 묶인 ONNX/최소 runtime만 로컬 후보 ZIP으로 묶는다."""
import argparse
import json
from pathlib import Path
import re
import zipfile

from .common import digest, write_once
from .package_metadata import public_payload
from .report import load_completed
from .validate_artifactnet_onnx import selection
from .verify_onnx_bundle import FILES, verify, verify_parity
from .verify_release import check_public, verify as verify_release


def check_bound_sources(inputs, paths):
    # 상대/절대 표기가 달라도 같은 파일이어야 하며 중복 기대값은 충돌할 수 없다.
    expected = {}
    for name, checksum in inputs['source_file_sha256'].items():
        path = Path(name).resolve()
        if path in expected and expected[path] != checksum:
            raise ValueError('Conflicting parity source bindings')
        expected[path] = checksum
    for path in paths:
        path = Path(path)
        if expected.get(path.resolve()) != digest(path):
            raise ValueError('Parity input binding mismatch: '+path.name)


def public_parity(model, parity_dir, release, reference_run):
    verified = verify_release(release)
    public = json.loads((release/'manifest.public.json').read_text())['bench']
    originals, identity = load_completed(reference_run, {r['id']: r for r in public})
    summary = json.loads((parity_dir/'summary.json').read_text())
    inputs = json.loads((parity_dir/'inputs.local.json').read_text())
    is_smoke = inputs['source_smoke']
    if type(is_smoke) is not bool:
        raise ValueError('Invalid parity scope marker')
    required_status = 'source-balanced smoke parity measured' if is_smoke else 'full rc2 ONNX parity measured'
    if (summary['status'] != required_status or summary['passed'] is not True
            or summary['full_parity_complete'] is not (not is_smoke)
            or summary['onnx_sha256'] != digest(model) or identity['model'] != 'artifactnet'
            or identity['public_manifest_sha256'] != verified['manifest_sha256']):
        raise ValueError('A successful completed parity run on this model and release is required')
    expected = {r['id'] for r in selection(public, is_smoke)}
    if (len(inputs['selected_ids']) != len(expected) or set(inputs['selected_ids']) != expected
            or set(summary['record_sha256']) != expected or summary['attempted'] != len(expected)):
        raise ValueError('Parity selection differs from the fixed score-blind selection')
    required_sources = (model, model.with_suffix('.json'), reference_run/'identity.json', reference_run/'summary.json')
    check_bound_sources(inputs, required_sources)
    public_by_id = {r['id']: r for r in public}
    result = {'schema': 'artifactnet-public-parity/1', 'scope': 'source-smoke' if is_smoke else 'full-rc2',
        'public_manifest_sha256': verified['manifest_sha256'], 'onnx_sha256': digest(model),
        'reference_run_summary_sha256': digest(reference_run/'summary.json'),
        'parity_summary_sha256': digest(parity_dir/'summary.json'), 'records': [],
        'maximum_absolute_error': summary['maximum_absolute_error'],
        'raw_05_decision_flips': summary['raw_05_decision_flips'],
        'original_failed_rows_retained': summary['original_failed_rows_retained']}
    for identifier in sorted(expected):
        path = parity_dir/'records'/(identifier+'.json')
        if digest(path) != summary['record_sha256'][identifier]:
            raise ValueError('Parity record changed')
        row = json.loads(path.read_text())
        original = originals[identifier]
        if (row['id'] != identifier or row['audio_sha256'] != original['audio_sha256']
                or row['source'] != public_by_id[identifier]['source']
                or row['reference_outcome'] != original['outcome']):
            raise ValueError('Parity/reference recording identity differs')
        clean = {key: row[key] for key in ('id', 'source', 'audio_sha256', 'chunks', 'onnx_p_ai', 'reference_outcome')}
        if original['outcome'] == 'scored':
            if row['original_gpu_p_ai'] != original['prob']:
                raise ValueError('Parity substituted the original reference score')
            clean['reference_p_ai'] = original['prob']
        else:
            clean['reference_error_type'] = original['error_type']
        result['records'].append(clean)
    verify_parity(result)
    check_public(result)
    return result


def inspect_graph(model):
    import onnx
    document = onnx.load(str(model), load_external_data=False)
    onnx.checker.check_model(document, full_check=True)
    graph = document.graph
    if (len(graph.input) != 1 or graph.input[0].name != 'audio_chunks'
            or len(graph.output) != 1 or graph.output[0].name != 'p_ai'
            or any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.initializer)
            or document.functions or document.training_info or any(n.domain for n in graph.node)):
        raise ValueError('Expected self-contained standard-operator single-output inference graph')
    input_type, output_type = graph.input[0].type.tensor_type, graph.output[0].type.tensor_type
    input_shape = [d.dim_param or d.dim_value for d in input_type.shape.dim]
    output_shape = [d.dim_param or d.dim_value for d in output_type.shape.dim]
    if (input_type.elem_type != onnx.TensorProto.FLOAT or input_shape != ['chunks', 176400]
            or output_type.elem_type != onnx.TensorProto.DOUBLE or output_shape != [1]):
        raise ValueError('Unexpected ONNX input/output type or shape')
    if re.search(rb'/(?:home|media|Users|Volumes)/[A-Za-z0-9_.-]', model.read_bytes()):
        raise ValueError('Machine-local path in ONNX bytes')
    return {'nodes': len(graph.node), 'initializers': len(graph.initializer), 'external_initializers': 0,
            'inputs': [graph.input[0].name], 'outputs': [graph.output[0].name]}


def build(root, model, parity_dir, release, reference_run, output):
    root = Path(root)
    model, parity_dir, release, reference_run, output = map(Path, (model, parity_dir, release, reference_run, output))
    report = public_parity(model, parity_dir, release, reference_run)
    graph = inspect_graph(model)
    sources = {name: root/name for name in FILES if name.startswith('artifactbench/')}
    sources.update({'README.md': root/'docs/ARTIFACTNET_MODEL_CARD_DRAFT.md',
        'RUNTIME_LICENSE.txt': root/'LICENSE', 'requirements.txt': root/'requirements-onnx-runtime.lock',
        'artifactnet_raw.onnx': model, 'artifactnet_raw.json': model.with_suffix('.json')})
    for path in sources.values():
        if path.resolve() != path.absolute() or not path.is_file():
            raise ValueError('Candidate source missing or symbolic')
    # 검증 뒤 바뀐 실행 코드가 검증된 모델과 함께 묶이지 않도록 고정한다.
    inputs = json.loads((parity_dir/'inputs.local.json').read_text())
    check_bound_sources(inputs, [root/'artifactbench/v12/artifactnet_onnx.py',
                                 root/'artifactbench/v12/common.py'])
    hashes = {name: digest(path) for name, path in sources.items()}
    payloads = {name: path.read_bytes() if name.endswith('.onnx') else public_payload(path)
                for name, path in sources.items()}
    name = 'artifactnet-raw-'+('SOURCE-SMOKE' if report['scope'] == 'source-smoke' else 'FULL-PARITY')+'-CANDIDATE'
    output.mkdir(parents=True, exist_ok=False)
    directory = output/name
    directory.mkdir()
    for filename, payload in payloads.items():
        target = directory/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    write_once(directory/'PARITY_REPORT.json', report)
    hashes['PARITY_REPORT.json'] = digest(directory/'PARITY_REPORT.json')
    write_once(directory/'BUNDLE_MANIFEST.json', {'schema': 'artifactnet-onnx-candidate-bundle/1',
        'publication_status': 'local candidate; not uploaded', 'parity_scope': report['scope'],
        'file_sha256': hashes, 'graph_inspection': graph,
        'model_license_status': 'Owner decision required before publication; runtime MIT does not cover weights'})
    verification = verify(directory)
    archive = output/(name+'.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for filename in sorted(FILES | {'BUNDLE_MANIFEST.json'}):
            member = zipfile.ZipInfo(name+'/'+filename, (2026, 9, 5, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            stream.writestr(member, (directory/filename).read_bytes(), compresslevel=9)
    if any(digest(path) != hashes[filename] for filename, path in sources.items()):
        raise ValueError('Candidate source changed during packaging')
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise ValueError('Candidate ZIP integrity failure')
    summary = {'status': 'local ONNX candidate packaged; not uploaded', 'archive': archive.name,
        'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size, 'verification': verification}
    write_once(output/'summary.json', summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'parity', 'release', 'reference-run', 'output'):
        ap.add_argument('--'+name, required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[2], args.model, args.parity, args.release,
                           args.reference_run, args.output), indent=2))


if __name__ == '__main__':
    main()

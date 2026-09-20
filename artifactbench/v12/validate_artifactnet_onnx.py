"""새 ONNX를 실제 rc2 오디오/저장 점수와 대조한다. 원래 run은 절대 수정하지 않는다."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .artifactnet_onnx import RawOnnx, song_chunks
from .common import digest, rank, write_once
from .report import load_completed
from .verify_release import verify


def selection(entries, source_smoke):
    if not source_smoke:
        return sorted(entries, key=lambda row: rank(row['id']))
    # 검출 점수는 보지 않는다. source별 hash-first + 길이 최소/최대 경계만 선택한다.
    first = {}
    for row in sorted(entries, key=lambda row: rank(row['id'])):
        first.setdefault(row['source'], row)
    selected = {row['id']: row for row in first.values()}
    for row in (min(entries, key=lambda r: (r['duration_seconds'], rank(r['id']))),
                max(entries, key=lambda r: (r['duration_seconds'], rank(r['id'])))):
        selected[row['id']] = row
    return sorted(selected.values(), key=lambda row: rank(row['id']))


def compare(got, reference, tolerance):
    return {'absolute_error': abs(got-reference), 'decision_flip_05': bool((got >= .5) != (reference >= .5)),
            'within_tolerance': bool(abs(got-reference) <= tolerance)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--reference-run', required=True, type=Path)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--source-smoke', action='store_true', help='Source-balanced engineering check, not full parity')
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This verifier is CPU-only; set CUDA_VISIBLE_DEVICES to an empty string')
    import torch
    from .audio import load_audio_float
    from . import audio as audio_module, artifactnet_onnx
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    verified = verify(args.release)
    public = json.loads((args.release/'manifest.public.json').read_text())['bench']
    local = {row['id']: row for row in json.loads((args.release/'manifest.local.json').read_text())['bench']}
    references, identity = load_completed(args.reference_run, {r['id']: r for r in public})
    export_inputs = json.loads((args.model.parent/'export_inputs.local.json').read_text())
    if (identity['model'] != 'artifactnet' or identity['public_manifest_sha256'] != verified['manifest_sha256']
            or digest(args.reference_run/'identity.json') != export_inputs['reference_identity_sha256']
            or digest(args.reference_run/'summary.json') != export_inputs['reference_summary_sha256']):
        raise ValueError('Export/reference/release identity mismatch')
    unchanged = dict(export_inputs['source_file_sha256'])
    unchanged.update(identity['code_hashes'])
    for repository in identity['model_assets']['source_repos'].values():
        unchanged.update(repository['python_files'])
    # 현재 hash를 추가하기 전에 과거 export/run의 기대값부터 확인한다.
    # 같은 key를 현재 값으로 덮어써 source drift 검사를 무력화하면 안 된다.
    if any(digest(path) != checksum for path, checksum in unchanged.items()):
        raise ValueError('Export/reference implementation changed')
    export_summary = json.loads((args.model.parent/'export_summary.json').read_text())
    if digest(args.model) != export_summary['onnx_sha256']:
        raise ValueError('Artifact differs from the completed export')
    unchanged.update({str(path): digest(path) for path in (Path(__file__), Path(audio_module.__file__),
        Path(artifactnet_onnx.__file__), args.model, args.model.with_suffix('.json'),
        args.release/'manifest.local.json', args.release/'manifest.public.json',
        args.reference_run/'summary.json', args.reference_run/'identity.json')})
    if any(digest(path) != checksum for path, checksum in unchanged.items()):
        raise ValueError('Export/reference implementation changed')
    policy = export_inputs['parity_policy']
    if policy['absolute_score_tolerance'] != 1e-3 or policy['maximum_raw_05_decision_flips'] != 0:
        raise ValueError('Unexpected predeclared parity policy')
    selected = selection(public, args.source_smoke)
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output/'inputs.local.json', {'source_file_sha256': unchanged,
        'selected_ids': [r['id'] for r in selected], 'policy': policy, 'source_smoke': args.source_smoke,
        'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu',
        'scope': 'Original GPU predictions preserved; CPU parity is a separate replication condition'})
    model = RawOnnx(args.model)
    reference_model = None
    if args.source_smoke:
        from artifactbench.models.artifactnet import ArtifactNetModel
        repo = Path(identity['model_assets']['source_repos']['ArtifactNet']['root']).parent
        reference_model = ArtifactNetModel(repo_dir=str(repo))
        reference_model.load('cpu')
    rows = []
    for index, entry in enumerate(selected, 1):
        original, path = references[entry['id']], Path(local[entry['id']]['path'])
        if digest(path) != entry['sha256']:
            raise ValueError('Audio hash changed: '+entry['id'])
        waveform = load_audio_float(path)
        if len(waveform) != original['decoded_frames']:
            raise ValueError('Decoder frame count changed: '+entry['id'])
        chunks = song_chunks(waveform)
        start = time.monotonic()
        got = model.predict_chunks(chunks)
        row = {'id': entry['id'], 'source': entry['source'], 'audio_sha256': entry['sha256'],
            'chunks': len(chunks), 'onnx_p_ai': got, 'onnx_seconds': time.monotonic()-start,
            'reference_outcome': original['outcome']}
        if original['outcome'] == 'scored':
            row['original_gpu_comparison'] = compare(got, original['prob'], policy['absolute_score_tolerance'])
            row['original_gpu_p_ai'] = original['prob']
        else:
            row['original_error_type'] = original['error_type']
            row['scope'] = 'Original failure retained; new probability-only graph output is not a comparable reference score'
        if reference_model is not None:
            # 현재 adapter의 실제 chunk 생성과 배열 자체를 비교한다.
            from .artifactnet_onnx import CHUNK_SAMPLES, MAX_SAMPLES
            reference_audio = np.ascontiguousarray(waveform[:MAX_SAMPLES], dtype=np.float32)
            if len(reference_audio) < CHUNK_SAMPLES:
                reference_audio = np.pad(reference_audio, (0, CHUNK_SAMPLES-len(reference_audio)))
            expected_chunks = reference_model._sliding(torch.from_numpy(reference_audio)).numpy()
            if not np.array_equal(chunks, expected_chunks):
                raise ValueError('Original adapter chunk policy differs')
            try:
                cpu = reference_model.forward(waveform)
            except RuntimeError as exc:
                if original['outcome'] == 'scored' or original['error_type'] != 'RuntimeError':
                    raise
                row['cpu_reference_error'] = {'type': type(exc).__name__, 'message': str(exc)}
            else:
                row['cpu_reference_p_ai'] = cpu
                row['cpu_reference_comparison'] = compare(got, cpu, policy['absolute_score_tolerance'])
        # Torch 없는 실행 환경을 검사할 실제 입력 하나를 private evidence에 보존한다.
        if args.source_smoke and index == 1:
            np.save(args.output/'runtime_waveform.private.npy', waveform, allow_pickle=False)
            row['runtime_waveform_sha256'] = digest(args.output/'runtime_waveform.private.npy')
        write_once(args.output/'records'/(entry['id']+'.json'), row)
        rows.append(row)
        print(json.dumps({'checked': index, 'expected': len(selected), 'chunks': len(chunks),
            'onnx_seconds': row['onnx_seconds'], 'original_gpu_comparison': row.get('original_gpu_comparison'),
            'cpu_reference_comparison': row.get('cpu_reference_comparison')}), flush=True)
    if any(digest(path) != checksum for path, checksum in unchanged.items()):
        raise ValueError('Inputs changed during ONNX verification')
    comparisons = [r['original_gpu_comparison'] for r in rows if 'original_gpu_comparison' in r]
    cpu_comparisons = [r['cpu_reference_comparison'] for r in rows if 'cpu_reference_comparison' in r]
    passed = all(r['within_tolerance'] and not r['decision_flip_05'] for r in comparisons+cpu_comparisons)
    summary = {'status': 'source-balanced smoke parity measured' if args.source_smoke else 'full rc2 ONNX parity measured',
        'passed': passed, 'full_parity_complete': not args.source_smoke and passed,
        'attempted': len(rows), 'comparable_original_scores': len(comparisons),
        'original_failed_rows_retained': len(rows)-len(comparisons),
        'maximum_absolute_error': max(r['absolute_error'] for r in comparisons),
        'raw_05_decision_flips': sum(r['decision_flip_05'] for r in comparisons),
        'cpu_reference_comparisons': len(cpu_comparisons),
        'maximum_cpu_reference_absolute_error': max((r['absolute_error'] for r in cpu_comparisons), default=None),
        'onnx_sha256': digest(args.model), 'record_sha256': {r['id']: digest(args.output/'records'/(r['id']+'.json')) for r in rows},
        'scope': 'Separate CPU replication; no rescue-stack reproduction, no upload, no changed benchmark results'}
    write_once(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    if not passed:
        raise ValueError('Predeclared parity tolerance or decision agreement failed; inspect records')


if __name__ == '__main__':
    main()

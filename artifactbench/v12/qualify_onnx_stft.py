"""보존된 표준-STFT probe를 기존 metadata-first 34개 표본으로 검증한다. 공개 승격은 하지 않는다."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .artifactnet_onnx import RawOnnx, song_chunks
from .common import digest, write_once
from .report import load_completed
from .validate_artifactnet_onnx import compare, selection
from .verify_release import verify


def prepare_check(probe, parent, reference_run, release):
    verify(release)
    public = json.loads((release/'manifest.public.json').read_text())['bench']
    local = {r['id']: r for r in json.loads((release/'manifest.local.json').read_text())['bench']}
    references, identity = load_completed(reference_run, {r['id']: r for r in public})
    exported = json.loads((parent/'export_inputs.local.json').read_text())
    measured = json.loads((probe/'summary.json').read_text())
    if (identity['model'] != 'artifactnet' or
            identity['public_manifest_sha256'] != digest(release/'manifest.public.json') or
            digest(reference_run/'identity.json') != exported['reference_identity_sha256'] or
            digest(reference_run/'summary.json') != exported['reference_summary_sha256']):
        raise ValueError('Original export/reference identity differs')
    artifact = probe/'stft_probe.onnx'
    if measured['status'] != 'single-recording STFT probe measured' or digest(artifact) != measured['model_sha256']:
        raise ValueError('Expected the unchanged measured STFT probe')
    policy = exported['parity_policy']
    if policy['absolute_score_tolerance'] != .001 or policy['maximum_raw_05_decision_flips'] != 0:
        raise ValueError('Cannot relax the original parity policy')
    pins = dict(exported['source_file_sha256'])
    prior_probe = json.loads((probe/'inputs.local.json').read_text())['source_file_sha256']
    for mapping in (prior_probe, identity['code_hashes'],
                    *(r['python_files'] for r in identity['model_assets']['source_repos'].values())):
        for path, sha in mapping.items():
            if path in pins and pins[path] != sha:
                raise ValueError('Conflicting source identity')
            pins[path] = sha
    if any(digest(path) != sha for path, sha in pins.items()):
        raise ValueError('Preserved export/probe/reference source changed')
    selected = selection(public, True)
    if set(local) != {r['id'] for r in public}:
        raise ValueError('Local/public release membership differs')
    for row in selected:
        if local[row['id']]['sha256'] != row['sha256'] or digest(local[row['id']]['path']) != row['sha256']:
            raise ValueError('Selected frozen audio differs')
    for path in (Path(__file__), Path(__file__).with_name('validate_artifactnet_onnx.py'),
                 Path(__file__).with_name('artifactnet_onnx.py'), Path(__file__).with_name('report.py'),
                 artifact, artifact.with_suffix('.json'), probe/'summary.json',
                 probe/'inputs.local.json', parent/'export_inputs.local.json',
                 release/'manifest.local.json', release/'manifest.public.json', release/'release.json',
                 reference_run/'summary.json', reference_run/'identity.json'):
        if str(path) in pins and pins[str(path)] != digest(path):
            raise ValueError('Pinned input changed before qualification')
        pins[str(path)] = digest(path)
    return selected, local, references, identity, pins, policy


def measure_entry(runtime, reference_model, waveform, original, tolerance=.001):
    chunks = song_chunks(waveform)
    started = time.monotonic()
    score = runtime.predict_chunks(chunks)
    row = {'chunks': len(chunks), 'onnx_p_ai': score, 'onnx_seconds': time.monotonic()-started,
           'reference_outcome': original['outcome']}
    if original['outcome'] == 'scored':
        row.update(original_gpu_p_ai=original['prob'], original_gpu_comparison=compare(score, original['prob'], tolerance))
    else:
        row['original_error_type'] = original['error_type']
    try:
        cpu = float(reference_model.forward(waveform))
    except RuntimeError as exc:
        if original['outcome'] == 'scored' or original['error_type'] != type(exc).__name__:
            raise
        row['cpu_reference_error_type'] = type(exc).__name__
    else:
        if not np.isfinite(cpu) or not 0 <= cpu <= 1:
            raise ValueError('Invalid original CPU probability')
        row.update(factory_cpu_p_ai=cpu, factory_cpu_comparison=compare(score, cpu, tolerance))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'parent', 'reference-run', 'release', 'snapshots', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('CPU-only/offline qualification is required')
    import torch
    from .audio import load_audio_float
    from .model_assets import build_pinned
    selected, local, references, reference_identity, pins, policy = prepare_check(
        args.probe, args.parent, args.reference_run, args.release)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model, assets = build_pinned('artifactnet', args.snapshots)
    if assets != reference_identity['model_assets']:
        raise ValueError('Factory CPU model assets differ from the original primary')
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output/'inputs.local.json', {'selected_ids': [r['id'] for r in selected],
        'source_file_sha256': pins, 'policy': policy, 'device': 'cpu', 'torch': torch.__version__,
        'cpu_reference': 'build_pinned ArtifactNet, GPU_BATCH_MAX=1, not direct-adapter default batch8',
        'scope': 'Metadata/source/boundary qualification only; no export_summary or public release promotion'})
    model.load('cpu')
    runtime = RawOnnx(args.probe/'stft_probe.onnx')
    rows = []
    for number, entry in enumerate(selected, 1):
        path = local[entry['id']]['path']
        if digest(path) != entry['sha256']:
            raise ValueError('Frozen audio changed')
        waveform = load_audio_float(path)
        original = references[entry['id']]
        if len(waveform) != original['decoded_frames']:
            raise ValueError('Original decoder frame count differs')
        row = dict(measure_entry(runtime, model, waveform, original),
                   id=entry['id'], source=entry['source'], audio_sha256=entry['sha256'])
        if digest(path) != entry['sha256']:
            raise ValueError('Frozen audio changed during inference')
        write_once(args.output/'records'/(entry['id']+'.json'), row)
        rows.append(row)
        print(json.dumps({'checked': number, 'expected': len(selected), **row}), flush=True)
    if any(digest(path) != sha for path, sha in pins.items()):
        raise ValueError('Qualification source/input changed')
    comparisons = {}
    for name in ('original_gpu_comparison', 'factory_cpu_comparison'):
        values = [r[name] for r in rows if name in r]
        comparisons[name] = {'comparable': len(values), 'failures': sum(not r['within_tolerance'] for r in values),
            'flips': sum(r['decision_flip_05'] for r in values),
            'max_absolute_error': max((r['absolute_error'] for r in values), default=None)}
    write_once(args.output/'summary.json', {'status': 'fixed-source STFT qualification complete; not full parity',
        'attempted': len(rows), 'comparisons': comparisons, 'inputs_sha256': digest(args.output/'inputs.local.json'),
        'record_sha256': {r['id']: digest(args.output/'records'/(r['id']+'.json')) for r in rows},
        'limits': 'CPU provider only; retain original failed coverage; full scored-reference qualification still required'})


if __name__ == '__main__':
    main()

"""원래 full run을 보존하고 별도 버전의 native-rate 입력 조건을 전수 평가한다."""
import argparse
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import soundfile as sf
import torch

from .common import digest, rank, write_once
from .deepfense_native import CONDITION, CONTRACT, load_native, score_prepared
from .model_assets import build_pinned
from .report import load_completed
from .run import measured_parameters, metrics, run_identity, seed_record, snapshot_sources
from .verify_release import verify


def check_original_condition(current, reference):
    """새 입력 경로를 추가하기 전 원래 실행과 device/선택 이외의 차이를 금지한다."""
    ignored = {'device', 'gpu', 'expected', 'selected_ids', 'smoke_per_source', 'scope'}
    current = {k: v for k, v in json.loads(json.dumps(current)).items() if k not in ignored}
    original = {k: v for k, v in reference.items() if k not in ignored}
    if current != original:
        changed = sorted(k for k in set(current) | set(original) if current.get(k) != original.get(k))
        raise ValueError('Native evaluation differs beyond the declared input/device condition: '+', '.join(changed))


def selected_entries(local, public, per_source):
    if (len(local) != len(public) or len({r['id'] for r in local}) != len(local)
            or {r['id'] for r in local} != set(public)):
        raise ValueError('Frozen membership mismatch')
    for entry in local:
        if any(entry[k] != public[entry['id']][k] for k in ('sha256', 'source', 'label', 'track_id',
                'partition', 'dependence_cluster', 'recording_group', 'recording_representative')):
            raise ValueError('Frozen local/public row mismatch')
    selected = sorted(local, key=lambda r: rank(r['id']))
    if per_source:
        counts, kept = Counter(), []
        for entry in selected:
            if counts[entry['source']] < per_source:
                kept.append(entry)
                counts[entry['source']] += 1
        selected = kept
    return selected


def infer_entry(model, entry):
    """입력 실패와 모델 실패를 분리하고 OOM/손상된 CUDA 상태는 실행 중단으로 남긴다."""
    if digest(entry['path']) != entry['sha256']:
        raise ValueError('Frozen audio hash changed: '+entry['id'])
    row = {key: entry[key] for key in ('id', 'source', 'track_id', 'label', 'partition',
           'recording_group', 'recording_representative', 'dependence_cluster')}
    row.update(audio_sha256=entry['sha256'], seed=seed_record(entry['id']))
    started = time.monotonic()
    try:
        audio, preprocessing = load_native(entry['path'])
    except (sf.LibsndfileError, RuntimeError, ValueError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, KeyError) as exc:
        row.update(outcome='input_execution_error', error_type=type(exc).__name__,
                   error=str(exc), traceback=traceback.format_exc(), decode_seconds=time.monotonic()-started)
        return row
    decoded = time.monotonic()
    row.update(input_preprocessing=preprocessing, decoded_frames=preprocessing['native_frames'],
               decode_seconds=decoded-started)
    try:
        probability, logits = score_prepared(model, audio)
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid probability')
        row.update(outcome='scored', prob=probability, model_diagnostics={'logits': logits, 'spoof_index': 0})
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, ValueError, TypeError, IndexError, ZeroDivisionError) as exc:
        if isinstance(exc, RuntimeError) and any(s in str(exc).lower() for s in
                ('cuda error', 'illegal memory access', 'device-side assert')):
            raise
        row.update(outcome='model_execution_error', error_type=type(exc).__name__,
                   error=str(exc), traceback=traceback.format_exc())
    row['inference_seconds'] = time.monotonic()-decoded
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'snapshots', 'reference-run', 'output'):
        ap.add_argument('--'+name, required=True, type=Path)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    ap.add_argument('--smoke-per-source', type=int, default=0)
    args = ap.parse_args()
    if args.smoke_per_source < 0:
        ap.error('smoke-per-source must be nonnegative')
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('HF_HUB_OFFLINE=1 is required')
    if args.device == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU evaluation requires CUDA_VISIBLE_DEVICES empty')
    verify(args.release)
    public = {r['id']: r for r in json.loads((args.release/'manifest.public.json').read_text())['bench']}
    local = json.loads((args.release/'manifest.local.json').read_text())['bench']
    selected = selected_entries(local, public, args.smoke_per_source)
    reference, previous = load_completed(args.reference_run, public)
    if set(reference) != set(public) or previous['model'] != 'deepfense':
        raise ValueError('Expected the complete original DeepFense run')
    args.model = 'deepfense'
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_record('model-initialization')
    model, original_assets = build_pinned('deepfense', args.snapshots)
    identity = run_identity(model, original_assets, args, selected)
    check_original_condition(identity, previous)
    import deepfense
    from . import deepfense_native
    package = Path(deepfense.__file__).parent
    package_sources = {str(p): digest(p) for p in package.rglob('*.py')}
    assets = copy.deepcopy(original_assets)
    assets['source_repos']['DeepFense-package'] = {'root': str(package), 'git_head': None,
                                                  'python_files': package_sources}
    identity.update(model_assets=assets, input_condition=CONTRACT,
        audio=CONDITION+': native float64 mono, direct librosa soxr_hq 16k, upstream prefix/repeat, float32; no clip; AAC/Opus-only native FFmpeg extension',
        scope='separately versioned reference-consistent input condition; original adapter scores retained; official demos separate',
        reference_summary_sha256=digest(args.reference_run/'summary.json'),
        reference_identity_sha256=digest(args.reference_run/'identity.json'))
    identity['code_hashes'].update({str(p): digest(p) for p in
        (Path(__file__), Path(deepfense_native.__file__), Path(__file__).with_name('report.py'))})
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output/'identity.json', identity)
    snapshot_sources(args.output, identity)
    model.load(args.device)
    write_once(args.output/'model_info.json', dict(model.info(), parameter_measurement=measured_parameters(model)))
    rows, consecutive_errors = [], 0
    for number, entry in enumerate(selected, 1):
        try:
            row = infer_entry(model, entry)
        except torch.cuda.OutOfMemoryError as exc:
            write_once(args.output/('infrastructure-'+entry['id']+'.json'),
                       {'outcome': 'cuda_oom_not_a_prediction', 'id': entry['id'], 'error': str(exc)})
            raise
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Audio changed during native evaluation')
        if args.device == 'cuda':
            row['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        write_once(args.output/'records'/(entry['id']+'.json'), row)
        rows.append(row)
        consecutive_errors = 0 if row['outcome'] == 'scored' else consecutive_errors+1
        print(json.dumps({'condition': CONDITION, 'attempted': number, 'expected': len(selected),
                          'source': entry['source'], 'outcome': row['outcome']}), flush=True)
        if consecutive_errors >= 5:
            raise RuntimeError('Five consecutive input/model failures; inspect before continuing')
    _, end_assets = build_pinned('deepfense', args.snapshots)
    if end_assets != original_assets or any(digest(path) != checksum for path, checksum in
            dict(identity['code_hashes'], **package_sources).items()):
        raise ValueError('Native evaluation source/assets changed')
    for path, expected in ((args.release/'manifest.local.json', identity['manifest_sha256']),
            (args.release/'manifest.public.json', identity['public_manifest_sha256']),
            (args.release/'release.json', identity['release_sha256']),
            (args.reference_run/'summary.json', identity['reference_summary_sha256']),
            (args.reference_run/'identity.json', identity['reference_identity_sha256'])):
        if digest(path) != expected:
            raise ValueError('Native evaluation reference changed')
    partitions = defaultdict(list)
    for row in rows:
        partitions[row['partition']].append(row)
    summary = {'model': 'deepfense', 'input_condition': CONDITION,
        'status': 'smoke complete' if args.smoke_per_source else 'full fixed-checkpoint run complete',
        'attempted': len(rows), 'expected': len(selected), 'outcomes': dict(Counter(r['outcome'] for r in rows)),
        'decoder_routes': dict(Counter(r.get('input_preprocessing', {}).get('decoder', 'input_failure') for r in rows)),
        'partitions': {part: metrics(group) for part, group in partitions.items()},
        'per_source': {source: metrics([r for r in rows if r['source'] == source]) for source in sorted({r['source'] for r in rows})},
        'identity_sha256': digest(args.output/'identity.json'),
        'record_sha256': {r['id']: digest(args.output/'records'/(r['id']+'.json')) for r in rows}}
    write_once(args.output/'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('per_source', 'record_sha256')}), flush=True)


if __name__ == '__main__':
    main()

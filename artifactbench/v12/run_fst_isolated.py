"""FST를 파일별 독립 CUDA worker로 실행해 장시간 GPU 점유를 명시적으로 제한한다.

원래 ``run`` 출력은 바꾸지 않는다. 이 실행기는 새 디렉터리에서만 사용하며,
각 worker가 단 하나의 고정 입력을 처리한다. worker가 시간 상한을 넘으면 점수는
만들지 않고 ``model_execution_error``로 보존한다.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

from .audio import load_audio_float
from .common import digest, rank, write_once
from .model_assets import build_pinned
from .run import measured_parameters, metrics, run_identity, seed_record, snapshot_sources
from .verify_release import verify


CONDITION = 'fst-isolated-worker/1'
DEFAULT_TIMEOUT_SECONDS = 600


def entries(release):
    local = json.loads((release/'manifest.local.json').read_text())['bench']
    public = {r['id']: r for r in json.loads((release/'manifest.public.json').read_text())['bench']}
    if len(local) != len(public) or {r['id'] for r in local} != set(public):
        raise ValueError('Frozen membership mismatch')
    for entry in local:
        if any(entry[k] != public[entry['id']][k] for k in
               ('sha256', 'source', 'label', 'track_id', 'partition', 'recording_group',
                'recording_representative', 'dependence_cluster')):
            raise ValueError('Frozen local/public row mismatch')
    return sorted(local, key=lambda row: rank(row['id']))


def infer_one(model, entry, device):
    if digest(entry['path']) != entry['sha256']:
        raise ValueError('Frozen audio hash changed: '+entry['id'])
    row = {key: entry[key] for key in ('id', 'source', 'track_id', 'label', 'partition',
           'recording_group', 'recording_representative', 'dependence_cluster')}
    started = time.monotonic()
    audio = load_audio_float(entry['path'])
    decoded = time.monotonic()
    row.update(audio_sha256=entry['sha256'], seed=seed_record(entry['id']),
               decoded_frames=len(audio), decode_seconds=decoded-started)
    try:
        probability = float(model.forward(audio))
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid probability')
        row.update(outcome='scored', prob=probability, model_diagnostics=model.last_diagnostics)
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, ValueError, IndexError, TypeError, ZeroDivisionError) as exc:
        if isinstance(exc, RuntimeError) and any(part in str(exc).lower() for part in
                ('cuda error', 'illegal memory access', 'device-side assert')):
            raise
        row.update(outcome='model_execution_error', error_type=type(exc).__name__,
                   error=str(exc), traceback=traceback.format_exc())
    row['inference_seconds'] = time.monotonic()-decoded
    if device == 'cuda':
        row['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
    if digest(entry['path']) != entry['sha256']:
        raise ValueError('Audio changed during inference: '+entry['id'])
    return row


def worker(args):
    verify(args.release)
    entry = next((row for row in entries(args.release) if row['id'] == args.entry_id), None)
    if entry is None:
        raise ValueError('Unknown frozen entry: '+args.entry_id)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_record('model-initialization')
    model, assets = build_pinned('fst', args.snapshots)
    model.load(args.device)
    row = infer_one(model, entry, args.device)
    _, current_assets = build_pinned('fst', args.snapshots)
    if current_assets != assets:
        raise ValueError('FST assets changed during isolated worker')
    write_once(args.response, row)


def timeout_row(entry, timeout_seconds):
    row = {key: entry[key] for key in ('id', 'source', 'track_id', 'label', 'partition',
           'recording_group', 'recording_representative', 'dependence_cluster')}
    row.update(audio_sha256=entry['sha256'], seed=seed_record(entry['id']),
               outcome='model_execution_error', error_type='TimeoutExpired',
               error=f'FST isolated worker exceeded fixed {timeout_seconds}-second wall-clock limit',
               timeout_seconds=timeout_seconds)
    return row


def parent(args):
    if args.timeout_seconds <= 0:
        raise ValueError('--timeout-seconds must be positive')
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('HF_HUB_OFFLINE=1 is required')
    verify(args.release)
    selected = entries(args.release)
    args.model, args.smoke_per_source = 'fst', 0
    seed_record('model-initialization')
    model, assets = build_pinned('fst', args.snapshots)
    identity = run_identity(model, assets, args, selected)
    from artifactbench.models import fst
    identity.update(input_condition={'name': CONDITION, 'timeout_seconds': args.timeout_seconds,
                                     'worker_scope': 'fresh Python process and CUDA model per fixed input'},
                    scope='separately versioned FST fixed-checkpoint evaluation; timeout yields no score')
    identity['code_hashes'].update({str(Path(__file__)): digest(__file__), str(Path(fst.__file__)): digest(fst.__file__)})
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output/'identity.json', identity)
    snapshot_sources(args.output, identity)
    rows, failures = [], 0
    for number, entry in enumerate(selected, 1):
        target = args.output/'records'/(entry['id']+'.json')
        with tempfile.TemporaryDirectory(prefix='artifactbench-fst-worker-') as temporary:
            response = Path(temporary)/'response.json'
            command = [sys.executable, '-m', 'artifactbench.v12.run_fst_isolated', '--worker',
                       '--release', str(args.release), '--snapshots', str(args.snapshots),
                       '--entry-id', entry['id'], '--response', str(response), '--device', args.device]
            try:
                completed = subprocess.run(command, cwd=Path.cwd(), timeout=args.timeout_seconds,
                                           capture_output=True, text=True, check=False)
            except subprocess.TimeoutExpired:
                row = timeout_row(entry, args.timeout_seconds)
            else:
                if completed.returncode != 0 or not response.exists():
                    raise RuntimeError('FST worker failed for '+entry['id']+': '+completed.stderr[-1000:])
                row = json.loads(response.read_text())
                if row['id'] != entry['id'] or row['audio_sha256'] != entry['sha256']:
                    raise ValueError('FST worker response identity mismatch')
        write_once(target, row)
        rows.append(row)
        failures = failures+1 if row['outcome'] != 'scored' else 0
        print(json.dumps({'condition': CONDITION, 'attempted': number, 'expected': len(selected),
                          'source': entry['source'], 'outcome': row['outcome']}), flush=True)
        if failures >= 5:
            raise RuntimeError('Five consecutive FST failures; inspect before continuing')
    _, end_assets = build_pinned('fst', args.snapshots)
    if end_assets != assets or any(digest(path) != checksum for path, checksum in identity['code_hashes'].items()):
        raise ValueError('FST assets/source changed during isolated evaluation')
    for name, field in (('manifest.local.json', 'manifest_sha256'), ('manifest.public.json', 'public_manifest_sha256'), ('release.json', 'release_sha256')):
        if digest(args.release/name) != identity[field]:
            raise ValueError('Frozen evaluation input changed during FST evaluation')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['partition']].append(row)
    summary = {'model': 'fst', 'expected': len(selected), 'attempted': len(rows),
        'outcomes': dict(Counter(row['outcome'] for row in rows)),
        'partitions': {part: metrics(group) for part, group in grouped.items()},
        'per_source': {source: metrics([r for r in rows if r['source'] == source]) for source in sorted({r['source'] for r in rows})},
        'status': 'full fixed-checkpoint run complete', 'identity_sha256': digest(args.output/'identity.json'),
        'record_sha256': {row['id']: digest(args.output/'records'/(row['id']+'.json')) for row in rows}}
    write_once(args.output/'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--snapshots', required=True, type=Path)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--entry-id')
    parser.add_argument('--response', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout-seconds', type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.worker:
        if not args.entry_id or args.response is None or args.output is not None:
            parser.error('worker requires --entry-id and --response, and does not accept --output')
        worker(args)
    else:
        if args.output is None or args.entry_id or args.response is not None:
            parser.error('parent requires --output and does not accept worker-only arguments')
        parent(args)


if __name__ == '__main__':
    main()

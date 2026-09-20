"""고정 평가 입력·가중치로 실제 추론하며 실패와 재시작 근거를 파일별로 보존한다."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .audio import load_audio_float
from .common import digest, rank, write_once
from .model_assets import build_pinned


def seed_record(record_id):
    seed = int(hashlib.sha256(('260905:'+record_id).encode()).hexdigest()[:8], 16)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def metrics(rows):
    scored = [r for r in rows if r['outcome'] == 'scored']
    tp = sum(r['label'] == 'ai' and r['prob'] >= .5 for r in scored)
    fn = sum(r['label'] == 'ai' and r['prob'] < .5 for r in scored)
    fp = sum(r['label'] == 'real' and r['prob'] >= .5 for r in scored)
    tn = sum(r['label'] == 'real' and r['prob'] < .5 for r in scored)
    both = bool(tp+fn and fp+tn)
    return {'attempted': len(rows), 'scored': len(scored), 'failed': len(rows)-len(scored),
            'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
            'coverage': len(scored)/len(rows) if rows else None,
            'TPR': tp/(tp+fn) if tp+fn else None, 'FPR': fp/(fp+tn) if fp+tn else None,
            'F1': 2*tp/(2*tp+fp+fn) if both and 2*tp+fp+fn else None,
            'AUROC': float(roc_auc_score([r['label'] == 'ai' for r in scored], [r['prob'] for r in scored])) if both else None,
            'all_attempt_correctness': (tp+tn)/len(rows) if rows else None}


def run_identity(model, assets, args, selected):
    from . import audio, model_assets, common
    paths = [Path(__file__), Path(inspect.getfile(type(model))), Path(audio.__file__),
             Path(model_assets.__file__), Path(common.__file__)]
    return {'model': args.model, 'model_assets': assets, 'device': args.device,
        'manifest_sha256': digest(args.release/'manifest.local.json'),
        'public_manifest_sha256': digest(args.release/'manifest.public.json'),
        'release_sha256': digest(args.release/'release.json'), 'expected': len(selected),
        'selected_ids': [r['id'] for r in selected], 'smoke_per_source': args.smoke_per_source,
        'code_hashes': {str(p): digest(p) for p in paths}, 'python': sys.version,
        'packages': sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()),
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
        'ffmpeg': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
        'rng': 'SHA256(260905:<release-entry-id>) first 32 bits, reset before every recording',
        'audio': 'soundfile float32, mono mean, torchaudio resample; ffmpeg explicit f32 fallback; clip [-1,1]',
        'threshold': .5, 'torch_threads': 2, 'cudnn_deterministic': True,
        'cudnn_benchmark': False, 'tf32_matmul': False, 'tf32_cudnn': False,
        'hf_hub_offline': True, 'scope': 'fixed checkpoint evaluation; official demos reported separately'}


def snapshot_sources(output, identity):
    sources = dict(identity['code_hashes'])
    for repo in identity['model_assets']['source_repos'].values():
        sources.update(repo['python_files'])
    for repo in identity['model_assets']['snapshots'].values():
        sources.update({str(Path(repo['snapshot'])/name): checksum
                        for name, checksum in repo['files'].items() if name.endswith('.py')})
    mapping = {}
    directory = output/'source_snapshots'
    directory.mkdir(parents=True, exist_ok=True)
    for original, checksum in sources.items():
        snapshot = directory/(checksum+'.py')
        if not snapshot.exists():
            shutil.copyfile(original, snapshot)
        if digest(snapshot) != checksum:
            raise ValueError('Source snapshot differs from pinned hash')
        mapping[original] = str(snapshot.relative_to(output))
    write_once(output/'source_snapshots.json', mapping)


def measured_parameters(model):
    modules = {name: value for name, value in vars(model).items() if isinstance(value, torch.nn.Module)}
    if getattr(model, '_net', None) is not None:
        modules.update({name: getattr(model._net, name) for name in ('_unet', '_cnn')})
    if getattr(model, 'file2beats', None) is not None:
        modules['beat_this'] = model.file2beats.model
    seen, counts = set(), {}
    for name, module in sorted(modules.items()):
        parameters = [p for p in module.parameters() if id(p) not in seen]
        counts[name] = sum(p.numel() for p in parameters)
        seen.update(id(p) for p in parameters)
    return {'module_counts': counts, 'total_unique_torch_parameters': sum(counts.values()) if counts else None,
            'scope': 'loaded PyTorch modules; excludes ONNX/LGBM parameters and non-parameter buffers'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--snapshots', required=True, type=Path)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    ap.add_argument('--smoke-per-source', type=int, default=0)
    args = ap.parse_args()
    if args.smoke_per_source < 0:
        ap.error('smoke-per-source must be nonnegative')
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('HF_HUB_OFFLINE=1 is required for fixed local asset loading')
    release = json.loads((args.release/'release.json').read_text())
    for name, expected in release['public_file_sha256'].items():
        if digest(args.release/name) != expected:
            raise ValueError(f'Frozen release changed: {name}')
    selected = json.loads((args.release/'manifest.local.json').read_text())['bench']
    public = {r['id']: r for r in json.loads((args.release/'manifest.public.json').read_text())['bench']}
    if len(selected) != len(public) or len(selected) != release['evaluation_entries']:
        raise ValueError('Frozen public/local manifest count mismatch')
    for row in selected:
        if any(row[k] != public[row['id']][k] for k in ('sha256', 'source', 'label', 'track_id', 'partition', 'dependence_cluster', 'recording_group')):
            raise ValueError('Frozen public/local row identity mismatch')
    selected = sorted(selected, key=lambda r: rank(r['id']))
    if args.smoke_per_source:
        counts, subset = Counter(), []
        for row in selected:
            if counts[row['source']] < args.smoke_per_source:
                subset.append(row)
                counts[row['source']] += 1
        selected = subset
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_record('model-initialization')
    model, assets = build_pinned(args.model, args.snapshots)
    identity = run_identity(model, assets, args, selected)
    write_once(args.output/'identity.json', identity)
    snapshot_sources(args.output, identity)
    model.load(args.device)
    rescue = None
    if args.model == 'artifactnet':
        import lightgbm as lgb
        rescue = lgb.Booster(model_file=assets['rescue_checkpoint'])
    write_once(args.output/'model_info.json', dict(model.info(), parameter_measurement=measured_parameters(model)))
    done, consecutive_errors = [], 0
    for index, entry in enumerate(selected):
        target = args.output/'records'/(entry['id']+'.json')
        if target.exists():
            cached = json.loads(target.read_text())
            if cached['id'] != entry['id'] or cached['audio_sha256'] != entry['sha256']:
                raise ValueError('Cached prediction identity mismatch')
            done.append(cached)
            continue
        if digest(entry['path']) != entry['sha256']:
            raise ValueError(f'Audio changed after freeze: {entry["id"]}')
        begin = time.monotonic()
        audio = load_audio_float(entry['path'])
        decoded = time.monotonic()
        seed = seed_record(entry['id'])
        row = {k: entry[k] for k in ('id', 'source', 'track_id', 'label', 'partition', 'recording_group', 'recording_representative', 'dependence_cluster')}
        row.update(audio_sha256=entry['sha256'], seed=seed, decoded_frames=len(audio), decode_seconds=decoded-begin)
        try:
            prob = float(model.forward(audio))
            if not np.isfinite(prob) or not 0 <= prob <= 1:
                raise ValueError(f'Non-finite or out-of-range probability: {prob}')
            row.update(prob=prob, outcome='scored')
            if rescue is not None:
                rp = float(rescue.predict(np.array([model.last_features['verdict_feat']], dtype=np.float32), num_threads=1)[0])
                if not np.isfinite(rp) or not 0 <= rp <= 1:
                    raise ValueError('Invalid rescue score')
                thresholds = assets['operating_thresholds']
                row.update(rescue_prob=rp, production_ai=bool(prob >= thresholds['cnn'] and rp >= thresholds['rescue']),
                           n_chunks=model.last_features['n_chunks'])
            if hasattr(model, 'last_diagnostics'):
                row['model_diagnostics'] = model.last_diagnostics
            consecutive_errors = 0
        except torch.cuda.OutOfMemoryError as exc:
            write_once(args.output/('infrastructure-'+entry['id']+'.json'), {'id': entry['id'], 'error': str(exc), 'outcome': 'cuda_oom_not_a_prediction'})
            raise
        except (RuntimeError, ValueError, IndexError, TypeError, ZeroDivisionError) as exc:
            if isinstance(exc, RuntimeError) and any(s in str(exc).lower() for s in ('cuda error', 'illegal memory access', 'device-side assert')):
                raise
            row.pop('prob', None)
            row.update(outcome='model_execution_error', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            consecutive_errors += 1
        row['inference_seconds'] = time.monotonic()-decoded
        if args.device == 'cuda':
            row['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        write_once(target, row)
        done.append(row)
        print(json.dumps({'model': args.model, 'attempted': index+1, 'expected': len(selected),
                          'outcome': row['outcome'], 'source': row['source'], 'seconds': row['inference_seconds']}), flush=True)
        if consecutive_errors >= 5:
            raise RuntimeError('Five consecutive execution errors: inspect adapter/runtime before continuing')
    # 실행 중 가중치나 구현이 바뀌었다면 완료 산출물을 만들지 않는다.
    _, end_assets = build_pinned(args.model, args.snapshots)
    if end_assets != assets or any(digest(p) != expected for p, expected in identity['code_hashes'].items()):
        raise ValueError('Model assets/code changed during evaluation')
    for name, field in (('manifest.local.json', 'manifest_sha256'), ('manifest.public.json', 'public_manifest_sha256'), ('release.json', 'release_sha256')):
        if digest(args.release/name) != identity[field]:
            raise ValueError('Frozen evaluation input changed during execution')
    grouped = defaultdict(list)
    for row in done:
        grouped[row['partition']].append(row)
    summary = {'model': args.model, 'expected': len(selected), 'attempted': len(done),
        'outcomes': dict(Counter(r['outcome'] for r in done)),
        'partitions': {part: metrics(rows) for part, rows in grouped.items()},
        'per_source': {source: metrics([r for r in done if r['source'] == source]) for source in sorted({r['source'] for r in done})},
        'status': 'smoke complete' if args.smoke_per_source else 'full fixed-checkpoint run complete',
        'identity_sha256': digest(args.output/'identity.json'),
        'record_sha256': {r['id']: digest(args.output/'records'/(r['id']+'.json')) for r in done}}
    write_once(args.output/'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('record_sha256', 'per_source')}), flush=True)


if __name__ == '__main__':
    main()

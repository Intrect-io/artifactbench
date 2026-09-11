"""완료된 주 평가와 동일한 모델·런타임으로 해시 고정된 전송 변형을 평가한다."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .audio import TRANSFORMS
from .common import digest, write_once
from .model_assets import build_pinned
from .report import load_completed
from .run import measured_parameters, run_identity, seed_record, snapshot_sources


def transport_cases(manifest, public):
    cases = []
    controlled_ids, native_ids = set(), set()
    for record in manifest['controlled']:
        entry = record['entry']
        if entry != public[entry['id']] or entry['id'] in controlled_ids:
            raise ValueError('Controlled entry differs from release or is repeated')
        if set(record['variants']) != set(TRANSFORMS):
            raise ValueError('Controlled transform membership mismatch')
        controlled_ids.add(entry['id'])
        for view in TRANSFORMS:
            cache = record['variants'][view]
            if cache['frames'] != record['base']['frames']:
                raise ValueError('Controlled variants must have equal frame counts')
            if view == 'float_identity' and cache['pcm_f32le_sha256'] != record['base']['pcm_f32le_sha256']:
                raise ValueError('Float identity differs from prepared base')
            cases.append({'case_id': entry['id'], 'kind': 'controlled', 'view': view,
                          'entry': entry, 'cache': cache})
    for record in manifest['native']:
        entry = record['entry']
        if entry != public[entry['id']] or record['id'] in native_ids:
            raise ValueError('Native entry differs from release or pair is repeated')
        native_ids.add(record['id'])
        for view in ('primary', 'variant'):
            cases.append({'case_id': record['id'], 'kind': record['pair_kind'], 'view': view,
                          'entry': entry, 'cache': record[view]})
    identifiers = [case['case_id']+'--'+case['view'] for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('Duplicate transport attempt identifier')
    return cases


def load_cached(cache):
    if digest(cache['path']) != cache['npy_sha256']:
        raise ValueError('Prepared waveform hash changed')
    audio = np.load(cache['path'], allow_pickle=False)
    if (audio.ndim != 1 or audio.dtype != np.float32 or len(audio) != cache['frames'] or
            cache['sample_rate'] != 44100 or not len(audio) or not np.isfinite(audio).all()):
        raise ValueError('Invalid prepared waveform shape/dtype/frames/sample rate')
    if np.max(np.abs(audio)) > 1:
        raise ValueError('Prepared waveform violates clipping policy')
    return audio


def infer_view(model, audio, entry_id, rescue=None, thresholds=None):
    """변형명과 무관한 recording seed; 실패를 확률로 대체하지 않는다."""
    seed = seed_record(entry_id)
    row = {'seed': seed}
    begin = time.monotonic()
    try:
        prob = float(model.forward(audio))
        if not np.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError('Invalid AI probability')
        row.update(outcome='scored', prob=prob)
        if rescue is not None:
            rp = float(rescue.predict(np.array([model.last_features['verdict_feat']], dtype=np.float32), num_threads=1)[0])
            if not np.isfinite(rp) or not 0 <= rp <= 1:
                raise ValueError('Invalid rescue probability')
            row.update(rescue_prob=rp, production_ai=bool(prob >= thresholds['cnn'] and rp >= thresholds['rescue']),
                       n_chunks=model.last_features['n_chunks'])
        if hasattr(model, 'last_diagnostics'):
            row['model_diagnostics'] = model.last_diagnostics
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, ValueError, IndexError, TypeError, ZeroDivisionError) as exc:
        if isinstance(exc, RuntimeError) and any(s in str(exc).lower() for s in ('cuda error', 'illegal memory access', 'device-side assert')):
            raise
        row = {'seed': seed, 'outcome': 'model_execution_error', 'error_type': type(exc).__name__,
               'error': str(exc), 'traceback': traceback.format_exc()}
    row['inference_seconds'] = time.monotonic()-begin
    return row


def assert_reference_runtime(identity, reference):
    # JSON로 저장된 tuple/list 표현 차이만 정규화한다. 버전·소스·가중치는 완전히 같아야 한다.
    normalized = json.loads(json.dumps(identity))
    fields = ('model', 'model_assets', 'device', 'manifest_sha256', 'public_manifest_sha256',
              'release_sha256', 'code_hashes', 'python', 'packages', 'torch', 'cuda_runtime',
              'gpu', 'ffmpeg', 'rng', 'audio', 'threshold', 'torch_threads', 'cudnn_deterministic',
              'cudnn_benchmark', 'tf32_matmul', 'tf32_cudnn', 'hf_hub_offline')
    for field in fields:
        if normalized[field] != reference[field]:
            raise ValueError(f'Transport runtime differs from primary run: {field}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--snapshots', required=True, type=Path)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--prepared', required=True, type=Path)
    ap.add_argument('--reference-run', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    ap.add_argument('--deepfense-native', action='store_true',
                    help='Use the completed native-soxr DeepFense primary condition; native pairs read encoded originals')
    args = ap.parse_args()
    args.smoke_per_source = 0
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('HF_HUB_OFFLINE=1 is required')
    if args.deepfense_native and (args.model != 'deepfense' or
            (args.device == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') != '')):
        raise ValueError('Native DeepFense requires its model and explicit empty CUDA visibility for CPU')
    release = json.loads((args.release/'release.json').read_text())
    for name, expected in release['public_file_sha256'].items():
        if digest(args.release/name) != expected:
            raise ValueError('Frozen release changed')
    public = {r['id']: r for r in json.loads((args.release/'manifest.public.json').read_text())['bench']}
    reference_rows, reference_identity = load_completed(args.reference_run, public)
    if set(reference_rows) != set(public):
        raise ValueError('Transport reference must cover the complete primary release')
    preparation = json.loads((args.prepared/'summary.json').read_text())
    prepared_manifest = args.prepared/'manifest.local.json'
    if digest(prepared_manifest) != preparation['manifest_sha256']:
        raise ValueError('Prepared manifest changed')
    manifest = json.loads(prepared_manifest.read_text())
    selection = json.loads((args.prepared/'selection.json').read_text())
    if (selection['public_manifest_sha256'] != digest(args.release/'manifest.public.json') or
            selection['selected'] != [r['entry'] for r in manifest['controlled']]):
        raise ValueError('Prepared cohort does not match score-blind selection')
    cases = transport_cases(manifest, public)
    if args.deepfense_native:
        from . import deepfense_transport
        cases = deepfense_transport.bind_native_cases(cases, manifest,
            json.loads((args.release/'manifest.local.json').read_text())['bench'],
            json.loads((args.release/'native_pairs.local.json').read_text()),
            json.loads((args.release/'native_pairs.public.json').read_text()))
    selected = list({c['entry']['id']: c['entry'] for c in cases}.values())
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_record('model-initialization')
    model, assets = build_pinned(args.model, args.snapshots)
    identity = run_identity(model, assets, args, selected)
    if args.deepfense_native:
        identity = deepfense_transport.native_reference_identity(identity, reference_identity)
    assert_reference_runtime(identity, reference_identity)
    extra_paths = [Path(__file__), Path(__file__).with_name('report.py')]
    identity['code_hashes'].update({str(path): digest(path) for path in extra_paths})
    if args.deepfense_native:
        identity['code_hashes'][str(Path(deepfense_transport.__file__))] = digest(deepfense_transport.__file__)
        identity['native_pair_input_hashes'] = {name: digest(args.release/name) for name in
                                              ('native_pairs.local.json', 'native_pairs.public.json')}
    identity.update(scope='paired transport evaluation, not independent additional recordings',
        reference_summary_sha256=digest(args.reference_run/'summary.json'),
        prepared_input_hashes={name: digest(args.prepared/name) for name in
                              ('manifest.local.json', 'summary.json', 'selection.json', 'materialization_inputs.json')},
        expected_waveforms=len(cases), input_audio='exact cached float32 waveform; no second resampling')
    if args.deepfense_native:
        identity['input_audio'] = ('controlled: verified common float32/44100 master promoted to float64, native-soxr preparation; '
                                  'native pairs: original encoded file, identical primary preparation; no old adapter resampling')
    write_once(args.output/'identity.json', identity)
    snapshot_sources(args.output, identity)
    model.load(args.device)
    rescue = None
    if args.model == 'artifactnet':
        import lightgbm as lgb
        rescue = lgb.Booster(model_file=assets['rescue_checkpoint'])
    write_once(args.output/'model_info.json', dict(model.info(), parameter_measurement=measured_parameters(model)))
    rows, record_hashes, consecutive_errors = [], {}, 0
    case_had_success = False
    for number, case in enumerate(cases):
        entry, cache = case['entry'], case['cache']
        identifier = case['case_id']+'--'+case['view']
        path = args.output/'records'/(identifier+'.json')
        expected = {'id': identifier, 'entry_id': entry['id'], 'case_id': case['case_id'],
                    'kind': case['kind'], 'view': case['view'], 'source': entry['source'],
                    'label': entry['label'], 'dependence_cluster': entry['dependence_cluster'],
                    'waveform_sha256': cache['npy_sha256'], 'decoded_frames': cache['frames']}
        if args.deepfense_native and case['kind'] != 'controlled':
            if digest(case['encoded_input']['path']) != case['encoded_input']['sha256']:
                raise ValueError('Frozen native transport file changed')
            expected.pop('waveform_sha256')
            expected.pop('decoded_frames')
            expected.update(input_sha256=case['encoded_input']['sha256'],
                            input_hash_scope='original encoded native file')
            audio = None
        else:
            audio = load_cached(cache)
        if path.exists():
            row = json.loads(path.read_text())
            if any(row[k] != value for k, value in expected.items()):
                raise ValueError('Cached prediction identity differs')
        else:
            try:
                inference = (deepfense_transport.infer_native_view(model, case, audio) if args.deepfense_native else
                             infer_view(model, audio, entry['id'], rescue, assets.get('operating_thresholds')))
                row = dict(expected, **inference)
            except torch.cuda.OutOfMemoryError as exc:
                write_once(args.output/('infrastructure-'+identifier+'.json'), {'id': identifier,
                    'outcome': 'cuda_oom_not_a_prediction', 'error': str(exc)})
                raise
            if case['view'] in ('float_identity', 'primary'):
                prior = reference_rows[entry['id']]
                row['primary_run_comparison'] = {'primary_outcome': prior['outcome'],
                    'score_difference': row['prob']-prior['prob'] if row['outcome'] == prior['outcome'] == 'scored' else None,
                    'scope': 'decode/runtime repeatability diagnostic; not a codec effect'}
                if args.deepfense_native:
                    eligible = case['kind'] != 'controlled'
                    row['primary_run_comparison']['repeatability_eligible'] = eligible
                    if not eligible:
                        row['primary_run_comparison']['scope'] = 'common-master versus native input-path difference; not repeatability or a codec effect'
            write_once(path, row)
        rows.append(row)
        record_hashes[identifier] = digest(path)
        case_had_success = case_had_success or row['outcome'] == 'scored'
        print(json.dumps({'model': args.model, 'attempted': number+1, 'expected': len(cases),
                          'kind': case['kind'], 'view': case['view'], 'outcome': row['outcome']}), flush=True)
        # 한 짧은 곡의 8개 변형 실패는 8개의 독립적인 인프라 이상 신호가 아니다.
        if number+1 == len(cases) or cases[number+1]['case_id'] != case['case_id']:
            consecutive_errors = 0 if case_had_success else consecutive_errors+1
            case_had_success = False
            if consecutive_errors >= 5:
                raise RuntimeError('Five consecutive complete-case failures: inspect model/runtime')
    _, end_assets = build_pinned(args.model, args.snapshots)
    if end_assets != assets or any(digest(p) != sha for p, sha in identity['code_hashes'].items()):
        raise ValueError('Model assets/code changed during transport evaluation')
    if args.deepfense_native:
        for repo in identity['model_assets']['source_repos'].values():
            if any(digest(p) != sha for p, sha in repo['python_files'].items()):
                raise ValueError('Native transport package source changed')
        for name, sha in identity['native_pair_input_hashes'].items():
            if digest(args.release/name) != sha:
                raise ValueError('Frozen native pair binding changed')
    for name, sha in identity['prepared_input_hashes'].items():
        if digest(args.prepared/name) != sha:
            raise ValueError('Prepared transport metadata changed during evaluation')
    if digest(args.reference_run/'summary.json') != identity['reference_summary_sha256']:
        raise ValueError('Primary-run reference changed during transport evaluation')
    for filename, field in (('manifest.local.json', 'manifest_sha256'),
                            ('manifest.public.json', 'public_manifest_sha256'), ('release.json', 'release_sha256')):
        if digest(args.release/filename) != identity[field]:
            raise ValueError('Frozen release changed during transport evaluation')
    write_once(args.output/'summary.json', {'model': args.model, 'status': 'paired transport run complete',
        'expected': len(cases), 'attempted': len(rows), 'outcomes': dict(Counter(r['outcome'] for r in rows)),
        'identity_sha256': digest(args.output/'identity.json'), 'record_sha256': record_hashes})


if __name__ == '__main__':
    main()

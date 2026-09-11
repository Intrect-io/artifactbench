"""고정 DeepFense의 두 입력 경로를 CPU에서 비교한다. 원래 점수/모델은 바꾸지 않는다."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import librosa
import numpy as np
import soundfile as sf
import torch

from .audio import load_audio_float
from .common import digest, rank, write_once
from .model_assets import build_pinned
from .report import load_completed
from .run import run_identity, seed_record, snapshot_sources
from .verify_release import verify


def select_cases(entries):
    """점수 없이 real source마다 1개, legacy AI 1개, native source마다 1개 선택."""
    groups = {}
    seen = set()
    for entry in sorted(entries, key=lambda r: rank(r['id'])):
        if entry['id'] in seen:
            raise ValueError('Duplicate selection entry')
        seen.add(entry['id'])
        if entry['partition'] == 'legacy':
            group = 'legacy_real:'+entry['source'] if entry['label'] == 'real' else 'legacy_ai'
        elif entry['partition'] == 'contemporary_native':
            group = 'native:'+entry['source']
        else:
            continue
        groups.setdefault(group, {'group': group, 'entry': entry})
    if not groups:
        raise ValueError('No eligible diagnostic entries')
    return [groups[key] for key in sorted(groups)]


def check_reference_runtime(identity, reference):
    if identity['device'] != 'cpu' or identity['gpu'] is not None or reference['device'] != 'cuda':
        raise ValueError('Expected CPU diagnostic against the original CUDA run')
    # 여기서 달라지는 것은 장치와 진단 subset뿐이다. 가중치/전처리/버전은 유지한다.
    ignored = {'device', 'gpu', 'expected', 'selected_ids', 'smoke_per_source', 'scope'}
    current = {k: v for k, v in json.loads(json.dumps(identity)).items() if k not in ignored}
    original = {k: v for k, v in reference.items() if k not in ignored}
    if current != original:
        changed = sorted(k for k in set(current) | set(original) if current.get(k) != original.get(k))
        raise ValueError('Diagnostic runtime differs beyond device/subset: '+', '.join(changed))


def score_output(output, spoof_index):
    logits = output.get('logits')
    if (spoof_index not in (0, 1) or not isinstance(logits, torch.Tensor) or
            logits.shape != (1, 2) or not torch.isfinite(logits).all()):
        raise ValueError('Expected finite labelled two-class logits')
    probability = torch.softmax(logits, dim=-1)[0, spoof_index].item()
    return {'outcome': 'scored', 'p_ai': probability,
            'logits': logits.detach().cpu().tolist()[0], 'spoof_index': spoof_index}


def describe_waveform(audio):
    if audio.shape != (64000,) or audio.dtype != np.float32 or not np.isfinite(audio).all():
        raise ValueError('Expected finite float32 four-second 16-kHz model input')
    canonical = np.ascontiguousarray(audio, dtype='<f4')
    return {'frames': len(audio), 'sample_rate': 16000,
            'pcm_f32le_sha256': hashlib.sha256(canonical.tobytes()).hexdigest(),
            'rms': float(np.sqrt(np.mean(audio.astype(np.float64)**2))),
            'peak': float(np.max(np.abs(audio)))}


@torch.no_grad()
def capture_adapter(model, audio, identifier):
    captured = []
    def observe(module, inputs, output):
        captured.append((inputs[0].detach().cpu().numpy().copy()[0], score_output(output, model.spoof_idx)))
    handle = model.model.register_forward_hook(observe)
    try:
        seed_record(identifier)
        probability = model.forward(audio)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0][1]['p_ai'] != probability:
        raise ValueError('Captured model output differs from actual adapter probability')
    waveform, result = captured[0]
    return waveform, dict(result, waveform=describe_waveform(waveform))


def upstream_waveform(path, loader, padder):
    """실제 upstream 함수를 호출한다. 지원하지 않는 포맷에 fallback을 넣지 않는다."""
    try:
        waveform = loader(str(path), target_sr=16000, mono=True)
        waveform = padder(waveform, max_len=64000, random_pad=False, pad_type='repeat')
        waveform = np.asarray(waveform, dtype=np.float32)
        description = describe_waveform(waveform)
    except (sf.LibsndfileError, RuntimeError, ValueError) as exc:
        return None, {'outcome': 'upstream_input_error', 'error_type': type(exc).__name__, 'error': str(exc)}
    return waveform, {'waveform': description}


def comparison(first, second):
    if not np.isfinite([first, second]).all() or not (0 <= first <= 1 and 0 <= second <= 1):
        raise ValueError('Invalid comparison probability')
    return {'signed_shift': second-first, 'absolute_shift': abs(second-first),
            'raw_05_flip': (first >= .5) != (second >= .5), 'direction': 'second minus first'}


def control_waveforms(common_audio, captured_input, padder):
    """공통 waveform/precision/clip은 고정하고 마지막 resampler만 교체한다."""
    if common_audio.dtype != np.float32 or common_audio.ndim != 1 or not np.isfinite(common_audio).all():
        raise ValueError('Expected the unchanged common float32 mono waveform')
    alternate = librosa.resample(common_audio, orig_sr=44100, target_sr=16000,
                                 res_type='soxr_hq', fix=True, scale=False)
    alternate = padder(alternate, max_len=64000, random_pad=False, pad_type='repeat')
    outputs = {'benchmark_input_replay': captured_input.copy(),
               'common44_librosa': np.asarray(alternate, dtype=np.float32)}
    for waveform in outputs.values():
        describe_waveform(waveform)
    return outputs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--snapshots', required=True, type=Path)
    ap.add_argument('--reference-run', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('Require CUDA_VISIBLE_DEVICES empty and HF_HUB_OFFLINE=1')
    verify(args.release)
    public = json.loads((args.release/'manifest.public.json').read_text())['bench']
    selected = select_cases(public)
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output/'selection.json', {'policy': 'metadata-only SHA256(260905:<id>) first per declared group',
               'manifest_sha256': digest(args.release/'manifest.public.json'), 'selected': selected,
               'scope': 'Small input-path diagnostic selected before reading reference scores; not a performance estimate'})
    # 선택 파일을 저장한 후에만 기존 점수를 연다.
    reference, prior_identity = load_completed(args.reference_run, {r['id']: r for r in public})
    if set(reference) != {r['id'] for r in public} or prior_identity['model'] != 'deepfense':
        raise ValueError('Expected a completed full DeepFense reference')
    local = {r['id']: r for r in json.loads((args.release/'manifest.local.json').read_text())['bench']}
    for case in selected:
        entry = case['entry']
        if digest(local[entry['id']]['path']) != entry['sha256']:
            raise ValueError('Selected input audio changed')
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.model, args.device, args.smoke_per_source = 'deepfense', 'cpu', 0
    seed_record('model-initialization')
    model, assets = build_pinned('deepfense', args.snapshots)
    identity = run_identity(model, assets, args, [r['entry'] for r in selected])
    check_reference_runtime(identity, prior_identity)
    import deepfense
    from deepfense.data.transforms.transforms import load_audio, pad_combined
    code = {str(path): digest(path) for path in Path(deepfense.__file__).parent.rglob('*.py')}
    code.update({str(Path(__file__).resolve()): digest(__file__),
                 str(Path(__file__).with_name('report.py').resolve()): digest(Path(__file__).with_name('report.py'))})
    identity['code_hashes'].update(code)
    identity.update(scope='CPU input-path diagnostic only; original benchmark predictions remain unchanged',
                    reference_summary_sha256=digest(args.reference_run/'summary.json'),
                    selection_sha256=digest(args.output/'selection.json'),
                    controls={'benchmark_input_replay': 'Exact captured model input on the same CPU model',
                              'common44_librosa': 'Only the final common44k-to-16k resampler changes to librosa soxr_hq; decoder, float32 mono, clipping, checkpoint and validation padding stay fixed'})
    write_once(args.output/'identity.local.json', identity)
    snapshot_sources(args.output, identity)
    model.load('cpu')
    if model.max_len != 64000 or model.spoof_idx != 0 or torch.cuda.is_initialized():
        raise ValueError('Unexpected checkpoint input/class/device contract')
    records = {}
    for case in selected:
        started = time.monotonic()
        entry = case['entry']
        identifier, path = entry['id'], local[entry['id']]['path']
        common_audio = load_audio_float(path)
        waveform, cpu = capture_adapter(model, common_audio, identifier)
        controls = {}
        for name, control_input in control_waveforms(common_audio, waveform, pad_combined).items():
            seed_record(identifier)
            with torch.no_grad():
                result = score_output(model.model(torch.from_numpy(control_input).unsqueeze(0)), model.spoof_idx)
            controls[name] = dict(result, waveform=describe_waveform(control_input),
                                  comparison=comparison(cpu['p_ai'], result['p_ai']))
        direct, upstream = upstream_waveform(path, load_audio, pad_combined)
        if direct is not None:
            seed_record(identifier)
            with torch.no_grad():
                upstream.update(score_output(model.model(torch.from_numpy(direct).unsqueeze(0)), model.spoof_idx))
            difference = direct.astype(np.float64)-waveform.astype(np.float64)
            upstream['input_difference'] = {'max_absolute': float(np.max(np.abs(difference))),
                'rms': float(np.sqrt(np.mean(difference**2))),
                'exact_pcm_match': np.array_equal(direct, waveform)}
        prior = reference[identifier]
        row = {'id': identifier, 'group': case['group'], 'source': entry['source'], 'label': entry['label'],
               'audio_sha256': entry['sha256'], 'original_gpu': {'outcome': prior['outcome'], 'p_ai': prior.get('prob')},
               'benchmark_cpu': cpu, 'upstream_cpu': upstream,
               'common44_pcm_f32le_sha256': hashlib.sha256(np.ascontiguousarray(common_audio, dtype='<f4').tobytes()).hexdigest(),
               'resampler_controls': controls,
               'gpu_to_cpu': comparison(prior['prob'], cpu['p_ai']) if prior['outcome'] == 'scored' else None,
               'input_path_comparison': comparison(cpu['p_ai'], upstream['p_ai']) if direct is not None else None,
               'seconds': time.monotonic()-started}
        if digest(path) != entry['sha256']:
            raise ValueError('Audio changed during diagnostic')
        write_once(args.output/'records'/(identifier+'.json'), row)
        records[identifier] = row
        print(json.dumps({'checked': len(records), 'expected': len(selected), 'group': case['group'],
                          'benchmark_cpu': cpu['p_ai'], 'upstream_cpu': upstream.get('p_ai'),
                          'common44_librosa': controls['common44_librosa']['p_ai'],
                          'upstream_outcome': upstream['outcome'], 'seconds': row['seconds']}), flush=True)
    _, end_assets = build_pinned('deepfense', args.snapshots)
    if end_assets != assets or any(digest(p) != checksum for p, checksum in identity['code_hashes'].items()):
        raise ValueError('Model assets or source changed during diagnostic')
    for name, checksum in ((args.output/'selection.json', identity['selection_sha256']),
                           (args.reference_run/'summary.json', identity['reference_summary_sha256']),
                           (args.release/'manifest.local.json', identity['manifest_sha256']),
                           (args.release/'manifest.public.json', identity['public_manifest_sha256']),
                           (args.release/'release.json', identity['release_sha256'])):
        if digest(name) != checksum:
            raise ValueError('Diagnostic reference metadata changed')
    if torch.cuda.is_initialized():
        raise ValueError('Diagnostic unexpectedly initialized CUDA')
    compared = [r['input_path_comparison'] for r in records.values() if r['input_path_comparison'] is not None]
    gpu = [r['gpu_to_cpu'] for r in records.values() if r['gpu_to_cpu'] is not None]
    summary = {'status': 'bounded CPU input-path diagnostic complete', 'attempted': len(records),
               'upstream_outcomes': dict(Counter(r['upstream_cpu']['outcome'] for r in records.values())),
               'compared_input_paths': len(compared), 'input_path_flips': sum(r['raw_05_flip'] for r in compared),
               'max_input_path_score_difference': max((r['absolute_shift'] for r in compared), default=None),
               'gpu_cpu_flips': sum(r['raw_05_flip'] for r in gpu),
               'max_gpu_cpu_score_difference': max((r['absolute_shift'] for r in gpu), default=None),
               'resampler_controls': {name: {
                   'compared': len(records),
                   'flips': sum(r['resampler_controls'][name]['comparison']['raw_05_flip'] for r in records.values()),
                   'max_score_difference': max(r['resampler_controls'][name]['comparison']['absolute_shift'] for r in records.values())}
                   for name in ('benchmark_input_replay', 'common44_librosa')},
               'identity_sha256': digest(args.output/'identity.local.json'),
               'record_sha256': {i: digest(args.output/'records'/(i+'.json')) for i in records},
               'limits': 'Nine metadata-selected examples, not a population estimate or proof of cause. Input paths differ in decode, precision, clipping and resampling. No score/threshold/model substitution.',
               'cuda_initialized': False}
    write_once(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

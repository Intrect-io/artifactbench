"""고정 native DeepFense 조건을 전송 실험에 연결한다. 실행 중 주 평가 코드는 수정하지 않는다."""
import copy
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .common import digest
from .deepfense_native import CONDITION, CONTRACT, prepare_waveform, score_prepared
from .run import seed_record
from .run_deepfense_native import infer_entry


def native_reference_identity(identity, reference):
    """현재 파일/패키지로 주 평가 identity를 재구성한 후 기존 strict runtime 검사에 넘긴다."""
    if identity['model'] != 'deepfense' or reference.get('input_condition') != CONTRACT:
        raise ValueError('Expected the pinned native DeepFense reference condition')
    import deepfense
    from . import deepfense_native, run_deepfense_native, report
    identity = copy.deepcopy(identity)
    package = Path(deepfense.__file__).parent
    identity['model_assets']['source_repos']['DeepFense-package'] = {
        'root': str(package), 'git_head': None,
        'python_files': {str(p): digest(p) for p in package.rglob('*.py')}}
    identity['code_hashes'].update({str(Path(module.__file__)): digest(module.__file__)
        for module in (deepfense_native, run_deepfense_native, report)})
    identity.update(input_condition=CONTRACT,
        audio=CONDITION+': native float64 mono, direct librosa soxr_hq 16k, upstream prefix/repeat, float32; no clip; AAC/Opus-only native FFmpeg extension')
    return identity


def bind_native_cases(cases, manifest, local_entries, local_pairs, public_pairs):
    """캐시의 native view를 실제 동결 파일에 바인딩한다. 공개 pair와 모든 비경로 필드를 대조한다."""
    local = {r['id']: r for r in local_entries}
    pairs = {(r['primary_id'], r['variant_id']): r for r in local_pairs}
    published = {(r['primary_id'], r['variant_id']): r for r in public_pairs}
    if len(pairs) != len(local_pairs) or len(published) != len(public_pairs) or set(pairs) != set(published):
        raise ValueError('Native local/public pair membership differs')
    bound = {}
    for record in manifest['native']:
        pair = record['native_pair']
        key = (pair['primary_id'], pair['variant_id'])
        if key not in pairs or pair != pairs[key] or any(pair.get(k) != v for k, v in published[key].items()):
            raise ValueError('Prepared native pair differs from frozen local/public pair')
        entry = record['entry']
        primary = local[entry['id']]
        if (pair['primary_id'] != entry['id'] or pair['primary_sha256'] != entry['sha256']
                or primary['sha256'] != entry['sha256'] or pair['primary_path'] != primary['path']):
            raise ValueError('Native primary does not match the frozen primary file')
        if record['id'] != pair['primary_id']+'-'+pair['variant_id'] or record['id'] in bound:
            raise ValueError('Duplicate native case')
        bound[record['id']] = pair
    if len(bound) != len(pairs):
        raise ValueError('Prepared native pair inventory incomplete')
    result = []
    for case in cases:
        case = dict(case)
        if case['kind'] != 'controlled':
            pair = bound[case['case_id']]
            if case['view'] not in ('primary', 'variant'):
                raise ValueError('Invalid native transport view')
            prefix = 'primary_' if case['view'] == 'primary' else ''
            case['encoded_input'] = {'path': pair[prefix+'path'], 'sha256': pair[prefix+'sha256']}
        result.append(case)
    return result


def infer_native_view(model, case, audio=None):
    if case['kind'] != 'controlled':
        encoded = case['encoded_input']
        row = infer_entry(model, dict(case['entry'], **encoded))
        if digest(encoded['path']) != encoded['sha256']:
            raise ValueError('Native file changed during transport inference')
        # entry id/encoded hash는 호출자의 전송 view identity를 덮어쓰지 않는다.
        fields = ('seed', 'outcome', 'prob', 'error_type', 'error', 'traceback', 'decode_seconds',
                  'inference_seconds', 'decoded_frames', 'input_preprocessing', 'model_diagnostics')
        return {k: row[k] for k in fields if k in row}
    if not isinstance(audio, np.ndarray) or audio.dtype != np.float32 or audio.ndim != 1:
        raise ValueError('Controlled native input requires the verified float32 master')
    seed = seed_record(case['entry']['id'])
    begin = time.monotonic()
    # materialization cache를 float64로 승격할 뿐, native 원본 precision을 복원했다고 주장하지 않는다.
    prepared, metadata = prepare_waveform(audio.astype(np.float64), 44100)
    metadata['decoder'] = 'controlled_float32_master_promoted_f64'
    row = {'seed': seed, 'input_preprocessing': metadata, 'decode_seconds': time.monotonic()-begin}
    started = time.monotonic()
    try:
        prob, logits = score_prepared(model, prepared)
        row.update(outcome='scored', prob=prob, model_diagnostics={'logits': logits, 'spoof_index': 0})
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, ValueError, TypeError, IndexError, ZeroDivisionError) as exc:
        if isinstance(exc, RuntimeError) and any(s in str(exc).lower() for s in
                ('cuda error', 'illegal memory access', 'device-side assert')):
            raise
        row.update(outcome='model_execution_error', error_type=type(exc).__name__,
                   error=str(exc), traceback=traceback.format_exc())
    row['inference_seconds'] = time.monotonic()-started
    return row

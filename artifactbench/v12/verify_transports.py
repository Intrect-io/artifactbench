"""준비된 변형의 실제 PCM과 native pair의 정렬 근거를 검증한다. 검출 점수는 읽지 않는다."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .common import digest, write_once
from .duplicates import compare_waveforms, normalized_correlation
from .run_transports import load_cached, transport_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--prepared', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    summary = json.loads((args.prepared/'summary.json').read_text())
    if digest(args.prepared/'manifest.local.json') != summary['manifest_sha256']:
        raise ValueError('Prepared manifest differs from completion hash')
    manifest = json.loads((args.prepared/'manifest.local.json').read_text())
    public = {r['id']: r for r in json.loads((args.release/'manifest.public.json').read_text())['bench']}
    inputs = [args.prepared/'manifest.local.json', args.prepared/'summary.json',
              args.release/'manifest.public.json', Path(__file__), Path(__file__).with_name('run_transports.py'),
              Path(__file__).with_name('duplicates.py')]
    write_once(args.output/'inputs.json', {'input_sha256': {str(p): digest(p) for p in inputs},
        'alignment_scope': 'full cached waveform, scipy resample_poly 44100->8000, lag search +/-0.5s',
        'criterion': 'existing duplicate-review correlation and median 5s correlation >=0.985; no score-based filtering'})
    cases = transport_cases(manifest, public)
    for case in cases:
        load_cached(case['cache'])
    for record in manifest['controlled']:
        base = load_cached(record['base'])
        identity = load_cached(record['variants']['float_identity'])
        if not np.array_equal(base, identity):
            raise ValueError('Numerically inexact float identity')
    pairs = []
    for record in manifest['native']:
        x = load_cached(record['primary'])
        y = load_cached(record['variant'])
        a, b = resample_poly(x, 80, 441), resample_poly(y, 80, 441)
        shared = min(len(a), len(b))
        result = {'id': record['id'], 'entry_id': record['entry']['id'], 'kind': record['pair_kind'],
            'duration_difference_seconds': (len(y)-len(x))/44100,
            'native_unaligned_correlation': normalized_correlation(a[:shared], b[:shared]),
            'alignment': compare_waveforms(a, b, 0.),
            'scope': 'waveform evidence only; alignment is diagnostic, not applied to detector inputs'}
        write_once(args.output/'native_pairs'/(record['id']+'.json'), result)
        pairs.append(result)
        print(json.dumps(result), flush=True)
    write_once(args.output/'summary.json', {'status': 'prepared transport waveform verification passed',
        'controlled_recordings': len(manifest['controlled']), 'float_identity_exact': len(manifest['controlled']),
        'inference_waveforms_checked': len(cases), 'native_pairs': len(pairs),
        'native_pair_kinds': dict(Counter(p['kind'] for p in pairs)),
        'native_alignment_confirmed': sum(p['alignment']['confirmed_shared_excerpt'] for p in pairs),
        'native_pair_result_sha256': {p['id']: digest(args.output/'native_pairs'/(p['id']+'.json')) for p in pairs},
        'scope': 'no detector scores or causal pure-codec interpretation'})


if __name__ == '__main__':
    main()

"""완성된 paired transport 점수의 실패 분모와 recording/creator bootstrap을 보존한다."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .audio import CODECS, TRANSFORMS
from .common import digest, write_once
from .statistics import bootstrap_weights, interval


def summarize_values(entries, values, replicates=2000):
    values = np.asarray(values, dtype=np.float64)
    if len(entries) != len(values) or not entries:
        raise ValueError('Expected nonempty aligned recording values')
    valid = np.isfinite(values)
    weights, strata = bootstrap_weights(entries, replicates)
    denominator = weights @ valid
    numerator = weights @ np.where(valid, values, 0.)
    means = np.divide(numerator, denominator, out=np.full(replicates, np.nan), where=denominator > 0)
    observed = values[valid]
    return {'attempted_recordings': len(entries), 'complete_recordings': int(valid.sum()),
        'coverage': float(valid.mean()), 'mean': interval(means, observed.mean() if len(observed) else np.nan),
        'median_descriptive': float(np.median(observed)) if len(observed) else None,
        'p90_descriptive': float(np.quantile(observed, .9)) if len(observed) else None,
        'max_descriptive': float(observed.max()) if len(observed) else None, 'strata': strata}


def paired_values(first, second):
    if first['outcome'] != 'scored' or second['outcome'] != 'scored':
        return np.nan, np.nan, np.nan
    a, b = first['prob'], second['prob']
    if not np.isfinite([a, b]).all() or not (0 <= a <= 1 and 0 <= b <= 1):
        raise ValueError('Invalid scored probability')
    return b-a, abs(b-a), float((a >= .5) != (b >= .5))


def summarize_pairs(entries, first, second, replicates=2000):
    if len(entries) != len(first) or len(entries) != len(second):
        raise ValueError('Paired transport rows must align with entries')
    values = [paired_values(a, b) for a, b in zip(first, second)]
    result = {name: summarize_values(entries, [v[index] for v in values], replicates)
              for index, name in enumerate(('signed_shift', 'absolute_shift', 'raw_05_flip_rate'))}
    flips = [v[2] for v in values if np.isfinite(v[2])]
    result['raw_05_flip_count'] = int(sum(flips)) if flips else None
    return result


def staging_contrasts(entries, rows, replicates=2000):
    """같은 staging의 코덱 증가분과 녹음별 4-view interaction을 분리한다."""
    views = {view: [rows[e['id']+'--'+view] for e in entries] for view in TRANSFORMS}
    def pair(first, second):
        return {'first_view': first, 'second_view': second, 'direction': 'second minus first',
                'metrics': summarize_pairs(entries, views[first], views[second], replicates)}
    result = {'quantization_only': pair('float_identity', 'pcm16_only'), 'codecs': {}}
    for codec in CODECS:
        staged = 'pcm16_'+codec
        # 두 평균/CI를 빼지 않고 같은 녹음의 네 점수로 contrast를 먼저 만든다.
        interactions = [paired_values(s, cs)[0]-paired_values(f, cf)[0]
                        for f, s, cf, cs in zip(views['float_identity'], views['pcm16_only'],
                                               views[codec], views[staged])]
        result['codecs'][codec] = {
            'codec_from_float': pair('float_identity', codec),
            'codec_from_pcm16': pair('pcm16_only', staged),
            'staging_with_codec': pair(codec, staged),
            'signed_interaction': {
                'formula': '(p_codec_pcm16 - p_pcm16) - (p_codec_float - p_float)',
                'direction': 'codec increment after PCM16 minus codec increment after float staging',
                'required_views': ['float_identity', 'pcm16_only', codec, staged],
                'statistics': summarize_values(entries, interactions, replicates)}}
    return result


def primary_comparisons(rows):
    """native 주 평가와 common-master 차이를 반복 실행 오차와 섞지 않는다."""
    groups = {'primary_repeatability': [], 'common_master_input_path_difference': []}
    for row in rows:
        comparison = row.get('primary_run_comparison', {})
        value = comparison.get('score_difference')
        if value is None:
            continue
        name = ('primary_repeatability' if comparison.get('repeatability_eligible', True)
                else 'common_master_input_path_difference')
        groups[name].append(value)
    return {name: {'comparable_cases': len(values),
                  'max_absolute_difference': max(abs(v) for v in values) if values else None}
            for name, values in groups.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared', required=True, type=Path)
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    identity = json.loads((args.run/'identity.json').read_text())
    summary = json.loads((args.run/'summary.json').read_text())
    if (summary['status'] != 'paired transport run complete' or
            summary['identity_sha256'] != digest(args.run/'identity.json') or
            identity['prepared_input_hashes']['manifest.local.json'] != digest(args.prepared/'manifest.local.json')):
        raise ValueError('A completed hash-bound transport run is required')
    manifest = json.loads((args.prepared/'manifest.local.json').read_text())
    rows = {}
    for identifier, checksum in summary['record_sha256'].items():
        path = args.run/'records'/(identifier+'.json')
        if digest(path) != checksum:
            raise ValueError('Transport prediction changed')
        row = json.loads(path.read_text())
        if row['id'] != identifier:
            raise ValueError('Transport prediction ID mismatch')
        rows[identifier] = row
    expected = {r['entry']['id']+'--'+view for r in manifest['controlled'] for view in TRANSFORMS}
    expected.update(r['id']+'--'+view for r in manifest['native'] for view in ('primary', 'variant'))
    if (set(rows) != expected or len(rows) != summary['expected'] or len(rows) != identity['expected_waveforms'] or
            dict(Counter(r['outcome'] for r in rows.values())) != summary['outcomes']):
        raise ValueError('Incomplete transport prediction membership or coverage')
    write_once(args.output/'inputs.json', {'run_summary_sha256': digest(args.run/'summary.json'),
        'prepared_manifest_sha256': digest(args.prepared/'manifest.local.json'),
        'tool_sha256': digest(__file__), 'bootstrap_tool_sha256': digest(Path(__file__).with_name('statistics.py')),
        'replicates': 2000, 'seed': 260905})
    entries = [r['entry'] for r in manifest['controlled']]
    controlled = {}
    for view in TRANSFORMS:
        controlled[view] = summarize_pairs(entries, [rows[e['id']+'--float_identity'] for e in entries],
                                          [rows[e['id']+'--'+view] for e in entries])
    ranges = []
    for entry in entries:
        views = [rows[entry['id']+'--'+view] for view in TRANSFORMS]
        ranges.append(max(r['prob'] for r in views)-min(r['prob'] for r in views)
                      if all(r['outcome'] == 'scored' for r in views) else np.nan)
    controlled['whole_eight_view_range'] = summarize_values(entries, ranges)
    native_entries = [r['entry'] for r in manifest['native']]
    native = summarize_pairs(native_entries, [rows[r['id']+'--primary'] for r in manifest['native']],
                             [rows[r['id']+'--variant'] for r in manifest['native']]) if native_entries else {}
    result = {'model': summary['model'], 'coverage': summary['outcomes'], 'controlled': controlled,
        'controlled_contrasts': staging_contrasts(entries, rows),
        'native': native, 'native_pairs': len(native_entries),
        'native_case_predictions': [{view: rows[r['id']+'--'+view] for view in ('primary', 'variant')} for r in manifest['native']],
        **primary_comparisons(rows.values()),
        'input_condition': identity.get('input_condition'),
        'limits': 'Three native pairs are descriptive, not population-level codec evidence. Controlled source history may already be lossy. No failed probability is imputed.'}
    write_once(args.output/'statistics.json', result)
    write_once(args.output/'summary.json', {'status': 'paired transport statistics complete', 'model': summary['model'],
        'statistics_sha256': digest(args.output/'statistics.json')})
    print(json.dumps({'model': summary['model'], 'controlled': len(entries), 'native': len(native_entries)}))


if __name__ == '__main__':
    main()

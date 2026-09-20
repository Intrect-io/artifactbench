"""완료된 checkpoint 실행만 읽어 coverage·층화 집단 CI·동일 표본 차이를 생성한다."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

import numpy as np

from .common import digest, write_once
from .statistics import metric_arrays, paired_difference, summarize_bootstrap

MODELS = ('artifactnet', 'spectttra', 'spectttra_beta5s', 'ast_60s', 'deepfense', 'deezer_ismir', 'clam', 'fst')


def load_completed(directory, public, allow_smoke=False):
    summary = json.loads((directory/'summary.json').read_text())
    identity = json.loads((directory/'identity.json').read_text())
    expected_status = 'smoke complete' if allow_smoke else 'full fixed-checkpoint run complete'
    if summary['status'] != expected_status or digest(directory/'identity.json') != summary['identity_sha256']:
        raise ValueError('Report requires a completed hash-bound run of the requested scope')
    if len(identity['selected_ids']) != summary['expected'] or len(summary['record_sha256']) != summary['expected']:
        raise ValueError('Incomplete or duplicate run membership')
    rows = {}
    for identifier in identity['selected_ids']:
        path = directory/'records'/(identifier+'.json')
        if digest(path) != summary['record_sha256'][identifier]:
            raise ValueError(f'Prediction changed: {path}')
        row, entry = json.loads(path.read_text()), public[identifier]
        if row['id'] in rows:
            raise ValueError('Duplicate prediction ID')
        for field in ('id', 'label', 'source', 'track_id', 'partition', 'recording_group', 'recording_representative', 'dependence_cluster'):
            if row[field] != entry[field]:
                raise ValueError(f'Prediction/manifest {field} mismatch')
        if row['audio_sha256'] != entry['sha256']:
            raise ValueError('Prediction audio hash mismatch')
        rows[identifier] = row
    if dict(Counter(r['outcome'] for r in rows.values())) != summary['outcomes']:
        raise ValueError('Run coverage summary differs from saved predictions')
    return rows, identity


def wilson(successes, trials):
    if not trials:
        return {'lower': None, 'upper': None}
    z = 1.959963984540054
    p, correction = successes/trials, z*z/trials
    center = (p + correction/2)/(1+correction)
    radius = z*np.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/(1+correction)
    return {'lower': float(max(0, center-radius)), 'upper': float(min(1, center+radius))}


def scalar_point(entries, rows):
    return {key: float(value[0]) if np.isfinite(value[0]) else None
            for key, value in metric_arrays(entries, rows, np.ones((1, len(entries)))).items()}


def generate_reports(public, runs, identities, output, allow_smoke=False):
    """호출자가 검증한 로컬/공개 예측을 같은 통계 경로로 계산한다."""
    if set(runs) != set(MODELS) or set(identities) != set(MODELS):
        raise ValueError('All eight model inputs are required')
    ids = sorted(runs[MODELS[0]])
    if any(set(rows) != set(ids) for rows in runs.values()) or (not allow_smoke and set(ids) != set(public)):
        raise ValueError('All models must attempt the same fixed membership')
    replicates = 20 if allow_smoke else 2000
    cohorts = {
        'legacy': [i for i in ids if public[i]['partition'] == 'legacy'],
        'contemporary_native': [i for i in ids if public[i]['partition'] == 'contemporary_native'],
        'non_demo': [i for i in ids if public[i]['partition'] != 'official_demo'],
        'non_demo_recording_representative': [i for i in ids if public[i]['partition'] != 'official_demo' and public[i]['recording_representative']],
    }
    all_results = {}
    for name in MODELS:
        result = {'model': name, 'coverage': dict(Counter(r['outcome'] for r in runs[name].values())),
                  'cohorts': {}, 'sources': {}, 'official_demonstrations': {}}
        for cohort, selected in cohorts.items():
            result['cohorts'][cohort] = summarize_bootstrap([public[i] for i in selected], [runs[name][i] for i in selected], replicates)
        for source in sorted({public[i]['source'] for i in ids}):
            selected = [i for i in ids if public[i]['source'] == source]
            entries, rows = [public[i] for i in selected], [runs[name][i] for i in selected]
            if entries[0]['partition'] == 'official_demo':
                point = scalar_point(entries, rows)
                result['official_demonstrations'][source] = {'point': point, 'TPR_descriptive_wilson': wilson(point['TP'], point['scored']),
                    'warning': 'Provider-selected examples; interval is descriptive and does not establish population generalization'}
            else:
                result['sources'][source] = summarize_bootstrap(entries, rows, replicates)
        for metric, worst in (('TPR', min), ('FPR', max)):
            eligible = [(source, value['metrics'][metric]['estimate']) for source, value in result['sources'].items()
                        if value['metrics'][metric]['estimate'] is not None]
            result['worst_observed_'+metric] = ({'source': (item := worst(eligible, key=lambda x: x[1]))[0],
                 'estimate': item[1], 'entries': result['sources'][item[0]]['entries']} if eligible else None)
        if name == 'artifactnet':
            if identities[name]['model_assets']['operating_thresholds'] != {'cnn': .225, 'rescue': .02}:
                raise ValueError('Declared operating-stack thresholds differ from the evaluated checkpoint identity')
            result['operating_stacks'] = {}
            selected = cohorts['non_demo']
            for stack in ('raw_0225', 'cnn_0225_rescue_002'):
                rows = []
                for identifier in selected:
                    row = dict(runs[name][identifier])
                    if row['outcome'] == 'scored':
                        # 내부 confusion-matrix 계산용 hard decision; 확률 예측을 새로 만들지 않는다.
                        row['prob'] = float(row['prob'] >= .225 if stack == 'raw_0225' else row['production_ai'])
                    rows.append(row)
                stack_result = summarize_bootstrap([public[i] for i in selected], rows, replicates)
                del stack_result['metrics']['AUROC']
                stack_result['scope'] = 'Existing hard decision stack; no threshold tuning or implied probability/AUROC'
                result['operating_stacks'][stack] = stack_result
        write_once(output/(name+'.json'), result)
        all_results[name] = result
        print(json.dumps({'model': name, 'phase': 'statistical_report_complete', 'scope': 'smoke wiring' if allow_smoke else 'full'}), flush=True)
    paired = {}
    for first, second in itertools.combinations(MODELS, 2):
        comparison = {}
        for cohort in ('legacy', 'contemporary_native', 'non_demo'):
            selected = cohorts[cohort]
            comparison[cohort] = paired_difference([public[i] for i in selected],
                [runs[first][i] for i in selected], [runs[second][i] for i in selected], replicates)
        paired[first+'__minus__'+second] = comparison
    write_once(output/'paired_differences.json', paired)
    write_once(output/'summary.json', {'status': 'SMOKE REPORT WIRING ONLY' if allow_smoke else 'statistical report complete',
        'models': list(MODELS), 'attempted_per_model': len(ids), 'replicates': replicates,
        'report_sha256': {name: digest(output/(name+'.json')) for name in MODELS},
        'paired_differences_sha256': digest(output/'paired_differences.json'),
        'limitations': 'Descriptive multiple comparisons; no universal unseen status, no causal architecture ranking, no demo pooling.'})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--runs', required=True, type=Path)
    ap.add_argument('--run-override', action='append', default=[], help='Explicit NAME=DIRECTORY for separately retried run')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--allow-smoke', action='store_true', help='20-replicate report-wiring check, never final inferential evidence')
    args = ap.parse_args()
    public_path = args.release/'manifest.public.json'
    release = json.loads((args.release/'release.json').read_text())
    if digest(public_path) != release['public_file_sha256']['manifest.public.json']:
        raise ValueError('Public release identity mismatch')
    public = {r['id']: r for r in json.loads(public_path.read_text())['bench']}
    paths = {name: args.runs/name for name in MODELS}
    for override in args.run_override:
        name, directory = override.split('=', 1)
        if name not in paths:
            raise ValueError(f'Unknown run override: {name}')
        paths[name] = Path(directory)
    runs, identities = {}, {}
    for name, directory in paths.items():
        runs[name], identities[name] = load_completed(directory, public, args.allow_smoke)
        if identities[name]['model'] != name:
            raise ValueError('Run directory belongs to a different model')
        if identities[name]['public_manifest_sha256'] != digest(public_path):
            raise ValueError('Runs use different frozen data')
    inputs = [public_path, args.release/'release.json', Path(__file__), Path(__file__).with_name('statistics.py'),
              Path('docs/STATISTICS_v1.2.md'), *(directory/'summary.json' for directory in paths.values())]
    write_once(args.output/'inputs.json', {'input_hashes': {str(p): digest(p) for p in inputs},
        'replicates': 20 if args.allow_smoke else 2000, 'seed': 260905,
        'status': 'SMOKE REPORT WIRING ONLY' if args.allow_smoke else 'completed-run statistical report'})
    generate_reports(public, runs, identities, args.output, args.allow_smoke)


if __name__ == '__main__':
    main()

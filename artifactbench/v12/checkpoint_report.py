"""완료된 단일 checkpoint의 검증된 중간 보고서. 전체 8모델 비교와 구분한다."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .common import digest, write_once
from .report import load_completed, scalar_point, wilson
from .statistics import summarize_bootstrap
from .verify_release import verify


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    verified = verify(args.release)
    public = {r['id']: r for r in json.loads((args.release/'manifest.public.json').read_text())['bench']}
    rows, identity = load_completed(args.run, public)
    if set(rows) != set(public) or identity['public_manifest_sha256'] != verified['manifest_sha256']:
        raise ValueError('A complete run on this release is required')
    ids = sorted(rows)
    groups = {'legacy': [i for i in ids if public[i]['partition'] == 'legacy'],
              'non_demo': [i for i in ids if public[i]['partition'] != 'official_demo']}
    groups.update({s: [i for i in ids if public[i]['source'] == s] for s in sorted({
        r['source'] for r in public.values() if r['partition'] == 'contemporary_native'})})
    inputs = {'manifest_sha256': verified['manifest_sha256'], 'run_summary_sha256': digest(args.run/'summary.json'),
              'tool_sha256': digest(__file__), 'statistics_tool_sha256': digest(Path(__file__).with_name('statistics.py')),
              'report_tool_sha256': digest(Path(__file__).with_name('report.py')), 'replicates': 2000, 'seed': 260905}
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output/'inputs.json', inputs)
    result = {'status': 'single completed checkpoint; not the eight-model comparison', 'model': identity['model'],
              'coverage': dict(Counter(r['outcome'] for r in rows.values())), 'raw_05': {}, 'operating_stacks': {}, 'official_demos': {}}
    for group, selected in groups.items():
        entries, predictions = [public[i] for i in selected], [rows[i] for i in selected]
        result['raw_05'][group] = summarize_bootstrap(entries, predictions)
        if identity['model'] == 'artifactnet':
            if identity['model_assets']['operating_thresholds'] != {'cnn': .225, 'rescue': .02}:
                raise ValueError('Unexpected ArtifactNet operating thresholds')
            stacks = {}
            for name in ('raw_0225', 'cnn_0225_rescue_002'):
                decisions = [dict(r) for r in predictions]
                for row in decisions:
                    if row['outcome'] == 'scored':
                        row['prob'] = float(row['prob'] >= .225 if name == 'raw_0225' else row['production_ai'])
                stack = summarize_bootstrap(entries, decisions)
                del stack['metrics']['AUROC']
                stacks[name] = stack
            result['operating_stacks'][group] = stacks
    for source in sorted({r['source'] for r in public.values() if r['partition'] == 'official_demo'}):
        selected = [i for i in ids if public[i]['source'] == source]
        point = scalar_point([public[i] for i in selected], [rows[i] for i in selected])
        result['official_demos'][source] = {'raw_05': point, 'TPR_descriptive_wilson': wilson(point['TP'], point['scored'])}
    if digest(args.run/'summary.json') != inputs['run_summary_sha256']:
        raise ValueError('Run summary changed during inspection')
    result['limits'] = 'Native cohorts have no contemporary real controls. Official demos are provider-selected. No new threshold or model change; other model results still required.'
    write_once(args.output/'checkpoint.json', result)
    write_once(args.output/'summary.json', {'status': result['status'], 'model': identity['model'],
               'checkpoint_sha256': digest(args.output/'checkpoint.json')})
    print(json.dumps({'model': identity['model'], 'coverage': result['coverage'],
        'raw_05': {g: {k: r['metrics'][k] for k in ('F1', 'TPR', 'FPR')} for g, r in result['raw_05'].items()},
        'operating_stacks': {g: {s: {k: r['metrics'][k]['estimate'] for k in ('F1', 'TPR', 'FPR')}
                               for s, r in stacks.items()} for g, stacks in result['operating_stacks'].items()}}, indent=2))


if __name__ == '__main__':
    main()

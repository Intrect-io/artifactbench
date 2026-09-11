"""완료·해시 검증된 통계만 논문 표/그림으로 변환한다. smoke는 명시적 배선 검사다."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from artifactbench.v12.common import digest, write_once
from artifactbench.v12.report import MODELS
from artifactbench.v12.public_results import load_public_results

LABELS = {'artifactnet': 'ArtifactNet', 'spectttra': 'SpecTTTra alpha-120s',
          'spectttra_beta5s': 'SpecTTTra beta-5s', 'ast_60s': 'AST-60s',
          'deepfense': 'DeepFense EAT', 'deezer_ismir': 'Fakeprint LR (lofcz)',
          'clam': 'CLAM', 'fst': 'FST'}
NATIVE = (('suno_v5.5_native_260905', 'Suno v5.5', '#24638a'),
          ('udio_2026_version_unknown_native_260905', 'Udio 2026 (version unknown)', '#c16928'))


def verify_prediction_binding(directory, release, predictions):
    """최종 그림은 검증된 공개 예측과 정확히 같은 9개 통계 파일에서만 만든다."""
    if predictions is None:
        raise ValueError('Final rendering requires --predictions with validated public results')
    _, _, _, published = load_public_results(predictions, release)
    expected = published['reference_statistics']['file_sha256']
    if set(expected) != {name+'.json' for name in MODELS} | {'paired_differences.json'}:
        raise ValueError('Public predictions must bind all nine statistical files')
    for name, sha in expected.items():
        if digest(directory/name) != sha:
            raise ValueError('Rendered statistics differ from validated public predictions: '+name)


def load_report(directory, release, allow_smoke=False, predictions=None):
    summary = json.loads((directory/'summary.json').read_text())
    inputs = json.loads((directory/'inputs.json').read_text())
    expected = 'SMOKE REPORT WIRING ONLY' if allow_smoke else 'statistical report complete'
    if summary['status'] != expected or summary['replicates'] != (20 if allow_smoke else 2000):
        raise ValueError('Final rendering requires a completed 2000-replicate report; smoke must be explicit')
    if set(summary['models']) != set(MODELS) or set(summary['report_sha256']) != set(MODELS):
        raise ValueError('All eight model reports are required')
    manifest_path = release/'manifest.public.json'
    manifest_hashes = [v for k, v in inputs['input_hashes'].items() if Path(k).name == 'manifest.public.json']
    if manifest_hashes != [digest(manifest_path)]:
        raise ValueError('Statistical report belongs to a different release')
    release_info = json.loads((release/'release.json').read_text())
    if release_info['public_file_sha256']['manifest.public.json'] != digest(manifest_path):
        raise ValueError('Frozen public manifest changed')
    if not allow_smoke and summary['attempted_per_model'] != release_info['evaluation_entries']:
        raise ValueError('Full report has an incomplete denominator')
    reports = {}
    for name in MODELS:
        path = directory/(name+'.json')
        if digest(path) != summary['report_sha256'][name]:
            raise ValueError('Statistical result hash changed: '+name)
        result = json.loads(path.read_text())
        if result['model'] != name or sum(result['coverage'].values()) != summary['attempted_per_model']:
            raise ValueError('Model identity or attempted denominator mismatch')
        reports[name] = result
    if digest(directory/'paired_differences.json') != summary['paired_differences_sha256']:
        raise ValueError('Paired statistical result hash changed')
    if not allow_smoke:
        verify_prediction_binding(directory, release, predictions)
    return reports, summary, release_info


def tex_escape(value):
    mapping = {'&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#', '$': r'\$',
               '{': r'\{', '}': r'\}', '\\': r'\textbackslash{}'}
    return ''.join(mapping.get(char, char) for char in str(value))


def formatted(metric, percent=False):
    if metric['estimate'] is None:
        return '--'
    scale, digits = (100, 1) if percent else (1, 3)
    result = f"{scale*metric['estimate']:.{digits}f}"
    if metric.get('lower') is not None and metric.get('upper') is not None:
        result += f" [{scale*metric['lower']:.{digits}f}, {scale*metric['upper']:.{digits}f}]"
    return result


def table(path, headers, rows, smoke):
    lines = [r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}', r'\toprule']
    if smoke:
        lines.extend([r'\multicolumn{'+str(len(headers))+r'}{c}{SMOKE WIRING ONLY --- not benchmark results} \\', r'\midrule'])
    lines.extend([' & '.join(headers)+r' \\', r'\midrule'])
    lines.extend(' & '.join(row)+r' \\' for row in rows)
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    path.write_text('\n'.join(lines)+'\n')


def save_figure(fig, path, smoke):
    if smoke:
        fig.suptitle('SMOKE WIRING ONLY — not benchmark results', color='#a12626', fontsize=11)
    fig.savefig(path, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def tables(reports, output, smoke):
    for cohort in ('legacy', 'non_demo', 'non_demo_recording_representative'):
        rows = []
        for name in MODELS:
            metrics = reports[name]['cohorts'][cohort]['metrics']
            rows.append([tex_escape(LABELS[name]), f"{metrics['scored']['estimate']:.0f}/{metrics['attempted']['estimate']:.0f}",
                         formatted(metrics['F1']), formatted(metrics['AUROC']), formatted(metrics['FPR'], True)])
        table(output/(cohort+'_table.tex'), ['Detector', 'Scored/attempted', r'F1 [95\% CI]', r'AUROC [95\% CI]', r'FPR (\%) [95\% CI]'], rows, smoke)
    rows = []
    for name in MODELS:
        row = [tex_escape(LABELS[name])]
        for source, _, _ in NATIVE:
            metrics = reports[name]['sources'][source]['metrics']
            row.extend([f"{metrics['scored']['estimate']:.0f}/{metrics['attempted']['estimate']:.0f}", formatted(metrics['TPR'], True)])
        rows.append(row)
    table(output/'native_table.tex', ['Detector', 'Suno scored', r'Suno TPR (\%)', 'Udio scored', r'Udio TPR (\%)'], rows, smoke)
    demos = sorted(next(iter(reports.values()))['official_demonstrations'])
    rows = []
    for name in MODELS:
        row = [tex_escape(LABELS[name])]
        for source in demos:
            point = reports[name]['official_demonstrations'][source]['point']
            row.append(f"{point['TP']:.0f}/{point['scored']:.0f}/{point['attempted']:.0f}")
        rows.append(row)
    table(output/'demo_table.tex', ['Detector', 'Lyria TP/scored/all', 'Stable Audio TP/scored/all'], rows, smoke)
    rows = []
    for stack, result in reports['artifactnet']['operating_stacks'].items():
        metrics = result['metrics']
        rows.append([tex_escape(stack), *(formatted(metrics[k], k in ('TPR', 'FPR')) for k in ('F1', 'TPR', 'FPR'))])
    table(output/'operating_stacks_table.tex', ['ArtifactNet stack', 'F1', r'TPR (\%)', r'FPR (\%)'], rows, smoke)


def native_plot(reports, output, smoke):
    fig, ax = plt.subplots(figsize=(7.6, 4.1), layout='constrained')
    for source_index, (source, label, color) in enumerate(NATIVE):
        for index, name in enumerate(MODELS):
            metric = reports[name]['sources'][source]['metrics']['TPR']
            y = index+(-.14 if source_index == 0 else .14)
            if metric['estimate'] is not None:
                if metric['lower'] is not None:
                    # percentile interval에 point가 포함된다고 가정해 음수 xerr를 만들지 않는다.
                    ax.hlines(y, metric['lower'], metric['upper'], color=color, linewidth=1.5)
                ax.plot(metric['estimate'], y, 'o' if source_index == 0 else 's', color=color,
                        markersize=4)
    ax.set_yticks(range(len(MODELS)), labels=[LABELS[n] for n in MODELS])
    ax.invert_yaxis()
    ax.set(xlim=(-.02, 1.02), xlabel='TPR at raw threshold 0.5; 95% cluster bootstrap interval')
    ax.grid(axis='x', alpha=.2)
    handles = [Line2D([], [], marker='o' if index == 0 else 's', color=color, label=label)
               for index, (_, label, color) in enumerate(NATIVE)]
    fig.legend(handles=handles, loc='outside lower center', ncols=2, fontsize=8)
    save_figure(fig, output/'native_tpr.pdf', smoke)


def source_heatmaps(reports, output, smoke, source_labels):
    first = reports[MODELS[0]]
    for label, metric, cmap in (('ai', 'TPR', 'Blues'), ('real', 'FPR', 'Reds')):
        # 모든 모델이 실패한 cell도 metadata label로 선택하여 행을 유지한다.
        sources = sorted(s for s in first['sources'] if source_labels[s] == label)
        values = np.array([[reports[n]['sources'][s]['metrics'][metric]['estimate']
                            if reports[n]['sources'][s]['metrics'][metric]['estimate'] is not None else np.nan
                            for n in MODELS] for s in sources])
        fig, ax = plt.subplots(figsize=(9.5, max(3.7, .29*len(sources)+1.9)), layout='constrained')
        image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=cmap, aspect='auto')
        for i, s in enumerate(sources):
            for j, n in enumerate(MODELS):
                value = values[i, j]
                ax.text(j, i, '--' if not np.isfinite(value) else f'{100*value:.1f}', ha='center', va='center',
                        fontsize=7, color='white' if np.isfinite(value) and value > .65 else 'black')
        ax.set_yticks(range(len(sources)), labels=[s.replace('_native_260905', '').replace('_', ' ')+f" (n={first['sources'][s]['entries']})" for s in sources], fontsize=7)
        ax.set_xticks(range(len(MODELS)), labels=[LABELS[n] for n in MODELS], rotation=35, ha='right', fontsize=8)
        ax.set_title(f"{metric} by {label}-labelled source cell (%) — {'higher' if label == 'ai' else 'lower'} is better", fontsize=10)
        fig.colorbar(image, ax=ax, fraction=.025, pad=.02, label=metric)
        save_figure(fig, output/(label+'_source_rates.pdf'), smoke)


def export_csv(reports, output):
    with (output/'all_metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'scope', 'cohort', 'metric', 'estimate', 'lower', 'upper', 'finite_replicates', 'replicates'])
        writer.writeheader()
        for name, report in reports.items():
            for scope in ('cohorts', 'sources', 'operating_stacks'):
                for cohort, result in report.get(scope, {}).items():
                    for metric, values in result['metrics'].items():
                        writer.writerow(dict(model=name, scope=scope, cohort=cohort, metric=metric, **values))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--statistics', required=True, type=Path)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--allow-smoke', action='store_true')
    ap.add_argument('--predictions', type=Path,
                    help='Required for final figures: validated public predictions binding these exact statistics')
    args = ap.parse_args()
    paper = Path(__file__).resolve().parent
    if args.allow_smoke and args.output.resolve().is_relative_to(paper):
        raise ValueError('Smoke render outputs must stay outside the manuscript tree')
    reports, summary, release_info = load_report(args.statistics, args.release, args.allow_smoke, args.predictions)
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output/'inputs.json', {'statistics_summary_sha256': digest(args.statistics/'summary.json'),
        'statistics_inputs_sha256': digest(args.statistics/'inputs.json'),
        'manifest_sha256': digest(args.release/'manifest.public.json'), 'release_version': release_info['version'],
        'tool_sha256': digest(__file__), 'matplotlib': matplotlib.__version__,
        'public_predictions_summary_sha256': digest(args.predictions/'summary.json') if not args.allow_smoke else None,
        'scope': 'SMOKE WIRING ONLY' if args.allow_smoke else 'completed statistical report rendering'})
    tables(reports, args.output, args.allow_smoke)
    native_plot(reports, args.output, args.allow_smoke)
    entries = json.loads((args.release/'manifest.public.json').read_text())['bench']
    source_labels = {}
    for entry in entries:
        if entry['source'] in source_labels and source_labels[entry['source']] != entry['label']:
            raise ValueError('Mixed-label source requires explicit plotting policy')
        source_labels[entry['source']] = entry['label']
    source_heatmaps(reports, args.output, args.allow_smoke, source_labels)
    export_csv(reports, args.output)
    outputs = {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.name != 'inputs.json'}
    write_once(args.output/'summary.json', {'status': 'SMOKE WIRING ONLY' if args.allow_smoke else 'completed results rendered',
        'attempted_per_model': summary['attempted_per_model'], 'output_sha256': outputs})
    print(json.dumps({'status': 'smoke wiring' if args.allow_smoke else 'completed report', 'outputs': list(outputs)}))


if __name__ == '__main__':
    main()

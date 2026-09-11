"""완료된 실제 예측을 공개 필드로 투영하고 공개 파일만으로 동일 통계를 재계산한다."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from .common import digest, write_once
from .report import MODELS, generate_reports, load_completed
from .verify_bundle import safe_member
from .verify_release import check_public, verify

RECORD_FIELDS = ('id', 'label', 'source', 'track_id', 'partition', 'recording_group',
                 'recording_representative', 'dependence_cluster', 'audio_sha256', 'seed', 'outcome')
RUNTIME_FIELDS = ('model', 'device', 'expected', 'smoke_per_source', 'public_manifest_sha256',
                  'release_sha256', 'python', 'packages', 'torch', 'cuda_runtime', 'gpu', 'ffmpeg',
                  'rng', 'audio', 'threshold', 'torch_threads', 'cudnn_deterministic',
                  'cudnn_benchmark', 'tf32_matmul', 'tf32_cudnn', 'hf_hub_offline', 'scope')


def probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid probability in public prediction')
    return value


def require_final_input_condition(identity, rows):
    """원래 DeepFense adapter audit를 최종 공개 비교로 승격하지 못하게 한다."""
    if identity['model'] != 'deepfense':
        return
    condition = 'deepfense-native-soxr/2'
    if identity.get('input_condition', {}).get('name') != condition:
        raise ValueError('Final DeepFense results require the corrected native input condition; original adapter remains an audit')
    for row in rows.values():
        if row['outcome'] != 'input_execution_error':
            metadata = row.get('input_preprocessing', {})
            if not isinstance(metadata, dict) or metadata.get('condition') != condition:
                raise ValueError('Final DeepFense row uses a different native input condition')
            public_preprocessing(metadata)


def public_record(row):
    result = {field: row[field] for field in RECORD_FIELDS}
    if row['seed'] != int(hashlib.sha256(('260905:'+row['id']).encode()).hexdigest()[:8], 16):
        raise ValueError('Prediction seed differs from declared per-entry policy')
    if row['outcome'] == 'scored':
        result['prob'] = probability(row['prob'])
        if 'rescue_prob' in row:
            result['rescue_prob'] = probability(row['rescue_prob'])
            if type(row['production_ai']) is not bool or row['production_ai'] != (row['prob'] >= .225 and row['rescue_prob'] >= .02):
                raise ValueError('Production decision differs from fixed thresholds')
            result['production_ai'] = row['production_ai']
            result['n_chunks'] = row['n_chunks']
    elif row['outcome'] in ('model_execution_error', 'input_execution_error'):
        if 'prob' in row or 'rescue_prob' in row or 'production_ai' in row:
            raise ValueError('Failed prediction carries a probability or decision')
        if not re.fullmatch('[A-Za-z][A-Za-z0-9_]*', row['error_type']):
            raise ValueError('Unsafe error class')
        result['error_type'] = row['error_type']
    else:
        raise ValueError('Unknown prediction outcome')
    if 'input_preprocessing' in row:
        result['input_preprocessing'] = public_preprocessing(row['input_preprocessing'])
    for field in ('decoded_frames', 'decode_seconds', 'inference_seconds', 'peak_gpu_bytes'):
        if field in row:
            if type(row[field]) not in (int, float) or not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError('Invalid measured runtime field')
            result[field] = row[field]
    check_public(result)
    return result


def public_preprocessing(value):
    """native DeepFense의 공개 가능한 입력 계약과 수치만 남긴다. 파형·경로는 제외한다."""
    fields = ('condition', 'decoder', 'upstream_decoder_error_type', 'native_sample_rate',
              'native_channels', 'native_frames', 'resampled_frames', 'model_sample_rate',
              'model_frames', 'model_input_sha256', 'peak', 'rms')
    result = {key: value[key] for key in fields}
    if (result['condition'] not in ('deepfense-native-soxr/1', 'deepfense-native-soxr/2') or result['model_sample_rate'] != 16000
            or result['model_frames'] != 64000 or not re.fullmatch('[0-9a-f]{64}', result['model_input_sha256'])):
        raise ValueError('Invalid native preprocessing contract')
    for key in ('native_sample_rate', 'native_channels', 'native_frames', 'resampled_frames',
                'model_sample_rate', 'model_frames'):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError('Invalid native preprocessing size/rate')
    for key in ('peak', 'rms'):
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError('Invalid native preprocessing amplitude')
    routes = {'soundfile_native_f64': None, 'ffmpeg_aac_native_f64': 'LibsndfileError'}
    if result['condition'] == 'deepfense-native-soxr/2':
        routes['ffmpeg_opus_native_f64'] = 'LibsndfileError'
    if result['decoder'] not in routes or result['upstream_decoder_error_type'] != routes[result['decoder']]:
        raise ValueError('Invalid native decoder route')
    check_public(result)
    return result


def relative_hashes(files, root):
    mapped = {}
    for original, checksum in files.items():
        name = Path(original).relative_to(root).as_posix()
        safe_member(name)
        if name in mapped:
            raise ValueError('Ambiguous relative source identity')
        mapped[name] = checksum
    return mapped


def public_identity(identity):
    result = {key: identity[key] for key in RUNTIME_FIELDS}
    if 'input_condition' in identity:
        result['input_condition'] = identity['input_condition']
    anchors = [Path(p) for p in identity['code_hashes'] if p.endswith('/artifactbench/v12/run.py')]
    if len(anchors) != 1:
        raise ValueError('Cannot identify the benchmark source root')
    result['code_hashes'] = relative_hashes(identity['code_hashes'], anchors[0].parents[2])
    assets = identity['model_assets']
    public_assets = {'snapshots_file_sha256': assets['snapshots_file_sha256'],
                     'snapshots': {}, 'source_repos': {}, 'files': {}}
    for name, snapshot in assets['snapshots'].items():
        public_assets['snapshots'][name] = {k: snapshot[k] for k in ('repo', 'revision', 'files')}
        for filename in snapshot['files']:
            safe_member(filename)
    roots = {}
    for name, source in assets['source_repos'].items():
        root = Path(source['root'])
        public_assets['source_repos'][name] = {
            'git_head': source['git_head'], 'python_files': relative_hashes(source['python_files'], root)}
        roots[name] = root.parent if name == 'ArtifactNet' else root
    local_names = {}
    for original, checksum in assets['files'].items():
        path = Path(original)
        candidates = [name+'/'+path.relative_to(root).as_posix() for name, root in roots.items()
                      if path.is_relative_to(root)]
        if len(candidates) > 1:
            raise ValueError('Ambiguous model-asset source root')
        name = candidates[0] if candidates else 'standalone/'+checksum[:12]+'/'+path.name
        safe_member(name)
        if name in public_assets['files']:
            raise ValueError('Colliding logical asset names')
        public_assets['files'][name] = checksum
        local_names[original] = name
    for field in ('operating_thresholds', 'beat_checkpoint_url'):
        if field in assets:
            public_assets[field] = assets[field]
    if 'rescue_checkpoint' in assets:
        public_assets['rescue_checkpoint_file'] = local_names[assets['rescue_checkpoint']]
    result['model_assets'] = public_assets
    result['timing_scope'] = 'Per-entry wall times; peak_gpu_bytes is cumulative torch allocator peak, not isolated model memory.'
    result['privacy_scope'] = 'Absolute paths and error text/tracebacks omitted; original content hashes retained.'
    check_public(result)
    return result


def report_reference(statistics, release, paths, allow_smoke):
    summary = json.loads((statistics/'summary.json').read_text())
    inputs = json.loads((statistics/'inputs.json').read_text())
    expected = 'SMOKE REPORT WIRING ONLY' if allow_smoke else 'statistical report complete'
    if summary['status'] != expected or summary['replicates'] != (20 if allow_smoke else 2000):
        raise ValueError('Statistics scope differs from requested export')
    if set(summary['models']) != set(MODELS) or set(summary['report_sha256']) != set(MODELS):
        raise ValueError('Statistics requires all eight models')
    for source in (release/'manifest.public.json', release/'release.json', *(p/'summary.json' for p in paths.values())):
        matching = [checksum for filename, checksum in inputs['input_hashes'].items()
                    if Path(filename).resolve() == source.resolve()]
        if matching != [digest(source)]:
            raise ValueError('Statistics is not bound to these exact runs/release')
    files = {name+'.json': checksum for name, checksum in summary['report_sha256'].items()}
    files['paired_differences.json'] = summary['paired_differences_sha256']
    for name, checksum in files.items():
        if digest(statistics/name) != checksum:
            raise ValueError('Reference statistics changed')
    return {'replicates': summary['replicates'], 'attempted_per_model': summary['attempted_per_model'],
            'statistics_summary_sha256': digest(statistics/'summary.json'), 'file_sha256': files}


def export_results(release, paths, statistics, output, allow_smoke=False):
    verification = verify(release)
    public = {r['id']: r for r in json.loads((release/'manifest.public.json').read_text())['bench']}
    reference = report_reference(statistics, release, paths, allow_smoke)
    results, selected_ids = {}, None
    for name in MODELS:
        rows, identity = load_completed(paths[name], public, allow_smoke)
        if not allow_smoke:
            require_final_input_condition(identity, rows)
        if identity['model'] != name or identity['public_manifest_sha256'] != verification['manifest_sha256']:
            raise ValueError('Run identity mismatch')
        ids = sorted(rows)
        if selected_ids is None:
            selected_ids = ids
        if ids != selected_ids or len(ids) != reference['attempted_per_model'] or (not allow_smoke and set(ids) != set(public)):
            raise ValueError('Incomplete or inconsistent eight-model membership')
        source = json.loads((paths[name]/'summary.json').read_text())
        records = [dict(public_record(rows[i]), source_record_sha256=source['record_sha256'][i]) for i in ids]
        result = {'model': name, 'identity': public_identity(identity), 'records': records,
                  'source_run_summary_sha256': digest(paths[name]/'summary.json'),
                  'source_run_identity_sha256': digest(paths[name]/'identity.json'), 'outcomes': source['outcomes']}
        check_public(result)
        results[name] = result
    output.mkdir(parents=True, exist_ok=False)
    for name, result in results.items():
        write_once(output/(name+'.json'), result)
    summary = {'schema': 'artifactbench-public-predictions/1',
               'status': 'SMOKE EXPORT WIRING ONLY' if allow_smoke else 'completed public prediction export',
               'manifest_sha256': verification['manifest_sha256'], 'release_sha256': digest(release/'release.json'),
               'models': list(MODELS), 'attempted_per_model': len(selected_ids), 'reference_statistics': reference,
               'file_sha256': {name+'.json': digest(output/(name+'.json')) for name in MODELS},
               'tool_sha256': digest(__file__),
               'limits': 'Saved-score/statistics reproduction, not fresh audio inference or a universal unseen test.'}
    check_public(summary)
    write_once(output/'summary.json', summary)
    return summary


def load_public_results(directory, release, allow_smoke=False):
    verification = verify(release)
    summary = json.loads((directory/'summary.json').read_text())
    check_public(summary)
    expected = 'SMOKE EXPORT WIRING ONLY' if allow_smoke else 'completed public prediction export'
    if summary['status'] != expected or summary['reference_statistics']['replicates'] != (20 if allow_smoke else 2000):
        raise ValueError('Public prediction export scope mismatch')
    expected_reports = {n+'.json' for n in MODELS} | {'paired_differences.json'}
    if set(summary['reference_statistics']['file_sha256']) != expected_reports:
        raise ValueError('All nine reference statistical files are required')
    if summary['manifest_sha256'] != verification['manifest_sha256'] or summary['release_sha256'] != digest(release/'release.json'):
        raise ValueError('Public prediction release mismatch')
    if set(summary['models']) != set(MODELS) or set(summary['file_sha256']) != {n+'.json' for n in MODELS}:
        raise ValueError('All eight public prediction files are required')
    public = {r['id']: r for r in json.loads((release/'manifest.public.json').read_text())['bench']}
    runs, identities, selected_ids = {}, {}, None
    for name in MODELS:
        path = directory/(name+'.json')
        if digest(path) != summary['file_sha256'][name+'.json']:
            raise ValueError('Public prediction hash mismatch')
        result = json.loads(path.read_text())
        check_public(result)
        if result['model'] != name or result['identity']['model'] != name or result['identity']['public_manifest_sha256'] != summary['manifest_sha256']:
            raise ValueError('Public model identity mismatch')
        rows = {}
        for row in result['records']:
            identifier = row['id']
            if identifier in rows or identifier not in public:
                raise ValueError('Duplicate or unknown public prediction ID')
            cleaned = public_record(row)
            if set(row) != set(cleaned) | {'source_record_sha256'}:
                raise ValueError('Unapproved public prediction field')
            if not re.fullmatch('[0-9a-f]{64}', row['source_record_sha256']):
                raise ValueError('Invalid source record hash')
            for field in ('id', 'label', 'source', 'track_id', 'partition', 'recording_group', 'recording_representative', 'dependence_cluster'):
                if row[field] != public[identifier][field]:
                    raise ValueError('Public prediction membership field mismatch')
            if row['audio_sha256'] != public[identifier]['sha256']:
                raise ValueError('Public prediction audio identity mismatch')
            rows[identifier] = cleaned
        if dict(Counter(r['outcome'] for r in rows.values())) != result['outcomes']:
            raise ValueError('Public coverage mismatch')
        if not allow_smoke:
            require_final_input_condition(result['identity'], rows)
        ids = sorted(rows)
        if selected_ids is None:
            selected_ids = ids
        if (ids != selected_ids or len(ids) != summary['attempted_per_model'] or
                len(ids) != result['identity']['expected'] or
                (not allow_smoke and (set(ids) != set(public) or result['identity']['smoke_per_source'] != 0))):
            raise ValueError('Public prediction frame incomplete')
        runs[name], identities[name] = rows, result['identity']
    return public, runs, identities, summary


def reproduce_results(directory, release, output, allow_smoke=False):
    public, runs, identities, summary = load_public_results(directory, release, allow_smoke)
    output.mkdir(parents=True, exist_ok=False)
    write_once(output/'inputs.json', {'public_prediction_summary_sha256': digest(directory/'summary.json'),
        'input_hashes': {'manifest.public.json': digest(release/'manifest.public.json')},
        'tool_sha256': digest(__file__), 'report_tool_sha256': digest(Path(__file__).with_name('report.py')),
        'statistics_tool_sha256': digest(Path(__file__).with_name('statistics.py')),
        'scope': 'Saved public scores only; no audio or model inference'})
    generate_reports(public, runs, identities, output, allow_smoke)
    for name, checksum in summary['reference_statistics']['file_sha256'].items():
        safe_member(name)
        if digest(output/name) != checksum:
            raise ValueError('Recomputed statistics differ from original: '+name)
    result = {'status': 'SMOKE REPRODUCTION WIRING ONLY' if allow_smoke else 'public-score statistical reproduction passed',
              'compared_statistical_files': len(summary['reference_statistics']['file_sha256']),
              'comparison': 'byte-identical JSON including all intervals and paired model differences',
              'scope': 'Does not re-run detectors or establish audio availability'}
    write_once(output/'reproduction.json', result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--runs', required=True, type=Path)
    export.add_argument('--statistics', required=True, type=Path)
    export.add_argument('--run-override', action='append', default=[])
    reproduce = sub.add_parser('reproduce')
    reproduce.add_argument('--predictions', required=True, type=Path)
    for parser in (export, reproduce):
        parser.add_argument('--release', required=True, type=Path)
        parser.add_argument('--output', required=True, type=Path)
        parser.add_argument('--allow-smoke', action='store_true')
    args = ap.parse_args()
    if args.allow_smoke and args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]/'paper'):
        raise ValueError('Smoke artifacts must stay outside manuscript tree')
    if args.command == 'export':
        paths = {name: args.runs/name for name in MODELS}
        for override in args.run_override:
            name, directory = override.split('=', 1)
            if name not in paths:
                raise ValueError('Unknown model override')
            paths[name] = Path(directory)
        result = export_results(args.release, paths, args.statistics, args.output, args.allow_smoke)
    else:
        result = reproduce_results(args.predictions, args.release, args.output, args.allow_smoke)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

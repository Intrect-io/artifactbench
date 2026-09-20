"""완료된 공개 점수·동일 통계·최소 실행 소스를 결정적 ZIP으로 묶는다."""
import argparse
import json
from pathlib import Path
import re
import stat
import zipfile

from .common import digest, write_once
from .package_metadata import MANIFEST_SHA256, PUBLIC_FILES, public_payload
from .public_results import load_public_results
from .report import MODELS
from .verify_bundle import safe_member
from .verify_release import check_public

SOURCE_FILES = {
    'README.md': 'docs/RESULTS_BUNDLE_README.md',
    **{name: name for name in (
        'LICENSE', 'requirements-notebook.lock', 'requirements-results.lock',
        'artifactbench/__init__.py', 'artifactbench/v12/__init__.py',
        'artifactbench/v12/common.py', 'artifactbench/v12/report.py',
        'artifactbench/v12/statistics.py', 'artifactbench/v12/public_results.py',
        'artifactbench/v12/verify_release.py', 'artifactbench/v12/verify_bundle.py',
        'artifactbench/v12/package_metadata.py', 'artifactbench/v12/package_results.py',
        'artifactbench/v12/execute_results_notebook.py',
        'notebooks/v1.2_results_reproduction.ipynb')},
}
REPORT_FILES = tuple(name+'.json' for name in MODELS) + ('paired_differences.json',)
SCOPE = 'saved public scores and statistics; no audio, weights, or fresh inference'


def bundle_name(allow_smoke):
    return 'artifactbench-results-SMOKE-ONLY' if allow_smoke else 'artifactbench-1.2-rc2-results'


def expected_files():
    return (set(SOURCE_FILES) | {'release/'+name for name in PUBLIC_FILES}
            | {'predictions/'+name for name in (*[n+'.json' for n in MODELS], 'summary.json')}
            | {'statistics/'+name for name in (*REPORT_FILES, 'inputs.json', 'reproduction.json')})


def check_reproduction(statistics, predictions, summary, root, allow_smoke):
    proof = json.loads((statistics/'reproduction.json').read_text())
    inputs = json.loads((statistics/'inputs.json').read_text())
    expected = 'SMOKE REPRODUCTION WIRING ONLY' if allow_smoke else 'public-score statistical reproduction passed'
    if proof['status'] != expected or proof['compared_statistical_files'] != 9:
        raise ValueError('Bundle requires a successful nine-file reproduction')
    if (inputs['public_prediction_summary_sha256'] != digest(predictions/'summary.json')
            or inputs['input_hashes']['manifest.public.json'] != summary['manifest_sha256']):
        raise ValueError('Reproduction is not bound to these public predictions')
    for key, name in (('tool_sha256', 'public_results.py'), ('report_tool_sha256', 'report.py'),
                      ('statistics_tool_sha256', 'statistics.py')):
        if inputs[key] != digest(root/'artifactbench/v12'/name):
            raise ValueError('Bundled reporting source differs from verified reproduction')
    for name in REPORT_FILES:
        if digest(statistics/name) != summary['reference_statistics']['file_sha256'][name]:
            raise ValueError('Bundled statistics differ from reference: '+name)


def verify_results_bundle(directory, allow_smoke=False):
    directory = Path(directory).resolve()
    manifest_path = directory/'BUNDLE_MANIFEST.json'
    if manifest_path.is_symlink():
        raise ValueError('Symbolic bundle manifest')
    manifest = json.loads(manifest_path.read_text())
    check_public(manifest)
    if (manifest['schema'] != 'artifactbench-results-bundle/1' or manifest['scope'] != SCOPE
            or type(manifest['smoke_only']) is not bool or manifest['smoke_only'] != allow_smoke):
        raise ValueError('Results bundle scope mismatch')
    if set(manifest['file_sha256']) != expected_files():
        raise ValueError('Unexpected results bundle inventory')
    actual = set()
    for path in directory.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symbolic link in results bundle')
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != expected_files() | {'BUNDLE_MANIFEST.json'}:
        raise ValueError('Unlisted or missing results bundle payload')
    for name, checksum in manifest['file_sha256'].items():
        safe_member(name)
        if not re.fullmatch('[0-9a-f]{64}', checksum) or digest(directory/name) != checksum:
            raise ValueError('Results bundle hash mismatch: '+name)
        public_payload(directory/name)
    _, _, _, summary = load_public_results(directory/'predictions', directory/'release', allow_smoke)
    if summary['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Results bundle release identity mismatch')
    if not allow_smoke and summary['manifest_sha256'] != MANIFEST_SHA256:
        raise ValueError('Full results bundle requires pinned rc2')
    check_reproduction(directory/'statistics', directory/'predictions', summary, directory, allow_smoke)
    return {'status': 'SMOKE BUNDLE WIRING ONLY' if allow_smoke else 'public-score bundle verified',
            'checked_files': len(actual)-1, 'manifest_sha256': summary['manifest_sha256'],
            'attempted_per_model': summary['attempted_per_model'], 'scope': SCOPE}


def build_results_bundle(root, release, predictions, statistics, output, allow_smoke=False):
    root, release, predictions, statistics, output = (
        Path(p).resolve() for p in (root, release, predictions, statistics, output))
    if allow_smoke and output.is_relative_to(root/'paper'):
        raise ValueError('Smoke artifacts cannot enter the manuscript tree')
    _, _, _, summary = load_public_results(predictions, release, allow_smoke)
    if not allow_smoke and summary['manifest_sha256'] != MANIFEST_SHA256:
        raise ValueError('Full results bundle requires pinned rc2')
    check_reproduction(statistics, predictions, summary, root, allow_smoke)
    inputs = {name: root/source for name, source in SOURCE_FILES.items()}
    inputs.update({'release/'+name: release/name for name in PUBLIC_FILES})
    inputs.update({'predictions/'+name: predictions/name for name in (*[n+'.json' for n in MODELS], 'summary.json')})
    inputs.update({'statistics/'+name: statistics/name for name in (*REPORT_FILES, 'inputs.json', 'reproduction.json')})
    for name, path in inputs.items():
        safe_member(name)
        if path.resolve() != path or not path.is_file():
            raise ValueError('Input missing or symbolic link: '+name)
    hashes = {name: digest(path) for name, path in inputs.items()}
    payloads = {name: public_payload(path) for name, path in inputs.items()}
    output.mkdir(parents=True, exist_ok=False)
    bundle = output/bundle_name(allow_smoke)
    bundle.mkdir()
    for name, payload in payloads.items():
        target = bundle/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    write_once(bundle/'BUNDLE_MANIFEST.json', {
        'schema': 'artifactbench-results-bundle/1', 'scope': SCOPE, 'smoke_only': allow_smoke,
        'manifest_sha256': summary['manifest_sha256'], 'file_sha256': hashes,
        'archive_timestamp': '2026-09-05T00:00:00 (fixed packaging epoch, not retrieval time)',
    })
    verification = verify_results_bundle(bundle, allow_smoke)
    archive = output/(bundle.name+'.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for name in sorted(expected_files() | {'BUNDLE_MANIFEST.json'}):
            member = zipfile.ZipInfo(bundle.name+'/'+name, (2026, 9, 5, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            stream.writestr(member, (bundle/name).read_bytes(), compresslevel=9)
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None or len(stream.namelist()) != len(inputs)+1:
            raise ValueError('Results ZIP verification failed')
    if any(digest(path) != hashes[name] for name, path in inputs.items()):
        raise ValueError('Results bundle input changed during packaging')
    result = {'status': 'SMOKE ARCHIVE WIRING ONLY' if allow_smoke else 'public-score candidate packaged; not uploaded',
              'archive': archive.name, 'archive_sha256': digest(archive),
              'archive_bytes': archive.stat().st_size, 'verification': verification}
    write_once(output/'summary.json', result)
    return result


def extract_results_bundle(archive, destination, allow_smoke=False):
    """로컬 후보 ZIP도 전체 목록·경로·파일 타입을 확인한 뒤 새 경로에만 푼다."""
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Extraction destination must be empty')
    expected = {bundle_name(allow_smoke)+'/'+name for name in expected_files() | {'BUNDLE_MANIFEST.json'}}
    with zipfile.ZipFile(archive) as stream:
        members = stream.infolist()
        if len({m.filename for m in members}) != len(members):
            raise ValueError('Duplicate ZIP member names')
        for member in members:
            safe_member(member.filename)
            if not stat.S_ISREG(member.external_attr >> 16):
                raise ValueError('Non-regular file in results ZIP')
        if {m.filename for m in members} != expected:
            raise ValueError('Unexpected results ZIP inventory or scope')
        if sum(m.file_size for m in members) > 256*1024*1024:
            raise ValueError('Results ZIP exceeds declared 256 MiB payload limit')
        stream.extractall(destination)
    bundle = destination/bundle_name(allow_smoke)
    verify_results_bundle(bundle, allow_smoke)
    return bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    for name in ('release', 'predictions', 'reproduction', 'output'):
        build.add_argument('--'+name, required=True, type=Path)
    verify = sub.add_parser('verify')
    verify.add_argument('--bundle', required=True, type=Path)
    for parser in (build, verify):
        parser.add_argument('--allow-smoke', action='store_true')
    args = ap.parse_args()
    if args.command == 'build':
        result = build_results_bundle(Path(__file__).resolve().parents[2], args.release,
            args.predictions, args.reproduction, args.output, args.allow_smoke)
    else:
        result = verify_results_bundle(args.bundle, args.allow_smoke)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

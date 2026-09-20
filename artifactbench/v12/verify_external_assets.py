"""공개 prediction identity와 외부 모델 자산의 byte identity를 비교한다. 모델은 로드하지 않는다."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from .common import digest, write_once
from .verify_bundle import safe_member
from .verify_release import check_public


MODELS = {'spectttra', 'spectttra_beta5s', 'ast_60s', 'deepfense', 'deezer_ismir', 'clam', 'fst'}
REPOSITORIES = {'MoM-CLAM', 'FST'}


def checked_file(path, expected):
    if not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('Invalid expected SHA-256')
    if path is None:
        return {'status': 'location_not_supplied', 'expected_sha256': expected}
    path = Path(path)
    row = {'path': str(path.absolute()), 'expected_sha256': expected}
    try:
        before = path.stat()
        if not path.is_file():
            return dict(row, status='not_a_file')
        actual = digest(path)
        after = path.stat()
    except OSError as exc:
        return dict(row, status='unreadable', error_type=type(exc).__name__)
    stable = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    return dict(row, status='changed_during_read' if not stable else 'match' if actual == expected else 'hash_mismatch',
                actual_sha256=actual, bytes=after.st_size)


def verify(reference, snapshots_file, roots, beat_checkpoint=None):
    reference, snapshots_file = Path(reference), Path(snapshots_file)
    input_hashes = {str(p): digest(p) for p in (reference, snapshots_file)}
    document = json.loads(reference.read_text())
    identity = document['identity']
    check_public(identity)
    model = identity['model']
    if model not in MODELS or document['model'] != model:
        raise ValueError('Expected one public external-baseline prediction identity')
    assets = identity['model_assets']
    if not assets['snapshots'] or set(assets['source_repos']) - REPOSITORIES:
        raise ValueError('Unexpected external-model asset inventory')
    rows = json.loads(snapshots_file.read_text())
    snapshots = {r['repo']: r for r in rows}
    if len(rows) != len(snapshots):
        raise ValueError('Duplicate local snapshot repository')
    checks = {}
    for name, expected in assets['snapshots'].items():
        safe_member(name)
        if name != expected['repo'] or not re.fullmatch('[0-9a-f]{40}', expected['revision']) or not expected['files']:
            raise ValueError('Invalid expected snapshot identity')
        local = snapshots.get(name)
        prefix = 'hf/'+name
        if local is None:
            checks[prefix] = {'status': 'snapshot_not_supplied'}
        elif (local['revision'] != expected['revision'] or Path(local['snapshot']).name != expected['revision']
                or local['files'] != expected['files']):
            checks[prefix] = {'status': 'snapshot_identity_mismatch'}
        else:
            for filename, checksum in expected['files'].items():
                safe_member(filename)
                checks[prefix+'/'+filename] = checked_file(Path(local['snapshot'])/filename, checksum)
    for name, expected in assets['source_repos'].items():
        root = roots.get(name)
        if not expected['python_files'] or not re.fullmatch('[0-9a-f]{40}', expected['git_head']):
            raise ValueError('Invalid expected source identity')
        if root is not None:
            root = Path(root)
            # git metadata만 읽는다. 외부 Python 모듈을 import하거나 checkpoint를 역직렬화하지 않는다.
            command = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
            head = command.stdout.strip()
            checks['source/'+name+'/git_head'] = {'status': 'match' if command.returncode == 0 and head == expected['git_head']
                                                 else 'git_revision_mismatch', 'expected': expected['git_head'],
                                                 'actual': head if command.returncode == 0 else None}
            actual_files = {p.relative_to(root).as_posix() for p in root.rglob('*.py')
                            if not any(s.startswith('.') or s == '__pycache__' for s in p.relative_to(root).parts)}
            checks['source/'+name+'/python_inventory'] = {
                'status': 'match' if actual_files == set(expected['python_files']) else 'python_inventory_mismatch',
                'extra': sorted(actual_files-set(expected['python_files'])),
                'missing': sorted(set(expected['python_files'])-actual_files)}
        for filename, checksum in expected['python_files'].items():
            safe_member(filename)
            checks['source/'+name+'/'+filename] = checked_file(root/filename if root is not None else None, checksum)
    for name, checksum in assets['files'].items():
        parts = safe_member(name).parts
        if parts[0] in assets['source_repos'] and len(parts) > 1:
            root = roots.get(parts[0])
            path = Path(root).joinpath(*parts[1:]) if root is not None else None
        elif parts == ('standalone', checksum[:12], 'final0.ckpt') and model == 'fst':
            path = beat_checkpoint
        else:
            raise ValueError('Unknown logical external checkpoint: '+name)
        checks['checkpoint/'+name] = checked_file(path, checksum)
    if any(digest(path) != checksum for path, checksum in input_hashes.items()):
        raise ValueError('Verification inputs changed during inspection')
    passed = bool(checks) and all(row['status'] == 'match' for row in checks.values())
    return {'status': 'external asset identity matched' if passed else 'external asset identity incomplete or mismatched',
            'passed': passed, 'model': model, 'reference_expected_entries': identity['expected'],
            'reference_smoke_per_source': identity['smoke_per_source'],
            'inputs_sha256': input_hashes, 'checks': checks,
            'scope': 'Private path-bearing byte/code-identity audit only; no model execution, audio check, score reproduction, rights grant or independent publication authentication'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', required=True, type=Path, help='Public prediction JSON from a verified results bundle')
    ap.add_argument('--snapshots', required=True, type=Path)
    ap.add_argument('--clam-repo', type=Path, default=os.environ.get('ARTIFACTBENCH_CLAM_REPO'))
    ap.add_argument('--fst-repo', type=Path, default=os.environ.get('ARTIFACTBENCH_FST_REPO'))
    ap.add_argument('--beat-checkpoint', type=Path, default=os.environ.get('ARTIFACTBENCH_BEAT_CHECKPOINT'))
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('Use a new audit output directory')
    result = verify(args.reference, args.snapshots, {'MoM-CLAM': args.clam_repo, 'FST': args.fst_repo}, args.beat_checkpoint)
    write_once(args.output/'audit.local.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'inputs_sha256')}, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

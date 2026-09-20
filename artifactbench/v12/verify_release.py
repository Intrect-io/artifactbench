"""표준 라이브러리만으로 공개 메타데이터의 해시·분모·개인 경로 분리를 검증한다."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from .common import digest


def check_public(value):
    forbidden = {'path', 'original_path', 'primary_path', 'lyrics', 'prompt', 'creator_alias'}
    if isinstance(value, dict):
        if forbidden.intersection(value):
            raise ValueError('Private/raw-content field in public metadata')
        for child in value.values():
            check_public(child)
    elif isinstance(value, list):
        for child in value:
            check_public(child)
    elif isinstance(value, str):
        if value.startswith(('http://', 'https://')):
            parsed = urlsplit(value)
            if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query:
                raise ValueError('Public URL requires access/privacy review')
        elif any(prefix in value for prefix in ('/home/', '/media/', '/Volumes/', '/Users/')):
            raise ValueError('Machine-local path in public metadata')


def verify(directory):
    summary = json.loads((directory/'release.json').read_text())
    for filename, expected in summary['public_file_sha256'].items():
        if Path(filename).name != filename:
            raise ValueError('Unsafe relative public filename')
        if digest(directory/filename) != expected:
            raise ValueError(f'Public file hash mismatch: {filename}')
        check_public(json.loads((directory/filename).read_text()))
    entries = json.loads((directory/'manifest.public.json').read_text())['bench']
    if len(entries) != summary['evaluation_entries'] or len({r['id'] for r in entries}) != len(entries):
        raise ValueError('Entry count or uniqueness mismatch')
    for field, expected in (('partition', summary['partitions']), ('source', summary['source_counts']), ('label', summary['labels'])):
        if dict(Counter(r[field] for r in entries)) != expected:
            raise ValueError(f'Count mismatch: {field}')
    groups = {r['recording_group'] for r in entries}
    representatives = Counter(r['recording_group'] for r in entries if r['recording_representative'])
    if len(groups) != summary['identified_recording_groups'] or any(representatives[g] != 1 for g in groups):
        raise ValueError('Recording grouping/representative mismatch')
    if len({r['dependence_cluster'] for r in entries}) != summary['dependence_clusters']:
        raise ValueError('Dependence cluster count mismatch')
    for row in entries:
        if not re.fullmatch('[0-9a-f]{64}', row['sha256']) or row['duration_seconds'] <= 0 or row['sample_rate'] <= 0:
            raise ValueError('Invalid audio metadata')
    pairs = json.loads((directory/'native_pairs.public.json').read_text())
    ids = {r['id']: r for r in entries}
    if len(pairs) != summary['native_transport_pairs']:
        raise ValueError('Paired transport count mismatch')
    for row in pairs:
        if row['primary_id'] not in ids or row['primary_sha256'] != ids[row['primary_id']]['sha256']:
            raise ValueError('Unmatched paired transport primary')
    return {'status': 'metadata verification passed', 'entries': len(entries), 'recording_groups': len(groups),
            'paired_views': len(pairs), 'manifest_sha256': summary['public_file_sha256']['manifest.public.json'],
            'scope': 'metadata only; does not verify audio availability, model scores, or redistribution rights'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(verify(args.release), indent=2))


if __name__ == '__main__':
    main()

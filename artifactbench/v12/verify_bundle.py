"""공개 메타데이터 번들의 파일 목록·해시·고정 release를 표준 라이브러리로 검사한다."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re

from .common import digest
from .verify_release import check_public, verify


def safe_member(name):
    if not isinstance(name, str) or any(char in name for char in ('\\', ':', '\0')):
        raise ValueError('Unsafe bundle member name')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != name or not path.parts:
        raise ValueError('Unsafe bundle member name')
    return path


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'BUNDLE_MANIFEST.json').read_text())
    check_public(manifest)
    if manifest['scope'] != 'metadata-only; no detector result reproduction':
        raise ValueError('Unsupported bundle scope')
    for name, expected in manifest['file_sha256'].items():
        safe_member(name)
        path = directory/name
        if path.resolve() != path or not path.is_file():
            raise ValueError('Bundle file missing or symbolic link: '+name)
        if not re.fullmatch('[0-9a-f]{64}', expected) or digest(path) != expected:
            raise ValueError('Bundle file hash mismatch: '+name)
    release_name = manifest['release_directory']
    safe_member(release_name)
    required = {'release.json', 'manifest.public.json', 'native_pairs.public.json', 'exposure.public.json'}
    if not {release_name+'/'+name for name in required}.issubset(manifest['file_sha256']):
        raise ValueError('Release files missing from bundle inventory')
    result = verify(directory/release_name)
    if result['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Bundle release identity mismatch')
    return dict(status='metadata bundle verification passed', checked_files=len(manifest['file_sha256']),
                release=result, scope=manifest['scope'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', required=True, type=Path)
    print(json.dumps(verify_bundle(ap.parse_args().bundle), indent=2))


if __name__ == '__main__':
    main()

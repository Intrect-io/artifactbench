"""허용 목록의 공개 메타데이터만 결정적 ZIP으로 묶는다. 업로드는 하지 않는다."""
import argparse
import json
from pathlib import Path
import re
import zipfile

from .common import digest, write_once
from .verify_bundle import safe_member, verify_bundle
from .verify_release import check_public, verify


RELEASE_NAME = 'out/v1.2_frozen_rc2_260905'
MANIFEST_SHA256 = 'feab7c4c3d037919fd784dc40e46199470088abde532c5181e34573f20dceefd'
PUBLIC_FILES = ('release.json', 'manifest.public.json', 'native_pairs.public.json', 'exposure.public.json')
SOURCE_FILES = {
    'README.md': 'docs/METADATA_BUNDLE_README.md',
    **{name: name for name in (
        'LICENSE', 'requirements-notebook.lock', 'artifactbench/__init__.py',
        'artifactbench/v12/__init__.py', 'artifactbench/v12/common.py',
        'artifactbench/v12/verify_release.py', 'artifactbench/v12/verify_bundle.py',
        'artifactbench/v12/execute_notebook.py', 'notebooks/v1.2_metadata_quickstart.ipynb',
        'docs/DATASET_CARD_v1.2.md', 'docs/PROTOCOL_v1.2.md', 'docs/STATISTICS_v1.2.md',
        'docs/TRANSPORT_v1.2.md', 'docs/DUPLICATE_AUDIT_v1.2.md',
        'docs/UDIO_IDENTITY_CORRECTION_v1.2.md')},
}


def public_payload(path):
    payload = path.read_bytes()
    text = payload.decode('utf-8')
    if path.suffix in ('.json', '.ipynb'):
        document = json.loads(text)
        check_public(document)
        if path.suffix == '.ipynb':
            if any(c.get('outputs') or c.get('execution_count') is not None
                   for c in document['cells'] if c['cell_type'] == 'code'):
                raise ValueError('Bundle requires an unexecuted source notebook')
    else:
        # 공개 URL은 로컬 파일 경로가 아니다. JSON은 위의 구조 검사를 사용한다.
        without_urls = re.sub(r'''https?://[^\s<>"']+''', '', text)
        if re.search(r'/(?:home|media|Volumes|Users)/[A-Za-z0-9_.-]', without_urls):
            raise ValueError('Machine-local path in bundle input: '+path.name)
    return payload


def build_bundle(root, release, output):
    root, release, output = (Path(p).resolve() for p in (root, release, output))
    verified = verify(release)
    if verified['manifest_sha256'] != MANIFEST_SHA256:
        raise ValueError('Metadata bundle requires the pinned rc2 release')
    inputs = {name: root/source for name, source in SOURCE_FILES.items()}
    inputs.update({RELEASE_NAME+'/'+name: release/name for name in PUBLIC_FILES})
    for name, path in inputs.items():
        safe_member(name)
        if path.resolve() != path or not path.is_file():
            raise ValueError('Input missing or symbolic link: '+name)
    payloads = {name: public_payload(path) for name, path in inputs.items()}
    input_hashes = {name: digest(path) for name, path in inputs.items()}
    output.mkdir(parents=True, exist_ok=False)
    bundle = output/'artifactbench-1.2-rc2-metadata'
    bundle.mkdir()
    for name, payload in payloads.items():
        target = bundle/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if digest(target) != input_hashes[name]:
            raise ValueError('Input changed while being packaged: '+name)
    write_once(bundle/'BUNDLE_MANIFEST.json', {
        'schema': 'artifactbench-metadata-bundle/1',
        'scope': 'metadata-only; no detector result reproduction',
        'release_directory': RELEASE_NAME, 'manifest_sha256': MANIFEST_SHA256,
        'file_sha256': input_hashes, 'builder_sha256': digest(__file__),
        'archive_timestamp': '2026-09-05T00:00:00 (fixed packaging epoch, not retrieval time)',
    })
    verification = verify_bundle(bundle)
    archive = output/(bundle.name+'.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                member = zipfile.ZipInfo(bundle.name+'/'+path.relative_to(bundle).as_posix(), (2026, 9, 5, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o100644 << 16
                stream.writestr(member, path.read_bytes(), compresslevel=9)
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None or len(stream.namelist()) != len(inputs)+1:
            raise ValueError('Packaged ZIP verification failed')
    if any(digest(path) != input_hashes[name] for name, path in inputs.items()):
        raise ValueError('Bundle source changed during packaging')
    summary = {'status': 'metadata-only candidate packaged; not uploaded',
               'archive': archive.name, 'archive_sha256': digest(archive),
               'archive_bytes': archive.stat().st_size, 'verification': verification}
    write_once(output/'summary.json', summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(build_bundle(Path(__file__).resolve().parents[2], args.release, args.output), indent=2))


if __name__ == '__main__':
    main()

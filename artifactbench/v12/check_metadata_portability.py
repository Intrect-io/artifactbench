"""새 임시 디렉터리에 번들을 풀어 site 없는 검증과 별도 커널 실행을 실제 수행한다."""
import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import zipfile

from .common import digest, write_once
from .verify_bundle import safe_member


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', required=True, type=Path)
    ap.add_argument('--notebook-python', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    extracted = Path(tempfile.mkdtemp(prefix='artifactbench-metadata-'))
    # 로컬 생성 ZIP이어도 추출 경계와 symlink를 검사한다.
    with zipfile.ZipFile(args.archive) as archive:
        members = archive.infolist()
        if len({m.filename for m in members}) != len(members):
            raise ValueError('Duplicate ZIP member names')
        for member in members:
            safe_member(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Symbolic link in ZIP')
        roots = {safe_member(m.filename).parts[0] for m in members}
        if roots != {'artifactbench-1.2-rc2-metadata'}:
            raise ValueError('Unexpected archive root')
        archive.extractall(extracted)
    bundle = extracted/'artifactbench-1.2-rc2-metadata'
    manifest = json.loads((bundle/'BUNDLE_MANIFEST.json').read_text())
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    if actual != set(manifest['file_sha256']) | {'BUNDLE_MANIFEST.json'}:
        raise ValueError('Unlisted archive payload')
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    commands = {
        'stdlib': [sys.executable, '-S', '-m', 'artifactbench.v12.verify_bundle', '--bundle', '.'],
        # venv Python은 resolve하면 base interpreter로 바뀌므로 absolute만 사용한다.
        'notebook': [str(args.notebook_python.absolute()), '-m', 'artifactbench.v12.execute_notebook',
                     '--notebook', 'notebooks/v1.2_metadata_quickstart.ipynb',
                     '--release', manifest['release_directory'], '--output', str(output/'notebook')],
    }
    inputs = {'archive_sha256': digest(args.archive), 'tool_sha256': digest(__file__),
              'extracted_root': str(bundle), 'commands': commands,
              'scope': 'fresh extraction/import roots; existing isolated dependency environment, not a new package install'}
    write_once(output/'inputs.json', inputs)
    for name, command in commands.items():
        result = subprocess.run(command, cwd=bundle, env=env, capture_output=True, text=True, timeout=180)
        write_once(output/(name+'_process.json'), {'returncode': result.returncode,
                   'stdout': result.stdout, 'stderr': result.stderr})
        if result.returncode:
            raise RuntimeError(name+' failed; see preserved process output')
    stdlib_result = json.loads((output/'stdlib_process.json').read_text())
    verification = json.loads(stdlib_result['stdout'])
    notebook = json.loads((output/'notebook/summary.json').read_text())
    if notebook['executed_code_cells'] != 6:
        raise ValueError('Unexpected notebook execution count')
    if digest(args.archive) != inputs['archive_sha256']:
        raise ValueError('Archive changed during validation')
    summary = {'status': 'independent metadata extraction and notebook checks passed',
               'archive_sha256': inputs['archive_sha256'], 'stdlib': verification, 'notebook': notebook,
               'scope': 'metadata only; no audio availability, detector accuracy, or complete inference reproduction claim'}
    write_once(output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

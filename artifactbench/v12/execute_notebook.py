"""명시한 환경에서 실제 커널을 실행하고 오류를 성공으로 바꾸지 않은 채 산출물을 보존한다."""
import argparse
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

from .common import digest, write_once


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--notebook', required=True, type=Path)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    # 새 실행은 새 출력 디렉터리: 실패한 노트북도 삭제하거나 덮어쓰지 않는다.
    args.output.mkdir(parents=True, exist_ok=False)
    notebook = nbformat.read(args.notebook, as_version=4)
    nbformat.validate(notebook)
    if any(cell.get('outputs') for cell in notebook.cells if cell.cell_type == 'code'):
        raise ValueError('Execution input must be a source notebook with cleared outputs')
    inputs = {'notebook_sha256': digest(args.notebook), 'tool_sha256': digest(__file__),
              'release_sha256': digest(args.release/'release.json'),
              'manifest_sha256': digest(args.release/'manifest.public.json'),
              'python': sys.version, 'python_executable': sys.executable,
              'python_prefix': sys.prefix, 'nbformat': nbformat.__version__}
    write_once(args.output/'inputs.json', inputs)
    os.environ['ARTIFACTBENCH_RELEASE'] = str(args.release.resolve())
    os.environ['ARTIFACTBENCH_NOTEBOOK_PYTHON'] = sys.executable
    os.environ['ARTIFACTBENCH_NOTEBOOK_PREFIX'] = sys.prefix
    client = NotebookClient(notebook, timeout=120, kernel_name='python3',
                            resources={'metadata': {'path': str(root)}}, allow_errors=False)
    try:
        client.execute()
    finally:
        # 실제 커널이 기록한 출력만 저장한다. 실패 시 summary를 만들지 않는다.
        nbformat.write(notebook, args.output/'executed.ipynb')
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == 'code']
    if any(cell.execution_count is None or any(o.output_type == 'error' for o in cell.outputs) for cell in code):
        raise ValueError('Notebook did not execute all code cells successfully')
    summary = {'status': 'metadata notebook execution passed', 'executed_code_cells': len(code),
               'output_sha256': digest(args.output/'executed.ipynb'),
               'scope': 'real fresh-kernel metadata verification, not model inference or full result reproduction'}
    write_once(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

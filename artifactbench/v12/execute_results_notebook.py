"""공개 예측 노트북을 별도 커널로 실제 실행하고 원래 통계와의 일치를 확인한다."""
import argparse
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
import numpy as np

from .common import digest, write_once


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--notebook', required=True, type=Path)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--predictions', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--allow-smoke', action='store_true')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.allow_smoke and args.output.resolve().is_relative_to(root/'paper'):
        raise ValueError('Smoke notebook outputs cannot enter the manuscript tree')
    args.output.mkdir(parents=True, exist_ok=False)
    notebook = nbformat.read(args.notebook, as_version=4)
    nbformat.validate(notebook)
    if any(c.get('outputs') or c.get('execution_count') is not None for c in notebook.cells if c.cell_type == 'code'):
        raise ValueError('A cleared source notebook is required')
    inputs = {'notebook_sha256': digest(args.notebook), 'tool_sha256': digest(__file__),
              'release_sha256': digest(args.release/'release.json'),
              'predictions_summary_sha256': digest(args.predictions/'summary.json'),
              'python': sys.version, 'python_executable': sys.executable, 'python_prefix': sys.prefix,
              'numpy': np.__version__, 'nbformat': nbformat.__version__, 'allow_smoke': args.allow_smoke}
    write_once(args.output/'inputs.json', inputs)
    os.environ.update(ARTIFACTBENCH_RELEASE=str(args.release.resolve()),
        ARTIFACTBENCH_PUBLIC_PREDICTIONS=str(args.predictions.resolve()),
        ARTIFACTBENCH_RESULTS_OUTPUT=str((args.output/'statistics').resolve()),
        ARTIFACTBENCH_NOTEBOOK_PYTHON=sys.executable, ARTIFACTBENCH_NOTEBOOK_PREFIX=sys.prefix,
        ARTIFACTBENCH_ALLOW_SMOKE='1' if args.allow_smoke else '0')
    client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                            resources={'metadata': {'path': str(root)}}, allow_errors=False)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, args.output/'executed.ipynb')
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    if len(code) != 6 or any(c.execution_count is None or any(o.output_type == 'error' for o in c.outputs) for c in code):
        raise ValueError('Not every results notebook cell executed successfully')
    proof = json.loads((args.output/'statistics/reproduction.json').read_text())
    expected = 'SMOKE REPRODUCTION WIRING ONLY' if args.allow_smoke else 'public-score statistical reproduction passed'
    if proof['status'] != expected or proof['compared_statistical_files'] != 9:
        raise ValueError('Notebook did not prove all original statistical files')
    summary = {'status': 'SMOKE NOTEBOOK WIRING ONLY' if args.allow_smoke else 'public-score results notebook passed',
               'executed_code_cells': len(code), 'output_sha256': digest(args.output/'executed.ipynb'),
               'reproduction_sha256': digest(args.output/'statistics/reproduction.json'),
               'scope': 'Actual kernel and saved-score statistics; not fresh model inference'}
    write_once(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

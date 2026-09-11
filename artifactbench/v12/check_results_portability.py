"""공개 점수와 최소 소스만 새 디렉터리로 옮겨 결과 노트북을 실제 재현한다."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .common import digest, write_once
from .public_results import load_public_results

SOURCES = (
    'artifactbench/__init__.py', 'artifactbench/v12/__init__.py',
    'artifactbench/v12/common.py', 'artifactbench/v12/report.py',
    'artifactbench/v12/statistics.py', 'artifactbench/v12/public_results.py',
    'artifactbench/v12/verify_release.py', 'artifactbench/v12/verify_bundle.py',
    'artifactbench/v12/execute_results_notebook.py',
    'notebooks/v1.2_results_reproduction.ipynb',
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--predictions', required=True, type=Path)
    ap.add_argument('--notebook-python', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--allow-smoke', action='store_true')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if args.allow_smoke and output.is_relative_to(root/'paper'):
        raise ValueError('Smoke artifacts cannot enter the manuscript tree')
    _, _, _, predictions = load_public_results(args.predictions, args.release, args.allow_smoke)
    output.mkdir(parents=True, exist_ok=False)
    copied = Path(tempfile.mkdtemp(prefix='artifactbench-results-'))
    inputs = {name: root/name for name in SOURCES}
    inputs.update({'release/'+name: args.release/name for name in
                   ('release.json', 'manifest.public.json', 'native_pairs.public.json', 'exposure.public.json')})
    inputs.update({'predictions/'+name: args.predictions/name for name in (*predictions['file_sha256'], 'summary.json')})
    hashes = {name: digest(path) for name, path in inputs.items()}
    for name, original in inputs.items():
        target = copied/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        if digest(target) != hashes[name]:
            raise ValueError('Source changed during copy')
    command = [str(args.notebook_python.absolute()), '-m', 'artifactbench.v12.execute_results_notebook',
        '--notebook', 'notebooks/v1.2_results_reproduction.ipynb', '--release', 'release',
        '--predictions', 'predictions', '--output', str(output/'notebook')]
    if args.allow_smoke:
        command.append('--allow-smoke')
    write_once(output/'inputs.json', {'copied_root': str(copied), 'file_sha256': hashes, 'command': command,
        'tool_sha256': digest(__file__), 'scope': 'Copied public data and source; existing isolated notebook dependency environment'})
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    result = subprocess.run(command, cwd=copied, env=env, capture_output=True, text=True, timeout=900)
    write_once(output/'process.json', {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    if result.returncode:
        raise RuntimeError('Copied results notebook failed; see preserved process output')
    for name, original in inputs.items():
        if digest(copied/name) != hashes[name] or digest(original) != hashes[name]:
            raise ValueError('Public data or source changed during execution')
    notebook = json.loads((output/'notebook/summary.json').read_text())
    expected = 'SMOKE NOTEBOOK WIRING ONLY' if args.allow_smoke else 'public-score results notebook passed'
    if notebook['status'] != expected or notebook['executed_code_cells'] != 6:
        raise ValueError('Copied notebook completion mismatch')
    summary = {'status': 'SMOKE PORTABILITY WIRING ONLY' if args.allow_smoke else 'public-score notebook portability passed',
               'copied_files': len(inputs), 'notebook': notebook,
               'scope': 'No original source/data imports, audio, weights, or inference; saved-score reconstruction only'}
    write_once(output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

"""새 ZIP 추출본의 검증기와 결과 노트북을 실제 실행한다. 로컬 후보 전용이다."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .common import digest, write_once
from .package_results import extract_results_bundle, verify_results_bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', required=True, type=Path)
    ap.add_argument('--notebook-python', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--allow-smoke', action='store_true')
    args = ap.parse_args()
    output = args.output.resolve()
    if args.allow_smoke and output.is_relative_to(Path(__file__).resolve().parents[2]/'paper'):
        raise ValueError('Smoke artifacts cannot enter the manuscript tree')
    checksum = digest(args.archive)
    output.mkdir(parents=True, exist_ok=False)
    extracted = Path(tempfile.mkdtemp(prefix='artifactbench-results-zip-'))
    bundle = extract_results_bundle(args.archive, extracted, args.allow_smoke)
    inventory_hash = digest(bundle/'BUNDLE_MANIFEST.json')
    python = str(args.notebook_python.absolute())
    commands = {
        'verification': [python, '-B', '-m', 'artifactbench.v12.package_results', 'verify', '--bundle', '.'],
        'notebook': [python, '-B', '-m', 'artifactbench.v12.execute_results_notebook',
            '--notebook', 'notebooks/v1.2_results_reproduction.ipynb', '--release', 'release',
            '--predictions', 'predictions', '--output', str(output/'notebook')],
    }
    if args.allow_smoke:
        for command in commands.values():
            command.append('--allow-smoke')
    write_once(output/'inputs.json', {'archive_sha256': checksum,
        'inventory_sha256': inventory_hash, 'extracted_root': str(bundle),
        'commands': commands, 'tool_sha256': digest(__file__),
        'scope': 'Fresh ZIP extraction/import roots; existing isolated dependencies, not a fresh install'})
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    for name, command in commands.items():
        result = subprocess.run(command, cwd=bundle, env=env, capture_output=True, text=True, timeout=900)
        write_once(output/(name+'_process.json'), {'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr})
        if result.returncode:
            raise RuntimeError(name+' failed; see preserved process output')
    notebook = json.loads((output/'notebook/summary.json').read_text())
    proof = json.loads((output/'notebook/statistics/reproduction.json').read_text())
    expected = 'SMOKE NOTEBOOK WIRING ONLY' if args.allow_smoke else 'public-score results notebook passed'
    if notebook['status'] != expected or notebook['executed_code_cells'] != 6 or proof['compared_statistical_files'] != 9:
        raise ValueError('Extracted notebook did not prove all nine statistical files')
    if digest(args.archive) != checksum or digest(bundle/'BUNDLE_MANIFEST.json') != inventory_hash:
        raise ValueError('Results archive or inventory changed during execution')
    verification = verify_results_bundle(bundle, args.allow_smoke)
    summary = {'status': 'SMOKE ZIP PORTABILITY WIRING ONLY' if args.allow_smoke else 'public-score ZIP portability passed',
        'archive_sha256': checksum, 'verification': verification, 'notebook': notebook,
        'scope': 'No original source/data imports; saved-score reproduction only, not fresh inference'}
    write_once(output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

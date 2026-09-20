"""Torch/ONNX 패키지 없이 후보 번들의 byte identity와 공개 비교표를 검사한다."""
import argparse
import json
import math
from pathlib import Path
import re

from .common import digest


FILES = {'README.md', 'RUNTIME_LICENSE.txt', 'requirements.txt', 'artifactnet_raw.onnx',
         'artifactnet_raw.json', 'PARITY_REPORT.json', 'artifactbench/__init__.py',
         'artifactbench/v12/__init__.py', 'artifactbench/v12/common.py',
         'artifactbench/v12/artifactnet_onnx.py', 'artifactbench/v12/verify_onnx_bundle.py'}
FROZEN_MANIFEST = 'feab7c4c3d037919fd784dc40e46199470088abde532c5181e34573f20dceefd'


def score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid probability in ONNX parity report')
    return float(value)


def verify_parity(report):
    if (report['schema'] != 'artifactnet-public-parity/1'
            or report['public_manifest_sha256'] != FROZEN_MANIFEST
            or report['scope'] not in ('source-smoke', 'full-rc2')):
        raise ValueError('Unexpected parity scope or release')
    rows = report['records']
    expected = 34 if report['scope'] == 'source-smoke' else 2579
    if len(rows) != expected or len({r['id'] for r in rows}) != expected:
        raise ValueError('Parity membership does not match declared scope')
    errors, flips, failed = [], 0, 0
    for row in rows:
        if (not re.fullmatch('[0-9a-f]{24}', row['id'])
                or not re.fullmatch('[0-9a-f]{64}', row['audio_sha256'])
                or not isinstance(row['source'], str) or not row['source']
                or type(row['chunks']) is not int or not 1 <= row['chunks'] <= 15):
            raise ValueError('Invalid parity recording identity or chunk count')
        got = score(row['onnx_p_ai'])
        if row['reference_outcome'] != 'scored':
            if (row['reference_outcome'] != 'model_execution_error' or 'reference_p_ai' in row
                    or row['id'] != 'cfae4e3c1d5317fe58fe6422' or row['reference_error_type'] != 'RuntimeError'):
                raise ValueError('Invalid failed reference row')
            failed += 1
            continue
        reference = score(row['reference_p_ai'])
        errors.append(abs(got-reference))
        flips += (got >= .5) != (reference >= .5)
    if (not errors or failed != 1 or max(errors) > 1e-3 or flips
            or report['maximum_absolute_error'] != max(errors)
            or report['raw_05_decision_flips'] != flips or report['original_failed_rows_retained'] != failed):
        raise ValueError('ONNX parity evidence fails the declared rule')
    return {'scope': report['scope'], 'attempted': len(rows), 'comparable': len(errors),
            'maximum_absolute_error': max(errors), 'decision_flips': int(flips)}


def verify(directory):
    directory = Path(directory).resolve()
    manifest_path = directory/'BUNDLE_MANIFEST.json'
    if manifest_path.is_symlink():
        raise ValueError('Symbolic bundle manifest')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['schema'] != 'artifactnet-onnx-candidate-bundle/1'
            or manifest['publication_status'] != 'local candidate; not uploaded'
            or set(manifest['file_sha256']) != FILES):
        raise ValueError('Invalid candidate bundle inventory or status')
    actual = set()
    for path in directory.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symbolic link in candidate bundle')
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != FILES | {'BUNDLE_MANIFEST.json'}:
        raise ValueError('Unlisted or missing candidate payload')
    for name, checksum in manifest['file_sha256'].items():
        if digest(directory/name) != checksum:
            raise ValueError('Candidate payload hash mismatch: '+name)
    metadata = json.loads((directory/'artifactnet_raw.json').read_text())
    parity = json.loads((directory/'PARITY_REPORT.json').read_text())
    if (metadata['schema'] != 'artifactnet-raw-onnx/1'
            or metadata['onnx_sha256'] != manifest['file_sha256']['artifactnet_raw.onnx']
            or parity['onnx_sha256'] != metadata['onnx_sha256']):
        raise ValueError('Model/metadata/parity identity mismatch')
    result = verify_parity(parity)
    if manifest['parity_scope'] != result['scope']:
        raise ValueError('Manifest/report scope mismatch')
    return {'status': 'candidate bytes and paired-score arithmetic verified', 'files': len(FILES),
            'parity': result, 'scope': 'No detector execution, rights grant, or publication approval'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', required=True, type=Path)
    print(json.dumps(verify(ap.parse_args().bundle), indent=2))


if __name__ == '__main__':
    main()

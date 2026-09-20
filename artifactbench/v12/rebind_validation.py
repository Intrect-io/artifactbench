"""새 manifest에 실제로 포함된 동일 바이트 파일만 기존 완전 디코딩 근거로 재검증한다."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .common import digest, write_once


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--previous-release', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    provenance = args.previous_release/'provenance.local.json'
    previous = json.loads(provenance.read_text())
    rows, source_hashes, source_directories = {}, {}, set()
    for filename, sha in previous['validation_hashes'].items():
        path = Path(filename)
        if digest(path) != sha:
            raise ValueError('Previous validation record changed')
        row = json.loads(path.read_text())
        if row['outcome'] != 'valid':
            raise ValueError('Only successful full-decode records can be rebound')
        if row['path'] in rows and rows[row['path']] != row:
            raise ValueError('Conflicting validation records for same input path')
        rows[row['path']] = row
        source_hashes[row['path']] = {'validation_path': str(path), 'sha256': sha}
        source_directories.add(path.parent.parent)
    current_tool = digest(Path(__file__).with_name('validate_audio.py'))
    ffmpeg = subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
    fpcalc = subprocess.check_output(['fpcalc', '-version'], text=True).strip()
    for directory in source_directories:
        inputs = json.loads((directory/'inputs.json').read_text())
        if inputs['tool_sha256'] != current_tool or inputs['ffmpeg'] != ffmpeg or inputs['fpcalc'] != fpcalc:
            raise ValueError('Validation implementation or decoder version changed: fresh decode required')
    selected = json.loads(args.manifest.read_text())['bench']
    evidence = {}
    for entry in selected:
        row = rows[entry['path']]
        if digest(entry['path']) != row['sha256']:
            raise ValueError('Audio bytes differ from the completely decoded prior input')
        key = hashlib.sha256(entry['path'].encode()).hexdigest()
        write_once(args.output/'records'/(key+'.json'), row)
        evidence[key] = source_hashes[entry['path']]
    write_once(args.output/'cache_rebinding.json', {'new_manifest_sha256': digest(args.manifest),
        'previous_provenance_sha256': digest(provenance), 'tool_sha256': digest(__file__),
        'validator_sha256': current_tool, 'records': evidence,
        'scope': 'current audio byte hashes rechecked; same full-decode validation reused, no fake re-execution'})
    print(json.dumps({'rebound_records': len(selected), 'next': 'run validate_audio to check new membership and produce its own completion summary'}))


if __name__ == '__main__':
    main()

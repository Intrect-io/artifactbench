"""공개 메타데이터에 사용자가 제공한 로컬 오디오 경로를 바이트 검증 후 연결한다."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

from .common import digest, write_once
from .package_metadata import PUBLIC_FILES
from .verify_release import verify


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate key in local path map: '+key)
        result[key] = value
    return result


def read_mapping(path, primary_ids, native_ids):
    mapping = json.loads(path.read_text(), object_pairs_hook=unique_object)
    if (not isinstance(mapping, dict) or set(mapping) != {'schema', 'primary', 'native'}
            or mapping['schema'] != 'artifactbench-local-paths/1'
            or not isinstance(mapping['primary'], dict) or not isinstance(mapping['native'], dict)):
        raise ValueError('Unsupported local path map schema')
    for group, allowed in (('primary', primary_ids), ('native', native_ids)):
        unknown = set(mapping[group])-set(allowed)
        if unknown:
            raise ValueError('Unknown '+group+' path-map IDs: '+', '.join(sorted(unknown)))
    return mapping


def inspect_mapping(value, base, checksum, expected_bytes=None):
    if value is None:
        return {'outcome': 'not_provided'}
    if not isinstance(value, str) or not value.strip() or '\0' in value or '://' in value:
        return {'outcome': 'invalid_local_path'}
    result = {}
    try:
        supplied = Path(value).expanduser()
        path = (base/supplied).resolve()
        result['path'] = str(path)
        if not path.is_file():
            return dict(result, outcome='not_a_file')
        before = path.stat()
        result['bytes'] = before.st_size
        if expected_bytes is not None and before.st_size != expected_bytes:
            return dict(result, outcome='size_mismatch')
        actual = digest(path)
        after = path.stat()
        signature = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if signature(before) != signature(after):
            return dict(result, outcome='changed_during_hash')
        result['sha256'] = actual
        return dict(result, outcome='bound' if actual == checksum else 'hash_mismatch')
    except (OSError, RuntimeError) as exc:
        return dict(result, outcome='file_access_error', error_type=type(exc).__name__, error=str(exc))


def bind_audio(release, path_map, output):
    release, path_map, output = (Path(p).resolve() for p in (release, path_map, output))
    if output.is_relative_to(release):
        raise ValueError('Binding output must be separate from the frozen release')
    hashes = {name: digest(release/name) for name in PUBLIC_FILES}
    map_hash = digest(path_map)
    verification = verify(release)
    public = json.loads((release/'manifest.public.json').read_text())
    entries = public['bench']
    pairs = json.loads((release/'native_pairs.public.json').read_text())
    primary = {r['id']: r for r in entries}
    native = {r['primary_id']+'-'+r['variant_id']: r for r in pairs}
    if len(native) != len(pairs):
        raise ValueError('Duplicate native pair identity')
    mapping = read_mapping(path_map, primary, native)
    output.mkdir(parents=True, exist_ok=False)
    write_once(output/'binding_inputs.local.json', {'public_sha256': hashes, 'path_map_sha256': map_hash,
        'path_map': str(path_map), 'python': sys.version, 'tool_sha256': digest(__file__),
        'scope': 'Local byte-identity binding only; no download, decode, inference or rights grant'})
    audit, local, local_pairs, bound = [], [], [], {}
    for group, rows in (('primary', primary), ('native', native)):
        for index, (identifier, entry) in enumerate(rows.items(), 1):
            result = inspect_mapping(mapping[group].get(identifier), path_map.parent, entry['sha256'], entry.get('bytes'))
            audit.append(dict(result, kind=group, id=identifier, expected_sha256=entry['sha256']))
            if result['outcome'] == 'bound':
                if group == 'primary':
                    local.append(dict(entry, path=result['path']))
                    bound[identifier] = result['path']
                else:
                    # native 파일 자체가 맞더라도 primary가 없으면 완성된 pair가 아니다.
                    if entry['primary_id'] in bound:
                        local_pairs.append(dict(entry, path=result['path'], primary_path=bound[entry['primary_id']]))
            if index % 100 == 0 or index == len(rows):
                print(json.dumps({'kind': group, 'checked': index, 'expected': len(rows)}), flush=True)
    write_once(output/'binding_audit.local.json', audit)
    if digest(path_map) != map_hash or any(digest(release/name) != checksum for name, checksum in hashes.items()):
        raise ValueError('Public metadata or path map changed during binding')
    outcomes = dict(Counter(r['outcome'] for r in audit))
    complete = len(local) == len(entries) and len(local_pairs) == len(pairs)
    summary = {'status': 'local audio byte binding complete' if complete else 'local audio byte binding incomplete',
        'manifest_sha256': verification['manifest_sha256'], 'primary_expected': len(entries),
        'primary_bound': len(local), 'native_expected': len(pairs), 'native_pairs_bound': len(local_pairs),
        'outcomes': outcomes, 'audit_sha256': digest(output/'binding_audit.local.json'),
        'scope': 'Same bytes at declared local paths; no new decoding or model execution; output is private'}
    if not complete:
        write_once(output/'binding_summary.local.json', summary)
        raise ValueError('Incomplete local audio binding; inspect binding_audit.local.json. No runnable manifests written.')
    for name in PUBLIC_FILES:
        shutil.copyfile(release/name, output/name)
        if digest(output/name) != hashes[name]:
            raise ValueError('Public metadata copy mismatch')
    write_once(output/'manifest.local.json', {'schema': public['schema'], 'bench': local})
    write_once(output/'native_pairs.local.json', local_pairs)
    verify(output)
    summary['local_manifest_sha256'] = digest(output/'manifest.local.json')
    summary['local_native_pairs_sha256'] = digest(output/'native_pairs.local.json')
    write_once(output/'binding_summary.local.json', summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--path-map', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(bind_audio(args.release, args.path_map, args.output), indent=2))


if __name__ == '__main__':
    main()

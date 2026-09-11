"""rc1의 Udio 상위 디렉터리 결합 오류를 완전한 공개 페이지 URL로 교정한다."""
import argparse
from collections import Counter
import json
from pathlib import Path
from urllib.parse import urlsplit

from .common import digest, write_once
from .suno_metadata import page_objects


def url_key(value):
    if not isinstance(value, str):
        return None
    parts = urlsplit(value)
    if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password:
        return None
    # 번호 경로·trim 경로·query를 버리지 않는다. 도메인 대소문자만 정규화한다.
    return parts.scheme, parts.netloc.lower(), parts.path, parts.query


def udio_links(metadata):
    declared = {key: metadata.get(key) for key in ('song_path', 'video_path')}
    declared['original_song_path'] = (metadata.get('lineage') or {}).get('original_song_path')
    return {name: url_key(value) for name, value in declared.items() if url_key(value) is not None}


def choose_udio_variant(variants, metadata):
    links = udio_links(metadata)
    matched, rejected = [], []
    for variant in variants:
        fields = [name for name, key in links.items() if key == url_key(variant['source_url'])]
        if fields:
            matched.append((variant, fields))
        else:
            rejected.append(variant)
    if not matched:
        raise ValueError('No complete page-declared Udio URL in available variants')
    matched.sort(key=lambda item: (not any(k in item[1] for k in ('song_path', 'video_path')),
                                  item[0]['container'] != 'mp3', item[0]['variant_id']))
    return matched, rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', required=True, type=Path)
    ap.add_argument('--candidates', required=True, type=Path)
    ap.add_argument('--udio-metadata', required=True, type=Path)
    ap.add_argument('--suno-metadata', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    document = json.loads(args.selection.read_text())
    candidates = {r['track_id']: r for r in json.loads(args.candidates.read_text())}
    metadata_hashes, audit, entries = {}, [], []
    for original in document['bench']:
        entry = dict(original)
        if entry['partition'] != 'contemporary_native':
            entries.append(entry)
            continue
        directory = args.udio_metadata if entry['provider'] == 'udio' else args.suno_metadata
        metadata_path = directory/'metadata'/(entry['media_id']+'.json')
        metadata = json.loads(metadata_path.read_text())
        page_path = directory/metadata['page_file']
        if (digest(page_path) != metadata['page_sha256'] or
                metadata['page_sha256'] != entry['metadata_page_sha256']):
            raise ValueError('Pinned metadata page changed')
        metadata_hashes[str(metadata_path)] = digest(metadata_path)
        if entry['provider'] == 'suno':
            clip_urls = {url_key(obj[field]) for obj in page_objects(page_path.read_text())
                         if obj.get('id') == entry['media_id'] for field in ('audio_url', 'video_url') if obj.get(field)}
            if url_key(entry['source_url']) not in clip_urls:
                raise ValueError('Suno source URL not explicitly linked by its exact clip object')
            entry['source_identity_evidence'] = 'exact_clip_object_audio_or_video_url'
            audit.append({'track_id': entry['track_id'], 'provider': 'suno', 'changed': False,
                          'source_identity_evidence': entry['source_identity_evidence']})
        else:
            matched, rejected = choose_udio_variant(candidates[entry['track_id']]['variants'], metadata)
            primary, fields = matched[0]
            for field in ('path', 'container', 'source_url', 'source_page', 'variant_id', 'transport',
                          'original_manifest', 'original_split', 'split'):
                if field in primary:
                    entry[field] = primary[field]
            entry['additional_native_variants'] = [variant for variant, _ in matched[1:]]
            entry['source_identity_evidence'] = 'exact_page_'+','.join(fields)
            entry['source_identity_fields'] = fields
            audit.append({'track_id': entry['track_id'], 'provider': 'udio',
                'changed': entry['path'] != original['path'], 'old_source_url': original['source_url'],
                'new_source_url': entry['source_url'], 'matched_page_fields': fields,
                'retained_extra_views': len(matched)-1,
                'rejected_sibling_urls': [r['source_url'] for r in rejected]})
        entries.append(entry)
    if [(r['source'], r['track_id'], r['creator_group']) for r in entries] != [
            (r['source'], r['track_id'], r['creator_group']) for r in document['bench']]:
        raise ValueError('Correction changed the selected source/creator/recording cohort')
    inputs = [args.selection, args.candidates, Path(__file__), Path(__file__).with_name('suno_metadata.py'),
              Path('docs/UDIO_IDENTITY_CORRECTION_v1.2.md')]
    write_once(args.output/'inputs.json', {'input_sha256': {str(p): digest(p) for p in inputs},
        'metadata_sha256': metadata_hashes, 'read_set': 'saved public page objects and selected file URLs only; no detector scores'})
    write_once(args.output/'selection.local.json', {'bench': entries})
    write_once(args.output/'source_identity_audit.public.json', audit)
    summary = {'status': 'complete-URL identity correction prepared; rc2 validation/freeze pending',
        'entries': len(entries), 'checked_native': len(audit),
        'primary_files_changed': sum(r['changed'] for r in audit),
        'extra_page_linked_views': sum(r.get('retained_extra_views', 0) for r in audit),
        'rejected_sibling_urls': sum(len(r.get('rejected_sibling_urls', [])) for r in audit),
        'identity_evidence': dict(Counter(r['source_identity_evidence'] for r in entries if r['partition'] == 'contemporary_native')),
        'selection_sha256': digest(args.output/'selection.local.json')}
    write_once(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

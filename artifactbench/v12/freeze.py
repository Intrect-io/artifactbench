"""검증·중복 재검토를 통과한 평가 입력을 공개 메타데이터와 로컬 경로로 분리 동결한다."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit, urlunsplit

from .common import digest, manifest_rows, rank, write_once
from .duplicates import Groups, entry_key, compare_waveforms
from .inventory import EXPOSURE_INPUTS
from .review_duplicates import full_audio


def public_url(value):
    if not value:
        return None
    parsed = urlsplit(value)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Only credential-free public HTTPS source URLs can be released')
    if parsed.query:
        raise ValueError('URL query needs manual privacy/access review before release')
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', ''))


def validated(rows, directory):
    summary = json.loads((directory / 'summary.json').read_text())
    if summary['expected'] != len(rows) or summary['outcomes'] != {'valid': len(rows)}:
        raise ValueError('Cannot freeze incomplete or failed validation')
    records = {}
    for entry in rows:
        path = directory / 'records' / (hashlib.sha256(entry['path'].encode()).hexdigest() + '.json')
        row = json.loads(path.read_text())
        if row['outcome'] != 'valid' or row['path'] != entry['path']:
            raise ValueError('Validation identity mismatch')
        records[entry['path']] = row
    return records


def recording_groups(entries, audit, review):
    keys = [entry_key(row) for row in entries]
    groups = Groups(keys)
    for exact in audit['exact_groups']:
        for key in exact['entries'][1:]:
            groups.join(exact['entries'][0], key)
    for pair in review['pairs']:
        if pair['verdict'] == 'same_recording_or_excerpt':
            groups.join(*pair['entries'])
    labels = groups.labels(keys, 'recording:')
    representatives = {}
    for key in sorted(keys, key=rank):
        representatives.setdefault(labels[key], key)
    return labels, representatives


def local_exposure_audit(entries, inventory):
    root = Path('/home/unohee/dev/ArtifactNet')
    sys.path.insert(0, str(root))
    from src.data.dataset_guard import stem_key, same_seed_different_generator
    findings, input_hashes = [], {}
    for name in EXPOSURE_INPUTS:
        path = root / 'outputs' / name
        actual = digest(path)
        if inventory['input_hashes'][str(path)] != actual:
            raise ValueError(f'Exposure input changed after selection: {path}')
        input_hashes[name] = actual
        indexed = defaultdict(list)
        for split, row in manifest_rows(json.loads(path.read_text())):
            if row.get('path'):
                indexed[stem_key(row['path'])].append((split, row))
        for entry in entries:
            for split, row in indexed[stem_key(entry['path'])]:
                if not same_seed_different_generator(entry['path'], row['path'], entry['label']):
                    findings.append({'entry': entry_key(entry), 'manifest': name, 'split': split})
    return {'method': 'Path-stem candidate matches with explicit multigenerator seed exception; not an audio-level all-training audit',
            'input_hashes': input_hashes, 'candidate_collisions': findings,
            'guard_sha256': digest(root / 'src/data/dataset_guard.py')}


def public_entry(entry, validation, groups, audit, representative):
    key = entry_key(entry)
    return {
        'id': hashlib.sha256(key.encode()).hexdigest()[:24],
        'source': entry['source'], 'track_id': entry['track_id'], 'label': entry['label'],
        'partition': entry['partition'], 'provider': entry.get('provider'),
        'provider_recording_id': entry.get('song_id') if entry.get('provider') == 'udio' else
                                 (entry.get('media_id') if entry.get('provider') == 'suno' else None),
        'generator': entry.get('generator'), 'generator_version': entry.get('generator_version'),
        'model_size': entry.get('model_size'), 'label_scope': entry.get('label_scope'),
        'created_at': entry.get('created_at'), 'version_evidence': entry.get('version_evidence'),
        'source_identity_evidence': entry.get('source_identity_evidence'),
        'generation_task': entry.get('generation_task'), 'generation_type': entry.get('generation_type'),
        'audio_conditioning_type': entry.get('audio_conditioning_type'),
        'source_url': public_url(entry.get('source_url')),
        'evidence_url': public_url(entry.get('metadata_source_url') or entry.get('source_page')),
        'evidence_page_sha256': entry.get('metadata_page_sha256') or entry.get('page_sha256'),
        'evidence_retrieved_at': entry.get('metadata_retrieved_at') or entry.get('retrieved_at'),
        'recording_group': groups[key], 'recording_representative': representative[groups[key]] == key,
        'dependence_cluster': audit['groups'][key]['dependence_cluster'],
        'creator_group_hash': hashlib.sha256(entry['creator_group'].encode()).hexdigest()[:24],
        'creator_evidence': entry.get('creator_evidence', 'public user ID' if entry['partition'] == 'contemporary_native' else 'official demo selection group'),
        'sha256': validation['sha256'], 'bytes': validation['bytes'],
        'decoded_pcm_sha256_22050_mono_f32le': validation['decoded_pcm_sha256_22050_mono_f32le'],
        'duration_seconds': validation['decoded_duration_seconds'], 'sample_rate': validation['sample_rate'],
        'channels': validation['channels'], 'audio_codec': validation['audio_codec'],
        'container': validation['container'], 'sample_format': validation['sample_format'],
        'rights': 'metadata only; no audio redistribution authorization inferred',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for argument in ('selection', 'validation', 'duplicates', 'review', 'inventory', 'variants', 'variant-validation', 'output'):
        ap.add_argument('--'+argument, required=True, type=Path)
    args = ap.parse_args()
    entries = json.loads(args.selection.read_text())['bench']
    variations = json.loads(args.variants.read_text())['bench']
    for directory, manifest in ((args.validation, args.selection), (args.variant_validation, args.variants)):
        if json.loads((directory/'inputs.json').read_text())['manifest_sha256'] != digest(manifest):
            raise ValueError('Validation belongs to a different selection')
    if json.loads((args.duplicates/'inputs.json').read_text())['manifest_sha256'] != digest(args.selection):
        raise ValueError('Duplicate audit belongs to a different selection')
    review_inputs = json.loads((args.review/'inputs.json').read_text())
    if (review_inputs['selection_sha256'] != digest(args.selection) or
            review_inputs['audit_sha256'] != digest(args.duplicates/'report.json')):
        raise ValueError('Duplicate review belongs to different inputs')
    if any(not e.get('source_identity_evidence') for e in entries if e['partition'] == 'contemporary_native'):
        raise ValueError('Every native entry requires exact public-page source-URL evidence')
    validation = validated(entries, args.validation)
    variation_validation = validated(variations, args.variant_validation)
    audit = json.loads((args.duplicates / 'report.json').read_text())
    review = json.loads((args.review / 'report.json').read_text())
    if audit['label_conflicts'] or audit['known_lineage_exposure_links']:
        raise ValueError('Unresolved label/lineage exposure blocks freeze')
    if review['pairs_reviewed'] != audit['perceptual_candidates']:
        raise ValueError('All fingerprint candidates need explicit review')
    inventory = json.loads(args.inventory.read_text())
    exposure = local_exposure_audit(entries, inventory)
    if exposure['candidate_collisions']:
        raise ValueError('Unresolved exposure ID candidates require investigation')
    groups, representatives = recording_groups(entries, audit, review)
    public = [public_entry(e, validation[e['path']], groups, audit, representatives) for e in entries]
    if len({r['id'] for r in public}) != len(public):
        raise ValueError('Duplicate release IDs')
    local = [dict(e, **{k: p[k] for k in ('id', 'recording_group', 'recording_representative', 'dependence_cluster', 'sha256')})
             for e, p in zip(entries, public)]
    primary = {r['path']: r for r in local}
    paired_local, paired_public = [], []
    for variant in variations:
        original = primary[variant['primary_path']]
        info = variation_validation[variant['path']]
        row = {'primary_id': original['id'], 'primary_sha256': original['sha256'],
               'variant_id': variant['variant_id'], 'sha256': info['sha256'], 'audio_codec': info['audio_codec'],
               'duration_seconds': info['decoded_duration_seconds'], 'source_url': public_url(variant.get('source_url'))}
        for path, checksum in ((original['path'], original['sha256']), (variant['path'], info['sha256'])):
            if digest(path) != checksum:
                raise ValueError('Native pair audio changed before identity verification')
        x, y = full_audio(original['path']), full_audio(variant['path'])
        comparison = compare_waveforms(x, y, 0.)
        if (not comparison['confirmed_shared_excerpt'] or comparison['shorter_excerpt_coverage'] < .98 or
                min(len(x), len(y))/max(len(x), len(y)) < .98):
            raise ValueError('Native pair is not a verified same-recording transport')
        row['waveform_identity'] = {k: comparison[k] for k in ('waveform_correlation', 'window_correlation_median',
                                   'offset_samples_8000', 'shorter_excerpt_coverage')}
        paired_public.append(row)
        paired_local.append(dict(row, path=variant['path'], primary_path=original['path']))
    public_doc = {'schema': 'artifactbench-1.2-rc2', 'bench': public}
    write_once(args.output / 'manifest.public.json', public_doc)
    write_once(args.output / 'manifest.local.json', {'schema': public_doc['schema'], 'bench': local})
    write_once(args.output / 'native_pairs.public.json', paired_public)
    write_once(args.output / 'native_pairs.local.json', paired_local)
    write_once(args.output / 'exposure.public.json', {
        'artifactnet': {'local_manifest_check': exposure, 'cohort_status': 'retrospective researcher-observed/mined corpus; not a prospective blind test'},
        'external_checkpoints': {name: {'declared_training_corpus': corpus, 'recording_membership': 'unknown; no per-recording training manifests available'}
            for name, corpus in {'spectttra': 'SONICS', 'spectttra_beta5s': 'SONICS', 'clam': 'MoM',
                'deezer_ismir': 'SONICS, FMA medium, proprietary data (lofcz model card)',
                'fst': 'see upstream checkpoint documentation; membership unavailable',
                'ast_60s': 'model-card scope; membership unavailable', 'deepfense': 'FakeMusicCaps'}.items()},
        'warning': 'Neither disjoint filenames nor absent external training manifests establish universal unseen status.',
    })
    inputs = [args.selection, args.inventory, args.variants,
              args.validation/'summary.json', args.variant_validation/'summary.json',
              args.duplicates/'report.json', args.review/'report.json',
              Path('docs/PROTOCOL_v1.2.md'), Path('docs/DUPLICATE_AUDIT_v1.2.md'),
              Path('docs/UDIO_IDENTITY_CORRECTION_v1.2.md'),
              Path(__file__).with_name('review_duplicates.py'), Path(__file__).with_name('duplicates.py'), Path(__file__)]
    inputs.extend(p for directory in (args.validation, args.variant_validation)
                  if (p := directory/'cache_rebinding.json').exists())
    correction_inputs = args.selection.parent/'inputs.json'
    if correction_inputs.exists():
        inputs.append(correction_inputs)
    write_once(args.output / 'provenance.local.json', {'input_hashes': {str(p): digest(p) for p in inputs},
        'validation_hashes': {str(p): digest(p) for directory in (args.validation, args.variant_validation)
                              for p in sorted((directory/'records').glob('*.json'))}})
    summary = {
        'version': '1.2-rc2', 'status': 'frozen evaluation data; baseline scores and publication still pending',
        'evaluation_entries': len(entries), 'identified_recording_groups': len(set(groups.values())),
        'dependence_clusters': len({r['dependence_cluster'] for r in public}),
        'partitions': dict(Counter(r['partition'] for r in entries)),
        'source_counts': dict(Counter(r['source'] for r in entries)),
        'labels': dict(Counter(r['label'] for r in entries)), 'native_transport_pairs': len(paired_public),
        'duplicate_review': review['verdicts'],
        'limits': ['First-120s fingerprint candidate search, all candidates reviewed against full waveforms.',
                   'Recording-group counts mean identified groups, not proof of exhaustive unique recordings.',
                   'Shared edits are distinct recordings but not independent samples.',
                   'Legacy rows retained; representative-only view and dependence clustering required.',
                   'Audio files and raw song-page snapshots are not redistribution artifacts.'],
        'public_file_sha256': {name: digest(args.output/name) for name in
            ('manifest.public.json', 'native_pairs.public.json', 'exposure.public.json')},
    }
    write_once(args.output/'release.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

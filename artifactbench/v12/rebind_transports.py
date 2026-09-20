"""변경되지 않은 controlled PCM과 검증된 native pair를 새 release에 명시적으로 결합한다."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .common import digest, write_once
from .prepare_transports import sample_controlled
from .run_transports import load_cached


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--previous-prepared', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    previous = args.previous_prepared
    old_summary = json.loads((previous/'summary.json').read_text())
    if digest(previous/'manifest.local.json') != old_summary['manifest_sha256']:
        raise ValueError('Old prepared manifest changed')
    old = json.loads((previous/'manifest.local.json').read_text())
    old_inputs = json.loads((previous/'materialization_inputs.json').read_text())
    if old_inputs['audio_tool_sha256'] != digest(Path(__file__).with_name('audio.py')):
        raise ValueError('Audio implementation changed; generate new waveforms')
    release = json.loads((args.release/'release.json').read_text())
    if digest(args.release/'manifest.public.json') != release['public_file_sha256']['manifest.public.json']:
        raise ValueError('New public manifest changed')
    entries = json.loads((args.release/'manifest.public.json').read_text())['bench']
    public = {r['id']: r for r in entries}
    selected, eligible = sample_controlled(entries)
    old_controlled = {r['entry']['id']: r for r in old['controlled']}
    if set(old_controlled) != {r['id'] for r in selected}:
        raise ValueError('New score-blind sample is different; generate new waveforms')
    controlled = []
    for entry in selected:
        old_record = old_controlled[entry['id']]
        for field in ('sha256', 'duration_seconds', 'source', 'track_id', 'label', 'partition'):
            if old_record['entry'][field] != entry[field]:
                raise ValueError('Controlled recording identity changed')
        for cache in [old_record['base'], *old_record['variants'].values()]:
            load_cached(cache)
        record = dict(old_record, entry=entry)
        write_once(args.output/'controlled_records'/(entry['id']+'.json'), record)
        controlled.append(record)
    old_native = {r['id']: r for r in old['native']}
    native = []
    for pair in json.loads((args.release/'native_pairs.local.json').read_text()):
        identifier = pair['primary_id']+'-'+pair['variant_id']
        old_record = old_native[identifier]
        if any(old_record['native_pair'][k] != pair[k] for k in ('primary_sha256', 'sha256', 'primary_path', 'path')):
            raise ValueError('Native audio changed; generate new waveforms')
        if 'waveform_identity' not in pair:
            raise ValueError('New native pair lacks release-time full-waveform verification')
        for cache in (old_record['primary'], old_record['variant']):
            load_cached(cache)
        record = dict(old_record, entry=public[pair['primary_id']], native_pair=pair)
        write_once(args.output/'native_records'/(identifier+'.json'), record)
        native.append(record)
    write_once(args.output/'selection.json', {
        'public_manifest_sha256': digest(args.release/'manifest.public.json'),
        'specification_sha256': digest(Path('docs/TRANSPORT_v1.2.md')), 'tool_sha256': digest(__file__),
        'seed': 260905, 'scope': 'same lossless-container legacy cohort; no detector scores used',
        'eligible_ids': sorted(r['id'] for r in eligible), 'selected': selected,
        'sources': dict(Counter(r['source'] for r in selected)), 'labels': dict(Counter(r['label'] for r in selected)),
        'native_pairs': len(native)})
    write_once(args.output/'materialization_inputs.json', dict(old_inputs,
        selection_sha256=digest(args.output/'selection.json'),
        local_manifest_sha256=digest(args.release/'manifest.local.json'),
        native_pairs_sha256=digest(args.release/'native_pairs.local.json'),
        rebinding={'tool_sha256': digest(__file__), 'previous_manifest_sha256': digest(previous/'manifest.local.json'),
                  'previous_materialization_sha256': digest(previous/'materialization_inputs.json'),
                  'scope': 'verified cached waveforms reused; no claim of a new encoder execution'}))
    write_once(args.output/'manifest.local.json', {'controlled': controlled, 'native': native})
    summary = {'status': 'prepared inputs rebound to corrected release; no detector inference',
        'controlled_recordings': len(controlled), 'controlled_variants': sum(len(r['variants']) for r in controlled),
        'native_pairs': len(native), 'native_pair_kinds': dict(Counter(r['pair_kind'] for r in native)),
        'manifest_sha256': digest(args.output/'manifest.local.json')}
    write_once(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

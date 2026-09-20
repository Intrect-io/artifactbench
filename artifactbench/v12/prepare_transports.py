"""점수 없이 코덱 실험 표본을 고정하고 동일 float 입력의 변형을 로컬에 보존한다."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch

from .audio import TRANSFORMS, load_audio_float, transform_audio
from .common import digest, rank, write_once


def sample_controlled(entries, limit=50):
    if limit < 1:
        raise ValueError('Controlled sample size must be positive')
    eligible = [r for r in entries if r['partition'] == 'legacy' and
                (r['audio_codec'].startswith('pcm_') or r['audio_codec'] == 'flac')]
    selected_groups, by_source = set(), defaultdict(list)
    for row in sorted(eligible, key=lambda r: rank(r['id'])):
        if row['recording_group'] in selected_groups:
            continue
        selected_groups.add(row['recording_group'])
        by_source[row['source']].append(row)
    order = sorted(by_source, key=lambda source: rank('codec-cell:'+source))
    selected = []
    offset = 0
    while len(selected) < min(limit, len(selected_groups)):
        for source in order:
            if offset < len(by_source[source]):
                selected.append(by_source[source][offset])
                if len(selected) == limit:
                    break
        offset += 1
    return selected, eligible


def save_array(path, audio):
    """중단된 파일을 완성 파일로 오인하지 않도록 임시 파일에서 원자적으로 게시한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_hash = hashlib.sha256(np.asarray(audio, dtype='<f4').tobytes()).hexdigest()
    if path.exists():
        existing = np.load(path, allow_pickle=False)
        if existing.dtype != np.float32 or not np.array_equal(existing, audio):
            raise ValueError(f'Existing cached waveform differs: {path}')
    else:
        with tempfile.TemporaryDirectory(prefix='artifactbench-array-', dir=path.parent) as temporary:
            stage = Path(temporary)/'wave.npy'
            with stage.open('xb') as stream:
                np.save(stream, np.asarray(audio, dtype=np.float32), allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            # link는 같은 디렉터리 내 완성 파일을 원자적으로 게시하며 덮어쓰지 않는다.
            os.link(stage, path)
    return {'path': str(path.resolve()), 'npy_sha256': digest(path), 'pcm_f32le_sha256': raw_hash,
            'frames': len(audio), 'sample_rate': 44100, 'dtype': 'float32'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--selection-only', action='store_true')
    args = ap.parse_args()
    public_path = args.release/'manifest.public.json'
    release = json.loads((args.release/'release.json').read_text())
    if digest(public_path) != release['public_file_sha256']['manifest.public.json']:
        raise ValueError('Frozen public metadata changed')
    entries = json.loads(public_path.read_text())['bench']
    local = {r['id']: r for r in json.loads((args.release/'manifest.local.json').read_text())['bench']}
    native = json.loads((args.release/'native_pairs.local.json').read_text())
    selected, eligible = sample_controlled(entries)
    specification = Path('docs/TRANSPORT_v1.2.md')
    write_once(args.output/'selection.json', {
        'public_manifest_sha256': digest(public_path), 'specification_sha256': digest(specification),
        'tool_sha256': digest(__file__), 'seed': 260905,
        'scope': 'lossless input container, earlier encoding history not established; incremental transforms',
        'eligible_ids': sorted(r['id'] for r in eligible), 'selected': selected,
        'sources': dict(Counter(r['source'] for r in selected)), 'labels': dict(Counter(r['label'] for r in selected)),
        'native_pairs': len(native),
    })
    print(json.dumps({'controlled_selected': len(selected), 'eligible': len(eligible),
                      'sources': dict(Counter(r['source'] for r in selected)), 'labels': dict(Counter(r['label'] for r in selected))}), flush=True)
    if args.selection_only:
        return
    torch.set_num_threads(2)
    write_once(args.output/'materialization_inputs.json', {
        'selection_sha256': digest(args.output/'selection.json'),
        'local_manifest_sha256': digest(args.release/'manifest.local.json'),
        'native_pairs_sha256': digest(args.release/'native_pairs.local.json'),
        'audio_tool_sha256': digest(Path(__file__).with_name('audio.py')),
        'torch': torch.__version__, 'numpy': np.__version__, 'soundfile': sf.__version__,
        'ffmpeg': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0], 'torch_threads': 2,
    })
    args.output.mkdir(parents=True, exist_ok=True)
    required_seconds = sum(r['duration_seconds'] for r in selected) * 9
    approximate_bytes = required_seconds * 44100 * 4
    if shutil.disk_usage(args.output).free < approximate_bytes * 1.2:
        raise RuntimeError('Insufficient free space for controlled float waveforms')
    controlled = []
    for number, entry in enumerate(selected):
        row_path = args.output/'controlled_records'/(entry['id']+'.json')
        source = local[entry['id']]['path']
        if digest(source) != entry['sha256']:
            raise ValueError('Controlled source changed after freeze')
        if row_path.exists():
            record = json.loads(row_path.read_text())
            for cache in [record['base'], *record['variants'].values()]:
                if digest(cache['path']) != cache['npy_sha256']:
                    raise ValueError('Cached controlled waveform changed')
        else:
            audio = load_audio_float(source)
            root = args.output/'arrays'/entry['id']
            record = {'entry': entry, 'base': save_array(root/'base.npy', audio), 'variants': {}}
            for transform in TRANSFORMS:
                waveform, evidence = transform_audio(audio, transform)
                if transform == 'float_identity' and not np.array_equal(waveform, audio):
                    raise ValueError('Float identity is not numerically exact')
                record['variants'][transform] = dict(save_array(root/(transform+'.npy'), waveform), transform_evidence=evidence)
            write_once(row_path, record)
        controlled.append(record)
        print(json.dumps({'controlled_prepared': number+1, 'expected': len(selected)}), flush=True)
    native_records = []
    public = {r['id']: r for r in entries}
    for pair in native:
        identifier = pair['primary_id']+'-'+pair['variant_id']
        row_path = args.output/'native_records'/(identifier+'.json')
        if row_path.exists():
            record = json.loads(row_path.read_text())
            for cache in (record['primary'], record['variant']):
                if digest(cache['path']) != cache['npy_sha256']:
                    raise ValueError('Cached native waveform changed')
        else:
            for path, expected in ((pair['primary_path'], pair['primary_sha256']), (pair['path'], pair['sha256'])):
                if digest(path) != expected:
                    raise ValueError('Native source changed after freeze')
            a, b = load_audio_float(pair['primary_path']), load_audio_float(pair['path'])
            root = args.output/'native_arrays'/identifier
            record = {'id': identifier, 'entry': public[pair['primary_id']], 'native_pair': pair,
                'pair_kind': 'same_codec_native' if public[pair['primary_id']]['audio_codec'] == pair['audio_codec'] else 'cross_codec_native',
                'primary': save_array(root/'primary.npy', a), 'variant': save_array(root/'variant.npy', b),
                'alignment_policy': 'native inputs unmodified except common loader; no forced timing/length correction'}
            write_once(row_path, record)
        native_records.append(record)
    write_once(args.output/'manifest.local.json', {'controlled': controlled, 'native': native_records})
    write_once(args.output/'summary.json', {
        'status': 'transport inputs materialized; no detector inference', 'controlled_recordings': len(controlled),
        'controlled_variants': sum(len(r['variants']) for r in controlled), 'native_pairs': len(native_records),
        'native_pair_kinds': dict(Counter(r['pair_kind'] for r in native_records)),
        'manifest_sha256': digest(args.output/'manifest.local.json'),
    })


if __name__ == '__main__':
    main()

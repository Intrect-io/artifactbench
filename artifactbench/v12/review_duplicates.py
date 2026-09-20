"""지문 후보 전부를 전체 길이 파형으로 재확인하고 원본/확장판을 구분한다."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np

from .common import digest, write_once
from .duplicates import compare_waveforms, FP_STEP_SECONDS


def full_audio(path):
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-map', '0:a:0',
               '-vn', '-ac', '1', '-ar', '8000', '-c:a', 'pcm_f32le', '-f', 'f32le', 'pipe:1']
    result = subprocess.run(command, capture_output=True, check=True, timeout=180)
    return np.frombuffer(result.stdout, dtype='<f4')


def shared_window_spans(correlations, threshold=0.985, window_seconds=5):
    spans, start = [], None
    for index, value in enumerate([*correlations, 0]):
        if value >= threshold and start is None:
            start = index
        if value < threshold and start is not None:
            if index - start >= 2:
                spans.append({'start_seconds': start*window_seconds, 'end_seconds': index*window_seconds})
            start = None
    return spans


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', required=True, type=Path)
    ap.add_argument('--audit', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    entries = json.loads(args.selection.read_text())['bench']
    pair_paths = sorted((args.audit / 'pairs').glob('*.json'))
    write_once(args.output / 'inputs.json', {
        'selection_sha256': digest(args.selection), 'audit_sha256': digest(args.audit / 'report.json'),
        'tool_sha256': digest(__file__), 'matcher_sha256': digest(Path(__file__).with_name('duplicates.py')),
        'pair_hashes': {p.name: digest(p) for p in pair_paths},
        'review_policy': 'All candidates, full waveform. Shared >=10s contiguous five-second windows support dependence, not duplicate identity. Similar durations require ratio>=0.98; no detector scores inspected.',
    })
    rows = []
    for pair_path in pair_paths:
        pair = json.loads(pair_path.read_text())
        a, b = (entries[pair[key]] for key in ('a', 'b'))
        x, y = full_audio(a['path']), full_audio(b['path'])
        result = compare_waveforms(x, y, pair['offset_fp_frames']*FP_STEP_SECONDS)
        spans = shared_window_spans(result.get('window_correlations', []))
        # first-120s에서 확인한 관계를 전체 길이 불일치로 삭제하지 않는다.
        prefix_spans = shared_window_spans(pair.get('window_correlations', []))
        shared = bool(spans or prefix_spans or pair['confirmed_shared_excerpt'])
        legacy_ids_match = (a['partition'] == b['partition'] == 'legacy' and
                            a['track_id'].split('_', 1)[-1] == b['track_id'].split('_', 1)[-1])
        if result['confirmed_shared_excerpt'] and (min(len(x), len(y))/max(len(x), len(y)) >= 0.98 or legacy_ids_match):
            verdict = 'same_recording_or_excerpt'
        elif shared:
            verdict = 'shared_audio_lineage_not_identical_recording'
        else:
            verdict = 'not_confirmed_as_shared_audio'
        row = {'entries': pair['entries'], 'a': pair['a'], 'b': pair['b'],
               'duration_seconds': [len(x)/8000, len(y)/8000], 'full_waveform': result,
               'full_shared_spans_relative_to_alignment': spans, 'first120_shared_spans': prefix_spans,
               'same_creator': a['creator_group'] == b['creator_group'],
               'legacy_original_id_matches': legacy_ids_match, 'verdict': verdict}
        write_once(args.output / pair_path.name, row)
        rows.append(row)
        print(json.dumps({'entries': pair['entries'], 'verdict': verdict}), flush=True)
    summary = {'pairs_reviewed': len(rows), 'verdicts': dict(Counter(r['verdict'] for r in rows)), 'pairs': rows}
    write_once(args.output / 'report.json', summary)


if __name__ == '__main__':
    main()

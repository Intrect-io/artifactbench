"""오디오 지문 후보를 실제 파형으로 확인하며, 중복과 의존성 집단을 구분한다."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse

import numpy as np
from scipy.signal import correlate

from .common import digest, write_once

BIT_COUNTS = np.array([n.bit_count() for n in range(256)], dtype=np.uint8)
FP_STEP_SECONDS = 1365 / 11025  # Chromaprint default 4096 / 3 sample hop


def entry_key(row):
    return row['source'] + ':' + row['track_id']


class Groups:
    def __init__(self, keys):
        self.parents = {key: key for key in keys}

    def root(self, key):
        self.parents.setdefault(key, key)
        if self.parents[key] != key:
            self.parents[key] = self.root(self.parents[key])
        return self.parents[key]

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parents[max(a, b)] = min(a, b)

    def labels(self, keys, prefix):
        members = defaultdict(list)
        for key in keys:
            members[self.root(key)].append(key)
        return {key: prefix + hashlib.sha256('\n'.join(sorted(group)).encode()).hexdigest()[:24]
                for group in members.values() for key in group}


def fingerprint_candidates(fingerprints):
    """위치가 일치하는 부분 지문으로 후보를 만들고 전체 겹침의 Hamming 거리를 확인한다."""
    index = defaultdict(list)
    for row_id, raw in enumerate(fingerprints):
        if raw is None:
            continue
        fp = np.asarray(raw, dtype=np.int64).astype(np.uint32)
        for shift in (0, 16):
            half = (fp >> shift) & 65535
            tokens = (half[:-3].astype(np.uint64) << 16) | half[3:]
            for position, token in enumerate(tokens):
                index[(shift, int(token))].append((row_id, position))
    votes, skipped = Counter(), 0
    for postings in index.values():
        if len(postings) > 200 or len({i for i, _ in postings}) > 20:
            skipped += 1
            continue
        for (a, pa), (b, pb) in itertools.combinations(postings, 2):
            if a != b:
                # postings는 row_id 순서로 삽입된다.
                votes[(a, b, (pa - pb) // 2)] += 1
    del index
    offsets = defaultdict(set)
    for (a, b, offset_bin), count in votes.items():
        if count >= 8:
            offsets[(a, b)].update(range(offset_bin * 2 - 2, offset_bin * 2 + 4))
    candidates = []
    for (a, b), possible in sorted(offsets.items()):
        fa = np.asarray(fingerprints[a], dtype=np.int64).astype(np.uint32)
        fb = np.asarray(fingerprints[b], dtype=np.int64).astype(np.uint32)
        best = None
        for offset in sorted(possible):
            ia, ib = max(offset, 0), max(-offset, 0)
            n = min(len(fa) - ia, len(fb) - ib)
            if n < max(80, 0.5 * min(len(fa), len(fb))):
                continue
            different = np.bitwise_xor(fa[ia:ia+n], fb[ib:ib+n])
            distance = float(BIT_COUNTS[different.view(np.uint8)].sum()) / (32 * n)
            proposal = (distance, -n, offset)
            if best is None or proposal < best:
                best = proposal
        if best is not None and best[0] <= 0.20:
            candidates.append({'a': a, 'b': b, 'fingerprint_bit_error': best[0],
                               'overlap_fp_frames': -best[1], 'offset_fp_frames': best[2]})
    return candidates, {'discarded_common_tokens': skipped, 'offset_voted_pairs': len(offsets)}


def normalized_correlation(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def compare_waveforms(a, b, offset_seconds, sample_rate=8000):
    """지문과 독립적으로 샘플 단위 정렬·상관을 검증한다. 무음은 일치로 보지 않는다."""
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Empty or non-finite verification waveform')
    center = round(offset_seconds * sample_rate)
    lags = np.arange(center - sample_rate // 2, center + sample_rate // 2 + 1)
    ia, ib = np.maximum(lags, 0), np.maximum(-lags, 0)
    sizes = np.minimum(len(a) - ia, len(b) - ib)
    okay = sizes >= max(10 * sample_rate, 0.8 * min(len(a), len(b)))
    lags, ia, ib, sizes = (x[okay] for x in (lags, ia, ib, sizes))
    if not len(lags):
        return {'confirmed_shared_excerpt': False, 'reason': 'insufficient_waveform_overlap'}
    cross = correlate(a.astype(np.float64), b.astype(np.float64), method='fft')
    def moments(x, starts):
        x = x.astype(np.float64)
        sums = np.r_[0., np.cumsum(x)]
        squares = np.r_[0., np.cumsum(x*x)]
        mean_sum = sums[starts+sizes] - sums[starts]
        centered_energy = squares[starts+sizes] - squares[starts] - mean_sum**2/sizes
        return mean_sum, np.maximum(centered_energy, 0)
    sa, ea = moments(a, ia)
    sb, eb = moments(b, ib)
    covariance = cross[lags + len(b) - 1] - sa*sb/sizes
    corr = np.divide(covariance, np.sqrt(ea*eb), out=np.zeros_like(covariance), where=ea*eb > 1e-12)
    best = int(np.argmax(np.abs(corr)))
    n, start_a, start_b = int(sizes[best]), int(ia[best]), int(ib[best])
    x, y = a[start_a:start_a+n], b[start_b:start_b+n]
    width = 5 * sample_rate
    windows = [abs(normalized_correlation(x[k:k+width], y[k:k+width]))
               for k in range(0, n-width+1, width)]
    window_median = float(np.median(windows))
    audible = min(float(np.std(x)), float(np.std(y))) > 1e-5
    confirmed = audible and abs(float(corr[best])) >= 0.985 and window_median >= 0.985
    return {'confirmed_shared_excerpt': confirmed, 'waveform_correlation': float(corr[best]),
            'window_correlation_median': window_median, 'window_correlations': windows,
            'offset_samples_8000': int(lags[best]), 'overlap_seconds': n / sample_rate,
            'shorter_excerpt_coverage': n / min(len(a), len(b)), 'audible': audible}


def decode_excerpt(path):
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-t', '120',
           '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '8000', '-c:a', 'pcm_f32le', '-f', 'f32le', 'pipe:1']
    result = subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    return np.frombuffer(result.stdout, dtype='<f4')


def lineage_ids(row):
    """실제 식별자만 namespace에 넣어 null/sentinel로 모든 곡이 합쳐지는 것을 막는다."""
    provider = row.get('provider')
    if provider not in ('suno', 'udio'):
        return set()
    values = [row.get('media_id'), row.get('song_id'), *(row.get('lineage') or {}).values()]
    ids = set()
    for value in values:
        if not isinstance(value, str):
            continue
        tokens = urlparse(value).path.split('/') if value.startswith('https://') else [value]
        for token in tokens:
            compact = token.replace('-', '').lower()
            if re.fullmatch(r'[0-9a-f]{32}', compact) and compact != '0' * 32:
                ids.add(provider + ':' + compact)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--validation', required=True, type=Path)
    ap.add_argument('--inventory', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    entries = json.loads(args.manifest.read_text())['bench']
    summary = json.loads((args.validation / 'summary.json').read_text())
    if summary['expected'] != len(entries) or summary['outcomes'] != {'valid': len(entries)}:
        raise ValueError('Complete successful validation is required')
    validations, paths = [], []
    for entry in entries:
        key = hashlib.sha256(entry['path'].encode()).hexdigest()
        path = args.validation / 'records' / (key + '.json')
        row = json.loads(path.read_text())
        if row['path'] != entry['path'] or row['outcome'] != 'valid':
            raise ValueError('Validation identity mismatch')
        validations.append(row)
        paths.append(path)
    inputs = {'manifest_sha256': digest(args.manifest), 'inventory_sha256': digest(args.inventory),
              'tool_sha256': digest(__file__), 'common_sha256': digest(Path(__file__).with_name('common.py')),
              'specification_sha256': digest(Path('docs/DUPLICATE_AUDIT_v1.2.md')),
              'validation_hashes': {p.name: digest(p) for p in paths}}
    write_once(args.output / 'inputs.json', inputs)
    keys = [entry_key(r) for r in entries]
    duplicates, dependencies = Groups(keys), Groups(keys)
    exact = []
    for field in ('sha256', 'decoded_pcm_sha256_22050_mono_f32le'):
        indexed = defaultdict(list)
        for i, row in enumerate(validations):
            indexed[row[field]].append(i)
        for group in indexed.values():
            if len(group) > 1:
                exact.append({'evidence': field, 'entries': [keys[i] for i in group]})
                for i in group[1:]:
                    duplicates.join(keys[group[0]], keys[i])
    print(json.dumps({'phase': 'fingerprint_candidates', 'exact_groups': len(exact)}), flush=True)
    candidates, search = fingerprint_candidates([r['chromaprint_raw_120s'] for r in validations])
    write_once(args.output / 'fingerprint_candidates.json', {'search': search, 'candidates': candidates})
    verified = []
    for number, candidate in enumerate(candidates):
        a, b = candidate['a'], candidate['b']
        record_path = args.output / 'pairs' / f'{a:04d}-{b:04d}.json'
        if record_path.exists():
            evidence = json.loads(record_path.read_text())
        else:
            # 입력 해시를 재확인하여 검증 뒤 변경된 원본을 조용히 사용하지 않는다.
            for i in (a, b):
                if digest(entries[i]['path']) != validations[i]['sha256']:
                    raise ValueError('Candidate audio changed since validation')
            evidence = dict(candidate, entries=[keys[a], keys[b]])
            evidence.update(compare_waveforms(decode_excerpt(entries[a]['path']),
                            decode_excerpt(entries[b]['path']), candidate['offset_fp_frames'] * FP_STEP_SECONDS))
            write_once(record_path, evidence)
        verified.append(evidence)
        if evidence['confirmed_shared_excerpt']:
            duplicates.join(keys[a], keys[b])
        print(json.dumps({'pair': number+1, 'total': len(candidates), 'confirmed': evidence['confirmed_shared_excerpt']}), flush=True)
    duplicate_groups = duplicates.labels(keys, 'recording:')
    linked, exposure_links = {}, []
    exposed = {t.replace('-', '').lower() for t in json.loads(args.inventory.read_text())['known_exposure_tracks']}
    for key, entry in zip(keys, entries):
        dependencies.join(key, duplicate_groups[key])
        if entry.get('creator_group'):
            dependencies.join(key, 'creator:' + entry['creator_group'])
        for lineage in sorted(lineage_ids(entry)):
            dependencies.join(key, 'lineage:' + lineage)
            linked.setdefault(lineage, []).append(key)
            if lineage in exposed:
                exposure_links.append({'entry': key, 'lineage_id': lineage})
    dependence_groups = dependencies.labels(keys, 'cluster:')
    grouped = defaultdict(list)
    for key, row in zip(keys, entries):
        grouped[duplicate_groups[key]].append(row)
    conflicts = [group for group, rows in grouped.items() if len({r['label'] for r in rows}) > 1]
    report = {
        'evaluation_entries': len(entries), 'recording_groups': len(grouped),
        'dependence_clusters': len(set(dependence_groups.values())),
        'exact_groups': exact, 'perceptual_candidates': len(verified),
        'confirmed_shared_excerpt_pairs': sum(r['confirmed_shared_excerpt'] for r in verified),
        'unconfirmed_candidates': [r for r in verified if not r['confirmed_shared_excerpt']],
        'fingerprints_available': sum(r['chromaprint_raw_120s'] is not None for r in validations),
        'label_conflicts': conflicts, 'known_lineage_exposure_links': exposure_links,
        'shared_lineage_ids': {k: v for k, v in linked.items() if len(v) > 1},
        'groups': {k: {'recording_group': duplicate_groups[k], 'dependence_cluster': dependence_groups[k]} for k in keys},
        'scope': 'Full exact hashes; bounded first-120s near-identical excerpt search. Not proof of universal deduplication or train independence.',
    }
    write_once(args.output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('groups', 'exact_groups', 'unconfirmed_candidates')}), flush=True)
    if conflicts or exposure_links:
        raise SystemExit('Label/lineage exposure requires review before release')


if __name__ == '__main__':
    main()

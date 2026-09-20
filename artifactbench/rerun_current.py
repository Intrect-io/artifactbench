"""현재 ArtifactNet으로 누수 정정 test 전체를 재실행하고 실행 근거를 보존한다."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from .data.manifest import load_audio_mono, WAV_SOURCES
from .metrics.codec_pair import codec_pair_measure
from .metrics.failcheck import check_fail
from .metrics.source_level import summarize
from .models.artifactnet import ArtifactNetModel
from .report.markdown import single_model_report
from .report.roc import generate_roc_plot, generate_roc_report, threshold_sweep


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0 = full corrected test set")
    ap.add_argument("--codec-pairs", type=int, default=50)
    ap.add_argument("--recover-fma-from", type=Path,
                    help="reuse verified track results and retry HTML-disguised FMA files using local fma_full")
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rd = Path('/home/unohee/dev/ArtifactNet')
    sys.path.insert(0, str(rd))
    import src.pipeline.infer as infer
    from src.pipeline import model_config as config
    from src.data.dataset_guard import stem_key, same_seed_different_generator
    import lightgbm as lgb

    # 공유 GPU의 남은 메모리에 맞춰 UNet 배치만 제한한다. 청크/집계 프로토콜은 유지한다.
    infer.GPU_BATCH_MAX = 1
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    test_path = rd / 'outputs/artifactbench_test_clean_runner_260703.json'
    clean_path = rd / 'outputs/artifactbench_v1_manifest_v1.0.1.json'
    original = json.loads(test_path.read_text())['bench']
    clean = json.loads(clean_path.read_text())['bench']
    allowed = {(e['source'], e['track_id']) for e in clean}
    entries = [e for e in original if (e['source'], e['track_id']) in allowed]
    excluded = [e for e in original if (e['source'], e['track_id']) not in allowed]
    training_path = rd / 'outputs/bigset_260720/train_manifest.json'
    training = json.loads(training_path.read_text())
    exposed = defaultdict(list)
    for split in ('train', 'val', 'test'):
        for e in training[split]:
            exposed[stem_key(e['path'])].append(e)
    collisions = [(e['path'], x['path']) for e in entries for x in exposed[stem_key(e['path'])]
                  if not same_seed_different_generator(e['path'], x['path'], e['label'])]
    assert not collisions, f'Unresolved training overlap: {collisions[:5]}'
    if args.limit:
        # smoke도 여러 소스를 통과하도록 균등하게 선택한다.
        grouped = defaultdict(list)
        for e in entries:
            grouped[e['source']].append(e)
        entries = [grouped[s][0] for s in sorted(grouped)][:args.limit]
    assert entries and len({e['path'] for e in entries}) == len(entries)
    recovered = {}
    inherited = []
    parent_files = []
    if args.recover_fma_from:
        parent = args.recover_fma_from.resolve()
        assert parent != out.resolve() and not args.limit
        previous = json.loads((parent / 'provenance.json').read_text())
        snapshot = parent / 'runner_snapshot.py'
        for p, h in previous['files'].items():
            checked = snapshot if p == str(Path(__file__)) else Path(p)
            assert digest(checked) == h, f'Parent input changed: {p}'
        old_rows = [json.loads(line) for line in (parent / 'rows.jsonl').read_text().splitlines()]
        expected = {e['path']: e for e in entries}
        assert len(old_rows) == len(expected) == len({r['path'] for r in old_rows})
        assert {r['path'] for r in old_rows} == set(expected)
        assert previous['threshold'] == config.DEFAULT_THRESHOLD
        assert previous['rescue_tau'] == config.DEFAULT_RESCUE_TAU
        assert previous['device'] == args.device and previous['torch'] == torch.__version__
        for r in old_rows:
            assert all(r[k] == expected[r['path']][k] for k in ('source', 'label', 'track_id'))
            if 'error' not in r:
                inherited.append(r)
            elif r['source'] == 'fma_hardneg' and 'Audio decode failed:' in r['error']:
                p = Path(r['path'])
                assert p.stem.isdigit() and len(p.stem) == 6
                with p.open('rb') as fh:
                    assert fh.read(30).lstrip().lower().startswith(b'<!doctype html')
                original = Path('/media/unohee/Archive/dataset/fma_full') / p.stem[:3] / p.name
                assert original.is_file()
                recovered[r['path']] = str(original)
        parent_files = [parent / 'provenance.json', parent / 'rows.jsonl', snapshot]
    identity = {
        'artifactnet_commit': subprocess.check_output(['git', '-C', str(rd), 'rev-parse', 'HEAD'], text=True).strip(),
        'files': {str(p): digest(p) for p in [rd / config.DEFAULT_CNN, rd / config.DEFAULT_UNET,
                  rd / config.DEFAULT_RESCUE_LGBM, rd / 'src/pipeline/infer.py',
                  rd / 'src/pipeline/model_config.py', Path(__file__),
                  Path(__file__).parent / 'models/artifactnet.py', test_path, clean_path, training_path]},
        'cnn': config.DEFAULT_CNN, 'rescue': config.DEFAULT_RESCUE_LGBM,
        'threshold': config.DEFAULT_THRESHOLD, 'rescue_tau': config.DEFAULT_RESCUE_TAU,
        'device': args.device, 'unet_batch': 1, 'max_seconds': 60,
        'aggregation': 'RMS-weighted mean', 'raw_report_threshold': 0.5,
        'codec_pairs': args.codec_pairs, 'limit': args.limit, 'seed': 42,
        'expected_tracks': len(entries), 'per_source': dict(Counter(e['source'] for e in entries)),
        'excluded_from_original_test': len(excluded),
        'exposure_stem_collisions': len(collisions),
        'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'production_metric_scope': 'CNN threshold + matched rescue only; no AcoustID, integrity gate or codec TTA',
        'recovered_audio_paths': recovered, 'inherited_successes': len(inherited),
    }
    identity['files'].update({str(p): digest(p) for p in parent_files + list(recovered.values())})
    provenance = out / 'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text()) != identity:
        raise RuntimeError('Provenance changed; use a fresh output directory')
    write_json(provenance, identity)
    write_json(out / 'manifest.json', {'bench': entries, 'excluded': excluded})
    print(f"Expected {len(entries)} tracks / {len(identity['per_source'])} sources; excluded {len(excluded)}", flush=True)
    model = ArtifactNetModel(repo_dir=str(rd))
    model.load(args.device)
    rescue = lgb.Booster(model_file=str(rd / config.DEFAULT_RESCUE_LGBM))
    rows_path = out / 'rows.jsonl'
    if not rows_path.exists() and inherited:
        with rows_path.open('x') as fh:
            for r in inherited:
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    done = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done[r['path']] = r
    started = time.monotonic()
    with rows_path.open('a') as fh:
        for e in entries:
            if e['path'] in done:
                continue
            try:
                audio_path = recovered.get(e['path'], e['path'])
                audio = load_audio_mono(audio_path)
                if audio is None:
                    raise RuntimeError(f"Audio decode failed: {e['path']}")
                prob = model.forward(audio)
                rp = float(rescue.predict(np.array([model.last_features['verdict_feat']], dtype=np.float32), num_threads=1)[0])
                assert np.isfinite(prob) and 0 <= prob <= 1 and np.isfinite(rp)
                r = dict(e, prob=prob, rescue_prob=rp, n_chunks=model.last_features['n_chunks'],
                         resolved_audio_path=audio_path,
                         production_ai=bool(prob >= config.DEFAULT_THRESHOLD and rp >= config.DEFAULT_RESCUE_TAU))
            except torch.cuda.OutOfMemoryError:
                # 리소스 충돌을 모델 결함으로 채점하지 않는다. 같은 출력으로 재개 가능하다.
                raise
            except (RuntimeError, ValueError) as exc:
                r = dict(e, error=f'{type(exc).__name__}: {exc}')
                print(f"INFERENCE FAILURE {e['path']}: {r['error']}", flush=True)
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
            fh.flush()
            done[e['path']] = r
            if len(done) % 20 == 0 or len(done) == 1:
                print(f"Scored {len(done)}/{len(entries)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    attempts = [done[e['path']] for e in entries]
    assert len(attempts) == len(entries)
    errors = [r for r in attempts if 'error' in r]
    rows = [r for r in attempts if 'error' not in r]
    write_json(out / 'errors.json', errors)
    groups = defaultdict(list)
    for r in rows:
        groups[r['source']].append(r)
    per_source = {s: summarize(rr) for s, rr in groups.items()}
    track_probs = [dict(prob=r['prob'], label=int(r['label']=='ai'), source=r['source'], path=r['path']) for r in rows]
    codec_entries = [dict(e, path=recovered.get(e['path'], e['path']))
                     for e in entries if e['source'] in WAV_SOURCES]
    random.shuffle(codec_entries)
    codec_file = out / 'codec_pair.json'
    if codec_file.exists():
        pairs = json.loads(codec_file.read_text())
    else:
        pairs = codec_pair_measure(codec_entries, model, n_pair=args.codec_pairs)
        write_json(codec_file, pairs)
    assert len(pairs) == args.codec_pairs
    fails = check_fail(per_source, pairs)
    result = dict(model_info=model.info(), per_source=per_source, codec_pairs=pairs,
                  track_probs=track_probs, fails=fails, elapsed=time.monotonic()-started)
    for name, data in [('per_source', per_source), ('per_source_raw', groups), ('track_probs', track_probs)]:
        write_json(out / f'{name}.json', data)
    (out / 'report.md').write_text(single_model_report(model.info(), per_source, pairs, fails, result['elapsed']))
    (out / 'roc_analysis.md').write_text(generate_roc_report([result]))
    generate_roc_plot([result], out / 'roc_curves.png')
    def metrics(rr, production=False, threshold=0.5):
        ys = np.array([int(r['label']=='ai') for r in rr])
        ps = np.array([float(r['production_ai']) if production else r['prob'] for r in rr])
        return threshold_sweep(ps, ys, [0.5 if production else threshold])[0]
    summary = dict(expected=len(entries), attempted=len(attempts), scored=len(rows), errors=len(errors),
                   sources=len(groups), excluded=len(excluded),
                   raw_05=metrics(rows), raw_operating=metrics(rows, threshold=config.DEFAULT_THRESHOLD),
                   production=metrics(rows, production=True),
                   production_per_source={s: metrics(rr, production=True) for s, rr in groups.items()},
                   raw_sanity_fails=fails, codec_pairs=len(pairs), identity=identity)
    # 측정 도중 체크포인트나 코드가 바뀌면 완료로 간주하지 않는다.
    assert all(digest(p) == h for p, h in identity['files'].items()), 'Inputs changed during execution'
    write_json(out / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['scored', 'sources', 'raw_05', 'production']}, indent=2), flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()

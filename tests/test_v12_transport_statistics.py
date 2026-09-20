from collections import Counter
import json
import sys

import numpy as np
import pytest

from artifactbench.v12.audio import CODECS, TRANSFORMS
from artifactbench.v12.common import digest, write_once
from artifactbench.v12.transport_statistics import main, paired_values, staging_contrasts, summarize_pairs, summarize_values


def entries(n):
    return [{'source': 'one', 'label': 'ai', 'dependence_cluster': str(i)} for i in range(n)]


def test_paired_failures_stay_unscored_and_in_coverage():
    assert all(np.isnan(v) for v in paired_values({'outcome': 'model_execution_error'}, {'outcome': 'scored', 'prob': .9}))
    summary = summarize_values(entries(3), [0., .2, np.nan], replicates=20)
    assert summary['attempted_recordings'] == 3 and summary['complete_recordings'] == 2
    assert summary['coverage'] == 2/3 and summary['mean']['estimate'] == .1
    empty = summarize_values(entries(2), [np.nan, np.nan], replicates=20)
    assert empty['mean']['estimate'] is None and empty['mean']['finite_replicates'] == 0


def test_identity_shifts_and_bootstrap_are_zero():
    row = {'outcome': 'scored', 'prob': .7}
    assert paired_values(row, row) == (0., 0., 0.)
    summary = summarize_values(entries(4), [0., 0., 0., 0.], replicates=20)
    assert summary['mean']['estimate'] == summary['mean']['lower'] == summary['mean']['upper'] == 0.


def transport_fixture(probabilities):
    """집계 산술용 단위 fixture이며 실제 모델 결과가 아니다."""
    selected = [dict(e, id=str(i)) for i, e in enumerate(entries(len(probabilities)))]
    rows = {}
    for entry, scores in zip(selected, probabilities):
        for view in TRANSFORMS:
            value = scores[view]
            rows[entry['id']+'--'+view] = ({'outcome': 'scored', 'prob': value} if value is not None
                                        else {'outcome': 'model_execution_error'})
    return selected, rows


def staged_scores(f=.4, s=.7, cf=.5, cs=.9):
    return {'float_identity': f, 'pcm16_only': s,
            **{codec: cf for codec in CODECS}, **{'pcm16_'+codec: cs for codec in CODECS}}


def test_codec_increment_uses_same_staging_not_total_shift():
    selected, rows = transport_fixture([staged_scores()])
    contrasts = staging_contrasts(selected, rows, replicates=20)
    assert contrasts['quantization_only']['metrics']['signed_shift']['mean']['estimate'] == pytest.approx(.3)
    for codec in CODECS:
        result = contrasts['codecs'][codec]
        assert result['codec_from_pcm16']['first_view'] == 'pcm16_only'
        assert result['codec_from_pcm16']['second_view'] == 'pcm16_'+codec
        assert result['codec_from_float']['metrics']['signed_shift']['mean']['estimate'] == pytest.approx(.1)
        # .9-.4=.5는 전체 경로 차이, .9-.7=.2가 PCM16 baseline에 대한 증가분이다.
        assert result['codec_from_pcm16']['metrics']['signed_shift']['mean']['estimate'] == pytest.approx(.2)
        assert result['staging_with_codec']['metrics']['signed_shift']['mean']['estimate'] == pytest.approx(.4)
        assert result['signed_interaction']['statistics']['mean']['estimate'] == pytest.approx(.1)
        assert result['codec_from_float']['metrics']['raw_05_flip_count'] == 1
        assert result['codec_from_pcm16']['metrics']['raw_05_flip_count'] == 0


@pytest.mark.parametrize('failed_view', TRANSFORMS[:2]+('mp3_128', 'pcm16_mp3_128'))
def test_interaction_requires_all_four_views_and_retains_failed_coverage(failed_view):
    complete, incomplete = staged_scores(), staged_scores()
    incomplete[failed_view] = None
    selected, rows = transport_fixture([complete, incomplete])
    result = staging_contrasts(selected, rows, replicates=20)['codecs']['mp3_128']
    summary = result['signed_interaction']['statistics']
    assert summary['attempted_recordings'] == 2 and summary['complete_recordings'] == 1
    assert summary['coverage'] == .5 and summary['mean']['estimate'] == pytest.approx(.1)
    for name in ('codec_from_float', 'codec_from_pcm16', 'staging_with_codec'):
        pair = result[name]
        expected = 1 if failed_view in (pair['first_view'], pair['second_view']) else 2
        assert pair['metrics']['signed_shift']['complete_recordings'] == expected


def test_four_view_paired_bootstrap_preserves_recording_level_cancellation():
    selected, rows = transport_fixture([staged_scores(.1, .2, .3, .5),
                                       staged_scores(.4, .1, .8, .6),
                                       staged_scores(.6, .7, .2, .4)])
    result = staging_contrasts(selected, rows, replicates=200)
    repeated = staging_contrasts(selected, rows, replicates=200)
    assert result == repeated
    # 각 코덱 증가분은 녹음별로 달라도 그 차이는 항상 .1이다.
    summary = result['codecs']['mp3_128']['signed_interaction']['statistics']['mean']
    assert summary['estimate'] == pytest.approx(.1)
    assert summary['lower'] == pytest.approx(.1) and summary['upper'] == pytest.approx(.1)


def test_all_failed_contrasts_are_null_not_zero_effect():
    selected, rows = transport_fixture([dict.fromkeys(TRANSFORMS)])
    contrasts = staging_contrasts(selected, rows, replicates=20)
    pair = contrasts['quantization_only']['metrics']
    assert pair['raw_05_flip_count'] is None
    assert pair['signed_shift']['complete_recordings'] == 0
    assert pair['signed_shift']['mean']['estimate'] is None
    summary = contrasts['codecs']['mp3_128']['signed_interaction']['statistics']['mean']
    assert summary['estimate'] is None and summary['finite_replicates'] == 0


@pytest.mark.parametrize('probability', [np.nan, np.inf, -.01, 1.01])
def test_scored_invalid_values_cannot_enter_staging_contrasts(probability):
    scores = staged_scores()
    scores['pcm16_mp3_128'] = probability
    selected, rows = transport_fixture([scores])
    with pytest.raises(ValueError, match='Invalid scored probability'):
        staging_contrasts(selected, rows, replicates=20)


def test_paired_summary_does_not_silently_truncate_mismatched_inputs():
    with pytest.raises(ValueError, match='align'):
        summarize_pairs(entries(2), [{'outcome': 'scored', 'prob': .2}], [])


def test_cli_persists_matched_staging_and_interactions_from_hash_bound_unit_fixture(tmp_path, monkeypatch):
    prepared, run, output = (tmp_path/name for name in ('prepared', 'run', 'statistics'))
    selected, rows = transport_fixture([staged_scores()])
    write_once(prepared/'manifest.local.json', {'controlled': [{'entry': e} for e in selected], 'native': []})
    identity = {'prepared_input_hashes': {'manifest.local.json': digest(prepared/'manifest.local.json')},
                'expected_waveforms': len(rows)}
    write_once(run/'identity.json', identity)
    record_hashes = {}
    for identifier, row in rows.items():
        path = run/'records'/(identifier+'.json')
        write_once(path, dict(row, id=identifier))
        record_hashes[identifier] = digest(path)
    write_once(run/'summary.json', {'status': 'paired transport run complete', 'model': 'unit-fixture',
               'identity_sha256': digest(run/'identity.json'), 'record_sha256': record_hashes,
               'expected': len(rows), 'outcomes': dict(Counter(r['outcome'] for r in rows.values()))})
    monkeypatch.setattr(sys, 'argv', ['transport_statistics', '--prepared', str(prepared),
                                    '--run', str(run), '--output', str(output)])
    main()
    result = json.loads((output/'statistics.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    inputs = json.loads((output/'inputs.json').read_text())
    assert summary['statistics_sha256'] == digest(output/'statistics.json')
    assert inputs['replicates'] == 2000
    assert result['controlled']['pcm16_mp3_128']['signed_shift']['mean']['estimate'] == pytest.approx(.5)
    matched = result['controlled_contrasts']['codecs']['mp3_128']
    assert matched['codec_from_pcm16']['metrics']['signed_shift']['mean']['estimate'] == pytest.approx(.2)
    assert matched['signed_interaction']['statistics']['mean']['estimate'] == pytest.approx(.1)
    assert result['controlled']['mp3_128']['raw_05_flip_count'] == 1

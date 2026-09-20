import random

import numpy as np
import pytest

from artifactbench.v12.freeze import public_url, recording_groups
from artifactbench.v12.run import measured_parameters, metrics, seed_record
import torch
from types import SimpleNamespace
from artifactbench.v12.verify_release import check_public


def test_per_record_rng_is_independent_of_execution_order():
    seed_record('a')
    expected = (random.random(), np.random.random())
    seed_record('b')
    seed_record('a')
    assert (random.random(), np.random.random()) == expected


def test_failure_is_coverage_loss_not_a_fake_score():
    rows = [{'label': 'ai', 'outcome': 'scored', 'prob': .9},
            {'label': 'real', 'outcome': 'scored', 'prob': .1},
            {'label': 'ai', 'outcome': 'model_execution_error'}]
    result = metrics(rows)
    assert result['F1'] == 1 and result['AUROC'] == 1
    assert result['coverage'] == result['all_attempt_correctness'] == 2/3
    assert result['failed'] == 1 and result['FN'] == 0
    assert metrics(rows[:1])['AUROC'] is None


def test_shared_extension_is_not_counted_as_an_identical_recording():
    entries = [{'source': 'x', 'track_id': key} for key in ('a', 'b', 'c')]
    review = {'pairs': [{'entries': ['x:a', 'x:b'], 'verdict': 'same_recording_or_excerpt'},
                        {'entries': ['x:b', 'x:c'], 'verdict': 'shared_audio_lineage_not_identical_recording'}]}
    groups, representatives = recording_groups(entries, {'exact_groups': []}, review)
    assert groups['x:a'] == groups['x:b'] != groups['x:c']
    assert len(representatives) == 2


def test_public_url_never_exports_credentials_or_query_tokens():
    assert public_url(None) is None
    assert public_url('https://example.com/song#play') == 'https://example.com/song'
    for url in ('https://user:password@example.com/song', 'https://example.com/song?token=private', '/home/user/song.mp3'):
        with pytest.raises(ValueError):
            public_url(url)


def test_public_metadata_rejects_nested_raw_content_and_local_paths():
    check_public({'source_url': 'https://example.com/song', 'description': 'path-stem audit'})
    check_public({'source_url': 'https://storage.googleapis.com/bucket/media/song.webm'})
    for value in ({'nested': [{'lyrics': 'private'}]}, {'value': '/media/user/song.mp3'}, {'url': 'https://user:pass@example.com'}):
        with pytest.raises(ValueError):
            check_public(value)


def test_parameter_measurement_does_not_double_count_shared_module():
    layer = torch.nn.Linear(3, 2)
    result = measured_parameters(SimpleNamespace(a=layer, b=layer))
    assert result['total_unique_torch_parameters'] == 8

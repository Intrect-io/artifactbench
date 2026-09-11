import json

import pytest

from artifactbench.v12.common import rank
from artifactbench.v12.run_fst_isolated import CONDITION, entries, timeout_row


def test_entries_require_exact_public_identity_and_fixed_order(tmp_path):
    release = tmp_path/'release'
    release.mkdir()
    local = [{'id': key, 'sha256': key+'h', 'source': 's', 'label': 'ai', 'track_id': key,
              'partition': 'legacy', 'recording_group': key, 'recording_representative': key,
              'dependence_cluster': key, 'path': '/unit/'+key} for key in ('a', 'b')]
    public = [{key: value for key, value in row.items() if key != 'path'} for row in local]
    (release/'manifest.local.json').write_text(json.dumps({'bench': local}))
    (release/'manifest.public.json').write_text(json.dumps({'bench': public}))
    got = entries(release)
    assert [row['id'] for row in got] == sorted(('a', 'b'), key=rank)
    public[0]['source'] = 'changed'
    (release/'manifest.public.json').write_text(json.dumps({'bench': public}))
    with pytest.raises(ValueError, match='row mismatch'):
        entries(release)


def test_timeout_row_preserves_identity_but_never_fabricates_probability():
    entry = {'id': 'unit', 'sha256': 'frozen', 'source': 'source', 'track_id': 'track',
             'label': 'real', 'partition': 'legacy', 'recording_group': 'group',
             'recording_representative': 'rep', 'dependence_cluster': 'cluster'}
    row = timeout_row(entry, 600)
    assert row['outcome'] == 'model_execution_error'
    assert row['error_type'] == 'TimeoutExpired'
    assert row['timeout_seconds'] == 600
    assert 'prob' not in row and row['audio_sha256'] == 'frozen'
    assert CONDITION == 'fst-isolated-worker/1'

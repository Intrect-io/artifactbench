"""전체 소스 실행 옵션이 빈 결과로 성공하지 않는지 확인한다."""
import numpy as np
import pytest

from artifactbench.metrics import source_level


@pytest.mark.parametrize('limit, expected', [(0, 3), (1, 1), (100, 3)])
def test_source_limit(monkeypatch, limit, expected):
    entries = [
        dict(track_id=str(i), path=str(i), label='ai', source='test')
        for i in range(3)
    ]
    monkeypatch.setattr(source_level, 'load_audio_mono', lambda _: np.ones(10))

    class Model:
        def forward(self, audio):
            return 0.75

    selected = entries if limit == 0 else entries[:limit]
    result, failures = source_level.measure_source(selected, Model(), verbose=False)
    assert failures == []
    assert len(result) == expected
    assert [r['path'] for r in result] == [str(i) for i in range(expected)]

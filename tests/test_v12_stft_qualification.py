"""비교 경계용 unit 수치이며 detector 측정 결과가 아니다."""
from types import SimpleNamespace

import numpy as np
import pytest

from artifactbench.v12.qualify_onnx_stft import measure_entry


def test_saved_gpu_and_factory_cpu_comparisons_remain_separate():
    runtime = SimpleNamespace(predict_chunks=lambda _: .501)
    model = SimpleNamespace(forward=lambda _: .499)
    row = measure_entry(runtime, model, np.zeros(100, np.float32), {'outcome': 'scored', 'prob': .501})
    assert row['original_gpu_comparison']['within_tolerance']
    assert not row['factory_cpu_comparison']['within_tolerance']
    assert row['factory_cpu_comparison']['decision_flip_05']


def test_original_failure_stays_noncomparable():
    def fail(_):
        raise RuntimeError('unit short-input failure')
    row = measure_entry(SimpleNamespace(predict_chunks=lambda _: .8), SimpleNamespace(forward=fail),
                        np.zeros(100, np.float32), {'outcome': 'model_execution_error', 'error_type': 'RuntimeError'})
    assert 'original_gpu_comparison' not in row and 'factory_cpu_comparison' not in row
    assert row['cpu_reference_error_type'] == 'RuntimeError'
    with pytest.raises(RuntimeError):
        measure_entry(SimpleNamespace(predict_chunks=lambda _: .8), SimpleNamespace(forward=fail),
                      np.zeros(100, np.float32), {'outcome': 'scored', 'prob': .8})


def test_invalid_reference_probability_is_not_accepted():
    with pytest.raises(ValueError, match='probability'):
        measure_entry(SimpleNamespace(predict_chunks=lambda _: .8), SimpleNamespace(forward=lambda _: np.nan),
                      np.zeros(100, np.float32), {'outcome': 'scored', 'prob': .8})

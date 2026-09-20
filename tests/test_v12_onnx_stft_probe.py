import numpy as np
import pytest
import torch

onnx = pytest.importorskip('onnx')
ort = pytest.importorskip('onnxruntime')
from onnx import helper as h, TensorProto as T
from artifactbench.v12.diagnose_artifactnet_onnx import fft_magnitude
from artifactbench.v12.probe_onnx_stft import MAGNITUDE, native_prefix, replace_prefix


def test_standard_stft_executes_with_reference_frame_policy():
    window = torch.hann_window(2048)
    nodes, initializers = native_prefix(window.numpy())
    model = h.make_model(h.make_graph(nodes, 'unit-STFT',
        [h.make_tensor_value_info('audio_chunks', T.FLOAT, ['chunks', 4096])],
        [h.make_tensor_value_info(MAGNITUDE, T.FLOAT, ['chunks', 1, 1025, 9])], initializers),
        opset_imports=[h.make_opsetid('', 18)], ir_version=8)
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    runtime = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    # 정확도 표본이 아닌 결정론적 입력 경계 fixture다.
    wave = torch.linspace(-.5, .5, 8192).reshape(2, 4096)
    got = runtime.run([MAGNITUDE], {'audio_chunks': wave.numpy()})[0]
    expected = fft_magnitude(wave, window).numpy()
    assert got.shape == (2, 1, 1025, 9)
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-5)


def test_bad_window_and_unknown_boundary_rejected():
    with pytest.raises(ValueError, match='reference window'):
        native_prefix(np.zeros(2048, dtype=np.float64))
    parent = h.make_model(h.make_graph([], 'missing', [], []))
    with pytest.raises(ValueError, match='known v2'):
        replace_prefix(parent, np.zeros(2048, dtype=np.float32))


def test_replacement_preserves_downstream_weights_shared_constants_and_parent():
    from onnx import numpy_helper as nh
    # 수치 성능을 뜻하지 않는 작은 유효 ONNX fixture로 경계 교체만 검증한다.
    nodes = [h.make_node('Constant', [], ['axis'], name='shared', value=nh.from_array(np.array([1], np.int64))),
        h.make_node('Unsqueeze', ['audio_chunks', 'axis'], ['signal']),
        h.make_node('Conv', ['signal', 'cosine'], ['real']),
        h.make_node('Conv', ['signal', 'sine'], ['imaginary']),
        h.make_node('Add', ['real', 'imaginary'], ['combined']),
        h.make_node('Unsqueeze', ['combined', 'axis'], [MAGNITUDE]),
        h.make_node('Mul', [MAGNITUDE, 'downstream_weight'], ['weighted'], name='downstream'),
        h.make_node('ReduceSum', ['weighted', 'axis'], ['output'], keepdims=0, name='reduction')]
    tensors = [nh.from_array(np.ones((1, 1, 1), np.float32), name) for name in ('cosine', 'sine')]
    tensors.append(nh.from_array(np.array(.75, np.float32), 'downstream_weight'))
    parent = h.make_model(h.make_graph(nodes, 'unit-boundary',
        [h.make_tensor_value_info('audio_chunks', T.FLOAT, ['chunks', 4096])],
        [h.make_tensor_value_info('output', T.FLOAT, [None, None, None])], tensors),
        opset_imports=[h.make_opsetid('', 18)], ir_version=8)
    onnx.checker.check_model(parent, full_check=True)
    original = parent.SerializeToString()
    child, report = replace_prefix(parent, torch.hann_window(2048).numpy())
    assert parent.SerializeToString() == original
    assert report['removed_initializers'] == ['cosine', 'sine']
    assert report['unchanged_parent_initializers'] == 1
    assert {n.name for n in child.graph.node}.issuperset({'shared', 'downstream', 'reduction'})
    assert next(t for t in child.graph.initializer if t.name == 'downstream_weight') == tensors[-1]
    parent.graph.initializer.append(nh.from_array(np.array(1., np.float32), 'unexpected_unused'))
    with pytest.raises(ValueError, match='more than the DFT'):
        replace_prefix(parent, torch.hann_window(2048).numpy())

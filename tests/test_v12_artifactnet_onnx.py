"""ONNX 경계/수식의 unit fixture. 학습된 검출기의 성능 또는 변환 성공 증거가 아니다."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from artifactbench.v12.artifactnet_onnx import CHUNK_SAMPLES, MAX_SAMPLES, RawOnnx, song_chunks
from artifactbench.v12.artifactnet_onnx_graph import aggregate, features, median_filter


@pytest.mark.parametrize('length, expected', ((1, 1), (CHUNK_SAMPLES-1, 1), (CHUNK_SAMPLES, 1),
    (CHUNK_SAMPLES+CHUNK_SAMPLES//2-1, 1), (CHUNK_SAMPLES+CHUNK_SAMPLES//2, 2),
    (MAX_SAMPLES, 15), (MAX_SAMPLES+CHUNK_SAMPLES, 15)))
def test_song_chunk_boundary_and_prefix(length, expected):
    audio = np.full(length, .25, dtype=np.float32)
    got = song_chunks(audio)
    assert got.shape == (expected, CHUNK_SAMPLES) and got.dtype == np.float32
    assert got.flags.c_contiguous
    valid = min(length, MAX_SAMPLES, expected*CHUNK_SAMPLES)
    np.testing.assert_array_equal(got.reshape(-1)[:valid], audio[:valid])
    assert not np.any(got.reshape(-1)[valid:])


@pytest.mark.parametrize('audio', ([], [np.nan], [np.inf], [1.01], np.ones((2, 10)), [1, 2], ['a']))
def test_invalid_waveform_rejected(audio):
    with pytest.raises(ValueError):
        song_chunks(audio)


def test_rms_aggregation_matches_original_numpy_float64_formula():
    chunks = torch.tensor([[.1, .3, .4], [.001, .002, .005], [0, 0, 0]], dtype=torch.float32)
    probabilities = torch.tensor([.8, .1, .6], dtype=torch.float32)
    rms = chunks.square().mean(dim=1).sqrt().numpy().astype(np.float64)
    expected = np.dot(probabilities.numpy().astype(np.float64), rms/(rms.sum()+1e-9))
    got = aggregate(probabilities, chunks)
    assert got.shape == (1,) and got.dtype == torch.float64
    assert float(got[0]) == pytest.approx(expected, abs=1e-15)
    assert float(aggregate(probabilities, torch.zeros_like(chunks))[0]) == 0


def test_exported_aggregation_keeps_denominator_in_float64(tmp_path):
    # Export 환경에서 실제 실행한다. 일반 평가 환경에 새 dependency를 설치하지 않는다.
    onnx = pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')

    class UnitAggregation(torch.nn.Module):
        def forward(self, probabilities, chunks):
            return aggregate(probabilities, chunks)

    chunks = torch.tensor([[.1, .3, .4], [.001, .002, .005], [.11, .12, .15]], dtype=torch.float32)
    probabilities = torch.ones(3, dtype=torch.float32)
    target = tmp_path/'unit-aggregation.onnx'
    torch.onnx.export(UnitAggregation(), (probabilities, chunks), str(target), dynamo=False,
                      opset_version=18, input_names=['probabilities', 'chunks'], output_names=['p_ai'])
    graph = onnx.shape_inference.infer_shapes(onnx.load(str(target))).graph
    kinds = {v.name: v.type.tensor_type.elem_type for v in list(graph.value_info)+list(graph.output)}
    additions = [node for node in graph.node if node.op_type == 'Add']
    assert len(additions) == 1
    assert all(kinds[name] == onnx.TensorProto.DOUBLE for name in additions[0].input)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(target), sess_options=options, providers=['CPUExecutionProvider'])
    got = float(session.run(['p_ai'], {'probabilities': probabilities.numpy(), 'chunks': chunks.numpy()})[0][0])
    assert 0 <= got <= 1
    assert got == pytest.approx(float(aggregate(probabilities, chunks)[0]), abs=1e-12)


@pytest.mark.parametrize('dimension', (1, 2))
def test_static_slice_median_equals_unfold_reference(dimension):
    value = torch.arange(2*7*9, dtype=torch.float32).reshape(2, 7, 9).remainder(17)
    padding = (2, 2) if dimension == 2 else (0, 0, 2, 2)
    expected = F.pad(value, padding, mode='reflect').unfold(dimension, 5, 1).median(-1).values
    torch.testing.assert_close(median_filter(value, 5, dimension), expected, rtol=0, atol=0)


def test_median_rejects_unsupported_axis():
    with pytest.raises(ValueError, match='dimension'):
        median_filter(torch.zeros(1, 7, 9), 5, 0)


class UnitGlobalMel(torch.nn.Module):
    """Global top-dB clamp만 분리한 fixture. 실제 mel filterbank나 모델이 아니다."""
    def forward(self, value):
        db = 10*value.square().clamp(min=1e-10).log10()
        return db.clamp(min=db.max()-80)


def test_global_mel_floor_is_shared_across_components_and_chunks():
    residual = torch.full((2, 1, 3, 5), 100.)
    residual[1] *= 1e-8
    harmonic = torch.full_like(residual, 1e-4)
    percussive = torch.full_like(residual, 1e-5)
    mel = UnitGlobalMel()
    got = features(residual, harmonic, percussive, mel)
    assert got.shape == (2, 7, 3, 5)
    assert torch.all(got[:, 1:3] == -40.)
    assert torch.all(got[1, 0] == -40.)
    assert not torch.equal(got[:, 1:2], mel(harmonic))
    separately = features(residual[1:], harmonic[1:], percussive[1:], mel)
    assert not torch.equal(got[1:], separately)


@pytest.mark.parametrize('shape,dtype', (((0, CHUNK_SAMPLES), np.float32),
    ((16, CHUNK_SAMPLES), np.float32), ((1, 10), np.float32), ((1, CHUNK_SAMPLES), np.float64)))
def test_bad_chunks_fail_before_any_runtime_call(shape, dtype):
    model = RawOnnx.__new__(RawOnnx)
    with pytest.raises(ValueError, match='1–15'):
        model.predict_chunks(np.zeros(shape, dtype=dtype))


@pytest.mark.parametrize('value', (np.nan, np.inf, 1.1))
def test_nonfinite_or_unclipped_chunks_fail_before_runtime(value):
    model = RawOnnx.__new__(RawOnnx)
    with pytest.raises(ValueError, match='1–15'):
        model.predict_chunks(np.full((1, CHUNK_SAMPLES), value, dtype=np.float32))

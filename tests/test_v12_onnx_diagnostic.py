import numpy as np
import pytest
import torch

from artifactbench.v12.diagnose_artifactnet_onnx import capture, describe, difference, fft_magnitude


def test_description_and_difference_keep_dtype_shape_and_exactness():
    left = np.array([[1., 2.]], dtype=np.float32)
    right = left.copy()
    assert describe(left) == describe(right)
    assert difference(left, right)['exact']
    right[0, 1] = 4
    result = difference(left, right)
    assert result['max_absolute'] == 2 and result['mean_absolute'] == 1
    assert result['rms'] == pytest.approx(np.sqrt(2)) and not result['exact']
    assert describe(left)['sha256'] != describe(right)['sha256']
    with pytest.raises(ValueError, match='shapes or dtypes'):
        difference(left, right.astype(np.float64))
    with pytest.raises(ValueError, match='shapes or dtypes'):
        difference(left, right.reshape(-1))


@pytest.mark.parametrize('bad', (np.array([]), np.array([1]), np.array([np.nan]), np.array([np.inf])))
def test_description_rejects_empty_nonfinite_and_nonfloating(bad):
    with pytest.raises(ValueError, match='nonempty finite'):
        describe(bad)


def test_capture_concatenates_microbatches_and_cleans_hooks():
    unet, cnn = torch.nn.Identity(), torch.nn.Identity()
    value = torch.tensor([[.1], [.2]])
    def call():
        joined = torch.cat([unet(row[None]) for row in value])
        return cnn(joined).mean()
    arrays, result = capture(unet, cnn, call)
    np.testing.assert_array_equal(arrays['magnitude'], value.numpy())
    assert result['unet_calls'] == 2 and result['p_ai'] == pytest.approx(.15)
    assert not unet._forward_hooks and not cnn._forward_hooks
    def failing():
        unet(value)
        raise RuntimeError('unit forward failed')
    with pytest.raises(RuntimeError, match='unit forward failed'):
        capture(unet, cnn, failing)
    assert not unet._forward_hooks and not cnn._forward_hooks


def test_capture_rejects_unexecuted_modules():
    unet, cnn = torch.nn.Identity(), torch.nn.Identity()
    with pytest.raises(ValueError, match='Incomplete'):
        capture(unet, cnn, lambda: .5)
    assert not unet._forward_hooks and not cnn._forward_hooks


def test_native_fft_split_uses_identical_sample_and_frame_policy():
    # Unit signal checks API/frame policy, not a trained-model parity claim.
    value = torch.linspace(-.5, .5, 8192).reshape(2, 4096)
    window = torch.hann_window(2048)
    direct = torch.stft(value, 2048, 512, window=window, return_complex=True).abs().unsqueeze(1)
    torch.testing.assert_close(fft_magnitude(value, window), direct, rtol=0, atol=0)
    split = fft_magnitude(value, window, split=True)
    assert split.shape == (2, 1, 1025, 9)
    torch.testing.assert_close(split, direct, rtol=1e-5, atol=1e-5)

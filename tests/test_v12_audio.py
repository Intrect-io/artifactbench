"""코덱 실험 자체의 정확성을 신호 fixture로 검사한다. 모델 성능 데이터가 아니다."""
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from artifactbench.v12.audio import TRANSFORMS, load_audio_float, transform_audio
from artifactbench.v12.validate_audio import inspect_audio


@pytest.fixture
def tone():
    t = np.arange(44100 * 3) / 44100
    return (0.2 * np.sin(2 * np.pi * 443 * t)).astype(np.float32)


def test_float_wav_control_is_exact(tone):
    decoded, info = transform_audio(tone, "float_identity")
    np.testing.assert_array_equal(decoded, tone)
    assert info["staging_subtype"] == "FLOAT"
    assert info["tail_adjustment_frames"] == 0


def test_pcm16_control_only_quantizes(tone):
    decoded, info = transform_audio(tone, "pcm16_only")
    assert not np.array_equal(decoded, tone)
    assert np.max(np.abs(decoded - tone)) <= 1/32768 + 1e-7
    assert info["staging_subtype"] == "PCM_16"


@pytest.mark.parametrize("transform", TRANSFORMS[2:])
def test_lossy_transforms_have_explicit_staging_and_float_decode(tone, transform):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    output, info = transform_audio(tone, transform)
    assert output.shape == tone.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert info["decode_format"] == "pcm_f32le"
    assert info["staging_subtype"] == ("PCM_16" if transform.startswith("pcm16_") else "FLOAT")


def test_bad_audio_is_not_fake_probability(tmp_path):
    path = tmp_path / "error.mp3"
    path.write_text("<!DOCTYPE html><html>download error</html>")
    with pytest.raises(ValueError, match="Markup"):
        inspect_audio(path)
    with pytest.raises(subprocess.CalledProcessError):
        load_audio_float(path)


def test_validator_reads_real_pcm(tone, tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, tone, 44100, subtype="FLOAT")
    info = inspect_audio(path)
    assert info["sample_rate"] == 44100
    assert info["channels"] == 1
    assert info["decoded_samples_22050_mono"] == 66150
    assert info["rms"] > 0
    np.testing.assert_array_equal(load_audio_float(path), tone)

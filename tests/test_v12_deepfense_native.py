import copy
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from artifactbench.v12.deepfense_native import load_native, native_decode, prepare_waveform, score_prepared
from artifactbench.v12.run_deepfense_native import check_original_condition, infer_entry, selected_entries


@pytest.mark.parametrize('sample_rate,frames,channels', [(16000, 17000, 1), (44100, 180000, 2),
                                                       (48000, 7000, 2), (96000, 384000, 1)])
def test_supported_native_waveform_is_bit_exact_with_actual_upstream(tmp_path, sample_rate, frames, channels):
    pytest.importorskip('deepfense')
    from deepfense.data.transforms.transforms import load_audio, pad_combined
    # 전처리 unit fixture이며 실제 detector/benchmark 표본이나 성능은 아니다.
    wave = 1.05*np.sin(np.arange(frames, dtype=np.float64)*.041)
    if channels == 2:
        wave = np.stack([wave, wave*.25+.03], axis=1)
    path = tmp_path/'unit-native.wav'
    sf.write(path, wave, sample_rate, subtype='DOUBLE')
    got, metadata = load_native(path)
    expected = pad_combined(load_audio(str(path), target_sr=16000, mono=True),
                            max_len=64000, random_pad=False, pad_type='repeat').astype(np.float32)
    np.testing.assert_array_equal(got, expected)
    assert metadata['native_sample_rate'] == sample_rate and metadata['native_channels'] == channels
    assert metadata['native_frames'] == frames and metadata['model_frames'] == 64000
    assert metadata['decoder'] == 'soundfile_native_f64'
    if channels == 1:
        assert metadata['peak'] > 1  # 원본에 없는 clip을 추가하지 않는다.


@pytest.mark.parametrize('codec,encoder,suffix', [('aac', 'aac', 'm4a'), ('opus', 'libopus', 'webm')])
def test_fallback_preserves_native_rate_and_channels_in_actual_ffmpeg(tmp_path, codec, encoder, suffix):
    frames, sample_rate = 9000, 48000
    wave = np.sin(np.arange(frames, dtype=np.float64)*.047)*.4
    source, encoded = tmp_path/'unit-stereo.wav', tmp_path/('unit-'+codec+'.'+suffix)
    sf.write(source, np.stack([wave, wave*.25], axis=1), sample_rate, subtype='DOUBLE')
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(source), '-c:a', encoder,
                    '-b:a', '128k', str(encoded)], check=True, capture_output=True, timeout=30)
    with pytest.raises(sf.LibsndfileError):
        sf.read(encoded)
    got, rate, metadata = native_decode(encoded)
    reference = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(encoded),
        '-map', '0:a:0', '-vn', '-c:a', 'pcm_f64le', '-f', 'f64le', 'pipe:1'],
        check=True, capture_output=True, timeout=30)
    expected = np.frombuffer(reference.stdout, dtype='<f8').reshape(-1, 2)
    np.testing.assert_array_equal(got, expected)
    assert rate == 48000 and got.shape[1] == 2 and got.dtype == np.float64
    assert metadata == {'decoder': 'ffmpeg_'+codec+'_native_f64', 'upstream_decoder_error_type': 'LibsndfileError'}


def test_decoder_extension_does_not_swallow_bad_files_or_change_policy(tmp_path, monkeypatch):
    path = tmp_path/'not-audio.mp3'
    path.write_text('unit non-audio input')
    def probe(command, **kwargs):
        assert command[0] == 'ffprobe'  # 허용되지 않은 codec에는 FFmpeg fallback도 실행하지 않는다.
        return SimpleNamespace(stdout=json.dumps({'streams': [{'codec_name': 'mp3', 'sample_rate': '44100', 'channels': 2}]}))
    monkeypatch.setattr('artifactbench.v12.deepfense_native.subprocess.run', probe)
    with pytest.raises(ValueError, match='AAC or Opus only'):
        native_decode(path)
    with pytest.raises(ValueError, match='regular local'):
        native_decode(tmp_path)


@pytest.mark.parametrize('value,rate', [(np.array([], np.float64), 16000), (np.array([np.nan]), 16000),
    (np.ones(10, np.float32), 16000), (np.ones(10, np.float64), 0), (np.ones(10, np.float64), True)])
def test_invalid_waveform_rejected_before_upstream_padding(value, rate):
    with pytest.raises(ValueError, match='float64 native'):
        prepare_waveform(value, rate)


def test_prepared_score_uses_raw_input_and_labelled_logits_without_resampling():
    class UnitModel(torch.nn.Module):
        def forward(self, audio):
            assert audio.shape == (1, 64000) and audio.dtype == torch.float32
            assert audio[0, 0] == np.float32(1.05)
            return {'logits': torch.tensor([[1., 2.]]), 'probs': torch.tensor([.999])}
    model = SimpleNamespace(model=UnitModel(), device='cpu', max_len=64000, input_sr=16000, spoof_idx=0)
    audio = np.full(64000, 1.05, np.float32)
    prob, logits = score_prepared(model, audio)
    assert prob == torch.softmax(torch.tensor([1., 2.]), -1)[0].item() and logits == [1., 2.]
    model.spoof_idx = 1
    with pytest.raises(ValueError, match='contract'):
        score_prepared(model, audio)


def test_reference_assets_sources_and_packages_cannot_change():
    prior = {'device': 'cuda', 'gpu': 'unit GPU', 'expected': 2, 'selected_ids': ['a', 'b'],
             'model_assets': {'weight': 'original'}, 'packages': [['torch', 'version']]}
    current = copy.deepcopy(prior) | {'device': 'cpu', 'gpu': None, 'expected': 1, 'selected_ids': ['a']}
    check_original_condition(current, prior)
    current['model_assets']['weight'] = 'changed'
    with pytest.raises(ValueError, match='model_assets'):
        check_original_condition(current, prior)


def test_input_failure_is_not_a_model_score(tmp_path, monkeypatch):
    from artifactbench.v12.common import digest
    path = tmp_path/'unit.wav'
    path.write_bytes(b'unit invalid audio')
    entry = {key: 'unit' for key in ('id', 'source', 'track_id', 'label', 'partition', 'recording_group', 'dependence_cluster')}
    entry.update(recording_representative=True, path=str(path), sha256=digest(path))
    def fail(path):
        raise ValueError('unit decode failure')
    monkeypatch.setattr('artifactbench.v12.run_deepfense_native.load_native', fail)
    result = infer_entry(None, entry)
    assert result['outcome'] == 'input_execution_error' and 'prob' not in result
    assert result['error_type'] == 'ValueError' and result['audio_sha256'] == entry['sha256']


def test_selection_checks_full_membership_before_source_smoke():
    with pytest.raises(ValueError, match='membership'):
        selected_entries([], {'unit': {}}, 1)

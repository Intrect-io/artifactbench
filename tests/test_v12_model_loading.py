from types import SimpleNamespace

import pytest
import numpy as np
import torch

from artifactbench.models.deepfense import DeepFenseModel
from artifactbench.models.fst import pinned_mert_loaders


def test_pinned_fst_dependency_is_local_scoped_and_restored_on_error():
    seen = []
    original = SimpleNamespace(from_pretrained=lambda *a, **kw: seen.append((a, kw)))
    module = SimpleNamespace(AutoConfig=original, AutoModel=original)
    with pytest.raises(ValueError, match='Unexpected pretrained'):
        with pinned_mert_loaders(module, '/fixed/snapshot'):
            module.AutoModel.from_pretrained('m-a-p/MERT-v1-95M', trust_remote_code=True)
            module.AutoConfig.from_pretrained('unexpected/model')
    assert module.AutoConfig is module.AutoModel is original
    assert seen == [(('/fixed/snapshot',), {'trust_remote_code': True, 'local_files_only': True})]


def test_deepfense_partial_local_configuration_is_rejected():
    with pytest.raises(ValueError, match='together'):
        DeepFenseModel(config_path='/config.yaml')


def test_deepfense_uses_two_class_logits_not_single_bonafide_score():
    model = DeepFenseModel()
    audio = np.ones(44100, dtype=np.float32)
    expected = torch.softmax(torch.tensor([6., 5.]), dim=-1)[0].item()
    for shift in (0., 100.):
        model.model = lambda x: {'logits': torch.tensor([[6.+shift, 5.+shift]]),
                                'scores': torch.tensor([5.+shift])}
        assert model.forward(audio) == pytest.approx(expected)
    model.model = lambda x: {'scores': torch.tensor([5.])}
    with pytest.raises(ValueError, match='two-class'):
        model.forward(audio)
def test_asset_location_can_move_without_changing_model_configuration(monkeypatch, tmp_path):
    from artifactbench.v12.model_assets import asset_location
    monkeypatch.delenv('ARTIFACTBENCH_FST_REPO', raising=False)
    assert asset_location('ARTIFACTBENCH_FST_REPO', str(tmp_path/'original')) == tmp_path/'original'
    monkeypatch.setenv('ARTIFACTBENCH_FST_REPO', str(tmp_path/'moved'))
    assert asset_location('ARTIFACTBENCH_FST_REPO', str(tmp_path/'original')) == tmp_path/'moved'


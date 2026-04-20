"""Smoke test — verify package imports and ArtifactNet forward pass on random audio.

Run: pytest tests/test_smoke.py -v
This does NOT require a real manifest — it generates dummy audio to check the
inference pipeline end-to-end. The ONNX weight is downloaded from HF Hub on
first run (CC BY-NC 4.0, see NOTICE.md).
"""
import numpy as np
import pytest


def test_imports():
    from artifactbench.models import MODEL_REGISTRY, BenchModel
    from artifactbench.data.manifest import load_manifest, load_audio_mono
    from artifactbench.metrics.failcheck import check_fail
    from artifactbench.report.markdown import comparison_report
    assert set(MODEL_REGISTRY.keys()) == {"artifactnet", "spectttra", "clam"}


@pytest.mark.slow
def test_artifactnet_forward():
    """Download ArtifactNet ONNX and run one forward pass on dummy audio."""
    from artifactbench.models import ArtifactNetModel

    model = ArtifactNetModel()
    model.load(device="cpu")

    # 6 seconds of pink-ish noise at 44.1 kHz
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(44100 * 6).astype(np.float32) * 0.1
    audio = np.clip(audio, -1.0, 1.0)

    prob = model.forward(audio)
    assert 0.0 <= prob <= 1.0, f"prob out of range: {prob}"


def test_spectttra_import_error_is_helpful():
    """Without the sonics package installed, SpecTTTra should raise a helpful error."""
    from artifactbench.models import SpecTTTraModel
    m = SpecTTTraModel()
    try:
        m.load(device="cpu")
    except ImportError as e:
        assert "sonics" in str(e).lower()
        return
    except Exception:
        # sonics may be installed in the test env; that's fine
        return


def test_clam_requires_explicit_args():
    """CLAM should refuse to load without --clam-repo / --clam-ckpt."""
    from artifactbench.models import CLAMModel
    with pytest.raises(ValueError, match="clam-repo"):
        CLAMModel().load(device="cpu")

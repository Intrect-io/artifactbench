"""Smoke test — verify package imports and ArtifactNet forward pass on random audio.

Run: pytest tests/test_smoke.py -v
This does NOT require a real manifest — it generates dummy audio to check the
inference pipeline end-to-end. The ONNX weight is downloaded from HF Hub on
first run (CC BY-NC 4.0, see NOTICE.md).
"""
import numpy as np
import pytest


def test_imports():
    import artifactbench
    from artifactbench.models import MODEL_REGISTRY, BenchModel

    assert artifactbench.__version__ == "0.2.0"
    assert set(MODEL_REGISTRY.keys()) == {
        "artifactnet", "spectttra", "clam", "spectttra_beta5s",
        "deezer_ismir", "fst", "ast_60s", "deepfense",
    }
    assert BenchModel is not None


def test_reports_identify_v2():
    from artifactbench.report.markdown import comparison_report, single_model_report

    model_info = {
        "name": "dummy",
        "params": 1,
        "input_sr": 44100,
        "input_duration": 1.0,
        "paper_ref": "none",
    }
    single = single_model_report(model_info, {}, [], [], 0.0)
    comparison = comparison_report(
        [
            {
                "model_info": model_info,
                "per_source": {},
                "codec_pairs": [],
                "fails": [],
                "elapsed": 0.0,
            }
        ]
    )

    assert single.startswith("# ArtifactBench v2")
    assert comparison.startswith("# ArtifactBench v2")


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

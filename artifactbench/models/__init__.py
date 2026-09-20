"""Model adapters for ArtifactBench."""
from .base import BenchModel
from .artifactnet import ArtifactNetModel
from .spectttra import SpecTTTraModel
from .spectttra_variants import SpecTTTraVariantModel
from .clam import CLAMModel
from .deezer_ismir import DeezerISMIRModel
from .fst import FSTModel
from .ast_60s import ASTMusicDetectionModel
from .deepfense import DeepFenseModel


def _spectttra_beta5s():
    return SpecTTTraVariantModel(
        hf_repo="awsaf49/sonics-spectttra-beta-5s", duration=5.0,
        params=18_679_693, display_name="SpecTTTra beta-5s")


MODEL_REGISTRY = {
    "artifactnet":       ArtifactNetModel,
    "spectttra":         SpecTTTraModel,
    "clam":              CLAMModel,
    "deezer_ismir":      DeezerISMIRModel,
    "fst":               FSTModel,
    "ast_60s":           ASTMusicDetectionModel,
    "spectttra_beta5s":  _spectttra_beta5s,
    "deepfense":         DeepFenseModel,
}

__all__ = ["BenchModel", "MODEL_REGISTRY",
           "ArtifactNetModel", "SpecTTTraModel", "SpecTTTraVariantModel", "CLAMModel",
           "DeezerISMIRModel", "FSTModel", "ASTMusicDetectionModel", "DeepFenseModel"]

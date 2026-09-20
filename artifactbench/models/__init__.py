"""Model adapters for ArtifactBench."""
from .artifactnet import ArtifactNetModel
from .base import BenchModel
from .clam import CLAMModel
from .deezer_ismir import DeezerISMIRModel
from .spectttra import SpecTTTraModel

MODEL_REGISTRY = {
    "artifactnet": ArtifactNetModel,
    "spectttra":   SpecTTTraModel,
    "clam":        CLAMModel,
    "deezer_ismir": DeezerISMIRModel,
}

__all__ = [
    "MODEL_REGISTRY",
    "ArtifactNetModel",
    "BenchModel",
    "CLAMModel",
    "DeezerISMIRModel",
    "SpecTTTraModel",
]

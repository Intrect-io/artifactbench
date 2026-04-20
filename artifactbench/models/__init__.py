"""Model adapters for ArtifactBench."""
from .base import BenchModel
from .artifactnet import ArtifactNetModel
from .spectttra import SpecTTTraModel
from .clam import CLAMModel

MODEL_REGISTRY = {
    "artifactnet": ArtifactNetModel,
    "spectttra":   SpecTTTraModel,
    "clam":        CLAMModel,
}

__all__ = ["BenchModel", "MODEL_REGISTRY",
           "ArtifactNetModel", "SpecTTTraModel", "CLAMModel"]

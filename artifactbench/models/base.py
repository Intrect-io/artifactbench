"""BenchModel — abstract adapter interface."""
from abc import ABC, abstractmethod

import numpy as np


class BenchModel(ABC):
    """Every benchmarked model implements this interface.

    `load()` puts weights on device; `forward()` takes 44.1 kHz mono float32
    and returns P(AI) in [0, 1].
    """

    name: str = "unnamed"
    params: int = 0
    input_sr: int = 44100
    input_duration: float = 0.0  # 0 = variable
    paper_ref: str = ""

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        ...

    @abstractmethod
    def forward(self, audio_44k: np.ndarray) -> float:
        ...

    def info(self) -> dict:
        return {
            "name": self.name,
            "params": self.params,
            "input_sr": self.input_sr,
            "input_duration": self.input_duration,
            "paper_ref": self.paper_ref,
        }

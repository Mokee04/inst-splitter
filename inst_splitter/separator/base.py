from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class SeparatorBackend(ABC):
    """Abstract base class for audio stem separation backends."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir
        self.current_model_name: Optional[str] = None

    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> None:
        """Load the specified separation model."""
        pass

    @abstractmethod
    def separate_instrumental(self, audio_pcm: np.ndarray, sr: int = 44100) -> np.ndarray:
        """
        Separate input stereo audio PCM (shape: (2, N)) and return the instrumental stem PCM (shape: (2, N)).
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload current model from memory to free VRAM/RAM."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()

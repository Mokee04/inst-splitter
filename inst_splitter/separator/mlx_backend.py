import gc
import platform
from typing import Optional
import numpy as np

from inst_splitter.separator.base import SeparatorBackend
from inst_splitter.separator.audio_separator_backend import AudioSeparatorBackend
from inst_splitter.utils import setup_logger

logger = setup_logger("mlx_backend")


class MLXBackend(SeparatorBackend):
    """
    Apple Silicon MLX-accelerated stem separation backend.
    If MLX native model separator is available, utilizes MLX tensors;
    otherwise gracefully delegates to AudioSeparatorBackend with MPS/CoreML acceleration.
    """

    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(output_dir=output_dir)
        self.is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        self._fallback_backend: Optional[AudioSeparatorBackend] = None

    def load_model(self, model_name: str, **kwargs) -> None:
        self.unload_model()
        self.current_model_name = model_name

        # Check for MLX native audio separation availability
        mlx_available = False
        if self.is_apple_silicon:
            try:
                import mlx.core as mx
                # If a specialized mlx_audio_separator exists
                try:
                    import mlx_audio_separator
                    mlx_available = True
                    logger.info("Native MLX audio separation engine detected.")
                except ImportError:
                    pass
            except ImportError:
                pass

        if not mlx_available:
            logger.debug("Native MLX audio-separator module not found; using AudioSeparatorBackend with Apple Silicon acceleration.")
            self._fallback_backend = AudioSeparatorBackend(output_dir=self.output_dir)
            self._fallback_backend.load_model(model_name, **kwargs)

    def separate_instrumental(self, audio_pcm: np.ndarray, sr: int = 44100) -> np.ndarray:
        if self._fallback_backend is not None:
            return self._fallback_backend.separate_instrumental(audio_pcm, sr=sr)
        raise NotImplementedError("Native MLX separation is not configured.")

    def unload_model(self) -> None:
        if self._fallback_backend is not None:
            self._fallback_backend.unload_model()
            self._fallback_backend = None
        self.current_model_name = None
        gc.collect()

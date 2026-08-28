import platform
from typing import Optional
from inst_splitter.separator.base import SeparatorBackend
from inst_splitter.separator.audio_separator_backend import AudioSeparatorBackend
from inst_splitter.separator.mlx_backend import MLXBackend


def get_separator_backend(output_dir: Optional[str] = None, backend_type: Optional[str] = None) -> SeparatorBackend:
    """
    Factory to get the appropriate separator backend.
    Auto-detects Apple Silicon or uses specified backend.
    """
    if backend_type == "mlx":
        return MLXBackend(output_dir=output_dir)
    elif backend_type == "audio_separator":
        return AudioSeparatorBackend(output_dir=output_dir)

    # Automatic selection
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return MLXBackend(output_dir=output_dir)
    return AudioSeparatorBackend(output_dir=output_dir)

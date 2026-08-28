import hashlib
import re
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from inst_splitter.audio_io import load_audio_as_stereo_pcm, save_24bit_wav
from inst_splitter.utils import setup_logger

logger = setup_logger("cache")


def sanitize_filename(name: str) -> str:
    """Sanitize string to be safe for filenames."""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)


def compute_file_hash(file_path: str | Path) -> str:
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class StemCache:
    """Manages cached separated stems to avoid redundant model inference."""

    def __init__(self, cache_dir: str | Path = ".cache", enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

    def get_cache_path(self, source_hash: str, model_name: str, sr: int = 44100) -> Path:
        safe_model = sanitize_filename(model_name)
        return self.cache_dir / source_hash / f"{safe_model}_{sr}hz.wav"

    def get_cached_stem(
        self,
        source_hash: str,
        model_name: str,
        sr: int = 44100
    ) -> Optional[np.ndarray]:
        if not self.enabled:
            return None

        cache_path = self.get_cache_path(source_hash, model_name, sr)
        if cache_path.exists():
            try:
                pcm, cached_sr = load_audio_as_stereo_pcm(cache_path, target_sr=sr)
                logger.debug(f"Cache hit: {cache_path}")
                return pcm
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_path}: {e}")
                return None
        return None

    def save_cached_stem(
        self,
        source_hash: str,
        model_name: str,
        audio_pcm: np.ndarray,
        sr: int = 44100
    ) -> Optional[Path]:
        if not self.enabled:
            return None

        cache_path = self.get_cache_path(source_hash, model_name, sr)
        try:
            save_24bit_wav(cache_path, audio_pcm, sr=sr)
            logger.debug(f"Saved stem to cache: {cache_path}")
            return cache_path
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_path}: {e}")
            return None

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import yaml


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    if config_path is None:
        # Look for config.yaml in current dir or project root
        current_dir = Path(__file__).resolve().parent.parent
        default_config = current_dir / "config.yaml"
        if default_config.exists():
            config_path = default_config
        else:
            config_path = Path("config.yaml")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def setup_logger(name: str = "inst_splitter", debug: bool = False) -> logging.Logger:
    """Setup standard console logger."""
    logger = logging.getLogger(name)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def validate_audio_pcm(pcm: np.ndarray, name: str = "audio") -> None:
    """
    Validate audio PCM array.
    Expects shape (2, N) stereo float32.
    """
    if not isinstance(pcm, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(pcm)}")

    if pcm.size == 0:
        raise ValueError(f"{name} is empty (0 samples).")

    if pcm.ndim != 2 or pcm.shape[0] != 2:
        raise ValueError(f"{name} must be stereo shape (2, N), got shape {pcm.shape}")

    if np.isnan(pcm).any():
        raise ValueError(f"{name} contains NaN values.")

    if np.isinf(pcm).any():
        raise ValueError(f"{name} contains Infinite values.")


def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude ratio."""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float, eps: float = 1e-12) -> float:
    """Convert linear amplitude ratio to decibels."""
    return 20.0 * np.log10(max(float(linear), eps))

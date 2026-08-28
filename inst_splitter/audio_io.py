import math
from pathlib import Path
from typing import Tuple
import numpy as np
import scipy.signal
import soundfile as sf

from inst_splitter.utils import validate_audio_pcm

SUPPORTED_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}


def resample_audio(audio_pcm: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio PCM of shape (channels, samples) using scipy.signal.resample_poly.
    """
    if orig_sr == target_sr:
        return audio_pcm

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    # Apply along the sample axis (axis=1)
    resampled = scipy.signal.resample_poly(audio_pcm, up, down, axis=1)
    return resampled.astype(np.float32)


def load_audio_as_stereo_pcm(
    file_path: str | Path,
    target_sr: int = 44100
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file (.flac, .wav, .mp3, etc.) and convert to standardized:
    - Sample rate: target_sr (default 44100 Hz)
    - Channels: Stereo (shape: (2, N))
    - Data type: float32 in [-1.0, 1.0]

    Returns:
        tuple of (audio_pcm_stereo, target_sr)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Read audio with soundfile (supports WAV, FLAC, OGG, and MP3 via libsndfile)
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # data has shape (samples, channels)

    if data.size == 0:
        raise ValueError(f"Audio file '{path}' contains no audio data.")

    # Transpose to (channels, samples)
    pcm = data.T  # shape: (channels, samples)

    num_channels = pcm.shape[0]
    if num_channels == 1:
        # Mono to stereo
        pcm = np.vstack([pcm, pcm])
    elif num_channels > 2:
        # Downmix or take first 2 channels
        pcm = pcm[:2, :]

    # Resample if necessary
    if sr != target_sr:
        pcm = resample_audio(pcm, sr, target_sr)

    pcm = pcm.astype(np.float32)
    validate_audio_pcm(pcm, name=f"Loaded audio from {path.name}")

    return pcm, target_sr


def save_24bit_wav(
    file_path: str | Path,
    audio_pcm: np.ndarray,
    sr: int = 44100
) -> Path:
    """
    Save stereo float32 PCM of shape (2, N) as a 24-bit PCM WAV file.
    """
    validate_audio_pcm(audio_pcm, name="Output audio")

    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # soundfile expects (samples, channels)
    data_to_write = audio_pcm.T

    sf.write(str(out_path), data_to_write, samplerate=sr, subtype="PCM_24")
    return out_path

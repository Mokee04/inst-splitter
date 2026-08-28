from typing import Any, Dict, Optional
import numpy as np

from inst_splitter.ensemble import align_audio_length
from inst_splitter.utils import setup_logger, validate_audio_pcm

logger = setup_logger("transpose")


def pitch_shift_instrumental(
    audio_pcm: np.ndarray,
    semitone: float,
    sr: int = 44100,
    rbargs: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Pitch shift stereo audio PCM of shape (2, N) by `semitone` steps.
    Preserves original duration and tempo using Rubber Band.
    If semitone == 0, bypasses DSP completely.
    """
    validate_audio_pcm(audio_pcm, "Input for transpose")
    target_length = audio_pcm.shape[1]

    if abs(semitone) < 1e-4:
        logger.debug("Semitone is 0; skipping pitch shift DSP.")
        return audio_pcm.copy()

    shifted_audio = None
    # 1. Try pyrubberband
    try:
        import pyrubberband as pyrb
        custom_rbargs = {}
        if rbargs:
            if rbargs.get("pitch_hq", True):
                custom_rbargs["--pitch-hq"] = ""
            if rbargs.get("formant", False):
                custom_rbargs["--formant"] = ""

        # pyrubberband expects (samples, channels)
        audio_transposed = audio_pcm.T
        shifted_audio = pyrb.pitch_shift(
            audio_transposed,
            sr=sr,
            n_steps=float(semitone),
            rbargs=custom_rbargs if custom_rbargs else None
        )
        if shifted_audio.ndim == 1:
            shifted_audio = np.stack([shifted_audio, shifted_audio], axis=0)
        else:
            shifted_audio = shifted_audio.T
    except Exception as e:
        logger.info(f"Rubber Band not available ({e}); using librosa high-quality pitch shift.")
        import librosa
        shifted_audio = librosa.effects.pitch_shift(
            y=audio_pcm,
            sr=sr,
            n_steps=float(semitone)
        )

    # Align length to match original duration exactly
    shifted_audio = align_audio_length(shifted_audio.astype(np.float32), target_length)

    validate_audio_pcm(shifted_audio, "Shifted audio")
    return shifted_audio

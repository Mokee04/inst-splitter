from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from inst_splitter.audio_io import save_24bit_wav
from inst_splitter.utils import db_to_linear, linear_to_db, setup_logger, validate_audio_pcm

logger = setup_logger("output")


def format_semitone_string(semitone: float) -> str:
    """
    Format semitone string for output filename:
    -5 -> "-5st"
    +2 -> "+2st"
    0  -> "0st"
    """
    if abs(semitone - round(semitone)) < 1e-4:
        # Integer semitone
        val = int(round(semitone))
        if val > 0:
            return f"+{val}st"
        elif val < 0:
            return f"{val}st"
        else:
            return "0st"
    else:
        # Float semitone
        if semitone > 0:
            return f"+{semitone:.1f}st"
        elif semitone < 0:
            return f"{semitone:.1f}st"
        else:
            return "0st"


def apply_peak_safety(
    audio_pcm: np.ndarray,
    peak_ceiling_db: float = -1.0
) -> Tuple[np.ndarray, float]:
    """
    Check waveform peak. If peak exceeds peak_ceiling_db, smoothly reduce total gain.
    Does NOT hard-clip or squash dynamic range.
    Returns: (gain_adjusted_pcm, applied_gain_reduction_db)
    """
    validate_audio_pcm(audio_pcm, "Output audio for peak safety")

    max_peak = float(np.max(np.abs(audio_pcm)))
    ceiling_linear = db_to_linear(peak_ceiling_db)

    if max_peak > ceiling_linear:
        scale_factor = ceiling_linear / max_peak
        adjusted_pcm = audio_pcm * scale_factor
        gain_reduction_db = linear_to_db(scale_factor)
        logger.info(
            f"Peak safety triggered: peak {linear_to_db(max_peak):.2f} dBFS exceeds "
            f"{peak_ceiling_db:.1f} dBFS ceiling. Applied gain reduction: {gain_reduction_db:.2f} dB."
        )
        return adjusted_pcm.astype(np.float32), gain_reduction_db

    return audio_pcm, 0.0


def apply_linked_peak_safety(
    mr_pcm: np.ndarray,
    vocals_pcm: np.ndarray,
    peak_ceiling_db: float = -1.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Apply Linked Peak Safety across both MR and Vocals stems.
    Computes a single common gain factor from the maximum peak of both signals,
    preserving exact mixture consistency (MR_out + Vocals_out = g * Mixture).
    Returns: (safe_mr_pcm, safe_vocals_pcm, applied_gain_reduction_db)
    """
    validate_audio_pcm(mr_pcm, "MR PCM for linked peak safety")
    validate_audio_pcm(vocals_pcm, "Vocals PCM for linked peak safety")

    peak_mr = float(np.max(np.abs(mr_pcm)))
    peak_vocals = float(np.max(np.abs(vocals_pcm)))
    max_peak = max(peak_mr, peak_vocals)

    ceiling_linear = db_to_linear(peak_ceiling_db)

    if max_peak > ceiling_linear:
        scale_factor = ceiling_linear / max_peak
        safe_mr = (mr_pcm * scale_factor).astype(np.float32)
        safe_vocals = (vocals_pcm * scale_factor).astype(np.float32)
        gain_reduction_db = linear_to_db(scale_factor)
        logger.info(
            f"Linked peak safety triggered: max peak {linear_to_db(max_peak):.2f} dBFS "
            f"(MR: {linear_to_db(peak_mr):.2f} dBFS, Vocals: {linear_to_db(peak_vocals):.2f} dBFS) "
            f"exceeds {peak_ceiling_db:.1f} dBFS ceiling. Applied common gain reduction: {gain_reduction_db:.2f} dB."
        )
        return safe_mr, safe_vocals, gain_reduction_db

    return mr_pcm.astype(np.float32), vocals_pcm.astype(np.float32), 0.0


def check_mixture_consistency(
    mixture_pcm: np.ndarray,
    mr_pcm: np.ndarray,
    vocals_pcm: np.ndarray,
    tolerance: float = 1e-4
) -> float:
    """
    Verify software invariant: mixture = mr + vocals in float PCM precision.
    Returns maximum absolute error.
    """
    reconstructed = mr_pcm + vocals_pcm
    error = float(np.max(np.abs(mixture_pcm - reconstructed)))
    if error > tolerance:
        logger.warning(
            f"Mixture consistency check warning: max error {error:.2e} exceeds tolerance {tolerance:.2e}."
        )
    else:
        logger.debug(f"Mixture consistency verified: max error {error:.2e} <= {tolerance:.2e}")
    return error


def export_stem_wav(
    audio_pcm: np.ndarray,
    input_file_path: str | Path,
    semitone: float,
    stem_name: str = "MR",
    output_dir: Optional[str | Path] = None,
    sr: int = 44100
) -> Path:
    """
    Export 24-bit PCM WAV file for a given stem (e.g. 'MR' or 'Vocals').
    Output path: <original_dir>/MR_output/{stem}_{stem_name}_{semitone}st.wav
    """
    input_path = Path(input_file_path)
    if output_dir is None:
        target_dir = input_path.parent / "MR_output"
    else:
        target_dir = Path(output_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    semitone_str = format_semitone_string(semitone)
    output_filename = f"{input_path.stem}_{stem_name}_{semitone_str}.wav"
    output_file_path = target_dir / output_filename

    # Save 24-bit WAV
    saved_path = save_24bit_wav(output_file_path, audio_pcm, sr=sr)
    return saved_path


def export_mr_wav(
    audio_pcm: np.ndarray,
    input_file_path: str | Path,
    semitone: float,
    output_dir: Optional[str | Path] = None,
    sr: int = 44100,
    peak_ceiling_db: float = -1.0
) -> Path:
    """
    Apply peak safety and export final 24-bit PCM WAV file for MR (Backward compatibility).
    """
    safe_pcm, _ = apply_peak_safety(audio_pcm, peak_ceiling_db=peak_ceiling_db)
    return export_stem_wav(
        audio_pcm=safe_pcm,
        input_file_path=input_file_path,
        semitone=semitone,
        stem_name="MR",
        output_dir=output_dir,
        sr=sr,
    )

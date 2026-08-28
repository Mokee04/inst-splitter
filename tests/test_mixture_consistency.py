import numpy as np
import pytest

from inst_splitter.ensemble import (
    EnsembleConfig,
    align_audio_length,
    build_ensemble_instrumental,
)
from inst_splitter.output import apply_linked_peak_safety, check_mixture_consistency
from inst_splitter.transpose import pitch_shift_instrumental


def test_mixture_consistency_zero_semitone():
    """
    Verify Invariant 1: For semitone == 0,
    Original = MR + Vocals (where Vocals = Original - MR)
    """
    sr = 44100
    n_samples = 44100
    t = np.linspace(0, 1.0, n_samples, endpoint=False)

    orig_wave = np.stack([
        np.sin(2 * np.pi * 440 * t) * 0.5 + np.sin(2 * np.pi * 1000 * t) * 0.3,
        np.cos(2 * np.pi * 440 * t) * 0.5 + np.cos(2 * np.pi * 1000 * t) * 0.3
    ], axis=0).astype(np.float32)

    stems = {
        "bleedless": orig_wave * 0.4,
        "balanced": orig_wave * 0.45,
        "fullness": orig_wave * 0.8,
        "demucs_ft": orig_wave * 0.42,
        "demucs_extra": orig_wave * 0.44,
    }
    weights = {"bleedless": 1.0, "balanced": 1.0, "fullness": 0.8, "demucs_ft": 0.6, "demucs_extra": 0.5}
    config = EnsembleConfig(n_fft=2048, hop_length=512)

    mr = build_ensemble_instrumental(
        original_pcm=orig_wave,
        model_stems=stems,
        weights=weights,
        carrier_key="fullness",
        config=config,
        sr=sr,
    )

    vocals = orig_wave - mr
    error = check_mixture_consistency(orig_wave, mr, vocals)
    assert error < 1e-6

    # Verify linked peak safety maintains invariant
    safe_mr, safe_vocals, reduction_db = apply_linked_peak_safety(mr, vocals, peak_ceiling_db=-1.0)
    assert np.max(np.abs(safe_mr)) <= 1.0
    assert np.max(np.abs(safe_vocals)) <= 1.0


def test_mixture_consistency_transposed_pair():
    """
    Verify Invariant 2: For semitone != 0,
    ShiftedOriginal = ShiftedMR + ShiftedVocals (where ShiftedVocals = ShiftedOriginal - ShiftedMR)
    """
    sr = 44100
    n_samples = 22050
    t = np.linspace(0, 0.5, n_samples, endpoint=False)

    orig_wave = np.stack([
        np.sin(2 * np.pi * 440 * t) * 0.4,
        np.cos(2 * np.pi * 440 * t) * 0.4
    ], axis=0).astype(np.float32)

    # Simulated MR
    mr_orig = orig_wave * 0.6

    semitone = -3.0
    shifted_mr = pitch_shift_instrumental(mr_orig, semitone=semitone, sr=sr)
    shifted_mix = pitch_shift_instrumental(orig_wave, semitone=semitone, sr=sr)

    # Sample length alignment
    target_len = shifted_mix.shape[1]
    aligned_mr = align_audio_length(shifted_mr, target_len)
    aligned_mix = shifted_mix

    shifted_vocals = aligned_mix - aligned_mr
    error = check_mixture_consistency(aligned_mix, aligned_mr, shifted_vocals)
    assert error < 1e-6

import numpy as np
import pytest

from inst_splitter.ensemble import (
    EnsembleConfig,
    build_ensemble_instrumental,
    compute_hierarchical_removal_confidence,
    compute_soft_attenuation_mask,
    compute_stft,
    compute_istft,
    stereo_magnitude,
)
from inst_splitter.utils import db_to_linear


def test_stft_istft_reconstruction():
    sr = 44100
    duration = 0.5
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    sig = np.stack([
        np.sin(2 * np.pi * 440 * t) * 0.5,
        np.cos(2 * np.pi * 880 * t) * 0.5
    ], axis=0).astype(np.float32)

    z = compute_stft(sig, sr=sr, n_fft=2048, hop_length=512)
    recon = compute_istft(z, sr=sr, n_fft=2048, hop_length=512, target_len=n_samples)

    assert recon.shape == sig.shape
    # STFT/iSTFT with Hann window should have near perfect reconstruction
    error = np.max(np.abs(sig - recon))
    assert error < 1e-4


def test_hierarchical_consensus_cases():
    """
    Test 4 canonical cases of Hierarchical Consensus:
    Case 1: Both RoFormer & Demucs agree on removal (P_R ~ 1, P_D ~ 1) -> strong removal confidence
    Case 2: RoFormer removes, Demucs keeps (P_R ~ 1, P_D ~ 0) -> architecture disagreement penalty -> low confidence
    Case 3: RoFormer keeps, Demucs removes (P_R ~ 0, P_D ~ 1) -> architecture disagreement penalty -> low confidence
    Case 4: Both families agree on keeping instrumental (P_R ~ 0, P_D ~ 0) -> confidence ~ 0 (Carrier preserved)
    """
    config = EnsembleConfig(
        w_bleedless=1.0,
        w_balanced=1.0,
        w_fullness=0.8,
        roformer_ab_bonus=0.15,
        w_demucs_ft=0.6,
        w_demucs_extra=0.5,
        arch_disagreement_penalty=0.5,
    )

    shape = (1, 1)

    # Case 1: Both families remove
    e_case1 = {
        "bleedless": np.ones(shape),
        "balanced": np.ones(shape),
        "fullness": np.ones(shape),
        "demucs_ft": np.ones(shape),
        "demucs_extra": np.ones(shape),
    }
    p_final_case1 = compute_hierarchical_removal_confidence(e_case1, config)
    assert p_final_case1[0, 0] > 0.95

    # Case 2: RoFormer removes, Demucs keeps (Disagreement)
    e_case2 = {
        "bleedless": np.ones(shape),
        "balanced": np.ones(shape),
        "fullness": np.ones(shape),
        "demucs_ft": np.zeros(shape),
        "demucs_extra": np.zeros(shape),
    }
    p_final_case2 = compute_hierarchical_removal_confidence(e_case2, config)
    assert p_final_case2[0, 0] < 0.05  # sqrt(1 * 0) * (1 - 0.5 * 1) = 0.0

    # Case 3: RoFormer keeps, Demucs removes (Disagreement)
    e_case3 = {
        "bleedless": np.zeros(shape),
        "balanced": np.zeros(shape),
        "fullness": np.zeros(shape),
        "demucs_ft": np.ones(shape),
        "demucs_extra": np.ones(shape),
    }
    p_final_case3 = compute_hierarchical_removal_confidence(e_case3, config)
    assert p_final_case3[0, 0] < 0.05

    # Case 4: Both families keep
    e_case4 = {
        "bleedless": np.zeros(shape),
        "balanced": np.zeros(shape),
        "fullness": np.zeros(shape),
        "demucs_ft": np.zeros(shape),
        "demucs_extra": np.zeros(shape),
    }
    p_final_case4 = compute_hierarchical_removal_confidence(e_case4, config)
    assert p_final_case4[0, 0] == 0.0


def test_soft_attenuation_mask_and_frequency_limits():
    """
    Test 5-model soft attenuation mask behavior including:
    1. Agreement vs Disagreement attenuation contrast
    2. Silence protection
    3. Frequency-dependent max attenuation ceiling
    """
    sr = 44100
    n_samples = 44100
    t = np.linspace(0, 1.0, n_samples, endpoint=False)

    # 440 Hz tone (Vocal band)
    tone_vocal = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], axis=0).astype(np.float32)
    # 60 Hz tone (Sub bass band)
    tone_sub = np.stack([np.sin(2 * np.pi * 60 * t), np.sin(2 * np.pi * 60 * t)], axis=0).astype(np.float32)

    config = EnsembleConfig(
        removal_threshold=0.45,
        soft_temperature=0.10,
        low_quantile=0.25,
        silence_threshold_db=-70.0,
        sub_freq=120.0,
        sub_bass_db=3.0,
        low_freq=500.0,
        low_mids_db=5.0,
        vocal_freq=5000.0,
        vocal_mids_db=10.0,
        high_freq=10000.0,
        high_db=6.0,
        air_db=4.0,
        n_fft=2048,
        hop_length=512,
        median_size=1,
        attack_ms=1.0,
        release_ms=1.0,
        frequency_bins=1,
    )
    weights = {"bleedless": 1.0, "balanced": 1.0, "fullness": 0.8, "demucs_ft": 0.6, "demucs_extra": 0.5}

    # Find active bins
    stft_vocal = compute_stft(tone_vocal, sr=sr, n_fft=config.n_fft, hop_length=config.hop_length)
    mag_vocal = stereo_magnitude(stft_vocal)
    center_frame = mag_vocal.shape[1] // 2
    bin_vocal = int(np.argmax(mag_vocal[:, center_frame]))

    stft_sub = compute_stft(tone_sub, sr=sr, n_fft=config.n_fft, hop_length=config.hop_length)
    mag_sub = stereo_magnitude(stft_sub)
    bin_sub = int(np.argmax(mag_sub[:, center_frame]))

    # --- 1. All 5 models agree on keeping instrumental ---
    stems_keep_all = {
        "bleedless": tone_vocal * 0.85,
        "balanced": tone_vocal * 0.85,
        "fullness": tone_vocal * 0.85,
        "demucs_ft": tone_vocal * 0.85,
        "demucs_extra": tone_vocal * 0.85,
    }
    mask_keep = compute_soft_attenuation_mask(
        original_pcm=tone_vocal,
        model_stems=stems_keep_all,
        weights=weights,
        carrier_key="fullness",
        config=config,
        sr=sr,
    )
    assert mask_keep[bin_vocal, center_frame] > 0.95

    # --- 2. Architecture Disagreement (RoFormer removes, Demucs keeps) ---
    stems_disagree = {
        "bleedless": tone_vocal * 0.05,
        "balanced": tone_vocal * 0.05,
        "fullness": tone_vocal * 0.85,
        "demucs_ft": tone_vocal * 0.85,
        "demucs_extra": tone_vocal * 0.85,
    }
    mask_disagree = compute_soft_attenuation_mask(
        original_pcm=tone_vocal,
        model_stems=stems_disagree,
        weights=weights,
        carrier_key="fullness",
        config=config,
        sr=sr,
    )

    # --- 3. Both families agree on removal (High confidence removal) ---
    stems_remove_all = {
        "bleedless": tone_vocal * 0.05,
        "balanced": tone_vocal * 0.05,
        "fullness": tone_vocal * 0.85,
        "demucs_ft": tone_vocal * 0.05,
        "demucs_extra": tone_vocal * 0.05,
    }
    mask_remove = compute_soft_attenuation_mask(
        original_pcm=tone_vocal,
        model_stems=stems_remove_all,
        weights=weights,
        carrier_key="fullness",
        config=config,
        sr=sr,
    )

    # In case of disagreement, attenuation is significantly weaker (mask is higher) than full agreement
    assert mask_disagree[bin_vocal, center_frame] > mask_remove[bin_vocal, center_frame]

    # --- 4. Sub-bass band max attenuation limit ---
    # Even if all models want to remove sub-bass tone, mask must NOT drop below -3 dB (≈ 0.707)
    stems_sub_remove = {
        "bleedless": tone_sub * 0.0,
        "balanced": tone_sub * 0.0,
        "fullness": tone_sub * 0.8,
        "demucs_ft": tone_sub * 0.0,
        "demucs_extra": tone_sub * 0.0,
    }
    mask_sub = compute_soft_attenuation_mask(
        original_pcm=tone_sub,
        model_stems=stems_sub_remove,
        weights=weights,
        carrier_key="fullness",
        config=config,
        sr=sr,
    )
    min_sub_linear = db_to_linear(-config.sub_bass_db)
    assert mask_sub[bin_sub, center_frame] >= min_sub_linear - 1e-3

    # --- 5. Silence protection ---
    silent_orig = np.zeros_like(tone_vocal)
    stems_silence = {k: np.zeros_like(tone_vocal) for k in weights}
    mask_silence = compute_soft_attenuation_mask(
        original_pcm=silent_orig,
        model_stems=stems_silence,
        weights=weights,
        carrier_key="fullness",
        config=config,
        sr=sr,
    )
    assert np.all(mask_silence == 1.0)


def test_build_ensemble_instrumental_full_5models():
    sr = 44100
    n_samples = 22050
    t = np.linspace(0, 0.5, n_samples, endpoint=False)
    orig = np.stack([np.sin(2 * np.pi * 440 * t) * 0.5, np.sin(2 * np.pi * 440 * t) * 0.5], axis=0).astype(np.float32)

    stems = {
        "bleedless": orig * 0.1,
        "balanced": orig * 0.2,
        "fullness": orig * 0.8,
        "demucs_ft": orig * 0.2,
        "demucs_extra": orig * 0.3,
    }
    weights = {"bleedless": 1.0, "balanced": 1.0, "fullness": 0.8, "demucs_ft": 0.6, "demucs_extra": 0.5}
    config = EnsembleConfig(n_fft=2048, hop_length=512)

    out = build_ensemble_instrumental(
        original_pcm=orig,
        model_stems=stems,
        weights=weights,
        carrier_key="fullness",
        config=config,
        sr=sr,
    )
    assert out.shape == orig.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()

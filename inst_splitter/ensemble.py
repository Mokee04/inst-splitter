from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import scipy.ndimage
import scipy.signal

from inst_splitter.utils import db_to_linear, validate_audio_pcm


@dataclass
class EnsembleConfig:
    removal_threshold: float = 0.45
    soft_temperature: float = 0.10
    low_quantile: float = 0.25
    silence_threshold_db: float = -70.0

    # RoFormer family weights
    w_bleedless: float = 1.0
    w_balanced: float = 1.0
    w_fullness: float = 0.8
    roformer_ab_bonus: float = 0.15

    # Demucs family weights
    w_demucs_ft: float = 0.6
    w_demucs_extra: float = 0.5

    # Architecture disagreement penalty
    arch_disagreement_penalty: float = 0.5

    # Frequency-dependent max attenuation limits (dB)
    sub_freq: float = 120.0
    sub_bass_db: float = 3.0       # < 120 Hz
    low_freq: float = 500.0
    low_mids_db: float = 5.0       # 120 - 500 Hz
    vocal_freq: float = 5000.0
    vocal_mids_db: float = 10.0    # 500 - 5000 Hz
    high_freq: float = 10000.0
    high_db: float = 6.0           # 5000 - 10000 Hz
    air_db: float = 4.0            # > 10000 Hz

    # STFT parameters
    n_fft: int = 4096
    hop_length: int = 1024
    window: str = "hann"

    # Smoothing parameters
    median_size: int = 3
    attack_ms: float = 40.0
    release_ms: float = 100.0
    frequency_bins: int = 3


def compute_stft(
    signal_2ch: np.ndarray,
    sr: int = 44100,
    n_fft: int = 4096,
    hop_length: int = 1024,
    window: str = "hann"
) -> np.ndarray:
    """
    Compute STFT for 2-channel stereo audio.
    signal_2ch shape: (2, N)
    Returns: complex array of shape (2, n_freqs, n_frames)
    """
    noverlap = n_fft - hop_length
    _, _, z_left = scipy.signal.stft(
        signal_2ch[0],
        fs=sr,
        window=window,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    _, _, z_right = scipy.signal.stft(
        signal_2ch[1],
        fs=sr,
        window=window,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    return np.stack([z_left, z_right], axis=0)


def compute_istft(
    z_2ch: np.ndarray,
    sr: int = 44100,
    n_fft: int = 4096,
    hop_length: int = 1024,
    window: str = "hann",
    target_len: Optional[int] = None
) -> np.ndarray:
    """
    Compute inverse STFT for 2-channel complex spectrogram.
    z_2ch shape: (2, n_freqs, n_frames)
    Returns: stereo float32 PCM of shape (2, target_len)
    """
    noverlap = n_fft - hop_length
    _, x_left = scipy.signal.istft(
        z_2ch[0],
        fs=sr,
        window=window,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=True,
    )
    _, x_right = scipy.signal.istft(
        z_2ch[1],
        fs=sr,
        window=window,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=True,
    )

    reconstructed = np.stack([x_left, x_right], axis=0).astype(np.float32)

    if target_len is not None:
        if reconstructed.shape[1] > target_len:
            reconstructed = reconstructed[:, :target_len]
        elif reconstructed.shape[1] < target_len:
            pad_width = target_len - reconstructed.shape[1]
            reconstructed = np.pad(reconstructed, ((0, 0), (0, pad_width)), mode="constant")

    return reconstructed


def stereo_magnitude(z_2ch: np.ndarray) -> np.ndarray:
    """
    Compute combined stereo magnitude:
    A(f,t) = sqrt((|L(f,t)|^2 + |R(f,t)|^2) / 2)
    Returns: shape (n_freqs, n_frames)
    """
    mag_l_sq = np.abs(z_2ch[0]) ** 2
    mag_r_sq = np.abs(z_2ch[1]) ** 2
    return np.sqrt((mag_l_sq + mag_r_sq) / 2.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def align_audio_length(
    audio_pcm: np.ndarray,
    target_length: int
) -> np.ndarray:
    """Align audio length by trimming or zero-padding."""
    cur_len = audio_pcm.shape[1]
    if cur_len == target_length:
        return audio_pcm
    elif cur_len > target_length:
        return audio_pcm[:, :target_length]
    else:
        pad_width = target_length - cur_len
        return np.pad(audio_pcm, ((0, 0), (0, pad_width)), mode="constant")


def compute_frequency_dependent_min_gain(
    sr: int,
    n_fft: int,
    config: EnsembleConfig
) -> np.ndarray:
    """
    Compute smooth frequency-dependent minimum gain curve min_gain(f).
    - < sub_freq (120 Hz): sub_bass_db (3.0 dB)
    - sub_freq - low_freq (120 - 500 Hz): low_mids_db (5.0 dB)
    - low_freq - vocal_freq (500 - 5000 Hz): vocal_mids_db (10.0 dB)
    - vocal_freq - high_freq (5000 - 10000 Hz): high_db (6.0 dB)
    - > high_freq (10000 Hz): air_db (4.0 dB)
    Returns shape: (n_freqs, 1)
    """
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Control points (Hz, max_attenuation_db)
    xp = [0.0, config.sub_freq, config.low_freq, config.vocal_freq, config.high_freq, sr / 2.0]
    fp = [
        config.sub_bass_db,
        config.sub_bass_db,
        config.low_mids_db,
        config.vocal_mids_db,
        config.high_db,
        config.air_db,
    ]

    atten_db = np.interp(freqs, xp, fp)
    min_gain = db_to_linear(-atten_db)
    return min_gain[:, None]  # shape: (n_freqs, 1)


def apply_asymmetric_envelope_smoothing(
    mask: np.ndarray,
    sr: int,
    hop_length: int,
    attack_ms: float,
    release_ms: float,
    freq_bins: int = 3
) -> np.ndarray:
    """
    Apply asymmetric Attack / Release temporal envelope smoothing and frequency smoothing.
    - When mask drops (Vocal onset / Attack): Fast attack time constant
    - When mask rises (Vocal release / Recovery): Slow release time constant (eliminates chirping)
    """
    n_freqs, n_frames = mask.shape
    dt = hop_length / sr

    # Time constants
    alpha_attack = 1.0 - np.exp(-dt / max(attack_ms / 1000.0, 1e-4))
    alpha_release = 1.0 - np.exp(-dt / max(release_ms / 1000.0, 1e-4))

    # 1-pole recursive filter across time
    smoothed = np.zeros_like(mask)
    smoothed[:, 0] = mask[:, 0]

    for t in range(1, n_frames):
        target = mask[:, t]
        prev = smoothed[:, t - 1]
        is_attack = target < prev
        smoothed[:, t] = np.where(
            is_attack,
            prev + alpha_attack * (target - prev),
            prev + alpha_release * (target - prev)
        )

    # Frequency-axis smoothing
    if freq_bins > 1:
        sigma_freq = max(freq_bins / 2.0, 0.5)
        smoothed = scipy.ndimage.gaussian_filter1d(smoothed, sigma=sigma_freq, axis=0, mode="nearest")

    return smoothed


def compute_hierarchical_removal_confidence(
    e_dict: Dict[str, np.ndarray],
    config: EnsembleConfig
) -> np.ndarray:
    """
    Compute Hierarchical Consensus removal confidence (P_final):
    1. RoFormer Family Consensus (P_R)
    2. Demucs Family Consensus (P_D)
    3. Family Agreement (P_consensus = sqrt(P_R * P_D))
    4. Architecture Disagreement (D_arch = |P_R - P_D|)
    5. Final Removal Confidence (P_final = P_consensus * (1 - beta * D_arch))
    """
    sample_arr = next(iter(e_dict.values()))
    zeros_like_sample = np.zeros_like(sample_arr)

    # RoFormer Family
    has_roformer = False
    p_r_weighted_sum = np.zeros_like(sample_arr)
    w_r_total = 0.0

    if "bleedless" in e_dict:
        p_r_weighted_sum += config.w_bleedless * e_dict["bleedless"]
        w_r_total += config.w_bleedless
        has_roformer = True

    if "balanced" in e_dict:
        p_r_weighted_sum += config.w_balanced * e_dict["balanced"]
        w_r_total += config.w_balanced
        has_roformer = True

    if "fullness" in e_dict:
        p_r_weighted_sum += config.w_fullness * e_dict["fullness"]
        w_r_total += config.w_fullness
        has_roformer = True

    if has_roformer and w_r_total > 0:
        p_r = p_r_weighted_sum / w_r_total
        # Synergistic consensus bonus between Bleedless and Balanced
        if "bleedless" in e_dict and "balanced" in e_dict:
            p_r += config.roformer_ab_bonus * (e_dict["bleedless"] * e_dict["balanced"])
        p_r = np.clip(p_r, 0.0, 1.0)
    else:
        p_r = zeros_like_sample

    # Demucs Family
    has_demucs = False
    p_d_weighted_sum = np.zeros_like(sample_arr)
    w_d_total = 0.0

    if "demucs_ft" in e_dict:
        p_d_weighted_sum += config.w_demucs_ft * e_dict["demucs_ft"]
        w_d_total += config.w_demucs_ft
        has_demucs = True

    if "demucs_extra" in e_dict:
        p_d_weighted_sum += config.w_demucs_extra * e_dict["demucs_extra"]
        w_d_total += config.w_demucs_extra
        has_demucs = True

    if has_demucs and w_d_total > 0:
        p_d = p_d_weighted_sum / w_d_total
        p_d = np.clip(p_d, 0.0, 1.0)
    else:
        p_d = zeros_like_sample

    # Combine Families
    if has_roformer and has_demucs:
        # Both families present: compute consensus and architecture disagreement penalty
        p_consensus = np.sqrt(p_r * p_d)
        d_arch = np.abs(p_r - p_d)
        penalty_factor = np.clip(1.0 - config.arch_disagreement_penalty * d_arch, 0.0, 1.0)
        p_final = p_consensus * penalty_factor
    elif has_roformer:
        # Only RoFormer family present
        p_final = p_r
    elif has_demucs:
        # Only Demucs family present
        p_final = p_d
    else:
        p_final = zeros_like_sample

    return np.clip(p_final, 0.0, 1.0)


def compute_soft_attenuation_mask(
    original_pcm: np.ndarray,
    model_stems: Dict[str, np.ndarray],
    weights: Dict[str, float],
    carrier_key: str,
    config: EnsembleConfig,
    sr: int = 44100,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Compute smoothed soft attenuation mask M(f,t) using:
    1. Model-wise retention ratios r_i and soft removal scores e_i
    2. Hierarchical consensus removal confidence P_final (RoFormer vs Demucs)
    3. Soft target retention r_target interpolating Carrier retention and lower quantile
    4. Frequency-dependent max attenuation ceiling min_gain(f)
    5. Silence protection (-70 dB relative ceiling)
    6. 2D Median + Attack/Release asymmetric temporal envelope & frequency smoothing
    """
    target_length = original_pcm.shape[1]

    # Align all stems to original length
    aligned_stems = {k: align_audio_length(v, target_length) for k, v in model_stems.items()}

    # Compute STFTs
    stft_orig = compute_stft(original_pcm, sr=sr, n_fft=config.n_fft, hop_length=config.hop_length, window=config.window)
    mag_orig = stereo_magnitude(stft_orig)  # (F, T)

    model_keys = list(aligned_stems.keys())
    mag_stems = {}
    for k in model_keys:
        stft_k = compute_stft(aligned_stems[k], sr=sr, n_fft=config.n_fft, hop_length=config.hop_length, window=config.window)
        mag_stems[k] = stereo_magnitude(stft_k)

    # 1. Retention ratios: r_i = |I_i| / (|X| + eps)
    ratios = {}
    r_list = []
    for k in model_keys:
        r_i = np.clip(mag_stems[k] / (mag_orig + eps), 0.0, 1.2)
        ratios[k] = r_i
        r_list.append(r_i)

    r_stack = np.stack(r_list, axis=0)  # (K, F, T)

    # Carrier retention
    r_base = ratios.get(carrier_key, r_stack[0])

    # 2. Silence mask (Original relative dB)
    peak_orig = np.max(mag_orig)
    if peak_orig > 0:
        rel_db = 20.0 * np.log10(mag_orig / peak_orig + 1e-12)
        is_silence = rel_db < config.silence_threshold_db
    else:
        is_silence = np.ones_like(mag_orig, dtype=bool)

    # 3. Soft removal scores e_i
    e_dict = {}
    for k, r_val in ratios.items():
        e_dict[k] = sigmoid((config.removal_threshold - r_val) / config.soft_temperature)

    # 4. Hierarchical consensus removal confidence P_final
    p_final = compute_hierarchical_removal_confidence(e_dict, config)

    # 5. Target retention based on P_final and lower quantile
    low_pct = config.low_quantile * 100.0
    r_low = np.percentile(r_stack, low_pct, axis=0)
    r_target = (1.0 - p_final) * r_base + p_final * r_low

    # 6. Raw attenuation mask M_raw
    raw_mask = r_target / (r_base + eps)

    # 7. Frequency-dependent minimum gain ceiling
    min_gain_curve = compute_frequency_dependent_min_gain(sr=sr, n_fft=config.n_fft, config=config)
    bounded_mask = np.clip(raw_mask, min_gain_curve, 1.0)
    bounded_mask[is_silence] = 1.0

    # 8. 2D Median Filtering
    if config.median_size > 1:
        bounded_mask = scipy.ndimage.median_filter(bounded_mask, size=config.median_size)

    # 9. Asymmetric Attack/Release Temporal + Frequency Smoothing
    smoothed_mask = apply_asymmetric_envelope_smoothing(
        mask=bounded_mask,
        sr=sr,
        hop_length=config.hop_length,
        attack_ms=config.attack_ms,
        release_ms=config.release_ms,
        freq_bins=config.frequency_bins,
    )

    # Final bounding and silence protection
    final_mask = np.clip(smoothed_mask, min_gain_curve, 1.0)
    final_mask[is_silence] = 1.0

    return final_mask


def build_ensemble_instrumental(
    original_pcm: np.ndarray,
    model_stems: Dict[str, np.ndarray],
    weights: Dict[str, float],
    carrier_key: str,
    config: EnsembleConfig,
    sr: int = 44100
) -> np.ndarray:
    """
    Build final instrumental waveform from models using Hierarchical Consensus soft attenuation
    applied to the carrier model's complex STFT.
    """
    validate_audio_pcm(original_pcm, "Original PCM")
    for k, v in model_stems.items():
        validate_audio_pcm(v, f"Stem {k}")

    target_length = original_pcm.shape[1]

    if carrier_key not in model_stems:
        carrier_key = list(model_stems.keys())[0]

    # Compute soft attenuation mask
    mask = compute_soft_attenuation_mask(
        original_pcm=original_pcm,
        model_stems=model_stems,
        weights=weights,
        carrier_key=carrier_key,
        config=config,
        sr=sr,
    )  # (F, T)

    # Compute carrier complex STFT
    carrier_pcm = align_audio_length(model_stems[carrier_key], target_length)
    stft_carrier = compute_stft(
        carrier_pcm,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        window=config.window
    )  # (2, F, T)

    # Apply mask to complex STFT for both L and R channels
    stft_final = np.zeros_like(stft_carrier)
    stft_final[0] = stft_carrier[0] * mask
    stft_final[1] = stft_carrier[1] * mask

    # Inverse STFT
    reconstructed_pcm = compute_istft(
        stft_final,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        window=config.window,
        target_len=target_length,
    )

    validate_audio_pcm(reconstructed_pcm, "Ensemble Instrumental")
    return reconstructed_pcm

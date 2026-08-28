from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from inst_splitter.audio_io import load_audio_as_stereo_pcm
from inst_splitter.cache import StemCache, compute_file_hash
from inst_splitter.ensemble import EnsembleConfig, align_audio_length, build_ensemble_instrumental
from inst_splitter.output import (
    apply_linked_peak_safety,
    check_mixture_consistency,
    export_stem_wav,
)
from inst_splitter.separator import get_separator_backend
from inst_splitter.transpose import pitch_shift_instrumental
from inst_splitter.utils import load_config, setup_logger

logger = setup_logger("pipeline")


@dataclass
class PipelineResult:
    """Dataclass holding output paths for generated MR and complementary Vocals."""
    mr_path: Path
    vocals_path: Path


def run_pipeline(
    input_path: str | Path,
    semitone: float,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str | Path] = None,
    force: bool = False,
    debug: bool = False
) -> PipelineResult:
    """
    Execute the 9-step MR + Complementary Vocals generation pipeline (v3.1).
    Returns PipelineResult(mr_path=..., vocals_path=...).
    """
    if config is None:
        config = load_config()

    input_file = Path(input_path).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"\nLoading: {input_file.name}")

    # [1/9] Preparing audio
    print("[1/9] Preparing audio")
    target_sr = config.get("output", {}).get("sample_rate", 44100)
    original_pcm, sr = load_audio_as_stereo_pcm(input_file, target_sr=target_sr)

    # Initialize cache
    cache_conf = config.get("cache", {})
    cache_enabled = cache_conf.get("enabled", True) and not force
    cache_mgr = StemCache(cache_dir=cache_conf.get("dir", ".cache"), enabled=cache_enabled)
    source_hash = compute_file_hash(input_file)

    # Models config
    models_conf = config.get("models", {})
    model_order = ["bleedless", "balanced", "fullness", "demucs_ft", "demucs_extra"]
    step_numbers = {
        "bleedless": 2,
        "balanced": 3,
        "fullness": 4,
        "demucs_ft": 5,
        "demucs_extra": 6,
    }

    model_stems = {}
    weights = {}
    carrier_key = "fullness"

    # [2/9] - [6/9] Sequential Separation
    backend = get_separator_backend()

    for key in model_order:
        if key not in models_conf:
            continue

        m_conf = models_conf.get(key, {})
        m_name = m_conf.get("model", key)
        m_display = m_conf.get("name", key.capitalize())
        weights[key] = m_conf.get("weight", 1.0)
        if m_conf.get("carrier", False):
            carrier_key = key

        step_num = step_numbers.get(key, 2)
        print(f"[{step_num}/9] Separating — {m_display}")

        # Check cache
        cached_pcm = cache_mgr.get_cached_stem(source_hash, m_name, sr=sr)
        if cached_pcm is not None:
            model_stems[key] = cached_pcm
            continue

        # Inference
        try:
            backend.load_model(m_name)
            stem_pcm = backend.separate_instrumental(original_pcm, sr=sr)
            model_stems[key] = stem_pcm
            cache_mgr.save_cached_stem(source_hash, m_name, stem_pcm, sr=sr)
        except Exception as e:
            logger.warning(f"Model {m_display} ({m_name}) failed during separation: {e}")
        finally:
            backend.unload_model()

    # Model failure policy: require at least 2 successful models
    if len(model_stems) < 2:
        raise RuntimeError(
            f"Stem separation failed: only {len(model_stems)} model(s) succeeded. "
            "At least 2 models are required for ensemble."
        )

    # [7/9] Building ensemble (Hierarchical Consensus)
    print("[7/9] Building ensemble (Hierarchical Consensus)")
    ens_conf_dict = config.get("ensemble", {})
    atten_conf = ens_conf_dict.get("attenuation", {})
    freq_limits = atten_conf.get("frequency_limits", {})
    smoothing_conf = ens_conf_dict.get("smoothing", {})
    roformer_conf = ens_conf_dict.get("roformer", {})
    family_conf = ens_conf_dict.get("family", {})

    ens_config = EnsembleConfig(
        removal_threshold=ens_conf_dict.get("removal_threshold", 0.45),
        soft_temperature=ens_conf_dict.get("soft_temperature", 0.10),
        low_quantile=ens_conf_dict.get("low_quantile", 0.25),
        silence_threshold_db=ens_conf_dict.get("silence_threshold_db", -70.0),
        w_bleedless=models_conf.get("bleedless", {}).get("weight", 1.0),
        w_balanced=models_conf.get("balanced", {}).get("weight", 1.0),
        w_fullness=models_conf.get("fullness", {}).get("weight", 0.8),
        roformer_ab_bonus=roformer_conf.get("ab_consensus_bonus", 0.15),
        w_demucs_ft=models_conf.get("demucs_ft", {}).get("weight", 0.6),
        w_demucs_extra=models_conf.get("demucs_extra", {}).get("weight", 0.5),
        arch_disagreement_penalty=family_conf.get("architecture_disagreement_penalty", 0.5),
        sub_freq=freq_limits.get("sub", {}).get("max_frequency", 120.0),
        sub_bass_db=freq_limits.get("sub", {}).get("max_attenuation_db", 3.0),
        low_freq=freq_limits.get("low", {}).get("max_frequency", 500.0),
        low_mids_db=freq_limits.get("low", {}).get("max_attenuation_db", 5.0),
        vocal_freq=freq_limits.get("vocal", {}).get("max_frequency", 5000.0),
        vocal_mids_db=freq_limits.get("vocal", {}).get("max_attenuation_db", 10.0),
        high_freq=freq_limits.get("high", {}).get("max_frequency", 10000.0),
        high_db=freq_limits.get("high", {}).get("max_attenuation_db", 6.0),
        air_db=freq_limits.get("air", {}).get("max_attenuation_db", 4.0),
        n_fft=ens_conf_dict.get("stft", {}).get("n_fft", 4096),
        hop_length=ens_conf_dict.get("stft", {}).get("hop_length", 1024),
        window=ens_conf_dict.get("stft", {}).get("window", "hann"),
        median_size=smoothing_conf.get("median_size", 3),
        attack_ms=smoothing_conf.get("attack_ms", 40.0),
        release_ms=smoothing_conf.get("release_ms", 100.0),
        frequency_bins=smoothing_conf.get("frequency_bins", 3),
    )

    ensemble_mr = build_ensemble_instrumental(
        original_pcm=original_pcm,
        model_stems=model_stems,
        weights=weights,
        carrier_key=carrier_key,
        config=ens_config,
        sr=sr,
    )

    # [8/9] Pitch shifting & Complementary Vocals calculation
    transpose_conf = config.get("transpose", {})
    rbargs = transpose_conf.get("rbargs", {})
    is_zero_semitone = abs(semitone) < 1e-4

    if is_zero_semitone:
        print("[8/9] Pitch shift skipped (0st)")
        processed_mr = ensemble_mr
        processed_mix = original_pcm
    else:
        print(f"[8/9] Pitch shifting {semitone:+g} semitones (MR & Mixture)")
        processed_mr = pitch_shift_instrumental(
            ensemble_mr,
            semitone=semitone,
            sr=sr,
            rbargs=rbargs,
        )
        processed_mix = pitch_shift_instrumental(
            original_pcm,
            semitone=semitone,
            sr=sr,
            rbargs=rbargs,
        )

    # Align lengths to ensure len(processed_mr) == len(processed_mix)
    target_pair_len = processed_mix.shape[1]
    aligned_mr = align_audio_length(processed_mr, target_pair_len)
    aligned_mix = processed_mix

    # Complementary Vocals: V = X - MR (or V' = X' - MR')
    vocals_pcm = (aligned_mix - aligned_mr).astype(np.float32)

    # Verify mixture consistency before quantization
    check_mixture_consistency(aligned_mix, aligned_mr, vocals_pcm)

    # Apply Linked Peak Safety across MR and Vocals
    out_conf = config.get("output", {})
    peak_ceiling_db = out_conf.get("peak_ceiling_db", -1.0)
    safe_mr, safe_vocals, _ = apply_linked_peak_safety(
        aligned_mr,
        vocals_pcm,
        peak_ceiling_db=peak_ceiling_db,
    )

    # [9/9] Writing WAV files
    print("[9/9] Writing WAV files")
    mr_path = export_stem_wav(
        audio_pcm=safe_mr,
        input_file_path=input_file,
        semitone=semitone,
        stem_name="MR",
        output_dir=output_dir,
        sr=sr,
    )
    vocals_path = export_stem_wav(
        audio_pcm=safe_vocals,
        input_file_path=input_file,
        semitone=semitone,
        stem_name="Vocals",
        output_dir=output_dir,
        sr=sr,
    )

    print(f"\nDone.\n\nMR:\n{mr_path}\n\nVocals:\n{vocals_path}\n")
    return PipelineResult(mr_path=mr_path, vocals_path=vocals_path)

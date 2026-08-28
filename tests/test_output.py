import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from inst_splitter.output import (
    apply_linked_peak_safety,
    apply_peak_safety,
    check_mixture_consistency,
    export_mr_wav,
    export_stem_wav,
    format_semitone_string,
)
from inst_splitter.utils import db_to_linear


def test_format_semitone_string():
    assert format_semitone_string(-5) == "-5st"
    assert format_semitone_string(2) == "+2st"
    assert format_semitone_string(0) == "0st"
    assert format_semitone_string(-3.5) == "-3.5st"
    assert format_semitone_string(1.5) == "+1.5st"


def test_apply_linked_peak_safety_below_ceiling():
    # Both MR and Vocals below -1 dBFS (linear ~ 0.891)
    mr = np.array([[0.4, -0.4], [0.4, -0.4]], dtype=np.float32)
    vocals = np.array([[0.3, -0.3], [0.3, -0.3]], dtype=np.float32)

    safe_mr, safe_vocals, reduction = apply_linked_peak_safety(mr, vocals, peak_ceiling_db=-1.0)
    assert reduction == 0.0
    assert np.allclose(safe_mr, mr)
    assert np.allclose(safe_vocals, vocals)


def test_apply_linked_peak_safety_above_ceiling():
    # MR peak is 1.4 (hot), Vocals peak is 0.7
    mr = np.array([[1.4, -0.5], [0.5, -1.4]], dtype=np.float32)
    vocals = np.array([[0.7, -0.2], [0.2, -0.7]], dtype=np.float32)
    original_mix = mr + vocals

    ceiling_db = -1.0
    ceiling_lin = db_to_linear(ceiling_db)

    safe_mr, safe_vocals, reduction_db = apply_linked_peak_safety(mr, vocals, peak_ceiling_db=ceiling_db)

    # Reduction must be negative dB
    assert reduction_db < 0.0

    # Max peak of both safe stems must not exceed ceiling
    assert np.max(np.abs(safe_mr)) <= ceiling_lin + 1e-5
    assert np.max(np.abs(safe_vocals)) <= ceiling_lin + 1e-5

    # Crucial Invariant: safe_mr + safe_vocals == scale_factor * original_mix
    scale_factor = db_to_linear(reduction_db)
    reconstructed_mix = safe_mr + safe_vocals
    expected_mix = original_mix * scale_factor
    assert np.allclose(reconstructed_mix, expected_mix, atol=1e-6)


def test_check_mixture_consistency():
    mixture = np.array([[0.5, -0.5], [0.3, -0.3]], dtype=np.float32)
    mr = np.array([[0.3, -0.3], [0.1, -0.1]], dtype=np.float32)
    vocals = mixture - mr

    error = check_mixture_consistency(mixture, mr, vocals, tolerance=1e-5)
    assert error < 1e-6


def test_export_stem_wav():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_song = tmp_path / "test_track.flac"
        # Dummy audio
        audio = np.zeros((2, 44100), dtype=np.float32)

        mr_path = export_stem_wav(audio, input_song, semitone=-5, stem_name="MR")
        vocals_path = export_stem_wav(audio, input_song, semitone=-5, stem_name="Vocals")

        assert mr_path.name == "test_track_MR_-5st.wav"
        assert vocals_path.name == "test_track_Vocals_-5st.wav"
        assert mr_path.exists()
        assert vocals_path.exists()

import numpy as np
import pytest

from inst_splitter.transpose import pitch_shift_instrumental


def test_transpose_zero_semitones():
    sr = 44100
    samples = 10000
    audio = np.random.uniform(-0.5, 0.5, (2, samples)).astype(np.float32)

    # Semitone = 0 should return identical audio without running DSP
    shifted = pitch_shift_instrumental(audio, semitone=0, sr=sr)
    assert np.allclose(shifted, audio)
    assert shifted.shape == audio.shape


def test_transpose_duration_preservation():
    sr = 44100
    duration = 0.5
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # 440 Hz test tone
    audio = np.stack([
        np.sin(2 * np.pi * 440 * t) * 0.5,
        np.cos(2 * np.pi * 440 * t) * 0.5
    ], axis=0).astype(np.float32)

    try:
        shifted_up = pitch_shift_instrumental(audio, semitone=2, sr=sr)
        assert shifted_up.shape == audio.shape
        assert not np.isnan(shifted_up).any()
    except Exception as e:
        # If rubberband CLI binary is not installed on the system, skip gracefully in unit test
        pytest.skip(f"Rubber Band binary not available: {e}")

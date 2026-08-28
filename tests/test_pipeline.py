import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import soundfile as sf

from inst_splitter.output import apply_peak_safety, export_mr_wav, format_semitone_string
from inst_splitter.pipeline import PipelineResult, run_pipeline


def test_format_semitone_string():
    assert format_semitone_string(-5) == "-5st"
    assert format_semitone_string(2) == "+2st"
    assert format_semitone_string(0) == "0st"
    assert format_semitone_string(-3.5) == "-3.5st"
    assert format_semitone_string(1.5) == "+1.5st"


def test_apply_peak_safety():
    # Audio below -1 dBFS (linear ~ 0.891)
    safe_pcm = np.array([[0.5, -0.5], [0.5, -0.5]], dtype=np.float32)
    adjusted, reduction = apply_peak_safety(safe_pcm, peak_ceiling_db=-1.0)
    assert reduction == 0.0
    assert np.allclose(adjusted, safe_pcm)

    # Audio with peak 1.5 (exceeds ceiling)
    hot_pcm = np.array([[1.5, -1.0], [0.5, -1.5]], dtype=np.float32)
    adjusted_hot, reduction_hot = apply_peak_safety(hot_pcm, peak_ceiling_db=-1.0)
    assert reduction_hot < 0.0
    assert np.max(np.abs(adjusted_hot)) <= 10 ** (-1.0 / 20.0) + 1e-5


def test_pipeline_integration_mock_backend_mr_and_vocals():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_song = tmp_path / "my_song.wav"
        output_dir = tmp_path / "MR_output"

        # Generate 1 sec test audio
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        stereo_pcm = np.stack([np.sin(2 * np.pi * 440 * t) * 0.4, np.cos(2 * np.pi * 440 * t) * 0.4], axis=1)
        sf.write(str(input_song), stereo_pcm, sr)

        # Mock SeparatorBackend to return simulated stems for all 5 models
        mock_backend = MagicMock()
        stem_sample = np.stack([np.sin(2 * np.pi * 440 * t) * 0.35, np.sin(2 * np.pi * 440 * t) * 0.35], axis=0).astype(np.float32)
        mock_backend.separate_instrumental.side_effect = [
            stem_sample * 0.9,  # bleedless
            stem_sample * 0.95, # balanced
            stem_sample * 1.0,  # fullness
            stem_sample * 0.92, # demucs_ft
            stem_sample * 0.94, # demucs_extra
        ]

        with patch("inst_splitter.pipeline.get_separator_backend", return_value=mock_backend):
            result: PipelineResult = run_pipeline(
                input_path=input_song,
                semitone=0,  # 0 semitone to test clean bypass
                output_dir=output_dir,
                force=True,
            )

            assert isinstance(result, PipelineResult)
            assert result.mr_path.exists()
            assert result.vocals_path.exists()
            assert result.mr_path.name == "my_song_MR_0st.wav"
            assert result.vocals_path.name == "my_song_Vocals_0st.wav"

            # Check WAV metadata for both files
            info_mr = sf.info(str(result.mr_path))
            assert info_mr.samplerate == 44100
            assert info_mr.channels == 2
            assert info_mr.subtype == "PCM_24"

            info_voc = sf.info(str(result.vocals_path))
            assert info_voc.samplerate == 44100
            assert info_voc.channels == 2
            assert info_voc.subtype == "PCM_24"
            assert info_mr.frames == info_voc.frames

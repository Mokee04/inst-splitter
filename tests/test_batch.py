import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import soundfile as sf

from inst_splitter.cli import find_audio_files, run_batch
from inst_splitter.pipeline import PipelineResult


def test_find_audio_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create audio files
        (tmp_path / "song1.flac").touch()
        (tmp_path / "song2.wav").touch()
        (tmp_path / "song3.mp3").touch()
        (tmp_path / "not_audio.txt").touch()

        # Subdirectory
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        (sub_dir / "song4.m4a").touch()

        # Existing MR_output directory (should be excluded)
        mr_out = tmp_path / "MR_output"
        mr_out.mkdir()
        (mr_out / "song1_MR_0st.wav").touch()
        (mr_out / "song1_Vocals_0st.wav").touch()

        # 1. Non-recursive search
        files_flat = find_audio_files(tmp_path, recursive=False)
        assert len(files_flat) == 3
        filenames_flat = [f.name for f in files_flat]
        assert "song1.flac" in filenames_flat
        assert "song2.wav" in filenames_flat
        assert "song3.mp3" in filenames_flat
        assert "not_audio.txt" not in filenames_flat
        assert "song4.m4a" not in filenames_flat

        # 2. Recursive search
        files_rec = find_audio_files(tmp_path, recursive=True)
        assert len(files_rec) == 4
        filenames_rec = [f.name for f in files_rec]
        assert "song4.m4a" in filenames_rec
        assert "song1_MR_0st.wav" not in filenames_rec


def test_run_batch_success():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create 2 dummy audio files
        sr = 44100
        t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
        dummy_wave = np.stack([np.sin(2 * np.pi * 440 * t) * 0.3, np.cos(2 * np.pi * 440 * t) * 0.3], axis=1)

        f1 = tmp_path / "track1.wav"
        f2 = tmp_path / "track2.wav"
        sf.write(str(f1), dummy_wave, sr)
        sf.write(str(f2), dummy_wave, sr)

        # Mock run_pipeline to return dummy PipelineResult
        mock_result = PipelineResult(
            mr_path=tmp_path / "MR_output" / "track_MR_0st.wav",
            vocals_path=tmp_path / "MR_output" / "track_Vocals_0st.wav",
        )

        with patch("inst_splitter.cli.run_pipeline", return_value=mock_result) as mock_pipeline:
            exit_code = run_batch(
                folder_path=tmp_path,
                semitone=0,
                recursive=False,
            )
            assert exit_code == 0
            assert mock_pipeline.call_count == 2

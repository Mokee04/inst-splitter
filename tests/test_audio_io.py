import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from inst_splitter.audio_io import load_audio_as_stereo_pcm, resample_audio, save_24bit_wav
from inst_splitter.utils import validate_audio_pcm


def test_load_mono_to_stereo():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_wav = Path(tmp_dir) / "mono_test.wav"
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        mono_data = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        sf.write(str(tmp_wav), mono_data, sr)

        loaded_pcm, loaded_sr = load_audio_as_stereo_pcm(tmp_wav, target_sr=44100)
        assert loaded_sr == 44100
        assert loaded_pcm.shape[0] == 2
        assert loaded_pcm.shape[1] == len(mono_data)
        assert np.allclose(loaded_pcm[0], loaded_pcm[1])
        assert loaded_pcm.dtype == np.float32


def test_load_and_resample():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_wav = Path(tmp_dir) / "resample_test.wav"
        sr_orig = 48000
        sr_target = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr_orig * duration), endpoint=False)
        stereo_data = np.stack([np.sin(2 * np.pi * 440 * t), np.cos(2 * np.pi * 440 * t)], axis=1).astype(np.float32)

        sf.write(str(tmp_wav), stereo_data, sr_orig)

        loaded_pcm, loaded_sr = load_audio_as_stereo_pcm(tmp_wav, target_sr=sr_target)
        assert loaded_sr == sr_target
        assert loaded_pcm.shape[0] == 2
        assert abs(loaded_pcm.shape[1] - int(sr_target * duration)) <= 10


def test_save_24bit_wav():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_wav = Path(tmp_dir) / "out_24bit.wav"
        sr = 44100
        samples = 44100
        audio_pcm = np.random.uniform(-0.5, 0.5, (2, samples)).astype(np.float32)

        saved_path = save_24bit_wav(tmp_wav, audio_pcm, sr=sr)
        assert saved_path.exists()

        info = sf.info(str(saved_path))
        assert info.samplerate == sr
        assert info.channels == 2
        assert info.subtype == "PCM_24"


def test_validate_audio_pcm():
    valid = np.zeros((2, 1000), dtype=np.float32)
    validate_audio_pcm(valid)

    with pytest.raises(ValueError):
        validate_audio_pcm(np.zeros((1, 1000), dtype=np.float32))

    with pytest.raises(ValueError):
        invalid_nan = np.zeros((2, 1000), dtype=np.float32)
        invalid_nan[0, 10] = np.nan
        validate_audio_pcm(invalid_nan)

    with pytest.raises(ValueError):
        invalid_inf = np.zeros((2, 1000), dtype=np.float32)
        invalid_inf[1, 10] = np.inf
        validate_audio_pcm(invalid_inf)

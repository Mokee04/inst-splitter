"""
inst-splitter: 5-Model Heterogeneous Ensemble AI Instrumental & Vocals Splitter
"""

from inst_splitter.pipeline import PipelineResult, run_pipeline
from inst_splitter.audio_io import load_audio_as_stereo_pcm, save_24bit_wav
from inst_splitter.ensemble import build_ensemble_instrumental
from inst_splitter.transpose import pitch_shift_instrumental

__version__ = "1.0.0"
__all__ = [
    "run_pipeline",
    "PipelineResult",
    "load_audio_as_stereo_pcm",
    "save_24bit_wav",
    "build_ensemble_instrumental",
    "pitch_shift_instrumental",
]

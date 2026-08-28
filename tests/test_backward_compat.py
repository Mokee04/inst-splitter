import pytest

def test_mr_builder_backward_compatibility():
    """Verify that legacy imports from mr_builder continue to function seamlessly."""
    import mr_builder
    from mr_builder.pipeline import run_pipeline, PipelineResult
    from mr_builder.ensemble import EnsembleConfig, build_ensemble_instrumental
    from mr_builder.output import apply_linked_peak_safety
    from mr_builder.audio_io import load_audio_as_stereo_pcm
    from mr_builder.transpose import pitch_shift_instrumental
    from mr_builder.utils import load_config, setup_logger

    assert hasattr(mr_builder, "run_pipeline")
    assert hasattr(mr_builder, "PipelineResult")

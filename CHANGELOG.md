# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-28

### Added
- **5-Model Heterogeneous Ensemble**: Hierarchical Consensus combining 3 RoFormer models (`Bleedless`, `Balanced`, `Fullness`) and 2 Demucs models (`HTDemucs FT`, `Demucs MDX Extra`).
- **Complementary Residual Vocals**: Exact mixture consistency ($MR + Vocals = Mixture$) in 32-bit floating point precision.
- **Linked Peak Safety**: Single linked common gain reduction to prevent clipping while preserving vocal-to-instrumental relative balance.
- **Asymmetric Temporal Envelope Smoothing**: Attack (40ms) / Release (100ms) DSP preventing chirping artifacts.
- **Frequency-Dependent Attenuation**: Multi-band max attenuation curves (Sub-bass, Lows, Vocals, Highs, Air).
- **Batch Processing CLI**: `inst-splitter-batch` for folder scanning, recursive traversal, and batch conversion reports.
- **Unified CLI Entrypoints**: `inst-split`, `inst-splitter`, `inst-splitter-batch` (with backward-compatible `make-mr` aliases).
- **High Quality 24-bit PCM WAV Master Output**.

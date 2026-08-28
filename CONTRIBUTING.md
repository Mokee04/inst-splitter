# Contributing to inst-splitter

Thank you for your interest in contributing to `inst-splitter`! We welcome contributions from the community.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mokee04/inst-splitter.git
   cd inst-splitter
   ```

2. **System Dependencies**:
   - **macOS**:
     ```bash
     brew install ffmpeg rubberband libsndfile
     ```
   - **Ubuntu / Debian**:
     ```bash
     sudo apt-get update && sudo apt-get install -y ffmpeg rubberband-cli libsndfile1
     ```

3. **Install Python environment (with uv or pip)**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   # or with pip:
   # pip install -e ".[dev]"
   ```

4. **Run Tests**:
   ```bash
   pytest
   ```

## Workflow

1. Create a feature branch: `git checkout -b feature/my-new-feature`
2. Make your changes with clear, descriptive commit messages.
3. Ensure all tests pass: `pytest`
4. Push to your branch and open a Pull Request.

## Code Style & Principles
- Keep changes minimal and focused.
- Ensure audio math invariants (e.g. `Mixture = MR + Vocals`) and float32 PCM bounds are preserved.
- Write clean docstrings and comments.

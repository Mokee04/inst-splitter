#!/usr/bin/env python3
"""
inst_splitter CLI module
Provides CLI entry points for single track and batch processing.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from inst_splitter.audio_io import SUPPORTED_EXTENSIONS
from inst_splitter.pipeline import PipelineResult, run_pipeline
from inst_splitter.utils import load_config, setup_logger


def parse_single_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="inst-split",
        description="5-Model Heterogeneous Ensemble AI Instrumental & Vocals Splitter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  inst-split "song.flac" -5
  inst-split "song.wav" +2
  inst-split "song.mp3" 0 --output-dir "./results"
        """
    )

    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the input audio file (.flac, .wav, .mp3, .ogg, .m4a)"
    )
    parser.add_argument(
        "semitone",
        type=float,
        help="Semitone pitch shift value (e.g. -5, +2, 0, -1.5)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-separation bypassing cache"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (default: <file_folder>/MR_output)"
    )

    return parser.parse_args(args)


def validate_input_file(file_path_str: str) -> Path:
    file_path = Path(file_path_str).resolve()
    if not file_path.exists():
        print(f"Error: Input file does not exist: {file_path}", file=sys.stderr)
        sys.exit(1)

    if not file_path.is_file():
        print(f"Error: Input path is not a file: {file_path}", file=sys.stderr)
        sys.exit(1)

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(
            f"Error: Unsupported audio format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            file=sys.stderr
        )
        sys.exit(1)

    return file_path


def main_single(args: Optional[List[str]] = None) -> None:
    """CLI entry point for single file processing."""
    parsed = parse_single_args(args)
    logger = setup_logger("inst_splitter", debug=parsed.debug)

    input_file = validate_input_file(parsed.file_path)

    try:
        config = load_config(parsed.config)
        run_pipeline(
            input_path=input_file,
            semitone=parsed.semitone,
            config=config,
            output_dir=parsed.output_dir,
            force=parsed.force,
            debug=parsed.debug,
        )
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if parsed.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def parse_batch_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="inst-splitter-batch",
        description="Batch AI Instrumental & Vocals splitter for audio folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  inst-splitter-batch "/Music/Album" -5
  inst-splitter-batch "/Music/Library" 0 --recursive
  inst-splitter-batch "./inputs" +2 --output-dir "./results"
        """
    )

    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder containing audio files"
    )
    parser.add_argument(
        "semitone",
        type=float,
        help="Semitone pitch shift value (e.g. -5, +2, 0)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively search for audio files in subdirectories"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-separation bypassing cache"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for all generated files"
    )

    return parser.parse_args(args)


def find_audio_files(folder_path: Path, recursive: bool = False) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    all_files = folder_path.glob(pattern)

    audio_files = []
    for f in all_files:
        if not f.is_file():
            continue

        if "MR_output" in f.parts or "_MR_" in f.stem or "_Vocals_" in f.stem:
            continue

        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            audio_files.append(f.resolve())

    return sorted(audio_files)


def run_batch(
    folder_path: Path,
    semitone: float,
    recursive: bool = False,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    force: bool = False,
    debug: bool = False
) -> int:
    logger = setup_logger("inst_splitter_batch", debug=debug)
    config = load_config(config_path)

    audio_files = find_audio_files(folder_path, recursive=recursive)

    if not audio_files:
        print(f"\nNo supported audio files found in: {folder_path}", file=sys.stderr)
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}", file=sys.stderr)
        return 0

    total_files = len(audio_files)
    print("\n" + "=" * 60)
    print("inst-splitter Batch Processing")
    print(f"Target Folder : {folder_path}")
    print(f"Pitch Shift   : {semitone:+g} semitones")
    print(f"Audio Files   : {total_files} file(s) found")
    if output_dir:
        print(f"Output Dir    : {output_dir}")
    print("=" * 60)

    results: List[Tuple[Path, bool, str, Optional[PipelineResult]]] = []

    for idx, audio_file in enumerate(audio_files, start=1):
        print(f"\n[{idx}/{total_files}] Processing: {audio_file.name}")
        print("-" * 50)

        try:
            pipeline_result = run_pipeline(
                input_path=audio_file,
                semitone=semitone,
                config=config,
                output_dir=output_dir,
                force=force,
                debug=debug,
            )
            results.append((audio_file, True, "Success", pipeline_result))
        except KeyboardInterrupt:
            print("\nBatch process interrupted by user.", file=sys.stderr)
            return 130
        except Exception as e:
            logger.error(f"Failed processing {audio_file.name}: {e}")
            results.append((audio_file, False, str(e), None))
            if debug:
                import traceback
                traceback.print_exc()

    succeeded = sum(1 for _, ok, _, _ in results if ok)
    failed = total_files - succeeded

    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print(f"Total: {total_files} | Succeeded: {succeeded} | Failed: {failed}")
    print("=" * 60)

    for audio_file, ok, msg, pipe_res in results:
        status_str = "SUCCESS" if ok else f"FAILED ({msg})"
        print(f"- {audio_file.name}: {status_str}")
        if ok and pipe_res:
            print(f"    MR    : {pipe_res.mr_path}")
            print(f"    Vocals: {pipe_res.vocals_path}")
    print("=" * 60 + "\n")

    return 0 if failed == 0 else 1


def main_batch(args: Optional[List[str]] = None) -> None:
    """CLI entry point for batch processing."""
    parsed = parse_batch_args(args)
    folder_path = Path(parsed.folder_path).resolve()

    if not folder_path.exists():
        print(f"Error: Folder does not exist: {folder_path}", file=sys.stderr)
        sys.exit(1)

    if not folder_path.is_dir():
        print(f"Error: Path is not a directory: {folder_path}", file=sys.stderr)
        sys.exit(1)

    exit_code = run_batch(
        folder_path=folder_path,
        semitone=parsed.semitone,
        recursive=parsed.recursive,
        config_path=parsed.config,
        output_dir=parsed.output_dir,
        force=parsed.force,
        debug=parsed.debug,
    )
    sys.exit(exit_code)

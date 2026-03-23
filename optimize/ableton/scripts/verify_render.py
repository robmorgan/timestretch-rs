#!/usr/bin/env python3
"""
Validate exported WAV files from Ableton rendering.

Checks:
  - File exists and is non-empty
  - Valid WAV format (readable by soundfile)
  - Sample rate matches expected (default 44100)
  - Channel count matches expected (1 or 2)
  - Duration is within tolerance of expected duration (if provided)

Usage:
    python verify_render.py output.wav
    python verify_render.py output.wav --expected-duration 420.8 --expected-sr 44100 --expected-channels 2
"""
import argparse
import os
import sys

import soundfile as sf


def verify_render(wav_path, expected_sr=44100, expected_channels=None,
                  expected_duration=None, duration_tolerance=0.05):
    """
    Validate a rendered WAV file.

    Args:
        wav_path: Path to the WAV file
        expected_sr: Expected sample rate (default 44100)
        expected_channels: Expected channel count (None = don't check)
        expected_duration: Expected duration in seconds (None = don't check)
        duration_tolerance: Fractional tolerance for duration (default 5%)

    Returns:
        (valid: bool, issues: list[str])
    """
    issues = []

    # Check existence
    if not os.path.exists(wav_path):
        return False, [f"File not found: {wav_path}"]

    # Check non-empty
    file_size = os.path.getsize(wav_path)
    if file_size == 0:
        return False, [f"File is empty: {wav_path}"]

    if file_size < 44:  # WAV header minimum
        return False, [f"File too small to be valid WAV ({file_size} bytes)"]

    # Read audio info
    try:
        info = sf.info(wav_path)
    except Exception as e:
        return False, [f"Cannot read WAV file: {e}"]

    # Check sample rate
    if info.samplerate != expected_sr:
        issues.append(
            f"Sample rate mismatch: got {info.samplerate}, expected {expected_sr}"
        )

    # Check channels
    if expected_channels is not None and info.channels != expected_channels:
        issues.append(
            f"Channel count mismatch: got {info.channels}, expected {expected_channels}"
        )

    # Check duration
    actual_duration = info.duration
    if expected_duration is not None:
        ratio = actual_duration / expected_duration if expected_duration > 0 else 0
        if abs(ratio - 1.0) > duration_tolerance:
            issues.append(
                f"Duration mismatch: got {actual_duration:.2f}s, "
                f"expected {expected_duration:.2f}s "
                f"(ratio={ratio:.4f}, tolerance={duration_tolerance})"
            )

    # Basic sanity: duration > 0
    if actual_duration <= 0:
        issues.append(f"Invalid duration: {actual_duration}")

    valid = len(issues) == 0
    return valid, issues


def print_info(wav_path):
    """Print detailed info about a WAV file."""
    try:
        info = sf.info(wav_path)
    except Exception as e:
        print(f"Error reading {wav_path}: {e}")
        return

    print(f"File:       {wav_path}")
    print(f"Format:     {info.format} / {info.subtype}")
    print(f"Channels:   {info.channels}")
    print(f"Sample rate: {info.samplerate} Hz")
    print(f"Frames:     {info.frames}")
    print(f"Duration:   {info.duration:.3f}s")
    print(f"File size:  {os.path.getsize(wav_path):,} bytes")


def main():
    parser = argparse.ArgumentParser(description="Validate rendered WAV files")
    parser.add_argument("wav", nargs="+", help="WAV file(s) to validate")
    parser.add_argument("--expected-sr", type=int, default=44100,
                        help="Expected sample rate (default: 44100)")
    parser.add_argument("--expected-channels", type=int, default=None,
                        help="Expected channel count")
    parser.add_argument("--expected-duration", type=float, default=None,
                        help="Expected duration in seconds")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Duration tolerance as fraction (default: 0.05 = 5%%)")
    parser.add_argument("--info", action="store_true",
                        help="Print detailed file info")

    args = parser.parse_args()

    all_valid = True
    for wav_path in args.wav:
        if args.info:
            print_info(wav_path)
            print()
            continue

        valid, issues = verify_render(
            wav_path,
            expected_sr=args.expected_sr,
            expected_channels=args.expected_channels,
            expected_duration=args.expected_duration,
            duration_tolerance=args.tolerance,
        )

        if valid:
            print(f"OK: {wav_path}")
        else:
            print(f"FAIL: {wav_path}")
            for issue in issues:
                print(f"  - {issue}")
            all_valid = False

    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()

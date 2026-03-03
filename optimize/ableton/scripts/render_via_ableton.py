#!/usr/bin/env python3
"""
Render a single track through Ableton Live via GUI automation.

Pipeline per track:
  1. Generate an .als project from the template
  2. Open the project in Ableton
  3. Wait for Ableton to load
  4. Run the export AppleScript
  5. Wait for the output file to appear and stabilize
  6. Validate the output
  7. Close Ableton

Usage:
    python render_via_ableton.py template.als input.wav 115.0 output.wav
    python render_via_ableton.py --catalog track_catalog.json --entry music-sounds-better --target-bpm 115.0
"""
import argparse
import logging
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABLETON_APP = "Ableton Live 11 Suite"
ABLETON_APP_PATH = f"/Applications/{ABLETON_APP}.app"
EXPORT_APPLESCRIPT = os.path.join(SCRIPT_DIR, "export_dialog.applescript")

# Timeouts and retries
ABLETON_LOAD_TIMEOUT = 45  # seconds to wait for Ableton to load a project
EXPORT_TIMEOUT = 300  # seconds to wait for export to complete (5 min)
FILE_STABLE_CHECKS = 3  # number of stable size checks before considering done
FILE_STABLE_INTERVAL = 2  # seconds between size checks
MAX_RETRIES = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def is_ableton_running():
    """Check if Ableton is currently running."""
    result = subprocess.run(
        ["pgrep", "-f", ABLETON_APP],
        capture_output=True, text=True
    )
    return result.returncode == 0


def open_project_in_ableton(als_path):
    """Open an .als project in Ableton Live."""
    abs_path = os.path.abspath(als_path)
    log.info(f"Opening project in Ableton: {abs_path}")
    subprocess.run(["open", "-a", ABLETON_APP_PATH, abs_path], check=True)


def wait_for_ableton_ready(timeout=ABLETON_LOAD_TIMEOUT):
    """Wait for Ableton to finish loading (heuristic: wait for it to be responsive)."""
    log.info(f"Waiting up to {timeout}s for Ableton to be ready...")
    start = time.time()

    # First, wait for the process to appear
    while time.time() - start < timeout:
        if is_ableton_running():
            break
        time.sleep(1)

    # Then wait a fixed amount for project loading
    # This is a heuristic -- Ableton has no CLI signal for "ready"
    remaining = max(10, timeout - (time.time() - start))
    wait_time = min(remaining, 20)
    log.info(f"Ableton process found, waiting {wait_time}s for project to load...")
    time.sleep(wait_time)


def run_export(output_dir, output_filename):
    """Run the AppleScript export automation."""
    abs_output_dir = os.path.abspath(output_dir)
    log.info(f"Running export: dir={abs_output_dir}, file={output_filename}")

    result = subprocess.run(
        ["osascript", EXPORT_APPLESCRIPT, abs_output_dir, output_filename],
        capture_output=True, text=True, timeout=60
    )

    if result.returncode != 0:
        log.error(f"AppleScript failed: {result.stderr}")
        return False

    log.info("Export AppleScript completed")
    return True


def wait_for_file(file_path, timeout=EXPORT_TIMEOUT):
    """Wait for an output file to appear and stabilize (size stops growing)."""
    log.info(f"Waiting for output file: {file_path}")
    start = time.time()

    # Wait for file to appear
    while time.time() - start < timeout:
        if os.path.exists(file_path):
            break
        time.sleep(2)
    else:
        log.error(f"Timeout waiting for file to appear: {file_path}")
        return False

    # Wait for file to stabilize (size stops changing)
    stable_count = 0
    last_size = -1
    while time.time() - start < timeout:
        try:
            current_size = os.path.getsize(file_path)
        except OSError:
            time.sleep(FILE_STABLE_INTERVAL)
            continue

        if current_size == last_size and current_size > 0:
            stable_count += 1
            if stable_count >= FILE_STABLE_CHECKS:
                log.info(f"File stabilized at {current_size} bytes")
                return True
        else:
            stable_count = 0

        last_size = current_size
        time.sleep(FILE_STABLE_INTERVAL)

    log.error(f"Timeout waiting for file to stabilize: {file_path}")
    return False


def quit_ableton():
    """Quit Ableton Live gracefully."""
    log.info("Quitting Ableton Live...")
    try:
        subprocess.run(
            ["osascript", "-e",
             f'tell application "{ABLETON_APP}" to quit'],
            capture_output=True, text=True, timeout=10
        )
    except subprocess.TimeoutExpired:
        log.warning("Quit command timed out, force killing...")
        subprocess.run(["pkill", "-f", ABLETON_APP], capture_output=True)

    # Wait for Ableton to fully close
    for _ in range(30):
        if not is_ableton_running():
            log.info("Ableton closed")
            return
        time.sleep(1)

    log.warning("Ableton did not close within 30s, force killing...")
    subprocess.run(["pkill", "-9", "-f", ABLETON_APP], capture_output=True)
    time.sleep(2)


def render_single_track(template_path, wav_path, target_bpm, output_wav_path,
                        close_after=True):
    """
    Full pipeline: generate .als -> open in Ableton -> export -> validate.

    Returns True on success, False on failure.
    """
    # Import here to avoid circular dependency
    from create_als_project import create_project

    output_dir = os.path.dirname(os.path.abspath(output_wav_path))
    output_filename = os.path.splitext(os.path.basename(output_wav_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate .als project
    als_path = os.path.join(output_dir, f"{output_filename}.als")
    log.info(f"Step 1: Generating .als project -> {als_path}")
    if not create_project(template_path, wav_path, target_bpm, als_path):
        return False

    # Step 2: Open in Ableton
    log.info("Step 2: Opening project in Ableton")
    open_project_in_ableton(als_path)

    # Step 3: Wait for loading
    log.info("Step 3: Waiting for Ableton to load")
    wait_for_ableton_ready()

    # Step 4: Run export
    log.info("Step 4: Running export automation")
    if not run_export(output_dir, output_filename):
        if close_after:
            quit_ableton()
        return False

    # Step 5: Wait for output
    log.info("Step 5: Waiting for output file")
    if not wait_for_file(output_wav_path):
        if close_after:
            quit_ableton()
        return False

    # Step 6: Validate
    log.info("Step 6: Validating output")
    try:
        from verify_render import verify_render
        valid, issues = verify_render(output_wav_path)
        if not valid:
            log.error(f"Validation failed: {issues}")
            if close_after:
                quit_ableton()
            return False
        log.info("Validation passed")
    except ImportError:
        log.warning("verify_render not available, skipping validation")

    # Step 7: Close Ableton
    if close_after:
        log.info("Step 7: Closing Ableton")
        quit_ableton()

    # Clean up temporary .als
    try:
        os.remove(als_path)
    except OSError:
        pass

    log.info(f"Successfully rendered: {output_wav_path}")
    return True


def render_with_retries(template_path, wav_path, target_bpm, output_wav_path,
                        max_retries=MAX_RETRIES):
    """Render with retry logic."""
    for attempt in range(max_retries + 1):
        if attempt > 0:
            log.info(f"Retry attempt {attempt}/{max_retries}")
            # Make sure Ableton is fully closed before retrying
            if is_ableton_running():
                quit_ableton()
            time.sleep(5)

        if render_single_track(template_path, wav_path, target_bpm,
                               output_wav_path, close_after=True):
            return True

        log.warning(f"Attempt {attempt + 1} failed")

    log.error(f"All {max_retries + 1} attempts failed for {output_wav_path}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Render a single track through Ableton Live"
    )
    parser.add_argument("template", help="Path to .als template")
    parser.add_argument("wav", help="Path to source WAV file")
    parser.add_argument("bpm", type=float, help="Target tempo (BPM)")
    parser.add_argument("output", help="Output WAV path")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES,
                        help=f"Max retry attempts (default: {MAX_RETRIES})")
    parser.add_argument("--no-close", action="store_true",
                        help="Don't close Ableton after rendering")
    parser.add_argument("--log-file", help="Write logs to file")

    args = parser.parse_args()

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(file_handler)

    if args.no_close:
        success = render_single_track(
            args.template, args.wav, args.bpm, args.output, close_after=False
        )
    else:
        success = render_with_retries(
            args.template, args.wav, args.bpm, args.output,
            max_retries=args.retries
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

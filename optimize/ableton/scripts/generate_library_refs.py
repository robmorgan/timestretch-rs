#!/usr/bin/env python3
"""
Generate timestretch-rs library outputs for the same tracks/ratios in the
Ableton track catalog. This produces the "test" side of the comparison.

Runs the timestretch-cli binary in both batch and streaming modes.

Usage:
    python generate_library_refs.py track_catalog.json refs/library/
    python generate_library_refs.py track_catalog.json refs/library/ --streaming-only
"""
import argparse
import json
import logging
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def find_cli_binary():
    """Find the timestretch-cli binary, building if necessary."""
    release_path = os.path.join(REPO_ROOT, "target", "release", "timestretch-cli")
    debug_path = os.path.join(REPO_ROOT, "target", "debug", "timestretch-cli")

    if os.path.exists(release_path):
        return release_path

    if os.path.exists(debug_path):
        log.warning("Using debug build (slower). Build with --release for best results.")
        return debug_path

    # Try to build
    log.info("Building timestretch-cli (release)...")
    result = subprocess.run(
        ["cargo", "build", "--release", "--features", "cli"],
        cwd=REPO_ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error(f"Build failed: {result.stderr}")
        return None

    if os.path.exists(release_path):
        return release_path

    return None


def load_catalog(catalog_path):
    """Load the track catalog."""
    with open(catalog_path, "r") as f:
        return json.load(f)


def resolve_source_path(source, catalog_dir):
    """Resolve relative source paths."""
    if os.path.isabs(source):
        return source
    return os.path.normpath(os.path.join(catalog_dir, source))


def run_timestretch(cli_path, source, output, ratio, streaming=False):
    """Run the timestretch-cli binary."""
    cmd = [cli_path, source, output, "--ratio", str(ratio), "--no-normalize"]
    if streaming:
        cmd.append("--streaming")

    log.info(f"Running: {' '.join(os.path.basename(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error(f"timestretch-cli failed: {result.stderr}")
        return False

    if not os.path.exists(output):
        log.error(f"Output not created: {output}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate timestretch-rs library outputs for catalog entries"
    )
    parser.add_argument("catalog", help="Path to track_catalog.json")
    parser.add_argument("output_dir", help="Output directory for library WAVs")
    parser.add_argument("--batch-only", action="store_true",
                        help="Only generate batch mode outputs")
    parser.add_argument("--streaming-only", action="store_true",
                        help="Only generate streaming mode outputs")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already exist")
    parser.add_argument("--cli-path", help="Path to timestretch-cli binary")
    parser.add_argument("--manifest-out",
                        help="Write library manifest JSON to this path")

    args = parser.parse_args()

    # Find CLI binary
    cli_path = args.cli_path or find_cli_binary()
    if not cli_path:
        log.error("timestretch-cli not found. Run: cargo build --release --features cli")
        sys.exit(1)
    log.info(f"Using CLI: {cli_path}")

    catalog = load_catalog(args.catalog)
    catalog_dir = os.path.dirname(os.path.abspath(args.catalog))
    os.makedirs(args.output_dir, exist_ok=True)

    do_batch = not args.streaming_only
    do_streaming = not args.batch_only

    results = {"success": [], "failed": []}
    manifest = []

    for track in catalog:
        source_path = resolve_source_path(track["source"], catalog_dir)
        if not os.path.exists(source_path):
            log.warning(f"Source not found, skipping: {source_path}")
            continue

        for target in track["targets"]:
            track_id = track["id"]
            target_bpm = target["target_bpm"]
            ratio = target["ratio"]

            # Batch mode output
            if do_batch:
                batch_name = f"{track_id}_{target_bpm}bpm_batch.wav"
                batch_path = os.path.join(args.output_dir, batch_name)

                if args.skip_existing and os.path.exists(batch_path):
                    log.info(f"Skipping (exists): {batch_name}")
                else:
                    if run_timestretch(cli_path, source_path, batch_path, ratio):
                        results["success"].append(batch_name)
                    else:
                        results["failed"].append(batch_name)

                manifest.append({
                    "track_id": track_id,
                    "target_bpm": target_bpm,
                    "ratio": ratio,
                    "mode": "batch",
                    "path": batch_path,
                    "stereo": track.get("stereo", False),
                    "description": f"{track_id} @ {target_bpm} BPM (batch)",
                })

            # Streaming mode output
            if do_streaming:
                stream_name = f"{track_id}_{target_bpm}bpm_stream.wav"
                stream_path = os.path.join(args.output_dir, stream_name)

                if args.skip_existing and os.path.exists(stream_path):
                    log.info(f"Skipping (exists): {stream_name}")
                else:
                    if run_timestretch(cli_path, source_path, stream_path,
                                       ratio, streaming=True):
                        results["success"].append(stream_name)
                    else:
                        results["failed"].append(stream_name)

                manifest.append({
                    "track_id": track_id,
                    "target_bpm": target_bpm,
                    "ratio": ratio,
                    "mode": "streaming",
                    "path": stream_path,
                    "stereo": track.get("stereo", False),
                    "description": f"{track_id} @ {target_bpm} BPM (streaming)",
                })

    # Summary
    total = len(results["success"]) + len(results["failed"])
    log.info(f"Complete: {len(results['success'])}/{total} succeeded")
    if results["failed"]:
        log.error(f"Failed: {results['failed']}")

    # Write manifest
    if args.manifest_out:
        with open(args.manifest_out, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info(f"Manifest written: {args.manifest_out}")

    sys.exit(0 if not results["failed"] else 1)


if __name__ == "__main__":
    main()

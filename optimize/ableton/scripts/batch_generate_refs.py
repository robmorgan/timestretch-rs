#!/usr/bin/env python3
"""
Batch orchestrator for generating Ableton Complex Pro references.

Processes all entries in track_catalog.json sequentially (Ableton can only
render one project at a time). Supports resuming interrupted batches.

Usage:
    python batch_generate_refs.py track_catalog.json refs/ableton/
    python batch_generate_refs.py track_catalog.json refs/ableton/ --skip-existing
    python batch_generate_refs.py --dry-run track_catalog.json refs/ableton/
"""
import argparse
import json
import logging
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from render_via_ableton import render_with_retries, is_ableton_running, quit_ableton

DEFAULT_TEMPLATE = os.path.join(SCRIPT_DIR, "..", "template", "base_template.als")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_catalog(catalog_path):
    """Load and validate the track catalog."""
    with open(catalog_path, "r") as f:
        catalog = json.load(f)

    if not isinstance(catalog, list):
        raise ValueError("Catalog must be a JSON array")

    for entry in catalog:
        if "id" not in entry:
            raise ValueError(f"Catalog entry missing 'id': {entry}")
        if "source" not in entry:
            raise ValueError(f"Catalog entry missing 'source': {entry}")
        if "targets" not in entry:
            raise ValueError(f"Catalog entry missing 'targets': {entry}")

    return catalog


def ref_filename(track_id, target_bpm):
    """Generate a consistent reference filename."""
    return f"{track_id}_{target_bpm}bpm.wav"


def resolve_source_path(source, catalog_dir):
    """Resolve the source path relative to the catalog directory."""
    if os.path.isabs(source):
        return source
    return os.path.normpath(os.path.join(catalog_dir, source))


def generate_manifest(catalog, output_dir):
    """Generate an ableton_manifest.json mapping references to library outputs."""
    manifest = []
    for track in catalog:
        catalog_dir = os.path.dirname(os.path.abspath(track.get("_catalog_path", ".")))
        source_path = resolve_source_path(track["source"], catalog_dir)

        for target in track["targets"]:
            filename = ref_filename(track["id"], target["target_bpm"])
            ref_path = os.path.join(output_dir, filename)

            manifest.append({
                "track_id": track["id"],
                "source": source_path,
                "source_bpm": track.get("source_bpm"),
                "target_bpm": target["target_bpm"],
                "ratio": target["ratio"],
                "stereo": track.get("stereo", False),
                "ref_path": ref_path,
                "description": f"{track['id']} @ {target['target_bpm']} BPM",
            })

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate Ableton Complex Pro references"
    )
    parser.add_argument("catalog", help="Path to track_catalog.json")
    parser.add_argument("output_dir", help="Output directory for reference WAVs")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE,
                        help=f"Path to .als template (default: {DEFAULT_TEMPLATE})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tracks that already have output files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be rendered without doing it")
    parser.add_argument("--retries", type=int, default=2,
                        help="Max retries per track (default: 2)")
    parser.add_argument("--log-dir",
                        default=os.path.join(SCRIPT_DIR, "..", "logs"),
                        help="Directory for log files")
    parser.add_argument("--manifest-out",
                        help="Write ableton_manifest.json to this path")

    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    catalog_dir = os.path.dirname(os.path.abspath(args.catalog))

    # Attach catalog path for source resolution
    for track in catalog:
        track["_catalog_path"] = args.catalog

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up file logging
    log_path = os.path.join(args.log_dir, "render.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Validate template
    template_path = os.path.abspath(args.template)
    if not os.path.exists(template_path):
        log.error(f"Template not found: {template_path}")
        log.error("Create the template manually in Ableton first. See README.md.")
        sys.exit(1)

    # Build render queue
    queue = []
    for track in catalog:
        source_path = resolve_source_path(track["source"], catalog_dir)
        if not os.path.exists(source_path):
            log.warning(f"Source not found, skipping: {source_path}")
            continue

        for target in track["targets"]:
            filename = ref_filename(track["id"], target["target_bpm"])
            output_path = os.path.join(args.output_dir, filename)

            if args.skip_existing and os.path.exists(output_path):
                log.info(f"Skipping (exists): {filename}")
                continue

            queue.append({
                "track_id": track["id"],
                "source": source_path,
                "target_bpm": target["target_bpm"],
                "ratio": target["ratio"],
                "output": output_path,
                "filename": filename,
            })

    if not queue:
        log.info("Nothing to render (all files exist or no valid sources)")
        sys.exit(0)

    log.info(f"Render queue: {len(queue)} tracks")

    if args.dry_run:
        for item in queue:
            print(f"  {item['filename']}: {item['source']} -> "
                  f"{item['target_bpm']} BPM (ratio={item['ratio']})")
        print(f"\n{len(queue)} tracks would be rendered.")
        sys.exit(0)

    # Ensure Ableton is not already running
    if is_ableton_running():
        log.warning("Ableton is already running, closing it first...")
        quit_ableton()
        time.sleep(5)

    # Process queue
    results = {"success": [], "failed": []}
    total = len(queue)

    for i, item in enumerate(queue, 1):
        log.info(f"[{i}/{total}] Rendering {item['filename']}...")
        start_time = time.time()

        success = render_with_retries(
            template_path, item["source"], item["target_bpm"],
            item["output"], max_retries=args.retries
        )

        elapsed = time.time() - start_time

        if success:
            results["success"].append(item["filename"])
            log.info(f"[{i}/{total}] OK: {item['filename']} ({elapsed:.1f}s)")
        else:
            results["failed"].append(item["filename"])
            log.error(f"[{i}/{total}] FAILED: {item['filename']} ({elapsed:.1f}s)")

        # Brief pause between tracks
        if i < total:
            time.sleep(3)

    # Summary
    log.info(f"\nBatch complete: {len(results['success'])}/{total} succeeded")
    if results["failed"]:
        log.error(f"Failed: {results['failed']}")

    # Write manifest if requested
    if args.manifest_out:
        manifest = generate_manifest(catalog, args.output_dir)
        with open(args.manifest_out, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info(f"Manifest written: {args.manifest_out}")

    sys.exit(0 if not results["failed"] else 1)


if __name__ == "__main__":
    main()

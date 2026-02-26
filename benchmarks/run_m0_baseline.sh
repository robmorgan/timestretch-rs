#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TIMESTRETCH_STRICT_REFERENCE_BENCHMARK=1
export TIMESTRETCH_REFERENCE_MAX_SECONDS=30

echo "Running strict reference-quality benchmark..."
cargo test --test reference_quality -- --nocapture

REPORT_PATH="$ROOT_DIR/benchmarks/audio/output/report.json"
if [[ ! -f "$REPORT_PATH" ]]; then
  echo "ERROR: benchmark report not found at $REPORT_PATH" >&2
  exit 1
fi

ARCHIVE_DIR="$ROOT_DIR/benchmarks/baselines"
mkdir -p "$ARCHIVE_DIR"

TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
LATEST_PATH="$ARCHIVE_DIR/m0_baseline_latest.json"
STAMPED_PATH="$ARCHIVE_DIR/m0_baseline_${TIMESTAMP}.json"

cp "$REPORT_PATH" "$LATEST_PATH"
cp "$REPORT_PATH" "$STAMPED_PATH"

echo "Baseline archived:"
echo "  $LATEST_PATH"
echo "  $STAMPED_PATH"

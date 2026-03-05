#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TIMESTRETCH_STRICT_REFERENCE_BENCHMARK=1
export TIMESTRETCH_REFERENCE_MAX_SECONDS=30
TIMEOUT_SECS="${TIMESTRETCH_REFERENCE_TIMEOUT_SECS:-900}"

echo "Running strict reference-quality benchmark..."
if command -v python3 >/dev/null 2>&1; then
  python3 - "$TIMEOUT_SECS" <<'PY'
import subprocess
import sys

timeout = float(sys.argv[1])
cmd = ["cargo", "test", "--release", "--test", "reference_quality", "--", "--nocapture"]
try:
    completed = subprocess.run(cmd, check=False, timeout=timeout)
except subprocess.TimeoutExpired:
    print(f"ERROR: reference_quality timed out after {timeout:.0f}s", file=sys.stderr)
    sys.exit(124)
sys.exit(completed.returncode)
PY
else
  cargo test --release --test reference_quality -- --nocapture
fi

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

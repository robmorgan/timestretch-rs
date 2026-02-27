#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: check_tiers.sh [fast|quality|full] [--dry-run]

Modes:
  fast      Run formatting, clippy, and all-target tests.
  quality   Run fast tier plus quality-focused integration tests.
  full      Run quality tier plus heavy benchmark/reference checks.

Options:
  --dry-run Print commands without executing.
  -h, --help Show this help message.
EOF
}

mode="fast"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    fast|quality|full)
      mode="$1"
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run_cmd() {
  local label="$1"
  shift
  echo
  echo "==> ${label}"
  echo "+ $*"
  if [[ "$dry_run" -eq 0 ]]; then
    "$@"
  fi
}

echo "Running tier '${mode}' checks..."
[[ "$dry_run" -eq 1 ]] && echo "Dry-run mode enabled."

run_cmd "Format check" cargo fmt --all --check
run_cmd "Clippy (warnings denied)" cargo clippy --all-targets -- -D warnings
run_cmd "All-target test suite" cargo test --all-targets

if [[ "$mode" == "quality" || "$mode" == "full" ]]; then
  run_cmd "Quality integration tests" cargo test --test quality -- --nocapture
  run_cmd "Quality gates" cargo test --test quality_gates -- --nocapture
fi

if [[ "$mode" == "full" ]]; then
  echo
  echo "Full tier includes long-running DSP checks."
  run_cmd "Reference-quality benchmark test" cargo test --test reference_quality -- --nocapture
  run_cmd "Benchmark quality example" cargo run --release --example benchmark_quality
fi

echo
echo "Tier '${mode}' completed."

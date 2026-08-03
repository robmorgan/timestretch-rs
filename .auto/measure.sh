#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Fast pre-check: compile errors surface quickly.
cargo build --release --example stream_quality_bench 2>&1 | tail -20

# Deterministic streaming quality benchmark (engine Keylock profile).
cargo run -q --release --example stream_quality_bench

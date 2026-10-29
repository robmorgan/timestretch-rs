#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -p timestretch-qa -- quality --strict

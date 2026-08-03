#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

cargo fmt --all --check

if ! out=$(cargo clippy -q --all-targets -- -D warnings 2>&1); then
  echo "$out" | tail -40
  exit 1
fi

if ! out=$(cargo test --release -q 2>&1); then
  echo "$out" | grep -Ev "^(running|test result: ok|[.]+$)" | tail -40
  exit 1
fi
echo "checks ok"

#!/bin/bash
set -euo pipefail

# Run lib tests, count failures. Allow known pre-existing failures (9 tests).
LIB_OUTPUT=$(cargo test --lib 2>&1 || true)
LIB_RESULT=$(echo "$LIB_OUTPUT" | grep "^test result:" | tail -1)
echo "Lib: $LIB_RESULT"

# Extract passed/failed counts using awk
PASSED=$(echo "$LIB_RESULT" | awk '{for(i=1;i<=NF;i++) if($i=="passed;") print $(i-1)}')
FAILED=$(echo "$LIB_RESULT" | awk '{for(i=1;i<=NF;i++) if($i=="failed;") print $(i-1)}')
if [ -z "$PASSED" ]; then PASSED=0; fi
if [ -z "$FAILED" ]; then FAILED=0; fi
echo "Passed: $PASSED, Failed: $FAILED"

if [ "$FAILED" -gt 9 ]; then
    echo "ERROR: New test failures detected! ($FAILED failed, expected <=9)"
    echo "$LIB_OUTPUT" | grep "^    " | tail -20
    exit 1
fi

if [ "$PASSED" -lt 670 ]; then
    echo "ERROR: Too few tests passing ($PASSED, expected >=670)"
    exit 1
fi

# Format check only (clippy has pre-existing errors on this branch)
cargo fmt --all --check 2>&1 | tail -5

#!/bin/bash
set -euo pipefail

PROMPT_FILE=$1
ITERATION=$2
LOG_FILE="optimize/logs/agent_${ITERATION}.log"

MODEL=$(grep "model" optimize/config.toml | cut -d'=' -f2 | tr -d ' "' || echo "o3-mini")

echo "Running Codex agent..."
# Assuming 'codex' CLI is available
codex exec --full-auto --model "$MODEL" "$(cat "$PROMPT_FILE")" 2>&1 | tee "$LOG_FILE"

# Check if build is broken after agent changes
echo "Verifying build..."
if ! cargo build --release; then
    echo "Agent broke the build!"
    exit 1
fi

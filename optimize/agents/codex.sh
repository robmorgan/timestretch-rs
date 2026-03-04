#!/bin/bash
set -euo pipefail

PROMPT_FILE=$1
ITERATION=$2

# Derive log directory from the prompt file location (works for both rubberband and ableton loops)
LOG_DIR="$(dirname "$PROMPT_FILE")"
LOG_FILE="$LOG_DIR/agent_${ITERATION}.log"

MODEL=$(grep "model" optimize/config.toml | cut -d'=' -f2 | tr -d ' "' || echo "o3-mini")
MAX_TURNS=$(grep "max_turns" optimize/config.toml | cut -d'=' -f2 | tr -d ' ' || echo 30)

echo "Running Codex agent (model=$MODEL, max_turns=$MAX_TURNS)..."
codex exec --full-auto --model "$MODEL" --max-turns "$MAX_TURNS" "$(cat "$PROMPT_FILE")" 2>&1 | tee "$LOG_FILE"

# Check if build is broken after agent changes
echo "Verifying build..."
if ! cargo build --release; then
    echo "Agent broke the build!"
    exit 1
fi

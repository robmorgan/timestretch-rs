#!/bin/bash
set -euo pipefail

PROMPT_FILE=$1
ITERATION=$2
LOG_FILE="optimize/logs/agent_${ITERATION}.log"

MAX_TURNS=$(grep "max_turns" optimize/config.toml | cut -d'=' -f2 | tr -d ' ' || echo 10)

echo "Running Claude Code agent..."
# Assuming 'claude' CLI is available
# Using --non-interactive if supported or just piping/passing prompt
claude -p "$(cat "$PROMPT_FILE")" --allowedTools Edit,Write,Bash --max-turns "$MAX_TURNS" 2>&1 | tee "$LOG_FILE"

# Check if build is broken after agent changes
echo "Verifying build..."
if ! cargo build --release; then
    echo "Agent broke the build!"
    exit 1
fi

#!/bin/bash
set -euo pipefail

PROMPT_FILE=$1
ITERATION=$2
LOG_FILE="optimize/logs/agent_${ITERATION}.log"

MAX_TURNS=$(grep "max_turns" optimize/config.toml | cut -d'=' -f2 | tr -d ' ' || echo 10)

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate venv so the agent subprocess inherits it
if [ -f "optimize/.venv/bin/activate" ]; then
    source "optimize/.venv/bin/activate"
fi

echo "=== Agent Prompt (Iteration $ITERATION) ==="
cat "$PROMPT_FILE"
echo ""
echo "==========================================="
echo "Running Claude Code agent..."
claude -p "$(cat "$PROMPT_FILE")" --model claude-opus-4-6 --allowedTools Edit,Write,Bash,Read,Grep,Glob --max-turns "$MAX_TURNS" --verbose --output-format stream-json 2>&1 \
    | tee "$LOG_FILE" \
    | python3 "$AGENT_DIR/stream_filter.py"

# Check if build is broken after agent changes
echo "Verifying build..."
if ! cargo build --release; then
    echo "Agent broke the build!"
    exit 1
fi

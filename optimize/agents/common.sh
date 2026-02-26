#!/bin/bash
# common.sh - Shared agent interface

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_PATH="$REPO_ROOT/optimize/config.toml"

get_config_val() {
    local key=$1
    grep "^$key" "$CONFIG_PATH" | cut -d'=' -f2 | tr -d ' ",' | xargs
}

run_agent() {
    local prompt_file=$1
    local iteration=$2
    local agent_type=$(get_config_val "agent")
    
    echo "Invoking agent: $agent_type (Iteration $iteration)"
    
    case "$agent_type" in
        "claude_code")
            "$SCRIPT_DIR/claude_code.sh" "$prompt_file" "$iteration"
            ;;
        "codex")
            "$SCRIPT_DIR/codex.sh" "$prompt_file" "$iteration"
            ;;
        *)
            echo "Error: Unknown agent type $agent_type"
            return 1
            ;;
    esac
}

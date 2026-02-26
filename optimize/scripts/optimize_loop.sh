#!/bin/bash
set -euo pipefail

# optimize_loop.sh - Main orchestration loop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="$REPO_ROOT/config.toml"
LOG_DIR="$REPO_ROOT/logs"
PROGRESS_CSV="$LOG_DIR/progress.csv"

cd "$REPO_ROOT/.." # Go to real repo root

# Activate venv if present
if [ -f "optimize/.venv/bin/activate" ]; then
    source "optimize/.venv/bin/activate"
fi

# Load common agent functions
source "optimize/agents/common.sh"

# Helper to get config values
get_config() {
    python3 -c "import tomllib; 
with open('$CONFIG_PATH', 'rb') as f:
    config = tomllib.load(f)
    keys = '$1'.split('.')
    val = config
    for k in keys:
        val = val.get(k, {})
    print(val if not isinstance(val, dict) else '')
" 2>/dev/null || 
    python3 -c "import toml; 
with open('$CONFIG_PATH', 'r') as f:
    config = toml.load(f)
    keys = '$1'.split('.')
    val = config
    for k in keys:
        val = val.get(k, {})
    print(val if not isinstance(val, dict) else '')
" 2>/dev/null || 
    grep "^$1" "$CONFIG_PATH" | cut -d'=' -f2 | tr -d ' ",' | xargs
}

MAX_ITERATIONS=$(get_config "general.max_iterations")
TARGET_SCORE=$(get_config "general.target_score")
DRY_RUN=0
RESUME=0

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --dry-run  Score without agent modification"
    echo "  --resume   Resume from last iteration"
    echo "  --help     Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=1; shift ;;
        --resume) RESUME=1; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Pre-flight checks
if [ ! -f "test_manifest.json" ]; then
    echo "Error: test_manifest.json not found. Run optimize/scripts/generate_references.sh --generate-manifest first."
    exit 1
fi

mkdir -p "$LOG_DIR"
if [ ! -f "$PROGRESS_CSV" ]; then
    echo "iteration,timestamp,avg_score,worst_score,worst_case,agent,git_sha" > "$PROGRESS_CSV"
fi

START_ITER=1
if [ $RESUME -eq 1 ]; then
    START_ITER=$(tail -n +2 "$PROGRESS_CSV" | wc -l | xargs -I{} echo "{}+1" | bc || echo 1)
    echo "Resuming from iteration $START_ITER"
fi

for (( i=START_ITER; i<=MAX_ITERATIONS; i++ )); do
    echo "--- Iteration $i ---"
    
    # 1. Run timestretch-rs
    ./optimize/scripts/run_test_suite.py
    
    # 2. Score
    SCORES_JSON="$LOG_DIR/scores_$i.json"
    ./optimize/scripts/score.py --batch "$SCORES_JSON"
    
    AVG_SCORE=$(python3 -c "import json; data=json.load(open('$SCORES_JSON')); print(sum(r['total_score'] for r in data)/len(data))")
    WORST_SCORE=$(python3 -c "import json; data=json.load(open('$SCORES_JSON')); print(min(r['total_score'] for r in data))")
    WORST_CASE=$(python3 -c "import json; data=json.load(open('$SCORES_JSON')); print(min(data, key=lambda x: x['total_score'])['description'])")
    
    echo "Iteration $i Average Score: $AVG_SCORE (Worst: $WORST_SCORE - $WORST_CASE)"
    
    # Log progress
    GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")
    echo "$i,$(date -u +%Y-%m-%dT%H:%M:%SZ),$AVG_SCORE,$WORST_SCORE,"$WORST_CASE",$(get_config "general.agent"),$GIT_SHA" >> "$PROGRESS_CSV"
    
    if python3 -c "import sys; sys.exit(0 if $AVG_SCORE >= $TARGET_SCORE else 1)"; then
        echo "Target score reached! Converged at iteration $i."
        break
    fi
    
    if [ $DRY_RUN -eq 1 ]; then
        echo "Dry run enabled. Stopping."
        break
    fi
    
    # 3. Generate spectrograms for worst 3 cases
    mkdir -p "$LOG_DIR/spectrograms/$i"
    python3 -c "import json, os, subprocess;
data = json.load(open('$SCORES_JSON'))
sorted_data = sorted(data, key=lambda x: x['total_score'])[:3]
for item in sorted_data:
    desc_slug = item['description'].replace(' ', '_').lower()
    ratio = item['ratio']
    source_base = os.path.basename(item['description']).replace(' ', '_') # Fallback if source not in score
    # Finding actual paths is better:
    # We need to re-find the paths or include them in score.py output
    # For now, let's assume standard naming
    ref = f'optimize/references/{source_base}_ref_{ratio}.wav'
    test = f'optimize/outputs/{source_base}_test_{ratio}.wav'
    out = f'$LOG_DIR/spectrograms/$i/{desc_slug}.png'
    # This is a bit brittle, score.py should ideally output paths
"
    
    # 4. Prepare Agent Prompt
    PROMPT_FILE="$LOG_DIR/prompt_$i.md"
    SCORE_HISTORY=$(tail -n 3 "$PROGRESS_CSV" | cut -d',' -f3 | tr '
' ' ' | xargs)
    JSON_SCORES=$(cat "$SCORES_JSON")
    
    # Export vars for envsubst
    export ITERATION=$i
    export AVG_SCORE=$AVG_SCORE
    export TARGET_SCORE=$TARGET_SCORE
    export SCORE_HISTORY="$SCORE_HISTORY"
    export JSON_SCORES="$JSON_SCORES"
    export WORST_CASES=$(python3 -c "
import json
data = json.load(open('$SCORES_JSON'))
sorted_data = sorted(data, key=lambda x: x['total_score'])[:3]
for d in sorted_data:
    print(f'- {d[\"description\"]}: {d[\"total_score\"]:.2f}')
")
    
    envsubst < "optimize/scripts/agent_prompt.md.tmpl" > "$PROMPT_FILE"
    
    # 5. Run Agent (agent self-scores and commits if improved)
    AGENT_OK=1
    if ! run_agent "$PROMPT_FILE" "$i"; then
        echo "Agent failed or build broken."
        AGENT_OK=0
    fi

    # 6. Safety net: verify and commit any uncommitted improvements
    if git diff --quiet src/ && git diff --cached --quiet src/; then
        echo "No uncommitted changes in src/."
    elif [ $AGENT_OK -eq 0 ]; then
        echo "Agent failed. Reverting uncommitted changes..."
        git checkout -- src/
    else
        echo "Uncommitted changes detected. Re-scoring to check for improvement..."
        if cargo build --release 2>/dev/null; then
            ./optimize/scripts/run_test_suite.py
            VERIFY_JSON="$LOG_DIR/scores_${i}_verify.json"
            ./optimize/scripts/score.py --batch "$VERIFY_JSON"
            NEW_AVG=$(python3 -c "import json; data=json.load(open('$VERIFY_JSON')); print(sum(r['total_score'] for r in data)/len(data))")
            echo "Post-agent score: $NEW_AVG (was: $AVG_SCORE)"
            if python3 -c "import sys; sys.exit(0 if $NEW_AVG > $AVG_SCORE else 1)"; then
                echo "Score improved! Committing uncommitted agent changes..."
                git add src/
                git commit -m "opt(loop): auto-commit agent improvement, score=$NEW_AVG (was $AVG_SCORE)"
            else
                echo "Score did not improve ($NEW_AVG <= $AVG_SCORE). Reverting..."
                git checkout -- src/
            fi
        else
            echo "Build failed with uncommitted changes. Reverting..."
            git checkout -- src/
        fi
    fi

    # Cool off before next iteration
    if [ $i -lt $MAX_ITERATIONS ]; then
        echo "Cooling off for 30 seconds..."
        sleep 30
    fi
done

# 7. Final Report
./optimize/scripts/report.py

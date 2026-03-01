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
PLATEAU_LIMIT=8  # Break after this many consecutive non-improving iterations
STALE_COUNT=0
BEST_SCORE="0"

if [ $RESUME -eq 1 ]; then
    START_ITER=$(tail -n +2 "$PROGRESS_CSV" | wc -l | xargs -I{} echo "{}+1" | bc || echo 1)
    # Recover best score from progress CSV
    BEST_SCORE=$(tail -n +2 "$PROGRESS_CSV" | cut -d',' -f3 | sort -rn | head -1 || echo "0")
    echo "Resuming from iteration $START_ITER (best score so far: $BEST_SCORE)"
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
    BATCH_AVG_SCORE=$(python3 -c "import json; data=json.load(open('$SCORES_JSON')); b=[r['total_score'] for r in data if r.get('mode')=='batch']; print(sum(b)/len(b) if b else 0)")
    STREAM_AVG_SCORE=$(python3 -c "import json; data=json.load(open('$SCORES_JSON')); s=[r['total_score'] for r in data if r.get('mode')=='streaming']; print(sum(s)/len(s) if s else 0)")

    echo "Iteration $i Average Score: $AVG_SCORE (Batch: $BATCH_AVG_SCORE, Streaming: $STREAM_AVG_SCORE, Worst: $WORST_SCORE - $WORST_CASE)"

    # Plateau detection: track consecutive non-improving iterations
    if python3 -c "import sys; sys.exit(0 if float('$AVG_SCORE') > float('$BEST_SCORE') + 0.01 else 1)" 2>/dev/null; then
        BEST_SCORE="$AVG_SCORE"
        STALE_COUNT=0
    else
        STALE_COUNT=$((STALE_COUNT + 1))
        echo "  [plateau: $STALE_COUNT/$PLATEAU_LIMIT consecutive non-improving iterations]"
    fi

    # Log progress
    GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")
    echo "$i,$(date -u +%Y-%m-%dT%H:%M:%SZ),$AVG_SCORE,$WORST_SCORE,"$WORST_CASE",$(get_config "general.agent"),$GIT_SHA" >> "$PROGRESS_CSV"
    
    if python3 -c "import sys; sys.exit(0 if $AVG_SCORE >= $TARGET_SCORE else 1)"; then
        echo "Target score reached! Converged at iteration $i."
        break
    fi

    if [ $STALE_COUNT -ge $PLATEAU_LIMIT ]; then
        echo "Plateau detected: no improvement in $PLATEAU_LIMIT consecutive iterations (best: $BEST_SCORE). Stopping."
        echo "Consider: adjusting the algorithm architecture, changing the scoring methodology, or targeting specific weak test cases."
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

    # Load previous attempts log (if it exists) — show ALL attempts so the
    # agent doesn't repeat approaches from early iterations
    ATTEMPTS_FILE="$LOG_DIR/attempts.log"
    if [ -f "$ATTEMPTS_FILE" ] && [ -s "$ATTEMPTS_FILE" ]; then
        PREVIOUS_ATTEMPTS=$(cat "$ATTEMPTS_FILE")
    else
        PREVIOUS_ATTEMPTS="No previous attempts logged yet. You are the first agent."
    fi

    # Extract key source code sections for the prompt
    HYBRID_HEAD=$(sed -n '1,200p' src/stretch/hybrid.rs)
    HYBRID_STRETCH_CORE=$(sed -n '585,677p' src/stretch/hybrid.rs)
    PV_PROCESS=$(sed -n '299,500p' src/stretch/phase_vocoder.rs)
    WSOLA_CORE=$(sed -n '99,465p' src/stretch/wsola.rs)

    # Diversity pressure: if score unchanged for 3+ iterations, analyse the
    # attempts log to suggest directions that have NOT been tried or that
    # showed partial promise (small regressions that might work differently)
    DIVERSITY_STALE=$(tail -n 5 "$PROGRESS_CSV" | cut -d',' -f3 | sort -u | wc -l | tr -d ' ')
    if [ "$DIVERSITY_STALE" -eq 1 ] && [ "$i" -gt 3 ]; then
        DIVERSITY_HINTS=$(python3 -c "
import re, collections

# Read all attempts
attempts = open('$ATTEMPTS_FILE').read() if '$ATTEMPTS_FILE' != '' else ''

# Categorize past attempts by area
categories = {
    'PV hop/overlap': ['hop', 'overlap', 'hop_size', 'fft/'],
    'HPSS params': ['hpss', 'harmonic_width', 'percussive_width', 'wiener', 'mask power'],
    'Window function': ['window', 'hann', 'blackman'],
    'Crossfade/blending': ['crossfade', 'fade', 'blend', 'taper'],
    'Mirror padding': ['mirror', 'padding', 'pad_mult', 'start_pad', 'end_pad'],
    'Phase locking': ['phase lock', 'phase_gradient', 'adaptive phase', 'roi', 'identity'],
    'Transient detection': ['transient', 'sensitivity', 'onset'],
    'WSOLA search/overlap': ['wsola', 'search_ms', 'seg_size', 'wsola overlap'],
    'Edge correction': ['edge correct', 'gain ramp', 'correction_len', 'cubic hermite'],
    'Sub-bass handling': ['sub.bass', 'cutoff', '120hz', '85hz'],
    'Multi-resolution': ['multi_res', 'multi.resolution', '3-band'],
    'Normalization': ['normaliz', 'window_sum', 'floor_ratio'],
    'Spectral envelope': ['cepstr', 'spectral envelope'],
    'Band-split': ['band.split', 'crossover'],
    'Streaming chunk size': ['chunk_size', 'streaming chunk'],
}

# Count attempts per category
cat_counts = collections.Counter()
cat_results = collections.defaultdict(list)
lines = attempts.strip().split('\\n')
for line in lines:
    lower = line.lower()
    for cat, keywords in categories.items():
        if any(kw in lower for kw in keywords):
            cat_counts[cat] += 1
            # Extract score delta if possible
            m = re.search(r'score=([\\d.]+)\\s*\\(was\\s*([\\d.]+)', line)
            if m:
                delta = float(m.group(1)) - float(m.group(2))
                cat_results[cat].append(delta)

# Find under-explored and unexplored areas
all_cats = list(categories.keys())
unexplored = [c for c in all_cats if cat_counts[c] == 0]
under_explored = [(c, cat_counts[c]) for c in all_cats if 0 < cat_counts[c] <= 2]
# Near-misses: categories where at least one attempt was close to improving
near_miss = [(c, max(cat_results[c])) for c in all_cats
             if cat_results[c] and max(cat_results[c]) > -0.5]

print('## Diversity Suggestions (score has been flat — try something NEW)')
print()
if unexplored:
    print('### Unexplored Areas (never tried):')
    for c in unexplored[:4]:
        print(f'- **{c}**')
    print()
if under_explored:
    print('### Under-Explored Areas (tried only 1-2x):')
    for c, n in sorted(under_explored, key=lambda x: x[1])[:4]:
        print(f'- **{c}** ({n} attempt(s))')
    print()
if near_miss:
    print('### Near-Misses (small regressions — might work with a different angle):')
    for c, delta in sorted(near_miss, key=lambda x: x[1], reverse=True)[:3]:
        print(f'- **{c}** (best delta: {delta:+.2f})')
    print()
# Over-explored warning
over_explored = [(c, cat_counts[c]) for c in all_cats if cat_counts[c] >= 5]
if over_explored:
    print('### Over-Explored (diminishing returns — AVOID these):')
    for c, n in sorted(over_explored, key=lambda x: x[1], reverse=True)[:4]:
        print(f'- ~~{c}~~ ({n} attempts, none successful recently)')
")
    else
        DIVERSITY_HINTS=""
    fi

    # Calculate per-metric impact analysis: which metric+case has the most
    # leverage on the overall average score
    IMPACT_ANALYSIS=$(python3 -c "
import json

data = json.load(open('$SCORES_JSON'))
n = len(data)
weights = {'spectral_convergence': 0.30, 'log_spectral_distance': 0.25,
           'mfcc_distance': 0.20, 'transient_preservation': 0.25}

# For each test case and metric, compute how much the overall average
# would improve if that metric reached 100
impacts = []
for item in data:
    for metric, weight in weights.items():
        current = item['metrics'][metric]
        headroom = 100.0 - current
        # Impact on overall average = (headroom * weight) / n
        impact = (headroom * weight) / n
        if impact > 0.3:  # Only show meaningful opportunities
            impacts.append((impact, item['description'], metric, current))

impacts.sort(reverse=True)
print('## Highest-Impact Opportunities')
print('Improving these specific metrics would have the largest effect on the overall score:')
print('| Impact | Test Case | Metric | Current | Headroom |')
print('|--------|-----------|--------|---------|----------|')
for impact, desc, metric, current in impacts[:8]:
    print(f'| +{impact:.2f} | {desc} | {metric} | {current:.1f} | {100-current:.1f} |')
print()
print('**Focus on the top 2-3 rows** — these are where your changes will move the needle most.')
")

    # Export vars for envsubst
    export ITERATION=$i
    export AVG_SCORE=$AVG_SCORE
    export TARGET_SCORE=$TARGET_SCORE
    export BATCH_AVG_SCORE=$BATCH_AVG_SCORE
    export STREAM_AVG_SCORE=$STREAM_AVG_SCORE
    export SCORE_HISTORY="$SCORE_HISTORY"
    export JSON_SCORES="$JSON_SCORES"
    export PREVIOUS_ATTEMPTS="$PREVIOUS_ATTEMPTS"
    export HYBRID_HEAD="$HYBRID_HEAD"
    export HYBRID_STRETCH_CORE="$HYBRID_STRETCH_CORE"
    export PV_PROCESS="$PV_PROCESS"
    export WSOLA_CORE="$WSOLA_CORE"
    export DIVERSITY_HINTS="$DIVERSITY_HINTS"
    export IMPACT_ANALYSIS="$IMPACT_ANALYSIS"
    export WORST_CASES=$(python3 -c "
import json
data = json.load(open('$SCORES_JSON'))
sorted_data = sorted(data, key=lambda x: x['total_score'])[:3]
for d in sorted_data:
    print(f'- {d[\"description\"]}: {d[\"total_score\"]:.2f}')
")

    envsubst '$ITERATION $AVG_SCORE $TARGET_SCORE $BATCH_AVG_SCORE $STREAM_AVG_SCORE $SCORE_HISTORY $JSON_SCORES $PREVIOUS_ATTEMPTS $HYBRID_HEAD $HYBRID_STRETCH_CORE $PV_PROCESS $WSOLA_CORE $DIVERSITY_HINTS $IMPACT_ANALYSIS $WORST_CASES' \
        < "optimize/scripts/agent_prompt.md.tmpl" > "$PROMPT_FILE"
    
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

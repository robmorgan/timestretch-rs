#!/bin/bash
set -euo pipefail

# optimize_loop.sh - Ableton Complex Pro optimization loop
# Scores library outputs against pre-generated Ableton references.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLETON_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ABLETON_DIR/../.." && pwd)"
CONFIG_PATH="$REPO_ROOT/optimize/config.toml"
LOG_DIR="$ABLETON_DIR/logs"
PROGRESS_CSV="$LOG_DIR/progress.csv"

cd "$REPO_ROOT"

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
ABLETON_MANIFEST="$ABLETON_DIR/ableton_manifest.json"
TRACK_CATALOG="$ABLETON_DIR/track_catalog.json"
LIBRARY_REF_DIR="$ABLETON_DIR/refs/library"

if [ ! -f "$ABLETON_MANIFEST" ]; then
    echo "Error: $ABLETON_MANIFEST not found. Run 'cd optimize/ableton && make ableton-refs' first."
    exit 1
fi

if [ ! -f "$TRACK_CATALOG" ]; then
    echo "Error: $TRACK_CATALOG not found."
    exit 1
fi

mkdir -p "$LOG_DIR"
if [ ! -f "$PROGRESS_CSV" ]; then
    echo "iteration,timestamp,avg_score,worst_score,worst_case,agent,git_sha" > "$PROGRESS_CSV"
fi

START_ITER=1
STALE_COUNT=0
BEST_SCORE="0"

if [ $RESUME -eq 1 ]; then
    START_ITER=$(tail -n +2 "$PROGRESS_CSV" | wc -l | xargs -I{} echo "{}+1" | bc || echo 1)
    # Recover best score from progress CSV
    BEST_SCORE=$(tail -n +2 "$PROGRESS_CSV" | cut -d',' -f3 | sort -rn | head -1 || echo "0")
    echo "Resuming from iteration $START_ITER (best score so far: $BEST_SCORE)"
fi

for (( i=START_ITER; i<=MAX_ITERATIONS; i++ )); do
    echo "--- Iteration $i (Ableton) ---"

    # 1. Build the release binary (src/ may have changed)
    echo "Building release binary..."
    cargo build --release

    # 2. Clean stale library outputs and regenerate
    echo "Cleaning stale library outputs..."
    rm -f "$LIBRARY_REF_DIR"/*.wav
    echo "Generating library outputs..."
    NCPU=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
    python3 "$ABLETON_DIR/scripts/generate_library_refs.py" \
        "$TRACK_CATALOG" "$LIBRARY_REF_DIR" \
        --parallel "$NCPU" \
        --manifest-out "$ABLETON_DIR/library_manifest.json"

    # 3. Score against Ableton references
    SCORES_JSON="$LOG_DIR/scores_$i.json"
    python3 optimize/scripts/score.py \
        --ref-source ableton \
        --ableton-manifest "optimize/ableton/$( basename "$ABLETON_MANIFEST" )" \
        --batch "$SCORES_JSON"

    AVG_SCORE=$(python3 -c "import json; data=json.load(open('$SCORES_JSON')); print(sum(r['total_score'] for r in data)/len(data))")
    WORST_SCORE=$(python3 -c "import json; data=json.load(open('$SCORES_JSON')); print(min(r['total_score'] for r in data))")
    WORST_CASE=$(python3 -c "import json; data=json.load(open('$SCORES_JSON')); print(min(data, key=lambda x: x['total_score'])['description'])")
    BATCH_AVG_SCORE=$(python3 -c "import json; data=json.load(open('$SCORES_JSON')); b=[r['total_score'] for r in data if r.get('mode')=='batch']; print(sum(b)/len(b) if b else 0)")
    STREAM_AVG_SCORE=$(python3 -c "import json; data=json.load(open('$SCORES_JSON')); s=[r['total_score'] for r in data if r.get('mode')=='streaming']; print(sum(s)/len(s) if s else 0)")

    echo "Iteration $i Average Score: $AVG_SCORE (Batch: $BATCH_AVG_SCORE, Streaming: $STREAM_AVG_SCORE, Worst: $WORST_SCORE - $WORST_CASE)"

    # Plateau detection: track consecutive non-improving iterations
    ATTEMPTS_FILE="$LOG_DIR/attempts.log"
    TOTAL_ATTEMPTS=$(wc -l < "$ATTEMPTS_FILE" 2>/dev/null | tr -d ' ' || echo "0")
    PLATEAU_LIMIT=$(python3 -c "print(max(3, 8 - int('$TOTAL_ATTEMPTS') // 20))" 2>/dev/null || echo "8")

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

    # 4. Prepare Agent Prompt
    PROMPT_FILE="$LOG_DIR/prompt_$i.md"
    SCORE_HISTORY=$(tail -n 3 "$PROGRESS_CSV" | cut -d',' -f3 | tr '
' ' ' | xargs)
    JSON_SCORES=$(cat "$SCORES_JSON")

    # Load previous attempts log — summarize categories + show last 20
    if [ -f "$ATTEMPTS_FILE" ] && [ -s "$ATTEMPTS_FILE" ]; then
        PREVIOUS_ATTEMPTS=$(python3 -c "
import re, collections, sys

lines = open('$ATTEMPTS_FILE').readlines()
total = len(lines)

categories = {
    'PV hop/overlap': ['hop', 'overlap', 'hop_size', 'fft/'],
    'HPSS params': ['hpss', 'harmonic_width', 'percussive_width', 'wiener', 'mask power', 'mask smooth'],
    'Window function': ['window', 'hann', 'blackman'],
    'Crossfade/blending': ['crossfade', 'fade', 'blend', 'taper'],
    'Mirror padding': ['mirror', 'padding', 'pad_mult', 'start_pad', 'end_pad'],
    'Phase locking': ['phase lock', 'phase_gradient', 'adaptive phase', 'roi', 'identity'],
    'Transient detection': ['transient', 'sensitivity', 'onset'],
    'WSOLA search/overlap': ['wsola', 'search_ms', 'seg_size', 'wsola overlap'],
    'Edge correction': ['edge correct', 'gain ramp', 'correction_len', 'cubic hermite'],
    'Sub-bass handling': ['sub.bass', 'cutoff', '120hz', '85hz'],
    'Multi-resolution': ['multi_res', 'multi.resolution', '3-band'],
    'Normalization/RMS': ['normaliz', 'window_sum', 'floor_ratio', 'rms'],
    'Spectral envelope': ['cepstr', 'spectral envelope', 'spectral tilt'],
    'Band-split': ['band.split', 'crossover'],
    'Streaming chunk/accum': ['chunk_size', 'streaming chunk', 'accumulation'],
    'Output format': ['float wav', 'pcm_16', '16-bit', 'quantiz'],
}

cat_counts = collections.Counter()
cat_best = {}
cat_last = {}
for idx, line in enumerate(lines, 1):
    lower = line.lower()
    for cat, keywords in categories.items():
        if any(kw in lower for kw in keywords):
            cat_counts[cat] += 1
            cat_last[cat] = idx
            m = re.search(r'score=([\d.]+)', line)
            if m:
                sc = float(m.group(1))
                if cat not in cat_best or sc > cat_best[cat]:
                    cat_best[cat] = sc

# Print summary
print(f'Total attempts: {total}')
print()
print('### Attempts by Category')
print('| Category | Attempts | Best Score | Last Attempt |')
print('|----------|----------|------------|-------------|')
for cat in sorted(cat_counts, key=lambda c: cat_counts[c], reverse=True):
    best = f'{cat_best[cat]:.2f}' if cat in cat_best else 'n/a'
    print(f'| {cat} | {cat_counts[cat]} | {best} | #{cat_last[cat]} |')
print()

# Show last 20 attempts in full
n = min(20, total)
print(f'### Last {n} Attempts (full details)')
for line in lines[-n:]:
    print(line.rstrip())
")
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
lines = attempts.strip().split('\n')
for line in lines:
    lower = line.lower()
    for cat, keywords in categories.items():
        if any(kw in lower for kw in keywords):
            cat_counts[cat] += 1
            # Extract score delta if possible
            m = re.search(r'score=([\d.]+)\s*\(was\s*([\d.]+)', line)
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

    # Calculate per-metric impact analysis
    IMPACT_ANALYSIS=$(python3 -c "
import json, os

data = json.load(open('$SCORES_JSON'))
n = len(data)

# Read weights from config.toml
all_weights = {
    'spectral_convergence': 0.30, 'log_spectral_distance': 0.25,
    'mfcc_distance': 0.20, 'transient_preservation': 0.25,
    'stereo_width': 0.10, 'interchannel_correlation': 0.10,
    'panning_consistency': 0.05
}
config_path = '$CONFIG_PATH'
if os.path.exists(config_path):
    try:
        import tomllib
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
        if 'scoring' in config:
            all_weights.update(config['scoring'])
    except (ImportError, Exception):
        pass

mono_keys = ['spectral_convergence', 'log_spectral_distance', 'mfcc_distance', 'transient_preservation']
stereo_keys = ['stereo_width', 'interchannel_correlation', 'panning_consistency']

# For each test case and metric, compute how much the overall average
# would improve if that metric reached 100
impacts = []
for item in data:
    is_stereo = item.get('stereo', False)
    active_keys = mono_keys + stereo_keys if is_stereo else mono_keys
    raw_sum = sum(all_weights.get(k, 0) for k in active_keys)
    if raw_sum <= 0:
        continue
    norm = {k: all_weights.get(k, 0) / raw_sum for k in active_keys}

    for metric in active_keys:
        current = item['metrics'].get(metric, 100.0)
        headroom = 100.0 - current
        w = norm.get(metric, 0)
        impact = (headroom * w) / n
        if impact > 0.2:
            impacts.append((impact, item['description'], metric, current))

impacts.sort(reverse=True)
print('## Highest-Impact Opportunities')
print('Improving these specific metrics would have the largest effect on the overall score:')
print('| Impact | Test Case | Metric | Current | Headroom |')
print('|--------|-----------|--------|---------|----------|')
for impact, desc, metric, current in impacts[:10]:
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
        < "$ABLETON_DIR/scripts/agent_prompt.md.tmpl" > "$PROMPT_FILE"

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
            rm -f "$LIBRARY_REF_DIR"/*.wav
            NCPU=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
            python3 "$ABLETON_DIR/scripts/generate_library_refs.py" \
                "$TRACK_CATALOG" "$LIBRARY_REF_DIR" \
                --parallel "$NCPU" \
                --manifest-out "$ABLETON_DIR/library_manifest.json"
            VERIFY_JSON="$LOG_DIR/scores_${i}_verify.json"
            python3 optimize/scripts/score.py \
                --ref-source ableton \
                --ableton-manifest "optimize/ableton/$( basename "$ABLETON_MANIFEST" )" \
                --batch "$VERIFY_JSON"
            NEW_AVG=$(python3 -c "import json; data=json.load(open('$VERIFY_JSON')); print(sum(r['total_score'] for r in data)/len(data))")
            echo "Post-agent score: $NEW_AVG (was: $AVG_SCORE)"
            if python3 -c "import sys; sys.exit(0 if $NEW_AVG > $AVG_SCORE else 1)"; then
                echo "Score improved! Committing uncommitted agent changes..."
                git add src/
                git commit -m "opt(ableton): auto-commit agent improvement, score=$NEW_AVG (was $AVG_SCORE)"
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

echo "=== Ableton Optimization Complete ==="
echo "Results in: $PROGRESS_CSV"

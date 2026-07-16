#!/usr/bin/env bash
# Run timestretch-rs vs. rubberband head-to-head across DJ-relevant ratios.
#
# Requires: rubberband CLI (brew install rubberband / apt install rubberband-cli)
# Output:   target/rubberband_benchmark/summary.csv
#
# Usage:
#   ./scripts/compare_rubberband.sh
#   ./scripts/compare_rubberband.sh --ratios "0.95 1.05 1.08"
#   TIMESTRETCH_RUBBERBAND_ENGINE=realtime ./scripts/compare_rubberband.sh --ratios "0.95 1.05"
#
# Mode selection (TIMESTRETCH_RUBBERBAND_ENGINE, inherited by the test):
#   batch (default) - offline stretch() over the engine graph.
#   realtime - the engine's keylock chain at constant rate, callback by
#     callback. Only meaningful at DJ ratios (0.92-1.08, +/-20%
#     secondary): beyond ~+/-10% rate deviation keylock deliberately
#     fades toward plain varispeed (pitch follows tempo) while RubberBand
#     always preserves pitch.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# --- Config -------------------------------------------------------------------

RATIOS="${1:-}"
if [[ "$RATIOS" == "--ratios" ]]; then
    RATIOS="$2"
    shift 2
fi
RATIOS="${RATIOS:-0.75 1.5 2.0}"

TRACKS=(
    "edm_mix:test_audio/edm_mix.wav"
    "percussive:test_audio/kick_pattern_128bpm.wav"
    "harmonic:test_audio/sweep_20_20k.wav"
)

OUT_DIR="target/rubberband_benchmark"
SUMMARY="$OUT_DIR/summary.csv"

# --- Preflight ----------------------------------------------------------------

if ! command -v rubberband >/dev/null 2>&1; then
    echo "ERROR: rubberband not found."
    echo "  macOS:  brew install rubberband"
    echo "  Linux:  apt install rubberband-cli"
    exit 1
fi

RUBBERBAND_BIN="rubberband"

# Generate test audio if any track file is missing.
needs_generate=0
for entry in "${TRACKS[@]}"; do
    wav="${entry#*:}"
    [[ -f "$ROOT_DIR/$wav" ]] || needs_generate=1
done

if [[ "$needs_generate" -eq 1 ]]; then
    echo "Generating test audio..."
    cargo run --example generate_test_audio --quiet
fi

mkdir -p "$OUT_DIR"

# --- Run comparisons ----------------------------------------------------------

echo "scenario,ratio,sample_rate,spectral_similarity,perceptual_spectral_similarity,cross_correlation,transient_within_10ms,transient_total,lufs_difference,spectral_flux_similarity,overall_grade" \
    > "$SUMMARY"

for entry in "${TRACKS[@]}"; do
    track_name="${entry%%:*}"
    wav_rel="${entry#*:}"
    original="$ROOT_DIR/$wav_rel"

    for ratio in $RATIOS; do
        ref_wav="$OUT_DIR/${track_name}_rubberband_${ratio}x.wav"

        echo "--- $track_name @ ${ratio}x ---"

        # Stretch with rubberband. The --time flag sets a time ratio:
        # values >1 lengthen the audio, <1 shorten it. Verify with your
        # installed version: rubberband --help | grep -A1 '\-\-time'
        "$RUBBERBAND_BIN" --time "$ratio" "$original" "$ref_wav" 2>/dev/null \
            || { echo "  rubberband failed for $track_name @ ${ratio}x"; continue; }

        row_csv="$OUT_DIR/${track_name}_${ratio}x_report.csv"

        # Remove stale report so a failed test doesn't contaminate the next row.
        rm -f "$OUT_DIR/rubberband_comparison_report.csv"

        # Run the built-in comparison test.
        TIMESTRETCH_RUBBERBAND_ORIGINAL_WAV="$original" \
        TIMESTRETCH_RUBBERBAND_REFERENCE_WAV="$ref_wav" \
        TIMESTRETCH_RUBBERBAND_RATIO="$ratio" \
        TIMESTRETCH_RUBBERBAND_MAX_SECONDS="20.0" \
            cargo test \
                --features qa-harnesses \
                --release \
                --test rubberband_comparison \
                -- --nocapture 2>&1 \
            | grep -E "^rubberband comparison:|wrote:" || true

        # The test writes target/rubberband_benchmark/rubberband_comparison_report.csv.
        # Extract the data row and prepend the scenario name.
        if [[ -f "$OUT_DIR/rubberband_comparison_report.csv" ]]; then
            # Skip the header line, prepend scenario name.
            tail -n1 "$OUT_DIR/rubberband_comparison_report.csv" \
                | sed "s/^external_rubberband_render/${track_name}/" \
                >> "$SUMMARY"
            cp "$OUT_DIR/rubberband_comparison_report.csv" "$row_csv"
        fi
    done
done

# --- Summary ------------------------------------------------------------------

echo ""
echo "Results written to: $SUMMARY"
echo ""

# Pretty-print if python3 is available.
if command -v python3 >/dev/null 2>&1; then
python3 - "$SUMMARY" <<'PY'
import csv, sys

path = sys.argv[1]
with open(path) as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("(no results)")
    sys.exit(0)

col_w = 18
headers = ["scenario", "ratio", "spectral", "perceptual", "xcorr", "transients", "lufs_diff", "flux", "grade"]
print("  ".join(h.ljust(col_w) for h in headers))
print("-" * (col_w * len(headers) + 2 * (len(headers) - 1)))

for r in rows:
    t_pct = ""
    tot = int(float(r.get("transient_total", "0")))
    hit = int(float(r.get("transient_within_10ms", "0")))
    if tot > 0:
        t_pct = f"{100.0 * hit / tot:.0f}% ({hit}/{tot})"
    cols = [
        r.get("scenario", ""),
        r.get("ratio", ""),
        r.get("spectral_similarity", "")[:6],
        r.get("perceptual_spectral_similarity", "")[:6],
        r.get("cross_correlation", "")[:6],
        t_pct or "n/a",
        r.get("lufs_difference", "")[:7],
        r.get("spectral_flux_similarity", "")[:6],
        r.get("overall_grade", ""),
    ]
    print("  ".join(str(c).ljust(col_w) for c in cols))
PY
fi

#!/usr/bin/env bash
# Stage 11 falsification experiment driver (ROADMAP Stage 11): renders the
# wide-range Master Tempo listening matrix — per track and tempo rate, one
# WAV per arm (varispeed release, shipped wide PV, the reset-driven
# prototype at FFT 2048 and 1024, the free-low-band variant, and a
# rubberband reference) — plus summary.csv with the objective sidecar.
#
# Requires: rubberband CLI for the reference arm (brew install rubberband /
#           apt install rubberband-cli); the harness renders the other arms
#           without it.
# Output:   target/wide_falsification/<track>/<rate_tag>/<arm>.wav
#           target/wide_falsification/summary.csv
#
# Usage:
#   ./scripts/wide_falsification.sh
#   ./scripts/wide_falsification.sh --rates "1.5 0.5"
#   TIMESTRETCH_WIDE_TRACKS="mytag=/path/to/track.wav" ./scripts/wide_falsification.sh
#
# Tempo rates are DJ rates (1.5 = +50%); the harness stretches time by the
# reciprocal. Corpus tracks default to the local bass-heavy bpm-corpus
# entries plus one CC public-corpus track; the synthetic bass fixture is
# always rendered. bpm-corpus renders are commercial material — everything
# stays in gitignored target/.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RATES=""
if [[ "${1:-}" == "--rates" ]]; then
    RATES="$2"
    shift 2
fi

if ! command -v rubberband >/dev/null 2>&1; then
    echo "WARNING: rubberband not found — the reference arm will be skipped."
    echo "  macOS:  brew install rubberband"
    echo "  Linux:  apt install rubberband-cli"
fi

OUT_DIR="target/wide_falsification"
mkdir -p "$OUT_DIR"
{
    echo "git $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_DIR/run_info.txt"

TIMESTRETCH_WIDE_RATES="${RATES:-${TIMESTRETCH_WIDE_RATES:-}}" \
    cargo test \
        --features qa-harnesses \
        --release \
        --test wide_falsification \
        -- --ignored --nocapture

SUMMARY="$OUT_DIR/summary.csv"
echo ""
echo "Results written to: $SUMMARY"
echo ""

# Pretty-print if python3 is available.
if command -v python3 >/dev/null 2>&1 && [[ -f "$SUMMARY" ]]; then
python3 - "$SUMMARY" <<'PY'
import csv, sys

path = sys.argv[1]
with open(path) as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("(no results)")
    sys.exit(0)

headers = ["track", "rate", "arm", "lufs_d", "low_band", "clicks", "rb_spec", "rb_perc", "rb_lufs"]
widths = [16, 8, 16, 8, 9, 8, 8, 8, 8]
print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
print("-" * (sum(widths) + 2 * (len(headers) - 1)))
for r in rows:
    cols = [
        r.get("track", ""),
        r.get("rate", ""),
        r.get("arm", ""),
        r.get("lufs_delta_source", "") or "n/a",
        r.get("low_band_ratio", ""),
        r.get("clicks_per_million", ""),
        r.get("rb_spectral", "") or "n/a",
        r.get("rb_perceptual", "") or "n/a",
        r.get("rb_lufs_diff", "") or "n/a",
    ]
    print("  ".join(str(c)[:w].ljust(w) for c, w in zip(cols, widths)))
PY
fi

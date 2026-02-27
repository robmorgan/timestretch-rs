#!/bin/bash
set -euo pipefail

# generate_samples.sh - Synthesize test WAV files using sox
# All files: 44100 Hz, 16-bit, mono, ~5 seconds

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/samples"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$SAMPLES_DIR"

if ! command -v sox >/dev/null 2>&1; then
    echo "Error: sox is not installed. Run 'make setup' first."
    exit 1
fi

SOX="sox -r 44100 -b 16 -c 1"

echo "Generating synthetic test samples in $SAMPLES_DIR..."

# 1. sine_440.wav — Pure 440 Hz sine tone
echo "  sine_440.wav"
sox -n -r 44100 -b 16 -c 1 "$SAMPLES_DIR/sine_440.wav" \
    synth 5.0 sine 440

# 2. drums_break.wav — Layered kick + snare pattern with sharp transients
echo "  drums_break.wav"
# Create a single kick hit (60 Hz sine, fast decay)
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/kick_hit.wav" \
    synth 0.1 sine 60:30 fade l 0.001 0.1 0.08 gain -3
# Pad to 0.25s and repeat to fill ~5s (20 hits at 4 Hz)
sox "$TMP_DIR/kick_hit.wav" "$TMP_DIR/kick_padded.wav" pad 0 0.15
sox "$TMP_DIR/kick_padded.wav" "$TMP_DIR/kicks.wav" repeat 19

# Create a single snare hit (noise burst, bandpassed)
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/snare_hit.wav" \
    synth 0.05 pinknoise fade l 0.001 0.05 0.04 gain -6
# Offset snare by 0.25s to alternate with kick
sox "$TMP_DIR/snare_hit.wav" "$TMP_DIR/snare_padded.wav" pad 0.25 0.2
sox "$TMP_DIR/snare_padded.wav" "$TMP_DIR/snares.wav" repeat 9

# Mix kick and snare, trim to 5s
sox -m "$TMP_DIR/kicks.wav" "$TMP_DIR/snares.wav" \
    "$SAMPLES_DIR/drums_break.wav" trim 0 5.0

# 3. vocal_phrase.wav — Multi-harmonic signal with amplitude modulation
echo "  vocal_phrase.wav"
# Create each harmonic separately then mix
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/h1.wav" synth 5.0 sine 220 gain -6
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/h2.wav" synth 5.0 sine 440 gain -10
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/h3.wav" synth 5.0 sine 660 gain -14
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/h4.wav" synth 5.0 sine 880 gain -18
# Mix harmonics
sox -m "$TMP_DIR/h1.wav" "$TMP_DIR/h2.wav" "$TMP_DIR/h3.wav" "$TMP_DIR/h4.wav" \
    "$TMP_DIR/harmonics.wav"
# Apply slow amplitude modulation (tremolo at 3 Hz)
sox "$TMP_DIR/harmonics.wav" "$SAMPLES_DIR/vocal_phrase.wav" tremolo 3 60

# 4. full_mix.wav — Layered bass + mid + high + noise bursts
echo "  full_mix.wav"
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/bass.wav" synth 5.0 sine 80 gain -8
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/mid.wav" synth 5.0 sine 440 gain -10
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/high.wav" synth 5.0 sine 2000 gain -12
# Create noise bursts: short noise hit repeated periodically
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/burst_hit.wav" \
    synth 0.08 pinknoise fade l 0.002 0.08 0.06 gain -12
sox "$TMP_DIR/burst_hit.wav" "$TMP_DIR/burst_padded.wav" pad 0 0.545
sox "$TMP_DIR/burst_padded.wav" "$TMP_DIR/bursts.wav" repeat 7
# Mix all layers, trim to 5s
sox -m "$TMP_DIR/bass.wav" "$TMP_DIR/mid.wav" "$TMP_DIR/high.wav" "$TMP_DIR/bursts.wav" \
    "$SAMPLES_DIR/full_mix.wav" trim 0 5.0

# 5. silence_transient.wav — ~4 seconds silence then a sharp click
echo "  silence_transient.wav"
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/silence.wav" trim 0 4.0
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/click.wav" \
    synth 0.01 pinknoise fade l 0 0.01 0.005 gain -1 pad 0 0.99
sox "$TMP_DIR/silence.wav" "$TMP_DIR/click.wav" \
    "$SAMPLES_DIR/silence_transient.wav"

echo "Done. Generated 5 samples in $SAMPLES_DIR:"
ls -lh "$SAMPLES_DIR"/*.wav

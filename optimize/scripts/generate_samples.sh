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

# =====================================================================
# STEREO SAMPLES — meaningful L/R differences for stereo metric testing
# =====================================================================

# 6. stereo_drums.wav — Kick center, hihat panned right, snare center-left
echo "  stereo_drums.wav"
# Kick (center = equal L/R)
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/s_kick_hit.wav" \
    synth 0.1 sine 60:30 fade l 0.001 0.1 0.08 gain -3
sox "$TMP_DIR/s_kick_hit.wav" "$TMP_DIR/s_kick_padded.wav" pad 0 0.15
sox "$TMP_DIR/s_kick_padded.wav" "$TMP_DIR/s_kicks.wav" repeat 19
# Hihat (panned right: L=-18dB, R=0dB)
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/s_hh_hit.wav" \
    synth 0.03 pinknoise fade l 0.001 0.03 0.02 gain -8
sox "$TMP_DIR/s_hh_hit.wav" "$TMP_DIR/s_hh_padded.wav" pad 0.125 0.095
sox "$TMP_DIR/s_hh_padded.wav" "$TMP_DIR/s_hh_seq.wav" repeat 19
# Create L/R versions with different gains for hihat
sox "$TMP_DIR/s_hh_seq.wav" "$TMP_DIR/s_hh_L.wav" gain -18
sox "$TMP_DIR/s_hh_seq.wav" "$TMP_DIR/s_hh_R.wav" gain 0
# Snare (center-left: L=0dB, R=-6dB)
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/s_snare_hit.wav" \
    synth 0.05 pinknoise fade l 0.001 0.05 0.04 gain -6
sox "$TMP_DIR/s_snare_hit.wav" "$TMP_DIR/s_snare_padded.wav" pad 0.25 0.2
sox "$TMP_DIR/s_snare_padded.wav" "$TMP_DIR/s_snare_seq.wav" repeat 9
sox "$TMP_DIR/s_snare_seq.wav" "$TMP_DIR/s_snare_L.wav" gain 0
sox "$TMP_DIR/s_snare_seq.wav" "$TMP_DIR/s_snare_R.wav" gain -6
# Build L and R channels by mixing
sox -m "$TMP_DIR/s_kicks.wav" "$TMP_DIR/s_hh_L.wav" "$TMP_DIR/s_snare_L.wav" \
    "$TMP_DIR/s_drums_L.wav" trim 0 5.0
sox -m "$TMP_DIR/s_kicks.wav" "$TMP_DIR/s_hh_R.wav" "$TMP_DIR/s_snare_R.wav" \
    "$TMP_DIR/s_drums_R.wav" trim 0 5.0
# Merge L/R into stereo
sox -M "$TMP_DIR/s_drums_L.wav" "$TMP_DIR/s_drums_R.wav" \
    "$SAMPLES_DIR/stereo_drums.wav"

# 7. stereo_vocal.wav — Center vocal with stereo chorus effect
echo "  stereo_vocal.wav"
# Base vocal harmonics (center)
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/sv_h1.wav" synth 5.0 sine 220 gain -6
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/sv_h2.wav" synth 5.0 sine 440 gain -10
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/sv_h3.wav" synth 5.0 sine 660 gain -14
sox -m "$TMP_DIR/sv_h1.wav" "$TMP_DIR/sv_h2.wav" "$TMP_DIR/sv_h3.wav" \
    "$TMP_DIR/sv_center.wav"
# Chorus L: slightly detuned (+3 Hz) and delayed
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/sv_ch_l1.wav" synth 5.0 sine 223 gain -12
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/sv_ch_l2.wav" synth 5.0 sine 443 gain -16
sox -m "$TMP_DIR/sv_ch_l1.wav" "$TMP_DIR/sv_ch_l2.wav" "$TMP_DIR/sv_chorus_L.wav"
sox "$TMP_DIR/sv_chorus_L.wav" "$TMP_DIR/sv_chorus_L_delayed.wav" pad 0.012 0
# Chorus R: detuned (-2 Hz) with different delay
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/sv_ch_r1.wav" synth 5.0 sine 218 gain -12
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/sv_ch_r2.wav" synth 5.0 sine 438 gain -16
sox -m "$TMP_DIR/sv_ch_r1.wav" "$TMP_DIR/sv_ch_r2.wav" "$TMP_DIR/sv_chorus_R.wav"
sox "$TMP_DIR/sv_chorus_R.wav" "$TMP_DIR/sv_chorus_R_delayed.wav" pad 0.008 0
# L = center + chorus_L, R = center + chorus_R
sox -m "$TMP_DIR/sv_center.wav" "$TMP_DIR/sv_chorus_L_delayed.wav" \
    "$TMP_DIR/sv_vocal_L.wav" trim 0 5.0
sox -m "$TMP_DIR/sv_center.wav" "$TMP_DIR/sv_chorus_R_delayed.wav" \
    "$TMP_DIR/sv_vocal_R.wav" trim 0 5.0
sox -M "$TMP_DIR/sv_vocal_L.wav" "$TMP_DIR/sv_vocal_R.wav" \
    "$SAMPLES_DIR/stereo_vocal.wav"

# 8. stereo_mix.wav — Bass center, synth pad L/R, percussion wide
echo "  stereo_mix.wav"
# Bass (center)
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/sm_bass.wav" synth 5.0 sine 80 gain -8
# Synth pad L (mid-freq, slow tremolo)
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/sm_pad_L_raw.wav" synth 5.0 sine 330 gain -10
sox "$TMP_DIR/sm_pad_L_raw.wav" "$TMP_DIR/sm_pad_L.wav" tremolo 2.5 50
# Synth pad R (different freq, different tremolo rate)
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/sm_pad_R_raw.wav" synth 5.0 sine 392 gain -10
sox "$TMP_DIR/sm_pad_R_raw.wav" "$TMP_DIR/sm_pad_R.wav" tremolo 3.5 50
# Percussion: noise bursts, alternating L/R
sox -n -r 44100 -b 16 -c 1 "$TMP_DIR/sm_perc_hit.wav" \
    synth 0.05 pinknoise fade l 0.001 0.05 0.03 gain -10
sox "$TMP_DIR/sm_perc_hit.wav" "$TMP_DIR/sm_perc_L_pad.wav" pad 0 0.45
sox "$TMP_DIR/sm_perc_L_pad.wav" "$TMP_DIR/sm_perc_L_seq.wav" repeat 9
sox "$TMP_DIR/sm_perc_hit.wav" "$TMP_DIR/sm_perc_R_pad.wav" pad 0.25 0.2
sox "$TMP_DIR/sm_perc_R_pad.wav" "$TMP_DIR/sm_perc_R_seq.wav" repeat 9
# L channel: bass + pad_L + perc_L
sox -m "$TMP_DIR/sm_bass.wav" "$TMP_DIR/sm_pad_L.wav" "$TMP_DIR/sm_perc_L_seq.wav" \
    "$TMP_DIR/sm_mix_L.wav" trim 0 5.0
# R channel: bass + pad_R + perc_R
sox -m "$TMP_DIR/sm_bass.wav" "$TMP_DIR/sm_pad_R.wav" "$TMP_DIR/sm_perc_R_seq.wav" \
    "$TMP_DIR/sm_mix_R.wav" trim 0 5.0
sox -M "$TMP_DIR/sm_mix_L.wav" "$TMP_DIR/sm_mix_R.wav" \
    "$SAMPLES_DIR/stereo_mix.wav"

echo "Done. Generated 8 samples in $SAMPLES_DIR:"
ls -lh "$SAMPLES_DIR"/*.wav

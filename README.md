# timestretch

Pure Rust audio time-stretching library optimized for electronic dance music.

Stretches audio in time without changing its pitch, using a hybrid algorithm that
combines phase vocoder (for tonal content) with WSOLA (for transients). The only
external DSP dependency is [`rustfft`](https://crates.io/crates/rustfft).

## Features

- **Hybrid algorithm** — automatically switches between phase vocoder and WSOLA
  at transient boundaries so kicks stay punchy while pads stretch smoothly
- **Exact timeline fidelity** — explicit segment timeline bookkeeping and
  crossfade compensation keep output duration locked to target tempo
- **EDM presets** — tuned parameter sets for DJ beatmatching, house loops,
  halftime effects, ambient stretches, and vocal chops
- **Persistent hybrid streaming** — optional high-quality stream mode that keeps
  rolling state across calls instead of re-instantiating per chunk
- **Stateful streaming PV core** — phase state and overlap tails persist across
  stream chunks for smoother continuity
- **Streaming API** — process audio in chunks for real-time use with dynamic
  stretch ratio and tempo changes
- **Offline pre-analysis pipeline** — optional reusable artifact (BPM, phase,
  confidence, transient map) for safer beat/onset alignment at runtime
- **Stereo coherence hardening** — shared onset/timing map and deterministic
  channel length agreement in mid/side mode
- **Sub-bass phase locking** — locks phase below 120 Hz to prevent bass smearing
- **Quality gates** — benchmark-style pass/fail regression checks for duration,
  transient alignment, timing coherence, loudness, and spectral similarity
- **WAV I/O** — built-in reader/writer for 16-bit, 24-bit, and 32-bit float WAV files
- **Safe Rust** — `#![forbid(unsafe_code)]`, no panics in library code

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
timestretch = "0.1"
```

### One-Shot Stretching

```rust
use timestretch::{StretchParams, EdmPreset};

// Generate or load audio (f32 samples, -1.0 to 1.0)
let input: Vec<f32> = load_audio();

let params = StretchParams::new(1.5) // 1.5x longer (slower)
    .with_sample_rate(44100)
    .with_channels(1)
    .with_preset(EdmPreset::HouseLoop);

let output = timestretch::stretch(&input, &params).unwrap();
```

### DJ Beatmatching (126 BPM to 128 BPM)

```rust
use timestretch::{StretchParams, EdmPreset, bpm_ratio};

let original_bpm = 126.0_f64;
let target_bpm = 128.0_f64;
let ratio = bpm_ratio(original_bpm, target_bpm); // source / target = ~0.984

let params = StretchParams::new(ratio)
    .with_preset(EdmPreset::DjBeatmatch)
    .with_sample_rate(44100)
    .with_channels(2); // stereo

let output = timestretch::stretch(&input, &params).unwrap();
```

### Real-Time Streaming

```rust
use timestretch::{QualityMode, StreamProcessor, StretchParams, EdmPreset};

let params = StretchParams::new(1.02)
    .with_preset(EdmPreset::DjBeatmatch)
    .with_sample_rate(44100)
    .with_channels(2)
    .with_quality_mode(QualityMode::Balanced);

let mut processor = StreamProcessor::new(params);
processor.set_hybrid_mode(true); // optional: persistent hybrid streaming path
let mut output_chunk = Vec::with_capacity(8192); // pre-allocate once

// Feed chunks as they arrive from your audio driver
loop {
    let input_chunk = read_audio_chunk(1024);
    output_chunk.clear();
    processor.process_into(&input_chunk, &mut output_chunk).unwrap();
    play_audio(&output_chunk);
}

// Change ratio on the fly (e.g. DJ pitch fader)
processor.set_stretch_ratio(1.05);

// Flush remaining samples when done
let mut remaining = Vec::with_capacity(8192);
processor.flush_into(&mut remaining).unwrap();
```

### Tempo-Aware Streaming (DJ)

```rust
use timestretch::StreamProcessor;

let mut processor = StreamProcessor::from_tempo(126.0, 128.0, 44100, 2);
processor.set_hybrid_mode(true); // optional: higher-quality hybrid stream path

// Move the target deck tempo during playback
processor.set_tempo(130.0);

println!(
    "Current target BPM: {:.2}, latency: {:.1} ms",
    processor.target_bpm().unwrap_or(0.0),
    processor.latency_secs() * 1000.0
);
```

### AudioBuffer API

```rust
use timestretch::{AudioBuffer, StretchParams};

let buffer = AudioBuffer::from_mono(samples, 44100);
let params = StretchParams::new(2.0);
let output = timestretch::stretch_buffer(&buffer, &params).unwrap();

println!("Duration: {:.2}s -> {:.2}s", buffer.duration_secs(), output.duration_secs());
```

### Pitch Shifting

```rust
use timestretch::{EnvelopePreset, StretchParams};

let params = StretchParams::new(1.0)
    .with_sample_rate(44100)
    .with_channels(1)
    .with_envelope_preset(EnvelopePreset::Vocal) // stronger formant retention
    .with_envelope_strength(1.4)
    .with_adaptive_envelope_order(true);

// Shift up one octave (2x frequency), preserving duration
let output = timestretch::pitch_shift(&input, &params, 2.0).unwrap();
assert_eq!(output.len(), input.len());
```

Envelope control quick guide:
- Default profile is `EnvelopePreset::Balanced` (`envelope_strength = 1.0`, adaptive order enabled).
- Use `.with_envelope_preset(EnvelopePreset::Off)` for classic behavior with no formant correction.
- Use `.with_envelope_preset(EnvelopePreset::Vocal)` for stronger vocal formant retention.
- Use `.with_envelope_strength(x)` to scale correction (`0.0..=2.0`), and `.with_adaptive_envelope_order(true)` for content-adaptive cepstral detail.

### BPM-Based Stretching

```rust
use timestretch::{StretchParams, EdmPreset};

let params = StretchParams::new(1.0) // ratio computed automatically
    .with_sample_rate(44100)
    .with_channels(2)
    .with_preset(EdmPreset::DjBeatmatch);

// Stretch a 126 BPM track to 128 BPM
let output = timestretch::stretch_to_bpm(&input, &params, 126.0, 128.0).unwrap();
```

### Offline Pre-Analysis (Optional)

```rust
use timestretch::{
    analyze_for_dj, read_preanalysis_json, stretch, write_preanalysis_json,
    StretchParams, EdmPreset,
};
use std::path::Path;

// Build a reusable analysis artifact once (offline)
let artifact = analyze_for_dj(&input, 44100);
write_preanalysis_json(Path::new("track.preanalysis.json"), &artifact).unwrap();

// Load artifact at runtime and attach it to params
let loaded = read_preanalysis_json(Path::new("track.preanalysis.json")).unwrap();
let params = StretchParams::new(126.0 / 128.0)
    .with_preset(EdmPreset::DjBeatmatch)
    .with_sample_rate(44100)
    .with_pre_analysis(loaded)
    .with_beat_snap_confidence_threshold(0.35)
    .with_beat_snap_tolerance_ms(5.0);

let output = stretch(&input, &params).unwrap();
```

### WAV File I/O

```rust
use timestretch::io::wav;

// Read a WAV file
let buffer = wav::read_wav_file("input.wav").unwrap();

// Stretch it
let params = timestretch::StretchParams::new(2.0)
    .with_preset(timestretch::EdmPreset::Halftime);
let output = timestretch::stretch_buffer(&buffer, &params).unwrap();

// Write the result (16-bit, 24-bit, or float)
wav::write_wav_file_16bit("output_16.wav", &output).unwrap();
wav::write_wav_file_24bit("output_24.wav", &output).unwrap();
wav::write_wav_file_float("output_32.wav", &output).unwrap();

// Or use the one-liner convenience API
timestretch::stretch_wav_file("input.wav", "output.wav", &params).unwrap();
```

## EDM Presets

| Preset | Use Case | Stretch Range | FFT | Transient Sensitivity |
|--------|----------|---------------|-----|----------------------|
| `DjBeatmatch` | Live mixing tempo sync | ±1–8% | 4096 | Low (0.3) |
| `HouseLoop` | General house/techno loops | ±5–25% | 4096 | Medium (0.5) |
| `Halftime` | Bass music halftime effect | 2x | 4096 | High (0.7) |
| `Ambient` | Ambient transitions/builds | 2x–4x | 8192 | Low (0.2) |
| `VocalChop` | Vocal samples & one-shots | ±10–50% | 2048 | Medium-high (0.6) |

## How It Works

The library uses a hybrid segmented pipeline:

1. **Transient detection** — spectral flux with adaptive threshold identifies
   attack transients (kicks, snares, hi-hats). High-frequency bins (2–8 kHz)
   are weighted more heavily to catch percussive onsets.

2. **Beat-aware segmentation (optional)** — transient boundaries can be merged
   with beat-grid positions and snapped to subdivisions. If provided, an
   offline pre-analysis artifact is preferred when confidence is high.

3. **Segment-wise stretching** — the audio is split at boundaries.
   Transient segments are stretched with WSOLA (preserves waveform shape and
   attack character). Tonal segments are stretched with a phase vocoder
   (preserves frequency content with identity phase locking).

4. **Sub-bass treatment** — frequencies below 120 Hz always use phase-locked
   processing to prevent phase cancellation that would weaken the bass.

5. **Timeline correction** — explicit timeline bookkeeping compensates boundary
   overlap so concatenation preserves target output duration exactly.

Segment joins use fixed or adaptive raised-cosine crossfades.

## Parameters

`StretchParams` supports a builder pattern for full control:

```rust
let params = StretchParams::new(1.5)
    .with_sample_rate(48000)
    .with_channels(2)
    .with_preset(EdmPreset::HouseLoop)   // apply preset first
    .with_fft_size(4096)                 // then override individual params
    .with_hop_size(1024)
    .with_transient_sensitivity(0.6)
    .with_elastic_timing(true)
    .with_crossfade_mode(timestretch::CrossfadeMode::Adaptive)
    .with_hpss(true)
    .with_multi_resolution(true)
    .with_sub_bass_cutoff(100.0)
    .with_stereo_mode(timestretch::StereoMode::MidSide)
    .with_phase_locking_mode(timestretch::PhaseLockingMode::RegionOfInfluence)
    .with_wsola_segment_size(960)
    .with_wsola_search_range(480)
    .with_beat_aware(true)
    .with_beat_snap_confidence_threshold(0.35)
    .with_beat_snap_tolerance_ms(5.0);
```

**Defaults:** 44100 Hz, stereo, FFT 4096, hop 1024 (75% overlap), 120 Hz
sub-bass cutoff, ~20ms WSOLA segments, ~10ms search range.

## Performance

Performance depends heavily on preset, ratio, and mode (PV-only streaming vs
hybrid streaming vs offline batch).

Run built-in benchmark tests:

```sh
# Throughput-oriented benchmark suite (use release for realistic timing)
cargo test --release --test benchmarks -- --nocapture

# M0 baseline command (strict corpus validation + archive)
./benchmarks/run_m0_baseline.sh

# Quality-gate benchmark subset (CI-enforced)
cargo test --test quality_gates -- --nocapture

# Strict callback-budget gate (same mode used in CI quality-gates job)
TIMESTRETCH_STRICT_CALLBACK_BUDGET=1 cargo test --release --test quality_gates -- --nocapture

# Emit quality dashboard CSV artifacts (one file per quality gate)
TIMESTRETCH_QUALITY_DASHBOARD_DIR=target/quality_dashboard cargo test --test quality_gates -- --nocapture

# Reference-quality comparison (strict corpus required)
TIMESTRETCH_STRICT_REFERENCE_BENCHMARK=1 TIMESTRETCH_REFERENCE_MAX_SECONDS=30 cargo test --test reference_quality -- --nocapture

# Ad-hoc reference-quality run (non-strict, short window)
TIMESTRETCH_REFERENCE_MAX_SECONDS=5 cargo test --test reference_quality -- --nocapture

# Single-scenario comparison against an external Rubber Band render
TIMESTRETCH_RUBBERBAND_ORIGINAL_WAV=benchmarks/audio/originals/loop.wav \
TIMESTRETCH_RUBBERBAND_REFERENCE_WAV=benchmarks/audio/references/loop_rubberband.wav \
TIMESTRETCH_RUBBERBAND_RATIO=1.113043478 \
cargo test --test rubberband_comparison -- --nocapture
```

See `benchmarks/README.md` for corpus setup and manifest/checksum requirements.

## API Reference

### Core Types

- **`StretchParams`** — builder-pattern configuration: stretch ratio, sample rate,
  channels, FFT size, hop size, EDM preset, WSOLA parameters, beat-snap controls,
  optional pre-analysis artifact, and tempo helpers like `from_tempo()`
- **`AudioBuffer`** — holds interleaved sample data with metadata (sample rate,
  channel layout)
- **`EdmPreset`** — enum of tuned parameter sets for EDM workflows
- **`EnvelopePreset`** — formant/envelope profile (`Off`, `Balanced`, `Vocal`)
- **`QualityMode`** — explicit streaming profile: `LowLatency` (lean path, HPSS off), `Balanced`, `MaxQuality` (HPSS + adaptive crossfade/phase-lock enabled)
- **`StreamProcessor`** — chunked real-time processor with on-the-fly ratio/tempo
  changes, `from_tempo()`/`set_tempo()`, `process_into()`, and optional persistent hybrid mode
- **`PreAnalysisArtifact`** — serializable offline beat/onset analysis artifact
- **`StretchError`** — error type covering invalid parameters, I/O failures,
  and input-too-short conditions

### Functions

**Time stretching:**
- `stretch(&[f32], &StretchParams)` — stretch raw sample data
- `stretch_into(&[f32], &StretchParams, &mut Vec<f32>)` — append stretched output into caller buffer
- `stretch_buffer(&AudioBuffer, &StretchParams)` — stretch an `AudioBuffer`
- `stretch_to_bpm(&[f32], &StretchParams, source_bpm, target_bpm)` — BPM-based stretch
- `stretch_to_bpm_auto(&[f32], &StretchParams, target_bpm)` — auto-detect BPM and stretch
- `stretch_bpm_buffer(&AudioBuffer, &StretchParams, source_bpm, target_bpm)` — BPM stretch for `AudioBuffer`
- `stretch_bpm_buffer_auto(&AudioBuffer, &StretchParams, target_bpm)` — auto BPM stretch for `AudioBuffer`

**Pitch shifting:**
- `pitch_shift(&[f32], &StretchParams, factor)` — shift pitch without changing duration
- `pitch_shift_buffer(&AudioBuffer, &StretchParams, factor)` — pitch shift an `AudioBuffer`

**BPM detection:**
- `detect_bpm(&[f32], sample_rate)` — detect tempo from raw samples
- `detect_bpm_buffer(&AudioBuffer)` — detect tempo from an `AudioBuffer`
- `detect_beat_grid(&[f32], sample_rate)` — detect beat grid positions
- `detect_beat_grid_buffer(&AudioBuffer)` — detect beat grid from an `AudioBuffer`
- `bpm_ratio(source_bpm, target_bpm)` — compute stretch ratio for BPM change

**Pre-analysis artifact pipeline:**
- `analyze_for_dj(&[f32], sample_rate)` — generate offline beat/onset artifact
- `write_preanalysis_json(path, &PreAnalysisArtifact)` — write artifact JSON
- `read_preanalysis_json(path)` — read artifact JSON

**WAV file convenience:**
- `stretch_wav_file(input, output, &StretchParams)` — read, stretch, and write a WAV file
- `stretch_to_bpm_wav_file(input, output, &StretchParams, source_bpm, target_bpm)` — WAV BPM stretch
- `stretch_to_bpm_auto_wav_file(input, output, &StretchParams, target_bpm)` — WAV auto BPM stretch
- `pitch_shift_wav_file(input, output, &StretchParams, factor)` — read, pitch-shift, and write

See the [API documentation](https://docs.rs/timestretch) for full details.

## Examples

Run the included examples:

```sh
cargo run --example basic_stretch      # Simple time stretch
cargo run --example benchmark_quality  # Offline quality benchmark helper
cargo run --example dj_beatmatch       # 126 → 128 BPM tempo sync
cargo run --example dj_mix             # Streaming DJ transition demo
cargo run --example sample_halftime    # 2x halftime effect
cargo run --example pitch_shift        # Pitch shifting demo
cargo run --example realtime_stream    # Streaming API demo
```

## Audio Format

- Sample format: `f32` (32-bit float, range -1.0 to 1.0)
- Channel layout: mono or stereo (interleaved)
- Sample rates: any standard rate (44100, 48000, etc.)
- WAV I/O: 16-bit PCM, 24-bit PCM, and 32-bit float

## License

MIT

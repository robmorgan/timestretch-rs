# timestretch

[![crates.io](https://img.shields.io/crates/v/timestretch.svg)](https://crates.io/crates/timestretch)
[![docs.rs](https://docs.rs/timestretch/badge.svg)](https://docs.rs/timestretch)
[![CI](https://github.com/robmorgan/timestretch-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/robmorgan/timestretch-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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
- **Stateful streaming PV core** — phase state and overlap tails persist across
  stream chunks for smoother continuity
- **Multi-resolution streaming engine** — optional three-band Linkwitz-Riley
  filterbank with per-band phase vocoders (16384-point sub-bass FFT at the
  Quality profile) for tighter low-end phase coherence, at a higher
  buffering latency
- **Streaming API** — process audio in chunks for real-time use with dynamic
  stretch ratio and tempo changes
- **Analyze-on-load pre-analysis** — analyze a track once (CLI `analyze` or
  `analyze_for_dj`), persist the artifact (BPM, beat grid, transient onsets
  with strengths, content hash), and every stretch path — batch and
  streaming — consumes it in place of online detection, with
  `set_source_position` keeping onsets aligned across seeks
- **Loudness-robust onset detection** — log-compressed spectral flux with a
  robust `median + k·MAD` threshold and an energy-channel gate, so dense
  mastered material yields usable onsets/BPM (where a plateau-and-multiply
  detector reads zero) while sustained tones correctly report no beat
- **Warm-start seek/cue/loop** — `warm_start_seek` re-primes the streaming
  processor from the audio preceding a jump so playback resumes converged
  (no cold buffering gap, no phase-vocoder warm-up, ratio/pitch state
  preserved); `notify_source_jump` wraps loops gaplessly
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
timestretch = "0.6.0"
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
use timestretch::{EdmPreset, QualityMode, StreamProcessor, StretchParams};

let params = StretchParams::new(1.02)
    .with_preset(EdmPreset::DjBeatmatch)
    .with_sample_rate(44100)
    .with_channels(2)
    .with_quality_mode(QualityMode::Balanced);

let mut processor = StreamProcessor::new(params);
let (_, _, _, pending_capacity) = processor.capacities();
let ratio_hint = processor
    .current_stretch_ratio()
    .max(processor.target_stretch_ratio())
    .max(1.0);
let input_samples_per_chunk = 1024 * 2; // 1024 stereo frames
let estimated_output = (input_samples_per_chunk as f64 * ratio_hint).ceil() as usize
    + pending_capacity;
let mut output_chunk = Vec::with_capacity(estimated_output);

// Feed chunks as they arrive from your audio driver
loop {
    let input_chunk = read_audio_chunk(1024);
    output_chunk.clear();
    processor.process_into(&input_chunk, &mut output_chunk).unwrap();
    play_audio(&output_chunk);
}

// Change ratio on the fly (e.g. DJ pitch fader)
processor.set_stretch_ratio(1.05).expect("valid ratio");

// Flush remaining samples when done
let mut remaining = Vec::with_capacity(pending_capacity * 2);
processor.flush_into(&mut remaining).unwrap();
```

If you do not want to manage `Vec` capacity yourself, use
`StreamProcessor::process()` / `StreamProcessor::flush()` instead.

### Fixed-Buffer Realtime Callbacks

Use these deterministic APIs when the host owns the callback buffer and you
need bounded output budgets instead of `Vec` append semantics.

```rust
use timestretch::{EdmPreset, StreamProcessor, StretchParams};

let params = StretchParams::new(1.02)
    .with_preset(EdmPreset::DjBeatmatch)
    .with_sample_rate(44_100)
    .with_channels(2);

let mut processor = StreamProcessor::new(params);

let input_chunk = vec![0.0f32; 256 * 2];
let callback_capacity = processor
    .max_next_process_interleaved_output_samples(input_chunk.len())
    .unwrap();
let mut callback_output = vec![0.0f32; callback_capacity];

let written = processor
    .process_interleaved_into(&input_chunk, &mut callback_output)
    .unwrap();
host_submit(&callback_output[..written]);

loop {
    let written = processor
        .flush_interleaved_into(&mut callback_output)
        .unwrap();
    if written == 0 {
        break;
    }
    host_submit(&callback_output[..written]);
}
```


### Tempo-Aware Streaming (DJ)

```rust
use timestretch::StreamProcessor;

let mut processor = StreamProcessor::from_tempo(126.0, 128.0, 44100, 2);

// Move the target deck tempo during playback
processor.set_tempo(130.0);

println!(
    "Current target BPM: {:.2}, latency: {:.1} ms",
    processor.target_bpm().unwrap_or(0.0),
    processor.latency_secs() * 1000.0
);
```

### Streaming Profiles & Latency

Streaming latency and quality are selected together through
`StreamProfile`. Every profile carries the full DJ tuning bundle; they
differ only in the FFT/hop configuration that sets the buffering latency.

```rust
use timestretch::{StreamProcessor, StreamProfile};

let mut processor =
    StreamProcessor::try_from_tempo_with_profile(126.0, 128.0, 44100, 2, StreamProfile::Live)
        .expect("valid BPM inputs");
assert!(processor.latency_secs() * 1000.0 < 40.0);

// Or via params: StretchParams::new(ratio).with_stream_profile(StreamProfile::Club)
// `try_from_tempo_low_latency` is shorthand for the Live profile.
```

Measured figures at 44.1 kHz (`tests/streaming_latency.rs` verifies that the
first output sample appears exactly at the reported gate; `latency_samples()`
/ `latency_report()` report the real current gate, not a formula):

| Profile | FFT/hop | Buffering gate (ratio in `[0.9, 1.1]`) | Gate beyond ±10% | Steady-ratio similarity | Ratio-ride similarity |
|---------|-----------|--------------------|--------------------|------|------|
| Live    | 1024/256  | 1536 fr = 34.8 ms  | 2048 fr = 46.4 ms  | 0.9982 | 0.9982 |
| Club    | 2048/512  | 3072 fr = 69.7 ms  | 4096 fr = 92.9 ms  | 0.9976 | 0.9969 |
| Quality | 4096/1024 | 6144 fr = 139.3 ms | 8192 fr = 185.8 ms | 0.9991 | 0.9932 |

Similarity = mean spectral similarity to the source on synthetic EDM material
(`qa/profile_quality.rs`). At steady stretch the larger windows win; under a
0.92–1.08 ratio ride the smaller windows track the modulation better — which
is exactly why `Live` is the DJ control profile.

### Streaming Engines

`StreamProcessor` renders with one of two engines, selected via
`set_streaming_engine` (at build time, not from the audio callback —
switching starts a fresh stream):

- `StreamingEngine::Deterministic` (default) — single persistent phase
  vocoder per channel with scheduled per-band transient phase resets.
  All figures in the profile table above are this engine.
- `StreamingEngine::MultiResolution` — three-band Linkwitz-Riley filterbank
  (sub-bass/mid/high at 200 Hz / 4 kHz) with a per-band phase vocoder whose
  FFT is tuned to its range: sub-bass `mid × 4`, high `mid / 4`. The
  buffering gate is set by the sub-bass FFT, so it needs the Club profile
  or larger — selecting it on Live returns an error.

```rust
use timestretch::{StreamProcessor, StreamProfile, StreamingEngine, StretchParams};

let params = StretchParams::new(1.05)
    .with_sample_rate(44100)
    .with_channels(2)
    .with_stream_profile(StreamProfile::Club);
let mut processor = StreamProcessor::new(params);
processor
    .set_streaming_engine(StreamingEngine::MultiResolution)
    .expect("Club/Quality profiles support multi-resolution");
```

Measured multi-resolution figures at 44.1 kHz (`tests/streaming_latency.rs`
verifies the first output sample lands exactly at the reported gate):

| Profile | Band FFTs (sub/mid/high) | Buffering gate (ratio in `[0.9, 1.1]`) | Steady-ratio similarity | Ratio-ride similarity |
|---------|--------------------------|--------------------|------|------|
| Club    | 8192/2048/512   | 12288 fr = 278.6 ms | 0.9898 | 0.9876 |
| Quality | 16384/4096/1024 | 24576 fr = 557.3 ms | 0.9898 | 0.9928 |

Trade-off profile on the same synthetic EDM material: the multi-res engine
ties the single-PV engine at the sub-bass band's similarity ceiling and wins
the mid band (0.9982 vs 0.9964 at Quality), but gives some broadband
similarity back in the bands containing the 200 Hz / 4 kHz crossover seams,
where the independently-stretched bands overlap. Reach for it on
sub-bass-critical material where low-end phase coherence matters more than
control latency; keep the deterministic engine for beatmatching feel.

Control-to-audio behavior on the default path (all profiles): ratio and
pitch controls glide with a ~50 ms time constant. Pitch changes reach the
output almost immediately (the pitch resampler sits after the vocoder, so
only the glide applies, plus 16–80 samples of sinc kernel lookahead); ratio
changes take roughly the buffering gate plus the glide.

### Control Paths (Varispeed-First Keylock)

`StreamProcessor` implements tempo through one of two control paths,
selected via `set_control_path` (at build time, like engine selection):

- `ControlPath::VocoderTempo` (default) — the phase vocoder implements the
  tempo change, so tempo control latency is the buffering gate plus the
  ~50 ms glide. The right path for tape-mode/no-keylock use.
- `ControlPath::VarispeedFirst` — the tempo fader drives a varispeed sinc
  resampler at the input (instant, sample-accurate retargets, no glide on
  the tempo axis), and the vocoder only pitch-corrects at a delay-matched
  transposition. The buffering gate becomes a **constant pipeline delay**
  the host compensates in its timeline; tempo control-to-audio collapses to
  the resampler's kernel lookahead (16 samples at DJ ratios) plus the
  caller's callback size. Tempo ratios are bounded to `[0.25, 4.0]`.

```rust
use timestretch::{ControlPath, StreamProcessor, StreamProfile, StretchParams};

let params = StretchParams::new(1.05)
    .with_sample_rate(44100)
    .with_channels(2)
    .with_stream_profile(StreamProfile::Live);
let mut processor = StreamProcessor::new(params);
processor
    .set_control_path(ControlPath::VarispeedFirst)
    .expect("ratio within the varispeed range");

let report = processor.latency_report();
// Constant content delay to compensate in the deck timeline:
let _ = report.pipeline_delay_frames;
// Tempo fader feel — resampler lookahead only:
assert!(report.control_to_audio_frames <= 80);
```

`latency_report()` splits the two figures: `pipeline_delay_frames` (constant
content delay, equals `latency_samples()`) versus `control_to_audio_frames`
(tempo-change latency, excluding the caller's callback buffering). Works
behind both streaming engines.

### Realtime Pitch Control

```rust
use timestretch::StreamProcessor;

let mut processor = StreamProcessor::from_tempo(126.0, 128.0, 44100, 2);
processor.set_pitch_scale(1.05).expect("valid pitch scale");
println!("Current pitch scale control: {:.3}", processor.pitch_scale());
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

Performance depends heavily on preset, ratio, and mode (streaming vs offline
batch).

Run opt-in QA harnesses:

```sh
# These harnesses are excluded from default `cargo test`.

# Throughput-oriented benchmark suite (use release for realistic timing)
cargo test --features qa-harnesses --release --test benchmarks -- --nocapture

# M0 baseline command (strict corpus validation + archive)
./benchmarks/run_m0_baseline.sh

# Quality-gate benchmark subset (CI-enforced)
cargo test --features qa-harnesses --release --test quality_gates -- --nocapture

# Strict callback-budget gate (same mode used in CI quality-gates job)
TIMESTRETCH_STRICT_CALLBACK_BUDGET=1 cargo test --features qa-harnesses --release --test quality_gates -- --nocapture

# Emit quality dashboard CSV artifacts (one file per quality gate)
TIMESTRETCH_QUALITY_DASHBOARD_DIR=target/quality_dashboard cargo test --features qa-harnesses --release --test quality_gates -- --nocapture

# Reference-quality comparison (strict corpus required)
TIMESTRETCH_STRICT_REFERENCE_BENCHMARK=1 TIMESTRETCH_REFERENCE_MAX_SECONDS=30 cargo test --features qa-harnesses --test reference_quality -- --nocapture

# Ad-hoc reference-quality run (non-strict, short window)
TIMESTRETCH_REFERENCE_MAX_SECONDS=5 cargo test --features qa-harnesses --test reference_quality -- --nocapture

# Single-scenario comparison against an external Rubber Band render
TIMESTRETCH_RUBBERBAND_ORIGINAL_WAV=benchmarks/audio/originals/loop.wav \
TIMESTRETCH_RUBBERBAND_REFERENCE_WAV=benchmarks/audio/references/loop_rubberband.wav \
TIMESTRETCH_RUBBERBAND_RATIO=1.113043478 \
cargo test --features qa-harnesses --test rubberband_comparison -- --nocapture
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
  changes, `from_tempo()`/`set_tempo()`, plus both
  `Vec`-append (`process_into()`/`flush_into()`) and fixed-buffer interleaved
  (`process_interleaved_into()`/`flush_interleaved_into()`) deterministic APIs
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

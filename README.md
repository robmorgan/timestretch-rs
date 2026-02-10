# timestretch

Pure Rust audio time-stretching library optimized for electronic dance music.

Stretches audio in time without changing its pitch, using a hybrid algorithm that
combines phase vocoder (for tonal content) with WSOLA (for transients). The only
external dependency is [`rustfft`](https://crates.io/crates/rustfft).

## Features

- **Hybrid algorithm** — automatically switches between phase vocoder and WSOLA
  at transient boundaries so kicks stay punchy while pads stretch smoothly
- **EDM presets** — tuned parameter sets for DJ beatmatching, house loops,
  halftime effects, ambient stretches, and vocal chops
- **Streaming API** — process audio in chunks for real-time use with dynamic
  stretch ratio changes
- **Sub-bass phase locking** — locks phase below 120 Hz to prevent bass smearing
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
use timestretch::{StretchParams, EdmPreset};

let original_bpm = 126.0_f64;
let target_bpm = 128.0_f64;
let ratio = target_bpm / original_bpm; // ~1.016

let params = StretchParams::new(ratio)
    .with_preset(EdmPreset::DjBeatmatch)
    .with_sample_rate(44100)
    .with_channels(2); // stereo

let output = timestretch::stretch(&input, &params).unwrap();
```

### Real-Time Streaming

```rust
use timestretch::{StreamProcessor, StretchParams, EdmPreset};

let params = StretchParams::new(1.02)
    .with_preset(EdmPreset::DjBeatmatch)
    .with_sample_rate(44100)
    .with_channels(2);

let mut processor = StreamProcessor::new(params);

// Feed chunks as they arrive from your audio driver
loop {
    let input_chunk = read_audio_chunk(1024);
    let output_chunk = processor.process(&input_chunk).unwrap();
    play_audio(&output_chunk);
}

// Change ratio on the fly (e.g. DJ pitch fader)
processor.set_stretch_ratio(1.05);

// Flush remaining samples when done
let remaining = processor.flush().unwrap();
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
use timestretch::StretchParams;

let params = StretchParams::new(1.0)
    .with_sample_rate(44100)
    .with_channels(1);

// Shift up one octave (2x frequency), preserving duration
let output = timestretch::pitch_shift(&input, &params, 2.0).unwrap();
assert_eq!(output.len(), input.len());
```

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

The library uses a three-stage hybrid approach:

1. **Transient detection** — spectral flux with adaptive threshold identifies
   attack transients (kicks, snares, hi-hats). High-frequency bins (2–8 kHz)
   are weighted more heavily to catch percussive onsets.

2. **Segment-wise stretching** — the audio is split at transient boundaries.
   Transient segments are stretched with WSOLA (preserves waveform shape and
   attack character). Tonal segments are stretched with a phase vocoder
   (preserves frequency content with identity phase locking).

3. **Sub-bass treatment** — frequencies below 120 Hz always use phase-locked
   processing to prevent phase cancellation that would weaken the bass.

Segments are joined with 5ms raised-cosine crossfades.

## Parameters

`StretchParams` supports a builder pattern for full control:

```rust
let params = StretchParams::new(1.5)
    .with_sample_rate(48000)
    .with_channels(2)
    .with_preset(EdmPreset::HouseLoop)  // apply preset first
    .with_fft_size(4096)                // then override individual params
    .with_hop_size(1024)
    .with_transient_sensitivity(0.6)
    .with_sub_bass_cutoff(100.0)
    .with_wsola_segment_size(960)
    .with_wsola_search_range(480);
```

**Defaults:** 44100 Hz, stereo, FFT 4096, hop 1024 (75% overlap), 120 Hz
sub-bass cutoff, ~20ms WSOLA segments, ~10ms search range.

## Performance

Measured on a modern CPU (release build), processing 5 seconds of 44.1 kHz audio:

| Configuration | Speed | Real-time Factor |
|---------------|-------|-----------------|
| Phase vocoder, mono | ~28ms | 176x |
| Phase vocoder, stereo | ~54ms | 93x |
| Streaming, stereo | ~59ms | 169x |

All buffers are pre-allocated on init. The processing loop performs zero
heap allocations.

## API Reference

### Core Types

- **`StretchParams`** — builder-pattern configuration: stretch ratio, sample rate,
  channels, FFT size, hop size, EDM preset, and WSOLA parameters
- **`AudioBuffer`** — holds interleaved sample data with metadata (sample rate,
  channel layout)
- **`EdmPreset`** — enum of tuned parameter sets for EDM workflows
- **`StreamProcessor`** — chunked real-time processor with on-the-fly ratio changes
- **`StretchError`** — error type covering invalid parameters, I/O failures,
  and input-too-short conditions

### Functions

**Time stretching:**
- `stretch(&[f32], &StretchParams)` — stretch raw sample data
- `stretch_buffer(&AudioBuffer, &StretchParams)` — stretch an `AudioBuffer`
- `stretch_to_bpm(&[f32], &StretchParams, source_bpm, target_bpm)` — BPM-based stretch
- `stretch_to_bpm_auto(&[f32], &StretchParams, target_bpm)` — auto-detect BPM and stretch

**Pitch shifting:**
- `pitch_shift(&[f32], &StretchParams, factor)` — shift pitch without changing duration
- `pitch_shift_buffer(&AudioBuffer, &StretchParams, factor)` — pitch shift an `AudioBuffer`

**BPM detection:**
- `detect_bpm(&[f32], sample_rate)` — detect tempo from raw samples
- `detect_bpm_buffer(&AudioBuffer)` — detect tempo from an `AudioBuffer`
- `detect_beat_grid(&[f32], sample_rate)` — detect beat grid positions
- `bpm_ratio(source_bpm, target_bpm)` — compute stretch ratio for BPM change

**WAV file convenience:**
- `stretch_wav_file(input, output, &StretchParams)` — read, stretch, and write a WAV file
- `pitch_shift_wav_file(input, output, &StretchParams, factor)` — read, pitch-shift, and write

See the [API documentation](https://docs.rs/timestretch) for full details.

## Examples

Run the included examples:

```sh
cargo run --example basic_stretch      # Simple time stretch
cargo run --example dj_beatmatch       # 126 → 128 BPM tempo sync
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

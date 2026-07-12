# Changelog

## 0.6.0

### Breaking changes

- Removed the legacy hybrid streaming engine from `StreamProcessor`:
  - `StreamingEngine::LegacyHybridRerender` variant removed.
  - `StreamProcessor::set_hybrid_mode` removed.
  - `StreamProcessor::set_streaming_engine` now returns
    `Result<(), StretchError>` (engine selection can fail; see below).

  `StreamingEngine::Deterministic` remains the default streaming path. The
  offline hybrid stretcher used by `stretch_buffer` and the batch APIs is
  unaffected.

### Added

- `StreamingEngine::MultiResolution`: an opt-in streaming engine that runs
  the three-band Linkwitz-Riley filterbank (`MultiResolutionStretcher`) in
  the real-time path, with per-profile band FFT sizing (Club 8192 sub-bass,
  Quality 16384). Selecting it with a mid FFT below 2048 (the Live profile)
  returns `StretchError::InvalidFormat`. Latency reporting, warm-start
  preroll, and the allocation-free steady-state guarantee cover the new
  engine; `StreamingEngine` is now re-exported at the crate root.
- `MultiResolutionStretcher` streaming API: `process_streaming_into`,
  `flush_streaming_into`, `reset_streaming_state`,
  `streaming_frames_consumed`, `reserve_streaming_capacity`,
  `last_frame_flux` (combined three-band flux), `total_spectral_bins`,
  `latency_samples`, and a `with_sub_bass_fft_cap` constructor.
- `ControlPath::VarispeedFirst` (re-exported at the crate root): inverts the
  keylock control architecture so the tempo fader drives a varispeed sinc
  resampler at the input (sample-accurate retargets, no glide on the tempo
  axis) and the phase vocoder only pitch-corrects at a delay-matched
  transposition. Selected via `StreamProcessor::set_control_path` /
  `control_path`. Tempo control-to-audio latency collapses to the resampler
  kernel lookahead (16 samples at DJ ratios); the buffering gate becomes a
  constant, host-compensated pipeline delay. Source-timeline mapping stays
  exact through tempo rides (onsets, `notify_source_jump`,
  `warm_start_seek`).
- `StreamLatencyReport` now splits `control_to_audio_secs` from
  `pipeline_delay_secs` so hosts can compensate the constant part;
  `latency_samples` keeps meaning content delay.
- `PhaseVocoder::set_smooth_ratio_updates`: disables seam-masking
  continuity heuristics for smooth correction streams (the varispeed path's
  torture-ride pitch wobble drops from ~100 to ~12 cents p95 on Live).
- `PhaseVocoder::reserve_streaming_capacity`: preallocates the deterministic
  engine's streaming buffers, closing a latent audio-thread allocation.

### QA

- Manifest-driven BPM detection accuracy harness (`qa/bpm_accuracy.rs`):
  scores the detector against `benchmarks/manifest.toml` tracks (new
  `bpm_only = true` entries; wav/mp3/aiff via a `symphonia`
  dev-dependency, sha256-verified) and reports exact/octave accuracy.
- Varispeed keylock pitch-stability gate (`qa/varispeed_keylock.rs`).

## 0.5.0

See the [release notes](https://github.com/robmorgan/timestretch-rs/releases)
for changes in 0.5.0 and earlier.

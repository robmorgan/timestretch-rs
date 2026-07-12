# Changelog

## Unreleased

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

## 0.5.0

See the [release notes](https://github.com/robmorgan/timestretch-rs/releases)
for changes in 0.5.0 and earlier.

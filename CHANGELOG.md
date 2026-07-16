# Changelog

## 0.8.0

### Breaking changes — old engine deleted (ROADMAP Stage 9)

The push-based streaming engine and the hybrid batch stretcher are gone;
the pull engine (`timestretch::engine`) is the only engine, serving both
real-time and batch.

- Removed: `StreamProcessor`, `StreamingEngine`, `StreamProfile`,
  `ControlPath`, `StreamLatencyReport`, `StreamPitchQuality`,
  `TransientResetStats`, `HybridStretcher`, `MultiResolutionStretcher`,
  `Wsola`, `StereoMode`, and the `stream` module.
- `StretchParams`: `stereo_mode` / `with_stereo_mode` and
  `with_stream_profile` removed (the engine processes channels in
  lockstep natively; profiles were an old-engine latency knob).
- `timestretch-cli`: `--streaming` and `--chunk-size` removed — batch
  runs on the engine graph with exact output length.
- Desktop app: the deck engine selector loses `Legacy (push)`; pull
  tape/keylock remain. Legacy-only pitch-shift slider removed.
- QA: old-engine harnesses (`streaming_quality`, `profile_quality`,
  `varispeed_keylock`, `quality_gates`) deleted; the engine A/B matrix
  is the quality dashboard, re-anchored on absolute thresholds derived
  from new-engine measurements.
- The live keylock chain drops its phase-vocoder corrector entirely
  (owner listening rejected it at every threshold; SOLA carries the
  whole corrected range) — chain WCET drops accordingly.

### Breaking changes — EDM presets removed

The preset system predates the engine cutover: none of the tuning knobs
`with_preset` set ever reached the pull engine, and the docs promised
behavior (HPSS, elastic timing, multi-resolution FFT, "kick punch") that
no longer existed. Batch quality is a property of the engine, not a
parameter matrix.

- Removed: `EdmPreset`, `StretchParams::with_preset`, and the CLI
  `--preset` flag (plus the legacy positional preset argument).
  Migration: `VocalChop`'s one live effect survives as
  `with_envelope_preset(EnvelopePreset::Vocal)` — or `--envelope vocal`
  on the CLI (new flag: `off`, `balanced`, `vocal`). The other presets
  need no replacement; the engine ignored them.
- `StretchParams` loses the dead tuning fields and their builders:
  `hop_size`, `transient_sensitivity`, `wsola_segment_size`,
  `wsola_search_range`, `beat_aware`, `band_split`, `multi_resolution`,
  `transient_region_secs`, `elastic_timing`, `elastic_anchor`,
  `hpss_enabled`, `transient_class_adaptive_wsola`, `residual_branch`,
  `residual_mix`, `crossfade_mode`, `dynamic_wsola_search`,
  `adaptive_phase_locking`, and the `effective_wsola_search_*` helpers.
  `CrossfadeMode` is removed with them. (`fft_size`, `window_type`, and
  the envelope fields stay — the `pitch_shift` formant path reads them.)
- The analytic tonal fast path in `stretch()` is deleted (it only ever
  fired with a preset set): pure-tone inputs now render through the real
  engine like everything else, and
  `tests/stretch_quality_regressions.rs` is rebaselined against measured
  engine output instead of the analytic bypass.
- Behavior note (owner decision 2026-07-16): offline `stretch()` shares
  the live keylock semantics by construction — within the corrected
  range (ratios ~0.833–1.25) content below the 150 Hz crossover is not
  pitch-corrected; its pitch follows tempo, offline exactly as on a
  deck. Wide ratios render on the batch PV path with full-spectrum
  pitch preservation. Relative level across the crossover is clean
  (two-tone balance within ~1% of ideal at ratio 1.25); the wide-PV
  path loses some low-band level at heavy compression (ratio 0.5),
  which is quality-secondary per the product boundary.

### Kept

- Batch API surface: `stretch`, `stretch_into`, `stretch_buffer`, BPM
  helpers, `pitch_shift` (still phase-vocoder based), analysis
  (`analyze_for_dj`, beat tracking), WAV I/O.
- `tests/streaming_batch_parity.rs` is superseded by
  `tests/streaming_offline_determinism.rs`: streaming and offline are
  sample-identical by construction.

## 0.7.0

### Breaking changes

- `BeatGrid` rebuilt around a piecewise tempo model instead of a single BPM
  value:
  - `beats: Vec<usize>` becomes `beats: Vec<f64>` (fractional-sample
    positions, following tempo drift); the separate `beats_fractional`
    field is removed.
  - New fields: `downbeats: Vec<usize>`, `segments: Vec<TempoSegment>`
    (piecewise-constant tempo, new type re-exported at the crate root),
    `confidence: f32`, `downbeat_confidence: f32`.
  - New `BeatGrid::empty(sample_rate)` and `BeatGrid::bpm_at(position)` for
    querying the tempo curve at a point.
- `detect_bpm` / `detect_beat_grid`: tempo estimation moves from an
  EDM-tuned 100–160 BPM inter-onset detector to a general-purpose
  autocorrelation tempogram (50–220 BPM, soft log-normal octave prior, no
  hard range folding) — a 90 BPM hip-hop track now reports 90 instead of
  being folded into the EDM range. Detected values on existing material may
  shift; use the new `detect_beat_grid_with_options` (below) to narrow the
  range or hint a genre. `analyze_for_dj` keeps its 100–160 hint.

### Added

- New `engine` module (`src/engine/`): a pull-based, allocation-free,
  real-time-first stage-graph engine — the first cutover milestone of the
  roadmap's varispeed-first architecture. Coexists with the existing
  `StreamProcessor`/`StreamingEngine` surface (frozen, unaffected) rather
  than replacing it yet.
  - `Engine::build(EngineConfig)` returns `EngineHandles { controller,
    processor, source }`. `EngineProcessor::process(&mut [f32])` fills
    exactly the requested frames — infallible and lock-free on the audio
    thread. `EngineController` writes tempo changes through a lock-free,
    timestamped mailbox (`set_tempo_rate`, `set_tempo_rate_at` for a
    sample-accurate landing point) and drives allocation-free warm-start
    priming for seeks and loop wraps (`warm_start`). `SourceProducer` feeds
    the source ring from the host thread.
  - `EngineProfile::Tape`: varispeed-only chain (pitch follows tempo), zero
    pipeline delay.
  - `EngineProfile::Keylock`: band-split at 150 Hz; the low band stays
    un-keylocked (pure delay) while the high band is pitch-corrected by a
    corrector chosen from transposition magnitude — beat-synchronous
    time-domain SOLA across the full DJ range (rate deviation up to 9%),
    a small-FFT (512/128) phase vocoder with identity phase locking beyond
    it, hysteresis on the handoff, and a graceful fade to plain varispeed
    past 12–22%. Constant pipeline delay of 560 frames (12.7 ms at
    44.1 kHz).
  - `EngineConfig.pre_analysis` attaches a `PreAnalysisArtifact` as the
    engine's primary transient-control signal (splice/phase-reset
    placement steers around onsets), falling back to online spectral-flux
    detection when absent.
  - Measured against the old streaming engine on identical fixtures: kick
    transient sharpness ~70% sharper (1.21 vs 0.68 at ±4% rate), ±8% cents
    ride p95 0.23 vs 1.86 (old engine 12.19), top-octave retention -0.41 vs
    -0.79 dB, envelope swing under unity-crossing rides 0.65 vs 10.36 dB,
    and click-free modulation torture at 1.5x/1.1x theoretical slew (tape)
    and 3x/1.3x (keylock) vs the old engine's 6x/1.5x bound. Final A/B
    matrix run: 9 of 9 parity rows won.
- `TempoTrackingOptions` (re-exported at the crate root): tune the tempo
  search range, octave-prior center/width, or add a soft genre hint range.
- `detect_beat_grid_with_options(samples, sample_rate, &TempoTrackingOptions)`:
  `detect_beat_grid` with explicit tracking options.
- `TempoSegment` (re-exported at the crate root): a piecewise-constant
  tempo stretch (`start_beat`, `bpm`); serialized in the
  `PreAnalysisArtifact` and used as `BeatGrid`'s tempo model.

### QA

- New engine gate suites: `qa/engine_keylock.rs`, `qa/engine_transients.rs`,
  `qa/engine_wcet.rs` (per-callback WCET budget gates wired into CI),
  `qa/engine_ab.rs` and `qa/engine_ab_matrix.rs` (machine-readable
  old-vs-new parity dashboard), plus `tests/engine_latency.rs`,
  `tests/engine_modulation_torture.rs`, and
  `tests/engine_realtime_allocations.rs`.
- `tests/beat_tracking.rs`: new accuracy suite for the tempogram tracker;
  `qa/bpm_accuracy.rs` gains the 5-track BPM corpus wired into the
  manifest.
- RubberBand comparison anomaly explained: the historic ~-24 LUFS /
  ~0.15-similarity rows on chirp/sweep content were the frozen offline
  hybrid driver (`src/stretch/hybrid.rs`) attenuating uniformly ~28 dB
  whenever a preset was set, independent of the new engine.
  `qa/rubberband_comparison.rs` gains a `TIMESTRETCH_RUBBERBAND_ENGINE=new`
  mode that renders through the new keylock chain for comparison.

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

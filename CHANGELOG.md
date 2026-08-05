# Changelog

## 0.11.0

### Breaking changes

- `timestretch-cli analyze` writes the binary `.tsa` analysis container
  by default (`<input>.tsa`) instead of `<input>.tsanalysis.json`; an
  explicit `-o out.json` keeps the legacy JSON format. `--pre-analysis`
  accepts either format.
- Desktop: tracks now converge to a single `.tsa` sidecar. Valid legacy
  `.tsanalysis.json` artifacts are absorbed into the container on load
  and both legacy sidecars (`.tsanalysis.json`, `.tspeaks`) are deleted
  once the on-disk container supersedes them.

### Deprecated

- `read_preanalysis_json` / `write_preanalysis_json`: use the `.tsa`
  analysis container (`io::tsa`) instead. The JSON pair keeps working
  while downstream consumers migrate.

### Added

- `EngineProfile::WideKeylock`: wide-range Master Tempo deck profile
  (ROADMAP Stage 11) — a full-spectrum FFT-2048 identity-locked
  phase-vocoder corrector with artifact-driven per-band phase resets,
  keylocking across the engine's whole tempo range (rates 0.25–2.0)
  with no correction fade. Constant pipeline delay 2144 frames
  (48.6 ms at 44.1 kHz), its own honest latency contract alongside the
  untouched 12.7 ms keylock chain. Semver note: adding the enum variant
  breaks external exhaustive `match`es over `EngineProfile`.
- `StreamingSincResampler::set_step_anchor`: pins the step-ramp anchor
  so uniformly-produced chunks are consumed uniformly (the wide
  corrector's stream-balance requirement).
- `StreamingSincResampler::flush_into`: drains the sinc lookahead tail
  (feeding just enough zeros to release every output covering real
  input), then resets — so callers can end a stream without losing the
  final half-window of audio.
- `PhaseVocoder::set_wide_ratio_coherence_blend`: holds the
  phase-gradient coherence blend at wide stretch ratios instead of the
  shipped taper (the falsification-confirmed fix for robotic wide
  slowdowns).
- `Stage::warm_start_settle_frames`: per-stage warm-start history need;
  the graph's preroll now takes the chain maximum (keylock preroll
  unchanged).
- Desktop: a Range selector (Standard | Wide) with seek-priced engine
  rebuild that preserves the playhead, and a live pipeline-latency
  readout next to it.
- `.tsa` analysis container (`io::tsa`): one content-bound file per
  track holding the pre-analysis artifact and the 3-band waveform peaks
  as versioned chunks (unknown chunks skip forward-compatibly; readers
  reject-don't-panic on hostile input). Two API layers: bytes
  (`AnalysisFile::to_bytes`/`from_bytes`/`from_bytes_validated`, for
  apps that store analysis blobs in their own database keyed by
  `content_hash`) and sidecar file wrappers with atomic writes
  (`read_analysis_file`, `read_analysis_file_validated`,
  `write_analysis_file`, `analysis_file_path` — `<audio>.tsa`).
  `timestretch-cli analyze` writes both chunks, making it a complete
  offline pre-analysis tool.
- `analysis::waveform`: the desktop app's 3-band waveform peaks pyramid
  moved into the library (`BandPeaks`, `PeakLevel`, `NUM_BANDS`) so any
  frontend gets display peaks without reimplementing the analyzer.
- `PreAnalysisArtifact::matches_identity`: `matches_source` semantics
  for callers that already hold the (rate, length, hash) identity;
  `matches_source` now delegates to it.

### Removed

- Desktop `.tspeaks` sidecar format (introduced on an unreleased
  branch): superseded by the `.tsa` container's PEAK chunk; existing
  files are deleted after migration, peaks recompute in milliseconds.

## 0.10.0

### Added

- `MomentaryLoudness`: a real-time-safe streaming BS.1770-4 momentary
  (400 ms) loudness meter. Construction allocates; the
  `push_stereo`/`process`/`momentary_lufs`/`reset` paths never allocate,
  lock, or panic, so the meter can live inside an audio callback (e.g.
  per-deck LUFS meters in a DJ mixer). Silence and insufficient data
  report the finite floor `MomentaryLoudness::SILENCE_LUFS` (-100.0)
  instead of -inf.
- Rigid beat grids for quantized material: `analyze_for_dj` now fits a
  constant-BPM grid (small BPM search around the tracked tempo × full
  phase circle, scored by kick-band onset energy) and adopts it when the
  phase fit is decisive, replacing the tracked beats. Live/drifting
  material keeps the tracked grid (a tempo ramp never adopts). Corpus
  beat F-measure rose from 71.5% to 93.8% and downbeat F from 20.7% to
  76.4%; adopted grids align to the annotations at sub-millisecond mean
  offset. Public API: `fit_rigid_grid`, `refine_grid_rigid`,
  `RigidGridFit`; `AnalysisReport::rigid_grid_adopted` reports the
  decision and `timestretch-cli analyze` prints it.

### QA

- Signed beat-offset diagnostics in the BPM accuracy harness: per-track
  mean/std/drift of the signed beat error vs annotations
  (`beat_offset_mean_ms`, `beat_offset_std_ms`,
  `beat_offset_drift_ms_per_min`), distinguishing constant phase offset
  from jitter from period drift.

## 0.9.1

### Added

- ITU-R BS.1770-4 / EBU R128 loudness metering via the `ebur128` crate
  (pure Rust): `measure_loudness(interleaved, channels, sample_rate)`
  returns gated integrated LUFS, oversampled true peak (dBTP), and
  loudness range (LU) as a `LoudnessMeasurement`, with `gain_db_to` /
  `gain_linear_to` helpers for track autogain. Measured on the original
  interleaved channels — BS.1770 sums per-channel energies, so the mono
  analysis downmix would read up to ~3 dB low.
- `PreAnalysisArtifact` gains an optional `loudness` field (schema v6,
  additive — v4/v5 sidecars stay compatible). `analyze_for_dj` does not
  fill it (it only sees the mono signal); callers measure and store it,
  as `timestretch-cli analyze` now does. The simplified RMS
  `estimate_lufs` in the comparison module is unchanged and remains a
  benchmark-only A/B utility.
- Ranked tempo candidates: the tempo tracker now scores the half/double
  metrical alternatives of its chosen path with the same normalized
  tempogram-salience measure and exposes them as `TempoCandidate`s
  (`BeatGrid::tempo_candidates`, `TempoTrack::octave_saliences`, and
  `PreAnalysisArtifact::tempo_candidates` — schema v7, additive).
  Alternatives outside the tempo search range are not offered. Enables a
  one-tap "halve/double BPM" correction ranked by measured evidence
  instead of blind ×2/÷2 buttons; `timestretch-cli analyze` prints the
  alternatives.

## 0.9.0

### Breaking changes

- `PreAnalysisArtifact` gains a public `key: Option<KeyEstimate>` field
  (schema v5). Code constructing the artifact with an exhaustive struct
  literal must add the field or use `..Default::default()`; existing v4
  JSON sidecars remain compatible and deserialize with `key = None`.

### Added

- Musical key detection: `detect_key(samples, sample_rate)` estimates the
  key of a mono signal via HPCP-style chroma (spectral peaks attributed to
  their candidate fundamentals with decaying weight) scored against the 24
  rotated Krumhansl-Kessler profiles by Pearson correlation, with global
  tuning correction for off-A440 masters and a conservatively gated
  parallel-mode check on a harmonically masked view of the spectrogram.
  On the key-annotated benchmark corpus it matches Mixed In Key on 6 of 7
  tracks (85.7% exact, 88.6% MIREX-weighted).
- `KeyEstimate` / `KeyMode`: root pitch class, mode, and confidence, with
  `name()` ("A minor") and `camelot()` ("8A") for harmonic-mixing UIs.
- `analyze_for_dj` now computes the key and stores it in the artifact;
  `timestretch-cli analyze` prints it (plus `METRIC key=...` lines with
  `--verbose`).

### QA

- Key ground truth in `benchmarks/manifest.toml` (`key = "3A"` Camelot
  notation) and MIREX-style key scoring in the BPM accuracy harness:
  EXACT/FIFTH/RELATIVE/PARALLEL/OTHER classes, a weighted score, and a
  `TIMESTRETCH_KEY_MIN_EXACT` CI floor.
- Beat/downbeat ground-truth annotations for 13 corpus tracks plus the
  `annotate_rigid_grid` example that generated them (rigid-grid phase fit
  on kick-band onset energy, independent of the production detector), and
  recorded BPM accuracy baselines under `benchmarks/baselines/`.

## 0.8.1

### Added

- `PreAnalysisArtifact::resample_to`: rescales every frame-domain position
  (beats, onsets, downbeat offset, analysis hop, source length) to a target
  sample rate, so a track can be analyzed once at its native rate and
  reused at any playback rate. Rate-invariant fields (BPM, confidence,
  downbeat indices, tempo segments, transient strengths, band flux) pass
  through untouched; the returned artifact's content binding is cleared,
  since it no longer corresponds to a concrete signal.

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

### Added

- Keylock is a live engine parameter: `EngineController::set_keylock(bool)`
  (with `keylock_target()` to read the requested state) toggles high-band
  pitch correction click-free during playback via a ~12 ms per-sample
  crossfade to delay-matched varispeed. Deck-style Tape/Keylock switching
  no longer needs an engine rebuild; SOLA stays warm while bypassed so
  re-engage is instant.

### Changed

- **MSRV raised from 1.82 to 1.85**; the crate (and the desktop app)
  now use Rust edition 2024, which also enables cargo's MSRV-aware
  dependency resolver (v3).
- Development toolchain pinned to Rust 1.97.0 via `rust-toolchain.toml`;
  CI's stable jobs build with the pinned compiler instead of floating on
  latest stable (the MSRV job still tests 1.85.0 explicitly).

### Fixed

- Detected BPM no longer carries the analysis frame-grid bias: the
  representative and per-segment BPMs are computed as the median of
  K-beat-baseline intervals, so position errors telescope instead of
  skewing adjacent intervals (a ground-truth 125.000 BPM kick train now
  reads 125.03 instead of 125.23; a real 125.0 EDM track 124.99 instead
  of 125.22).
- Onset and beat positions are compensated for the analysis window's
  detection latency, so markers land on the audible attack instead of
  ~29 ms early (measured bias drops from -31.6 ms to -2.6 ms on a
  ground-truth kick train). `PREANALYSIS_VERSION` bumps to 4; older
  cached sidecars regenerate automatically.

### Removed

- The `web/` WASM demo. It was built against the old engine API deleted
  in 0.8.0 (`StreamProcessor`, `EdmPreset`) and had no CI coverage, so it
  no longer compiled. Last present at commit `22e5117` if a web demo is
  ever revived.

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

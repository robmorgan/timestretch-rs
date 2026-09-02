# Changelog

## Unreleased

### Changed

- `analyze_for_dj` / `analyze_for_dj_with_report` are renamed to `analyze`
  / `analyze_with_report`. The `_for_dj` suffix dated from when the
  offline pass was framed as a DJ-specific feature; it is now simply
  *the* analysis entry point, used by the CLI, the desktop app, the
  offline engine, and the QA harnesses. The old names remain as
  `#[deprecated]` aliases, so downstream code keeps compiling.

## 0.14.0

### Performance

- `AudioBuffer::resample` / `resample_sinc` precompute the windowed-sinc
  kernel as a polyphase row table (the streaming kernel's machinery)
  instead of evaluating `sin` per tap per output sample: a 5-minute
  stereo 44.1→96 kHz conversion drops from ~6.7 s to ~0.9 s (7.5x),
  with the same anti-aliasing cutoff policy and edge renormalization.

### Changed

- Downbeat election (both the accent scorer and the rigid grid's
  rotation) gains a sustained low-band ENERGY feature, self-gated by its
  own cross-beat contrast. Flux/onset features tie on four-on-floor
  material — every beat carries an equal kick — and the log-difference
  onset actively favors the snare (a bass fill before the kick
  suppresses its jump), so beat one landed on beats 2/4 and a DJ host's
  auto-cue parked on a snare. The bassline's emphasis on the true one
  is sustained energy (measured 30–280 ms after the beat, past the
  shared kick attack), which difference features cannot see. Cached
  artifacts regenerate: `PREANALYSIS_VERSION` and
  `MIN_COMPATIBLE_VERSION` rise to 13 — sampled house-corpus elections
  changed on 2 of 5 tracks, both toward the measured bass phase, with
  backbeat and accent-driven material unchanged.
- Waveform peaks are 10x denser: `BASE_BUCKETS_PER_SEC` rises from 150
  to 1500, so zoomed deck views (~1000 px/s at 1-bar zoom) get
  near-per-pixel peak data instead of 6–7 px flat-topped buckets.
  Compute cost is unchanged (the crossover biquads dominate); the costs
  are RAM (~26 MB pyramid per 6-min track) and a ~10x larger `.tsa`
  PEAK chunk. No `.tsa` format bump: stale cached peaks fail
  `decode_peaks` validation and recompute on next load, while the
  beat-grid artifact survives.

### Fixed

- The keylock chain's frame-domain tuning was fixed at 44.1 kHz values;
  on builds at higher sample rates every window, trigger, and fade
  described half (96 kHz) or a quarter (192 kHz) of its designed time.
  Two audible failures at 96 kHz, both found via Halo running its deck
  engines at the device rate: the Stage 21 bass corrector's period
  search bottomed out at ~74 Hz, so a real bass fundamental was
  unsearchable and the low end flapped between corrected and
  pitch-follow under a sustained DJ offset; and the high-band SOLA
  corrector's correlation window and search range could not align
  content below ~300 Hz, so low-mid splices landed at random phase
  (measured: a 150 Hz tone corrected at purity 1.000 at 44.1 kHz vs
  0.005 at 96 kHz — muddled, wobbly, "underwater" mids). Both
  correctors now scale their reference constants to the build rate at
  construction, and the chain's nominal lag scales above 44.1 kHz so
  the 12.7 ms latency contract — and the trigger corridor it anchors —
  is time-true at every rate (`pipeline_latency_frames` reports the
  scaled figure; hosts that DISPLAY it in milliseconds see 12.7 ms at
  every rate now). At and below 44.1 kHz everything is bit-identical
  to the blind-validated behavior. Regression tests pin sub-band
  correction at 48/96/192 kHz and 96 kHz low-mid purity.
- The transient cursor's retention and lookahead windows
  (`KEEP_BEHIND_FRAMES`, `HORIZON_FRAMES`) scale with the build rate
  the same way (with the graph's timeline-eviction margin following):
  unscaled, a 96 kHz build dropped onsets while the correctors' scaled
  masked windows still addressed them, silently weakening
  masked-window splice placement.

## 0.13.0

### Changed

- The keylock profile's low band (sub-120 Hz) is now pitch-corrected by
  a period-aligned SOLA-class bass corrector (ROADMAP Stage 21): splices
  jump whole bass periods, correlation-aligned, hidden in quiet moments
  away from kick onsets, with one lockstep splice decision across
  channels. Correction engages beyond ~±1–2% tempo deviation — mild
  nudges keep the traditional pitch-follow bass and a rigid crossover
  seam, while sustained DJ offsets (±8%) play in key. Latency contract
  unchanged (12.7 ms). Supersedes the Stage 2 scope line, whose blind
  verdict rejected a vocoder bass: the time-domain corrector won the
  Stage 21 blind re-match in all four ±8% conditions.

### QA

- `scripts/ab.sh` gains `--env-arm label:VAR=val`: extra blind arms
  rendered from the current tree under env settings, for env-gated
  prototypes (used by both Stage 20 and Stage 21 kill experiments).
- Stage 20 (bounded stereo-width treatment) was falsified blind and
  closed without shipping: side-level-matched mid-derived injection
  reads underwater at any gain, and the width preference itself did
  not reproduce against the current wide head. Faithful stereo stays
  the shipped behavior; evidence archived in LEARNINGS.

## 0.12.0

The quality-closure roadmap release: every open stage from the 2026-08-05
review (Stages 10, 12–19) closed with a recorded blind-listening verdict.

### Breaking changes

- `EngineProfile::WideKeylock` has a new head: a direct-ratio
  phase-vocoder demand inverter (`WidePvHead`) owns the tempo axis
  instead of a varispeed-then-correct chain. Its latency contract
  changed from a 48.6 ms pipeline delay to **0 ms** — the analysis
  window is source-side lookahead, so like tape the first delivered
  frame is source frame 0. Hosts that compensated the old wide delay
  externally must drop that offset (position queries were and remain
  self-compensating). Retarget landing in the wide profile is
  hop-quantized (≈5.8 ms); emission counts stay frame-exact.
- Cached analysis artifacts regenerate: `PREANALYSIS_VERSION` is now 12
  and `MIN_COMPATIBLE_VERSION` rose to 12 (from 7/4 at v0.11.0), because
  beat grids and tempo estimates changed materially across this span
  (phase hygiene, rigid-grid adoption, metrical-level second pass).
  `.tsa` sidecars from earlier versions are re-analyzed on load.

### Added

- `EngineController::retargets_degraded()`: counts timestamped
  retargets degraded to immediate latest-wins because more than
  `MAX_PENDING_RETARGETS` were in flight (#45 — previously silent;
  `dropped_events()` only covers mailbox overflow). The constant is now
  public, and `set_tempo_rate_at` documents the cap and the
  ≤8-in-flight scheduling pattern for long tempo curves.
- Metrical-level second pass in beat tracking: when the 3/2 tempo
  candidate's salience clears a measured threshold, tracking re-runs
  with the prior centered at that level and adopts on convergence —
  drum & bass now detects its true ~174 BPM instead of the 2/3
  sub-level. DJ tempo hint range widened 100–160 → 100–182 BPM.
- Stereo in the wide profile runs mid/side: source-faithful width
  (per-channel processing manufactured ~16 dB of side energy by
  decorrelation).

### Changed

- Keylock chain: steady-rate cadence stretch for SOLA splices — at
  sustained slowdowns the corrector splices at twice the steady cadence
  with a tapered stretch budget, removing the audible splice-rate
  artifact (gated off during tempo ramps to protect ride cymbals).
- Phase-vocoder phase hygiene (Stage 13): per-sample fade ramps and a
  modulation-hold contract fix batch-path phase artifacts.
- Rigid-grid beat corroboration adopts rigid grids on syncopated
  material, and estimator disagreement caps stored artifact confidence
  at 0.5 so hosts can flag uncertain grids.
- Keylock seam: mild-motion bounded recenter — the SOLA seam survives
  sustained ride cymbals (Stage 15).

### QA

- Blind A/B harness: `scripts/ab.sh` renders level-matched, blinded,
  sealed-key arm sets (current tree, any git ref, rubberband reference);
  `tools/ab-tui` is an interactive two-pane listening TUI writing
  machine-readable verdicts (`results.json`).
- Non-EDM benchmark corpus rows (hip-hop, rock, funk/live) plus two
  CC-licensed drum & bass public-corpus rows; CI now enforces ≥90%
  tempo-accuracy floors on the public corpus.
- Engine soak hardening (Stage 12): position-drift gate, no-panic
  audit, weekly re-seeded fuzz campaign; the soak also gates
  `retargets_degraded() == 0`.

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

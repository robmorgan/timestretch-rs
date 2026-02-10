# Progress

## Completed — Phase 1: Foundation (agent-3)
- [x] Cargo.toml with `rustfft` dependency, `#![forbid(unsafe_code)]`
- [x] Core types: `AudioBuffer`, `Sample`, `Channels`, `StretchParams`, `EdmPreset`
- [x] Error types: `StretchError` enum with descriptive variants
- [x] Window functions: Hann, Blackman-Harris, Kaiser-Bessel (with Bessel I0 series expansion)
- [x] WAV reader: 16-bit PCM, 24-bit PCM, 32-bit float IEEE
- [x] WAV writer: 16-bit PCM and 32-bit float
- [x] Linear and cubic (Hermite) resampling

## Completed — Phase 2: Algorithms (agent-3)
- [x] Phase vocoder with identity phase locking and sub-bass phase locking
- [x] WSOLA with normalized cross-correlation matching
- [x] Transient detector: spectral flux with adaptive threshold, HF-weighted bins (2-8kHz)
- [x] Beat detection with BPM estimation and grid quantization

## Completed — Phase 3: EDM Optimization (agent-3)
- [x] Hybrid stretcher: transient-aware switching between WSOLA and PV
- [x] Crossfade engine: raised-cosine crossfade between algorithm segments
- [x] Frequency band splitter (sub/low/mid/high)
- [x] All 5 EDM presets: DjBeatmatch, HouseLoop, Halftime, Ambient, VocalChop

## Completed — Phase 4: Streaming & API (agent-3)
- [x] `StreamProcessor` with chunk-based processing
- [x] Real-time stretch ratio changes (smooth interpolation)
- [x] Latency reporting API
- [x] Public API: `stretch()`, `stretch_buffer()`, builder pattern for params
- [x] CLI binary (behind `cli` feature flag)

## Completed — Phase 5: Testing (agent-3)
- [x] 65 unit tests across all modules (64 original + 1 new WSOLA extreme compression)
- [x] 11 identity tests (mono, stereo, 48kHz, all presets, frequency, SNR, sub-bass, near-unity, DC, clicks)
- [x] 5 quality tests (RMS preservation, length proportionality, clipping, DJ quality, sub-bass)
- [x] 6 EDM preset integration tests
- [x] 6 streaming tests (basic, stereo, ratio change, reset, small chunks, latency)
- [x] 6 benchmark tests
- [x] 1 doc-test
- [x] Total: 100 tests, all passing
- [x] Zero clippy warnings

## Agent-4 Contributions
- [x] Implemented full test suite: 53 integration tests across identity, quality, EDM presets, streaming, and benchmarks
- [x] Applied clippy fixes to src/ files (frequency.rs, transient.rs, phase_vocoder.rs, wsola.rs, lib.rs)
- [x] 95 unit tests across all modules
- [x] 148 total tests, all passing
- [x] `#![forbid(unsafe_code)]` enforced

## Agent-2 Contributions
- [x] Fixed clippy warnings in library code (frequency.rs, resample.rs, hybrid.rs)
- [x] Fixed clippy warnings in tests (streaming.rs, edm_presets.rs)
- [x] Fixed clippy warnings in examples (dj_beatmatch.rs, realtime_stream.rs, sample_halftime.rs)
- [x] Fixed example files to match current API (removed `.expect()` from non-Result returns)
- [x] Fixed WSOLA compression ratio accuracy (fractional output position tracking)
- [x] Implemented FFT-accelerated cross-correlation for WSOLA (auto-selects FFT vs direct based on search range)
- [x] Added comprehensive identity tests (merged with agent-3's tests: frequency, SNR, multi-freq, transients, stereo, sub-bass, DC offset)
- [x] Added 15 edge case tests (boundary inputs, extreme ratios, silence, impulse, DC offset, all presets with compression)

## Agent-5 Refactoring
- [x] Added builder methods to `StretchParams`: `with_sub_bass_cutoff()`, `with_wsola_segment_size()`, `with_wsola_search_range()`
- [x] Extracted `reconstruct_spectrum()` and `normalize_output()` from `PhaseVocoder::process()` (was 104 lines)
- [x] Extracted `stretch_segment()` and `stretch_with_wsola()` from `HybridStretcher::process()` (was 88 lines)
- [x] Added getter methods to `PhaseVocoder` (`fft_size()`, `hop_analysis()`, `hop_synthesis()`)
- [x] Added getter methods to `Wsola` (`segment_size()`, `search_range()`, `stretch_ratio()`)
- [x] Removed unused `ProcessingError` variant from `StretchError`
- [x] Extracted `write_wav_header()` helper to eliminate WAV writer duplication (wav.rs)
- [x] Extracted `create_vocoders()` helper to eliminate 3x duplicated PhaseVocoder init (processor.rs)
- [x] Simplified redundant guard in `segment_audio()` (hybrid.rs)
- [x] All 66 unit tests passing, zero clippy warnings

### Second pass (agent-5)
- [x] Split `read_wav()` into `validate_riff_header()`, `parse_wav_chunks()`, `convert_samples()`
- [x] Extracted `io_error()` helper to deduplicate 6 identical WAV I/O error formatting calls
- [x] Deduplicated `write_wav_file_16bit`/`write_wav_file_float` via shared `write_wav_file()`
- [x] Extracted `compute_spectral_flux()` from `detect_transients()` (was 64 lines, now 24)
- [x] Added `StretchParams::output_length()` helper, used in hybrid stretcher (eliminates 2 duplicated calculations)
- [x] Added doc comments for `with_preset()` and `with_sample_rate()` documenting override behavior
- [x] Removed unused `sub_bass_bin` field and `#[allow(dead_code)]` from `PhaseVocoder`
- [x] All 68 unit tests passing, zero clippy warnings

## Completed — Performance Optimization (agent-3)
- [x] Comprehensive benchmarks: PV mono/stereo, EDM presets, FFT sizes, streaming, signal scaling
- [x] PhaseVocoder: reuse FFT/magnitude/phase buffers, pre-compute phase advance, efficient wrap_phase
- [x] StreamProcessor: persistent per-channel PhaseVocoder instances, reusable deinterleave buffers
- [x] HybridStretcher: reuse single PV instance across tonal segments, fast path for single segment
- [x] Transient detector: reusable sort buffer in adaptive_threshold
- [x] Result: **5-7x speedup** across all processing paths
  - PV mono: 161ms → 28ms (176x realtime vs 31x before)
  - PV stereo: 321ms → 54ms (93x realtime vs 16x before)
  - Streaming: 436ms → 59ms (169x realtime vs 23x before)
- [x] Fixed WSOLA compression accuracy for ratio < 0.5 (trim to target length)
- [x] Added extreme compression tests (ratios 0.25-0.5)

## Completed — Comprehensive Identity Tests (agent-3)
- [x] 11 identity tests (was 4): frequency preservation, SNR, sub-bass coherence, near-unity ratios, DC offset, click timing
- [x] Total: 100 tests, all passing
- [x] Zero clippy warnings

## Streaming Fix & Equivalence Tests (agent-3)
- [x] Fixed StreamProcessor overlap handling: drain processed input correctly instead of re-processing overlap, which caused output to be ~2x expected length
- [x] Added 8 new streaming tests (14 total, was 6):
  - Streaming vs batch output length comparison (multiple ratios: 0.75, 1.0, 1.25, 1.5, 2.0)
  - Streaming vs batch RMS energy preservation
  - Streaming vs batch frequency content preservation (440Hz DFT check)
  - Chunk size consistency (512, 4096, 16384 produce similar RMS)
  - Stereo streaming vs batch equivalence (channel separation verified)
  - Streaming with EDM preset (DjBeatmatch)
  - Flush produces remaining output
  - Reset-then-reprocess consistency
- [x] Added 4 streaming edge case tests (18 total):
  - Compression ratio correctness (0.5, 0.75)
  - Empty/double flush safety
  - Single-sample chunks
  - Large FFT size (8192)
- [x] Total: 131 tests, all passing
- [x] Zero clippy warnings

## Agent-1 Contributions
- [x] Fixed WSOLA compression ratio accuracy: added early loop termination for compression and removed `overlap_size` padding from output trimming
- [x] Added `Clone`, `PartialEq`, `Eq` derives to `StretchError` for better ergonomics
- [x] Merged and expanded edge case tests (combined with agent-2's tests):
  - Extreme compression (0.25x) and stretch (4.0x, 10.0x)
  - Silence input, DC offset input, impulse input
  - Very short input (50 and 100 samples), single sample input
  - Parameter boundary validation (min/max ratios, invalid ratios)
  - WSOLA compression accuracy across multiple ratios
  - NaN/Inf output detection across all ratios
  - Stereo channel independence and mono/stereo consistency
  - Frequency edge cases (20 Hz sub-bass, 15 kHz near-Nyquist)
  - All EDM presets with compression
  - Alternating silence/tone patterns
  - FFT size variations (256, 8192)
- [x] Total: 137 tests, all passing

## Agent-2 Phase Locking & Streaming Fixes
- [x] Implemented sub-bass phase locking in PhaseVocoder (bins below cutoff get rigid phase propagation instead of standard deviation tracking)
- [x] Sub-bass bin index computed from cutoff frequency and sample rate at construction time
- [x] Identity phase locking now only applies to bins above sub-bass range
- [x] Added `set_stretch_ratio()` to PhaseVocoder for in-place ratio updates without phase reset
- [x] Fixed StreamProcessor to use `set_stretch_ratio()` instead of recreating vocoders on ratio change (prevents clicks)
- [x] Added 4 new unit tests: sub-bass bin calculation, low-freq preservation, high-freq independence, click-free ratio changes
- [x] Fixed identity phase locking bug: was a no-op (`phases[bin] = phases[peak] + phases[bin] - phases[peak]`); now correctly uses analysis phase relationships per Laroche & Dolson (1999)
- [x] Added `pitch_shift()` public API: time-stretch + cubic resample for pitch shifting without duration change
- [x] Added 4 pitch_shift tests: length preservation, empty input, invalid factor, stereo
- [x] Total: 166 tests, all passing
- [x] Zero clippy warnings on library code

## Documentation (agent-3)
- [x] Comprehensive README.md with feature overview, quick start, streaming example, EDM presets table, algorithm explanation, parameter reference, performance data
- [x] Applied cargo fmt across entire codebase (21 files)

## Enhanced Identity Tests (agent-3)
- [x] 9 new identity tests (21 total, was 12):
  - Waveform cross-correlation (must be > 0.9)
  - Max per-sample error bound (< 0.5)
  - Silence preservation (RMS and peak both < 1e-6)
  - Peak amplitude preservation (ratio within [0.5, 1.5])
  - No spectral coloring (low/high energy ratio preserved)
  - Streaming vs batch identity equivalence
  - Stereo with one silent channel (no channel leakage)
  - Click timing preservation (transient positions maintained)
  - Per-segment energy distribution (no energy redistribution)
- [x] Total: 153 tests, all passing
- [x] Zero clippy warnings

## Documentation (agent-1)
- [x] Merged and improved README (combined agent-1 and agent-3 versions)
- [x] Added crate-level rustdoc with compilable examples (quick start + streaming)
- [x] Added rustdoc examples to `stretch()`, `stretch_buffer()`, `AudioBuffer`, `StretchParams`, `EdmPreset`, `StreamProcessor`
- [x] Added `# Errors` sections to public functions

### Third pass (agent-5)
- [x] Extracted `fft_cross_correlate()` and `find_best_candidate()` from `Wsola::find_best_position_fft()` (was 91 lines, now 47)
- [x] Extracted `analyze_frame()` and `advance_phases()` from `PhaseVocoder::process()` (was 83 lines, now 38); includes sub-bass rigid phase locking and analysis phase tracking
- [x] Extracted `process_channels()`, `deinterleave_channel()`, `drain_consumed_input()`, `interleave_output()` from `StreamProcessor::process()` (was 80 lines, now 28)
- [x] Replaced magic numbers with named constants: `ENERGY_EPSILON`, `FFT_CANDIDATE_THRESHOLD`, `FFT_OVERLAP_THRESHOLD` (wsola.rs), `MIN_WINDOW_SUM_RATIO`, `WINDOW_SUM_EPSILON` (phase_vocoder.rs), `RATIO_SNAP_THRESHOLD`, `RATIO_INTERPOLATION_ALPHA` (processor.rs)
- [x] All tests passing, zero clippy warnings

## CI/CD Pipeline (agent-1)
- [x] GitHub Actions workflow: `.github/workflows/ci.yml`
  - Test job: runs `cargo test --all-targets` on ubuntu, macOS, windows + MSRV (1.65)
  - Clippy job: `cargo clippy --all-targets -- -D warnings`
  - Format job: `cargo fmt --all --check`
  - Documentation job: `cargo doc --no-deps` with `-D warnings`
- [x] Added `rust-version = "1.65"` MSRV to Cargo.toml
- [x] Fixed MSRV incompatibility: replaced `is_multiple_of()` (1.87+) with `% 2 != 0` in wav.rs
- [x] Applied `cargo fmt` to fix formatting drift from recent refactors
- [x] All 165 tests passing, zero clippy warnings, docs build clean

### Fourth pass (agent-5)
- [x] Added `ms_to_samples()` helper and `WSOLA_SEGMENT_MS`/`WSOLA_SEARCH_MS_*` constants to `types.rs`, replacing 6 inline `(sample_rate * seconds)` calculations in preset configuration
- [x] Extracted 7 named constants in `hybrid.rs`: `CROSSFADE_SECS`, `TRANSIENT_REGION_SECS`, `MIN_SEGMENT_FOR_STRETCH`, `MIN_WSOLA_SEGMENT`, `MIN_WSOLA_SEARCH`, `TRANSIENT_MAX_FFT`, `TRANSIENT_MAX_HOP`
- [x] Simplified `stretch_segment()` conditional chain from nested if-else-if to single `use_phase_vocoder` boolean (eliminates duplicate WSOLA branch)
- [x] Extracted 3 named constants in `transient.rs`: `MEDIAN_WINDOW_FRAMES`, `MIN_ONSET_GAP_FRAMES`, `THRESHOLD_FLOOR`
- [x] Extracted 5 named constants in `beat.rs`: `BEAT_FFT_SIZE`, `BEAT_HOP_SIZE`, `BEAT_SENSITIVITY`, `MIN_EDM_BPM`, `MAX_EDM_BPM`
- [x] Extracted shared `deinterleave()` and `interleave()` helpers in `lib.rs`, deduplicating identical channel processing in `stretch()` and `pitch_shift()` (net -23 lines)
- [x] Extracted `process_buffer()` helper in `lib.rs`, deduplicating the params-override-and-wrap pattern across `stretch_buffer()`, `stretch_bpm_buffer()`, and `stretch_bpm_buffer_auto()` (net -7 lines)
- [x] Extracted `Band` enum and `classify_bin()` helper in `frequency.rs`, deduplicating the if-elif band classification pattern used 3x in `split_spectrum_into_bands()` and 1x in `compute_band_energy()` (net -12 lines)
- [x] Removed unused `snr_db()` function and `#[allow(dead_code)]` from `tests/quality.rs`
- [x] All tests passing, zero clippy warnings

## BPM-Aware Stretch API (agent-3)
- [x] `stretch_to_bpm(input, source_bpm, target_bpm, params)` — stretch audio between known BPM values
- [x] `stretch_to_bpm_auto(input, target_bpm, params)` — auto-detect source BPM via beat detection, then stretch
- [x] `stretch_bpm_buffer()` and `stretch_bpm_buffer_auto()` — AudioBuffer convenience wrappers
- [x] `bpm_ratio(source, target)` — utility to compute stretch ratio from BPM pair
- [x] `BpmDetectionFailed` error variant for invalid BPM values and failed auto-detection
- [x] 14 BPM integration tests: DJ beatmatch (126→128), slowdown (128→126), halftime (128→64), doubletime (128→256), stereo, all presets, buffer API, auto-detection with clicks, invalid BPMs, 48kHz
- [x] 12 BPM unit tests: ratio utility, speedup, slowdown, same BPM identity, empty input, silence auto-detect, buffer API
- [x] Total: 189 tests, all passing
- [x] Zero clippy warnings

## WAV Round-Trip Integration Tests (agent-1)
- [x] 11 new integration tests in `tests/wav_roundtrip.rs`:
  - 16-bit WAV encode/decode round-trip (mono)
  - 32-bit float WAV encode/decode round-trip (stereo)
  - Stretch through 16-bit WAV pipeline (mono, 1.5x)
  - Stretch through float WAV pipeline (stereo, 0.75x compression)
  - EDM kick pattern DJ beatmatch stretch (128→126 BPM)
  - Kick pattern halftime effect (2.0x)
  - 16-bit vs float stretch consistency
  - 48 kHz sample rate WAV stretch
  - All 5 EDM presets WAV round-trip
  - Double stretch pipeline (stretch → WAV → stretch back)
  - Stretched output preservation through WAV encode/decode
- [x] Applied cargo fmt to fix formatting drift from other agents' changes
- [x] Total: 177 tests (168 integration/unit + 9 doc-tests), all passing

## Spectral Quality Tests (agent-3)
- [x] 13 spectral quality tests covering:
  - Frequency preservation: 440Hz energy dominant after stretch at 5 ratios (0.75-2.0)
  - Multi-tone: two-tone signal preserves both frequencies, no broadband smearing
  - Sub-bass: 60Hz energy preserved after 1.5x stretch
  - Dominant frequency preservation: 3-tone signal maintains frequency hierarchy at 4 ratios
  - No new harmonics: pure sine doesn't introduce strong 2nd/3rd harmonics
  - Band energy distribution: two-tone signal preserves relative energy ordering
  - Transient attack: click at 0.5s appears at ~0.75s after 1.5x stretch
  - Click train spacing: regular clicks maintain proportional intervals
  - DJ transparency: small-ratio stretch preserves 3-tone spectral content
  - Extreme stretch: 4x stretch retains fundamental frequency
  - Compression: 0.5x preserves fundamental
  - Stereo spectral independence: L=440Hz R=880Hz stay in correct channels
  - Frequency sweep: no silent "holes" in chirp output
- [x] Total: 205+ tests, all passing
- [x] Zero clippy warnings

## Pitch Shift Integration Tests (agent-1)
- [x] 14 new integration tests in `tests/pitch_shift.rs`:
  - Length preservation across multiple pitch factors (mono and stereo)
  - Frequency shift verification: octave up (440→880Hz) and octave down (880→440Hz)
  - Identity test (pitch factor 1.0 preserves frequency content)
  - Small DJ-style adjustments (±2%)
  - Stereo channel independence
  - Extreme factors (4.0x up, 0.25x down)
  - No clipping across all factors
  - Silence preservation
  - All 5 EDM presets compatibility
  - 48kHz sample rate
  - NaN/Inf sweep across 10 pitch factors
- [x] Total: 206 tests (197 integration/unit + 9 doc-tests), all passing
- [x] Zero clippy warnings

## Agent-2: API Improvements & Input Validation
- [x] Added `StretchParams::from_tempo(source_bpm, target_bpm)` convenience constructor for DJ workflow
- [x] Added `StretchError::NonFiniteInput` variant for NaN/Inf rejection
- [x] Added NaN/Inf input validation to `stretch()`, `pitch_shift()`, `stretch_to_bpm_auto()`, and `StreamProcessor::process()`
- [x] Added Cargo.toml metadata: `keywords`, `categories`, `readme`
- [x] Added 7 new tests: `from_tempo` unit test, NaN/Inf rejection (stretch, pitch_shift, streaming), from_tempo integration
- [x] All 226 tests passing, zero clippy warnings

### Fifth pass (agent-5)
- [x] Extracted `convert_pcm_16bit()`, `convert_pcm_24bit()`, `convert_ieee_float_32bit()` from `convert_samples()` in wav.rs (was 53 lines, now 10-line dispatch)
- [x] Extracted PCM scaling constants in wav.rs: `PCM_16BIT_SCALE`, `PCM_16BIT_MAX_OUT`, `PCM_24BIT_SCALE`, `PCM_24BIT_SIGN_BIT`, `PCM_24BIT_MASK`, `WAV_MIN_HEADER_SIZE`, `WAV_FMT_MIN_SIZE`
- [x] Extracted frequency band weight constants in transient.rs: `WEIGHT_SUB_BASS`, `WEIGHT_BASS_MID`, `WEIGHT_MID`, `WEIGHT_HIGH_MID`, `WEIGHT_VERY_HIGH` and band boundary constants
- [x] Extracted Blackman-Harris window coefficients as module-level constants in window.rs: `BH_A0`..`BH_A3`
- [x] All tests passing, zero clippy warnings

## BPM Detection Public API (agent-3)
- [x] `detect_bpm(samples, sample_rate)` — convenience function returning BPM as f64
- [x] `detect_beat_grid(samples, sample_rate)` — returns full BeatGrid with beat positions
- [x] Re-exported `BeatGrid` from top-level `timestretch` module
- [x] 5 new unit tests: silence, empty, short input, beat grid, click train BPM detection
- [x] 2 new doctests for `detect_bpm()` and `detect_beat_grid()`
- [x] Total: 243 tests, all passing
- [x] Zero clippy warnings

## Test Signal Generator (agent-1)
- [x] `test_audio/generate.rs` example binary — generates 8 WAV test files:
  - `sine_440hz.wav` — 2s mono 440 Hz sine
  - `sine_60hz.wav` — 2s mono 60 Hz sub-bass
  - `sine_stereo.wav` — 2s stereo (L=440Hz, R=880Hz)
  - `click_train_128bpm.wav` — 4s click train at 128 BPM
  - `kick_pattern_128bpm.wav` — 4s 808-style EDM kick at 128 BPM
  - `white_noise.wav` — 2s deterministic pseudo-random noise
  - `sweep_20_20k.wav` — 4s logarithmic frequency sweep 20Hz–20kHz
  - `edm_mix.wav` — 4s layered kick + sub-bass + hi-hat pattern
- [x] Run with `cargo run --example generate_test_audio`
- [x] Generated WAV files gitignored (`test_audio/.gitignore`)

### Sixth pass (agent-5)
- [x] Extracted `validate_input()` helper in lib.rs — deduplicates empty-check + NaN/Inf validation from `stretch()`, `pitch_shift()`, and `stretch_to_bpm_auto()` (net -9 lines)
- [x] Extracted `validate_bpm()` helper in lib.rs — deduplicates BPM positivity checks from `stretch_to_bpm()` and `stretch_to_bpm_auto()` (net -8 lines)
- [x] Added `#[inline]` hints to hot-path helpers: `deinterleave()`, `interleave()`, `validate_input()`, `validate_bpm()` (lib.rs), `overlap_add()` (wsola.rs), `snap_to_grid()` (beat.rs)
- [x] Applied cargo fmt to fix formatting drift from other agents' code
- [x] All 238 tests passing (98 unit + 126 integration + 14 doc), zero clippy warnings

## API Wrappers (agent-3)
- [x] `pitch_shift_buffer(buffer, params, pitch_factor)` — AudioBuffer convenience wrapper for pitch shifting
- [x] `detect_bpm_buffer(buffer)` — BPM detection from AudioBuffer (handles stereo by extracting left channel)
- [x] 4 new unit tests: pitch_shift_buffer mono/stereo, detect_bpm_buffer silence/stereo
- [x] Total: 249+ tests, all passing

## Agent-2: Performance & Beat-Aware Segmentation
- [x] Pre-allocated output/window_sum buffers in PhaseVocoder — eliminates 2 Vec allocations per `process()` call; buffers grow as needed but never shrink
- [x] Integrated beat-aware segmentation into HybridStretcher (Task 14):
  - Beat grid positions merged with transient onsets for segment boundaries
  - Nearby positions (<512 samples apart) deduplicated to avoid micro-segments
  - Short inputs (<1s) skip beat detection to avoid unreliable results
  - Enabled by default for all EDM presets (`beat_aware` field on StretchParams)
  - Added `with_beat_aware()` builder method for manual control
- [x] Added 7 new unit tests: merge logic (empty, no overlap, dedup, bounds), beat-aware integration, short-input safety, flag defaults
- [x] Fixed clippy warning (needless_range_loop → abs_diff)
- [x] Total: 252+ tests, all passing
- [x] Zero clippy warnings

## Streaming DJ API & detect_beat_grid_buffer (agent-1)
- [x] `StreamProcessor::from_tempo(source_bpm, target_bpm, sample_rate, channels)` — convenience constructor for DJ workflow, auto-applies `DjBeatmatch` preset
- [x] `StreamProcessor::set_tempo(target_bpm)` — smoothly change target BPM during playback (requires `from_tempo`)
- [x] `StreamProcessor::source_bpm()` — returns source BPM if set via `from_tempo`
- [x] `StreamProcessor::params()` — read-only access to current parameters
- [x] `detect_beat_grid_buffer(buffer)` — AudioBuffer convenience wrapper for beat grid detection (handles stereo)
- [x] 8 new unit tests: from_tempo, from_tempo_stereo, set_tempo, set_tempo_no_source, set_tempo_invalid, from_tempo_produces_output, params_accessor, detect_beat_grid_buffer mono/stereo
- [x] 5 new streaming integration tests: from_tempo DJ workflow, from_tempo stereo, set_tempo mid-stream, set_tempo without source, from_tempo slowdown
- [x] Updated `realtime_stream` example to showcase `from_tempo()` + `set_tempo()` (3-part demo: BPM matching, live tempo fader, manual ratio)
- [x] Fixed rustdoc ambiguous link warning on `bpm_ratio` (`stretch` → `stretch()`)
- [x] `cargo doc --no-deps` now builds with zero warnings
- [x] Total: 270 tests (118 unit + 138 integration + 14 doc), all passing
- [x] Zero clippy warnings, cargo fmt clean

### Seventh pass (agent-5)
- [x] Moved `preset_description()` standalone function to `EdmPreset::description()` method (more idiomatic Rust)
- [x] Extracted `RATIO_MIN`/`RATIO_MAX`/`FFT_SIZE_MIN`/`SAMPLE_RATE_MIN`/`SAMPLE_RATE_MAX` constants in params.rs — eliminates magic numbers in `validate_params()` and `pitch_shift()`
- [x] Simplified `validate_params()` ratio validation from 3 separate if-checks to single `contains()` range check
- [x] Extracted `BESSEL_MAX_TERMS`/`BESSEL_CONVERGENCE` constants in window.rs
- [x] Extracted `extract_mono()` helper in lib.rs — deduplicates stereo-to-mono extraction in `detect_bpm_buffer()`, `detect_beat_grid_buffer()`, and `stretch_to_bpm_auto()`
- [x] Deduplicated `pitch_shift_buffer()` via `process_buffer()` helper (was 10 lines, now 1-line delegation)
- [x] All 263+ tests passing, zero clippy warnings, cargo fmt clean

## AudioBuffer API & WAV Enhancements (agent-3)
- [x] `AudioBuffer::is_empty()` — returns true if buffer has no samples
- [x] `AudioBuffer::is_mono()` / `is_stereo()` — channel layout predicates
- [x] `AudioBuffer::left()` / `right()` — extract L/R channel samples (mono returns all samples for both)
- [x] `AudioBuffer::mix_to_mono()` — average all channels to mono
- [x] `write_wav_24bit()` / `write_wav_file_24bit()` — 24-bit PCM WAV output (completes read/write symmetry)
- [x] Fixed rustdoc ambiguity warning (`stretch` is both function and module)
- [x] `stretch_wav_file(input, output, params)` — high-level WAV file stretch convenience
- [x] `pitch_shift_wav_file(input, output, params, factor)` — high-level WAV file pitch shift convenience
- [x] `AudioBuffer::to_stereo()` — mono-to-stereo conversion (duplicates signal to both channels)
- [x] `AudioBuffer::total_samples()` — total sample count (frames * channels)
- [x] Updated README with full API reference, pitch shifting and BPM examples
- [x] 21 new tests total, zero clippy warnings, cargo fmt clean, zero doc warnings

### Eighth pass (agent-5)
- [x] Extracted `init_wav_buffer()` helper in wav.rs — deduplicates header setup from `write_wav_16bit()`, `write_wav_24bit()`, `write_wav_float()` (net -26 lines)
- [x] Replaced verbose `iter_mut().for_each(|x| *x = 0.0)` with idiomatic `fill(0.0)` in PhaseVocoder::process() (4 occurrences)
- [x] Replaced `.cloned()` with `.copied()` for f32 iteration in normalize_output()
- [x] Removed dead guard `if mirror < self.fft_size` in PhaseVocoder::reconstruct_spectrum() (always true since mirror = fft_size - bin, bin >= 1)
- [x] Removed dead guard `if mirror < spectrum.len()` in frequency.rs split_spectrum_into_bands() (always true when spectrum.len() == fft_size)
- [x] All 126 unit tests passing, zero clippy warnings

### Ninth pass (agent-5)
- [x] Extracted `DEFAULT_SAMPLE_RATE`, `DEFAULT_FFT_SIZE`, `DEFAULT_HOP_SIZE`, `DEFAULT_TRANSIENT_SENSITIVITY`, `DEFAULT_SUB_BASS_CUTOFF` constants in types.rs — eliminates 5 magic numbers in `StretchParams::new()`
- [x] Extracted `PresetConfig` struct and `EdmPreset::config()` method — replaces 5-arm match with data-driven dispatch in `with_preset()` (net -20 lines)
- [x] Extracted `LATENCY_FFT_MULTIPLIER` constant in processor.rs — replaces 3 occurrences of magic `* 2`
- [x] Extracted `PEAKS_CAPACITY_DIVISOR` constant in phase_vocoder.rs — documents why peaks buffer is allocated at 1/4 of bins
- [x] Extracted `WAV_HEADER_WRITE_SIZE` and `WAV_RIFF_OVERHEAD` constants in wav.rs — replaces magic `44` and `36` in header writing
- [x] Used `WAV_FMT_MIN_SIZE` for fmt chunk size write (was inline `16u32`)
- [x] Moved `PCM_24BIT_MAX_OUT` to top of wav.rs with other PCM constants (better organization)
- [x] All 291 tests passing (133 unit + 144 integration + 14 doc), zero clippy warnings

## CLI Enhancements (agent-1)
- [x] `--from-bpm <f> --to-bpm <f>` for DJ BPM matching (auto-selects DjBeatmatch preset)
- [x] `--pitch <f>` for pitch shifting (preserves audio duration)
- [x] `--24bit` and `--float` output format flags (default remains 16-bit)
- [x] Named flags (`--ratio`, `--preset`) alongside backward-compatible positional syntax
- [x] Usage help with examples
- [x] Tested with ratio, BPM, pitch, 24-bit, and legacy modes

## Agent-2: Sub-Bass Band-Split Processing (Task 17)
- [x] Implemented `separate_sub_bass()` FFT-based band-splitting filter in hybrid.rs
  - Overlap-add with Hann window, 4096 FFT, 75% overlap
  - Correctly handles conjugate-symmetric negative frequency mirroring
- [x] Added `band_split` field to `StretchParams` (default false, enabled by all EDM presets)
- [x] Added `with_band_split()` builder method
- [x] Added `process_band_split()` to `HybridStretcher`:
  - Separates sub-bass below cutoff (default 120 Hz) via FFT filtering
  - Sub-bass processed exclusively through PV with rigid phase locking
  - Remainder processed through normal hybrid (WSOLA for transients, PV for tonal)
  - Results summed (longer output zero-padded to match)
  - Prevents WSOLA from smearing sub-bass during kick drum transients
- [x] 10 new unit tests: energy preservation, high-freq passthrough, reconstruction,
      band-split stretch output, split vs no-split length, compression, all presets,
      short input fallback, zero cutoff, flag defaults
- [x] Reduced hot-path allocations: avoid `to_vec()` in merge_onsets_and_beats,
      avoid `clone()` for transient onsets when beat-aware is disabled
- [x] 15 integration tests in tests/band_split.rs: EDM signal, sub-bass energy,
      high-freq passthrough, DJ beatmatch, halftime, compression, stereo, 48kHz,
      ambient extreme stretch, vocal chop, custom cutoff, pitch shift, builder toggling
- [x] Total: 355 tests, all passing
- [x] Zero clippy warnings

## AudioBuffer Utilities, Trait Impls & CLI Auto-BPM (agent-1)
- [x] `AudioBuffer::slice(start_frame, num_frames)` — extract sub-region by frame range
- [x] `AudioBuffer::concatenate(buffers)` — join multiple buffers (validates sample rate/channels match)
- [x] `AudioBuffer::normalize(target_peak)` — scale peak to target amplitude
- [x] `AudioBuffer::apply_gain(gain_db)` — apply dB gain (positive amplifies, negative attenuates)
- [x] `AudioBuffer::trim_silence(threshold)` — remove leading/trailing silence (works for mono and stereo)
- [x] `impl Display for AudioBuffer` — e.g. `AudioBuffer(44100 frames, 44100Hz, Mono, 1.000s)`
- [x] `impl Display for StretchParams` — e.g. `StretchParams(ratio=1.5000, 48000Hz, Stereo, preset=DjBeatmatch, fft=4096, hop=1024)`
- [x] `impl Display for EdmPreset` — variant name as string
- [x] `impl Default for StretchParams` — ratio 1.0, stereo, 44100 Hz
- [x] CLI `--auto-bpm` flag — auto-detect source BPM from input WAV, use with `--to-bpm`
- [x] Fixed clippy warnings in edm_presets.rs and identity.rs (from other agents' code)
- [x] Applied cargo fmt across all modified files
- [x] 24 new unit tests (slice, concatenate, normalize, apply_gain, trim_silence, Display, Default)
- [x] All 316+ tests passing, zero clippy warnings, zero doc warnings

### Tenth pass (agent-5)
- [x] Extracted `encode_wav_samples()` helper in wav.rs — deduplicates init+loop+return pattern from `write_wav_16bit()`, `write_wav_24bit()`, `write_wav_float()` (each caller now a 3-line closure delegation)
- [x] Extracted `assign` closure in frequency.rs `split_spectrum_into_bands()` — deduplicates the identical classify-and-assign pattern from positive and negative frequency loops
- [x] Fixed clippy warnings in test files: `needless_range_loop` in edm_presets.rs, `collapsible_if` in identity.rs
- [x] All 302+ tests passing, zero clippy warnings

## Ergonomic API Traits & CLI Verbose (agent-1)
- [x] `impl PartialEq for AudioBuffer` — compare data, sample rate, and channels
- [x] `impl AsRef<[f32]> for AudioBuffer` — direct access to underlying sample slice
- [x] `AudioBuffer::frames()` → `FrameIter` — iterate over frames (1 sample for mono, 2 for stereo)
- [x] `FrameIter` implements `Iterator + ExactSizeIterator`
- [x] `FrameIter` re-exported from `timestretch::FrameIter`
- [x] CLI `--verbose`/`-v` flag — shows full parameter dump, processing time, realtime factor
- [x] Applied cargo fmt to hybrid.rs (other agents' code)
- [x] 9 new unit tests (PartialEq, AsRef, frames iterator, exact size)
- [x] All 335+ tests passing, zero clippy warnings, zero doc warnings

## Audio Analysis, Fades & IntoIterator (agent-1)
- [x] `impl IntoIterator for &AudioBuffer` — enables `for frame in &buf` syntax
- [x] `AudioBuffer::peak()` — peak absolute amplitude
- [x] `AudioBuffer::rms()` — root mean square amplitude (f64 precision)
- [x] `AudioBuffer::fade_in(duration_frames)` — linear fade-in
- [x] `AudioBuffer::fade_out(duration_frames)` — linear fade-out
- [x] Both fades work correctly with mono and stereo, handle edge cases
- [x] Applied cargo fmt to frequency.rs (other agents' code)
- [x] 14 new unit tests (IntoIterator, peak, RMS, fade in/out, stereo fades)
- [x] All 190+ unit tests passing, zero clippy/doc warnings

## API Polish & Documentation Examples (agent-1)
- [x] `Channels::from_count(n)` — construct `Channels` from numeric count (1=Mono, 2=Stereo)
- [x] Added runnable doc examples to: `slice`, `concatenate`, `peak`, `rms`, `fade_in`, `fade_out`, `from_count`
- [x] 22 doc tests now passing (up from 14)
- [x] 191 unit tests passing, zero clippy/doc warnings

## TODO
- [ ] SIMD-friendly inner loop layout

## Notes
- Hann window used for all PV processing (works well for EDM kicks)
- Phase vocoder window-sum normalization clamped to prevent amplification in low-overlap regions
- For very short segments, hybrid falls back to linear resampling
- WSOLA cross-correlation uses FFT for search ranges > 64 candidates, direct computation otherwise
- WSOLA output position tracked fractionally to prevent cumulative rounding drift at all stretch ratios
- wrap_phase uses floor-based modulo instead of while loops (more predictable for large phase values)
- Sub-bass bins (< 120 Hz by default) use rigid phase propagation to prevent phase cancellation — critical for EDM mono-bass compatibility
- StreamProcessor ratio changes are now phase-continuous (no vocoder recreation), enabling smooth DJ pitch fader behavior
- Sub-bass band splitting (when enabled via `band_split`) separates sub-bass before stretch processing, preventing WSOLA from smearing bass during kick transients
- Band splitting uses Hann-window overlap-add FFT filter (4096-point, 75% overlap) with correct conjugate-symmetric negative frequency handling

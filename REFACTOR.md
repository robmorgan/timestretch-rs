# Timestretch Refactor Plan

## Goal
Improve DJ beatmatching audio quality and timing correctness for both offline and realtime use, with explicit quality gates and measurable regression protection.

## Scope
- Primary use case: realtime beatmatching in a DJ engine.
- Secondary use case: offline pre-analysis and higher-quality render path.
- Non-goal for this phase: adding new creative effects before timing/coherence are fixed.

## Milestones

### M0: Baseline + Measurement Integrity
Status: `completed`

Deliverables:
- Fix benchmark manifest/path inconsistencies so reference comparison always runs.
- Freeze a reproducible benchmark corpus (same tracks, same reference renders).
- Add a single command that outputs timing, spectral, transient, loudness, and length metrics.

Acceptance criteria:
- `reference_quality_benchmark` processes real files (no silent skip).
- Baseline report archived for comparison against each milestone.

Resolved in M0:
- Fixed benchmark manifest path consistency:
  - Track/reference paths are now validated as relative to `benchmarks/audio/`.
  - `audio/...` prefix mistakes are detected and fail in strict mode.
- Froze a reproducible corpus via manifest checksums:
  - Added `original_sha256` and `file_sha256` fields in `benchmarks/manifest.toml`.
  - Benchmark validates SHA-256 values in strict mode.
- Added a single baseline command that outputs full metrics and archives results:
  - `./benchmarks/run_m0_baseline.sh`
  - Enables strict mode (`TIMESTRETCH_STRICT_REFERENCE_BENCHMARK=1`)
  - Uses fixed benchmark window (`TIMESTRETCH_REFERENCE_MAX_SECONDS=30`) for reproducible runtime
  - Archives to:
    - `benchmarks/baselines/m0_baseline_latest.json`
    - `benchmarks/baselines/m0_baseline_<timestamp>.json`

Acceptance check:
- `reference_quality_benchmark` processes real files with strict no-skip validation.
- Baseline report is archived for milestone comparison.

---

### M1: Stateful Streaming Phase Vocoder Core
Status: `completed`

Problem addressed:
- Realtime PV quality is limited by per-call state handling and chunk-boundary overlap behavior.

Deliverables:
- Add stateful streaming PV API that preserves phase across calls.
- Carry synthesis overlap/window tails across calls and flush tail at stream end.
- Update stream processor to consume hop-sized analysis progress (`N * hop`) instead of consuming full FFT windows per call.

Acceptance criteria:
- Streaming tests pass for mono/stereo, chunk-size variation, and ratio changes.
- No regression in existing unit/integration tests for stream processor and PV internals.

Files (initial pass):
- `src/stretch/phase_vocoder.rs`
- `src/stream/processor.rs`
- `tests/streaming_edge_cases.rs`

Completed validation for prior remaining work:
- Tail/flush continuity validated under rapid ratio and tempo automation:
  - `test_streaming_flush_continuity_under_rapid_ratio_automation`
  - `test_streaming_flush_continuity_under_rapid_tempo_automation`

---

### M2: Timeline/Length Correctness (Tempo Fidelity)
Status: `pending`

Problem addressed:
- Output duration drift and per-segment crossfade subtraction produce tempo mismatch.

Deliverables:
- Introduce explicit timeline bookkeeping so expected output length is exact (or bounded by <= 1 frame error).
- Rework segment concatenation to preserve global duration exactly.
- Add invariant checks: cumulative synthesis hops + boundary handling == target duration.

Acceptance criteria:
- Duration error <= 0.1% on long-form material.
- No multi-second drift against target tempo in benchmark tracks.

---

### M3: Persistent Hybrid Streaming (Not Re-instantiated Per Call)
Status: `pending`

Problem addressed:
- Hybrid path recreates stretcher state each call, breaking continuity.

Deliverables:
- Persistent hybrid streaming state with rolling analysis buffers.
- Unified transient map and segment policy in stream mode.
- Consistent output path between offline and streaming modes.

Acceptance criteria:
- Streaming hybrid timing metrics materially improve vs current state.
- Reduced chunk-boundary artifacts in transient-rich material.

---

### M4: Beat/Onset Alignment Pipeline for DJ Use
Status: `pending`

Problem addressed:
- BPM/subdivision snapping lacks robust phase/downbeat anchoring and can suppress valid onsets.

Deliverables:
- Optional offline pre-analysis artifact (BPM, phase/downbeat, confidence, transient map).
- Runtime uses pre-analysis when available; live fallback when unavailable.
- Safer snapping policy with confidence/tolerance controls.

Acceptance criteria:
- Beat-aligned segmentation improves transient timing metrics without increased false suppression.
- Stable behavior on tracks outside strict 100-160 BPM assumptions.

---

### M5: Stereo Coherence Hardening
Status: `pending`

Problem addressed:
- Independent M/S processing can desync lengths/phase behavior.

Deliverables:
- Shared segmentation/timing map for stereo pair paths.
- Deterministic channel length agreement before decode.
- Add stereo image/coherence tests (mid/side energy consistency, inter-channel phase drift bounds).

Acceptance criteria:
- No channel-length mismatch in stereo processing.
- Improved stereo coherence metrics on benchmark material.

---

### M6: Quality Gates and Release Criteria
Status: `pending`

Deliverables:
- Tighten tests from coarse sanity checks to DJ-grade thresholds.
- Introduce pass/fail gates for:
  - duration error
  - transient alignment
  - cross-correlation windowed timing
  - loudness deviation
  - spectral similarity by band
- Add CI target for benchmark subset.

Acceptance criteria:
- Regressions fail CI.
- Measurable improvement vs M0 baseline across primary tracks.

## Execution Order
1. M0 baseline integrity
2. M1 stateful PV streaming core
3. M2 timeline/length correctness
4. M3 persistent hybrid streaming
5. M4 beat/onset alignment
6. M5 stereo hardening
7. M6 quality gates

## Current Progress (Updated)
Current focus:
- Next milestone to execute: `M2: Timeline/Length Correctness (Tempo Fidelity)`.

Completed for M0:
- Added strict benchmark validation mode in `tests/reference_quality.rs`:
  - Missing manifest/files/references/checksums fail in strict mode (no silent skip).
  - Path safety/consistency checks enforce audio-base-relative manifest paths.
- Added corpus lock checksums in `benchmarks/manifest.toml`.
- Added baseline runner script `benchmarks/run_m0_baseline.sh`.
- Added baseline archive docs in `benchmarks/baselines/README.md`.
- Archived baseline from strict run:
  - `benchmarks/baselines/m0_baseline_latest.json`
  - `benchmarks/baselines/m0_baseline_20260226T021840Z.json`
- Latest baseline snapshot (strict, 30s window):
  - `tracks_tested`: 1
  - `references_tested`: 1
  - `skipped`: 0
  - `average_spectral_similarity`: 0.6276
  - Best preset on current corpus track: `Ambient` (spectral similarity: 0.6422)

Completed for M1:
- Added stateful streaming PV methods:
  - `PhaseVocoder::process_streaming()`
  - `PhaseVocoder::flush_streaming()`
- Added carry-over synthesis tail buffers in PV for chunk-boundary continuity.
- Refactored PV core processing into shared internal pass (`process_core`) used by batch and streaming paths.
- Updated `StreamProcessor` PV path to call `process_streaming()`.
- Changed streaming input drain policy to hop-based consumption (`frames * hop`) to keep required analysis context.
- Added stream-tail flush integration in `StreamProcessor::flush()` and `flush_into()`.
- Added streaming chunk-boundary continuity regression test (join-jump vs local p95 adjacent-diff threshold).
- Added rapid automation flush continuity coverage (ratio + tempo automation).

Validation run:
- `./benchmarks/run_m0_baseline.sh` (pass; strict mode, no skips, baseline archived)
- `cargo test -q --test streaming` (pass)
- `cargo test -q --test streaming_edge_cases` (pass)
- `cargo test -q --test hybrid_streaming` (pass)
- `cargo test -q test_set_stretch_ratio_preserves_phase_state` (pass)
- `cargo test -q test_streaming_flush_continuity_under_rapid_ratio_automation` (pass)
- `cargo test -q test_streaming_flush_continuity_under_rapid_tempo_automation` (pass)

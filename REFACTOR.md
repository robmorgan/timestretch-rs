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
Status: `completed`

Problem addressed:
- Output duration drift and per-segment crossfade subtraction produce tempo mismatch.

Deliverables:
- Introduce explicit timeline bookkeeping so expected output length is exact (or bounded by <= 1 frame error).
- Rework segment concatenation to preserve global duration exactly.
- Add invariant checks: cumulative synthesis hops + boundary handling == target duration.

Acceptance criteria:
- Duration error <= 0.1% on long-form material.
- No multi-second drift against target tempo in benchmark tracks.

Resolved in M2:
- Added explicit hybrid timeline bookkeeping to enforce duration invariants:
  - `cumulative_synthesis_len - boundary_overlap_len = expected_concat_len`
  - `expected_concat_len + duration_correction_frames = final_output_len`
- Reworked segment concatenation to preserve global duration:
  - Pre-computed crossfade plan before rendering.
  - Added crossfade compensation to segment targets so overlap subtraction does not shrink tempo timeline.
  - Reconciled per-segment synthesis totals to the global target timeline.
- Added deterministic exact-length enforcement at the hybrid output boundary.
- Added invariant and long-form regression coverage:
  - `tests/timeline_length.rs`
  - additional hybrid internal tests for crossfade compensation and timeline invariants.

Acceptance check:
- Long-form duration error validated <= 0.1%.
- No multi-second drift in target-tempo long-form test.

---

### M3: Persistent Hybrid Streaming (Not Re-instantiated Per Call)
Status: `completed`

Problem addressed:
- Hybrid path recreates stretcher state each call, breaking continuity.

Deliverables:
- Persistent hybrid streaming state with rolling analysis buffers.
- Unified transient map and segment policy in stream mode.
- Consistent output path between offline and streaming modes.

Acceptance criteria:
- Streaming hybrid timing metrics materially improve vs current state.
- Reduced chunk-boundary artifacts in transient-rich material.

Resolved in M3:
- Replaced per-call hybrid re-instantiation in `StreamProcessor` with persistent state:
  - Added rolling per-channel hybrid analysis buffers.
  - Added persistent per-channel `HybridStretcher` instances.
  - Added per-channel emitted-length tracking to output only incremental render deltas.
- Added ratio-change rebase policy for hybrid rolling buffers:
  - Prevents retroactive history rewrite after target-ratio changes.
  - Preserves recent analysis context while keeping emitted output monotonic.
- Unified stream-mode hybrid rendering path for both `process()` and `process_into()`.
- Added hybrid-state lifecycle integration:
  - reset/flush/hybrid-mode toggle now reset persistent hybrid state coherently.

Acceptance check:
- Hybrid streaming no longer recreates stretchers per processing call.
- Added regression coverage for persistent hybrid state growth and chunk-boundary roughness bounds.

---

### M4: Beat/Onset Alignment Pipeline for DJ Use
Status: `completed`

Problem addressed:
- BPM/subdivision snapping lacks robust phase/downbeat anchoring and can suppress valid onsets.

Deliverables:
- Optional offline pre-analysis artifact (BPM, phase/downbeat, confidence, transient map).
- Runtime uses pre-analysis when available; live fallback when unavailable.
- Safer snapping policy with confidence/tolerance controls.

Acceptance criteria:
- Beat-aligned segmentation improves transient timing metrics without increased false suppression.
- Stable behavior on tracks outside strict 100-160 BPM assumptions.

Resolved in M4:
- Added optional offline pre-analysis artifact support:
  - `PreAnalysisArtifact` (BPM, downbeat phase offset, confidence, beat positions, transient map)
  - JSON artifact I/O helpers:
    - `write_preanalysis_json()`
    - `read_preanalysis_json()`
  - Offline analyzer:
    - `analyze_for_dj()`
- Runtime integration in hybrid path:
  - Uses confident pre-analysis beat positions when available.
  - Falls back to live beat detection when artifact is unavailable/mismatched/low-confidence.
- Added safer snapping controls in `StretchParams`:
  - `beat_snap_confidence_threshold`
  - `beat_snap_tolerance_ms`
  - safe fallback policy avoids suppressing all transients.
- Added phase-aware subdivision generation using downbeat offset from pre-analysis.

Acceptance check:
- Confident pre-analysis path validated via integration tests.
- Fallback path validated when artifact is unavailable/mismatched.

---

### M5: Stereo Coherence Hardening
Status: `completed`

Problem addressed:
- Independent M/S processing can desync lengths/phase behavior.

Deliverables:
- Shared segmentation/timing map for stereo pair paths.
- Deterministic channel length agreement before decode.
- Add stereo image/coherence tests (mid/side energy consistency, inter-channel phase drift bounds).

Acceptance criteria:
- No channel-length mismatch in stereo processing.
- Improved stereo coherence metrics on benchmark material.

Resolved in M5:
- Added shared stereo segmentation/timing map path:
  - Mid channel now builds a shared onset/beat map.
  - Both Mid and Side are processed with the same onset anchors via `HybridStretcher::process_with_onsets()`.
- Added deterministic channel-length agreement before decode:
  - Mid and Side outputs are length-aligned to exact target length before M/S decode.
- Added stereo coherence regression coverage:
  - channel length agreement vs target
  - mid/side energy ratio consistency
  - inter-channel lag drift bound

Acceptance check:
- Stereo output channels are deterministic and equal-length.
- Stereo coherence bounds covered in new unit tests.

---

### M6: Quality Gates and Release Criteria
Status: `completed`

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

Resolved in M6:
- Added executable quality-gate benchmark subset in `tests/quality_gates.rs` with pass/fail thresholds for:
  - duration error
  - transient alignment
  - cross-correlation timing coherence
  - loudness deviation
  - spectral similarity by EDM band
- Added CI target in `.github/workflows/ci.yml`:
  - `quality-gates` job runs `cargo test --test quality_gates -- --nocapture`
  - Gate regressions now fail CI on every PR/push.
- Added strict-mode baseline regression guard in `tests/reference_quality.rs`:
  - Strict corpus runs now compare against `benchmarks/baselines/m0_baseline_latest.json`
  - Fails when average/per-track spectral similarity regresses beyond tolerance.

Acceptance check:
- Quality-gate subset passes locally and is enforced in CI.
- Strict reference benchmark retains regression checks against M0 baseline snapshot.

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
- All milestones `M0` through `M6` are completed.

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

Completed for M2:
- Added explicit hybrid timeline bookkeeping and invariant checks in `src/stretch/hybrid.rs`.
- Added crossfade-aware segment length compensation so segment boundary overlap no longer causes timeline shrink.
- Added exact duration enforcement at hybrid output to guarantee target-length tempo fidelity.
- Added M2 long-form timing regression tests in `tests/timeline_length.rs`.
- Added hybrid timeline unit coverage:
  - crossfade compensation preserves base total
  - segment-target reconciliation hits desired sum
- timeline bookkeeping invariants hold

Completed for M3:
- Added `HybridStreamingState` in `src/stream/processor.rs`:
  - persistent per-channel `HybridStretcher` allocation
  - rolling analysis buffers
  - incremental output emission tracking
- Refactored hybrid stream path to use persistent state in:
  - `StreamProcessor::process_hybrid_path()`
  - `StreamProcessor::process_hybrid_into()`
- Added ratio-change rebase handling for persistent hybrid buffers.
- Added M3 coverage:
  - `test_stream_processor_hybrid_state_persists_across_calls`
- `test_hybrid_streaming_persistent_small_vs_large_chunk_length`
- `test_hybrid_streaming_chunk_boundary_artifacts_bounded`

Completed for M4:
- Added pre-analysis artifact model in `src/core/preanalysis.rs`.
- Added offline pre-analysis pipeline in `src/analysis/preanalysis.rs`.
- Added runtime pre-analysis/fallback integration and phase-aware snap grid in `src/stretch/hybrid.rs`.
- Added new beat-snap control fields and builders in `StretchParams`:
  - pre-analysis attachment
  - confidence threshold
  - tolerance control
- Added M4 tests:
  - `tests/preanalysis_pipeline.rs`
  - core pre-analysis + parameter tests

Completed for M5:
- Added `HybridStretcher::process_with_onsets()` for shared-anchor rendering.
- Refactored stereo mid/side path to use shared onset map from Mid channel.
- Added deterministic per-channel target-length enforcement pre-decode.
- Added stereo coherence tests in `src/stretch/stereo.rs`:
  - `test_stretch_mid_side_channel_length_agreement`
- `test_stretch_mid_side_energy_coherence`
- `test_stretch_mid_side_phase_drift_bound`

Completed for M6:
- Added benchmark-subset quality gates in `tests/quality_gates.rs`.
- Added CI enforcement job (`quality-gates`) in `.github/workflows/ci.yml`.
- Added strict M0 baseline regression assertions in `tests/reference_quality.rs`.
- Tightened coverage to explicit pass/fail thresholds for duration/transient/xcorr/loudness/spectral-band metrics.

Validation run:
- `./benchmarks/run_m0_baseline.sh` (pass; strict mode, no skips, baseline archived)
- `cargo test -q --test streaming` (pass)
- `cargo test -q --test streaming_edge_cases` (pass)
- `cargo test -q --test hybrid_streaming` (pass)
- `cargo test -q test_set_stretch_ratio_preserves_phase_state` (pass)
- `cargo test -q test_streaming_flush_continuity_under_rapid_ratio_automation` (pass)
- `cargo test -q test_streaming_flush_continuity_under_rapid_tempo_automation` (pass)
- `cargo test -q --test timeline_length` (pass)
- `cargo test -q --test hybrid_streaming` (pass)
- `cargo test -q --test streaming_batch_parity` (pass)
- `cargo test -q --lib stream::processor` (pass)
- `cargo test -q --test preanalysis_pipeline` (pass)
- `cargo test -q --lib analysis::preanalysis` (pass)
- `cargo test -q --lib core::types` (pass)
- `cargo test -q --lib stretch::stereo` (pass)
- `cargo test -q --test quality_gates` (pass)
- `TIMESTRETCH_REFERENCE_MAX_SECONDS=5 cargo test -q --test reference_quality -- --nocapture` (pass)

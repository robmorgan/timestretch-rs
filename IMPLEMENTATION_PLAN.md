# Implementation Plan

Date: 2026-02-28
Last Updated: 2026-03-05

## Goals
- Reach professional-grade hybrid time-stretch quality for EDM/house and DJ beatmatching.
- Keep stream path real-time safe (bounded, allocation-free after warmup).
- Preserve API simplicity while allowing advanced internal routing.

## Current Baseline (Completed in This Overhaul)
- [x] Integrate research, architecture, and implementation-plan artifacts into this repository.
- [x] Preserve and validate existing hybrid streaming implementation in this repo.
- [x] Add objective quality benchmark harness with WAV + spectrogram CSV output.
- [x] Add conservative transient lookahead confirmation in detector.
- [x] Expose transient lookahead confirmation controls in `StretchParams`.

## Phase A - Foundation

### A1. Transient Detector
- [x] Multi-feature detector (spectral flux + energy + phase deviation).
- [x] Adaptive thresholding and onset strength scoring.
- [x] Add lookahead confirmation to suppress single-frame spikes.
- [x] Expose detector lookahead/threshold policy as public config knobs.

Acceptance:
- Stable onset detection on click-train/noise-burst tests.
- No regression in current transient test suite.

### A2. Steady-State Path (Phase Vocoder)
- [x] Phase vocoder with instantaneous-frequency propagation.
- [x] Identity/ROI phase locking implementation.
- [x] Streaming stateful processing API.
- [x] Add explicit confidence-driven locking profile selection per frame.

Acceptance:
- Harmonic material preserves pitch center and reduced phasiness relative to PV baseline.

### A3. Transient Path (WSOLA)
- [x] WSOLA with normalized correlation search.
- [x] FFT-accelerated correlation path for wider searches.
- [x] Add transient-class-aware WSOLA parameter adaptation (kick/snare/hat classes).

Acceptance:
- Transient timing MAE stays bounded in benchmark suite.

### A4. Hybrid Routing + Recombination
- [x] Segment routing between transient and tonal paths.
- [x] Crossfade and timeline compensation.
- [x] Persistent streaming hybrid mode.
- [x] Fix chunk-size parity regression in hybrid streaming.
- [x] Add optional explicit residual/noise branch in default runtime path.

Acceptance:
- Streaming-vs-batch length delta < 5% across tested chunk sizes.
- No NaN/Inf across stress tests.

## Phase B - Quality

### B1. Adaptive Locking and Decomposition
- [x] Adaptive phase locking mode (identity vs ROI vs selective) driven by local SNR/peak confidence.
- [x] Promote HPSS-guided routing as first-class in high-quality mode.
- [x] Improve transition handling when detector confidence is low.

### B2. Transition & Boundary Quality
- [x] Refine boundary crossfades with content-adaptive curves.
- [x] Add explicit transient-center anchoring for attack alignment.
- [x] Introduce boundary artifact detector for automated regression tests.

### B3. Pitch-Shifting Quality
- [x] Extend formant preservation controls for vocal content.
- [x] Add envelope-preservation presets and validation scenarios.

Acceptance:
- Better objective metrics than baseline on benchmark suite for 0.75x/1.5x/2.0x.
- Improved subjective results on EDM kick/hat clarity and vocal integrity.

## Phase C - Performance & Production Hardening

### C1. Real-Time Safety
- [x] Verify no allocations in steady callback path via allocation-tracing tests.
- [x] Add CI gate for worst-case callback processing budget.
- [x] Add explicit max-iteration bounds where dynamic loops exist.

### C2. SIMD / Hot-Loop Optimization
- [x] Optimize overlap-add/window kernels (scalar + SIMD backends).
- [x] Optimize WSOLA correlation hot path for AVX2/NEON with scalar fallback.
- [x] Evaluate FFT planning reuse and cache friendliness.

### C3. Benchmark & QA Infrastructure
- [x] Create quality benchmark harness (`tests/quality_benchmark.rs`).
- [x] Add comparative benchmarking target against Rubber Band outputs (non-code-copy black-box reference).
- [x] Add longitudinal quality dashboard artifacts in CI.

Acceptance:
- Faster-than-real-time processing at target settings on dev hardware.
- Stable quality metrics and no major regressions across releases.

## Benchmark Snapshot (2026-02-28)
Run command:
- `cargo test --features qa-harnesses --test quality_benchmark -- --ignored --nocapture`

Generated artifacts:
- `target/quality_benchmark/quality_report.csv`
- `target/quality_benchmark/pitch_formant_report.csv`
- `target/quality_benchmark/pitch_formant_delta.csv`

Time-stretch deltas (overhauled hybrid minus baseline PV, lower is better):
- `transient_mae_ms_mean_delta = -1.378055` (`n=15`)
- `spectral_distortion_mean_delta = -0.287285` (`n=5`, tone-stack cases only)
- `phase_coherence_std_mean_delta = +0.001410` (`n=5`, tone-stack cases only)
- `unexpected_energy_ratio_mean_delta = -0.086521` (`n=15`)

Pitch/formant deltas (vocal preset minus off preset):
- `formant_profile_similarity_mean_delta = +0.120029` (`n=4`, higher is better)
- `spectral_distortion_mean_delta = -0.112904` (`n=4`, lower is better)
- `unexpected_energy_ratio_mean_delta = +0.000002` (`n=4`, lower is better)

Notes:
- `tone_stack` transient MAE at the previously problematic ratios now tracks baseline closely:
  - `0.50x`: `5.493 ms` (baseline `5.867 ms`)
  - `4.00x`: `7.046 ms` (baseline `6.797 ms`)
- Vocal mode high-band leakage was tightened in upward shifts (`1.35x` delta reduced to `+0.000004` from `+0.000007`), while preserving formant gains.

## Prioritized Next Tasks (Immediate)
- [x] Run `tests/quality_benchmark.rs` with vocal/formant scenarios and record baseline deltas.
- [x] Add concise API docs/examples for new envelope controls (`EnvelopePreset`, strength, adaptive order).

## Next Tasks
- [x] Reduce remaining transient timing deltas on `tone_stack` cases (especially 0.50x and 4.00x) in `quality_report.csv`.
- [x] Tighten high-band leakage in vocal envelope mode (`delta_unexpected_energy_vocal_minus_off`).

## Phase D - Streaming Runtime Cleanup (Open)

### D1. Consolidate on the Plain Streaming Runtime
- [x] Remove the dual-plane deterministic runtime and its public API.
- [x] Keep `StreamProcessor` as the single supported realtime path.
- [ ] Revisit any remaining historical planning text that assumes dual-plane still exists.

Acceptance:
- `StreamProcessor` is the only supported realtime runtime surface.
- No public docs or tests require dual-plane-specific configuration.

### D2. Complete Callback/Control Decoupling
- [x] Replace callback-time `Mutex<Receiver<...>>` snapshot polling with lock-free snapshot handoff (RCU-style latest-value read).
- [x] Keep control-plane publish semantics non-blocking and drop-oldest-safe under bursty control updates.
- [x] Document RT-thread invariants and enforce them in tests.

Acceptance:
- RT callback path contains no locks and no blocking synchronization primitives.
- Control snapshot updates remain wait-free for RT reads.

### D3. Close Feature Gaps in Dual-Plane Deterministic Path
- [x] Add deterministic-path support for `pitch_scale != 1.0` with bounded per-callback cost.
- [x] Align tempo + pitch modulation behavior and quality with existing deterministic stream guarantees.
- [x] Add parity tests versus current deterministic stream output under tempo/pitch modulation sweeps.

Acceptance:
- `set_pitch_scale(...)` works when deterministic dual-plane backend is active.
- Modulation quality/callback gates pass in strict mode for deterministic dual-plane path.

### D4. Expand Multi-Lane Engine Beyond 3 Lanes
- [x] Add optional stem-aware lane plumbing (feature-gated/off by default initially).
- [x] Integrate stem/noise confidence into lane weighting policy without callback branching explosions.
- [x] Add deterministic blend/crossfade tests for lane transitions and tier changes.

Acceptance:
- Stem-aware lane can be enabled without RT reallocations after warmup.
- Lane mixing remains deterministic and artifact-gated at modulation boundaries.

### D5. Finish Policy Extraction from Legacy Hybrid Path
- [x] Continue moving adaptive policy logic from `src/stretch/hybrid.rs` into reusable analysis snapshots consumed by dual-plane.
- [x] Minimize duplicated adaptive decisions between legacy hybrid and dual-plane paths.
- [x] Keep `src/stretch/phase_vocoder.rs` as tonal core while policy stays outside callback.

Acceptance:
- Shared adaptive analysis module is the single source for transient/beat/tonal confidence and hint generation.
- Legacy path uses shared policy where practical; no callback-only policy divergence in deterministic mode.

### D6. Latency Profiles as First-Class Runtime Products
- [x] Add context-aware profile switching policy (`Scratch`, `Mix`, `Render`) with smooth transitions.
- [x] Define and enforce profile-specific callback budgets and tier crossfade settings.
- [x] Expose profile state/telemetry in API for host integration.

Acceptance:
- Automatic profile switching can occur without audible zipper artifacts.
- Profile transitions remain within callback-budget gates in CI.

### D7. Production Gates and Rollout Guardrails
- [x] Make dual-plane deterministic quality gates mandatory in CI for default stream path.
- [x] Add explicit CI assertions for long-run drift + fast-mod artifacts on the default deterministic route.
- [x] Add a release checklist for retiring the opt-in deterministic toggle and documenting migration.

Acceptance:
- CI fails if default deterministic path regresses on p99/p999, drift, allocation, or artifact gates.
- Release notes document default-route switch and legacy compatibility behavior.

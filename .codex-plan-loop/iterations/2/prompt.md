You are working in the Git repository at /Users/robbym/go/src/github.com/robmorgan/timestretch-rs on branch research at commit fe4aa1b95eed91c9027b5f3b6edc064c6c33e800.

Execute exactly one small slice from the active roadmap stage only.

Requirements:
- Implement exactly one small slice inside the active stage.
- Do not jump to later stages.
- Run the most relevant local checks for the files you touch.
- Update the active stage status from [ ] to [~] when work begins.
- Update [~] to [x] only if the stage exit criteria are satisfied and the dedicated stage verification passes.
- Do not edit /Users/robbym/go/src/github.com/robmorgan/timestretch-rs/.codex-plan-loop.
- Leave the repository in a state where the outer loop can run its smoke and stage-specific tests.
- End with a concise summary of files changed, checks run, and remaining risk.

This is loop iteration 2.
This is the active roadmap stage: Stage 1 - Stabilize Fast Modulation and Transition Quality

## Goal

Make `timestretch-rs` competitive with production-grade realtime stretchers by
closing the remaining gaps in audible quality, modulation stability, realtime
contract quality, external quality evidence, and API strictness.

## Principles

- Fix audible regressions before adding features.
- Make the RT-safe path the default, obvious path.
- Reject malformed input instead of silently truncating or falling back.
- Prefer reference-driven quality gates over self-comparison alone.
- Preserve the EDM-first focus unless there is a deliberate decision to expand
  into a broader general-purpose stretcher.

## Active Stage
## [~] Stage 1: Stabilize Fast Modulation and Transition Quality

Automation: auto

### Why

This is the clearest current signal that the library is not yet
production-stable. If dynamic ratio changes still produce obvious boundary
artifacts, improvements elsewhere will not matter.

### Primary Files

- `src/dual_plane/rt.rs`
- `src/stream/processor.rs`
- `src/stream/transient_scheduler.rs`
- `src/stretch/phase_vocoder.rs`
- `tests/quality_gates.rs`

### Work

- Fix ratio-transition continuity in the dual-plane deterministic path.
- Reduce profile-switch churn and improve hysteresis for automation-heavy use.
- Tighten transient reset scheduling so fast modulation does not over-trigger
  phase resets.
- Review how phase state is preserved or reseeded during rapid ratio changes.
- Add additional focused tests around repeated short-interval modulation, not
  just the existing one gate.

### Exit Criteria

- `cargo test --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture`
  passes with margin, not barely.
- Release-mode modulation no longer produces obvious clicks, roughness, or
  discontinuities on synthetic DJ-like material.
- Fixes do not regress steady-state deterministic streaming quality.

## Previous Iteration Summary
Files changed: [ROADMAP.md](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/ROADMAP.md#L33) and [tests/quality_gates.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/tests/quality_gates.rs#L762). Stage 1 is now marked `[~]`, and I added one Stage 1 slice only: a reusable short-step ratio runner plus a new release regression gate for repeated short-interval dual-plane modulation at [tests/quality_gates.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/tests/quality_gates.rs#L980).

Checks run:
- `cargo test --release --test quality_gates quality_gate_dual_plane_short_interval_step_modulation_artifacts -- --nocapture`  
  Passed. Metrics: baseline `p95=1.132`, modulated `p95=2.600`, baseline `mean=0.459`, modulated `mean=0.926`.
- `cargo test --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture`  
  Failed. Metrics: baseline `p95=1.132`, modulated `p95=4.936`; baseline `p98=1.182`, modulated `p98=5.468`.

Remaining risk: Stage 1 exit criteria are still not met because the existing dedicated fast-modulation gate is failing in release mode. The repo now has better coverage for short-interval modulation, but the underlying dual-plane modulation continuity issue remains unresolved.
## Last Failure Context
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.07s
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.13s
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.98s
thread 'quality_gate_dual_plane_fast_modulation_artifacts' (175086278) panicked at tests/quality_gates.rs:960:5:
dual-plane modulation artifact gate failed (p95): modulated 4.936 vs baseline 1.132
test quality_gate_dual_plane_fast_modulation_artifacts ... FAILED
failures:
failures:
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 5 filtered out; finished in 0.29s
error: test failed, to rerun pass `--test quality_gates`

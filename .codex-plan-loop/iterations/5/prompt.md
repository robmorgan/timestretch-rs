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

This is loop iteration 5.
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
Implemented one Stage 1 slice in [src/stream/transient_scheduler.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/stream/transient_scheduler.rs#L249): a detected transient no longer burns one cooldown frame immediately on the same analysis frame that triggered the reset. I also added a focused regression at [src/stream/transient_scheduler.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/stream/transient_scheduler.rs#L495) that tail-aligns a click to the final scanned frame and asserts the full configured cooldown is preserved for subsequent frames. `ROADMAP.md` was already at `[~]`, and it remains `[~]` because the stage exit gate is still failing.

Checks run:
- `cargo test scheduler_ -- --nocapture` passed (`8 passed`).
- `cargo test --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture` failed unchanged: baseline `p95=1.132`, modulated `p95=3.613`, `p98=5.433`, `mean=2.269`.

Files changed this turn: [src/stream/transient_scheduler.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/stream/transient_scheduler.rs). Remaining risk: this slice tightens transient reset scheduling, but the larger deterministic dual-plane fast-modulation artifact regression is still present, so Stage 1 is not ready to close.
## Last Failure Context
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.19s
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.89s
thread 'quality_gate_dual_plane_fast_modulation_artifacts' (175215924) panicked at tests/quality_gates.rs:960:5:
dual-plane modulation artifact gate failed (p95): modulated 3.613 vs baseline 1.132
test quality_gate_dual_plane_fast_modulation_artifacts ... FAILED
failures:
failures:
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 5 filtered out; finished in 0.29s
error: test failed, to rerun pass `--test quality_gates`

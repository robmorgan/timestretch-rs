You are working in the Git repository at /Users/robbym/go/src/github.com/robmorgan/timestretch-rs on branch research at commit b48ad28439b523eb5a38d499573a5764b0af2c76.

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
- `qa/quality_gates.rs`

### Work

- Fix ratio-transition continuity in the dual-plane deterministic path.
- Reduce profile-switch churn and improve hysteresis for automation-heavy use.
- Tighten transient reset scheduling so fast modulation does not over-trigger
  phase resets.
- Review how phase state is preserved or reseeded during rapid ratio changes.
- Add additional focused tests around repeated short-interval modulation, not
  just the existing one gate.

### Exit Criteria

- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture`
  passes with margin, not barely.
- Release-mode modulation no longer produces obvious clicks, roughness, or
  discontinuities on synthetic DJ-like material.
- Fixes do not regress steady-state deterministic streaming quality.

## Previous Iteration Summary
Implemented one Stage 1 slice: added a focused release QA gate in [qa/quality_gates.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/qa/quality_gates.rs#L1297) for callback-to-callback ratio toggling. It compares a constant-step baseline (`1.0` every callback) against alternating short-interval steps (`0.965/1.035/0.975/1.025` every callback) and asserts boundary artifact stats. `ROADMAP.md` was left unchanged because Stage 1 was already `[~]`, and `.codex-plan-loop` was not edited.

Checks run:
- `cargo fmt --all --check`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_callback_toggle_modulation_artifacts -- --nocapture`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_short_interval_step_modulation_artifacts -- --nocapture`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture`

Remaining risk: this iteration only strengthens QA coverage; it does not change the modulation path itself. The existing fast-modulation gate still passes narrowly on mean artifact ratio (`1.812` vs limit `1.818`), so Stage 1 should remain `[~]`.
## Last Failure Context
No prior failure context is recorded.

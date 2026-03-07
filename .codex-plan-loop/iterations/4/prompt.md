You are working in the Git repository at /Users/robbym/go/src/github.com/robmorgan/timestretch-rs on branch research at commit a383f1039cbdaf52a2bac102958161da0df2c65b.

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

This is loop iteration 4.
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
Implemented one Stage 1 slice in [src/dual_plane/rt.rs:594](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/dual_plane/rt.rs#L594) and [src/dual_plane/rt.rs:2535](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/dual_plane/rt.rs#L2535): fast ratio-motion freeze now biases auto-mode `Mix` into a temporary `Scratch` hold, and repeated freeze re-arms preserve that scratch hold instead of snapping back to the old mix policy. I also added a regression test covering that exact modulation case. `ROADMAP.md` was left unchanged because Stage 1 was already `[~]`, and I am not marking it `[x]`.

Checks run:
- `cargo test ratio_motion_freeze_ -- --nocapture`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_deterministic_long_run_drift -- --nocapture`
- `cargo fmt --all --check`

Result: the Stage 1 fast-modulation gate improved from the prior modulated `p95=2.695 / p98=4.026 / mean=1.812` to `p95=1.837 / p98=2.780 / mean=1.436`, while the long-run deterministic drift gate still passes. Remaining risk: there is still a very large rare outlier (`p99/max=63.591`), so this slice improves the main modulation margin but does not close Stage 1 by itself.
## Last Failure Context
No prior failure context is recorded.

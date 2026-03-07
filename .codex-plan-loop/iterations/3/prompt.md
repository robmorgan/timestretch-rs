You are working in the Git repository at /Users/robbym/go/src/github.com/robmorgan/timestretch-rs on branch research at commit f50c1063e5b32b9c50b124d51aea75a0422e1195.

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

This is loop iteration 3.
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
Added one Stage 1 slice in [src/dual_plane/rt.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/dual_plane/rt.rs#L2504): a new regression test, `ratio_motion_freeze_rearms_under_callback_rate_modulation`, which verifies callback-rate ratio toggling re-arms the existing ratio-motion freeze and prevents auto-profile flapping until modulation stops. I left `ROADMAP.md` unchanged because Stage 1 was already `[~]`, and the stage exit criteria are still not satisfied. I did not edit `.codex-plan-loop`.

Checks run:
- `cargo test ratio_motion_freeze_ -- --nocapture`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture`
- `cargo fmt --all --check`

Remaining risk: this iteration strengthens Stage 1 regression coverage only; it does not improve the fast-modulation path itself. The dedicated release gate still passes at the prior narrow margin (`modulated p95=2.695`, `mean=1.812`), so Stage 1 should remain `[~]`.
## Last Failure Context
No prior failure context is recorded.

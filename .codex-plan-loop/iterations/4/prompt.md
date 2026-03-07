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
Implemented one Stage 1 slice in [src/stretch/phase_vocoder.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/stretch/phase_vocoder.rs#L135): the streaming PV now tracks a carried `streaming_tail_ratio` so overlap tails that span a ratio change are normalized with the more expansion-prone ratio instead of snapping to the newest ratio at the chunk boundary. I also added a focused regression test at [src/stretch/phase_vocoder.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/stretch/phase_vocoder.rs#L2237). I left the existing unrelated dirty files untouched, and Stage 1 remains `[~]` because the exit criteria are not met.

Checks run:
- `cargo test --release stretch::phase_vocoder::tests::test_streaming_tail_ratio_preserves_overlap_history_across_ratio_change -- --nocapture` passed.
- `cargo test --release stretch::phase_vocoder::tests::test_process_streaming_and_flush_produce_finite_output -- --nocapture` passed.
- `cargo test --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture` failed unchanged: baseline `p95=1.132`; modulated `p95=3.613`, `p98=5.433`, `mean=2.269`, `max/p99=59.994`.
- `cargo test --release --test quality_gates quality_gate_dual_plane_deterministic_long_run_drift -- --nocapture` passed.

Remaining risk: the dedicated Stage 1 fast-modulation gate is still failing, so the next slice likely needs a deeper deterministic dual-plane transition fix beyond PV tail normalization alone.
## Last Failure Context
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.23s
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.90s
thread 'quality_gate_dual_plane_fast_modulation_artifacts' (175193544) panicked at tests/quality_gates.rs:960:5:
dual-plane modulation artifact gate failed (p95): modulated 3.613 vs baseline 1.132
test quality_gate_dual_plane_fast_modulation_artifacts ... FAILED
failures:
failures:
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 5 filtered out; finished in 0.30s
error: test failed, to rerun pass `--test quality_gates`

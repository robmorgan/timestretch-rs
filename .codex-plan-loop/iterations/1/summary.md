Implemented one Stage 1 slice: added a focused release QA gate in [qa/quality_gates.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/qa/quality_gates.rs#L1297) for callback-to-callback ratio toggling. It compares a constant-step baseline (`1.0` every callback) against alternating short-interval steps (`0.965/1.035/0.975/1.025` every callback) and asserts boundary artifact stats. `ROADMAP.md` was left unchanged because Stage 1 was already `[~]`, and `.codex-plan-loop` was not edited.

Checks run:
- `cargo fmt --all --check`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_callback_toggle_modulation_artifacts -- --nocapture`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_short_interval_step_modulation_artifacts -- --nocapture`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture`

Remaining risk: this iteration only strengthens QA coverage; it does not change the modulation path itself. The existing fast-modulation gate still passes narrowly on mean artifact ratio (`1.812` vs limit `1.818`), so Stage 1 should remain `[~]`.
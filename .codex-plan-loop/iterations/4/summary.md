Implemented one Stage 1 slice in [src/dual_plane/rt.rs:594](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/dual_plane/rt.rs#L594) and [src/dual_plane/rt.rs:2568](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/dual_plane/rt.rs#L2568): when ratio-motion freeze biases auto-mode `Mix` into `Scratch`, the RT path now snaps `current_tier`, `target_tier`, and the raw lane weights to the scratch ladder immediately instead of leaving a stale tier crossfade active. I also tightened the existing regression to assert that immediate tier snap. `ROADMAP.md` was left unchanged because Stage 1 was already `[~]`, and I am not marking it `[x]`.

Checks run:
- `cargo fmt --all --check`
- `cargo test ratio_motion_freeze_ -- --nocapture`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_callback_toggle_modulation_artifacts -- --nocapture`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_deterministic_long_run_drift -- --nocapture`

Remaining risk: the Stage 1 fast-modulation gate still shows the same rare outlier (`modulated max/p99 = 63.591`, `p95 = 1.837`, `p98 = 2.780`, `mean = 1.436`), so this slice improves tier-state coherence during the scratch hold but does not close the remaining fast-modulation artifact spike.
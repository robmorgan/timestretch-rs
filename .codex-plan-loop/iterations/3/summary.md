Implemented one Stage 1 slice in [src/dual_plane/rt.rs:594](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/dual_plane/rt.rs#L594) and [src/dual_plane/rt.rs:2535](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/dual_plane/rt.rs#L2535): fast ratio-motion freeze now biases auto-mode `Mix` into a temporary `Scratch` hold, and repeated freeze re-arms preserve that scratch hold instead of snapping back to the old mix policy. I also added a regression test covering that exact modulation case. `ROADMAP.md` was left unchanged because Stage 1 was already `[~]`, and I am not marking it `[x]`.

Checks run:
- `cargo test ratio_motion_freeze_ -- --nocapture`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_deterministic_long_run_drift -- --nocapture`
- `cargo fmt --all --check`

Result: the Stage 1 fast-modulation gate improved from the prior modulated `p95=2.695 / p98=4.026 / mean=1.812` to `p95=1.837 / p98=2.780 / mean=1.436`, while the long-run deterministic drift gate still passes. Remaining risk: there is still a very large rare outlier (`p99/max=63.591`), so this slice improves the main modulation margin but does not close Stage 1 by itself.
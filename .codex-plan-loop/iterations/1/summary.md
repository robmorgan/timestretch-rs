Files changed: [ROADMAP.md](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/ROADMAP.md#L33) and [tests/quality_gates.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/tests/quality_gates.rs#L762). Stage 1 is now marked `[~]`, and I added one Stage 1 slice only: a reusable short-step ratio runner plus a new release regression gate for repeated short-interval dual-plane modulation at [tests/quality_gates.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/tests/quality_gates.rs#L980).

Checks run:
- `cargo test --release --test quality_gates quality_gate_dual_plane_short_interval_step_modulation_artifacts -- --nocapture`  
  Passed. Metrics: baseline `p95=1.132`, modulated `p95=2.600`, baseline `mean=0.459`, modulated `mean=0.926`.
- `cargo test --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture`  
  Failed. Metrics: baseline `p95=1.132`, modulated `p95=4.936`; baseline `p98=1.182`, modulated `p98=5.468`.

Remaining risk: Stage 1 exit criteria are still not met because the existing dedicated fast-modulation gate is failing in release mode. The repo now has better coverage for short-interval modulation, but the underlying dual-plane modulation continuity issue remains unresolved.
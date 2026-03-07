Files changed: [src/stream/processor.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/stream/processor.rs#L24), [src/stream/processor.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/stream/processor.rs#L399), and [src/stream/processor.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/stream/processor.rs#L1571). This Stage 1 slice keeps a rolling chunk-ratio window sized to one dual-plane RT kernel and feeds the backend a lightly slewed average ratio instead of the latest callback ratio alone. I left the existing unrelated dirty files untouched, and Stage 1 remains `[~]`.

Checks run:
- `cargo test --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture`
  Failed, but improved: baseline `p95=1.132`; modulated `p95=3.613` (down from `4.936`), `p98=5.433`, `mean=2.269`.
- `cargo test --release --test quality_gates quality_gate_dual_plane_deterministic_long_run_drift -- --nocapture`
  Passed.

Remaining risk: the dedicated Stage 1 fast-modulation gate still fails, and large outliers remain, so the next slice likely needs a deeper RT/kernel transition continuity fix rather than more adapter-only ratio shaping.
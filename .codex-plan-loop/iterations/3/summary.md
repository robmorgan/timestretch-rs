Implemented one Stage 1 slice in [src/stretch/phase_vocoder.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/stretch/phase_vocoder.rs#L135): the streaming PV now tracks a carried `streaming_tail_ratio` so overlap tails that span a ratio change are normalized with the more expansion-prone ratio instead of snapping to the newest ratio at the chunk boundary. I also added a focused regression test at [src/stretch/phase_vocoder.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/stretch/phase_vocoder.rs#L2237). I left the existing unrelated dirty files untouched, and Stage 1 remains `[~]` because the exit criteria are not met.

Checks run:
- `cargo test --release stretch::phase_vocoder::tests::test_streaming_tail_ratio_preserves_overlap_history_across_ratio_change -- --nocapture` passed.
- `cargo test --release stretch::phase_vocoder::tests::test_process_streaming_and_flush_produce_finite_output -- --nocapture` passed.
- `cargo test --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture` failed unchanged: baseline `p95=1.132`; modulated `p95=3.613`, `p98=5.433`, `mean=2.269`, `max/p99=59.994`.
- `cargo test --release --test quality_gates quality_gate_dual_plane_deterministic_long_run_drift -- --nocapture` passed.

Remaining risk: the dedicated Stage 1 fast-modulation gate is still failing, so the next slice likely needs a deeper deterministic dual-plane transition fix beyond PV tail normalization alone.
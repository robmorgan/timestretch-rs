Added one Stage 1 slice in [src/dual_plane/rt.rs](/Users/robbym/go/src/github.com/robmorgan/timestretch-rs/src/dual_plane/rt.rs#L2504): a new regression test, `ratio_motion_freeze_rearms_under_callback_rate_modulation`, which verifies callback-rate ratio toggling re-arms the existing ratio-motion freeze and prevents auto-profile flapping until modulation stops. I left `ROADMAP.md` unchanged because Stage 1 was already `[~]`, and the stage exit criteria are still not satisfied. I did not edit `.codex-plan-loop`.

Checks run:
- `cargo test ratio_motion_freeze_ -- --nocapture`
- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture`
- `cargo fmt --all --check`

Remaining risk: this iteration strengthens Stage 1 regression coverage only; it does not improve the fast-modulation path itself. The dedicated release gate still passes at the prior narrow margin (`modulated p95=2.695`, `mean=1.812`), so Stage 1 should remain `[~]`.
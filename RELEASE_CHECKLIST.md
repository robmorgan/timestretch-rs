# Release Checklist

## Dual-Plane Deterministic Default

Use this checklist when cutting a release that verifies the
deterministic dual-plane backend is working correctly.

- [ ] Verify CI quality gates are green for the default deterministic route:
  - `cargo test --test realtime_allocations -- --nocapture`
  - `TIMESTRETCH_STRICT_CALLBACK_BUDGET=1 cargo test --features qa-harnesses --release --test quality_gates -- --nocapture`
  - `cargo test --test dual_plane_parity -- --nocapture`
- [ ] Confirm default routing behavior in tests:
  - New `StreamProcessor` starts on deterministic dual-plane backend.
- [ ] Confirm migration notes are present in release notes:
  - Deterministic stream path uses dual-plane backend by default.
  - `set_dual_plane_deterministic(...)` is compatibility-only control and not required for default behavior.
- [ ] Re-run API docs/examples to ensure they reflect default deterministic behavior.
- [ ] Record benchmark deltas versus previous release baseline:
  - callback p99/p999 latency
  - fast-mod boundary artifact percentiles (p95/p98)
  - long-run drift

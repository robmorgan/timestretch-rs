# Release Checklist

## Dual-Plane Deterministic Default Migration

Use this checklist when cutting a release that keeps
`StreamingEngine::Deterministic` as the default and phases out the old
opt-in deterministic toggle workflow.

- [ ] Verify CI quality gates are green for the default deterministic route:
  - `cargo test --test realtime_allocations -- --nocapture`
  - `TIMESTRETCH_STRICT_CALLBACK_BUDGET=1 cargo test --release --test quality_gates -- --nocapture`
  - `cargo test --test dual_plane_parity -- --nocapture`
- [ ] Confirm default routing behavior in tests:
  - New `StreamProcessor` starts on deterministic dual-plane backend.
  - `StreamingEngine::LegacyHybridRerender` remains available as explicit fallback.
- [ ] Confirm migration notes are present in release notes:
  - Deterministic stream path now uses dual-plane backend by default.
  - Legacy rerender path is now explicit (`StreamingEngine::LegacyHybridRerender`).
  - `set_dual_plane_deterministic(...)` is compatibility-only control and not required for default behavior.
- [ ] Re-run API docs/examples to ensure they use `set_streaming_engine(...)` semantics and do not require deterministic opt-in.
- [ ] Record benchmark deltas versus previous release baseline:
  - callback p99/p999 latency
  - fast-mod boundary artifact percentiles (p95/p98)
  - long-run drift
- [ ] Add a rollback note in release ops doc:
  - If regressions are found, instruct hosts to switch to `StreamingEngine::LegacyHybridRerender` while fixes are prepared.

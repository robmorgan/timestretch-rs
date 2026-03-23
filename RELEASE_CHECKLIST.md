# Release Checklist

## Streaming Runtime

Use this checklist when cutting a release that verifies the
streaming backend is working correctly.

- [ ] Verify CI quality gates are green for the shipping streaming route:
  - `cargo test --test realtime_allocations -- --nocapture`
  - `cargo test --features qa-harnesses --release --test streaming_quality -- --nocapture`
- [ ] Re-run API docs/examples to ensure they reflect the plain `StreamProcessor` behavior.
- [ ] Record benchmark deltas versus previous release baseline:
  - callback p99/p999 latency
  - streaming artifact percentiles on modulation-heavy material
  - end-of-stream length drift

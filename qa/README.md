# QA Harnesses

These harnesses are excluded from default `cargo test`.

Run them explicitly with `--features qa-harnesses`, for example:

- `cargo test --features qa-harnesses --release --test benchmarks -- --nocapture`
- `cargo test --features qa-harnesses --test quality_gates -- --nocapture`
- `cargo test --features qa-harnesses --test reference_quality -- --nocapture`
- `cargo test --features qa-harnesses --test quality_benchmark -- --ignored --nocapture`
- `cargo test --features qa-harnesses --release --test varispeed_keylock -- --nocapture`

## Varispeed Keylock (`varispeed_keylock`)

Stage 15 pitch-stability gate: streams a pure 440 Hz tone through the
±8%/2 s DJ ratio ride on both control paths and measures instantaneous
frequency deviation in cents (interpolated zero crossings, 100 ms windows).
The varispeed-first path is gated absolutely on the Live profile (p95/max)
and relative to the vocoder-tempo baseline on every profile; the baseline
rows are printed for comparison.

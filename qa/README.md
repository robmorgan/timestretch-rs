# QA Harnesses

These harnesses are excluded from default `cargo test`.

Run them explicitly with `--features qa-harnesses`, for example:

- `cargo test --features qa-harnesses --release --test benchmarks -- --nocapture`
- `cargo test --features qa-harnesses --test quality_gates -- --nocapture`
- `cargo test --features qa-harnesses --test reference_quality -- --nocapture`
- `cargo test --features qa-harnesses --test quality_benchmark -- --ignored --nocapture`

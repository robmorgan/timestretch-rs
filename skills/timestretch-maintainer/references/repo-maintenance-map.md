# Repository Maintenance Map

## Core Commands

1. Fast hygiene checks:
- `cargo fmt --all --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test --all-targets`

2. Quality-focused checks:
- `cargo test --test quality -- --nocapture`
- `cargo test --test quality_gates -- --nocapture`

3. Heavy DSP checks (explicit opt-in):
- `cargo test --test reference_quality -- --nocapture`
- `cargo run --release --example benchmark_quality`

## Key Project Locations

1. Rust library and DSP core:
- `src/`

2. Integration and quality tests:
- `tests/`
- `tests/quality.rs`
- `tests/quality_gates.rs`
- `tests/reference_quality.rs`

3. Benchmarks and corpus metadata:
- `benchmarks/`
- `benchmarks/manifest.toml`
- `benchmarks/README.md`

4. Existing optimization loop tooling:
- `optimize/`
- `optimize/Makefile`
- `optimize/scripts/`

5. CI policy and required checks:
- `.github/workflows/ci.yml`

## Maintenance Priorities

1. Keep CI-required checks green first (`fmt`, `clippy`, all-target tests).
2. Verify quality gates when DSP logic changes.
3. Run heavy reference and benchmark checks only when requested or when investigating quality regressions.
4. Prefer minimal patches that preserve existing behavior.

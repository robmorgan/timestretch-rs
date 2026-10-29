# Development

The workspace provides one `StreamSession` adapter. Consult [README](README.md),
[architecture](docs/CORE_ARCHITECTURE.md) and [session contracts](docs/SESSION_CONTRACT.md).
Desktop, CLI, examples and QA are workspace consumers. The library owns media
semantics; applications own threads and devices. Analysis is independent of DSP.

## Analysis version policy

`crates/timestretch-analysis/src/analysis/preanalysis.rs` defines
`PREANALYSIS_VERSION` and `MIN_COMPATIBLE_VERSION`. Increment the former when
analysis output changes materially; raise the latter when older cached results
should be regenerated. A naming-only change does not alter either version.

## Verification

```sh
cargo test --release --workspace --all-features
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
python3 -B -m unittest discover -s tools -p 'test_*.py'
```

Library MSRV is 1.85; the development compiler is pinned in rust-toolchain.toml.
[QA usage](qa/README.md) describes self-contained qualification and optional
public corpus/device checks. The standard checks use this checkout and generated signals.

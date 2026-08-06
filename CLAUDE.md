# CLAUDE.md

## Analysis Version Policy

`src/core/preanalysis.rs` carries two constants that control cached
analysis artifacts (`.tsa` sidecars and Halo's stored blobs):

- Bump `PREANALYSIS_VERSION` on any release whose **analysis output**
  changes materially (beat grids, onsets, key, tempo candidates) — not
  just on schema changes.
- Raise `MIN_COMPATIBLE_VERSION` with it whenever cached results from the
  previous version would be *worse* than re-analysis, so stale sidecars
  regenerate. Schema compatibility alone is not the bar.

Learned from v0.10.0 shipping the rigid-grid beat fit without a bump,
which left v7 artifacts ambiguous between old wandering and new rigid
grids (LEARNINGS.md). Checked at release time via RELEASE_CHECKLIST.md.

## CI Checks

The following checks run on every push to `main` and on every pull request. All must pass.

### Tests
```bash
cargo test --all-targets
```
Runs on Ubuntu, macOS, and Windows with the pinned toolchain from
`rust-toolchain.toml` (currently 1.97.0 — local builds use the same compiler),
plus Ubuntu with MSRV (1.85.0).

### Clippy
```bash
cargo clippy --all-targets -- -D warnings
```
All warnings are treated as errors.

### Format
```bash
cargo fmt --all --check
```
Code must be formatted with `rustfmt`.

### Documentation
```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```
Documentation must build without warnings.

### Desktop
```bash
cd desktop && cargo test --all-targets && cargo clippy --all-targets -- -D warnings
```
The `desktop/` crate is excluded from the workspace, so the root checks never
build it; CI checks it separately on macOS (its target platform).

# Release Checklist

Complements the `/release` flow (CHANGELOG fold, version bump, CI checks,
signed tag, publish). Human judgment items that the automation cannot
decide:

- [ ] **Analysis version policy** (CLAUDE.md "Analysis Version Policy"):
  if any analysis output changed materially this release, bump
  `PREANALYSIS_VERSION` in `src/core/preanalysis.rs`; if cached artifacts
  from the previous version would now be *worse* than re-analysis, raise
  `MIN_COMPATIBLE_VERSION` too so sidecars regenerate.
- [ ] CI green on the shipping surface: `cargo test --all-targets`,
  clippy `-D warnings`, `cargo fmt --check`, docs with
  `RUSTDOCFLAGS="-D warnings"`, and the desktop crate checks.
- [ ] Quality gates: the CI quality-gates job (`engine_ab_matrix`,
  `engine_wcet`) and public-corpus job (`bpm_accuracy`,
  `rubberband_reference_gate`) are green on the release commit.
- [ ] If DSP changed audibly this release: owner listen recorded in the
  relevant ROADMAP stage note before tagging, with the implementation
  state (commit) noted alongside the verdict.
- [ ] README latency table and RT-contract claims still match the code
  (profile latencies, tempo range, MSRV/toolchain pins).

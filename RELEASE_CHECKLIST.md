# Release checklist

0.16 is a breaking timestretch-only migration. Publishing remains a separate
action. Implementation completion does not establish realtime qualification.

- [ ] Review [qualification results](benchmarks/QUALIFICATION.md),
  including failed cells, native/converted rates, actual device coverage and
  the unchanged physical deadline/50% headroom gate.
- [ ] Checked-in native fixtures and generated-signal regression checks pass.
  All applicable retained quality gates pass without private inputs.
- [ ] Workspace tests, portable Linux/Windows checks, macOS provider checks,
  Rust 1.85 library build, supported features, rustdoc, Clippy and formatting pass.
- [ ] Package verification succeeds for all four publishable library crates.
- [ ] Callback and moved-session Rust/system allocation audits pass with a
  positive control. Transport/concurrency and extended campaigns pass.
- [ ] Record human listening results against the concealed comparison bundle.
- [ ] Review the migration guide and declared joint ratio/rate/channel limits.
- [ ] Analysis changes follow the existing version policy: increment
  `PREANALYSIS_VERSION` in the analysis crate when analysis output materially
  changes, and raise `MIN_COMPATIBLE_VERSION` only when old artifacts should be
  regenerated. Extraction alone does not change artifact versions.
- [ ] Preserve source, binary and raw-evidence hashes outside disposable target/.

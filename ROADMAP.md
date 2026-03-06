# Roadmap

## Goal

Make `timestretch-rs` competitive with production-grade realtime stretchers by
closing the remaining gaps in realtime contract quality, external quality
evidence, channel/spatial handling, and CPU efficiency.

## Guiding Rules

- No silent audio loss in realtime paths without explicit telemetry.
- Realtime-safe means from first callback after preparation, not only after
  warmup.
- Latency must be reported exactly enough for host compensation.
- Quality claims must be backed by mandatory external comparisons on real music,
  not only synthetic regressions or self-parity.
- Stereo and multichannel behavior must be first-class, not mono-derived.

## Phase 1: RT Observability And Host Contract

Status: `in_progress`

- Add cumulative overload/xrun telemetry for the dual-plane RT core.
- Surface deterministic runtime telemetry through `StreamProcessor`.
- Stop treating callback failures as silent black holes from the host point of
  view.
- Add exact delay reporting:
  - algorithmic delay
  - buffered delay
  - current profile/tier contribution
- Add fixed-buffer host-facing APIs where `Vec` append semantics are still the
  default.

Acceptance:

- Hosts can detect dropped input/output, callback errors, and current RT state.
- Deterministic mode reports exact current delay, not only a nominal minimum.
- No silent discard path remains unobservable.

## Phase 2: External Quality Proof

Status: `planned`

- Build a locked benchmark corpus with checksummed references for:
  - drums and loops
  - full mixes
  - vocals
  - bass-heavy material
  - wide stereo ambience
- Make external comparisons mandatory in CI where licensing allows.
- Fix current reference asset and checksum drift.
- Expand beyond one-track and one-reference coverage.

Acceptance:

- The repo can reproduce comparison reports against external references without
  manual setup drift.
- Realtime and offline paths are both evaluated on real-world material.

## Phase 3: Stereo, Multichannel, And Spatial Quality

Status: `planned`

- Replace the mono/stereo-only public channel model with general channel counts.
- Add channel-coupled transient and tonal analysis instead of mono fold-down
  only.
- Revisit the current mid/side assumption that Side is always PV-only.
- Add true stereo and multichannel quality tests, not mono-derived proxies.

Acceptance:

- 2-channel, stem, and surround-style workflows have explicit behavior and
  tests.
- Stereo width and transient placement survive modulation and large ratios.

## Phase 4: Residual, Noise, And Pitch Quality

Status: `planned`

- Improve the residual and noise path beyond linear-resampled leftovers.
- Add a stronger ambience and reverb preservation strategy.
- Replace the current realtime linear pitch resampler with a higher-quality RT
  resampling path.
- Add content-aware policies for noisy, diffuse, and vocal material.

Acceptance:

- Reverb tails, air, and noisy textures hold up better at large ratios.
- Realtime pitch modulation quality is no longer the obvious weak point.

## Phase 5: Performance And Determinism

Status: `planned`

- Add explicit SIMD or backend dispatch for hot loops.
- Make callback-budget gates mandatory by default.
- Tie RT budgets to the actual host period instead of static constants alone.
- Remove remaining first-use growth from callback-reachable paths after
  preparation.

Acceptance:

- p99 and p999 callback timing is enforced continuously.
- RT behavior remains bounded across supported callback sizes and ratios.

## Phase 6: Packaging And Integration

Status: `planned`

- Improve host integration docs and examples around exact delay, overload
  handling, and profile selection.
- Decide whether a stable C ABI or plugin-facing shim is required.
- Keep the RT facade crate aligned with the main host-facing contract.

Acceptance:

- Embedding the library in a DAW, engine, or plugin host does not require
  source-level guesswork.

## Current Implementation Slice

This change set continues Phase 1 by adding a fixed-buffer interleaved flush
API to the RT core and dual-plane facade, so end-of-stream drain no longer
depends on `Vec` append semantics in the deterministic path.

# Roadmap

## Goal

Make `timestretch-rs` competitive with production-grade realtime stretchers by
closing the remaining gaps in audible quality, modulation stability, realtime
contract quality, external quality evidence, and API strictness.

## Current Status

The repository is already beyond a toy implementation:

- The core hybrid design is real: phase vocoder, WSOLA, HPSS, multi-resolution,
  stereo handling, streaming, and a deterministic RT path all exist.
- CI, quality gates, regression tests, and allocation tests are already in
  place.
- The main gap is not "missing DSP ideas". The main gap is the last 20%:
  stronger continuity under modulation, tighter routing/decomposition, stricter
  invariants, and reference-driven tuning.

The current branch should be treated as not yet production-grade until the
fast-modulation artifact regression is fixed.

## Principles

- Fix audible regressions before adding features.
- Make the RT-safe path the default, obvious path.
- Reject malformed input instead of silently truncating or falling back.
- Prefer reference-driven quality gates over self-comparison alone.
- Preserve the EDM-first focus unless there is a deliberate decision to expand
  into a broader general-purpose stretcher.

## [~] Stage 1: Stabilize Fast Modulation and Transition Quality

Automation: auto

### Why

This is the clearest current signal that the library is not yet
production-stable. If dynamic ratio changes still produce obvious boundary
artifacts, improvements elsewhere will not matter.

### Primary Files

- `src/dual_plane/rt.rs`
- `src/stream/processor.rs`
- `src/stream/transient_scheduler.rs`
- `src/stretch/phase_vocoder.rs`
- `qa/quality_gates.rs`

### Work

- Fix ratio-transition continuity in the dual-plane deterministic path.
- Reduce profile-switch churn and improve hysteresis for automation-heavy use.
- Tighten transient reset scheduling so fast modulation does not over-trigger
  phase resets.
- Review how phase state is preserved or reseeded during rapid ratio changes.
- Add additional focused tests around repeated short-interval modulation, not
  just the existing one gate.

### Exit Criteria

- `cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture`
  passes with margin, not barely.
- Release-mode modulation no longer produces obvious clicks, roughness, or
  discontinuities on synthetic DJ-like material.
- Fixes do not regress steady-state deterministic streaming quality.

## [ ] Stage 2: Replace Binary Segment Routing with Confidence-Based Blending

Automation: auto

### Why

The current hybrid engine still routes whole segments as either transient or
tonal. That is too coarse for production quality because attacks, decays, and
mixed-content regions need softer treatment.

### Primary Files

- `src/analysis/adaptive_snapshot.rs`
- `src/analysis/transient.rs`
- `src/stretch/hybrid.rs`

### Work

- Replace hard transient-versus-tonal segment routing with event-centered masks.
- Use transient confidence to create a transient core, blended shoulders, and
  tonal sustain regions.
- Stop relying on post-render truncation and padding as the main way to enforce
  target length.
- Make better use of fractional onset timing when placing transitions.
- Reduce crossfade plans that assume a segment is homogeneous from start to end.

### Exit Criteria

- Boundary artifact metrics improve on click-pad, drum-loop, and vocal fixtures.
- Hybrid rendering produces fewer audible handoff artifacts around transient
  tails.
- Exact-length output is achieved without heavy dependence on hard truncation or
  last-sample padding.

## [ ] Stage 3: Upgrade Analysis from Fixed EDM Heuristics to Rolling Adaptive Analysis

Automation: auto

### Why

The transient and confidence front end is currently too static. Production
libraries usually make better decisions because they use rolling, multi-scale,
content-adaptive analysis instead of a small set of fixed weights and one-shot
confidence estimates.

### Primary Files

- `src/analysis/transient.rs`
- `src/analysis/adaptive_snapshot.rs`
- `src/analysis/beat.rs`
- `src/dual_plane/analysis_plane.rs`

### Work

- Replace single-resolution assumptions with rolling multi-resolution analysis.
- Revisit fixed spectral weights and band boundaries used for transient
  detection.
- Make tonal, transient, and noise confidence evolve over time instead of being
  estimated from only a narrow view of the signal.
- Improve beat confidence so beat-aware behavior is useful outside ideal EDM
  material.
- Expose enough telemetry to inspect analysis mistakes during tuning.

### Exit Criteria

- Routing decisions become more stable across mixed material and changing song
  sections.
- False positives and missed onsets drop on non-trivial material such as
  vocal-plus-drums and bright, noisy mixes.
- Beat-aware logic improves timing when helpful and backs off when confidence is
  low.

## [ ] Stage 4: Give Harmonic, Percussive, and Residual Content Real Independent Paths

Automation: auto

### Why

The repository already has HPSS and multiresolution processing, but the
decomposition remains static and the residual branch is still weak. Cymbals,
reverb tails, and noisy material are where shortcuts become obvious.

### Primary Files

- `src/analysis/hpss.rs`
- `src/stretch/multi_resolution.rs`
- `src/stretch/hybrid.rs`
- `src/core/crossover.rs`

### Work

- Replace fixed HPSS defaults with adaptive decomposition parameters.
- Improve the multi-resolution strategy so the split points and behavior are not
  purely static.
- Give residual and noise-like content a real processing path instead of linear
  resampling.
- Revisit how harmonic, percussive, and residual outputs are recombined so phase
  relationships survive better.
- Add targeted fixtures for bright percussion, reverb-heavy stems, and noisy
  vocals.

### Exit Criteria

- Spectral-flux similarity and subjective quality improve on bright/noisy
  content.
- Metallic artifacts and smeared air-band content are reduced.
- The residual path contributes audible quality instead of acting as a fallback
  patch.

## [ ] Stage 5: Replace Hard Transient Classes with Continuous Event Shaping

Automation: auto

### Why

The current kick/snare/hat classifier and attack-copy heuristic are useful, but
too coarse for production-grade event handling.

### Primary Files

- `src/stretch/hybrid.rs`
- `src/stretch/wsola.rs`
- `src/analysis/transient.rs`

### Work

- Replace three hard transient classes with continuous descriptors such as
  attack duration, low-band dominance, noisiness, and periodicity.
- Scale attack-copy length, WSOLA segment size, search range, and crossfade
  length continuously per event.
- Make transient rendering respond to event confidence rather than assuming
  every detected onset deserves the same type of intervention.
- Reduce cases where attacks are preserved but decays or body content are
  mismatched.

### Exit Criteria

- Attacks stay sharp on more than narrow EDM cases.
- Transient preservation improves on mixed and non-EDM material.
- WSOLA mismatch and repetition artifacts become less obvious on event tails.

## [ ] Stage 6: Raise Streaming Pitch Quality

Automation: auto

### Why

Realtime pitch currently depends on a linear resampler. That is acceptable as a
control mechanism, but not as a production-quality pitch stage for bright
material.

### Primary Files

- `src/stream/processor.rs`
- `src/core/resample.rs`

### Work

- Replace linear realtime pitch resampling with a bounded-latency higher-quality
  resampler.
- Keep the current linear path only as an explicit low-quality or emergency
  fallback.
- Measure CPU cost and callback safety after the new resampler is introduced.
- Add quality checks for hats, vocals, and sustained bright tones under stream
  pitch modulation.

### Exit Criteria

- High-frequency roughness drops when `pitch_scale != 1.0`.
- Pitch modulation sounds materially cleaner on hats, cymbals, and vocals.
- Callback-safe behavior is preserved.

## [ ] Stage 7: Harden API Contracts and Make Silent Failure Impossible

Automation: auto

### Why

Production libraries usually fail loudly on malformed input. Silent truncation,
implicit channel coercion, and soft fallbacks turn host mistakes into bad audio
that is difficult to debug.

### Primary Files

- `src/lib.rs`
- `src/core/types.rs`
- `src/error.rs`
- `tests/edge_cases.rs`
- `tests/algorithm_edge_cases.rs`

### Work

- Reject buffers whose sample count is not divisible by channel count.
- Stop silently truncating to the shortest channel during interleave paths.
- Tighten `AudioBuffer` invariants so malformed frame layouts are impossible to
  construct accidentally.
- Replace boolean or silent fallback behavior with explicit `Result` where
  state changes can fail.
- Audit every "helpful fallback" that can hide a host integration bug.

### Exit Criteria

- Malformed channel and frame layouts fail deterministically.
- Host misuse becomes easy to diagnose from returned errors.
- Public API behavior is stricter and easier to reason about.

## [ ] Stage 8: Make External Quality Evidence Mandatory

Automation: auto

### Why

The repository already has useful benchmark infrastructure, but too much of it
is optional, synthetic, or dependent on private local setup. Production quality
needs authoritative, repeatable evidence.

### Primary Files

- `qa/reference_quality.rs`
- `qa/rubberband_comparison.rs`
- `qa/quality_benchmark.rs`
- `benchmarks/manifest.toml`
- `benchmarks/README.md`
- `.github/workflows/ci.yml`

### Work

- Define a small redistributable public corpus that can run in CI.
- Keep the larger private corpus for deeper local tuning, but stop relying on it
  as the only meaningful reference test.
- Promote at least one external-reference comparison from optional to required.
- Tighten streaming-vs-batch and batch-vs-reference tolerances so they reflect
  audible defects, not just rough parity.
- Produce machine-readable reports that make regressions obvious in PRs.

### Exit Criteria

- CI fails when external-reference quality regresses on the public corpus.
- Synthetic self-regression is no longer the main quality signal.
- Listening tests and objective benchmarks point in the same direction.

## [ ] Stage 9: Decide the Product Boundary

Automation: manual

### Why

The codebase currently mixes two ambitions: "excellent EDM-focused stretcher"
and "general-purpose production-grade library". Those are related, but not the
same target.

### Decision

Make an explicit choice:

- Stay EDM-first and optimize hard for DJ workflows, stereo mixes, and tempo
  automation.
- Or broaden into a general-purpose library and retune analysis, presets, and
  validation around wider material classes.

### Impact

- The right benchmark corpus depends on this choice.
- The right API defaults depend on this choice.
- The right quality gates depend on this choice.

## Not a Priority Yet

These should stay secondary until the quality roadmap above is complete:

- SIMD and architecture-specific acceleration
- Desktop and web tooling polish
- Additional presets
- Wider API surface
- New convenience wrappers

## Definition of Success

`timestretch-rs` should be considered production-grade when all of the following
are true:

- Release-mode fast modulation is stable.
- Streaming and batch quality are tightly aligned on the supported use cases.
- Bright, noisy, and mixed-content material no longer exposes obvious weak
  fallback paths.
- The public API rejects malformed input instead of silently degrading.
- At least one external-reference benchmark is mandatory in CI.
- Realtime-safe usage is clearly separated from non-RT or legacy behavior.

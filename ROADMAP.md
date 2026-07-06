# Roadmap

## Goal

Make `timestretch-rs` competitive with production-grade realtime stretchers by
closing the remaining gaps in audible quality, modulation stability, realtime
contract quality, external quality evidence, and API strictness — and then
close the deck-integration gaps (latency, seek/cue behavior, transport
completeness) that separate a quality stretcher from an engine a real DJ deck
can be built on.

## Current Status

The repository is already beyond a toy implementation:

- The core hybrid design is real: phase vocoder, WSOLA, HPSS, multi-resolution,
  stereo handling, streaming, and a deterministic RT path all exist.
- CI, quality gates, regression tests, and allocation tests are already in
  place, including callback-budget and zero-allocation gates for the realtime
  pitch path.
- The realtime pitch stage is production-grade (Stage 6, complete): an
  anti-aliased Kaiser-sinc resampler with a click-free ~50 ms control glide,
  with the old linear resampler kept as an explicit fallback. Mono streams get
  the same transient-driven phase resets as stereo.
- Fast-modulation stability (Stage 1, complete): per-callback jog-wheel
  gestures (nudge/ride/snap) are click-free under hard CI bounds
  (`tests/modulation_torture.rs`), the scheduler's modulation-hold is wired,
  and the flush-tail splices are fixed. The remaining PV ratio-step seam
  residual (≤4x a tone's natural slew on instant 8% snaps) is documented in
  Stage 1 and deferred to Stage 12.
- The main gap is not "missing DSP ideas". The main gap is the last 20%:
  tighter routing/decomposition, stricter invariants, and reference-driven
  tuning — then the DJ deck readiness stages.

## Principles

- Fix audible regressions before adding features.
- Make the RT-safe path the default, obvious path.
- Reject malformed input instead of silently truncating or falling back.
- Prefer reference-driven quality gates over self-comparison alone.
- Preserve the EDM-first focus unless there is a deliberate decision to expand
  into a broader general-purpose stretcher.

## [x] Stage 1: Stabilize Fast Modulation and Transition Quality

Automation: auto

### Why

This is the clearest current signal that the library is not yet
production-stable. If dynamic ratio changes still produce obvious boundary
artifacts, improvements elsewhere will not matter.

### Primary Files

- `src/stream/processor.rs`
- `src/stream/transient_scheduler.rs`
- `src/stretch/phase_vocoder.rs`
- `qa/streaming_quality.rs`

### Work

- [x] Fix ratio-transition continuity in the streaming path. Three click
  sources found and fixed via the torture test: (1) the WSOLA overlay armed
  on PV-side flux (which spikes on every ratio step) and spliced in at 0.90
  weight instantly — now gated on input-domain onsets during slews and
  ramped in over ~3 ms; (2) the bit-exact unity passthrough re-engaged
  mid-stream when a nudge settled back to 1.0, hard-switching between raw
  input and the PV stream (a full-scale splice) — the DSP path now stays
  engaged for the life of the stream once used (`dsp_engaged`); (3) flush
  discontinuities (below).
- [x] Tighten transient reset scheduling so fast modulation does not
  over-trigger phase resets: the modulation-hold machinery is now wired —
  `modulation_hold_overlap_windows()` maps the in-flight ratio+pitch slew to
  scheduler hold windows, activating low-band suppression and trigger
  tightening that previously existed only as dead code.
- Review how phase state is preserved or reseeded during rapid ratio changes.
  **Remaining (deferred)**: the PV ratio-step seam — `hop_synthesis` jumps
  immediately while phase slews over ~3 frames. Measured residual after all
  fixes: 4.0x a pure tone's natural slew on instant 8% snaps (was 39x),
  1.8-2.8x on nudges/rides. Inaudible-to-marginal; revisit with Stage 12.
- [x] Add focused tests around short-interval modulation and callback
  boundaries (`tests/modulation_torture.rs`).
- [x] Add a jog-wheel-style torture test: continuous per-callback ratio
  modulation (nudge, ride, snap back) with hard click/slew/length bounds
  over the full output including flush, plus reset over/under-trigger
  budgets on percussive material.
- [x] Fix the end-of-stream flush splice click: fractional-period,
  gain-matched, crossfaded tonal-tail splice (was integer-period,
  amplitude-floored hard rewrite); faded input padding instead of a hard cut
  to zeros; fade-out after truncation. The pitch-sweep zipper test now scans
  the flush region instead of excluding it.

### Exit Criteria

- [x] `cargo test --features qa-harnesses --release --test streaming_quality -- --nocapture`
  passes with margin, including the new `streaming_modulation_quality_benchmark`
  (clicks=0, max slew 2.75x theoretical vs a 12x soft gate).
- [x] Release-mode modulation no longer produces obvious clicks, roughness,
  or discontinuities on synthetic DJ-like material: worst-case gesture click
  reduced from 1.03 (full scale) to 0.063 (residual seam) on a 0.5-amplitude
  tone.
- [x] Fixes do not regress steady-state streaming quality: full suite,
  strict callback-budget gates, and existing METRIC scores unchanged.

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

## [x] Stage 6: Raise Streaming Pitch Quality

Automation: auto

### Why

Realtime pitch previously depended on a linear resampler — acceptable as a
control mechanism, but not as a production-quality pitch stage for bright
material.

### Primary Files

- `src/stream/processor.rs`
- `src/core/resample.rs`

### Work

- [x] Replace linear realtime pitch resampling with a bounded-latency
  higher-quality resampler (`StreamingSincResampler`: interpolated
  Kaiser-windowed sinc, 16 half-taps, ratio-adaptive anti-aliasing cutoff,
  default via `StreamPitchQuality::Sinc`).
- [x] Keep the current linear path only as an explicit low-quality or emergency
  fallback (`StreamPitchQuality::Linear` via `set_pitch_resampler_quality`).
- [x] Measure CPU cost and callback safety after the new resampler is
  introduced (`quality_gate_streaming_pitch_callback_budget`: avg callback
  ratio 0.032 vs 0.034 baseline; pitch-path zero-alloc tests in
  `tests/realtime_allocations.rs`).
- [x] Add quality checks for hats, vocals, and sustained bright tones under
  stream pitch modulation (`streaming_pitch_quality_benchmark` in
  `qa/streaming_quality.rs`).
- [x] (Bonus) `set_pitch_scale` now glides over ~50 ms instead of hard-resetting
  the resampler, removing clicks/zipper on DJ pitch nudges and sweeps.

### Exit Criteria

- [x] High-frequency roughness drops when `pitch_scale != 1.0`: spurious
  (alias/image) energy on a bright tone stack is ~265x lower than linear at
  pitch 1.06 and ~460x lower at 1.30.
- [x] Pitch modulation sounds materially cleaner on hats, cymbals, and vocals:
  in-band interpolation images (e.g. 13.1 kHz -> 11.2 kHz at 1.06) are
  suppressed below the PV noise floor; sweeps are click-free.
- [x] Callback-safe behavior is preserved: zero allocations in the pitch path
  (steady and swept) and callback budget unchanged within noise.

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
- `scripts/compare_rubberband.sh`
- `benchmarks/manifest.toml`
- `benchmarks/README.md`
- `.github/workflows/ci.yml`

### Work

- Investigate the harmonic-track anomaly in the RubberBand comparison first:
  `scripts/compare_rubberband.sh` reports ~-24 LUFS difference and ~0.15
  spectral similarity on the harmonic sweep, which smells like a harness or
  fixture bug rather than a real audio gap. Benchmarks cannot be trusted or
  tightened until this is explained.
- Define a small redistributable public corpus that can run in CI.
- Keep the larger private corpus for deeper local tuning, but stop relying on it
  as the only meaningful reference test.
- Promote at least one external-reference comparison from optional to required.
- Tighten batch-vs-reference tolerances so they reflect audible defects, not
  just rough parity. (Streaming-vs-batch tightening is driven by Stage 12,
  which changes the streaming engine itself.)
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
- The DJ Deck Readiness stages below are written assuming the EDM/DJ-first
  choice; broadening instead would reprioritize them.

## DJ Deck Readiness

The stages below close the gap between "quality streaming stretcher" and
"engine a real DJ deck can be built on" (the Serato Pitch 'n Time DJ /
Elastique bar). They assume the EDM/DJ-first outcome of Stage 9. Ordering:
Stage 10 and Stage 11 are DJ-blocking and should follow Stage 1 directly;
Stage 12 is the long-tail quality work; Stage 13 is completeness.

## [ ] Stage 10: Make Low-Latency Streaming First-Class

Automation: auto

### Why

Default streaming latency is `fft * 3/2` = 6144 samples (~139 ms at 44.1 kHz).
A DJ nudging against a beat needs roughly 10-40 ms control-to-audio. The
low-latency constructor (1024 FFT, ~35 ms) exists but trades quality blindly,
and `QualityMode` barely changes streaming DSP today, so latency and quality
cannot be traded deliberately.

### Primary Files

- `src/stream/processor.rs`
- `src/core/types.rs`
- `desktop/src/processor.rs`

### Work

- Make `QualityMode` meaningfully reconfigure the streaming path (FFT/hop,
  lookahead, enabled DSP features) instead of only sizing buffers.
- Tune the transient scheduler and the sinc pitch stage for the 1024-FFT
  low-latency profile so it is a supported mode, not a degraded one.
- Revisit the `effective_min_frames` gating (`fft * 2` off-unity) that
  inflates latency exactly when a DJ is off-unity, which is always.
- Measure and document actual control-to-audio latency per mode (ratio
  change, pitch change, first-sample-out), not just buffer arithmetic.

### Exit Criteria

- A supported low-latency mode at or under ~40 ms with acceptable quality on
  DJ material at typical ratios (0.92-1.08).
- Quality modes are audibly and measurably distinct in streaming.
- A published latency table covers each mode and control path.

## [ ] Stage 11: Warm-Start Seek, Cue, and Loop Support

Automation: auto

### Why

DJ decks jump constantly: cue points, loops, beat jumps. Today every jump
means `reset()` plus a cold 1.5x-FFT prebuffer and PV warm-up transient — the
desktop app rebuilds the whole processor per seek. Commercial engines re-prime
from surrounding audio so a jump is seamless.

### Primary Files

- `src/stream/processor.rs`
- `src/stretch/phase_vocoder.rs`
- `desktop/src/processor.rs`

### Work

- Add a warm-start API: prime processor state from audio preceding the seek
  target so the first emitted frame is already converged.
- Bound the CPU cost of a jump so rapid cue drumming stays realtime-safe.
- Keep the warm-start path allocation-free (extend
  `tests/realtime_allocations.rs`).
- Preserve pitch/ratio control state across jumps (no re-glide from unity).
- Convert the desktop app's seek handling from rebuild-processor to
  warm-start as the reference integration.

### Exit Criteria

- Cue jumps and loop wraps produce no audible warm-up transient or gap.
- A beat-jump/loop torture test passes clean at DJ ratios.
- Desktop seek no longer rebuilds the processor.

## [ ] Stage 12: Port Hybrid Quality Into the Deterministic Stream Engine

Automation: auto

### Why

Offline rendering gets multi-resolution FFT, HPSS, and segmented transient
WSOLA; the streaming path approximates all of that with a single-FFT PV plus
a stack of corrective heuristics (EMA gain matching, spectral shelving, dry
blend, WSOLA overlay). Kick and hat sharpness under +/-8% stretch is judged in
the streaming path, because that is where beatmatching happens. This is the
streaming-side counterpart of Stages 2-5.

### Primary Files

- `src/stream/processor.rs`
- `src/stretch/multi_resolution.rs`
- `src/analysis/hpss.rs`
- `src/stream/transient_scheduler.rs`

### Work

- Design a bounded-latency multi-band PV for streaming (larger FFT for
  sub-bass phase coherence, smaller FFT for transient-sharp highs).
- Evaluate an incremental/causal HPSS variant for the streaming path, gated
  behind `QualityMode`.
- As structural quality lands, retire corrective heuristics instead of
  stacking more (each heuristic removed is a win in itself).
- Keep per-callback cost bounded and allocation-free; extend the callback
  budget gates to the new configuration.

### Exit Criteria

- Streaming-vs-batch tolerances in `tests/streaming_batch_parity.rs` and
  `tests/stretch_quality_regressions.rs` tighten measurably.
- Kick/hat transient sharpness in streaming is comparable to batch at DJ
  ratios.
- The corrective heuristic stack shrinks rather than grows.

## [ ] Stage 13: Deck Control Completeness

Automation: manual

### Why

Real decks are more than play-at-a-ratio: keylock has defined behavior at
extreme rates, playback can reverse, scratching hands control to plain
resampling, and pitch faders have hardware-grade resolution. These behaviors
need definitions and tests even where the answer is "the host does it".

### Primary Files

- `src/stream/processor.rs`
- `src/lib.rs` (docs)
- `desktop/src/processor.rs`

### Work

- Define keylock behavior beyond roughly +/-50%: blend to plain resampling
  (like commercial decks) rather than letting PV quality collapse; specify
  the crossover and make the transition inaudible.
- Define and test reverse playback: what the engine supports natively versus
  what the host feeds it.
- Document the scratch story: scratching is host-side variable-rate
  resampling; specify how the engine re-enters cleanly after a scratch
  (ties into Stage 11 warm-start).
- Verify 48 kHz and 96 kHz end-to-end (all tuned constants are Hz-based;
  confirm none assume 44.1 kHz).
- Verify CDJ-grade pitch resolution: 0.02% fader steps must ride the 50 ms
  control glide without artifacts or drift.

### Exit Criteria

- Documented, tested behavior for every transport control across its range.
- Keylock degrades gracefully (blend), never catastrophically.
- 48/96 kHz produce equivalent quality metrics to 44.1 kHz.

## Not a Priority Yet

These should stay secondary until the quality roadmap above is complete:

- SIMD and architecture-specific acceleration
- Desktop and web tooling polish (the desktop app does serve as the reference
  integration for Stages 10-13, but its UI/UX polish stays secondary)
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
- A deck built on the engine feels like hardware: low-latency nudges, seamless
  cue jumps, click-free pitch rides, and graceful keylock at extremes.

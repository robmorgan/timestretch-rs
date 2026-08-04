# Roadmap

## Goal

Rebuild `timestretch-rs` around a real-time-first engine architecture so that a
DJ deck built on it feels like hardware: tempo control at resampler latency,
total pipeline delay under ~15 ms, transparent keylock at DJ ratios, seamless
cue/loop behavior, and one engine whose live output is the same audio the
quality benchmarks measure.

This roadmap replaces the previous 16-stage document (see git history for the
full text). That roadmap treated streaming as a feature added to an offline
library and spent most of its open stages compensating for that inversion. An
architectural review (July 2026) concluded the remaining problems are
structural, not missing DSP: the streaming processor is a ~5.2k-line
orchestrator wrapping a phase vocoder in corrective heuristics; latency is
gated at 1.5–2x FFT size; the API is push-based with fallible calls in the
audio path; and the offline hybrid engine diverges from the live path
entirely. The fix is a new engine built on the inverted architecture that the
old roadmap's Stage 15 (varispeed-first keylock) already proved out as a
retrofit.

## Status (2026-07-19)

The engine rebuild is complete: Stages 1–9 are done, the old engine is
deleted, and the Definition of Success below holds for the rebuild scope
(15 ms budget, single engine both modes, machine-verified RT contract,
required external-reference evidence, 9/9 A/B matrix). The offline
low-band contract is settled: offline `stretch()` shares the live keylock
semantics by design — < 150 Hz follows tempo at DJ ratios (shipped in
0.8.0).

What separates the current state from production grade **for the DJ app**
is scope, not engine correctness:

- **Stage 10** (general-purpose beat tracking) — the algorithm side has
  landed; the annotated corpus and its gates have not. Wrong grids on
  non-EDM material poison beat jumps, loops, and the artifact-driven
  splice guidance the engine's quality depends on.
- **Stage 11** (wide-range Master Tempo) — promoted from the backlog now
  that its post-cutover condition is met.
- **Stage 12** (robustness hardening) — no-panic guarantees on arbitrary
  input, fuzzed parsers, long-run soak.

Production grade **as a public library** additionally requires the 1.0
path at the bottom of this file, which is gated on an explicit owner
decision about the crate's audience.

## Architecture Decision (settled July 2026)

The target architecture is settled. These are decisions, not open questions:

- **Stage-graph engine.** New code lives in `src/engine/`: small DSP stages
  with a uniform trait (fixed-block `process`, `latency_frames()`, `reset()`,
  `prime(history)`), composed into fixed per-profile chains. No god object.
- **Varispeed-first only.** The signal path is:
  source → varispeed sinc resampler (tempo axis: sample-accurate retargets,
  no control glide) → band split at ~150 Hz → low band **not keylocked**
  (pitch follows tempo; inaudible at DJ ratios, and it deletes the giant
  sub-bass FFT and its latency) → high band pitch-corrected at a small,
  slowly-varying transposition.
- **Corrector selected by transposition magnitude.** Below ~5% transposition:
  beat-synchronous time-domain SOLA (near-zero latency, transparent
  transients). Above: small-FFT phase vocoder with identity phase locking and
  artifact-driven transient phase resets. The ~150 Hz cutoff, the ~5%
  threshold, and the PV FFT size are tuning constants settled by QA evidence
  (Stages 2, 3, 7), not upfront.
- **RT-first pull API.** `process(&mut self, out: &mut [f32])` fills exactly
  the requested frames. Control lives on a separate handle with a lock-free,
  timestamped parameter mailbox (sample-accurate application). No `Result`
  and no allocation in the hot path: invariants are enforced at construction,
  input validation is debug-only.
- **Artifact-first analysis.** The `PreAnalysisArtifact`
  (`src/core/preanalysis.rs`: beatgrid, onsets, strengths, band flux) is the
  primary control signal — it drives splice-point selection, phase resets,
  and transient protection. Online detection is the fallback for
  artifact-less streams, not the default.
- **Single engine, both modes.** Offline/batch is the same graph with
  unlimited lookahead and a guaranteed artifact. Streaming-vs-batch parity
  becomes true by construction.
- **Latency budget as a design constraint.** Total pipeline delay ≤ ~15 ms at
  44.1 kHz; control-to-audio ≈ varispeed resampler lookahead (tens of
  samples). Any stage that cannot meet the budget does not ship.

## Binding Policies

- **Old-engine feature freeze from day zero.** `src/stream/processor.rs`,
  `src/stretch/hybrid.rs`, and `src/stretch/multi_resolution.rs` accept
  crash/correctness fixes only. All quality work lands in `src/engine/`.
- **Wrap, don't move.** The new engine consumes `src/core/*` and the clean
  DSP structs (`StreamingSincResampler` in `src/core/resample.rs`,
  `PhaseVocoder` in `src/stretch/phase_vocoder.rs`, `ThreeBandSplitter` in
  `src/core/crossover.rs`, `Wsola` in `src/stretch/wsola.rs`,
  `TransientEventScheduler` in `src/stream/transient_scheduler.rs`,
  `PreAnalysisArtifact`) where they are — no file moves until Stage 9
  cutover, so the frozen old engine and its tests never break underneath.
- **Parity means new ≥ old on the same fixture and metric** — never
  similarity to the old engine's output (that would reward its heuristics).
  Absolute QA thresholds are re-derived from new-engine measurements at
  cutover.
- **EDM/DJ-first stays binding** (settled in the previous roadmap's Stage 9):
  quality gates are DJ material, DJ ratios (0.92–1.08 primary, ±20%
  secondary), streaming path first. The public API may break freely — the
  crate is pre-1.0 and its customer is the author's DJ application.

## Principles

- Fix structure instead of stacking corrective heuristics; every heuristic
  removed is a win in itself.
- The RT-safe path is the only path: allocation-free, WCET-bounded,
  infallible by construction.
- Riskiest bets first: the falsifiable quality claims (un-keylocked low band,
  small-FFT budget, SOLA threshold) are validated in Stages 2–3, before
  anything is stacked on them.
- Every stage ends with the desktop app audibly playing through the new
  chain — vertical slices, never plumbing-only stages.
- CI stays green throughout: old suites run frozen until cutover; new gates
  are added per stage.

## Stage Sequence

Dependencies form a line: 1 → 2 → 3 → 4 → 5 → 6 → 7 → {8, 9}, with 8 → 9.
Stage 10 is a parallel analysis/UI track: it touches only the analysis front
end and the desktop app, has no dependency on Stages 4–9, and can be worked
at any point alongside the engine line.

Stages 11 and 12 are post-cutover tracks, independent of each other and of
Stage 10. Stage 11 builds on the settled engine graph (Stage 9); Stage 12
touches no DSP.

## [x] Stage 1: Walking Skeleton — Pull-Based Engine Core with Varispeed Tape Mode

Automation: auto

> **Completed 2026-07-12** (commit `8a1d214`). `src/engine/` (stage trait,
> SPSC mailbox + `EngineController`/`EngineProcessor` split, source ring
> with underrun policy and `TimelineMap` port of `RatioMapFifo`, varispeed
> head, Tape profile), desktop pull-native path behind a new "Deck"
> selector (`desktop/src/pull_deck.rs`, `AudioEngine::new_pull`), and the
> A/B adapter (`qa/ab/mod.rs`, smoke harness `qa/engine_ab.rs`).
> Ported gates pass: zero-alloc steady state under per-callback retargets
> (`tests/engine_realtime_allocations.rs`), first-sample-out == reported
> latency (0) and control-to-audio ≤ lookahead + one callback
> (`tests/engine_latency.rs`), tape torture clicks = 0 at 1.5x/1.1x
> theoretical slew vs the old engine's 6x/1.5x bounds
> (`tests/engine_modulation_torture.rs`). Old-engine suites untouched and
> green. Owner listening check on the desktop pull deck: passed.

### Why

Everything downstream depends on three structural decisions that must be
validated with real audio immediately: the stage trait, the pull-based
controller/processor split, and the source-supply contract. A pull engine
demands a ratio-dependent, variable amount of source per output block — this
inversion is the single biggest unknown in the design and has to be settled
before any DSP is layered on. A tape-mode chain (source → varispeed → out,
pitch follows tempo) is genuinely useful DJ behavior on its own and gives an
audible end-to-end result in the desktop app in the first stage.

### Primary Files

- New: `src/engine/stage.rs` (trait), `src/engine/graph.rs`,
  `src/engine/control.rs` (mailbox), `src/engine/source.rs`,
  `src/engine/stages/varispeed.rs`, `src/engine/profiles.rs`
- Reused in place: `src/core/resample.rs` (`StreamingSincResampler`),
  `src/core/ring_buffer.rs`
- Integration: `desktop/src/audio_engine.rs`, `desktop/src/processor.rs`
- Tests: new-engine ports of `tests/realtime_allocations.rs`,
  `tests/streaming_latency.rs`, `tests/modulation_torture.rs` (tape subset)

### Work

- `Stage` trait: fixed-block `process`, `latency_frames()`, `reset()`, and a
  `prime(history)` hook reserved for Stage 5. Construction-time invariant
  enforcement; debug-only assertions in the hot path; no `Result`, no
  allocation after construction.
- `EngineController` / `EngineProcessor` split with an SPSC timestamped
  parameter mailbox. `process(&mut self, out: &mut [f32])` fills exactly N
  frames; a block scheduler adapts caller callback sizes (64–1024) to the
  internal fixed block.
- `Source` contract: host-filled ring with explicit occupancy guarantees and
  underrun policy, plus absolute source-position bookkeeping. Port the
  resampled↔source timeline mapping idea from `RatioMapFifo`
  (`src/stream/processor.rs`) — it is proven and allocation-free.
- Varispeed stage wrapping `StreamingSincResampler` with sample-accurate
  retargets (no glide on the tempo axis — inherited from old Stage 15).
- Engine-agnostic **A/B QA adapter** in the qa harness commons: drives either
  the old `StreamProcessor` or the new engine over identical fixtures, so
  every later stage measures new-vs-old directly.
- Desktop wires the new engine behind the existing engine-selector UI pattern
  and goes pull-native for this path (no push-compat shim — the desktop is
  the reference integration and must surface API problems, not hide them).

### Exit Criteria

- Desktop audibly plays a real track through the new engine at 0.5–2.0x
  tempo in tape mode (seek excluded until Stage 5).
- Ported allocation gate: zero heap activity in steady state under
  per-callback tempo retargets (counting-allocator pattern from
  `tests/realtime_allocations.rs`).
- Ported latency gate: first-sample-out equals the reported latency exactly;
  control-to-audio ≤ resampler lookahead + one caller block.
- Tape-mode modulation torture (nudge/ride/snap): clicks = 0.
- Old-engine CI suites untouched and green.

## [x] Stage 2: Keylock Chain — Band Split and High-Band Small-FFT PV Corrector

Automation: auto

> **Completed 2026-07-12** (commit `0033214`). Chain:
> `src/engine/stages/{band_split,delay,pv_corrector,keylock}.rs`, composed
> as the `Keylock` profile; PV at FFT 512 / hop 128, identity locking,
> transposition delay-matched via `TimelineMap::rate_at` through the new
> `StageCtx`; constant pipeline delay 560 frames = **12.7 ms** at 44.1 kHz
> (≤ 15 ms budget). Desktop gains a "Pull — Keylock" deck.
> Measured gates (all green, `qa/engine_keylock.rs` +
> `tests/engine_{latency,modulation_torture,realtime_allocations}.rs`):
> cents wobble on the ±8%/2 s ride **p95 5.1 / max 5.2** vs old Live
> baseline 12.2 / 27.9 (and vs gates ≤ 15 / ≤ 22); steady-ratio residual
> 0.05 cents; unity band re-summation +0.00 dB / 0.0 % envelope after
> fixing a real defect found by the seam fixture — the legacy
> `LR8Crossover` cascades four Q=0.707 sections and notches −6 dB at the
> crossover; the new chain uses the corrected `LinkwitzRiley8` (proper LR
> Qs 0.5412/1.3066, true allpass re-sum) while the frozen multi-res engine
> keeps the old filter and its baselines. Keylock torture clicks = 0;
> zero-alloc steady state holds; control-to-audio unchanged (the low band
> additionally carries the LR8's ~5.5 ms dispersive group delay at DC —
> filter physics, documented in the gate).
>
> **Falsification experiment — result: the bet holds.** Metric half: a
> pure tone exactly at the 150 Hz seam at rate 1.06 re-sums at −3.0 dB
> with 39 % envelope beating — the expected power-sum of two copies
> detuned by the un-keylocked low band; narrow-band by construction (LR8
> overlap). Gated as a regression envelope, not a defect. Listening half
> (owner, 2026-07-12, A/B pairs new two-band vs old full-band keylock on
> bass-heavy material at ±8 % and ±20 %, rendered by the ignored
> `engine_keylock` test into `target/keylock_falsification/`): **the new
> keylock sounds better — less bit-crushing / artifact sound** than the
> old full-band correction; the un-keylocked low band was not the
> audible problem, the vocoder-processed sub was. The ~150 Hz cutoff is
> kept (final settlement with corpus evidence stays a Stage 7 item); the
> period-tracking low-band-corrector fallback is not needed.

### Why

This stage makes the architectural bet audible: split at ~150 Hz, leave the
low band un-keylocked, and pitch-correct only the high band with a small PV
at a small transposition. It either validates or falsifies the ≤ 15 ms
budget, and the falsification experiment — is the un-keylocked low band
really inaudible? — must run before SOLA, artifacts, and deck semantics are
stacked on top. The previous roadmap's Stage 16 (causal low-end at small
FFTs) dissolves here: sub-bass no longer passes through a vocoder at all.

### Primary Files

- New: `src/engine/stages/band_split.rs`, `src/engine/stages/pv_corrector.rs`,
  `src/engine/stages/delay.rs`
- Reused in place: `src/core/crossover.rs` (LR split machinery),
  `src/stretch/phase_vocoder.rs` (streaming mode, identity phase locking,
  `set_smooth_ratio_updates`), `src/stretch/phase_locking.rs`
- QA: ports of `qa/varispeed_keylock.rs` (cents-wobble gate) and the band
  similarity rows of `qa/profile_quality.rs`, via the A/B adapter

### Work

- Two-band split stage at ~150 Hz (a tuning constant; settled with evidence
  in Stage 7). Low band routes through a pure delay stage matched to the
  high-band corrector's latency so the bands re-sum in time.
- PV corrector stage: small FFT (settle 384/512/768 empirically against the
  budget), identity phase locking on, transposition driven by the
  delay-matched ratio map (ported from old Stage 15's mechanism).
- Latency accounting through the graph, inheriting the
  `StreamLatencyReport` split: constant pipeline delay (host-compensated) vs
  control-to-audio.
- **Falsification experiment (documented in this file when run):** A/B the
  un-keylocked low band on bass-heavy corpus material at ±8% and ±20%;
  record the audibility finding and the chosen cutoff. Named fallback if it
  fails beyond DJ ratios: a period-tracking low-band corrector as a stage
  swap — not a redesign.

### Exit Criteria

- Pitch wobble under the ±8%/2 s torture ride ≤ the old varispeed path on
  the ported cents gate (old Live baseline: p95 ≤ 15 / max ≤ 22 cents).
- Total pipeline delay ≤ 15 ms at 44.1 kHz, measured by the ported latency
  test; control-to-audio unchanged from Stage 1.
- Band re-summation artifact-free on a crossover-seam fixture (no dip or
  phase notch at the split beyond gate).
- Clicks = 0 under modulation torture with keylock engaged; zero-alloc
  steady state holds.

## [x] Stage 3: Beat-Synchronous SOLA Corrector and Corrector Selection

Automation: auto

> **Completed 2026-07-13** (commits `e93e5a5`, `db24c87`, `ffd370a`;
> owner listening passed: sub-threshold "sounds good", nudge bass loss
> fixed, muffling beyond threshold resolved by the threshold raise).
> `src/engine/stages/sola.rs`: elastic ring read at rate T with sinc
> interpolation; correlation-matched splices (raised-cosine, mono decision
> for stereo lockstep) at low-energy positions via an online energy gate;
> runs at the chain's shared 560-frame nominal lag so corrector handoff is
> latency-neutral (the splicer itself needs only the ~0.5 ms sinc margin,
> satisfying the ≤ 1 ms criterion). Selection in the keylock stage:
> hysteresis 4.5%/5.5% around the provisional 5% threshold, ~93 ms dwell,
> both correctors always warm; PV→SOLA engages via silent hard-recenter,
> SOLA→PV releases via a recentering splice, and both directions fade
> (48 frames) only at a measured phase-aligned instant (96-sample
> correlation gate, ~250 ms force backstop).
> Measured gates (all green): kick transient sharpness at ±4% — **new
> 1.21/1.18 vs old 0.68/0.74** (source 1.5) — the SOLA path preserves
> attacks ~70% sharper than the old engine; threshold-crossing torture
> clicks = 0 at 3×/1.3× theoretical slew; zero-alloc steady state across
> handoffs. Cents wobble: SOLA-band ±4% ride **p95 3.3 / max 3.6** (new
> gate); the ±8% ride — which now crosses the corrector threshold twice a
> second, a pathological selection gesture — measures p95 12.9 / max 16.5
> vs old 12.2 / 27.9 (absolute gates 15 / 22 still clear; threshold and
> hysteresis tuning is a named Stage 7 experiment).
> **Threshold settled early (2026-07-13, owner-reported):** beyond the 5%
> corrector threshold the deck sounded muffled — measured: the PV at FFT
> 512 loses ~2–4 dB in the top octave (multitone probe, 12–16 kHz, −3.8 dB
> at rate 1.10) while SOLA measures flat (−0.2 dB). The SOLA/PV threshold
> is raised from 5% to **9% rate deviation** (engage 8.5% / release 9.5%,
> now measured in rate space — transposition space is asymmetric), so the
> time-domain path covers the full DJ range (±8% with margin) and the PV
> serves only extreme rates, where Stage 5's corrector fade-out will apply
> anyway. Ripple effect: the ±8% cents torture ride is now all-SOLA and
> improved from p95 12.9 / max 16.5 to **p95 6.0 / max 6.3** (old engine
> 12.2 / 27.9). Final corpus settlement stays a Stage 7 item; improving
> the PV's top-octave fidelity (window/locking tuning) is on the Stage 7
> list too.
>
> **Post-landing fix (2026-07-13, owner-reported):** tempo nudges on the
> keylock deck thinned the bass. Cause: SOLA's elastic drift parks after a
> gesture, leaving the high band time-shifted against the low band's fixed
> delay — the crossover overlap (~100–300 Hz: kick body, bass harmonics)
> comb-filters indefinitely (reproduced at −6.7 dB on a seam tone under a
> dominant mid tone; a pure seam tone self-aligns and hides it). Fixed:
> splice correlation window widened to 2+ periods at the band edge (320)
> so splices align seam content too; distance-penalized tie-breaking
> (periodic content otherwise parks drift at the search edge); and
> sustained-rest drift bleed — a clean recenter splice for parked drift
> > 48 frames plus a sub-JND read-rate trim, gated on ~150 ms of rest so
> unity crossings mid-ride stay pure. Seam recovery after a nudge:
> **+0.00 dB** (was −6.7); all cents/torture/sharpness gates green.
>
> **Known issue found and pinned during gating:** the streaming PV sags
> up to −3 dB for ~1 s after fast unity-crossing transposition rides
> (envelope-gated in `pv_corrector` unit tests; the old engine measures
> **−10 dB / 12.7 dB swing** on the identical fixture, so this is an
> inherited, already-improved defect). Threshold-ride envelope gate set at
> 4 dB (measured 2.7) until the PV fix lands (Stage 4 phase resets /
> Stage 7 tuning are the levers). Beat-snapped splice positions arrive
> with the artifact cursor in Stage 4, per plan.

### Why

At DJ transpositions (< ~5%) a time-domain SOLA corrector is near-zero
latency and transparent on transients — this is what makes the engine feel
like Pitch 'n Time rather than a vocoder. The PV from Stage 2 remains the
wide-range corrector; this stage adds the transparent narrow-range path and
the selection/handoff machinery, which is where the quality of the whole
design is won or lost.

### Primary Files

- New: `src/engine/stages/sola.rs`; selection/handoff logic in
  `src/engine/graph.rs` or the corrector stages
- Reused in place: `src/stretch/wsola.rs` (splice search and crossfade
  machinery — adapted, not the offline driver)
- QA: transient-sharpness metrics from `qa/streaming_quality.rs`; a new
  threshold-crossing (handoff) torture fixture

### Work

- SOLA stage: fixed-block splice-based pitch correction at small
  transposition; splice candidates constrained to low-energy positions
  (online energy heuristic first; artifact guidance deepens in Stage 4).
- Corrector selection by transposition magnitude with hysteresis and a
  crossfaded handoff, so riding through the threshold mid-gesture is
  inaudible.
- The ~5% threshold is an explicit tuning constant with a QA experiment
  attached (provisional here, settled in Stage 7).

### Exit Criteria

- Kick/hat transient sharpness at ±4% tempo ≥ the old engine on the A/B
  adapter's streaming-quality transient metrics.
- Threshold-crossing torture (repeated rides through the SOLA/PV boundary):
  clicks = 0, no audible mode-switch signature (spectral-discontinuity gate).
- SOLA path adds ≤ 1 ms latency; selection adds zero allocation.

## [x] Stage 4: Artifact-First Control — Transient Protection and Splice Guidance

Automation: auto

> **Status (2026-07-13):** implementation landed; all measured exit
> criteria green. `src/engine/stages/transient.rs`: artifact cursor
> mapping onsets/beats track → ring (via a producer-side seqlock anchor,
> `SourceProducer::set_track_position` — seeks and loop wraps re-anchor
> without engine resets) → stage frames (via the inverted varispeed
> timeline map, exact under tempo rides; positions re-mapped every block).
> `StageCtx` carries the events plus a graph-level modulation-hold policy.
> Consumption: SOLA splice-candidate search excludes fades overlapping an
> onset protection window and takes pending splices opportunistically in
> the post-hit masked window; PV fires strength-gated per-band phase
> resets (inherited 0.45 low-band gate; upper bands always; exactly once
> per onset under re-mapping) with a flux-based online fallback when no
> artifact is attached. Desktop pull deck passes the analyze-on-load
> artifact and anchors start/seek/loop-wrap.
> Gates: onsets fire exactly once at mapped positions (cursor +
> `resets_fired` unit gates); artifact-guided transient sharpness ≥
> online (1.20 vs 1.13 at rate 0.96; equal at 1.04); no-artifact fallback
> ≥ old engine online path (1.15/1.13 vs 0.68/0.74); zero-alloc steady
> state with a 4 000-onset artifact attached. Owner listen passed
> (2026-07-13).

### Why

The `PreAnalysisArtifact` already exists, is produced by the CLI and desktop
on load (old Stage 14), and is strictly better than online detection. This
stage makes it the engine's primary control signal — artifact-driven PV
phase resets, SOLA splices snapped to onsets/beats, transient events
protected across both correctors — with the online scheduler as the fallback.
This is where the new engine starts beating the old one, not just matching
it.

### Primary Files

- New: `src/engine/stages/transient.rs`; artifact cursor in
  `src/engine/control.rs`
- Reused in place: `src/core/preanalysis.rs`,
  `src/stream/transient_scheduler.rs` (online fallback), `src/analysis/*`
  (front end unchanged)
- Tests: ports of `tests/streaming_preanalysis.rs` and
  `tests/preanalysis_pipeline.rs` semantics

### Work

- Map the artifact timeline through the varispeed ratio map so onset
  positions stay sample-accurate under tempo rides (mechanism proven in the
  old engine).
- PV corrector consumes artifact-scheduled per-band phase resets
  (strength-gated, inheriting the old `ARTIFACT_*_RESET_STRENGTH` tuning);
  SOLA chooses splice points from artifact onsets and beat positions.
- Online-detection fallback for artifact-less streams, with its own explicit
  parity gate.
- Re-express modulation-hold (suppress resets during fast control gestures)
  as a graph-level policy instead of processor-internal latches.

### Exit Criteria

- Artifact-driven transient preservation ≥ online detection on the ported
  preanalysis parity tests; onsets fire exactly once at mapped source
  positions under a tempo ride.
- No-artifact fallback ≥ the old engine's online path on transient metrics.
- Zero-alloc steady state with a large artifact attached (ported gate).

## [x] Stage 5: Deck Semantics — Warm Start, Loops, Timestamped Control, Rate Coverage

Automation: auto

> **Status (2026-07-13):** implementation landed; all measured exit
> criteria green. Warm-start seek: host protocol is reset → re-anchor →
> `EngineController::warm_start(preroll)` → push preroll + content; the
> processor primes the whole graph budgeted per callback (~1 ms CPU,
> exact stop at the mapped preroll boundary) and declick-fades in —
> post-seek level steady immediately, allocation-free (new gates in
> `engine_realtime_allocations`/graph tests). Loop wraps stay the Stage 4
> anchor mechanism: gapless and click-free across ratios (new gate).
> Timestamped control: `set_tempo_rate_at(rate, output_frame)` lands
> **exactly** on the requested output sample via capped resampler
> emission (`process_into_capped`, additive) — gated at a non-aligned
> frame. Keylock fades to plain varispeed over 12–22% rate deviation
> (old Stage 13 semantics); 48/96 kHz hold the cents gates; 0.02%
> fader steps are click- and drift-free. Reverse/scratch: the varispeed
> range is forward-only [0.25, 4.0] by design — hosts implement reverse
> by feeding reversed source and scratch re-entry as a warm-start seek
> (documented). Desktop: seeks warm-start (no more cold mute), and the
> deck DEFAULTS to Pull — Keylock (old engine one toggle away). Owner
> listen passed (2026-07-13): seeks resume instantly.

### Why

A DJ deck jumps constantly. The old engine's warm-start seek and gapless loop
wrap (old Stage 11) are hard-won semantics the new engine must reproduce, and
the pull architecture makes them cleaner: priming is "run history through the
graph, discard output" via the `prime` hook reserved in Stage 1.
Sample-accurate timestamped control completes the RT-first API. This stage
also absorbs the old Stage 13 completeness items.

### Primary Files

- `src/engine/control.rs`, `src/engine/graph.rs`, `src/engine/source.rs`
- `desktop/src/app.rs`, `desktop/src/processor.rs` (seek/loop/beat-jump
  rewired to the new engine)
- Tests: port of `tests/warm_start.rs`

### Work

- Warm-start seek: graph-wide prime-from-history, control state (targets and
  in-flight values) preserved, declick fade; loop-wrap equivalent of
  `notify_source_jump` (timeline re-anchor, no state reset).
- Timestamped control events applied at exact sample offsets within a block.
- Old Stage 13 items on the new engine: keylock at extreme rates = fade the
  corrector out toward plain varispeed (trivially clean in this
  architecture); reverse playback and scratch re-entry documented and
  tested; 48/96 kHz verified end-to-end; 0.02% pitch-fader steps artifact-
  and drift-free.
- Desktop defaults to the new engine from this stage (old engine one toggle
  away).

### Exit Criteria

- Ported warm-start tests pass: post-seek output at steady level
  immediately, loop-wrap seams click-free across ratios, allocation-free
  warm start.
- A timestamped tempo step lands on the requested output sample exactly
  (new test).
- 48/96 kHz produce equivalent gated metrics to 44.1 kHz.

## [x] Stage 6: WCET Flattening and Callback Budget Gates

Automation: auto

> **Completed 2026-07-13.** Both exit criteria machine-verified:
> `qa/engine_wcet.rs` (in the CI quality-gates job, strict + 2.0 hardware
> multiplier) measures per-callback wall/audio ratios across profiles and
> callback sizes 64–1024 under the full deck workload — threshold-riding
> gesture, artifact attached, a mid-run warm-start seek so priming
> callbacks are counted. Local reference measurements: keylock p99.9 ≤
> **0.20** at 64-frame callbacks (bound 0.5), tape ≤ 0.02; means 0.045 /
> 0.008. Flattening applied: the PV corrector renders at most ONE
> analysis+synthesis hop per block by construction (catch-up drains at 4×
> input rate inside the fixed window capacity), and warm-start priming is
> budgeted at 2× the callback's own size (clamped 256–2048) so seek
> priming stays inside the p99.9 bound at every callback size. Hot-path
> audit: the only non-debug assertion in engine code is a construction-
> time invariant (`SolaCorrector::new` ring sizing); all hot-path
> validation is debug-only, loops are const-bounded or guard-protected.
> No regression on any Stage 2–5 gate (cents, sharpness, seam, torture,
> latency, allocations re-run green).

### Why

Average-case CPU is already fine; a deck dies on the worst case. The old
engine's gate-then-render design bunches several FFT hops (occasionally an
8k-point sub-bass FFT) into single callbacks. The stage graph makes work
spreading tractable for the first time, and it must land before the parity
campaign so quality tuning happens under the real compute contract.

### Primary Files

- `src/engine/graph.rs` (hop scheduling), `src/engine/stages/pv_corrector.rs`
- New WCET gate extending the callback-budget pattern from
  `qa/streaming_quality.rs`

### Work

- Spread FFT hops: per-callback work bounded by construction (at most one
  analysis+synthesis hop per band per callback), verified across caller
  block sizes 64–1024.
- Callback budget gate measuring p99.9 and max (not mean) per profile, in CI.
- Audit remaining hot-path branches: debug-only validation, const-bounded
  loops (the old `LOOP_GUARD_SLACK` culture carries over).

### Exit Criteria

- Measured worst-case callback ratio (processing time / callback duration)
  under a hard bound — proposed ≤ 0.5 at 64-frame callbacks on the CI
  reference machine — with p99.9 gated.
- No latency or quality regression on the Stage 2–5 gates.

## [x] Stage 7: Parity Campaign — Tuning, External Evidence, Quality Sign-Off

Automation: manual

> **COMPLETE — owner sign-off recorded 2026-07-14.** Exit criteria:
>
> - New engine ≥ old on every gated A/B matrix row: **9/9** (cents rides
>   0.23/0.57 vs old 1.86/12.19, sharpness 1.16/1.29 vs 0.71/0.74, HF
>   −0.41 vs −0.79 dB, envelope 0.03 vs 10.36 dB, clicks 1.00 vs 2.40,
>   level −0.12 vs −0.29 dB).
> - External-reference comparison REQUIRED in CI and passing
>   (`qa/rubberband_reference_gate.rs`: spectral 0.965/0.970 vs the
>   ≥ 0.85 gate against Rubber Band CLI renders of the public corpus).
> - Owner listening sign-off: structured checklist passed on the pull
>   keylock deck — kicks, hats, vocals, sub bass, and full mixes across
>   0.92–1.08 and ±20% (2026-07-14, after the keylock range settlement
>   below; the ±20% edge carries audible-but-single-pitch splice
>   granularity, accepted as the specified behavior). Both previously
>   unscored public-corpus tempos ear-verified (Saucers 126, Arf 125)
>   and now scored in the BPM harness.
>
> Tuning constants settled: low-band cutoff 150 Hz (kept — Stage 2
> falsification verdict stands), SOLA/PV threshold = the fade end (SOLA
> owns the entire corrected range; three-step listening history in
> `src/engine/stages/keylock.rs`), PV FFT 512 (moot in the live path —
> the PV is never audible and is queued for removal at Stage 8/9).
> **Stages 8 and 9 are unlocked.**

> **Machine-side prep complete (2026-07-13):**
>
> **RubberBand anomaly EXPLAINED.** The historic ~−24 LUFS /
> ~0.15-similarity "harmonic" rows are the old hybrid batch driver
> (`src/stretch/hybrid.rs`, engaged whenever a preset is set) attenuating
> chirp content ~28 dB uniformly at any ratio — bisected: independent of
> HPSS/elastic/beat-aware/envelope/multi-res; `preset = None` with
> identical fields renders healthily (RMS 0.49 vs 0.02); `normalize=true`
> masks it. The defect retires with the hybrid at Stage 9. Confirmation:
> the comparison harness gained `TIMESTRETCH_RUBBERBAND_ENGINE=new`
> (pull-keylock render, DJ ratios only) — at 1.05×, edm_mix scores
> spectral **0.972 new vs 0.950 old**, and the sweep **0.954 new vs
> 0.154 old**. External comparisons are trustworthy for the new engine.
>
> **A/B matrix live** (`qa/engine_ab_matrix.rs`): one machine-readable
> report (`ab_matrix.csv` + parity verdicts) covering cents rides
> (±8/±4/steady), transient sharpness (±4%), top-octave retention,
> envelope stability under crossing rides, click ratios, and corpus level
> integrity, across old-Live / old-Club / new-keylock. First run: new
> wins 6 of 9 rows (wide-ride cents 6.0 vs 12.2, sharpness 1.15 vs 0.71,
> clicks 1.0 vs 2.4, steady cents 0.09 vs 0.64…); three tuning targets
> for the campaign: ±4%-ride cents (3.4 vs 1.9 — both sub-JND), top
> octave at 1.08 (−1.7 vs −0.8 dB, splice-phase scatter), envelope under
> ±11% crossing (11.8 vs 10.4 dB — both dominated by the pinned PV sag).
>
> **2026-07 campaign closed all three losing rows — the matrix is now 9/9
> for the new engine.** (1) Ride cents: root cause was the StageCtx
> embedded rate being read at the ring ingest position instead of the
> pipeline-latency-matched consumption position — a ride led the
> corrector by 560 frames (≈ latency × slope ≈ 2.8 cents at ±4% max
> slope). Delay-matching the read (plus a slope-tracked SOLA
> transposition that compensates the elastic drift term, and the rate
> history the timeline must now retain) took ride4 to **0.23 vs old
> 1.86** and ride8 to **0.57 vs old 12.19**. (2) Top octave: two causes —
> the 32-tap SOLA read kernel drooped ~1.5 dB in the top octave (now a
> dedicated 64-tap Kaiser table, flat past 0.9× Nyquist), and integer
> splice quantization scattered HF phase at every splice (now a
> sub-sample fine search over the fade window, sinc-interpolated at 1/8
> grid + parabolic vertex). hf_retention at 1.08: **−0.41 vs old
> −0.79 dB**. (3) Envelope under crossing rides: the PV's accumulated
> unity-crossing sag is now flushed (`flush_streaming_pipeline`,
> artifact bookkeeping preserved) whenever the ride settles near unity
> while SOLA alone is audible, with a release-handoff cooldown while the
> PV re-primes. env_swing at ±11%: **0.65 vs old 10.36 dB** — the sag no
> longer reaches the output at all on this fixture.
>
> **Public corpus + external evidence landed (2026-07-14).** The
> redistributable DJ corpus is defined (9 CC0/CC-BY/CC-BY-SA netlabel
> club tracks from the Internet Archive, SHA-256-pinned via
> `scripts/fetch_public_corpus.sh`; 7 scored with independently verified
> integer tempos, 119–147 BPM) and CI fetches it and runs the BPM
> harness on every push. The external-reference comparison is REQUIRED
> in CI: `qa/rubberband_reference_gate.rs` renders a public-corpus track
> through the keylock chain and gates against a Rubber Band CLI render
> (125→118 and 125→132 BPM; measured spectral 0.965/0.970 and level
> within 0.4 LUFS against gates of ≥ 0.85 and ≤ 1.5).
>
> **Keylock range settled (2026-07-14, owner listening on a full mix).**
> Three listening passes drove the selection/fade constants to their
> final shape: the PV was audibly phasey/robotic at every boundary it
> was placed behind (−9.7%, then −14.1%), and the old correction fade
> (starting at 12%) audibly doubled the top end — two pitches at
> complementary weights — long before its end. Final spec: **SOLA
> carries the entire corrected range; full single-pitch keylock through
> ±20% (the whole secondary DJ range); graceful release beyond**
> (fade 20.5%→35%, with SOLA's transposition clamp soft-flattening the
> corrected copy). Owner decision: this is the shipped scope line —
> CDJ-style wide-range (±100%) Master Tempo is explicitly out of scope
> for the parity campaign; a "wide keylock" deck profile (higher
> constant latency, big-FFT corrector) is recorded below as a
> post-cutover feature. The always-inaudible PV is now a removal
> candidate (chain WCET) at Stage 8/9.
>
> Remaining (manual): the structured listening checklist across the
> corpus, ear-verify the two unscored corpus tempos, and the owner
> listening sign-off.

### Why

This is the gate that authorizes deleting the old engine. It settles the open
tuning constants with QA evidence, runs the full A/B dashboard old-vs-new on
the DJ corpus, and absorbs the previous roadmap's Stage 8: a redistributable
public DJ corpus in CI, at least one mandatory external-reference comparison,
and the RubberBand harness anomaly explained before any comparison is
trusted.

### Primary Files

- `qa/*` (all harnesses via the A/B adapter; thresholds re-derived)
- `benchmarks/manifest.toml`, `scripts/compare_rubberband.sh`,
  `qa/rubberband_comparison.rs`
- `.github/workflows/ci.yml`

### Work

- Full A/B matrix: every gated QA row, old engine vs new, DJ corpus plus
  synthetic fixtures; machine-readable report surfaced in PRs.
- Settle and document the three tuning constants (low-band cutoff, SOLA/PV
  threshold, PV FFT size) with listening plus metric evidence.
- Investigate the ~-24 LUFS / ~0.15-similarity harmonic-track anomaly in
  `scripts/compare_rubberband.sh` first — benchmarks cannot be tightened
  until it is explained.
- Define the public, redistributable DJ corpus and promote one
  external-reference comparison to required in CI.
- Structured listening checklist: kicks, hats, vocals, sub bass, full mixes
  at 0.92–1.08 and ±20%; the sign-off is recorded in this file.

### Exit Criteria

- New engine ≥ old engine on every gated metric of the A/B dashboard; no
  row where the old engine wins by more than noise.
- External-reference comparison mandatory in CI on the public corpus and
  passing.
- Owner listening sign-off recorded here (the one deliberately manual gate
  in this roadmap).

## [x] Stage 8: Batch/Offline Rebase onto the Engine Graph

Automation: auto

> **Completed 2026-07-14.** `src/engine/offline.rs`: batch renders run on
> the pull engine graph — whole-file pre-analysis (artifact always
> present; a caller-provided one stays authoritative), constant-rate
> keylock for the DJ window (rate deviation ≤ 20%), a direct batch
> FFT-2048 PV for wide ratios, and **exact output length by
> construction** (`round(frames × ratio)`; latency prefix dropped
> structurally, source tail pushed through with a latency + kernel flush
> pad — no truncation heuristics). `stretch()`/`stretch_into()` rebased;
> per-channel hybrid and its mid/side mode retired (the graph processes
> channels in lockstep natively). `timestretch-cli` runs the new engine
> end-to-end with exact frame counts on both paths.
>
> Exit criteria: the external-reference gate now renders the frozen
> hybrid against the same Rubber Band reference — new batch **beats the
> captured baseline** (spectral 0.965 vs 0.933 at 125→118, 0.970 vs
> 0.950 at 125→132; assert new ≥ old − 0.02 in CI until the hybrid
> deletes at Stage 9). `tests/streaming_batch_parity.rs` replaced by
> `tests/streaming_offline_determinism.rs`: streaming and offline are
> **sample-identical** at equal rate/artifact, under irregular callback
> sizes. (The gate immediately caught a real engine defect: timeline
> eviction retained too little history for the transient cursor's
> look-behind, so onset mappings — and a SOLA splice — depended on the
> caller's callback sizes. Retention now covers the look-behind at the
> slowest rate.)
>
> Deferred to Stage 9 (the deletion stage): removing the never-audible
> PV corrector from the live keylock chain, and the old hybrid/stream
> code itself.

### Why

Single engine, both modes: offline is the same graph with unlimited lookahead
and a guaranteed artifact, which makes streaming-vs-batch parity true by
construction instead of a test suite. The hybrid engine is retired here, and
the old batch-quality ambitions (previous Stages 2–5) are formally closed.

### Primary Files

- New: `src/engine/offline.rs` (batch driver over the graph)
- `src/lib.rs` (batch API rebased), `src/cli.rs`
- Tests rebased: `tests/quality.rs`, `tests/spectral_quality.rs`,
  `tests/stretch_quality_regressions.rs`, `tests/timeline_length.rs`,
  `tests/bpm_stretch.rs`

### Work

- Batch driver: whole-file feed, artifact always computed, exact output
  length by construction (no post-render truncation/padding hacks); larger
  analysis lookahead where it measurably helps.
- Capture the hybrid engine's DJ-corpus baseline **before** deletion; rebase
  batch quality tests onto the new engine at parity-or-better (non-EDM
  regressions do not block, per the binding product boundary).
- Replace `tests/streaming_batch_parity.rs` with a same-graph
  streaming-vs-offline determinism test.

### Exit Criteria

- Batch DJ-corpus quality gates ≥ the captured hybrid baseline; exact-length
  output without truncation hacks.
- `timestretch-cli` runs on the new engine end-to-end.

## [x] Stage 9: Cutover and Deletion

Automation: auto

> **Completed 2026-07-15.** Deleted: `src/stream/` (processor +
> transient scheduler), `src/stretch/{hybrid,multi_resolution,stereo,
> wsola}.rs`, `src/analysis/adaptive_snapshot.rs`, the push API surface
> (`StreamProcessor`, `StreamingEngine`, `StreamProfile`, `ControlPath`,
> …), the frozen old-surface tests, and the old-engine QA harnesses.
> The live keylock chain dropped its PV corrector (SOLA carries the
> whole corrected range; the chain's constant delay is now its own
> `KEYLOCK_LATENCY_FRAMES`). Desktop runs pull-only (Legacy deck and
> legacy-only pitch-shift UI removed); CLI is batch-on-engine only.
> QA re-anchored: the A/B adapter collapsed to new-engine-only, the
> matrix gates on absolute thresholds derived from measured values, the
> torture envelope gate re-derived 13 dB → 1 dB (measured 0.03), and
> the external-reference gate dropped its hybrid-baseline arm. Version
> 0.8.0 with a breaking-change changelog. `grep` is clean of the
> removed API; all suites green.

### Why

Two engines is a temporary tax, not a feature. Once parity is signed off
(Stage 7) and batch is rebased (Stage 8), the old surface is deleted in one
stage, and with it the entire corrective-heuristic stack the new architecture
was built to obsolete.

### Primary Files

- Deleted: `src/stream/processor.rs`, `src/stretch/hybrid.rs`, the streaming
  pieces of `src/stretch/multi_resolution.rs`, the push API surface, and the
  frozen old-surface tests (`tests/streaming.rs`,
  `tests/streaming_edge_cases.rs`, `tests/stream_profiles.rs`,
  `tests/dj_workflows.rs`, `tests/realtime_dj_conditions.rs`, old-surface
  portions of `tests/edge_cases.rs` / `tests/public_api_workflows.rs`)
- Updated: `src/lib.rs`, `README.md`, `CHANGELOG.md`, `desktop/` (selector
  removed), `qa/` (A/B adapter collapses to new-engine-only)

### Work

- Remove old modules and API; version bump with a breaking-change changelog
  (pre-1.0; the DJ app is the customer).
- Prune/port remaining tests; re-derive every absolute QA threshold from
  new-engine measurements; delete the A/B adapter's old arm.
- Documentation pass: README latency table, RT contract, artifact workflow,
  pull-API examples.

### Exit Criteria

- Old engine fully deleted; `grep` clean of the removed API; CI green
  against re-baselined gates.
- Desktop and CLI run exclusively on the new engine.
- Net LOC substantially down (target: ≥ 8k lines removed).

## [ ] Stage 10: General-Purpose Beat Tracking and BPM Detection (parallel track)

Automation: auto

> **Status (2026-07-19): algorithm side landed; corpus and gates
> remaining.** Landed (commit `e0b2410` and follow-ups `36530c9`,
> `e0049ed`): `src/analysis/tempogram.rs` (autocorrelation tempogram with
> harmonic reinforcement, wide log-normal prior — no hard folding, the
> 100–160 EDM fold survives only as an optional soft `hint_range` — and a
> Viterbi tempo path that may drift or step), the DP beat tracker in
> `src/analysis/beat.rs`, tempo segments + downbeat indices in the
> artifact schema (`src/core/preanalysis.rs`), beat-level metrics in
> `qa/bpm_accuracy.rs` (F-measure ±70 ms, CMLt/AMLt-style continuity,
> downbeat F-measure) with the harness in CI on every push, and desktop
> consumption (grid marks in the waveform painters, grid-accurate beat
> jumps and loop spans in `desktop/src/app.rs`).
>
> Remaining before the stage closes — the evidence half of the exit
> criteria: `benchmarks/manifest.toml` still has **no non-EDM entries, no
> variable-tempo entries, and no annotated beat grids** (the harness
> supports annotations; the corpus doesn't exercise them). Needed: the
> widened corpus with annotated grids, the synthetic tempo-ramp fixture
> and at least one live-drummer recording, explicit acc2/F-measure floors
> enforced in CI on the non-EDM subset, and the owner check that desktop
> grids align on real non-EDM tracks.
>
> **External baseline settled (2026-07-19):** the QM beat tracker
> (qm-dsp's TempoTrackV2/DownBeat, as shipped in Mixxx) runs as a Vamp
> plugin via Sonic Annotator. It anchors Stage 10 the way the Rubber Band
> CLI anchors stretch quality: a baseline column in `qa/bpm_accuracy.rs`
> (ours ≥ QM's on the shared metrics) and machine-generated draft beat
> annotations for corpus tracks, hand-corrected into the manifest. We
> consume its *output* only — qm-dsp is GPL-2.0+ and this crate is MIT,
> so no code is ever ported from it; anything algorithmic we adopt is
> implemented from the underlying papers, with qm-dsp used solely to
> cross-check behavior.

### Why

The analysis front end's beat detector is EDM-only by construction: kick-tuned
onset sensitivity, BPM hard-folded into 100–160, and a constant 4/4 grid from
a single PLL pass. The DJ app needs trustworthy grids on everything a DJ
loads — hip-hop at 90, DnB at 174, older and live material whose tempo
drifts — and the deck UI needs to draw the grid on the waveform so a wrong
grid is visible instead of silently wrong beat jumps. The previous roadmap
parked a tempogram beat tracker in the backlog; this stage pulls it in as a
product feature. It touches only the analysis front end (inherited unchanged
by the engine work) and the desktop app, so it has **no dependency on Stages
4–9** and can be worked in parallel with the engine line. It does not move
the EDM/DJ-first *stretch-quality* boundary — non-EDM material must get a
correct grid, not a quality-gated stretch.

### Primary Files

- Rewritten: `src/analysis/beat.rs` (tempo model + tracker); new
  `src/analysis/tempogram.rs`
- Reused in place: `src/analysis/transient.rs` (spectral-flux novelty front
  end, widened from kick-tuned sensitivity)
- Schema: `src/core/preanalysis.rs` (beatgrid extended to segments +
  downbeats, artifact schema bump; coordinate with Stage 4's artifact
  cursor — whichever lands second adapts)
- API: `src/lib.rs` (`detect_bpm` / `detect_beat_grid` /
  `*_buffer` rebased onto the new model; pre-1.0, breaks freely)
- Desktop: `desktop/src/waveform.rs` (grid overlay),
  `desktop/src/app.rs` (beat-jump/loop on real grid positions)
- QA: `qa/bpm_accuracy.rs` (widened corpus + beat-level metrics),
  `benchmarks/manifest.toml`

### Work

- **Tempo model.** Replace the single-BPM `BeatGrid` with a piecewise model:
  fractional-sample beat positions, downbeat indices, per-segment BPM, and
  confidence. Constant-tempo tracks collapse to one segment; snap helpers
  (`snap_to_grid*`, subdivision grids) carry over on the new type.
- **Novelty front end.** Multi-band onset novelty built on the existing
  spectral-flux machinery, sensitivity generalized beyond kicks.
- **ODF experiments (gated, from the papers — no qm-dsp code).** Two
  candidate upgrades to the novelty front end, each accepted only if it
  moves beat F-measure on the non-EDM subset: a complex-domain detection
  function (Duxbury/Bello — fuses magnitude and phase prediction error;
  the known win is soft/tonal onsets, i.e. vocals and live material) as
  an alternative to the current flux + energy + phase-deviation combine;
  and adaptive whitening (Stowell & Plumbley — per-bin running-peak
  normalization ahead of the ODF) for dense mastered material. Both are
  small against the existing FFT machinery; if a gate doesn't move,
  the experiment is deleted, not kept as an option.
- **Downbeat feature A/B.** Compare the current bass-weighted accent
  feature against beat-synchronous spectral difference (Davies &
  Plumbley 2006) on annotated downbeats — different signal, sometimes
  wins where bass doesn't mark the bar (breakbeats, older funk). Keep
  whichever scores higher; don't maintain both.
- **Tempo estimation.** Autocorrelation/Fourier tempogram over the novelty
  curve; wide prior (~50–220 BPM); explicit octave-decision policy with
  confidence instead of hard folding. The 100–160 EDM fold survives only as
  an optional prior/hint for the DJ-app path.
- **Beat tracking.** Dynamic-programming beat tracker (Ellis-style DP or
  HMM/Viterbi) over the novelty curve with tempo allowed to vary along the
  tempogram ridge — this is what makes grids right on live and drifting
  material. The PLL quantizer is retired unless corpus evidence shows it
  winning on quantized EDM.
- **Downbeat estimation.** Bar-phase estimation from beat-synchronous
  features (energy/spectral pattern); 4/4 prior but confidence-marked, not
  hard-coded.
- **Artifact + analyze-on-load.** Artifact beatgrid carries the new model
  (schema bump + validation-gate update); CLI and desktop analyze-on-load
  produce it.
- **Desktop grid overlay.** Beat lines drawn on the waveform, downbeats
  emphasized, sample-accurate against the peaks buckets. Density-adaptive at
  the full-track view (bars/downbeats only until beat spacing clears a
  minimum pixel width). Beat-jump and loop-length math switch from fixed
  `60/bpm` intervals to real grid lookup.
- **QA.** Widen the corpus with non-EDM and variable-tempo entries
  (annotated grids in `benchmarks/manifest.toml`); add beat-level scoring —
  F-measure (±70 ms) and octave-tolerant continuity (CMLt/AMLt) — alongside
  acc1/acc2 in `qa/bpm_accuracy.rs`; keep the diffable JSON report.
  Bootstrap annotations via the QM Vamp baseline (draft grids from Sonic
  Annotator, hand-corrected into the manifest) and carry the QM scores as
  a baseline column in the report.

### Exit Criteria

- acc1/acc2 on the widened corpus ≥ the current detector's scores on the
  EDM subset (no regression where it already works), with explicit floors on
  the non-EDM subset (proposed acc2 ≥ 90%).
- Beat F-measure gate on annotated tracks (proposed ≥ 0.85), including a
  synthetic tempo-ramp fixture and at least one live-drummer recording
  tracked within tolerance.
- External baseline: ≥ the QM beat tracker (Vamp/Sonic Annotator renders)
  on beat F-measure and acc2 over the widened corpus — no row where QM
  wins by more than noise.
- Desktop draws the grid aligned on real tracks; beat jumps land on detected
  beats, not computed intervals; overlay stays responsive (no per-frame
  allocation churn in the paint path).
- Full-track analysis stays comfortably offline-budget (proposed ≥ 50×
  realtime on the CI reference machine).

## [x] Stage 11: Wide-Range Master Tempo Profile (post-cutover track)

Automation: auto

> **Falsification experiment — result: the bet holds (2026-08-04).
> Proceed to build-out.** Harness: `qa/wide_falsification.rs` via
> `./scripts/wide_falsification.sh` (renders + `summary.csv` in
> `target/wide_falsification/`, Rubber Band CLI 4.0.0 references incl.
> `--fine`, level-matched `norm/` copies for listening).
>
> Metric half, three rounds: the reset-driven FFT-2048 identity prototype
> holds 0.85–0.91 spectral / 0.82–0.88 perceptual vs Rubber Band at
> ±30/±50/+100 % with zero clicks. The −75 % blowup (+7 LUFS spurious
> energy, up to ~2 000 clicks/M) proved to be a **75 %-overlap artifact —
> hop = FFT/8 eliminates it on all five tracks** (zero clicks, normal
> level, RB similarity 0.83–0.89). FFT 4096 wins only on synthetic tones,
> is consistently worse on real mixes (transient smear), and its ~93 ms
> window is outside the profile's latency contract — dead end. Metrics
> twice failed to predict the ear verdicts (down-side robotiness scored
> *above* the clean up-side; blend 0.20 vs 0.40 metrically identical) —
> reconfirming owner listening as the binding gate.
>
> Listening half (owner, 2026-08-04, three passes; bass-heavy corpus
> msbwy / hot_stuff / somebody / saucers + the synthetic bass fixture,
> ~20 s excerpts at +30/−30/+50/−50/+100/−75 %): round 1 — all arms
> below Rubber Band; compressions clearly better than expansions, which
> sounded "roboty" (cause found: the phase-gradient coherence blend
> tapers to **zero** approaching ratio 2.5, so big slowdowns ran with no
> vertical coherence). Round 2 — blend held on + hop/8 rescued −30 %
> outright and kept −50 % together; residual robotiness remained. Round
> 3 — `blend_hop8` and `blend40_hop8` best in class and mutually
> near-indistinguishable (possible slight bass edge to 0.40, owner
> unsure). **Verdicts: −50 % is shippable ("the audience couldn't tell
> the difference, producers would"); +50 % is shippable (RB "cleaner,
> wider, more open," ours "a bit crowded," but subtle).** The RB gap is
> real and accepted: single-FFT identity PV vs R3's multi-resolution
> engine.
>
> Settled config for the build-out: FFT 2048, hop 256 (87.5 % overlap),
> Hann, identity locking, rigid sub-bass < 100 Hz, artifact-driven
> per-band resets (Stage-9 `begin_block` policy), coherence blend held
> at wide ratios via `PhaseVocoder::set_wide_ratio_coherence_blend`
> (strength 0.20–0.40; exact value settled by ear in build-out
> listening). WCET note: hop/8 doubles the hop rate — the flattening
> budget is one 2048-point FFT per 8 blocks per channel. Carried open:
> the wide-ratio low-band verdict (`widepv_lowfree` renders exist,
> not yet auditioned — exit criterion) and the ±100 %/−75 % edge
> listening characterization (metrically clean in the hop/8 family).
>
> **COMPLETE — build-out shipped, owner sign-off recorded 2026-08-04.**
> Live chain: `WideKeylockStage` (FFT 2048 / hop 256 / identity /
> full-spectrum, no correction fade, T clamp [0.5, 4.0] = the
> resampler's exact step range), artifact-only flux-gated per-band
> resets with a latency-delayed modulation-hold read, keylock-toggle
> crossfade, constant 2144-frame (48.6 ms) contract. Two mechanisms
> found and fixed by the gates during build-out: the resampler ramp's
> harmonic-mean consumption drifts stream balance by hop/2·ln(T₁/T₀)
> across sweeps (fixed: `set_step_anchor` — uniform chunks consumed
> uniformly), and instant full-range T steps tear the overlap-add seam
> (fixed: log-space transposition slew, full-range snap settles ≈30 ms).
> Metric half: strict WCET p99.9 0.328 vs 0.5 bound at 64-frame
> callbacks (no channel staggering needed — the named contingency stays
> unshipped); zero-alloc steady state + multi-callback warm start;
> torture rides/snaps click-free; wide matrix gates pinned (steady
> cents 0.09–0.50 p95 at rates 0.7–2.0; ride wobble 3.3 cents p95 at
> ±8 % / 38 at the full-range torture ride — the structural
> chunked-correction ceiling vs SOLA's per-sample elastic 0.5, a
> ±512-frame slope-lookahead scan confirmed zero offset optimal;
> synthetic-delta sharpness pinned loosely as big-window smear).
> Listening half (owner, 2026-08-04, live desktop deck): **keylock now
> holds beyond ±20 %** where the primary chain releases; range flip is
> a click-free seek-priced rebuild with the playhead preserved and the
> honest latency chip updating. **Low-band verdict (exit criterion):
> keylocked full spectrum — `widepv_resets` beat `widepv_lowfree` in
> the offline A/B; the stage is Corrected-only, no band split.**
> Coherence blend settled at **0.20** (live A/B: marginally better than
> 0.40, reversing the offline round-3 lean; both below metric
> resolution). Edge characterization at −75 %: pitch stays locked;
> bass reads bitcrushed/distorted, vocals robotic and raspy ("through
> a guitar amp") — documented degradation well outside the ±50 %
> shippable verdict, no silent cliff. The Rubber Band gap stands as
> accepted at the falsification: audibly behind R3 at all wide rates,
> shippable per the owner's call. Desktop ships the Range selector
> (Standard | Wide); README now carries the per-profile latency matrix.

### Why

Promoted from "Not a Priority Yet": the owner decision of 2026-07-14
deferred CDJ-3000-class wide-range keylock until after cutover, and cutover
completed 2026-07-15. The shipped keylock is single-pitch through ±20% with
graceful release beyond — correct for the primary deck path, but a real
competitive gap against hardware Master Tempo. Wide range wants a big-FFT
corrector (1024–2048) and therefore ~25–50 ms of constant latency, which is
structurally incompatible with the ≤ 15 ms budget — so it ships as a
separate user-selectable deck profile with its own honestly-reported
latency contract, like a CDJ range setting. The primary keylock chain is
untouched.

**Known risk, validated first:** Stage 7's listening passes found the
small-FFT PV audibly phasey at every boundary it was placed behind — that
is why SOLA owns the whole corrected range today. A wide-range corrector is
a *new* big-FFT PV with identity locking and artifact-driven phase resets
at a large transposition; whether it can sound acceptable is the
falsifiable bet of this stage, and it runs first, against CDJ/RubberBand
references, before any deck plumbing. Named fallback if it fails: keep the
profile unshipped and the scope line at ±20% — this stage is severable.

### Primary Files

- New: `src/engine/stages/` wide corrector stage; a `WideKeylock` profile
  in `src/engine/profiles.rs`
- Reused in place: `src/core/fft.rs`, the artifact reset machinery from
  `src/engine/stages/transient.rs`, band split from
  `src/engine/stages/band_split.rs`
- Integration: desktop profile selector (`desktop/src/app.rs`,
  `desktop/src/deck.rs`)
- QA: cents-wobble and transient-sharpness rows extended to ±30–100%;
  WCET gate for the big-FFT chain in `qa/engine_wcet.rs`

### Work

- Falsification experiment first: batch-render bass-heavy corpus material
  at ±30/±50/±100% through a prototype big-FFT corrector; A/B against
  Rubber Band renders and the plain-varispeed release; owner listen
  recorded here before the profile is built out.
- Wide corrector stage: big FFT (settle 1024/2048 empirically), identity
  phase locking, artifact-driven per-band phase resets, low band handled
  explicitly (the un-keylocked-low-band bet does not extend to ±100% —
  re-run the Stage 2 experiment at wide ratios and pick keylocked vs free
  per the evidence).
- Profile plumbing: separate constant-latency contract through
  `latency_frames()`; profile switching via the existing warm-start
  mechanism (no live morph between profiles — a switch is a seek-priced
  event, documented).
- WCET flattening for the big-FFT chain (at most one hop per callback, as
  Stage 6 established for the small chain).

### Exit Criteria

- Single-pitch keylock at ±50% on the structured listening checklist
  (kicks, vocals, full mixes), with the ±100% edge characterized and its
  behavior documented — no silent quality cliff.
- Cents gates hold at wide ratios; WCET p99.9 bound holds for the wide
  chain at 64-frame callbacks; zero-alloc steady state.
- Profile latency honestly reported; switching profiles is click-free via
  warm start.
- Owner listening sign-off recorded here, including the explicit verdict
  on the wide-ratio low-band question.

## [ ] Stage 12: Robustness Hardening — No-Panic Surface, Fuzzing, Soak (post-cutover track)

Automation: auto

### Why

The RT contract is machine-verified, but the crate's *input* surface has
never been hardened: the artifact JSON loader (`src/core/preanalysis.rs`),
the WAV reader (`src/io/wav.rs`), and the public batch API have no fuzz
coverage, and there is no enforced policy that arbitrary input produces
`Err`, never a panic. For a library embedded in a shipping app — and a
prerequisite for any 1.0 — "does not panic on hostile or degenerate input"
must be a tested property, not an intention. This stage touches no DSP.

### Primary Files

- New: `fuzz/` (cargo-fuzz targets), a soak harness in `qa/`
- Audited in place: `src/core/preanalysis.rs` (JSON load path),
  `src/io/wav.rs`, `src/lib.rs` (param validation), `src/error.rs`,
  engine constructors in `src/engine/`
- CI: `.github/workflows/ci.yml` (bounded fuzz on PRs, longer cron run)

### Work

- Fuzz targets: artifact JSON from arbitrary bytes; WAV parsing from
  arbitrary bytes; the batch `stretch()` API driven by arbitrary params ×
  degenerate audio (NaN/Inf/denormal samples, zero-length, one sample,
  extreme rates and sample rates).
- No-panic policy: every public entry point returns `Err` on invalid
  input; document the policy in `src/lib.rs`; `debug_assert` stays the
  hot-path idiom (construction-time invariants are the release-mode
  guard, per the Stage 1 contract).
- Long-run soak: hours-equivalent of randomized deck gestures (tempo
  rides, seeks, loop wraps, profile/keylock toggles, artifact swaps)
  gated on zero clicks, zero allocation, bounded drift — the
  torture-test generators already exist, this composes them.
- CI wiring: short bounded fuzz per PR; scheduled longer run with corpus
  persistence; crashes minimize into regression tests.

### Exit Criteria

- All fuzz targets run clean for the CI budget; every crash found during
  the campaign lands as a minimized regression test.
- Machine-checked: no panic reachable from the public API on arbitrary
  input (fuzz evidence + audit of `unwrap`/`expect`/`panic!` outside
  construction and tests).
- Soak harness green in CI (bounded variant) with the full-length recipe
  documented for local runs.

## Disposition of the Previous Roadmap's 16 Stages

| Old stage | Status then | Disposition now |
|---|---|---|
| 1 Fast modulation stability | Complete | Inherited as gates: the torture-test methodology and click/slew bounds port into new Stages 1–3. The deferred PV ratio-step seam is dissolved — the PV no longer implements tempo. |
| 2 Confidence-based blending | Open | **Cancelled** (hybrid retired). Event-mask ideas survive only as optional tuning levers in new Stage 4. |
| 3 Rolling adaptive analysis | Partial | Shipped loudness-robust onset front end inherited unchanged (it feeds artifacts). Rolling multi-res analysis and streaming-scheduler parity **cancelled** (artifact-first makes them fallback-only). Tempogram beat tracker → **Stage 10** (general-purpose beat tracking, parallel track). |
| 4 HPSS / residual paths | Open | **Cancelled** — HPSS is not in the target architecture. |
| 5 Continuous event shaping | Open | **Cancelled** as hybrid work; per-event descriptors (artifact strengths, band flux) already exist and are consumed in new Stage 4. |
| 6 Streaming pitch quality | Complete | Inherited: `StreamingSincResampler` is the new engine's varispeed and pitch stage. |
| 7 API contract hardening | Open | **Absorbed and inverted** into new Stages 1/6: the new API is strict by construction (invariants at build time, debug-only hot-path validation, no `Result` in the callback). Hardening the old surface is cancelled. |
| 8 External quality evidence | Open | **Carried over** into new Stage 7 (public corpus, mandatory reference comparison, RubberBand anomaly, machine-readable reports). |
| 9 Product boundary (EDM/DJ-first) | Complete | Inherited as binding policy. |
| 10 Low-latency streaming | Complete | Inherited: profiles become fixed chains; honest-latency-reporting culture and `tests/streaming_latency.rs` methodology port in new Stage 1. Old profile latencies superseded by the ≤ 15 ms budget. |
| 11 Warm-start seek/cue/loop | Complete | Inherited: semantics re-implemented on the graph in new Stage 5; `tests/warm_start.rs` ports as the gate. |
| 12 Port hybrid quality into stream engine | Open | **Superseded** — its goal (structural quality, shrinking heuristic stack) *is* this roadmap. The streaming multi-res engine is retired in new Stage 9. |
| 13 Deck control completeness | Open | **Absorbed** into new Stage 5 (reverse, scratch re-entry, 48/96 kHz, fader resolution) and Stages 2/5 (keylock extremes = corrector fade-out). |
| 14 Analyze-on-load pre-analysis | Complete | Inherited unchanged (artifact schema v2, CLI/desktop producers, validation gates); consumption re-implemented in new Stage 4. |
| 15 Varispeed-first keylock | Complete | **The foundation.** The control-path inversion, `RatioMapFifo` / delay-matched transposition, latency-report split, and `qa/varispeed_keylock.rs` all port into new Stages 1–2. The retrofit becomes the only architecture. |
| 16 Causal low-end at small FFTs | Open | **Dissolved** by the un-keylocked low band (new Stage 2). Multi-hop IF estimation → backlog, revived only if the Stage 2 falsification experiment fails beyond DJ ratios. |

## Migration Risks

1. **QA thresholds encode old-engine behavior.** Absolute gates (cents
   floors, similarity rows, budget baselines) were measured on the old PV.
   → Stage 1's A/B adapter; parity defined as new ≥ old on the same
   fixture/metric; thresholds re-derived at Stage 9.
2. **Metrics can reward the old heuristics.** Similarity-style metrics may
   score the corrective stack above structurally cleaner output. → Stage 7's
   scoreboard is source-referenced and external-referenced (clicks, cents
   wobble, transient sharpness, band similarity vs input, references), plus
   a recorded listening sign-off — never similarity-to-old-output.
3. **Two engines in CI: cost and drift.** → Wrap-don't-move policy; old
   suites run as a frozen CI job; the feature freeze means old baselines
   never move.
4. **Desktop coupling to the push API.** → Desktop goes pull-native in
   Stage 1 behind the proven engine-selector pattern; it is the reference
   integration, so no compat shim that would hide API problems.
5. **The pull `Source` contract is under-designed.** Varispeed makes
   per-block source demand variable; underruns, seeks, and loops all
   interact with the supplier. → Explicit, tested deliverable in Stage 1;
   revalidated under seek/loop in Stage 5.
6. **Freeze erosion during a long parity tail.** "Almost at parity" invites
   patching the old engine. → Freeze policy is binding from day zero;
   per-stage parity gates instead of one terminal gate; desktop defaults to
   the new engine from Stage 5 so dogfooding pressure lands there.
7. **Latency/quality bets failing late.** The ≤ 15 ms budget, the
   un-keylocked low band, and the SOLA/PV threshold could each be wrong.
   → All three falsification experiments are pinned to Stages 2–3, each
   with a named fallback that is a stage swap (period-tracking low-band
   corrector; wider SOLA range), not a rearchitecture.
8. **Artifact dependence degrading artifact-less streams.** → Stage 4
   carries an explicit no-artifact-fallback parity gate against the old
   engine's online path.

## QA and Test Gating Through the Migration

**Gate the new engine as-is (engine-independent or output-only):**

- `qa/reference_quality.rs`, `benchmarks/manifest.toml`,
  `scripts/compare_rubberband.sh`, `qa/rubberband_comparison.rs` — compare
  rendered audio to references; only the construction site changes (via the
  A/B adapter). Stage 7.
- `qa/track_analysis_qa.rs`, `qa/bpm_accuracy.rs`,
  `tests/dense_material_regression.rs` — the analysis front end is inherited
  unchanged; gate as-is throughout.

**Ported (coupled to the push API / `StreamProcessor` surface):**

- `tests/realtime_allocations.rs` — counting-allocator pattern ports
  verbatim; bodies rewritten for the pull API. Stage 1.
- `tests/streaming_latency.rs` — first-sample-out == report and
  control-to-audio methodology. Stage 1.
- `tests/modulation_torture.rs` — gesture generators and click/slew metrics
  reusable; control via the mailbox. Stages 1–3 (tape → keylock →
  threshold-crossing variants).
- `qa/varispeed_keylock.rs` — cents-wobble metric ports directly; the
  baseline becomes old-vs-new. Stage 2.
- `qa/streaming_quality.rs`, `qa/profile_quality.rs` — signal generators and
  metrics reusable; profile expectations re-derived for the ≤ 15 ms chains.
  Stages 2 and 7.
- `tests/streaming_preanalysis.rs`, `tests/preanalysis_pipeline.rs` —
  artifact-consumption semantics. Stage 4.
- `tests/warm_start.rs` — semantics. Stage 5.
- Batch suites (`tests/quality.rs`, `tests/spectral_quality.rs`,
  `tests/stretch_quality_regressions.rs`, `tests/timeline_length.rs`,
  `tests/bpm_stretch.rs`) — keep gating the old batch path until Stage 8,
  then rebase.
- `tests/streaming_batch_parity.rs` — retired at Stage 8, replaced by the
  same-graph streaming-vs-offline determinism test.

**Frozen with the old engine, deleted at Stage 9 (no port):**

`tests/streaming.rs`, `tests/streaming_edge_cases.rs`,
`tests/stream_profiles.rs`, `tests/dj_workflows.rs`,
`tests/realtime_dj_conditions.rs`, and the old-surface portions of
`tests/edge_cases.rs` / `tests/public_api_workflows.rs` — their intent is
covered by the per-stage new-engine gates.

## Not a Priority Yet

- ~~Wide-range Master Tempo~~ — **promoted to Stage 11** (2026-07-19; its
  post-cutover condition was met at Stage 9).
- **Musical key detection** (constant-Q chromagram + Krumhansl-style
  profile correlation — the GetKeyMode algorithm Mixxx ships). Table-stakes
  DJ-app feature for harmonic mixing and the natural next analysis feature
  after Stage 10: it reuses the analyze-on-load pipeline and artifact
  schema, and needs a chromagram front end this crate doesn't have yet.
  Implemented from the papers; qm-dsp (GPL-2.0+) is cross-check evidence
  only, never source (same policy as the Stage 10 baseline).
- SIMD and architecture-specific acceleration (revisit after Stage 6's WCET
  gates exist to measure it against)
- Desktop UI/UX polish beyond its role as the reference integration
- Additional presets, wider API surface, convenience wrappers
- General-purpose (non-EDM) material *stretch* quality (general-purpose
  analysis — beatgrid/BPM on any material — is Stage 10)

## Path to 1.0 (decision pending — not scheduled)

Everything above targets production grade **for the DJ app**, which is the
crate's binding customer. Production grade **as a public library** is a
separate, longer road that the pre-1.0 "API breaks freely" policy
deliberately defers. It does not get stages until the owner decides the
crate should take external customers; recorded here so the decision is
explicit rather than ambient:

- API freeze and semver discipline: audit the 19-function public surface,
  cut what only the DJ app needs, commit to the rest, 1.0 with a
  migration-noted changelog.
- Rustdoc completeness pass on every public item; the README latency
  table, RT contract, and artifact workflow kept as documented guarantees.
- The non-EDM *stretch quality* question answered one way or the other:
  either gated (a corpus and floors, like Stage 10 does for analysis) or
  documented as an explicit scope boundary in the crate docs — never
  silently variable.
- Quality sign-off bus factor: today several gates bottom out in one
  owner's listening checks on one CI machine class. A public 1.0 wants at
  least a second listener on the structured checklist and WCET/quality
  baselines sanity-checked on a second machine class.
- MSRV and platform support policy stated in the README (currently:
  pinned 1.97.0 toolchain, MSRV 1.85, CI on Ubuntu/macOS/Windows —
  implicit, not promised).

Stage 12 (no-panic hardening) is a prerequisite for this path but is
scheduled regardless — the DJ app benefits either way.

## Definition of Success

`timestretch-rs` is done with this roadmap when all of the following hold:

- Total pipeline delay ≤ 15 ms with control-to-audio at resampler lookahead,
  measured and reported honestly.
- One engine serves live and batch; streaming-vs-offline agreement is a
  determinism property, not a tolerance test.
- Zero corrective heuristics: no energy-EMA gain, no spectral shelves, no
  dry blends, no overlay splices — structure over patches.
- The RT contract is machine-verified: zero allocations, WCET-gated
  worst-case callbacks, no fallible calls in the audio path.
- External-reference quality evidence is mandatory in CI on a public DJ
  corpus.
- A deck built on the engine feels like hardware: instant tempo nudges,
  seamless cue jumps, click-free pitch rides, graceful keylock at extremes.

All of the above hold as of the Stage 9 cutover (2026-07-15). The
remaining stages extend the definition:

- Trustworthy beat grids on everything a DJ loads — non-EDM, drifting,
  live material — gated on an annotated corpus (Stage 10).
- Wide-range Master Tempo available as an opt-in profile with an honest
  latency contract, or explicitly ruled out by listening evidence
  (Stage 11).
- No panic reachable from the public API on arbitrary input,
  machine-verified by fuzzing (Stage 12).

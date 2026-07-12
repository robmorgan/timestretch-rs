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

## [ ] Stage 3: Beat-Synchronous SOLA Corrector and Corrector Selection

Automation: auto

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

## [ ] Stage 4: Artifact-First Control — Transient Protection and Splice Guidance

Automation: auto

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

## [ ] Stage 5: Deck Semantics — Warm Start, Loops, Timestamped Control, Rate Coverage

Automation: auto

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

## [ ] Stage 6: WCET Flattening and Callback Budget Gates

Automation: auto

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

## [ ] Stage 7: Parity Campaign — Tuning, External Evidence, Quality Sign-Off

Automation: manual

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

## [ ] Stage 8: Batch/Offline Rebase onto the Engine Graph

Automation: auto

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

## [ ] Stage 9: Cutover and Deletion

Automation: auto

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

## Disposition of the Previous Roadmap's 16 Stages

| Old stage | Status then | Disposition now |
|---|---|---|
| 1 Fast modulation stability | Complete | Inherited as gates: the torture-test methodology and click/slew bounds port into new Stages 1–3. The deferred PV ratio-step seam is dissolved — the PV no longer implements tempo. |
| 2 Confidence-based blending | Open | **Cancelled** (hybrid retired). Event-mask ideas survive only as optional tuning levers in new Stage 4. |
| 3 Rolling adaptive analysis | Partial | Shipped loudness-robust onset front end inherited unchanged (it feeds artifacts). Rolling multi-res analysis and streaming-scheduler parity **cancelled** (artifact-first makes them fallback-only). Tempogram beat tracker → backlog; pulled in only if Stage 7 corpus evidence demands it. |
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

- SIMD and architecture-specific acceleration (revisit after Stage 6's WCET
  gates exist to measure it against)
- Desktop UI/UX polish beyond its role as the reference integration
- Additional presets, wider API surface, convenience wrappers
- General-purpose (non-EDM) material quality

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

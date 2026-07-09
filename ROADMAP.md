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
- Analyze-on-load pre-analysis (Stage 14, complete): tracks analyzed once
  produce a persisted artifact consumed authoritatively by every stretch
  path; streaming has an absolute source-position API for seek alignment.
- Low-latency streaming is first-class (Stage 10, complete): library
  `StreamProfile` (Live ~35 ms / Club ~70 ms / Quality ~139 ms, all carrying
  the full DJ bundle), honest measured latency reporting, and time-based
  scheduler tuning at small hops.
- Warm-start seek/cue/loop (Stage 11, complete): `warm_start_seek` re-primes
  from preceding audio (no cold gap, no PV warm-up, control state preserved)
  and `notify_source_jump` wraps loops gaplessly; desktop seek, beat-jump
  buttons, and loop in/out are wired to them.
- Analysis robustness (Stage 3, partial): a loudness-robust onset front end
  (log-compressed flux, median+MAD threshold, energy gate) so dense mastered
  tracks analyze correctly; tempogram beat tracking and rolling adaptive
  analysis remain open.
- The main gap is not "missing DSP ideas". The main gap is the last 20%:
  tighter routing/decomposition, stricter invariants, and reference-driven
  tuning — then the DJ deck readiness stages.

## Principles

- Fix audible regressions before adding features.
- Make the RT-safe path the default, obvious path.
- Reject malformed input instead of silently truncating or falling back.
- Prefer reference-driven quality gates over self-comparison alone.
- Preserve the EDM/DJ-first focus (settled in Stage 9); general-purpose
  broadening is out of scope until the deck readiness stages are complete.

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

## [~] Stage 3: Upgrade Analysis from Fixed EDM Heuristics to Rolling Adaptive Analysis

Automation: auto

Status: **partially complete** — the loudness-robustness slice landed after
a real track ("dense mastered club mashup") produced an empty analysis
artifact (0 onsets, BPM 0, confidence 0), which the analyze-on-load feature
(Stage 14) then silently no-opped on. Root cause: linear-magnitude spectral
flux plateaued on continuously loud material and the multiplicative
`median * 3.1` threshold sat at the signal ceiling, so no frame stood out.
Every prior test used clicks-over-silence and never exercised that regime.

### Why

The transient and confidence front end is currently too static. Production
libraries usually make better decisions because they use rolling, multi-scale,
content-adaptive analysis instead of a small set of fixed weights and one-shot
confidence estimates.

### Primary Files

- `src/analysis/transient.rs`
- `src/analysis/adaptive_snapshot.rs`
- `src/analysis/beat.rs`

### Shipped (loudness-robustness slice)

- Log-magnitude compression `ln(1 + gamma*mag)` in the flux/energy/phase
  channels so onsets stay separable on dense mastered material.
- Robust additive threshold `median + k(sensitivity)*MAD + floor` over
  trailing windows (replacing the multiplicative threshold), with a global
  relative floor, a MAD floor, local-peak/masking gates, and startup-frame
  suppression.
- Energy-channel gate: sustained tonal material (held note, pure tone) emits
  no onsets, killing the spectral-leakage false positives that smeared pure
  tones in the stretcher.
- Telemetry: `AnalysisReport` (ODF median/MAD/max, onset count/rate, timing)
  via `analyze_for_dj_with_report`, surfaced by CLI `analyze -v` and the new
  `qa/track_analysis_qa.rs` real-track harness (scans
  `TIMESTRETCH_TRACK_QA_DIR`, filename `<n>bpm` tag accuracy, usability gate).
- Fixture `test_audio/dense_mastered_128bpm.wav` + `tests/dense_material_
  regression.rs` reproduce and lock in the fix.
- Result: the real failing track now analyzes to 123 BPM / confidence 0.90 /
  2.9 onsets/sec; the kick fixture no longer octave-doubles.

### Still Open

- **Rolling multi-resolution analysis** and **time-evolving tonal/noise
  confidence** (the original core of this stage) are untouched.
- **Beat-tracker redesign** (tempogram over the ODF, EDM-weighted octave
  disambiguation, widen the 100-160 BPM fold to ~70-180, tempo-salience
  confidence). Deferred deliberately: the failing track already resolves to
  the correct BPM with the current median-interval tracker, so this becomes
  valuable mainly for material outside 100-160 (DnB, hip-hop). A residual
  edge case (a 60 Hz sub-sine reporting false confidence) waits on this.
- **Streaming scheduler parity**: `src/stream/transient_scheduler.rs` is a
  separate incremental detector with its own `mean + 2.5*sigma` threshold and
  a bounded version of the same dense-material blindness — not yet aligned
  with the offline front end.

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

## [x] Stage 9: Decide the Product Boundary

Automation: manual

### Why

The codebase mixed two ambitions: "excellent EDM-focused stretcher" and
"general-purpose production-grade library". Those are related, but not the
same target, and the benchmark corpus, API defaults, and quality gates all
depend on which one is being built.

### Decision (settled July 2026)

**EDM/DJ-first.** The library exists to be the stretch engine inside a
custom Rust DJ application, with Serato Pitch 'n Time DJ / Elastique as the
quality bar. General-purpose broadening is explicitly out of scope until the
DJ Deck Readiness stages are complete; if it happens at all, it happens as a
deliberate later expansion, not as drift.

### Consequences (binding on later stages)

- **Benchmark corpus (Stage 8):** the public CI corpus is DJ material —
  four-on-the-floor kick patterns, house/techno loops, sustained bass,
  vocal-over-beat, full EDM mixes. Wider material (orchestral, sparse
  acoustic, speech) is not gated and regressions there do not block.
- **Quality gates:** judged at DJ ratios (0.92–1.08 primary, ±20% secondary)
  on stereo mixes, in the streaming path first — beatmatching happens there.
  Batch-path quality on non-EDM material is best-effort.
- **API defaults:** tuned for the DJ case — stereo M/S, DJ-ratio sweet spot,
  beat-aware behavior on by default where confidence allows. General-purpose
  callers adjust params; DJ callers get good behavior out of the box.
- **Analysis tuning (Stages 3-5):** transient/beat heuristics stay tuned for
  electronic material; adaptive analysis raises robustness *within* that
  focus (busy mixes, vocals over beats), not toward arbitrary material.
- **Priority order:** the DJ Deck Readiness stages (10-13) outrank the
  remaining batch-quality stages (2-5) whenever they conflict for attention.

## DJ Deck Readiness

The stages below close the gap between "quality streaming stretcher" and
"engine a real DJ deck can be built on" (the Serato Pitch 'n Time DJ /
Elastique bar). Stage 9 settled on EDM/DJ-first, so these stages are the
priority track. Ordering: Stage 10 and Stage 11 are DJ-blocking and should
follow Stage 1 directly; Stage 12 is the long-tail quality work; Stage 13 is
completeness. Stage 15 (varispeed-first keylock) and Stage 16 (causal
low-end quality at small FFTs) together target the élastique feel: instant
tempo control with Club-grade lows on the Live profile — Stage 15 is the
bigger win per unit of work and should go first.

## [x] Stage 10: Make Low-Latency Streaming First-Class

Automation: auto

### Why

Default streaming latency was `fft * 3/2` = 6144 samples (~139 ms at
44.1 kHz). A DJ nudging against a beat needs roughly 10-40 ms
control-to-audio. The low-latency constructor (1024 FFT, ~35 ms) existed but
traded quality blindly (it silently dropped the entire DjBeatmatch tuning
bundle), and `QualityMode` barely changed streaming DSP, so latency and
quality could not be traded deliberately.

Correction found during implementation: the original claim that the
`fft * 2` gate "inflates latency exactly when a DJ is off-unity, which is
always" was wrong — the wider gate applies only beyond ±10% ratio, so DJ
ratios (0.92-1.08) always had the `fft * 3/2` floor. The real defects were
the gate flipping mid-glide (it read the glided ratio, not the target), the
dishonest reporting, and the untuned 1024 configuration.

### Primary Files

- `src/stream/processor.rs`
- `src/stream/transient_scheduler.rs`
- `src/core/types.rs`
- `desktop/src/processor.rs`

### Shipped

- **`StreamProfile` (Live/Club/Quality)** replaces the QualityMode idea from
  the original stage text: profiles are the streaming latency/quality knob
  (1024/256 ~35 ms, 2048/512 ~70 ms, 4096/1024 ~139 ms), every tier carries
  the full DJ bundle, and `try_from_tempo_low_latency` delegates to Live.
- **Gate correctness + honest reporting**: the buffering gate reads the
  *target* processing ratio (no mid-glide flips at the ±10% boundary);
  `latency_samples()`/`latency_report()` report the real gate including the
  ratio band and sinc pitch lookahead.
- **Time-based scheduler tuning**: cooldown (~46 ms), statistics EMA
  (~104 ms), and the modulation-hold budget (~371 ms) derive from absolute
  time, identical at the reference 4096/1024 configuration and correctly
  scaled at 1024/256. Warmup stays a 3-frame statistical bootstrap.
- **Measured latency**: `tests/streaming_latency.rs` verifies first-sample-
  out equals the reported gate exactly (delta 0 at all profiles, in and out
  of band) and bounds control-to-audio for ratio (gate + 50 ms glide) and
  pitch (near-instant; resampler is post-PV). README carries the table.
- **Latent flush bug fixed**: vocoder tails bypassed the engaged pitch
  resampler at flush, splicing pre-resample-rate samples in (a full-scale
  click at most stream lengths); the pitch-sweep guard now runs at four
  lengths.
- Desktop uses the library profiles and shows the reported latency next to
  the profile selector; preroll derives from `latency_samples()`.

### Exit Criteria

- Live profile at ~35 ms in-band with the full DJ bundle; click-free under
  nudge/ride/snap torture and zero-alloc in steady state. ✓
- Profiles measurably distinct (`qa/profile_quality.rs`): steady-ratio
  similarity Quality 0.9991 / Club 0.9976 / Live 0.9982; under a 0.92-1.08
  ride Live 0.9982 / Club 0.9969 / Quality 0.9932 (small windows track
  modulation better — Live is the control profile by measurement, not just
  by latency). ✓
- Published latency table (README) with measured first-sample-out and
  control-path behavior. ✓

### Deferred Follow-Ups

- Sub-bass cutoff at 1024 FFT: raising 180 → 250 Hz measured only +0.0008
  low-band similarity on synthetic bass — not adopted; revisit with real
  material in Stage 8's corpus.
- Gate hysteresis at the ±10% boundary if target-toggling across it ever
  shows audible stalls (target-based gating removed the known oscillation
  source).

## [x] Stage 11: Warm-Start Seek, Cue, and Loop Support

Automation: auto

### Why

DJ decks jump constantly: cue points, loops, beat jumps. Every jump used to
mean `reset()` (which reallocates the vocoders) plus a cold `fft*3/2`
prebuffer refilled from silence and a PV warm-up transient — the desktop
rebuilt the whole processor per seek. Commercial engines re-prime from
surrounding audio so a jump is seamless.

### Primary Files

- `src/stream/processor.rs`
- `src/stretch/phase_vocoder.rs`
- `desktop/src/processor.rs`, `desktop/src/app.rs`, `desktop/src/state.rs`

### Shipped

- `StreamProcessor::warm_start_seek(target, preroll)` re-primes at a new
  source position by running the preceding audio through the full DSP path
  and discarding its rendered output, so the first emitted frame is already
  converged. Ratio/pitch control state — targets *and* in-flight glides —
  is preserved (no re-glide from unity). A short declick fade covers the
  mid-waveform cut.
- `StreamProcessor::notify_source_jump(next_frame)` handles gapless loop
  wraps (input-feed splice, timeline re-anchor only, no state reset).
- `PhaseVocoder::reset_streaming_state` clears all per-stream state without
  deallocating — the allocation-free equivalent of a rebuild.
- CPU bounded by `warm_start_preroll_frames()` (gate + one FFT window);
  the deterministic engine's warm start is allocation-free
  (`tests/realtime_allocations.rs`). The legacy hybrid engine rebuilds its
  rolling window (already non-RT) and is documented as such.
- Desktop: click-to-seek and beat-jump buttons (±4/±16 beats off the
  detected grid) use warm-start; loop in/out/exit wraps the region gaplessly
  via `notify_source_jump`.

### Exit Criteria

- Cue jumps and loop wraps produce no audible warm-up transient or gap. ✓
  (`tests/warm_start.rs`: post-seek level at steady state immediately, output
  within normal cadence, no gap.)
- A beat-jump/loop torture test passes clean at DJ ratios. ✓ (loop-wrap seams
  and cue-drumming click-free across ratios/profiles; click bound 6× ideal
  slew vs the PV's measured ~4.7× ripple and a cold-restart ~40× splice.)
- Desktop seek no longer rebuilds the processor. ✓ (warm-start, with a
  rebuild only as an error fallback.)

### Deferred Follow-Up

- ~~Allocation-free warm start in the legacy hybrid re-render engine.~~
  Obsolete: the legacy hybrid streaming engine has been removed.

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

- [x] Design a bounded-latency multi-band PV for streaming (larger FFT for
  sub-bass phase coherence, smaller FFT for transient-sharp highs).
  Landed as `StreamingEngine::MultiResolution`: streaming support on
  `MultiResolutionStretcher` (incremental LR8 split, per-band PVs, aligned
  band summation, combined flux report) wired into `StreamProcessor` with
  per-profile sub-bass FFT sizing (Club 8192 / Quality 16384; Live rejects
  the engine). Allocation-free steady state; measured gates: Club 12288
  frames (~279 ms), Quality 24576 frames (~557 ms) at 44.1 kHz.
  Known limit: similarity dips in the bands containing the 200 Hz / 4 kHz
  crossover seams (inherited from the offline multi-res band summation);
  sub-band similarity ties the single-PV ceiling, mid band wins.
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

## [x] Stage 14: Analyze-on-Load Pre-Analysis Everywhere

Automation: auto

### Why

Commercial DJ products analyze tracks once on import and let the engine
consume the stored beatgrid/onset map; offline analysis is non-causal and
strictly better than per-chunk online detection, and it frees realtime CPU.
The `PreAnalysisArtifact` existed but was only consumed by batch mono beat
snapping — no stretch path used its transient onsets, and nothing produced
or persisted it in real workflows.

### Primary Files

- `src/core/preanalysis.rs`
- `src/analysis/preanalysis.rs`
- `src/analysis/adaptive_snapshot.rs`
- `src/stretch/stereo.rs`
- `src/stream/processor.rs`
- `src/cli.rs`
- `desktop/src/app.rs`, `desktop/src/processor.rs`

### Work

- Artifact schema v2: version, content binding (source length + FNV-1a
  hash), per-onset strengths and band flux, analysis hop size; v1 JSON
  stays readable. Gates: `is_usable` (runtime) and `matches_source`
  (load boundary).
- All four stretch paths consume a usable artifact authoritatively
  (online detection skipped): batch mono, batch stereo M/S, deterministic
  RT engine (artifact-scheduled per-band phase resets, allocation-free
  cursor, strength-based band thresholds, modulation-hold suppression),
  legacy hybrid re-render engine (absolute window-base tracking maps
  artifact anchors into the rolling window; mono skips per-render
  adaptive analysis).
- `StreamProcessor::set_source_position` + `source_position`: absolute
  source-frame timeline (including unity passthrough) so artifact
  positions survive seek-rebuild flows. `set_pre_analysis` for rebuild
  wiring; artifact telemetry in `TransientResetStats`.
- Producers: `timestretch-cli analyze` writes a `.tsanalysis.json`
  sidecar; `--pre-analysis` validates and warn-falls-back when stale.
  Desktop analyzes on load in a background thread (generation-guarded),
  caches the sidecar, threads the artifact into every processor rebuild,
  and anchors rebuilds at the correct source position (fixing preset
  changes mid-track implicitly rebuilding at frame 0).

### Deferred Follow-Ups

- ~~Thread a sparse per-band transient map through `process_with_onsets` so
  the artifact path keeps per-band resets in the legacy hybrid engine.~~
  Obsolete: the legacy hybrid streaming engine has been removed.
- Tempo curve (per-region BPM) in the artifact for drifting material.
- Warm-start seek (Stage 11) builds on this stage's position bookkeeping.

### Exit Criteria

- All four paths consume artifacts with parity-or-better transient
  preservation vs online detection (`tests/preanalysis_pipeline.rs`,
  `tests/streaming_preanalysis.rs`). ✓
- Zero-alloc steady state with a large artifact attached
  (`tests/realtime_allocations.rs`). ✓
- Seek-offset streaming schedules exactly the onsets past the seek point. ✓
- CLI and desktop produce/persist/validate artifacts end-to-end. ✓
- Version 1 artifact JSON remains readable. ✓

## [ ] Stage 15: Varispeed-First Keylock (Instant Tempo Control)

Automation: auto

### Why

In the current architecture the phase vocoder implements the tempo change,
so the buffering gate (Live: 1536 frames ≈ 34.8 ms) plus the ~50 ms control
glide IS the tempo-control latency — a fader nudge takes ~85 ms to fully
land. Commercial decks (Traktor on élastique 3, Serato on Pitch 'n Time DJ)
invert this: the tempo fader drives a varispeed resampler (instant,
sample-accurate response), and the stretcher only pitch-corrects at a
slowly-varying transposition. The stretcher's buffering then becomes a
CONSTANT output delay the host compensates in its timeline — perceived
control latency drops to the soundcard buffer. Same DSP, radically better
beatmatching feel; no vocoder research required.

### Primary Files

- `src/stream/processor.rs` (control-path restructure)
- `src/core/resample.rs` (varispeed input stage on the streaming sinc
  resampler)
- `desktop/src/processor.rs`, `desktop/src/audio_engine.rs` (timeline/delay
  compensation, reference integration)
- `tests/streaming_latency.rs` (new control-to-audio budget)

### Work

- Add a keylock control mode to `StreamProcessor` (e.g.
  `ControlPath::VarispeedFirst` alongside the current PV-implements-tempo
  path, which stays for tape-mode/no-keylock use): input passes through a
  varispeed sinc resampler at the control ratio, then the PV runs at the
  compensating stretch to restore pitch.
- Tempo changes retarget the resampler sample-accurately (no 50 ms glide on
  the tempo axis — the glide survives only on the PV's transposition
  tracking, where its job is masking phase seams, not delaying tempo).
- Rework latency reporting: `latency_report()` gains a
  `control_to_audio_frames` (resampler-only, ~0) versus
  `pipeline_delay_frames` (PV gate, constant, host-compensated) split.
  Position/beat-grid bookkeeping (`source_position`, artifact onsets,
  Stage 14 timeline) must account for the moving resampler ratio ahead of
  the PV.
- Ratio-drift audit: the varispeed stage consumes source at a time-varying
  rate; `expected_total_output_samples` reconciliation and the modulation-
  hold machinery need to read the effective ratio from the resampler, not
  the PV.
- Desktop adopts it as the DJ default with keylock on; the engine-selection
  UI pattern from the multi-res work is the template.
- Interacts with Stage 13's extreme-rate keylock blend: beyond the blend
  threshold the PV correction fades toward plain varispeed — in this
  architecture that becomes simply fading the corrector out, which is the
  cleanest possible implementation of that item.

### Exit Criteria

- Measured control-to-audio for a tempo step in varispeed-first mode is
  bounded by one callback + resampler lookahead (tens of samples, not the
  PV gate) in `tests/streaming_latency.rs`.
- Output pipeline delay is constant and reported; beat-grid positions stay
  sample-aligned through tempo rides (artifact onsets keep firing at the
  right places).
- A/B against the current path under the qa ratio ride: no quality
  regression on the profile similarity rows; pitch wobble during fast
  tempo rides stays below audibility (define and gate a cents-deviation
  metric).
- Allocation-free steady state, both engines (deterministic and
  multi-resolution) behind the new control path.

## [ ] Stage 16: Causal Low-End Quality at Small FFTs

Automation: auto

### Why

The Live profile's 1024-point window has 43 Hz/bin resolution, which is why
it leans on a raised (180 Hz) rigid sub-bass locking cutoff and why the
multi-resolution engine (Club+) exists. But single-frame DFT resolution is
not a physical limit on a *causal* estimator: phase deltas across multiple
past hops give near-arbitrary frequency precision for sustained tones, and
sub-bass is quasi-periodic enough for time-domain period tracking. All of
the levers below consume only PAST samples — they raise low-end quality at
1024 with ZERO added output latency. Combined with Stage 15 this gets Live
toward "instant control AND Club-grade lows", which is the élastique bar.

### Primary Files

- `src/stretch/phase_vocoder.rs` (instantaneous-frequency estimation)
- `src/stream/processor.rs` (analysis-lookback wiring)
- `src/stretch/multi_resolution.rs` (shared band machinery where reusable)
- `qa/profile_quality.rs` (sub/low band rows are the scoreboard)

### Work

- **Multi-hop instantaneous-frequency estimation**: replace single-hop
  phase differencing with a tracked per-bin (or per-peak) frequency
  estimate over several past frames (with parabolic magnitude-peak
  interpolation); use it to drive `phase_accum` advance. Target: sustained
  bass at 1024 advances phase as accurately as a 4096 window does today.
- **Analysis-only lookback window**: an optional 4× analysis FFT over the
  trailing (already-received) samples — window ENDS at the newest sample,
  so it is causal — refining sub-500 Hz frequency estimates for the small
  synthesis PV. CPU cost is one extra FFT per hop; gate it behind
  `QualityMode` if the Live budget needs it.
- **Time-domain sub-bass option**: below ~150 Hz, evaluate a
  period-tracking resynthesis path (phase-continuous oscillator bank or
  period-synchronous splicing) against the PV-with-better-IF approach on
  the qa bass material; keep whichever wins, don't stack both.
- **Retire compensations as structure lands** (Stage 12 exit criterion
  applies here): each improvement must be measured against removing the
  corrective heuristics it obsoletes — the raised 180 Hz streaming cutoff,
  the mid-band additive boost, and the ratio-distance spectral shelf are
  the first candidates.
- Re-measure the multi-resolution engine after the IF work: better band
  PVs may shrink its seam losses too (the band vocoders de-phase less when
  their frequency estimates agree).

### Exit Criteria

- Live-profile sub+low band similarity on the qa bass material reaches the
  current Club deterministic numbers (sub ≥ 0.999, low ≥ 0.997) with the
  gate unchanged at 1536 frames.
- At least one corrective heuristic is removed with no regression in the
  quality dashboard (shrinking stack, per Stage 12).
- No new latency: `tests/streaming_latency.rs` figures unchanged; zero
  allocations in steady state.
- Modulation-torture and click gates stay green (better IF tracking must
  not fight the ratio-seam slew machinery).

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

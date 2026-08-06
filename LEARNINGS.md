# Learnings

Record of completed roadmap stages and the durable engineering lessons they
produced. The full stage texts (Why / Work / Exit Criteria as they were
planned) live in git history — `git log --follow ROADMAP.md`, or the tree at
tag `v0.10.0` for the last pre-rewrite version. ROADMAP.md carries only open
work; this file is where completed work and its evidence are recorded.

## Engine rebuild — Stages 1–9 (July 2026)

The July 2026 architectural review concluded the old streaming processor's
problems were structural (a ~5.2k-line orchestrator wrapping a phase vocoder
in corrective heuristics, latency gated at 1.5–2× FFT size, push-based
fallible audio path, divergent offline hybrid). Stages 1–9 rebuilt the crate
around a pull-based varispeed-first engine and deleted the old one.

- **Stage 1 — Walking skeleton** (2026-07-12, `8a1d214`). `src/engine/`:
  stage trait, SPSC mailbox + controller/processor split, source ring with
  underrun policy, `TimelineMap`, varispeed head, Tape profile; desktop went
  pull-native. Tape torture clicks = 0 at 1.5×/1.1× theoretical slew vs the
  old engine's 6×/1.5×.
- **Stage 2 — Keylock chain** (2026-07-12, `0033214`). Band split + 560-frame
  (12.7 ms) constant delay contract. Found and fixed a real crossover defect:
  the legacy `LR4²`-style `LR8Crossover` (four Q=0.707 sections) notches
  −6 dB at fc; replaced with a true `LinkwitzRiley8` (Qs 0.5412/1.3066,
  allpass re-sum). Cents wobble on the ±8% ride: p95 5.1 vs old 12.2.
  **Falsification verdict: the un-keylocked low band won the listening test**
  — the vocoder-processed sub was the audible problem, not the pitch-following
  bass.
- **Stage 3 — SOLA corrector** (2026-07-13, `e93e5a5`…). Elastic-cursor SOLA
  with correlation-matched splices; kick sharpness 1.21/1.18 vs old
  0.68/0.74. The small-FFT PV lost 2–4 dB in the top octave, so the SOLA
  threshold was raised to cover the full DJ range. **Post-landing lesson: the
  crossover seam combs when SOLA's elastic drift parks** — fixed with a
  320-frame correlation window (2+ periods at the band edge),
  distance-penalized tie-breaking, and sustained-rest drift bleed.
- **Stage 4 — Artifact-first control** (2026-07-13). Track→ring→stage onset
  mapping via seqlock anchor + inverted timeline; SOLA splice protection and
  strength-gated per-band PV resets. Artifact-guided sharpness ≥ online
  fallback ≥ old engine.
- **Stage 5 — Deck semantics** (2026-07-13). Warm-start seeks, gapless loop
  wraps, `set_tempo_rate_at` landing on the exact output sample via capped
  resampler emission, 48/96 kHz, keylock fade past the DJ range. Reverse is a
  host concern (feed reversed source) — the engine range is forward-only
  [0.25, 4.0] by design.
- **Stage 6 — WCET flattening** (2026-07-13). One PV hop max per block,
  budgeted warm-start priming. Keylock p99.9 ≤ 0.20 at 64-frame callbacks
  (bound 0.5).
- **Stage 7 — Parity campaign** (2026-07-14). Final A/B matrix 9/9 new ≥ old
  (ride cents 0.23/0.57 vs 1.86/12.19; sharpness 1.16/1.29 vs 0.71/0.74; …).
  Rubber Band reference gate required in CI (spectral 0.965/0.970 vs ≥ 0.85
  gate). Root causes closed: control read at ingest instead of the
  latency-matched position (≈ latency × slope ≈ 2.8 cents on rides); 32-tap
  read kernel drooping 1.5 dB in the top octave (→ 64-tap Kaiser); integer
  splice quantization scattering HF phase (→ sub-sample fine search).
  **Keylock range settled by ear: the small-FFT PV was audibly
  phasey/robotic at every boundary it was placed behind — SOLA carries the
  entire corrected range**, full keylock through ±20%, release fade
  20.5%→35%. The historic "RubberBand anomaly" was the old hybrid batch
  driver attenuating chirps ~28 dB — retired with the hybrid.
- **Stage 8 — Batch on the graph** (2026-07-14). `stretch()` renders on the
  engine graph with exact output length by construction
  (`round(frames × ratio)`, structural latency trim). Streaming and offline
  are sample-identical (the determinism gate immediately caught a real
  timeline-eviction bug that made splices depend on callback sizes).
- **Stage 9 — Cutover and deletion** (2026-07-15, v0.8.0). `src/stream/`,
  hybrid, multi-res, WSOLA, push API, and the never-audible live PV corrector
  all deleted; QA re-anchored on absolute thresholds from new-engine
  measurements.

## Stage 11 — Wide-range Master Tempo (2026-08-04)

Falsification first, build-out second; both halves recorded in detail at tag
`v0.10.0`'s ROADMAP. Shipped: `WideKeylockStage` — full-spectrum FFT-2048 /
hop-256 identity-locked PV + post-resampler, artifact-only flux-gated
per-band phase resets with a latency-delayed modulation-hold read, T clamp
[0.5, 4.0], constant 2144-frame (48.6 ms) contract, desktop Range selector.
Durable lessons:

- **hop = FFT/8 is mandatory** — 75% overlap caused the −75% tempo blowup
  (+7 LUFS spurious energy, ~2000 clicks/M) on every corpus track.
- **FFT 4096 loses on real mixes** (transient smear) despite winning on
  synthetic tones, and busts the latency contract.
- Big slowdowns ran "roboty" until the phase-gradient coherence blend was
  held on at wide ratios (it tapered to zero near ratio 2.5); blend settled
  at 0.20 by live A/B.
- Two mechanisms found by gates during build-out: the resampler ramp's
  harmonic-mean consumption drifts stream balance by `hop/2·ln(T₁/T₀)` per
  sweep (fixed: `set_step_anchor`); instant full-range T steps tear the OLA
  seam (fixed: log-space transposition slew, ~30 ms full-range settle).
- **Metrics twice failed to predict ear verdicts** (down-side robotiness
  scored above the clean up-side; blend 0.20 vs 0.40 metrically identical).
  Owner listening is the binding gate; metrics are regression tripwires.
- Verdicts: ±50% shippable; −75% documented degradation ("through a guitar
  amp"), no silent cliff. **The Rubber Band gap at wide rates is real and
  accepted**: single-FFT identity PV vs R3's multi-resolution engine.

## Other shipped work (outside the stage line)

- **Musical key detection** (v0.9.x): HPCP chroma + profile correlation,
  Camelot output; 85.7% exact vs Mixed In Key on the corpus; scored by the
  BPM harness.
- **Rigid beat grids** (v0.10.0, `026e758`): corpus diagnostics showed the
  beat-tracking failure mode was gross phase misplacement (60–180 ms) on
  quantized tracks — never wrong BPM. A rigid kick-band fit (constant BPM ×
  phase circle, adopted only at `phase_lock ≥ 0.3` plus a smeared-score
  floor) took beat F 71.5%→93.8%, downbeat F 20.7%→76.4%. The beat-F gain is
  partly self-confirming (annotations share the method); ear verification is
  the honest check. Known non-adopters (MSBWY, Hot Stuff, Somebody To Love)
  are exactly the annotator's own phase-indecisive tracks — offbeat disco
  bass under 150 Hz competes with the kick phase. Diagnosed live 2026-08-03:
  Hot Stuff fits 120.000 BPM exactly but phase_lock = 0.109, so the wandering
  DP grid ships (intervals 475–569 ms on a constant-tempo track).
- **Performance** (v0.10.x): SOLA/varispeed hot-loop auto-vectorization,
  polyphase kernel rows, lockstep multi-channel resampling — keylock mean
  2.2×↓, spikes 4×↓.
- **`.tsa` analysis container** (v0.11.0): single binary sidecar; JSON
  sidecar API deprecated. Bytes-first persistence (Halo stores blobs in
  sqlite; sidecars are a convenience layer).

## Quality review vs Rubber Band (2026-08-05)

Full-code audit of both stretch paths plus empirical probes through the
public API. This is the evidence base for ROADMAP Stages 13–17.

**Measured (probe harness, 44.1 kHz):**

- Duration is **sample-exact** at ratios 0.731 / 0.9375 / 1.137 / 1.333333 /
  1.618 over 30 s (0 samples error).
- Stereo through the DJ keylock path is **bit-identical lockstep**
  (max |L−R| = 0.0 on identical channels).
- Click trains at ±8%: no flams, placement deviation ≤ 1.8 ms.
- Tape at rate 1.0 is bit-transparent (347 dB SNR at lag 0).
- **Sustained-tone purity above the crossover collapses at DJ ratios**:
  3.3 kHz sine drops from 71.6 dB to 23.9 dB (0.92×) / 33.1 dB (1.08×) —
  SOLA splice granulation at ~20 splices/s. 440 Hz: 54–65 dB. 55 Hz clean
  (but pitch follows tempo, by design). This is the main audible gap vs
  Rubber Band inside the DJ window on pads/strings/vocal harmonics.

**Confirmed defects (all verified in source):**

1. **Unwrapped phase blends** — `phase_vocoder.rs:1624-1629` (gradient
   blend) and `:1295-1298` (IF blend) linearly interpolate raw phases that
   differ by arbitrary multiples of 2π; not a valid phase interpolation. The
   gradient blend runs at full 0.20 strength at every ratio in production.
2. **Never-wrapped f64 phase accumulator downcast to f32 each frame**
   (`phase_vocoder.rs:1552,1555`) — HF bin phase degrades to noise after
   minutes of accumulation; onset resets rescue busy material but the
   detector returns zero onsets on sustained tonal material by construction
   (`transient.rs:571-579` energy gate).
3. **DC/Nyquist break Hermitian symmetry** (`reconstruct_spectrum`,
   `phase_vocoder.rs:1637-1644`) — both bins get accumulated phases and are
   excluded from the mirror; `.re` extraction scales them by cos(φ).
4. **Offline wide path uses hop = FFT/4** (`offline.rs:169`) — the exact
   configuration `wide_keylock.rs:53-55` documents as the −75% blowup.
   Offline wide ratios also never exercise the shipped `WideKeylockStage`
   (independent per-channel batch PVs, no stereo coupling —
   `offline.rs:186-202`).
5. **Batch `resample_sinc` has no cutoff scaling** (`resample.rs:84-153`) —
   `pitch_shift()` downsamples through it (`lib.rs:680`) and aliases;
   `AudioBuffer::resample` is plain cubic (`types.rs:631`).
6. **Extreme-rate correction fade is a per-block gain step**
   (`keylock.rs:124-127,149`) between two differently-pitched signals; the
   toggle path got a per-sample 512-frame ramp, the fade path got none.
7. **`ctx.modulation_hold` is inert on the DJ profile** — SOLA never
   receives it (`sola.rs:413`, `keylock.rs:120`); `stage.rs:130-134`
   documents behavior the code doesn't have.
8. **The identity test suite tests the bypass, not the DSP** — all of
   `tests/identity.rs` and the "strict" ratio-1.0 regression go through
   `stretch()`, which returns `input.to_vec()` at exactly 1.0
   (`lib.rs:481-483`). No test anywhere runs the PV itself at ratio 1.0.
   Related: `stretch_into` lacks the 1.0 bypass entirely.

**Stage 13 follow-up finding (2026-08-05, during the fix):** hop = FFT/8
attenuates tones in the PV's rigid sub-bass region (< ~107 Hz at FFT 2048)
by 5–15× at ratio 2.0, while tones just above the boundary hold the ideal
balance (probe sweep 70–150 Hz; the old offline hop = FFT/4 kept sub-bass
clean at ratio 2 but is the documented blowup at ratio 4). This is a trait
of the *shipped live* wide configuration — consistent with Stage 11's
"−75%: bass reads bitcrushed" note — now pinned by
`tests/stretch_quality_regressions.rs` so it cannot silently worsen.
Improving sub-bass at heavy slowdown is Stage 14/16 material.

**Structural limitations (not bugs — recorded so they aren't re-litigated):**

- SOLA tonal-HF granulation is the mechanism working as designed; the fix
  space is a third band, tonality-adaptive splicing, or acceptance — decide
  from listening evidence (ROADMAP Stage 16), not tuning.
- The locking has no cross-frame peak continuity (`phase_locking.rs:389`
  recomputes per frame; overlapping regions last-write-wins) — the classic
  unstable-cymbals source at wide ratios; RB-class coherence needs peak
  tracking.
- SOLA's crossover-seam de-phasing under *continuous* rides is bounded but
  never recovered while the fader keeps moving (rest recentering needs
  ~150 ms of sustained near-unity).
- The correlation reference is sampled at the integer floor of the
  fractional cursor while candidates are sinc-interpolated
  (`sola.rs:587-588` vs `:645-668`) — sub-sample splice alignment is
  measured against a phase-offset reference.
- Minor dead weight: `hop_synthesis` field is DSP-dead; the envelope
  "correction" is provably ×1.0 (`phase_vocoder.rs:1333-1335`, dormant);
  ~200 lines of seam re-anchoring cases are dead in production but pinned by
  ~25 tests; three `bessel_i0` copies; stale `Makefile`/`qa/README.md`/
  `RESEARCH.md` references.

**Cross-cutting lessons this review adds:**

- A ratio-1.0 bypass in the public API makes every downstream "identity"
  test vacuous. Null tests must target the DSP object directly, and the
  meaningful offline null is *near*-1.0 or engine-at-1.0.
- Phase arithmetic rules that would be caught by a types pass: never
  linearly mix unwrapped phases; wrap accumulators; force DC/Nyquist real.
- When a constant is documented as mandatory in one path
  (`WIDE_HOP = FFT/8`), grep for the other paths that compute it
  independently.
- Sidecar/artifact caches need an analysis-version invalidation policy:
  the rigid-grid fix shipped without bumping `PREANALYSIS_VERSION`, so
  pre-v0.10 sidecars keep serving worse grids forever.
- **A listening verdict is evidence about the implementation auditioned,
  not the algorithm class.** Stage 7 rejected "phase vocoder in the
  corrected range" and Stage 11 accepted the wide-rate Rubber Band gap —
  both against a PV whose unwrapped-phase blends manufacture the exact
  phasiness heard. When a defect is later found in an auditioned
  implementation, every architecture decision that cites those listens is
  stale evidence and gets a cheap re-audition (ROADMAP Stages 13/16), not
  silent standing. Corollary: record the implementation state (commit,
  known defects) alongside every recorded listening verdict.

## Stage 13 — Wide-Path Phase Hygiene (2026-08-06)

Fixed the four confirmed phase-domain defects from the quality review
(commit `fb7dcfa`, merged PR #36; fade/hold/version follow-ups in PR #37):
wrapped `phase_accum`, wrapped-difference coherence blends, real
DC/Nyquist, offline wide hop = FFT/8, `stretch_into` bypass parity, plus
`tests/pv_null.rs` — the first tests to exercise the vocoder itself at
ratio 1.0.

- **Measured (A/B vs pre-fix)**: PV-direct null at ratio 1.0 went 59 →
  139 dB SER; 5 kHz purity at ratio 1.5 went ~25 → ~57 dB, flat over the
  render. Falsification sidecar re-run (300 rows vs the Stage 11
  baseline): shipped arm improved mean |LUFS error| 2.75 → 1.90 dB, RB
  similarity held, zero click increases; the diagnostic arms running the
  buggy paths hardest dropped ~80–125 clicks/M.
- **Owner listen (2026-08-06, ±50%, `norm/` level-matched renders,
  implementation `fb7dcfa` + sidecar rerun)**: "significantly better"
  than the Stage 11 renders; **the Rubber Band gap narrowed but R3
  remains ahead** — the wide-rate acceptance holds on fresher, smaller
  evidence.
- Re-pinning lesson: two wide-path baselines had encoded pre-fix behavior
  (an edge-ramp-biased pitch measurement window; a two-tone balance band
  built around the old low-band loss). Every re-pin was attributed by
  stash A/B before touching the band — one "regression" was an
  improvement, one was the pre-existing live-path sub-bass trait
  surfacing (recorded above).

## Stage 15 — DJ-Band Ride Polish (2026-08-07)

Fade-band per-sample ramp (D6) and the `modulation_hold` contract landed
in PR #37; the core seam item in PR #38 (`c767181`/`629d6d6` line).

- **Characterization before code**: sustained mild rides (0.5–1%
  deviation — the mix-in gesture) held the 120 Hz crossover seam at
  −7 dB for the whole ride; drift sawtooths to the full 192-frame
  trigger and rest recovery needs ~150 ms of stillness. Strong rides and
  parked faders were already fine.
- **Fix**: mild-motion bounded recenter — the rest-splice mechanism
  without its dwell, at drift > 96 when deviation < 1.2%. Measured:
  worst comb −7.1 → −4.4 dB, riding steady-state −2.5 → −1.2 dB, safe
  profiles bit-identical, ~5 bounded splices/s.
- **Negative result worth keeping**: simply tightening the general drift
  trigger made everything WORSE (−5.5 dB persistent, even at rest) —
  unbounded early splices park drift on the dominant-period grid on
  periodic content (the #62 pathology). The landing bound is the
  load-bearing part of early splicing, not the trigger level.
- Gates: `seam_survives_a_sustained_mild_ride` (fails pre-fix at
  −7.14 dB), `fade_band_rate_steps_are_click_free` (fails pre-fix at
  ~3.5× tone slew); `qa/engine_keylock` + `qa/engine_transients`
  promoted into CI. Cents-wobble and A/B-matrix gates unchanged.
- **Owner listen (2026-08-06, live desktop deck, branch `629d6d6`):
  "definitely an improvement" — bass body stable through a sustained
  gentle ride.** Optional items (fractional correlation reference,
  strength-aware protection, modulation_hold wiring) remain
  evidence-gated: each lands only if it moves a gate.

## Stage 10 — Tracked-Beat Corroboration for Rigid Grids (2026-08-07)

The three rigid-grid non-adopters (MSBWY, Hot Stuff, Somebody To Love —
syncopated disco, lock 0.109–0.181 vs the 0.3 gate) resisted both
roadmap disambiguation candidates, prototyped and measured before the
fix landed:

- **Slot exclusion failed**: rival-phase landscapes showed the
  competitors at *swung* subdivision offsets (Hot Stuff peaks at
  6–10/32 period, MSBWY at 25/32), not one excludable anti-phase slot;
  stacking exclusion zones until specific tracks pass is
  threshold-lowering in disguise.
- **Onset-sharpness weighting failed**: squared-envelope emphasis helped
  two tracks but made Hot Stuff WORSE (0.109 → 0.072) — disco bass is as
  punchy as the kick in the kick band.
- **What worked: corroboration by an independent estimator.** The DP
  tracker (full-band novelty, not the kick-band phase circle)
  independently lands ≥ 90% of its beats on the rigid grid for the truly
  rigid tracks (MSBWY 0.98, Hot Stuff 0.90) and collapses on genuinely
  non-rigid material (tempo-ramp control 0.25 — which phase_lock alone
  would wrongly trust at 0.77 — and Somebody 0.28, real estimator
  disagreement). Adoption gate: lock ≥ 0.3 OR agreement ≥ 0.6 (mid-gap
  between the measured ≤ 0.28 and ≥ 0.90 clusters), sanity floor
  unchanged.

Corpus results: MSBWY/Hot Stuff beat F 0.95/0.93 → **1.00**, downbeat F
→ 1.0, offsets sub-ms; 33rd Rate Revs X downbeat F 0 → 1.0; 13/16
harness rows byte-identical; ramp/jitter unit controls green.
PREANALYSIS_VERSION → 9/9 (first application of the CLAUDE.md policy).
Lessons: two agreeing independent estimators beat one reshaped metric —
and when a fixture needs a beater click before the tracker behaves like
it does on real kicks, the fixture was the problem, not the tracker.

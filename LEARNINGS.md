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
- Minor dead weight (all removed in the Stage 14 dead-code sweep): the
  DSP-dead `hop_synthesis` field; the dormant identity envelope
  "correction"; ~200 lines of production-dead seam re-anchoring pinned only
  by its own tests; three `bessel_i0` copies; stale
  `Makefile`/`qa/README.md`/`RESEARCH.md` references.

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

**Honest low-confidence display follow-up (2026-08-12, PR #44 +
review fixes).** Estimator disagreement on plausibly quantized material
now caps the stored artifact confidence at 0.5 via an explicit
`BeatGrid::phase_untrusted` verdict, and the desktop dims the grid below
0.6. Somebody To Love: artifact confidence 0.845 → 0.500 (its grid
measures beat F 0.31 — the old 0.845 was internal consistency, not
ground truth). Versions bumped 9 → 10 (second application of the
CLAUDE.md policy — the first cut of the PR shipped without the bump).
Two lessons the review caught before merge:

- **Verify a verdict at the layer its consumer reads.** The first cut
  capped `BeatGrid.confidence`, but the artifact stored
  `estimate_confidence(...).max(grid.confidence)` and the desktop
  displays the artifact value — interval regularity alone scores ~0.85
  on a smoothly wrong grid, so the max silently reinstated the
  confidence the cap had just revoked, and the feature was inert on the
  exact track it existed for. The commit's own verification number
  (0.772 → 0.500) was measured at the wrong layer. Verdicts that must
  survive downstream aggregation need to travel as explicit state
  (the flag), not be smuggled through a value other code maxes over.
- **A shared gate should be one expression.** The cap's "ramps never
  reach here" comment was wrong (the capped branch returned before the
  sanity floor ran) — a tempo ramp fails both adoption gates too and
  was wrongly capped. Fixed by gating the cap on the same smeared-score
  ratio the adoption floor uses, extracted so "plausibly quantized"
  means one thing in both places; ramp and drifting-tracker controls
  now pin confidence and flag from both sides.

Independent corroboration (QM baseline column, same day): QM's
bar/beat tracker also fails on Somebody To Love (beat F 0.37 vs our
0.31, both far below the ≥ 0.95 it and we score on confidently-gridded
rows) — the honest-uncertainty verdict, not a tracker deficiency.

## Stage 12 — Robustness Hardening (2026-08-13)

Adversarial input, no-panic, and soak coverage for every non-RT surface
(PRs #46, #48, and the completion PR). Deliberately no DSP changes.

**Landed:**

- `qa/robustness.rs` — seeded adversarial harness (cargo-fuzz stand-in,
  deterministic xorshift, no nightly) over the `.tsa` loader, deprecated
  JSON artifact, WAV parsing, and `stretch()`/`stretch_offline` under
  hostile params × degenerate audio. `qa/soak.rs` — randomized
  deck-gesture marathon (rides/nudges/snaps, warm-start seeks, keylock
  toggles, artifact swaps) gating underruns, finiteness, clicks, and
  **bounded drift**: source consumption tracks the commanded tempo
  integral within a constant 2048 frames (measured worst 197 frames =
  4.5 ms over hour-equivalent 60 s runs — the engine's stash is bounded,
  drift does not accumulate). Both run per-PR in the CI quality gates;
  a weekly re-seeded campaign (`fuzz-campaign.yml`,
  `TIMESTRETCH_FUZZ_SEED` = run id, logged for reproduction) runs the
  hour-equivalent soak plus the full adversarial sweep.
- Three real fixes found by writing the harnesses, each with a minimized
  regression test: `.tsa` PEAK bucket-count multiply overflow → checked
  `Err`; `stretch_offline` release-build infinite loop on a partial
  trailing frame plus missing channel/ratio validation (the old guard
  was a `debug_assert!` — release had nothing); varispeed
  `MAX_OUT_PER_FEED` missing the kernel-release term, silently
  truncating output on retargets from dilated kernels in release.
- No-panic contract + audit documented in `lib.rs`: all 13 explicit
  panic sites in `src/` outside tests are invariant-local; implicit
  panics on input surfaces are covered by the harness, not static
  audit.

**Lessons:**

- A feature-gated test target has ZERO CI coverage until a workflow
  names it — `cargo test --all-targets` skips `required-features`
  targets silently, and clippy doesn't even build them. PR #46 shipped
  1,200 lines of harness that CI never compiled; the review caught it.
  New gates land WITH their CI wiring, not before.
- The most valuable robustness bug was not in a parser: the varispeed
  capacity bound was an RT-contract hole (silent truncation swallowed
  by a `debug_assert!`). Adversarial thinking applies to internal
  contracts, not just input surfaces.
- Consumption-vs-tempo-integral is a cheap, strong soak invariant: it
  needed only the frames-pushed counter and ring occupancy, and its
  measured headroom (10x) still catches a one-frame-per-callback leak
  within ~6 s of audio time.

## Stage 17 — Pitch-Shift and Batch Resampler Correctness (2026-08-13)

Batch anti-aliasing (PR #42) plus the direction-inversion fix its
independent review surfaced (PR #47). Off the DJ hot path; public API
quality.

- **Anti-aliasing (PR #42)**: `resample_sinc` cutoff-scales when
  downsampling with the streaming kernel's stopband margin
  (`cutoff_for_step` policy), Kaiser window via a per-call lookup table
  (Bessel out of the tap loop); `AudioBuffer::resample` switched from
  unfiltered cubic to the band-limited sinc. Measured: 2:1 downsample
  alias rejection 1.9 → 89.8 dB; 48→44.1 ultrasonic fold gated; the
  short-input cubic fallback is a documented conscious edge (a signal
  shorter than the kernel cannot be band-limited by it).
- **Direction inversion (PR #47)**: `pitch_shift` set
  `stretch_ratio = 1.0 / pitch_factor` — a length ratio — so factor 2.0
  rendered an octave DOWN (probe: 440 Hz → 220 Hz), inverting the
  documented contract, and the real pitch-raising path aliased in-band
  at −1.2 dB rejection pre-#42. Fix: `stretch_ratio = pitch_factor`;
  the formant tapers were already written to the documented semantics
  and needed no change. Gates added: 10x dominance octave tests both
  ways, 44.1↔48 round-trip SNR > 40 dB (multi-tone).
- **Owner A/B (2026-08-13, renders from main `0e73acf`,
  `target/stage17_audition/`: shipped vs pre-fix-equivalent unfiltered
  cubic, +2/+5 semitones on a bright disco excerpt, RMS-matched,
  common-trimmed float)**: "the shipped drums sound higher quality, but
  the difference is still very small and hard to hear without a trained
  ear." Exit criterion met — the shipped path is audibly clean and the
  old path's aliasing, while real, was subtle on dense material.

Lessons:

- **An OR-assertion direction test is satisfiable by inverted output**
  through its own harmonics — `e_880 > e_220 || e_880 > e_1760` passed
  for months while factor 2.0 rendered 220 Hz. Direction tests must
  assert the target DOMINATES both the source and the
  opposite-direction frequency.
- **Probe the direction you claim**: the original "the shifter is not a
  beneficiary" scope note tested pitch-up as factor > 1, which — being
  inverted — never downsampled; the measurement was true and the
  conclusion backwards. A probe that can't fail the hypothesis both
  ways isn't evidence.
- Sine probes overstate audibility on dense mixes again (cf. Stage 16's
  purity numbers): −1.2 dB fold rejection sounds catastrophic on paper
  and was "very small" on a real club master. Characterize with tones,
  decide with ears.

## Stage 16 — Tonal-HF Granulation: Measure, Then Decide (2026-08-13)

The review's headline question answered by blind listening. Audition set:
PR #43 (renderer + purity characterization) with PR #49's validity fixes
(stereo PV arms, RB RMS match, no-clip common trim); renders from
`7f49a50`; 3 excerpts × ±4/±8% × 4 arms, letters shuffled per condition
(`BLIND_KEY.json` opened after the session).

**Characterization (pinned in `tests/tonal_purity_characterization.rs`):**
the granulation floor is strongly ASYMMETRIC — harmonic-15 purity 52 dB
at +8% tempo but 22 dB at −8% (slowdowns re-cross material, so splices
recur through sustained content); two-tone beating survives well both
ways (47/42 dB).

**Blind verdict (owner, 2026-08-13, 12 conditions):**

- **Q1 — granulation is audible in context.** Rubber Band was the
  cleanest arm in 9/12 conditions. The shipped SOLA path won twice
  (hot_stuff −4%, msbwy +4%), stayed "clean/good quality" on most of
  hot_stuff and cold_heart, but degraded to "roboty, bad quality" /
  "awful" on msbwy ±8% — sustained filtered strings, the exact
  content class the asymmetric floor predicts. Not a documented-scope
  close: the structural response is scoped as Stage 18
  (tonality-adaptive splice cadence), falsification-gated.
- **Q2 — Stage 7 RE-CONFIRMED on clean evidence.** The phase-fixed
  PV-512/1024 arms behind the 120 Hz split were the "robotic /
  underwater / vocoder" arms blind — the same artifact vocabulary that
  condemned the defective PV in July, now heard with the Stage 13
  fixes in and without knowing which arm was which. pv512 was the
  worst arm in most conditions; pv1024 followed. "SOLA carries the
  corrected range" now rests on an uncontaminated verdict, and the
  HF-small-PV-band candidate is dead. (Recorded bias: the PV arms ran
  without artifact resets, but the verdicts cite sustained-material
  phasiness, not transient smear — the deviation does not rescue
  them.)
- Side observation for Stage 14's listen: Rubber Band's msbwy wins came
  with "missing stereo, narrow, muted" notes — R3 trades image width
  for smoothness on that material; ours keeps the width.

Lessons:

- The blind protocol earned its cost: the letters Rob ranked cleanest
  were RB in 9/12 without knowing which arm was which, and the PV arms
  reproduced their July vocabulary exactly — both questions settled
  with zero suggestion risk.
- A listening verdict really is evidence about the implementation
  auditioned (the 2026-08-05 lesson, closed out): the July anti-PV
  verdict survived the re-audition, but only the re-audition made it
  citable — before it, every SOLA-vs-PV decision cited contaminated
  evidence.

## Stage 18 — Steady-Rate Splice-Cadence Stretch (2026-08-13)

The Stage 16 verdict's structural response, falsification-first
throughout (kill-experiment branch `proto/stage18-splice-cadence`;
build-out PR #56).

**Mechanism shipped**: SOLA's elastic drift triggers (normal /
opportunistic / hard) double at STEADY transposition inside the primary
DJ window — splices on sustained tonal material fire half as often and
land with larger, better period-aligned jumps. Three guards, each pinned
by the gate that motivated it:

- **Transposition taper (T 1.09 → 1.15)**: slowdown drift eats the
  560-frame nominal-lag gap to the write head; the prototype measured a
  cursor stall (sine −16 cents) at T=1.25. Ratio 1.25 in the quality
  sweep is the regression pin.
- **Steady-rate gate (`rate_slope == 0`)**: with the stretch active on
  rides the A/B matrix measured cents p95 5.65 vs the 1.5 bound — the
  slope-tracked pitch correction is drift-proportional and clamps.
  Rides keep shipped cadence; `ride_slope_restores_shipped_cadence`
  pins it.
- **Asymmetric force band (review-caught, HIGH)**: the stretched hard
  trigger (640) sits PAST the write head (lag 560) on slowdowns — under
  blocked-splice pressure the forced backstop could never fire before
  the read margin was violated. `SLOWDOWN_FORCE_CAP = 448` caps the
  force on negative drift (speed-ups keep the full band against ~6x
  ring headroom), with a write-head-side construction assert. The same
  review added the failed-attempt guard: a forced splice (which tears
  through onset protection) is honored only after an unforced attempt
  failed, so a ride starting under parked stretched-band drift cannot
  skip protection on its first block.

**Measured**: harmonic-15 purity −8% 22.1 → 62.8 dB, +8% 52.0 →
59.5 dB (asymmetry gone; floors re-pinned 12 → 45 dB); tone-pair +8%
47.3 → 62.1 dB, −8% unchanged. Splice cadence halved (pinned by unit
test). Full suite, A/B matrix, ride/seam, WCET, robustness, soak green.

**Blind owner listens** (sealed keys, same excerpts as Stage 16):

- Old vs new (6 conditions): new won or tied ALL — every "robot" rating
  landed on the old cadence.
- Exit listen, new vs Rubber Band (6 conditions): RB still cleanest on
  all six, but the granulation vocabulary collapsed (one "very subtle
  robot" across the set, vs Stage 16's "roboty bad quality/awful").
  The residual complaints are a DIFFERENT artifact class: "bass
  sometimes out of key" on 3/6 (±8%, bass-forward msbwy) — the
  sub-120 Hz pitch-follow scope line (±1.3 semitones against the
  corrected highs), audible now that granulation no longer masks it.
  Owner decision: stage closed as achieved; the bass finding is
  recorded against the scope line (ROADMAP Architecture note +
  Not a Priority Yet) for any future re-litigation — a time-domain
  low-band corrector was never falsified. Watch items recorded: msbwy
  "drums bitcrushed/smear a tiny bit" (transient gates green; may be
  relative-to-RB judgment) and an ours-narrow/RB-wide stereo note on
  cold_heart −4% that reverses Stage 16's reading — recheck during the
  Stage 14 listen.

**Lessons**:

- Fixing the loudest artifact re-litigates the ones it masked: the
  bass-detune scope line was inaudible under granulation and surfaced
  blind the moment the splices got clean. Budget for the next layer
  when closing a masking artifact.
- The A/B matrix caught the ride-pitch interaction and the independent
  review caught the write-head collision — neither appeared in the
  prototype's evidence because sines without onsets never block a
  splice and never ride. Synthetic evidence validates the mechanism;
  gates and adversarial review validate the envelope.
- A physical headroom bound beats a behavioral heuristic: the taper,
  the force cap, and the construction asserts all derive from the same
  560-frame lag geometry, which is why the mechanism ships with no
  tuning knobs.

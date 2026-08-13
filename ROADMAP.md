# Roadmap

## Goal

`timestretch-rs` powers a DJ deck that feels like hardware: tempo control at
resampler latency, ≤ 15 ms pipeline delay on the primary chain, transparent
keylock at DJ ratios, honest latency contracts everywhere, and one engine
whose live output is the same audio the quality benchmarks measure.

The engine rebuild that was this roadmap's original subject is **complete**
(Stages 1–9, July 2026; wide-range Master Tempo, August 2026). Completed
stages and their evidence live in [LEARNINGS.md](LEARNINGS.md); full stage
texts are in git history (tag `v0.10.0` has the last pre-rewrite version).

What remains is quality closure, not architecture: an August 2026
full-code review against Rubber Band (findings recorded in LEARNINGS.md)
identified confirmed defects in the wide-ratio phase vocoder, a
mischaracterized offline wide path, ride-polish gaps in the DJ chain, and a
measured tonal-HF granulation floor that needs a listening verdict — plus
the still-open analysis-trust track (Stage 10); the robustness track
(Stage 12) completed 2026-08-13.

## Status (2026-08-05)

Shipped and settled: pull-based stage-graph engine (Tape / Keylock /
WideKeylock profiles), SOLA keylock through ±20% at 12.7 ms, wide-range
Master Tempo at 48.6 ms, batch `stretch()` on the same graph with
sample-exact duration and streaming/offline determinism, artifact-first
transient control, `.tsa` analysis container (v0.11.0), musical key
detection, rigid beat grids for quantized material, machine-verified RT
contract (zero-alloc, WCET-gated), Rubber Band reference gate in CI.

Stage 13 (wide-path phase hygiene) completed 2026-08-06 — four confirmed
correctness bugs fixed, null SER 59→139 dB, owner ±50% verdict "significantly
better," Rubber Band gap narrowed (R3 still ahead); archived in LEARNINGS.md.
Stage 15 (DJ-band ride polish) completed 2026-08-07 (PRs #37/#38) — fade-band
clicks gated, seam comb during sustained mild rides −7.1→−4.4 dB with the
mild-motion bounded recenter, ride-quality harnesses in CI, owner mix-in
listen passed; archived in LEARNINGS.md. Its optional items (correlation
reference, strength gating, modulation_hold wiring) remain evidence-gated
ideas, not scheduled work.
Stage 12 (robustness hardening) completed 2026-08-13 (PRs #46/#48 + the
completion PR) — adversarial harness and deck-gesture soak in the CI
quality gates, bounded-drift gate (worst measured 4.5 ms over an
hour-equivalent), no-panic audit clean, weekly re-seeded fuzz campaign;
three real fixes landed on the way; archived in LEARNINGS.md.

Open, in priority order for the DJ app:

- **Stage 10** — beat-grid trust on everything a DJ loads (non-EDM
  corpus evidence half; the non-adopter class is closed).
- **Stage 14** — wide-path consolidation: behavior landed, owner ±30/±50
  listen remaining.
- **Stage 18** — tonality-adaptive splice cadence (the Stage 16 verdict's
  follow-up; falsification-gated).

Stage 16 (tonal-HF granulation: measure, then decide) completed
2026-08-13 — blind session on the validity-fixed set (12 conditions,
renders from `7f49a50`): granulation IS audible in context (Rubber Band
cleanest 9/12; ours degraded to "roboty" on the worst sustained-tonal
slowdowns, competitive elsewhere), and the corrected-range re-audition
RE-CONFIRMED Stage 7 — the phase-fixed PV was still the
"robotic/underwater/vocoder" arm blind. Verdict recorded, follow-up
scoped as Stage 18; archived in LEARNINGS.md.
Stage 17 (pitch-shift/batch-resampler correctness) completed 2026-08-13
(PRs #42/#47) — batch anti-aliasing (2:1 alias rejection 1.9 → 89.8 dB),
the `pitch_shift` direction inversion found by review and fixed, gates in
CI, owner bright-mix A/B passed (shipped drums "higher quality", the old
path's artifact subtle); archived in LEARNINGS.md.

## Architecture (settled — decisions, not open questions)

- **Stage-graph engine** in `src/engine/`: fixed-block stages
  (`process`, `latency_frames()`, `reset()`, `prime()`), fixed per-profile
  chains, varispeed head owning demand inversion.
- **Varispeed-first.** source → sinc varispeed (tempo axis, sample-accurate
  retargets, no control glide) → per-profile correction.
- **Keylock profile** (primary deck): LinkwitzRiley8 split at 120 Hz; low
  band delay-matched, **not** pitch-corrected (pitch follows tempo — the
  Stage 2 falsification verdict); high band corrected by elastic-cursor
  SOLA. Full keylock through ±20%, release fade 20.5%→35%, 560-frame
  (12.7 ms) contract.
- **WideKeylock profile** (opt-in range setting): full-spectrum FFT-2048 /
  hop-256 identity-locked PV + post-resampler, artifact-driven per-band
  phase resets, 2144-frame (48.6 ms) contract. Profile switch is a
  seek-priced rebuild, never a live morph.
- **Artifact-first analysis**: the `PreAnalysisArtifact` drives splice
  protection and phase resets; online detection is the fallback.
- **Single engine, both modes**: offline is the same graph with unlimited
  lookahead and a guaranteed artifact; streaming-vs-offline agreement is a
  determinism property.
- **RT contract**: pull API, no `Result` and no allocation in the audio
  path, WCET-gated, honest per-profile latency reporting.

**Evidence caveat (2026-08-05).** Two of these decisions were settled by
listening against a phase vocoder that carried the correctness defects
Stage 13 fixes — the unwrapped-phase blends in particular manufacture
exactly the "phasey" artifact that condemned it:

- *"SOLA carries the entire corrected range"* (Stage 7 verdict: the
  small-FFT PV was audibly phasey at every boundary it was placed behind).
- *The accepted wide-rate Rubber Band gap* (Stage 11 verdict: audibly
  behind R3, shippable).

Both verdicts stand as shipped behavior. The second was re-baselined at
Stage 13's exit listen (2026-08-06, commit `fb7dcfa`+sidecar rerun): the
fixed PV sounds "significantly better" at ±50% and the gap to R3
narrowed, though R3 remains ahead — the acceptance holds with fresher,
smaller evidence. The first was RE-CONFIRMED on clean evidence at the
Stage 16 blind re-audition (2026-08-13): the phase-fixed small/medium-FFT
PV behind the split was still the "robotic/underwater/vocoder" arm —
"SOLA carries the corrected range" now rests on an uncontaminated
verdict. The un-keylocked low band is **not**
similarly contaminated — its load-bearing justification at DJ ratios is
the ≤ 15 ms latency budget (a bass-resolving FFT cannot fit), which no
review finding touches; Stage 11's full-spectrum result already shows
corrected bass can win when latency permits.

## Binding Policies

- **EDM/DJ-first**: quality gates are DJ material at DJ ratios (0.92–1.08
  primary, ±20% secondary), streaming path first. The crate's customer is
  the author's DJ application; the public API breaks freely pre-1.0.
- **Owner listening is the binding quality gate.** Metrics are regression
  tripwires — they have twice failed to predict ear verdicts (LEARNINGS.md,
  Stage 11). Every quality-affecting stage ends with a recorded listen.
- **Falsification first**: risky bets get a cheap kill-experiment with a
  named fallback before build-out.
- **The accepted scope lines stay accepted until re-litigated with
  evidence**: sub-120 Hz follows tempo at DJ ratios; the wide-rate Rubber
  Band gap stands, re-baselined smaller at Stage 13 (2026-08-06: R3
  still ahead at ±50%, but the fixed PV is "significantly better" than
  the Stage 11 renders).

## Principles

- Fix structure instead of stacking corrective heuristics.
- Every stage ends with the desktop app audibly playing the change —
  vertical slices, never plumbing-only stages.
- CI stays green throughout; new gates land with the stage that motivates
  them.

## Stage Sequence

Stages are independent tracks. (Stages 12, 13, 15, 16, and 17 are
complete; Stage 16's verdict spawned and gates Stage 18.) Suggested
order: 10, then 14, 18.

## [ ] Stage 10: General-Purpose Beat Tracking and BPM Detection

Automation: auto

> **Status (2026-08-05): algorithm side landed and extended; evidence half
> and the non-adopter class remain.** Landed since the original stage text:
> tempogram + DP tracker (`src/analysis/tempogram.rs`,
> `src/analysis/beat.rs`), tempo segments + downbeats in the artifact,
> beat-level metrics (F-measure ±70 ms, continuity) in `qa/bpm_accuracy.rs`
> in CI, desktop grid overlay with grid-accurate jumps/loops, **rigid
> kick-band grid fitting for quantized material** (v0.10.0: beat F
> 71.5%→93.8%, downbeat F 20.7%→76.4%; adoption gated on
> `phase_lock ≥ 0.3`), and hand-corrected annotations wired into
> `benchmarks/manifest.toml` (13+ tracks). Caveat recorded in LEARNINGS.md:
> the beat-F gain is partly self-confirming — annotations share the
> rigid-fit method; ear verification stays the honest check.

### Why

The DJ app needs trustworthy grids on everything a DJ loads. The corpus is
still all-EDM (115–140 BPM club tracks), so the generalization claims are
unproven, and the known rigid-fit non-adopters (MSBWY, Hot Stuff, Somebody
To Love — offbeat disco bass competing with the kick phase) ship visibly
wandering grids on quantized material (diagnosed 2026-08-03: Hot Stuff fits
120.000 BPM exactly but phase_lock = 0.109, so the DP grid with 475–569 ms
intervals draws instead).

### Work

- **Corpus evidence half**: non-EDM entries with annotated grids (hip-hop
  ~90, DnB ~174, at least one live-drummer recording); explicit
  acc2/F-measure floors on the non-EDM subset in CI; ~~the QM Vamp
  baseline column~~ **done 2026-08-12** (output-only — qm-dsp is GPL,
  never source: `benchmarks/generate_qm_baseline.py` stores
  qm-barbeattracker beat times as JSON baselines, and the harness
  reports per-row `qm_beat_f` plus a worst-row margin with an optional
  `TIMESTRETCH_BPM_QM_MARGIN` floor; on the four locally-present EDM
  rows our tracker wins every confidently-gridded row by 4.4–5.0pp,
  and QM edges us +6.2pp only on Somebody To Love (beat F 0.37 vs
  0.31 — the annotated-phase-indecisive track where both estimators
  disagree with the annotation, consistent with its honest
  low-confidence handling). QM baselines for the non-EDM entries
  generate with the same script when that audio lands. The synthetic
  tempo-ramp fixture is ~~done~~ (2026-08-07, in-harness, no committed
  audio).
- ~~Offbeat-bass disambiguation~~ **done 2026-08-07** (branch
  `feat/rigid-grid-offbeat-disambiguation`): both roadmap candidates
  (slot exclusion, onset-sharpness weighting) were prototyped and failed
  on the class — swung rivals sit at scattered subdivision phases.
  Landed instead: **tracked-beat corroboration** — a below-threshold fit
  adopts when ≥ 60% of the DP tracker's beats (an independent estimator)
  land on the rigid grid within 25 ms. Corpus: MSBWY and Hot Stuff now
  adopt (beat F 0.95/0.93 → 1.00, downbeat F → 1.0, offsets sub-ms);
  33rd Rate Revs X downbeat F 0 → 1.0; Somebody To Love honestly stays
  out (agreement 0.28 — genuine estimator disagreement, the "annotated
  phase-indecisive" bucket); a tempo-ramp control that phase_lock alone
  would wrongly trust at 0.77 is rejected at 0.25. The 0.3 gate itself
  is untouched. PREANALYSIS_VERSION → 9/9 per the CLAUDE.md policy.
  **Desktop owner check passed 2026-08-07** ("the grids are spot on" —
  Hot Stuff/MSBWY markers dead-on after sidecar regeneration). The
  synthetic tempo-ramp fixture also landed (merged with PR #39): 120→132
  BPM ramp with exact ground truth, scored through the shipping path in
  the CI harness — beat F 0.956 vs the 0.85 floor, continuity 1.000,
  tempo-trend asserted.
- ~~Artifact invalidation policy~~ **done 2026-08-06**: versions bumped
  to 8/8 so ambiguous v4–v7 sidecars regenerate; policy written into the
  `src/core/preanalysis.rs` constant docs and RELEASE_CHECKLIST.md (which
  was also rebuilt — it still referenced the pre-cutover engine).
- Desktop owner check: grids visually aligned on real non-EDM tracks.

### Exit Criteria

- acc1/acc2 on the widened corpus ≥ current EDM-subset scores; non-EDM
  floors enforced in CI (proposed acc2 ≥ 90%, beat F ≥ 0.85).
- Tempo-ramp fixture and live-drummer recording tracked within tolerance;
  no row where the QM baseline wins by more than noise.
- ~~The three known non-adopters~~ two of three adopt with sub-ms offsets
  (2026-08-07); Somebody To Love is measured as genuine estimator
  disagreement — ~~annotate it as phase-indecisive and verify the
  desktop shows an honest low-confidence grid for it~~ **done
  2026-08-12** (PR #44 + review fixes): annotation marked
  `phase_indecisive`, estimator disagreement on plausibly quantized
  material caps stored confidence at 0.5 via an explicit
  `BeatGrid::phase_untrusted` verdict (artifact-level measured 0.845 →
  0.500, beat F 0.306; confident rows hold 0.88–0.91; ramps stay
  uncapped — gated on the adoption sanity ratio), and the desktop dims
  the grid and says "grid: low confidence" below 0.6. Versions bumped
  9 → 10 so cached wrongly-confident sidecars regenerate. Owner visual
  check on the real track pends the next desktop load (sidecars
  regenerate on open).
- Version-bump policy documented and applied on the next analysis change.

## [ ] Stage 14: Wide-Path Consolidation and Stereo Coherence

Automation: auto

> **Status (2026-08-07): behavior half landed** (branch
> `feat/wide-path-consolidation`): offline wide ratios inside the engine
> range now render through the shipped `WideKeylockStage` (the batch PV
> survives only beyond 4× either way); streaming-vs-offline determinism
> extended to wide rates (sample-identical at 0.5×/1.5×); the two-tone
> sub-bass attenuation pinned at Stage 13 is GONE through the live stage
> (balance at the ideal at every wide ratio — the live PV runs at unity
> with the resampler transposing); the corrected stereo path runs in
> mid/side (identical channels stay bit-identical, center cannot leak
> into side by construction — honest measurement: the per-channel leak
> was modest, 64.6 → 70.9 dB rejection, so the audible width verdict
> belongs to the owner listen); peak-magnitude floor unified across the
> vocoder and locking passes. Remaining: the dead-code/doc sweep
> (separate PR) and the owner ±30/±50 listen (note: M/S touches exactly
> the "a bit crowded vs R3" quality heard at Stage 11).

### Why

Offline wide ratios run a *different algorithm configuration* than the
shipped live wide stage — independent per-channel batch PVs with no stereo
coupling, no streaming resampler, and (until Stage 13) the rejected hop.
`lib.rs:493-494` claims batch stereo phase is preserved "natively"; for
this path it is not. Meanwhile the wide PV carries dead weight the review
inventoried: a provably-no-op envelope block, ~200 lines of
production-dead seam re-anchoring pinned by ~25 tests, a DSP-dead
`hop_synthesis` field, divergent peak-detection passes, and the legacy
broken `LR8Crossover` kept only for baselines that no longer exist.

### Primary Files

- `src/engine/offline.rs` (route wide ratios through `WideKeylockStage`),
  `src/stretch/phase_vocoder.rs` + `src/stretch/phase_locking.rs`
  (peak-set unification, dead-code removal), `src/stretch/envelope.rs`
  (delete or wire for real), `src/core/crossover.rs` (`LR8Crossover`
  removal), stale docs (`Makefile`, `qa/README.md`, `RESEARCH.md:309`,
  `benchmarks/README.md`)

### Work

- **Offline wide = live wide**: drive `stretch_offline`'s wide branch
  through the actual `WideKeylockStage` graph (host-emulation loop already
  exists for the keylock branch), giving the shipped corrector batch/QA
  coverage and making live and offline wide renders match. Supersedes the
  per-channel batch PV loop.
- **Stereo coupling in the wide PV**: shared peak selection and shared
  per-peak rotations across channels (rotation from the channel-summed
  spectrum, applied to both; magnitudes stay per-channel). Gate with a
  correlated-stereo-noise inter-channel phase test.
- **Unify the peak sets**: one thresholded peak pass feeding gradient
  integration, locking, and the IF-blend loop (`MIN_PEAK_MAGNITUDE` floor
  in `fill_spectral_peaks`); document last-write-wins region arbitration or
  fix it by magnitude.
- **Dead-code removal**: `hop_synthesis` field + stale doc comments; the
  envelope-preservation block (identity by construction — delete, with the
  cepstral machinery, unless a real synthesis-envelope source is wired);
  the `set_stretch_ratio` seam-case chain and its tests (production always
  uses `set_smooth_ratio_updates(true)`); `LR8Crossover`; two of the three
  `bessel_i0` copies.
- Doc truth pass: fix `Makefile` `TEST_CMD`, `qa/README.md` /
  `benchmarks/README.md` stale harness names, `RESEARCH.md` dormant-reset
  claim, `lib.rs` stereo claim.

### Exit Criteria

- `stretch()` beyond ±20% renders through `WideKeylockStage`;
  streaming-vs-offline determinism extended to a wide-rate fixture.
- Inter-channel phase gate green on stereo material at wide ratios.
- Net LOC down in `src/stretch/`; no production behavior change on the DJ
  chain (A/B matrix re-run green).
- Owner listen at ±30/±50% vs pre-consolidation renders.

## [ ] Stage 18: Tonality-Adaptive Splice Cadence

Automation: auto

> **Status (2026-08-13): kill-experiment SURVIVED — build-out is on.**
> Prototype (branch `proto/stage18-splice-cadence`, env-gated trigger
> scale, bit-identical when off): halving the splice cadence (scale 2 on
> the elastic drift triggers) lifted harmonic-15 purity at −8% tempo
> from 22.1 → **62.8 dB** (and +8%: 52.0 → 59.5 dB) — the asymmetric
> floor is gone, splice counts confirm the cadence halved. **Blind owner
> A/B (2026-08-13, msbwy + cold_heart × −8/−4/+8%, sealed key): the
> half-cadence arm won or tied all 6 conditions**, and every "robot"
> rating landed on the shipped cadence — the Stage 16 artifact
> attenuates exactly as the numbers predicted. One watch item: a
> background bass hum on msbwy ±8% in BOTH arms (likely the
> pitch-following low band on that track, not cadence-caused), slightly
> more exposed at half cadence on +8% — a build-out gate concern, not a
> kill. Two design facts from the prototype: (1) scale 4+ REGRESSES on
> slowdowns and scale 2 breaks at the ±20–25% range edge (sine pitch
> 218 vs 220 Hz at ratio 1.25) — both are the drift bound colliding with
> the 560-frame nominal lag, so the shippable mechanism is a
> **transposition-aware trigger scale** (full stretch in the primary
> window, tapering to 1 as headroom shrinks toward the range edge), NOT
> a fixed scale and NOT more latency; (2) a flat scale already wins on
> tonal material with ride/keylock gates green, so band-flux tonality
> detection is deferred unless the A/B matrix or transient gates object
> — simplest structure first.

### Why

The Stage 16 blind verdict (2026-08-13): SOLA's tonal-HF granulation IS
audible in context — Rubber Band was the cleanest arm in 9/12 blind
conditions, and the shipped path degraded to "roboty, bad quality" on
the worst sustained-tonal slowdowns (msbwy ±8%) while staying
competitive on brighter material (cleanest on 2 conditions). The same
session re-confirmed Stage 7 on clean evidence: the phase-FIXED
small/medium-FFT PV behind the split was still the "robotic /
underwater / vocoder" arm blind, so replacing SOLA with a PV in the
corrected range stays dead. The stage-16 decision tree therefore lands
here: keep SOLA, make its splice cadence tonality-aware — sustained
tonal material is exactly where splices recur through the same content
(~20/s, the measured 22 dB floor at −8%) and where the artifact's band
flux says "nothing is happening, don't splice".

### Falsification first (cheap kill-experiment before build-out)

Offline prototype only, batch graph + the stage-16 excerpt set: scale
SOLA's drift trigger by a tonality factor derived from the artifact's
band flux (high sustained-band energy, low flux → stretch the trigger;
transient-dense → unchanged). Render the msbwy/cold_heart slowdown
conditions and A/B against the stage-16 renders. Kill criteria: if
halving the splice rate on tonal passages does not move the ear verdict
(or trades it for seam drift/pitch wobble — the Stage 15 lesson says
the landing bound, not the trigger level, is load-bearing), record the
negative result in LEARNINGS.md and fall back to
acceptance-at-DJ-ratios with the wide profile as the documented escape
hatch.

### Work (kill-experiment survived; revised by its evidence)

- **Transposition-aware trigger scale** as shipped behavior (no env
  knob): scale 2 inside the primary DJ window, derived-from-constants
  taper to 1 as the drift bound approaches the 560-frame lag headroom
  toward the range edge (the ratio-1.25 sweep failure is the pin for
  where the taper must have fully landed). The hard/force trigger must
  always remain physically reachable — the prototype showed a force
  threshold beyond the write-head stall point is a latent parking risk.
  Mild-motion and rest recenters keep their own bounds (Stage 15).
- Band-flux tonality adaptivity: DEFERRED — only if the A/B matrix or
  transient gates object to the flat in-window scale.
- RT contract untouched (the scale is a per-retarget scalar, no
  allocation).
- Gates: purity characterization re-pinned at the improved floors
  (both directions ≥ ~55 dB); ride/seam/A-B-matrix suite green;
  ratio-sweep quality regressions green through the full range edge
  (the 1.25 case becomes the taper's regression test); splice-rate
  telemetry on the corpus recorded before/after; the msbwy bass-hum
  observation checked (must not worsen vs shipped).
- Exit: blind owner re-listen on the stage-16 protocol — the msbwy/
  cold_heart slowdown verdicts move toward Rubber Band, no new
  artifacts elsewhere.

## Not a Priority Yet

- SIMD / architecture-specific acceleration (WCET gates exist to measure
  any attempt against; current headroom is comfortable).
- Cross-frame peak tracking / multi-resolution wide path — the RB-class
  coherence work. Revisit only if post-Stage-14 listening still shows the
  wide gap mattering in practice; Stage 13 already narrowed it (owner
  verdict 2026-08-06) and the acceptance stands.
- Desktop UI/UX polish beyond its role as the reference integration.
- Additional presets, wider API surface, convenience wrappers.
- General-purpose (non-EDM) *stretch* quality (analysis generality is
  Stage 10; stretch quality gates stay DJ-material).

## Path to 1.0 (decision pending — not scheduled)

Production grade **as a public library** is a separate road, deferred until
the owner decides the crate should take external customers:

- API freeze and semver discipline; rustdoc completeness; README latency
  table, RT contract, and artifact workflow as documented guarantees.
- The non-EDM stretch-quality question answered (gated or documented as a
  scope boundary — never silently variable).
- Quality sign-off bus factor: a second listener on the structured
  checklist; baselines sanity-checked on a second machine class.
- MSRV and platform policy stated in the README.

Stage 12, a prerequisite for this path, completed 2026-08-13.

## Definition of Success

The engine-rebuild definition (≤ 15 ms primary chain, one engine both
modes, zero corrective heuristics, machine-verified RT contract, external
reference evidence in CI, hardware-feel deck) **holds as of the Stage 9
cutover (2026-07-15)** and must keep holding. This roadmap is done when,
in addition:

- Trustworthy beat grids on everything a DJ loads, gated on an annotated
  corpus that includes non-EDM and variable-tempo material (Stage 10).
- The wide-ratio path is free of known correctness defects (Stage 13,
  done), its offline and live renders are the same algorithm, and stereo
  coherence is gated (Stage 14).
- Riding the fader degrades nothing that holding it steady doesn't
  (Stage 15, done — seam and fade gates hold in CI).
- The tonal-HF granulation floor has a recorded listening verdict
  (Stage 16, done 2026-08-13 — audible in context; structural response
  scoped as Stage 18, falsification-gated).
- No public path resamples without anti-aliasing (Stage 17, done
  2026-08-13 — gates in CI, owner A/B passed).
- No panic is reachable from the public API on arbitrary input (Stage 12,
  done 2026-08-13 — adversarial harness in CI + audit).

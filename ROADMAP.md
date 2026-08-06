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
the still-open analysis-trust (Stage 10) and robustness (Stage 12) tracks.

## Status (2026-08-05)

Shipped and settled: pull-based stage-graph engine (Tape / Keylock /
WideKeylock profiles), SOLA keylock through ±20% at 12.7 ms, wide-range
Master Tempo at 48.6 ms, batch `stretch()` on the same graph with
sample-exact duration and streaming/offline determinism, artifact-first
transient control, `.tsa` analysis container (v0.11.0), musical key
detection, rigid beat grids for quantized material, machine-verified RT
contract (zero-alloc, WCET-gated), Rubber Band reference gate in CI.

Open, in priority order for the DJ app:

- **Stage 13** — wide-path phase hygiene: four confirmed correctness bugs,
  small diffs, zero DJ-chain risk.
- **Stage 10** — beat-grid trust on everything a DJ loads (evidence half +
  the offbeat-bass non-adopters).
- **Stage 15** — DJ-band ride polish (seam recovery under continuous rides,
  fade-band zipper).
- **Stage 14** — wide-path consolidation (offline uses the shipped stage,
  stereo coupling, dead-code removal).
- **Stage 16** — tonal-HF granulation: measure audibility on real material,
  then decide accept/document vs pursue.
- **Stage 17** — pitch-shift/batch-resampler correctness.
- **Stage 12** — robustness hardening (fuzzing, no-panic, soak).

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

Both verdicts stand as shipped behavior, but their evidence is stale:
Stage 16 re-auditions the first on a post-Stage-13 PV, and Stage 13's
exit listen re-baselines the second. The un-keylocked low band is **not**
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
  Band gap ("audibly behind R3, shippable") stands per the Stage 11
  sign-off.

## Principles

- Fix structure instead of stacking corrective heuristics.
- Every stage ends with the desktop app audibly playing the change —
  vertical slices, never plumbing-only stages.
- CI stays green throughout; new gates land with the stage that motivates
  them.

## Stage Sequence

Stages are independent tracks except: 13 → 14 (hygiene lands before
consolidation rebaselines wide-path QA), 13 → 16 (the re-audition needs
the fixed PV), and 16's verdict gates any future DJ-band tonal-quality
stage. Suggested order: 13 first (hours, pure win), then 10 and 15 in
parallel, then 14, 16, 17, 12.

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

- **Corpus evidence half** (unchanged from the original stage): non-EDM and
  variable-tempo entries with annotated grids (hip-hop ~90, DnB ~174, a
  synthetic tempo-ramp fixture, at least one live-drummer recording);
  explicit acc2/F-measure floors on the non-EDM subset in CI; the QM Vamp
  baseline column (output-only — qm-dsp is GPL, never source).
- **Offbeat-bass disambiguation for the rigid fit**: the low phase_lock on
  disco-bass tracks comes from sub-150 Hz bassline onsets scoring competing
  phases. Candidates: weight the kick envelope by onset sharpness, or test
  whether the competitor phase sits specifically at a half/quarter-period
  offset before counting it against decisiveness. Do **not** lower the 0.3
  gate to chase these tracks (recorded lesson).
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
- The three known non-adopters either adopt correctly under the
  disambiguation work or are annotated as genuinely phase-indecisive with
  the desktop showing an honest low-confidence grid.
- Version-bump policy documented and applied on the next analysis change.

## [ ] Stage 12: Robustness Hardening — No-Panic Surface, Fuzzing, Soak

Automation: auto

### Why

The RT contract is machine-verified, but the input surface has never been
hardened: the `.tsa` container loader (`src/io/tsa.rs`), the deprecated
JSON artifact loader, the WAV reader, and the batch API have no fuzz
coverage, and "arbitrary input produces `Err`, never a panic" is an
intention, not a tested property. Prerequisite for any 1.0; the DJ app
benefits regardless. Touches no DSP.

### Work

- Fuzz targets: `.tsa` from arbitrary bytes; deprecated JSON artifact; WAV
  parsing; `stretch()` driven by arbitrary params × degenerate audio
  (NaN/Inf/denormals, zero-length, one sample, extreme rates).
- No-panic policy documented in `src/lib.rs`; audit of
  `unwrap`/`expect`/`panic!` outside construction and tests.
- Long-run soak: hours-equivalent randomized deck gestures (rides, seeks,
  loop wraps, profile/keylock toggles, artifact swaps) gated on zero
  clicks, zero allocation, bounded drift — composes the existing torture
  generators.
- CI: bounded fuzz per PR, longer cron run with corpus persistence;
  crashes minimize into regression tests.

### Exit Criteria

- All fuzz targets clean for the CI budget; every campaign crash lands as
  a minimized regression test.
- Machine-checked: no panic reachable from the public API on arbitrary
  input.
- Soak harness green in CI (bounded), full-length recipe documented.

## [ ] Stage 13: Wide-Path Phase Hygiene

Automation: auto

### Why

The 2026-08-05 review confirmed four correctness bugs in the wide-ratio
phase vocoder (evidence and line references in LEARNINGS.md). All are
small, mechanical fixes that exclusively touch the wide path — the shipping
DJ ±20% chain has zero regression exposure — and they aim directly at the
one quality gap the project has formally conceded ("audibly behind R3 at
all wide rates"). This is the cheapest real quality work in the repo.

### Primary Files

- `src/stretch/phase_vocoder.rs` (accumulator wrap, blend fixes,
  DC/Nyquist), `src/engine/offline.rs` (hop constant), `src/lib.rs`
  (`stretch_into` bypass parity), new `tests/pv_null.rs`

### Work

- **Wrap `phase_accum`** into (−π, π] after each accumulation (three sites
  + seeds). Kills the f32-downcast precision decay on long
  streams/renders; makes the `:1555` downcast harmless.
- **Wrapped-difference blends**: replace both linear phase mixes with
  `φ₁ + b·wrap(φ₂ − φ₁)` (gradient blend `:1624-1629`, IF blend
  `:1295-1298`).
- **Force DC and Nyquist real** in `reconstruct_spectrum` (sign-preserving
  magnitude, zero imaginary part).
- **Offline wide hop → `WIDE_PV_FFT / 8`** (`offline.rs:169`) — the live
  stage documents FFT/4 as the −75% blowup configuration.
- **`stretch_into` gets the ratio-1.0 exact-passthrough fast path**
  matching `stretch()`.
- **PV-direct null test** (the review's biggest test gap): drive
  `PhaseVocoder` itself at ratio 1.0, batch and streaming, on a
  tone+noise mix; report peak residual, RMS residual, SER, latency offset,
  and output-length delta. Gate SER after the blend fix (the blends are
  the current dominant error term at 1.0). Add a long-render purity-decay
  test (10-min tone; last 10 s within 3 dB of first 10 s) gating the
  accumulator wrap.

### Exit Criteria

- All four fixes landed; PV null and purity-decay tests green and gated.
- Wide falsification objective sidecar re-run: no click/LUFS regression;
  offline ratio-4.0 render now matches the live-path click/level profile.
- Owner spot-listen at ±50% (the wrapped blends will shift wide output):
  no regression vs the Stage 11 renders — and the Rubber Band A/B re-run
  at the Stage 11 rates, since the accepted wide-rate gap was measured
  against the pre-fix PV. Record here whether the gap narrowed; the
  Stage 11 acceptance is a floor, not a ceiling.

## [ ] Stage 14: Wide-Path Consolidation and Stereo Coherence

Automation: auto

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

## [ ] Stage 15: DJ-Band Ride Polish — Seam Recovery and Fade Smoothing

Automation: auto

> **Status (2026-08-06): fade smoothing and the modulation-hold
> contradiction resolved** (branch `fix/fade-ramp-hold-doc-artifact-version`):
> the extreme-rate correction weight now chases its target per sample at
> the toggle-fade slew bound, gated by a fade-band rate-step click test
> (per-block steps measured ~3.5× the tone-slew bound; the chase ~0.9×).
> `modulation_hold` resolved as a doc fix: `stage.rs` now records that the
> wide stage is its sole consumer and WHY SOLA deliberately does not read
> it (suppressing opportunistic splices during rides would push drift into
> forced onset-unprotected splices) — wiring it remains the evidence-gated
> experiment below. Remaining: seam recovery under continuous rides, the
> optional correlation-reference/strength items, and promoting the two QA
> harnesses into CI.

### Why

Three review findings degrade exactly the gesture DJs perform most —
riding the fader (details in LEARNINGS.md): (1) the crossover seam combs
while SOLA's elastic drift is parked, and rest recentering requires
~150 ms of *sustained* near-unity, which a continuous ride never provides;
(2) beyond ±20.5% the correction fade steps its gain per 32-frame block
between two differently-pitched signals — the one unsmoothed control in
the chain; (3) `ctx.modulation_hold` is documented as suppressing
discretionary splices but is never delivered to SOLA.

### Primary Files

- `src/engine/stages/sola.rs`, `src/engine/stages/keylock.rs`,
  `src/engine/stage.rs` (doc or plumbing), `qa/engine_keylock.rs` (new
  ride-seam and fade-band gates; promote this harness plus
  `qa/engine_transients.rs` into the CI quality-gates job)

### Work

- **Seam recovery under motion**: extend the drift-bleed trim (currently
  rest-gated) to act — sub-JND — during sustained rides, or lower the
  effective drift ceiling during rides so parked offsets can't persist for
  the length of a mix-in. Measure with a seam-tone comb-depth metric over
  a 30 s continuous ±8% ride (the existing seam fixtures compose).
- **Per-sample correction-fade ramp** (`keylock.rs:124-127`), mirroring
  the `enable_w` per-sample chase; add a fade-band ride click gate
  (deviation sweep 0.18→0.30).
- **`modulation_hold` truth**: either forward it into SOLA's discretionary
  splice gating (rest recenter, quiet-gap opportunism) or fix the
  `stage.rs:130-134` doc to match reality. Decide by measuring splice
  audibility during rides with/without the hold.
- Optional, evidence-gated: fractionally align SOLA's correlation
  reference to the read cursor phase (`sola.rs:587-588`) and use
  `OnsetEvent.strength` to scale protection windows — each lands only if a
  gate moves.

### Exit Criteria

- Seam comb depth during a continuous ±8% ride bounded and gated (target
  from measurement; the post-nudge recovery gate stays green).
- Zero clicks through the fade band under fast rides.
- `qa/engine_keylock.rs` and `qa/engine_transients.rs` run in CI.
- Owner listen: bass body stable through a sustained mix-in ride.

## [ ] Stage 16: Tonal-HF Granulation — Measure, Then Decide

Automation: manual

### Why

The review's headline measurement: a 3.3 kHz tone through the DJ keylock
path drops from 71.6 dB to 24–33 dB spectral purity at ±8% — SOLA splicing
~20×/s granulates sustained tonal content above the crossover. This is the
mechanism working as designed, and Rubber Band's PV does not have this
artifact class. But a sine overstates audibility on dense mixes, the DJ
path passed the Rubber Band reference gate (spectral 0.965/0.970), and
the structural-fix risk assessment is stale: the Stage 7 listening that
killed the small-FFT PV — and settled "SOLA owns the corrected range" —
was run against a PV carrying the Stage 13 defects, whose unwrapped-phase
blends manufacture exactly the phasiness that condemned it (see the
Architecture evidence caveat). So: falsification-style listening first,
on fixed implementations, build-out only on evidence. Depends on
Stage 13.

### Work

- Extend the sine measurement to music-like probes (harmonic stacks, two
  nearby tones) and pin the purity numbers as characterization tests (not
  gates).
- Curate a pad/string/vocal-heavy excerpt set (sustained tonal HF over a
  beat); render ours vs Rubber Band R3 at ±4% and ±8% (the comparison
  harness exists: `qa/rubberband_comparison.rs`); blind owner listening
  with the same protocol as Stage 11.
- **Re-audition the corrected-range decision on the post-Stage-13 PV.**
  Prototype a small/medium-FFT PV corrector (512 and 1024, identity
  locking, artifact resets) behind the 120 Hz split at DJ transpositions —
  the Stage 7 experiment, re-run with the phase-hygiene fixes in. Render
  the same excerpt set through SOLA vs the fixed PV vs Rubber Band; blind
  A/B. This is evidence gathering, not build-out: no engine plumbing, the
  batch graph + `qa/ab` fixtures suffice. Record whether the July
  phasiness verdict survives the bug fixes.
- **Verdict recorded here.** If the granulation is inaudible-in-context:
  document the floor as a characterized scope line (like the −75% wide
  edge), record the re-audition result alongside it, and close. If
  audible: scope the smallest structural response as a new stage —
  candidate order now depends on the re-audition: if the fixed PV
  auditions clean, an HF small-PV band (120 Hz–~1.5 kHz SOLA / HF PV)
  becomes the front-runner; if it still sounds phasey, the SOLA decision
  is re-confirmed on clean evidence and tonality-adaptive splice cadence
  (stretch the drift triggers on tonal material via the artifact's band
  flux) leads, with acceptance-at-DJ-ratios (wide profile as the escape
  hatch) as the fallback.

### Exit Criteria

- Purity characterization tests landed; blind listening verdict recorded
  with renders archived; either a documented scope line or a scoped
  follow-up stage with a falsification plan.
- The corrected-range re-audition result recorded — the Stage 7 "SOLA
  owns the corrected range" decision either re-confirmed on clean
  evidence or reopened with the fixed-PV renders as the case.

## [ ] Stage 17: Pitch-Shift and Batch Resampler Correctness

Automation: auto

### Why

`pitch_shift()` downsamples through `resample_sinc`, whose cutoff never
scales with ratio — pitch-up of bright material aliases, and the artifact
would be blamed on the stretcher. `AudioBuffer::resample` (the public
44.1↔48 kHz conversion) is unfiltered cubic. The streaming resampler
already does this correctly (`cutoff_for_step`); the batch paths never got
the same care. Off the DJ hot path, but it is public API quality.

### Primary Files

- `src/core/resample.rs`, `src/lib.rs` (pitch-shift path),
  `src/core/types.rs` (`AudioBuffer::resample`)

### Work

- Cutoff-scale the batch sinc kernel for downsampling (mirror
  `cutoff_for_step`), or route `pitch_shift`/`AudioBuffer::resample`
  through `MultiSincResampler`; keep the per-tap Bessel out of the inner
  loop (precompute a table as the streaming path does).
- Alias-rejection regression tests: bright content pitch-up, Goertzel at
  the fold frequency, ≥ 60 dB rejection (pattern exists at
  `resample.rs:957-988`); `AudioBuffer::resample` 44.1↔48 kHz round-trip
  SNR gate.

### Exit Criteria

- No batch resampling path lacks anti-aliasing; gates green; pitch-shift
  A/B on a bright mix audibly clean of the pitch-up shimmer.

## Not a Priority Yet

- SIMD / architecture-specific acceleration (WCET gates exist to measure
  any attempt against; current headroom is comfortable).
- Cross-frame peak tracking / multi-resolution wide path — the RB-class
  coherence work. Revisit only if Stage 13+14 listening still shows the
  wide gap mattering in practice; the Stage 11 acceptance stands.
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

Stage 12 is a prerequisite for this path but is scheduled regardless.

## Definition of Success

The engine-rebuild definition (≤ 15 ms primary chain, one engine both
modes, zero corrective heuristics, machine-verified RT contract, external
reference evidence in CI, hardware-feel deck) **holds as of the Stage 9
cutover (2026-07-15)** and must keep holding. This roadmap is done when,
in addition:

- Trustworthy beat grids on everything a DJ loads, gated on an annotated
  corpus that includes non-EDM and variable-tempo material (Stage 10).
- The wide-ratio path is free of known correctness defects, its offline
  and live renders are the same algorithm, and stereo coherence is gated
  (Stages 13–14).
- Riding the fader degrades nothing that holding it steady doesn't
  (Stage 15).
- The tonal-HF granulation floor has a recorded listening verdict — fixed,
  or documented as scope (Stage 16).
- No public path resamples without anti-aliasing (Stage 17).
- No panic is reachable from the public API on arbitrary input (Stage 12).

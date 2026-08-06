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

Stage 13 (wide-path phase hygiene) completed 2026-08-06 — four confirmed
correctness bugs fixed, null SER 59→139 dB, owner ±50% verdict "significantly
better," Rubber Band gap narrowed (R3 still ahead); archived in LEARNINGS.md.
Stage 15 (DJ-band ride polish) completed 2026-08-07 (PRs #37/#38) — fade-band
clicks gated, seam comb during sustained mild rides −7.1→−4.4 dB with the
mild-motion bounded recenter, ride-quality harnesses in CI, owner mix-in
listen passed; archived in LEARNINGS.md. Its optional items (correlation
reference, strength gating, modulation_hold wiring) remain evidence-gated
ideas, not scheduled work.

Open, in priority order for the DJ app:

- **Stage 10** — beat-grid trust on everything a DJ loads (evidence half +
  the offbeat-bass non-adopters).
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

Both verdicts stand as shipped behavior. The second was re-baselined at
Stage 13's exit listen (2026-08-06, commit `fb7dcfa`+sidecar rerun): the
fixed PV sounds "significantly better" at ±50% and the gap to R3
narrowed, though R3 remains ahead — the acceptance holds with fresher,
smaller evidence. The first still awaits the Stage 16 re-audition on the
fixed PV. The un-keylocked low band is **not**
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

Stages are independent tracks except: 16's verdict gates any future
DJ-band tonal-quality stage. (Stages 13 and 15 are complete.) Suggested
order: 10, then 14, 16, 17, 12.

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
  disagreement — remaining: annotate it as phase-indecisive and verify
  the desktop shows an honest low-confidence grid for it.
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
on fixed implementations, build-out only on evidence. Stage 13 (the
dependency) is complete — the fixed PV is available to audition.

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

Stage 12 is a prerequisite for this path but is scheduled regardless.

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
- The tonal-HF granulation floor has a recorded listening verdict — fixed,
  or documented as scope (Stage 16).
- No public path resamples without anti-aliasing (Stage 17).
- No panic is reachable from the public API on arbitrary input (Stage 12).

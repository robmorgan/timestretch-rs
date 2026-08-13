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
- **Stage 19** — direct-ratio wide path (the Stage 14 attribution's
  fix; falsification-gated on live dynamics).

Stage 14 (wide-path consolidation) closed 2026-08-13 — the durable
deliverables stand (dead-code/doc sweep, wide determinism harness,
sub-bass fix, M/S machinery), but the owner ±30/±50 blind listen FAILED
against the pre-consolidation batch PV (~7/8), and the three-session
attribution chain that followed cleared resets, M/S, shared code, and
chunking — the live topology itself (varispeed prepass + PV +
post-resampler, the Stage 11 design) is the roboty floor. The
"offline = live" goal is superseded by Stage 19, which unifies both on
the direct-ratio configuration that auditioned clean; archived in
LEARNINGS.md.
Stage 18 (steady-rate splice-cadence stretch) completed 2026-08-13
(PR #56) — SOLA's elastic drift triggers double at steady transposition
inside the primary DJ window (asymmetric band: slowdown force capped at
the write-head headroom; taper released by T=1.15; shipped cadence on
rides). Harmonic-15 purity at −8%: 22.1 → 62.8 dB, asymmetry gone;
blind owner A/Bs: half cadence beat the old build 6/6, and the exit
listen's "robot" vocabulary collapsed to one "very subtle" mention
across 6 conditions. Rubber Band stays ahead on these excerpts — the
residual gap is dominated by the sub-120 Hz pitch-follow scope line
(see Not a Priority Yet), not granulation. Archived in LEARNINGS.md.
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
corrected bass can win when latency permits. **New evidence on file
(2026-08-13, Stage 18 exit listen, blind)**: with the splice granulation
fixed, "bass sometimes sounds out of key" surfaced on 3 of 6 conditions
at ±8% on bass-forward material (msbwy) — the ±1.3-semitone low-band
detune against the corrected highs is audible once it is no longer
masked. The scope line STANDS (the Stage 2 falsification rejected a
vocoder bass; the latency argument is untouched), but any future
re-litigation starts from this evidence — and a time-domain low-band
corrector was never falsified.

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

Stages are independent tracks. (Stages 12, 13, 14, 15, 16, 17, and 18
are complete or closed.) Remaining: 10, then 19.

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

## [ ] Stage 19: Direct-Ratio Wide Path — Remove the Live Topology Floor

Automation: auto

### Why

The Stage 14 listen chain attributed the wide path's "roboty background
noise" to the live topology itself, with every alternative eliminated
(2026-08-13, three blind sessions + probes, prototype branch
`proto/stage14-ablation`):

- Phase resets: innocent (disabling them changed nothing blind).
- M/S coupling: innocent for the noise (owns only the width difference —
  measured: M/S is source-faithful, per-channel manufactures ~16 dB of
  side energy by decorrelation).
- Shared PV code / peak unification: innocent (current-code batch render
  is bit-identical to the pre-consolidation one, SER ≈ 293 dB).
- Streaming chunking: innocent (chunked direct-ratio holds batch-level
  harmonic purity, 60.7–70.5 dB, and auditioned clean/BEST blind in all
  3 conditions).
- **Guilty: the varispeed-tempo-prepass + PV-stretch + post-resampler
  arrangement** — the Stage 11 live design. Its floor was accepted then
  against Rubber Band; the batch/direct-ratio comparison shows it is a
  self-inflicted floor, not a PV limitation.

A direct-ratio wide PV removes the roboty floor from BOTH live playback
and offline renders, and re-unifies the two paths at the better quality
(the consolidation goal, pointed the right way this time).

### Falsification first (the risk is dynamics, not quality)

Constant-rate quality is proven. What is NOT proven is deck feel: the
current design does tempo in the varispeed head precisely for
sample-accurate retargets, and instant PV-ratio steps tear OLA seams
(Stage 11, measured — hence the log-slew). Before build-out, prototype
the direct-ratio chunked PV under DYNAMIC rate: ratio steps, ±ride
glides, and seek/warm-start, offline-driven. Kill criteria: rate
transitions cannot be made click-free within the 48.6 ms contract, or
output-rate-varies-with-ratio demand inversion cannot be bounded for
the pull graph (WCET or buffer growth). Fallback if killed: revert
offline wide to the direct-ratio batch PV (the quality is free there)
and keep the live stage as Stage 11 accepted it, divergence documented.

### Work (only if the kill-experiment survives)

- WideKeylockStage rework: PV at the (slewed) direct ratio on the
  source-side audio; the varispeed head runs unity for this profile (or
  is bypassed); demand inversion accounts for ratio-dependent PV output
  rate; latency contract re-derived and re-gated.
- Stereo: decide faithful M/S vs per-channel width DELIBERATELY (the
  owner blind preference leaned wide; if wide wins, implement it as a
  bounded side treatment, not decorrelation-by-accident).
- Offline wide renders through the same reworked stage — determinism
  gate restored with teeth.
- Gates: wide purity probes pinned at direct-ratio levels; rate-step /
  ride torture on the wide profile; WCET; A/B matrix; the Stage 14
  blind conditions re-rendered and re-auditioned as the exit listen.

## Not a Priority Yet

- Corrected low band at DJ ratios — re-litigating the sub-120 Hz
  pitch-follow scope line on the Stage 18 exit-listen evidence
  ("bass out of key" blind at ±8% on bass-forward material, no longer
  masked by granulation). If ever scheduled, falsification first: a
  time-domain (SOLA-class, longer-window) low-band corrector inside the
  15 ms budget was never tried — the Stage 2 rejection was of a vocoder
  bass. The wide profile remains the documented escape hatch.
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

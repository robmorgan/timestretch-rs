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

The quality-closure phase that followed (Stages 10 and 12–19, spawned
by the August 2026 full-code review against Rubber Band) **completed
2026-08-18**: every stage closed with a recorded owner verdict and its
evidence archived in LEARNINGS.md. What remains in this file is the
settled architecture, the binding policies, the deliberately-parked
ideas (Not a Priority Yet), and the un-scheduled 1.0 path.

The **Parity Track** (Stages 22–29, opened 2026-09-02) extends the goal:
blind parity with Elastique Pro on DJ material, under two latency
budgets instead of one — the 12.7 ms gesture budget the deck already
has, and a 2048-frame corrected-playback budget in the class that
Rubber Band R3 and Elastique ship in. The offline render path is part
of the parity bar: issue #78 measured it behind the threaded
`rubberband` CLI.

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

**The quality-closure roadmap completed
2026-08-18** with the Stage 10 owner ear session: annotation click
renders confirmed on the beat for the hip-hop rows and teen-spirit
("everything seems pretty spot on"), and the desktop honest
low-confidence display verified on real material. Stage 10 archived in
LEARNINGS.md with the rest.

**Parity track opened 2026-09-02.** Every quality-closure exit listen
ended with Rubber Band still the cleanest arm (Stage 18: 6/6; Stage 19:
a tie at +50%; Stage 21: "Rubber Band remains the overall reference"),
the gap to Elastique Pro has never been measured because no Elastique
render exists in the corpus, and issue #78 reported `stretch_buffer`
about 2× behind the `rubberband` CLI in release builds — the CLI runs a
thread per channel offline and the graph is single-threaded. Stages
22–29 (Parity Track, below) are the response; nothing in the shipped
architecture is reopened without a kill experiment.

Stage 19 (direct-ratio wide path) completed 2026-08-14 (PR #60) — the
PV owns the tempo axis for the wide profile as the graph's demand
inverter; the Stage 11 topology is deleted. Blind exit listen (8
conditions, 4 arms, via the new ab-tui): the roboty floor is GONE from
every slowdown ("really nice / good bass / more open" vs the old arm's
"roboty / underwater / artifacts"); at +50% compression the new head
ties Rubber Band (both "slightly roboty") and only the
decorrelation-flattered batch arm escapes — measured sub-bass balance
is at the ideal (0.537 vs batch's 0.58–0.69), so the residual is the
width preference, not a defect. Zero pipeline latency (the analysis
window is source-side lookahead), determinism sample-identical, full
gate suite green; archived in LEARNINGS.md.
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
- **Tempo-axis ownership is per profile.** Keylock: source → sinc
  varispeed (tempo axis, sample-accurate retargets, no control glide) →
  correction. WideKeylock: the direct-ratio PV head owns the tempo axis
  as the graph's demand inverter (Stage 19 — the varispeed-prepass +
  post-resampler topology was the roboty floor). "Varispeed-first" was
  a Keylock property, never a global principle.
- **Keylock profile** (primary deck): LinkwitzRiley8 split at 120 Hz; low
  band corrected by a period-aligned SOLA-class corrector (Stage 21 —
  pitch-follow below ~±1% keeps the crossover seam rigid, full
  correction by ±2%; supersedes the Stage 2 pitch-follow verdict, which
  had only ever rejected a vocoder bass); high band corrected by
  elastic-cursor SOLA. Full keylock through ±20%, release fade
  20.5%→35%, 560-frame (12.7 ms) contract.
- **WideKeylock profile** (opt-in range setting): full-spectrum FFT-2048 /
  hop-256 identity-locked direct-ratio PV head, artifact-driven per-band
  phase resets, source-side lookahead (0 ms reported delay — the first
  delivered frame is source frame 0). Profile switch is a seek-priced
  rebuild, never a live morph (Stage 26 re-asks this for the Quality
  profile).
- **Artifact-first analysis**: the `PreAnalysisArtifact` drives splice
  protection and phase resets; online detection is the fallback.
- **Single engine, both modes**: offline is the same graph with unlimited
  lookahead and a guaranteed artifact; streaming-vs-offline agreement is a
  determinism property.
- **RT contract**: pull API, no `Result` and no allocation in the audio
  path, WCET-gated, honest per-profile latency reporting.
- **Two latency budgets.** Gestures (nudge, pitch bend, scratch, the
  release-to-varispeed region) live on the 560-frame Keylock contract —
  no shipping DJ software beats it. Corrected steady playback may take
  a 2048-frame (46 ms at 44.1 kHz) budget: Rubber Band R3's default
  start delay is 2048 frames, Elastique caps its output blocks at 1024
  frames behind a DirectAPI whose stated purpose is spreading that
  block's cost across small callbacks, and every DJ app hides the
  stretcher's delay on the timeline exactly as our compensated position
  queries do (RESEARCH.md §9.1). The larger budget is where the Quality
  profile (Stage 26) lives; it never replaces the gesture budget.
- **Numeric policy.** Signal buffers are `f32`; phase accumulators,
  cursors, and any state that integrates over time are `f64`, and
  accumulators are wrapped. The never-wrapped `f64` accumulator downcast
  to `f32` each frame shipped for months (2026-08-05 review, defect 2) —
  stated once so it is not re-learned.

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
  evidence**: the sub-120 Hz pitch-follow line was re-litigated on
  Stage 18's blind evidence and superseded by Stage 21 (2026-08-20,
  corrected low band); the wide-rate Rubber Band gap stands,
  re-baselined smaller at Stage 13 and tied at +50% by Stage 19 — it is
  now the Parity Track's subject, not an accepted line.
- **Parity is measured against Elastique renders in the corpus.** Until
  Stage 23 lands those renders and the written criterion, no stage may
  claim parity, and "Rubber Band is the reference" remains the only
  citable external comparison.
- **No design verdict by ear against a component that has not passed
  its own null and purity probes.** Two architecture decisions were
  settled in July 2026 by listening against a PV carrying the defects
  Stage 13 fixed; one needed a full blind re-audition (Stage 16) before
  it was citable. Identity, purity, and null gates precede A/B sessions.
- **Offline throughput is gated.** Realtime factor on the corpus is a
  CI tripwire alongside WCET for the live path (Stage 22). A quality
  stage that halves offline throughput says so in its exit note.

## Principles

- Fix structure instead of stacking corrective heuristics.
- Every stage ends with the desktop app audibly playing the change —
  vertical slices, never plumbing-only stages.
- CI stays green throughout; new gates land with the stage that motivates
  them.
- Verdicts are scoped to the mechanism and quality floor they were heard
  against, and expire when either changes (the Stage 2 bass verdict held
  a scope line for years against a mechanism it never tested; the
  Stage 14 width preference dissolved once Stage 19 removed the masking).

## Stage Sequence

Stages 10 and 12–21 are complete or closed; evidence in LEARNINGS.md.
The Parity Track (Stages 22–29) opened 2026-09-02 and follows Stage 21.

### Stage 20 — Bounded Width Treatment (CLOSED 2026-08-19: killed)

Promoted from Not-a-Priority-Yet 2026-08-18 by owner request; killed by
its own kill experiment the next day. Prototype `cf34fad` (env-gated
`TIMESTRETCH_PROTO_WIDTH`, velvet-decorrelated high-passed mid injected
into the side at the M/S decode seam, mono-exact), calibrated by
measurement to the Stage 14 reference levels (+5 dB ≈ Rubber Band,
+15 dB ≈ the preferred batch arm) and blind-listened 2026-08-19
(8 conditions × 5 arms, sealed key, results in
`target/ab/stage20-width/results.json`).

**Verdict — the mechanism dies, and the motivation has faded:**

- At batch-matched +15 dB the injection reads "underwater, smearing,
  robotic" in 7 of 8 conditions. Side LEVEL is not the preference
  driver: per-channel PV decorrelation is two independently COHERENT
  renders, and a diffuse mid-derived injection cannot fake that
  character at any gain.
- At +5 dB it only trades minor flaws with the shipped faithful path —
  one clear win, no consistent gain.
- The recurring width preference itself did not reproduce post-Stage-19:
  the shipped head now ties or beats the `887d854` batch arm in most
  conditions (winning MSBWY +50% outright, previously the batch arm's
  showcase), so the masking advantage that drove the old preference is
  largely gone. Rubber Band was the session's most consistent arm.

The proto code is removed; the shipped paths stay faithful. Any future
width attempt must start from coherent-channel processing (e.g. a true
per-channel blend, at double PV cost), not side injection — and first
re-establish that a preference still exists. Archived in LEARNINGS.md.

### Stage 21 — Corrected Low Band (CLOSED 2026-08-20: achieved)

Promoted from Not-a-Priority-Yet 2026-08-19 by owner request:
re-litigating the sub-120 Hz pitch-follow scope line on the Stage 18
exit-listen evidence ("bass out of key" blind at ±8% on bass-forward
material, no longer masked by granulation). The Stage 2 rejection that
set the scope line was of a VOCODER bass; a time-domain corrector was
never tried.

**Kill question:** does a SOLA-class low-band corrector — ring reader
at the transposition rate, drift repaid in NCC-aligned PERIOD-length
jumps under long raised-cosine crossfades, quiet-moment opportunistic
splicing — put the bass in key at ±8% while keeping the kick's punch
and phase? Or does any correction of the sub band lose to the honest
pitch-follow bass, vocoder or not?

**Prototype:** env-gated `TIMESTRETCH_PROTO_BASSLOCK=1`
(`bass_sola.rs`), replacing the low branch's pitch-follow delay at the
same nominal lag — band alignment and the 12.7 ms latency contract
unchanged; the read cursor wobbles elastically by up to ± one bass
period, the high-band corrector's contract at larger scale. Mechanism
pinned by unit tests (unity = pure delay; ±8% transposition moves a
bass fundamental within 1% with splice steps bounded by the tone's own
slope). Not RT-vetted (full NCC sweep per splice) and not wired to the
extreme-rate fade — those are build-out work, bought only by survival.

**Falsifier:** blind ±8% on bass-forward material (msbwy, cold heart) —
shipped pitch-follow vs bass-locked proto vs Rubber Band (which
corrects its full spectrum, so it is the "bass in key" reference). If
the corrected bass reads worse than the detuned one (wobble, lost
punch, seam artifacts), the scope line stands re-validated against the
stronger falsifier and the finding is recorded.

**Kill-experiment verdict (2026-08-19, blind, 4 conditions × 3 arms,
sealed key, `target/ab/stage21-bass/results.json`): SURVIVED.** The
corrected bass beat pitch-follow in ALL FOUR conditions — the shipped
detuned bass read as the artifact ("dirty, distorted bass", "bassline
sucks, distorted, hum noise", "strange bass hums") while the proto read
"bass hitting well / open, clean, bass ok / good drums". Measured: the
proto's low band lands within ~20 cents of Rubber Band's in-key
reference on the stable-bass track; shipped sits ~130 cents off (the
±8% detune). One residual on msbwy −8%: "kicks smearing a bit" — the
predicted failure mode of the un-wired onset protection, and still
preferred over the detuned arm.

**Build-out (bought by survival):** onset-protected splicing (wire
`ctx.onsets`, protect kick windows like the high-band corrector); an
RT-safe period estimator (budgeted/incremental NCC — the proto's full
sweep is not WCET-viable); extreme-rate fade and live keylock-toggle
wiring (bass must follow the same fades as the high band); A/B matrix
and WCET gates; blind exit listen. The Stage 2 scope line
("un-keylocked low band won") is superseded: that verdict rejected a
VOCODER bass, and the time-domain corrector wins where the vocoder
lost.

**Built out and shipped** (`dd8ca34` + review follow-ups `41f7ae7`):
lockstep splicing on the channel mean, flux-gated onset protection with
directional due-thresholds (early splices on the floor-draining side so
protection is never bypassed by forced splices), budgeted incremental
period sweep (WCET gate unchanged), correction engagement ramp
(pitch-follow below ~±1% keeps the crossover seam rigid — the Stage 15
contract at bass scale; full correction by ±2%), rest recentering
(quiet-gap splice + micro-trim), toggle/fade blend shared with the high
band. Independent adversarial review caught three interlocking HIGH
findings pre-listen (quiet-detector envelope tracking the waveform's
own ripple; an engagement-ramp dead zone classified as rest; corridor
mean riding up to a full period above nominal) — fixed as one unit,
each pinned by a regression test.

**Exit listen (2026-08-20, blind, 8 conditions × 3 arms, ±8/±4 on
msbwy + cold heart, `target/ab/stage21-exit/results.json`): PASSED.**
The detune vocabulary is gone from the corrected arm everywhere; at
±8% on the bass-forward evidence track the old chain read "bassline
hum, bad" while the new chain read "good, clean". Rubber Band remains
the overall reference. Watch items recorded: a very subtle "fuzz /
bitcrush" on the msbwy bassline (quieter than the hum it replaced), a
possible "minor bass wobble" on cold heart +4%, and kick smear on cold
heart −8% present in BOTH our arms (pre-existing, not the bass
corrector — likely high-band or material).

## Parity Track (opened 2026-09-02)

The quality-closure phase ended with every scope line either achieved
or re-litigated, and Rubber Band still cleanest on every exit listen.
The Parity Track is the next road: measure the gap to Elastique Pro
directly, then close it with the mechanism class the closure phase
never tried — a transient/tonal decomposition on a corrected-playback
latency budget — without touching the gesture budget that makes the
deck feel like hardware.

Ordering is deliberate. Stage 22 is cheap, has an external reporter
waiting, and builds the offline executor Stage 26 needs. Stage 23 must
precede any DSP stage because the wide path already ties Rubber Band
at +50% and the Elastique ordering may differ. Stage 24 upgrades the PV
on the path that already has gates, so it is the safe testbed and the
tonal engine Stage 25 borrows. Stage 25 is the parity bet and may die;
its fallback is named. Everything after it is bought by survival.

Two constraints bind every stage. The live audio path keeps the RT
contract — no threads, no synchronization, no allocation in the
callback — so parallelism lives in the offline executor around the
graph, never inside a stage. And the streaming-vs-offline determinism
gate stays sample-identical, so a first tier of any change must be
bit-identical by construction.

Each stage follows the Stage 21 template: kill question, prototype,
falsifier, named fallback, build-out bought by survival, gates landing
with the stage, sealed-key exit listen.

### Stage 22 — Offline Render Throughput (issue #78) (OPEN)

**Why.** Issue #78: release-mode `stretch_buffer` at ratio 2.0 measured
about 2× behind `rubberband --tempo 0.5` (R2). The `rubberband` CLI
runs one thread per channel in offline mode by default; our graph is
single-threaded, so a stereo file hands R2 a 2× head start. That does
not account for the whole single-thread cost: a 4-minute stereo track
at ratio 2.0 is roughly 41 k hops per channel through a 2048-point FFT
pair, which is a few seconds of transform work, not thirteen. The PV
also runs a complex FFT on real input, and analysis is 15–20% on top.

**Kill question.** Where do the seconds go? A profile before threads.

**Measure.** `qa/offline_throughput.rs` (feature `qa-harnesses`): the
corpus at ratios 0.5 / 2.0 / ±8%, analysis and render timed
separately, against the `rubberband` CLI with and without
`--no-threads`. A sampling profile of the wide path at ratio 2.0.
Exit of this step: a table that says where the 13 s go. If threads
alone cannot close the reported gap, the tiers below re-order.

**Tier 1 — bit-identical, ships with the stage.**
- Offline executor in `src/engine/offline.rs`: the wide head already
  holds one `PhaseVocoder` per channel (`HeadChannel` in
  `wide_pv_head.rs`) and loops over them per block; run those loops on
  scoped threads with one persistent worker per channel. `std::thread::scope`
  is inside the MSRV, no new dependency. The `> 4×` batch PV's
  per-channel loop gets the same treatment. The Keylock path splices in
  lockstep on the channel mean and is already cheap; it gains little
  and is left alone.
- Analysis: `detect_key` shares only the input with the
  transient → beat → rigid-grid chain; run it concurrently. The STFTs in
  `detect_transients` and the chroma are frame-independent;
  frame-parallelize them.
- Gate: `streaming_offline_determinism` proves each change lands
  sample-identical.

**Tier 2 — single-thread wins, behind quality pins.** Real-input FFT
(or two real channels packed into one complex transform). These change
rounding, so they ship behind `stretch_quality_regressions` and the
purity characterization pins rather than the determinism gate.

**Tier 3 — deferred, off by default.** Segment-parallel rendering:
split the input at artifact-driven full-band phase resets with
pre-roll, render segments on separate graph instances, stitch. Bit-exact
only if the wide head's state is history-free after a reset — a
hypothesis, not a promise. Not needed to close the issue; parked in
Not a Priority Yet.

**Gates.** Realtime-factor tripwire in CI (ratio, not wall-clock, since
CI hardware varies); determinism green; WCET unchanged (the live path
is untouched by construction).

**Exit.** The reporter's case — stereo, ratio 2.0 — at or better than
default-threaded `rubberband` on the same machine; the profile table
archived in LEARNINGS.md; README throughput table with the
release-build note (the issue's first cause was a debug build); reply
on #78 linking the numbers.

### Stage 23 — Elastique Reference Corpus and Parity Criterion (OPEN)

**Why.** Everything to date is measured against Rubber Band. The
Elastique gap is inferred from architecture, never observed.

**Deliverables.**
- Elastique renders of the CC corpus at the standard ratios (DJ window
  ±4/±8%, wide ±30/±50%) from an Elastique host — Ableton Live's
  Complex Pro warp mode is élastique and exports at a fixed tempo
  ratio — stored beside the Rubber Band references, level-matched by
  the Stage 16 RMS protocol, and wired into the reference-quality
  harness.
- One baseline sealed-key session, three arms (ours / Rubber Band /
  Elastique) across the full matrix, via ab-tui. Output: the ranked
  artifact classes in the owner's vocabulary, archived in LEARNINGS.md.
- The written parity criterion. Draft, owner finalizes at stage open:
  a sealed-key session of at least 12 conditions spanning both budgets,
  two listeners, in which ours is ranked below Elastique in no more
  than a quarter of conditions and never with the "robotic / underwater
  / vocoder" vocabulary. Ties count as parity.

**Exit.** Renders committed, harness green, session archived, criterion
written into Definition of Success. No DSP changes in this stage.

### Stage 24 — PV Coherence on the Wide Path (OPEN)

**Why.** The wide head's phase locking is recomputed per frame with no
cross-frame peak continuity (overlapping regions last-write-wins) and
no multi-resolution analysis — the classic unstable-cymbals source, and
why +50% compression ties Rubber Band at "slightly roboty" instead of
beating it. This PV is also the tonal engine Stage 25 borrows, so it is
upgraded first, on the path that already has determinism, WCET, and
±50% gates.

**Kill question.** Does cross-frame peak tracking with band-dependent
lock strength on a Bark-style partition, plus a two-resolution analysis
(long window below ~1.5 kHz, short above), remove the +50% "slightly
roboty" blind — against Rubber Band and the Stage 23 Elastique arm?

**Prototype.** Env-gated `TIMESTRETCH_PROTO_PEAKTRACK`, offline-only,
inside the existing head so the artifact-driven resets and M/S path are
unchanged. Peak continuity by nearest-peak assignment with a per-band
hysteresis; lock strength per partition; the two-resolution split as
two PV arms recombined at a fixed crossover with matched group delay.

**Falsifier.** Blind ±30/±50 on the Stage 14 excerpts, four arms:
shipped head / proto / Rubber Band / Elastique. Kill if the proto does
not beat the shipped head, or if any condition regains the "robotic /
underwater" vocabulary.

**Fallback.** Identity locking stays; the wide gap is re-baselined
against Elastique and recorded.

**Gates landing with the stage.** A null test through the PV itself at
ratio 1.0 (the 2026-08-05 review's finding 8: the identity suite tests
the bypass, not the DSP); the purity characterization re-pinned;
determinism; WCET; `wide_stereo_coherence`.

**Exit.** Sealed-key listen archived; the head's Architecture bullet
updated; sub-bass balance and two-tone pins re-derived (Stage 19's
lesson: when a fix explains an old pinned number, re-derive the pin).

### Stage 25 — Hybrid Decomposition Kill Experiment (OPEN)

**Why.** The residual DJ-window gap is sustained tonal material paying
for splices (Stage 16: granulation audible in context; Stage 18: cadence
halved, Rubber Band still cleanest 6/6). The mechanism class Elastique
uses — detect transient events, reinsert them in the time domain,
stretch the residual spectrally (RESEARCH.md §§1, 5, 7) — has never
been built here. Stage 16 killed a *small* PV behind the 120 Hz split;
it did not test a full-resolution tonal path with transients removed.

**Kill question.** At a 2048-frame budget, does a hybrid — artifact
timeline drives event segmentation; transient regions cut and
reinserted at their mapped timeline positions with Röbel-style
window-center alignment; residual through the Stage 24 PV; raised-cosine
recombination keeping the sample-exact timeline — beat the shipped
Keylock chain on the sustained-tonal excerpts where Rubber Band wins,
without losing the drums?

**Prototype.** Env-gated `TIMESTRETCH_PROTO_HYBRID`, offline-only, on
the Keylock ratios. The `hpss` module is the candidate for the residual
split if the event timeline alone leaves too much attack in the tonal
path. Tonal path at FFT ≥ 2048 — a smaller PV re-runs Stage 16 and is
out of scope by construction.

**Falsifier.** Blind on the Stage 16/18 excerpts (msbwy, cold_heart,
hot_stuff) at ±4/±8%, four arms: shipped Keylock / hybrid / Rubber Band
/ Elastique. Kill if the hybrid does not beat shipped Keylock on the
sustained-tonal conditions, or reads "robotic / underwater / vocoder"
on any, or smears the kicks the transient gates protect.

**Fallback.** Tonality-adaptive SOLA — a third band or a
tonality-gated splice cadence — the Stage 16 alternative that was
never tried. Falsified the same way.

**Exit.** Verdict archived; on survival Stage 26 opens.

### Stage 26 — Quality Profile Build-Out (bought by Stage 25 survival)

A third `EngineProfile` on the 2048-frame budget hosting the hybrid.
Keylock at 560 frames is untouched; Halo picks per deck or per mode,
the way zplane ships Efficient beside Pro.

**Build-out.**
- RT-safe implementation under the zero-alloc contract; WCET gate;
  artifact-first with the online detector as fallback.
- Stereo as a shared-analysis instance from the start — linked
  channels on the M/S machinery, one event timeline for both — not two
  mono paths (the zplane SDK documents the same recommendation).
- Ride and seam behavior against the existing ride harnesses; honest
  latency reporting; compensated position queries.
- Two decisions re-asked as kill experiments inside the stage: whether
  the profile switch stays a seek-priced rebuild or becomes a live
  morph sharing the varispeed head with Keylock (a crossfade between a
  12.7 ms and a 46 ms chain is a timeline-offset problem, not a DSP
  one); and whether the transient and tonal branches run on the Stage
  22 executor's workers offline.
- Desktop toggle so the change is audibly playable.

**Exit.** Sealed-key session against Elastique across the full matrix,
and a blind pitch-fader nudge test against a current Traktor or
rekordbox deck at the same interface buffer — the one place a 46 ms
profile is felt.

### Stage 27 — Pitch-Shift and Formant Parity

Elastique Pro's formant preservation is part of what its quality means.
The current formant path is the pre-rebuild PV behind `pitch_shift`.
Move it onto the Stage 24 PV with transient-aware handling on the shift
axis; gate vocal excerpts blind against Elastique's Pro and Monophonic
modes. Kill question: does the shifted vocal read closer to Elastique
than the current path at ±3 and ±7 semitones?

### Stage 28 — Material Generality and Second Listener

The binding policy is EDM at DJ ratios; Elastique's reputation is
vocals, acoustic, and speech. Expand the corpus with those classes,
bring the second listener onto the structured checklist (the 1.0 path's
bus-factor item), and either gate the new classes or write the scope
boundary down. Never silently variable.

### Stage 29 — Parity Sign-Off

Re-run the Stage 23 criterion on the shipped Quality profile with both
listeners. Either declare parity with the evidence archived, or record
the residual gap the way Stage 11's Rubber Band gap was recorded —
scoped to mechanism and floor, so the next re-litigation starts from
evidence.

## Not a Priority Yet

- SIMD / architecture-specific acceleration (WCET gates exist to measure
  any attempt against; current headroom is comfortable).
- Desktop UI/UX polish beyond its role as the reference integration.
- Additional presets, wider API surface, convenience wrappers.
- Segment-parallel offline rendering (Stage 22 tier 3) beyond what the
  channel-parallel executor delivers — only if a many-core customer
  appears and the history-free-after-reset hypothesis survives.

Promoted out of this list on 2026-09-02: cross-frame peak tracking /
multi-resolution wide path (now Stage 24) and general-purpose non-EDM
stretch quality (now Stage 28). Both were parked on the Rubber Band
acceptance; the Parity Track re-baselines against Elastique.

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

- Crate boundaries: a workspace split into stretch core (rustfft only),
  analysis (beat/key/loudness/waveform peaks, the `.tsa` container and
  its version policy), and I/O — so the analysis version policy and the
  engine stop sharing a release cadence by accident (the v0.10.0
  incident in CLAUDE.md).
- Parameter split: `StretchParams` keeps ratio, sample rate, channels,
  and the artifact; the FFT/window/envelope fields that only configure
  `pitch_shift` move to their own struct (issue #78's reporter went
  looking for stretch-quality knobs and found the pitch-shift ones).
- Stages 28 and 29 (material generality, parity sign-off) are
  prerequisites.

Stage 12, a prerequisite for this path, completed 2026-08-13.

## Definition of Success

The engine-rebuild definition (≤ 15 ms primary chain, one engine both
modes, zero corrective heuristics, machine-verified RT contract, external
reference evidence in CI, hardware-feel deck) **holds as of the Stage 9
cutover (2026-07-15)** and must keep holding. This roadmap is done when,
in addition:

- Trustworthy beat grids on everything a DJ loads, gated on an annotated
  corpus that includes non-EDM and variable-tempo material (Stage 10,
  done 2026-08-18 — hip-hop/rock/live/DnB rows, CI floors, owner
  ear-verified annotations; the hip-hop beat-PHASE class is a documented
  open frontier where the QM reference also scores zero, gated by the
  corpus for whenever it is re-attacked).
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
- The Stage 23 parity criterion is met by the shipped Quality profile
  in a sealed-key session with two listeners (Stage 29), or the residual
  gap is recorded the way the Stage 11 gap was.
- Offline `stretch()` on the corpus is at or above the default-threaded
  `rubberband` CLI's realtime factor on the same machine (Stage 22), and
  the determinism gate still holds sample-identical.

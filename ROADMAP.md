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

The **Parity Track** (Stages 23–29, opened 2026-09-02, reworked the
same day) extends the goal: blind parity with Elastique Pro on DJ
material. The deck gets one steady-state sound on a 46 ms
corrected-playback budget — the class Rubber Band R3 and Elastique ship
in, and the only class in which a splice engine can be replaced by a
transient/tonal decomposition — and a gesture lane on the 12.7 ms
budget it already has, which the engine crossfades to on nudge, bend,
and scratch. Both budgets are stated in time; Halo runs the engine at
96 kHz, so the frame counts scale with the rate. Offline render
throughput (issue #78) is handled outside this roadmap.

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
and the gap to Elastique Pro has never been measured because no
Elastique render exists in the corpus. Stages 23–29 (Parity Track,
below) are the response; nothing in the shipped architecture is
reopened without a kill experiment. **Reworked 2026-09-02, same day:**
the corrected-playback chain becomes the deck's only steady-state
sound and the Keylock chain becomes its gesture lane (Stage 26); the
decomposition kill experiment (Stage 25) runs before the PV coherence
upgrade (Stage 24) because it is the untried mechanism class; and the
offline-throughput stage (Stage 22, issue #78) is removed from the
roadmap and handled by the owner outside it. **Stage 23 closed
2026-09-03** (PR #81 + session): Elastique Pro renders scripted
through REAPER, 32 references in the manifest, criterion finalised;
baseline blind session has ours below Elastique 9/12 (DJ window) and
8/12 (wide), "robotic" still present at DJ ratios — archived in
LEARNINGS.md. Stage 25 is next.

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
  20.5%→35%, 560-frame (12.7 ms) contract. Becomes the gesture lane
  under Stage 26; the chain itself is unchanged.
- **WideKeylock profile** (opt-in range setting): full-spectrum FFT-2048 /
  hop-256 identity-locked direct-ratio PV head, artifact-driven per-band
  phase resets, source-side lookahead (0 ms reported delay — the first
  delivered frame is source frame 0). Profile switch is a seek-priced
  rebuild, never a live morph. Stage 26 replaces the seek-priced switch
  with a lane crossfade for the Keylock / quality-lane pair only;
  WideKeylock's switch behavior is unchanged until it is re-asked.
- **Artifact-first analysis**: the `PreAnalysisArtifact` drives splice
  protection and phase resets; online detection is the fallback.
- **Single engine, both modes**: offline is the same graph with unlimited
  lookahead and a guaranteed artifact; streaming-vs-offline agreement is a
  determinism property.
- **RT contract**: pull API, no `Result` and no allocation in the audio
  path, WCET-gated, honest per-profile latency reporting.
- **Two latency budgets, stated in time.** Gestures (nudge, pitch
  bend, scratch, the release-to-varispeed region) live on the 12.7 ms
  Keylock contract — 560 frames at 44.1 kHz, scaled with the rate by
  `keylock_latency_frames` (≈1219 frames at 96 kHz) — no shipping DJ
  software beats it. Corrected steady playback lives on a 46 ms budget —
  2048 frames at 44.1/48 kHz, 4096 at 88.2/96 kHz: Rubber Band R3's
  default start delay is 2048 frames, Elastique caps its output blocks
  at 1024 frames behind a DirectAPI whose stated purpose is spreading
  that block's cost across small callbacks, and every DJ app hides the
  stretcher's delay on the timeline exactly as our compensated position
  queries do (RESEARCH.md §9.1). The 46 ms chain is the deck's only
  steady-state sound (the quality lane, Stage 26); the 12.7 ms Keylock
  chain is the gesture lane the engine crossfades to when a control
  write exceeds a rate-slope or seek threshold, and back once the rate
  settles. Moving between the lanes is a timeline-offset problem on the
  shared source ring and timeline map, not a DSP one. The gesture
  budget is never given up; the quality budget is never opt-in.
  Frame counts alone are not the contract: the wide PV head's fixed
  FFT-2048 is a 46 ms / 21.5 Hz-bin window at 44.1 kHz but a 21 ms /
  47 Hz-bin window at 96 kHz, and Halo — the primary consumer — runs
  at 96 kHz. Time-domain budgets are what the bass argument rests on.
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
- **Parity is measured against Elastique renders in the corpus.**
  Stage 23 landed the renders (scripted through REAPER's élastique
  3.3.3 Pro) and the written criterion on 2026-09-03; every parity
  claim cites a sealed-key session against that arm. Rubber Band stays
  in every set as the cleanest-arm ceiling, but it is not the bar.
- **No design verdict by ear against a component that has not passed
  its own null and purity probes.** Two architecture decisions were
  settled in July 2026 by listening against a PV carrying the defects
  Stage 13 fixed; one needed a full blind re-audition (Stage 16) before
  it was citable. Identity, purity, and null gates precede A/B sessions.

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
The Parity Track (Stages 23–29) opened 2026-09-02 and follows Stage 21;
its stages are listed in execution order, not numeric order.

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

## Parity Track (opened 2026-09-02, reworked 2026-09-02)

The quality-closure phase ended with every scope line either achieved
or re-litigated, and Rubber Band still cleanest on every exit listen.
The gap is structural, not tuning: the 12.7 ms budget forces a splice
engine, and no splice engine reaches Elastique Pro or Rubber Band R3
on sustained tonal material — both ship at ~2048 frames of delay
(RESEARCH.md §9.1). The Parity Track is the next road: measure the gap
to Elastique Pro directly, then close it with the mechanism class the
closure phase never tried — a transient/tonal decomposition on a
corrected-playback budget — and make that chain the deck's default
sound, with the existing Keylock chain kept as the gesture lane that
makes the deck feel like hardware.

Ordering is deliberate, and the stages below are listed in execution
order rather than numeric order. Stage 23 must precede any DSP stage
because the wide path already ties Rubber Band at +50% and the
Elastique ordering may differ. Stage 25 runs next because
decomposition is the one mechanism class never built here and its
prototype is cheap on the shipped PV — it answers the biggest question
first. Stage 26 follows because the lane architecture is what lets a
46 ms chain be the default sound without giving up the gesture budget.
Stage 24 comes after: it upgrades the tonal engine the quality lane by
then hosts, and lands first on the wide path where the determinism,
WCET, and ±50% gates already exist. Everything after Stage 25 is
bought by its survival; its fallback is named.

Two constraints bind every stage. The live audio path keeps the RT
contract — no threads, no synchronization, no allocation in the
callback. And the streaming-vs-offline determinism gate stays
sample-identical, so a first tier of any change must be bit-identical
by construction.

Each stage follows the Stage 21 template: kill question, prototype,
falsifier, named fallback, build-out bought by survival, gates landing
with the stage, sealed-key exit listen.

### Stage 23 — Elastique Reference Corpus and Parity Criterion (CLOSED 2026-09-03: achieved)

**Verdict.** Renders in the corpus (REAPER's élastique 3.3.3 Pro,
scripted; 167/167), 32 references in the manifest, harness per-engine
block live, and the baseline sealed-key session listened blind
(2026-09-03, two sets × 12 conditions, three arms). Ours ranked below
Elastique in 9/12 DJ-window and 8/12 wide conditions, with "robotic"
on our arm in 3 DJ-window conditions; level or ahead only at ±50 %.
Ranked artifact classes and the finalised criterion are archived in
LEARNINGS.md; the criterion is in Definition of Success. Stage 25
opens next.

**Why.** Everything to date is measured against Rubber Band. The
Elastique gap is inferred from architecture, never observed.

**Deliverables.**
- Elastique renders of the corpus at the standard ratios (DJ window
  ±4/±8%, wide ±30/±50%) from an Elastique host. REAPER ships the
  engine as "élastique 3.3.3 Pro" and runs ReaScripts headlessly, so
  the renders are reproducible from the shell
  (`scripts/render_elastique.py`, 2026-09-03; Ableton's Complex Pro is
  the same engine and the manual fallback). Stored beside the Rubber
  Band references, level-matched by the Stage 16 RMS protocol, and
  wired into the reference-quality harness (per-engine summary).
- One baseline sealed-key session, three arms (ours / Rubber Band /
  Elastique) across the full matrix, via ab-tui. Output: the ranked
  artifact classes in the owner's vocabulary, archived in LEARNINGS.md.
- The written parity criterion. Draft, owner finalizes at stage open:
  a sealed-key session of at least 12 conditions spanning steady
  playback on the quality lane and gesture transitions through the
  gesture lane, two listeners, in which ours is ranked below Elastique
  in no more than a quarter of conditions and never with the "robotic
  / underwater / vocoder" vocabulary. Ties count as parity.

**Exit.** Renders committed, harness green, session archived, criterion
written into Definition of Success. No DSP changes in this stage.

### Stage 25 — Hybrid Decomposition Kill Experiment (OPEN)

**Why.** The residual DJ-window gap is sustained tonal material paying
for splices (Stage 16: granulation audible in context; Stage 18: cadence
halved, Rubber Band still cleanest 6/6). The mechanism class Elastique
uses — detect transient events, reinsert them in the time domain,
stretch the residual spectrally (RESEARCH.md §§1, 5, 7) — has never
been built here. Stage 16 killed a *small* PV behind the 120 Hz split;
it did not test a full-resolution tonal path with transients removed.
The tonal engine in this prototype is the *shipped* identity-locked
wide head, deliberately: the verdict must isolate decomposition from
the PV upgrades Stage 24 brings later.

**Kill question.** At a 46 ms budget, does a hybrid — artifact
timeline drives event segmentation; transient regions cut and
reinserted at their mapped timeline positions with Röbel-style
window-center alignment; residual through the shipped wide-head PV;
raised-cosine recombination keeping the sample-exact timeline — beat
the shipped Keylock chain on the sustained-tonal excerpts where Rubber
Band wins, without losing the drums?

**Prototype.** Env-gated `TIMESTRETCH_PROTO_HYBRID`, offline-only, on
the Keylock ratios. The `hpss` module is the candidate for the residual
split if the event timeline alone leaves too much attack in the tonal
path. Tonal path at FFT ≥ 2048 — a smaller PV re-runs Stage 16 and is
out of scope by construction. FFT ≥ 2048 is the 44.1 kHz figure; the
prototype sizes its FFT and hop from the sample rate so the window
stays ≈46 ms (4096 at 96 kHz). Two configurations are rendered:
- **Two-path**: transient events + tonal residual.
- **Three-path**: transient events + tonal peaks + a noise/residual
  path — the non-peak bins (or the `hpss` percussive component minus
  the cut events) stretched with relaxed or randomized phase instead
  of locked phase, recombined with the other two. Hats, reverb tails,
  and air through a locked PV are the classic "underwater" source, and
  RESEARCH.md §5 lists a relaxed-locking residual path as a probable
  Elastique component.

**Falsifier.** Blind on the Stage 16/18 excerpts (msbwy, cold_heart,
hot_stuff) at ±4/±8%, arms: shipped Keylock / two-path hybrid /
three-path hybrid / Rubber Band / Elastique. Kill if neither hybrid
beats shipped Keylock on the sustained-tonal conditions, or if the
surviving hybrid reads "robotic / underwater / vocoder" on any, or
smears the kicks the transient gates protect. Record separately
whether three-path beats two-path on the hat / reverb-tail excerpts;
the surviving configuration is what Stage 26 builds.

**Fallback.** Tonality-adaptive SOLA — a third band or a
tonality-gated splice cadence — the Stage 16 alternative that was
never tried. Falsified the same way.

**Exit.** Verdict archived; on survival Stage 26 opens.

### Stage 26 — Quality Lane and Gesture Lane (bought by Stage 25 survival)

The surviving hybrid becomes the deck's steady-state chain on the
46 ms budget — the quality lane. The existing Keylock chain becomes the
gesture lane. Both run behind one varispeed head, one source ring, and
one timeline map. The engine crossfades to the gesture lane when a
control write exceeds a rate-slope or seek threshold (nudge, pitch
bend, scratch, hot-cue jump) and crossfades back once the rate has
settled for a fixed hold. Slow pitch-fader moves stay on the quality
lane. This replaces the earlier design of an opt-in third profile with
a seek-priced switch: the quality budget is never opt-in, and the
gesture budget is never given up.

**Kill questions inside the stage** (Stage 21 template, each with the
named fallback below):
- Does the lane crossfade read as a seam? Falsified on the Stage 15
  ride harnesses (seam comb, fade-band clicks) and a blind nudge test:
  a nudge through the crossfade against the same nudge on the shipped
  Keylock chain alone.
- Does the quality lane track a slow fader ride without the crossfade
  firing? Falsified with the Stage 18 steady-transposition and ride
  cadence harnesses on the quality lane alone.

**Fallback.** The original design: a third `EngineProfile` on the 46 ms
budget, opt-in per deck, switched at seek price. Keylock stays the
default.

**Build-out.**
- RT-safe implementation under the zero-alloc contract; WCET gate;
  artifact-first with the online detector as fallback.
- FFT, hop, and latency derived from the sample rate the way Keylock
  derives its lag — 2048/256 at 44.1 and 48 kHz, 4096/512 at 88.2 and
  96 kHz — not inherited from the wide head's fixed `WIDE_FFT`. The
  WCET gate runs at 96 kHz as well as 44.1: twice the hops per second,
  and each hop twice the FFT, is the budget Halo actually pays.
- 96 kHz in the benchmark and pin matrix from the first commit, not
  only in the robustness fuzz. Halo runs the engine at 96 kHz; a
  lane pinned at 44.1 kHz alone is pinned for the wrong consumer.
- Stereo as a shared-analysis instance from the start — linked
  channels on the M/S machinery, one event timeline for both — not two
  mono paths (the zplane SDK documents the same recommendation).
- The lane-crossfade contract: the timeline offset between lanes is
  the quality lane's latency; the crossfade is delay-matched on the
  shared timeline map so the source position is continuous through
  it; the thresholds and hold are constants with harness-derived
  values, not tunables.
- Ride and seam behavior against the existing ride harnesses; honest
  latency reporting (the quality lane's delay is what the deck
  reports in steady state; the gesture lane's during a gesture);
  compensated position queries.
- Desktop toggle so the change is audibly playable, including a
  gesture-lane-only mode for A/B.

**Exit.** Sealed-key session against Elastique across the full matrix,
and a blind pitch-fader nudge test against a current Traktor or
rekordbox deck at the same interface buffer — the one place a 46 ms
chain is felt.

### Stage 24 — PV Coherence on the Quality Lane's Tonal Engine (OPEN)

**Why.** The wide head's phase locking is recomputed per frame with no
cross-frame peak continuity (overlapping regions last-write-wins) and
no multi-resolution analysis — the classic unstable-cymbals source, and
why +50% compression ties Rubber Band at "slightly roboty" instead of
beating it. After Stage 26 this PV is also the quality lane's tonal
engine, so the upgrade lands twice: first on the wide path, which
already has determinism, WCET, and ±50% gates, then re-pointed into
the lane.

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
Then the quality lane's tonal path is re-pointed at the upgraded PV
and the Stage 26 sealed-key session is re-run on the sustained-tonal
subset.

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

Re-run the Stage 23 criterion on the shipped quality lane with both
listeners. Either declare parity with the evidence archived, or record
the residual gap the way Stage 11's Rubber Band gap was recorded —
scoped to mechanism and floor, so the next re-litigation starts from
evidence.

## Not a Priority Yet

- SIMD / architecture-specific acceleration (WCET gates exist to measure
  any attempt against; current headroom is comfortable).
- Desktop UI/UX polish beyond its role as the reference integration.
- Additional presets, wider API surface, convenience wrappers.
- Offline render throughput (issue #78): handled outside the roadmap by
  the owner. Not a quality lever; a quality stage that materially
  changes offline throughput says so in its exit note.

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
- The quality lane is the deck's default steady-state sound and meets
  the **Stage 23 parity criterion** (finalised 2026-09-03): two
  sealed-key sets of 12 conditions — the DJ window (±4/±8 %) and the
  wide range (±30/±50 %) on the Stage 16 excerpts, three arms (ours /
  Rubber Band / Elastique Pro) — with two listeners, in which ours is
  ranked below Elastique in no more than 3 of 12 conditions per set
  and never with the "robotic / underwater / vocoder" vocabulary;
  ties count as parity (Stage 29). Baseline 2026-09-03: 9/12 and 8/12
  below, "robotic" present — recorded in LEARNINGS.md. Or the
  residual gap is recorded the way the Stage 11 gap was.
- The gesture lane keeps the 12.7 ms Keylock contract, and the lane
  crossfade is inaudible on the ride harnesses and in a blind nudge
  test (Stage 26).
- The streaming-vs-offline determinism gate still holds
  sample-identical through the lane architecture.

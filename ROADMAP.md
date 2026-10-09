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
evidence archived in LEARNINGS.md. This file covers the current
architecture and constraints, binding policies, parity experiments,
deferred ideas (Not a Priority Yet), and the un-scheduled 1.0 path.

The **Parity Track** (Stages 23–29, opened 2026-09-02, audited
2026-09-11 — Stages 23a/23b added) extends the goal: blind parity
with Elastique Pro on DJ material. The deck gets one steady-state
sound on a corrected-playback budget — the class Rubber Band R3 and
Elastique ship in, and the only class in which a splice engine can be
replaced by a transient/tonal decomposition — and a gesture lane on
the 12.7 ms budget it already has, which the engine crossfades to on
nudge, bend, and scratch. The ≈46 ms analysis window is the starting
experiment, not a proven delay figure: source lookahead, output delay,
and control response are measured separately for every candidate.
Halo runs at 96 kHz, so sample-rate-aware analysis and explicit 96 kHz
quality/callback gates travel with every stage. Offline render
throughput (issue #78) is handled outside this roadmap.

## Status (2026-10-08)

Shipped and settled: pull-based stage-graph engine (Tape / Keylock /
WideKeylock profiles), SOLA keylock through ±20% at 12.7 ms, wide-range
Master Tempo with source-side lookahead and 0 ms reported output delay,
batch `stretch()` on the same graph with sample-exact duration and
streaming/offline determinism, artifact-first Keylock transient
control, `.tsa` analysis container (v0.11.0), musical key
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

**Parity track opened 2026-09-02. Stage 23 closed 2026-09-03** (PR #81
and the owner session): Elastique Pro renders scripted through REAPER,
32 references in the manifest, criterion finalised. The baseline blind
session has ours below Elastique 9/12 (DJ window) and 8/12 (wide), with altered drum
attacks, unstable bass, and robotic tonal textures — archived in
LEARNINGS.md. Rubber Band remains the overall reference in those sets.

**Code/measurement review, 2026-09-11, main `2628090`.** The wide head
does not receive transient events: the graph sends them to its empty
downstream stage chain. A six-second mono fixture (61/220/1733 Hz tones,
decaying kick chirps, and clicks) rendered with accurate onsets versus
an empty artifact was bit-identical at tempo rates 0.5/0.7/1.3/1.5;
Keylock output changed at 0.92/1.08 as a positive control. All 23 tests
in `engine_ab_matrix`, `pitch_shift`, `pv_null`,
`tonal_purity_characterization`, and `wide_stereo_coherence` passed in
release mode. The listening evidence remains the archived Stage 23
session; its reference renders were absent from that checkout. The
missing guidance dates to Stage 19 part 2 (`408c734`, 2026-08-13): the
deleted `WideKeylockStage` carried the per-band onset resets, and the
commit deferred routing them into the new head to a part 3 that never
landed — so the Architecture bullet describing them as shipped was
stale from that day. Stage 23a restores the routing; Stage 23b fixes
the best-of reference gate. Neither gates Stage 25: at ±4/±8 %
`stretch()` and the deck run Keylock, which does receive onsets, so the
DJ-window baseline is not explained by the wide head's missing events,
and the Stage 14 ablation found those resets audibly innocent.

**Stage 25 closed and Stage 24's prototype killed, 2026-10-08.** Two
owner listens on the Stage 16 excerpts. At ±4/±8 % the full-PV control
(`9dbe45a`) and the two-path hybrid (`75b98ed`) both read
robotic/underwater; the three-path hybrid was dropped unheard. At
±30/±50 % both peak-track arms (`a9d2680`) regained robotic/underwater
vocabulary. No steady-state candidate beats the shipped engine, so
Stage 26 has nothing to promote yet. The shipped arm, DSP-identical to
the Stage 23 baseline, counted ~4/12 below Elastique (DJ) and 4–5/12
(wide), still with one "robot" mention per set. The recurring complaint
on it in both windows is the bassline sounding out of key. Archived in
LEARNINGS.md.

**Work in flight (2026-10-08).** Stage 27's pitch baseline set
(`stage27/pitch-refs`, 8 conditions × 5 arms) was heard 2026-10-08:
ours below the best formant-preserving reference in 7/8, robot
vocabulary in 4/8 (Stage 27 below). Stage 28's material rows
(`stage28/corpus-generality`) are on their branch with Elastique
references re-rendered after the decoder-alignment fix. The prototype
branches (`stage25/hybrid-proto`, `stage25/fullpv-control`,
`stage24/peaktrack-proto`) stay unmerged as the record. The remaining
order is 23a → 24 → 26 → 27 → 28 → 29; lane integration follows the
DSP verdicts.

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
across 6 conditions. Rubber Band stayed ahead on these excerpts; the
newly exposed bass detune motivated Stage 21's corrected low band.
Stage 23 records the remaining attack/bass/tonal gap after that change.
Archived in LEARNINGS.md.
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

## Architecture and Integration Constraints

- **Stage-graph engine** in `src/engine/`: fixed-block stages
  (`process`, `latency_frames()`, `reset()`, `prime()`), fixed per-profile
  chains, and a profile-specific head owning demand inversion.
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
  hop-256 identity-locked direct-ratio PV head, with source-side
  lookahead (0 ms reported delay — the first delivered frame is source
  frame 0). Artifact-driven phase resets are currently disconnected;
  Stage 23a repairs the event path. Profile switch is a seek-priced
  rebuild. Stage 26 tests a lane crossfade for the Keylock / quality pair;
  WideKeylock's switch behavior is unchanged until it is re-asked.
- **Artifact-first analysis**: the `PreAnalysisArtifact` drives Keylock
  splice protection; online detection is its fallback. Stage 23a must
  establish this contract for the direct-ratio wide head as well.
- **Stereo baseline**: the wide head encodes M/S, runs an independent PV
  per component, and decodes to L/R. This protects centered material
  but does not establish shared peak tracking or preservation of
  arbitrary interchannel phase relationships; Stage 24 extends it.
- **Single engine, both modes**: offline is the same graph with unlimited
  lookahead and a guaranteed artifact; streaming-vs-offline agreement is a
  determinism property.
- **RT contract**: pull API, no `Result` and no allocation in the audio
  path, WCET-gated, honest per-profile latency reporting.
- **Latency and analysis are separate contracts.** The Keylock gesture
  target stays at 12.7 ms — 560 frames at 44.1 kHz, scaled by
  `keylock_latency_frames` (≈1219 at 96 kHz). Quality prototypes start
  with FFT/hop 2048/256 at 44.1/48 kHz and 4096/512 at 88.2/96 kHz:
  analysis spans about 43–46 ms. The shipped fixed FFT-2048 instead
  spans 46.4 ms at 44.1 kHz and 21.3 ms at 96 kHz, with bin spacing
  worsening from 21.5 to 46.9 Hz. Longer bass analysis is an experiment
  with its own lookahead/cost, not a free extension of that budget.
  Measure source lookahead, output delay, control-to-audio response,
  startup, and seek recovery separately for every candidate. Elastique's
  internal architecture is inferred in RESEARCH.md §5; its output block
  cap does not establish a fixed delay or a minimum quality budget.
  Stage 26 must prove transitions between the quality and gesture
  paths: delay matching alone neither proves an inaudible crossfade nor
  preserves the shorter control response. The design goal stands as
  decided 2026-09-02: the gesture budget is never given up, and the
  quality budget is never opt-in. The opt-in profile is Stage 26's
  named fallback, recorded as a miss if it is what ships.
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
verdict. The low-band scope line was subsequently reopened by Stage
18's blind "bass sometimes sounds out of key" reports and superseded
by the Stage 21 time-domain bass corrector. Stage 23 still heard
unstable/out-of-key bass in both DJ and wide conditions, even with
correction engaged. Those reports motivate new coherence experiments;
neither the old scope line nor single-tone pitch accuracy establishes
polyphonic bass parity.

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
- **Every reference condition is accountable.** Stage 23b gates each
  required track × tempo rate × reference engine × candidate profile at
  the declared sample rate. A best result across references, presets,
  or ratios cannot substitute for a failing condition. Diagnostic
  averages remain useful, but do not decide acceptance. Development
  excerpts and held-out listening material are identified separately.
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

## Parity Track (opened 2026-09-02, updated 2026-10-08)

The Stage 23 blind baseline establishes a gap on drum attacks, bass,
and tonal texture. The mechanism behind each artifact remains an
experimental question. The shipped Keylock path pays for time-domain
splices; the wide path has disconnected transient guidance and a
single-resolution, frame-local peak-locking policy. Decomposition,
phase tracking, and separate noise treatment are candidates supported
by the research, not verified descriptions of Elastique's internals or
guarantees of parity.

Execution order is **23 (done) → 25 (done, killed) → 23a → 24 → 26 →
27 → 28 → 29**, with Stage 23b's gate fix landing alongside whatever is
in progress. Stage 23a precedes the rest of Stage 24 because both work
on the wide head and later Stage 24 candidates should be read against a
head with its guidance connected. Stage 24's first prototype is already
killed (below); its remaining hypotheses start with low-band coherence,
the recurring "bassline out of key" complaint.
Stage 28's material selection and listener recruitment start before
candidate selection. A simpler full-PV path can win Stage 25; the
hybrid is not a prerequisite for progress.

Every live implementation keeps the RT contract: no allocation,
synchronization, or worker-thread dependency in the callback. Offline
and streaming output remain sample-identical for equal source, rate
schedule, profile, and artifact. Each experiment records its source
revision, parameters, sample rate, reference configuration, and metric
and listening results. Null/purity checks precede blind selection.

**IP check before decomposition returns (proposed 2026-10-09).** The
Stage 25 hybrid cut each transient region out, stretched the residual,
and reinserted the original transient with crossfaded borders. That is
the pattern described by Fraunhofer's US 9,236,062 (priority
2008-03-10; Google Patents lists it active to 2029-09-29, an assumption,
not a legal conclusion). The same filing describes transient timing
stored as metadata; whether that is in the granted claims is unread,
so any resemblance to the `.tsa` onset artifacts is unassessed, not
established. Nothing shipped depends on cut-and-reinsert (the hybrid is
on an unmerged, killed branch). Get a freedom-to-operate read of that
family before any cut-and-reinsert design is revisited or promoted.
Expired and free: Laroche–Dolson peak-region pitch shifting (US
6,549,884, 2019) and frame-skip transient bypass (US 8,489,404, 2021).
Time-map anchoring (Stage 23a arm below) bends the ratio inside one PV
and neither removes nor reinserts audio. Patent links are in
RESEARCH.md.

### Stage 23 — Elastique Reference Corpus and Parity Criterion (CLOSED 2026-09-03: achieved)

**Verdict.** REAPER's élastique 3.3.3 Pro renders were generated
(167/167), 32 references entered the manifest, and the harness gained
per-engine summaries. The baseline sealed-key session was heard blind
(2026-09-03, two sets × 12 conditions, three arms). Ours ranked below
Elastique in 9/12 DJ-window and 8/12 wide conditions, with "robotic"
on our arm in three DJ-window conditions. Ranked artifact classes and
the finalised criterion are archived in LEARNINGS.md.

**Delivered.** `scripts/render_elastique.py` scripts REAPER renders at
±4/±8% and ±30/±50%; `scripts/ab.sh --ref-arm` assembles the references
with the Stage 16 RMS level-matching protocol. The owner baseline used
the Stage 16 excerpts and ours / Rubber Band R3 / Elastique Pro arms.
The final criterion is two sets of 12 conditions, two listeners, ours
below Elastique in no more than three conditions per set and never
with "robotic / underwater / vocoder" vocabulary; ties count as parity.
This criterion stays unchanged in Definition of Success.

**Boundary.** This stage established reference renders and a listening
baseline. It did not establish per-condition CI acceptance: the strict
harness currently chooses a best result across references/presets for
each track. Stages 23a/23b address the September review findings without
reopening the archived baseline verdict.

### Stage 23a — Wide-Path Transient Guidance (OPEN — after Stage 25's verdict, before Stage 24)

**Scope.** This is wide-path (±30/±50 %) and architecture work. It is
not the DJ-window fix: at ±4/±8 % the shipped path is Keylock, whose
splice protection does receive onsets, and the Stage 14 ablation found
the wide head's resets audibly innocent on the old topology. The
expected yield is an honest artifact-first contract and measurably
cleaner attacks at wide ratios; a DJ-window improvement would be a
surprise to record, not a prediction.

**Evidence.** [graph.rs](src/engine/graph.rs) publishes artifact events to
`StageCtx`; [profiles.rs](src/engine/profiles.rs) gives WideKeylock an
empty stage chain. [wide_pv_head.rs](src/engine/stages/wide_pv_head.rs)
accepts audio/rate/emission limits but no events. Its phase resets
occur at startup/reset, not at track onsets. The 2026-09-11
accurate-versus-empty-artifact probe was
bit-identical at four wide rates despite passing the existing gates.

**Work.** Route events on the source timeline into the direct-ratio
head before the affected analysis windows are synthesized. Account for
window-center alignment, strength/band selection, lookahead, seeks,
loop wraps, and rate changes. Establish an online detector fallback
when no artifact is attached; an explicitly empty artifact remains
authoritative. Keep the direct-ratio topology that survived Stage 19.

**Time-map anchoring arm (proposed 2026-10-09).** Resets restore
vertical coherence but not where the attack lands: frames that see an
onset each render it at a different output offset, spreading pre-echo
over up to one analysis window × |1/r − 1| (≈4–5 ms at the shipped FFT
2048, ±10 %, 44.1 kHz; double at 4096), most of it in the window's
middle half because of the Hann taper. Because this head owns the
tempo axis, it can instead hold the local ratio at exactly 1 from about half a window before each qualifying onset until
the attack has passed, and absorb the difference in the surrounding
sustain (at 128 BPM and −10 % the in-between ratio moves from 0.90 to
≈0.89). With equal analysis and synthesis hops every frame that sees
the attack places it at the same output time; sustained partials keep
integrating, so nothing is reset under them, and onsets land on their
nominal output times, which beat sync also wants. Treat seeks, loop
wraps, and direction flips as forced anchors. Render it as a third arm
beside reset-only and reset + anchoring, on the same fixture and
metrics; it changes the head's rate schedule, so verify timeline
accounting, the audible-position query, and determinism under it.
Hi-hats can stay unanchored if their short-band pre-echo measures
below ~1 ms; where onsets are too dense to anchor, fall back to the
reset policy. Measure that fallback rather than assuming it: at ±50 %
with dense onsets, holding unity around each attack pushes the
in-between ratio toward the clamp. In practice this is wide-path work;
±4/±8 % runs on Keylock.

**Gates and falsifier.** Land a discriminating regression for accurate,
empty, and deliberately shifted onset timelines on a controlled
kick/click-plus-tonal fixture. Verify the mapped event timing and a
measurable effect around qualifying attacks; equality between accurate
and empty guidance must not silently pass. Measure pre-echo energy,
attack spread, onset-position error, peak/energy retention, and tonal
continuity through attacks, at 44.1/48/96 kHz. Blind the guidance-only
candidate against the shipped wide head and both references at wide
ratios on the Stage 16 excerpts. A connected reset is necessary evidence
of wiring, but improved attacks without new tonal artifacts decide
whether its policy ships.

**Fallback.** If broad phase resets damage sustained content, test
selective/strength-gated resets before promotion (the deleted
`WideKeylockStage` gated low-band resets on onset strength and per-band
flux; start from that policy). Retain the shipped head as the explicit
control for Stage 24; do not reinstate the old varispeed-prepass
topology or describe disconnected resets as shipped.

**Exit.** Event routing and its regressions verified, guidance policy
and listening verdict archived, determinism/RT gates green for any
promoted implementation, and engine/API/test documentation corrected
to describe the policy actually running.

### Stage 23b — Reference Gates per Condition (OPEN — gate fix now, matrix before Stage 29)

**Evidence.** `assert_absolute_quality_floors` in
[reference_quality.rs](qa/reference_quality.rs)
selects one best result across a track's references and presets. A good
ratio or another reference engine can hide a failure. Required CI's
`rubberband_reference_gate` covers a 25-second mono excerpt from one
track at two DJ rates; per-engine summaries are diagnostic averages.

**Part 1 — the gate fix (small, lands now; does not block Stage 25).**

- Give each required track × tempo rate × reference engine × candidate
  profile its own result and acceptance bounds. Fix the candidate preset
  for an acceptance run; report exploratory presets separately. Keep
  aggregates for trends, without using a best result or average to
  overrule a failing condition.
- Fail strict runs on missing required rows, unavailable references,
  invalid metrics, or checksum/configuration mismatches. Add a harness
  regression showing that one strong condition cannot conceal a weak
  one.

**Part 2 — the matrix (required by Stage 29, not by Stage 25).**

- Extend required public-corpus CI beyond one mono track: include stereo,
  exposed attacks, sustained bass/tones, and the DJ/wide ratio matrix.
  Run relevant quality and callback-budget gates at 44.1/48/96 kHz.
- Require the complete Elastique matrix on a configured reference runner
  or as an archived local sign-off run; missing licensed-host renders
  cannot count as a pass. Record engine/host mode, source and render
  hashes, window, level matching, sample rate, and code revision.
- Track attack spread/pre-echo/timing, bass pitch and envelope stability,
  tonal sidebands, HF retention, and stereo relationships alongside
  spectral similarity. Derive artifact-specific bounds from measured and
  heard fixtures; record why each bound detects the intended failure.
  Keep blind listening as the perceptual gate.

**Fallback.** Unavailable conditions remain explicitly unvalidated and
block the corresponding parity claim. They are not replaced by an
easier ratio or by another engine's score.

### Stage 25 — Hybrid Decomposition Kill Experiment (CLOSED 2026-10-08: killed)

**Verdict.** Blind set `target/ab/stage25-fullpv` (12 conditions ×
current / fullpv / hy2 / Rubber Band / Elastique, ±4/±8 %). The full-PV
control (`stage25/fullpv-control`, `9dbe45a`) read robotic, underwater
or closed in 6/12; the two-path hybrid (`75b98ed`) was faulted in 8/12,
worst on hot_stuff where Keylock is clean. Both fail the
never-robotic bar, so neither the plain PV nor decomposition beats
shipped Keylock in the DJ window. The three-path hybrid's set
(`target/ab/stage25-hybrid`) was not heard: it shares hy2's
transient/residual split, and is the first listen if decomposition is
revisited. Keylock stays the shipped default. Open question carried
forward: the same wide head is robotic here and clean at ±30/±50 %
(Stage 24 set), which missing onset guidance alone does not explain.
The fallback below (tonality-adaptive SOLA) remains untried. Notes and
per-condition reads in LEARNINGS.md.

**Why.** Stage 16 rejected a small PV behind the 120 Hz split; it did
not establish how the current direct-ratio, full-resolution wide head
compares at DJ rates. Stage 23 heard both altered attacks and unstable
tonality. Establish the full-PV baseline before attributing an
improvement to transient/tonal/noise separation.

**Kill question.** Does separating attacks and sustained material beat
both shipped Keylock and a matched full-PV control on the failing DJ
excerpts, without weakening the drums? Does separate noise treatment
improve hats, breath, and reverb beyond the two-path hybrid?

**Prototype (built 2026-09-03, `stage25/hybrid-proto` `75b98ed`).**
`src/engine/hybrid.rs` behind `TIMESTRETCH_PROTO_HYBRID=2|3` in
`offline.rs`, Keylock ratios only: artifact timeline drives event
segmentation, transient regions are cut and reinserted at their mapped
timeline positions with window-center alignment, the residual runs
through the shipped identity-locked wide head at FFT ≥ 2048 (sized from
the sample rate so the window stays ≈46 ms), raised-cosine
recombination keeps the sample-exact timeline. Two-path = events +
tonal residual; three-path adds a noise/residual path with relaxed
phase. The blind set `target/ab/stage25-hybrid` is 12 conditions × 5
arms (shipped Keylock / two-path / three-path / Rubber Band /
Elastique) on msbwy, cold_heart, hot_stuff at ±4/±8 %.

**Control the hybrid must also beat before promotion.** `stretch()`
chooses Keylock in the DJ range, so no rendered arm shows what the
full-resolution direct-ratio PV alone does at ±4/±8 %. Stage 16 killed
a *small* PV behind the 120 Hz split, not this head. If either hybrid
survives the first listen, render a second set adding the shipped
WideKeylock head forced at DJ rates (matched FFT/hop, guidance, stereo
policy, and gain) so decomposition is the only variable between it and
the hybrid — a hybrid that only ties the plain PV is a PV win, not a
decomposition win, and the simpler candidate carries forward.

Record actual lookahead and output delay for every arm; include 96 kHz
in the confirmation set. `analysis::hpss` is an offline separation
candidate, not an RT-ready stage: it stores the whole spectrogram and
its centered time median needs future frames. Count separation
lookahead separately.

Evaluate relaxed locking or noise resynthesis on the residual. Any
random phase decisions must be deterministic from source position and
seed, respect stereo coherence, and preserve streaming/offline
agreement. A globally locked noise bed and independently randomized
channels are both hypotheses to test, not defaults to assume safe.

**Falsifier.** The rendered set, blind, with separate notes for
attacks, bass, sustained tones, and hats/tails. Kill if neither hybrid
beats shipped Keylock on the sustained-tonal conditions, or if the
surviving hybrid reads robotic/underwater/vocoder on any, or smears the
kicks the transient gates protect. Record whether three paths improve
on two; additional complexity must earn an audible benefit. Then the
full-PV control set above. Include the Stage 23b metrics and Stage
28's development excerpts before selecting the quality candidate; keep
its held-out set for final evaluation.

**Fallback.** If the full PV wins without decomposition, carry that
simpler candidate forward. If no spectral candidate improves on
Keylock, test tonality-adaptive SOLA (band/cadence policy) under the same
protocol. Keep Keylock as the shipped default until a candidate wins.

**Exit.** Archive the controlled comparison, chosen mechanism or failed
bet, and its latency/lookahead/cost. Stage 24 develops the surviving PV
candidate, or independently evaluates the shipped wide head when no
PV candidate survives; Stage 26 requires a winning steady-state path.

### Stage 24 — Tonal, Bass, and Stereo Coherence (OPEN — first prototype killed 2026-10-08; before lane integration)

**Verdict on the first prototype.** Blind set
`target/ab/stage24-peaktrack` (12 conditions, ±30/±50 %). trackband and
all1024 (`a9d2680`) were each robotic/underwater/wobbly in 5/12 and beat
the shipped head only once; both killed. Identity locking stays. The
shipped head was mostly "open, wide, clean" and ~4–5/12 below
Elastique, its faults mostly the bassline sounding out of key (msbwy
+30/−50, cold_heart −50). Low-band coherence is therefore the next
Stage 24 candidate; content-adaptive resolution and stereo linkage
follow. Per-condition reads in LEARNINGS.md.

**Prototype state (built 2026-09-07, `stage24/peaktrack-proto`).**
`TIMESTRETCH_PROTO_PEAKTRACK` (`1`/`all`, or a list of `track`, `band`,
`multires`) is read only in `offline.rs` and threaded through
`Engine::build_with_wide_proto` → `WidePvHead::with_proto`; live engines
never see it. Peak continuation and Bark-partition lock strength live
in `src/stretch/peak_track.rs`; the two-resolution arm is an LR4 split
at 1.5 kHz with a short PV above it. Both arms must seed phase from
frames with the same source centre or the crossover band cancels by
d·(1−r) samples (caught by a 1.5 kHz level unit test). **Measured
before the listen** (`examples/stage24_probes.rs`,
`target/ab/stage24-peaktrack/probes.txt`): the fixed split is
purity-destructive on dense HF harmonics — the 512 arm drops
harmonic-15 purity 67.7 → 17.3 dB at −50 % (1024 arm 37.5 dB) because a
short Hann window cannot resolve 220 Hz-spaced partials. A fixed split
is the wrong mechanism; content-adaptive resolution (R3-style) is the
hypothesis that remains. `track` alone is neutral to slightly
positive; `band` alone costs ~12 dB on pure tones but stays above
80 dB. The blind set (`target/ab/stage24-peaktrack`, 12 conditions,
±30/±50 %, arms current / trackband / all1024 / Rubber Band /
Elastique) dropped the 512 arm on that measurement.

**Evidence.** Peaks are detected afresh each frame in
`src/stretch/phase_vocoder.rs`; there are no persistent peak identities.
Trough-bounded locking in `src/stretch/phase_locking.rs` writes shared
boundary bins in peak order. The wide head configures a 100 Hz cutoff,
and the lowest bins are excluded from that peak-locking pass. M/S
components have independent PV analysis/state. These are candidates
behind the bass/tonal/image reports, not proven causes of every report.

**Kill question.** Do persistent peak tracking, deliberate low-band
coherence, multiple analysis resolutions, and linked stereo guidance
improve the selected PV and shipped wide head without new attack,
spatial, or modulation artifacts?

**Prototype.** `TIMESTRETCH_PROTO_PEAKTRACK`, offline-first, with
separately switchable changes. The rendered set is judged as-is; once
Stage 23a lands, the surviving changes are re-rendered on the head with
its guidance connected before promotion:

- Track peak identities across frames with bounded assignment and
  hysteresis; make bin ownership stable and unambiguous. Evaluate
  frequency-dependent lock strength and noise confidence.
- Test low-bin coherence explicitly on exposed bass, polyphonic bass,
  glides, and kick/bass overlap. Revisit the 100 Hz locking exclusion
  with evidence; accurate single-tone pitch alone cannot pass this gate.
- Size analysis windows by sample rate and content. A fixed
  two-resolution split is already measured out (above); the open
  question is content-adaptive resolution — longer windows where the
  spectrum is dense and stationary, shorter around attacks — with
  matched timing and measured recombination response. Begin at the
  Stage 25 window sizes; additional bass resolution carries an explicit
  latency/cost.
- Share peak/event guidance across stereo where appropriate, preserving
  the source's interchannel phase and level relationships. M/S remains
  a useful representation, not proof of complete channel linkage.

**Additional candidate arms (proposed 2026-10-09).** Each is a separate
switch under the existing prototype flag, tested alone before combining,
and judged under this stage's gates and falsifier. None replaces
identity locking unless it wins blind. Owner listens are the
bottleneck, so run them in tiers, not as five separate sets: (1) the
dispersion probe, measurement only, no listen; (2) PGHI fed by
reassignment gradients, rendered with and without the
sample-rate-scaled bass window; (3) shared-rotation stereo on the
winner of (2). Anchoring belongs to Stage 23a.

- **Near-unity dispersion probe (first, cheap).** Stage 25 left open
  why the same head reads robotic at ±4/±8 % and clean at ±30/±50 %.
  One hypothesis: away from unity, propagated phase drifts from
  analysis phase and inter-partial alignment randomizes even when the
  stretch is tiny, while the wide-ratio references are degraded enough
  that ours reads relatively clean. Render the wide head at
  1.00/1.01/1.02/1.04/1.08 and measure vertical-coherence loss
  (e.g. group-delay spread around partials, attack shape on the
  kick/click fixture) as a function of |r − 1|. If it appears as soon as
  the ratio leaves 1, test gradual coherence restoration toward analysis
  phase near unity (Rubber Band R3 restores vertical coherence gradually
  on return to 1.0; its changelog records the same change backported to
  R2) and keeping noise-classified bins at analysis phase in the DJ
  window. If coherence loss does not track |r − 1|, the hypothesis is
  falsified and the DJ-window gap lies elsewhere.
- **Phase-gradient heap integration (PGHI).** Replace frame-local
  peak locking plus the 0.20 nearest-peak gradient blend with
  integration of both phase derivatives outward from the strongest bins
  (Průša & Holighaus, "Phase Vocoder Done Right", 2022; RTPGHI for the
  causal variant). It needs no peak picking or tracking and claims no
  classic PV artifacts at extreme ratios. Target the low-band
  coherence complaint first; include the sub-100 Hz bins currently
  excluded from locking. Bin width still limits bass resolution at
  FFT 2048, so judge PGHI with and without the longer bass window. A
  quick search found academic sources only, no patents.
- **Reassignment-based gradients.** Per frame, take FFTs with the
  window w, its derivative w′, and t·w to get instantaneous frequency
  and local group delay per bin with no previous-frame dependency and no
  phase unwrapping. Feeds PGHI directly, gives per-bin attack timing for
  Stage 23a, and makes seeks/loop wraps/rate changes stateless for the
  analysis side. Costs two extra forward FFTs per frame per resolution
  (only for one channel if paired with the stereo arm below).
- **Shared-rotation stereo linkage.** Run the phase solution once on
  mid and express it as a per-bin unit rotation R[k] = output phase −
  analysis phase; apply Y_c[k] = X_c[k]·R[k] to every channel.
  Interchannel phase and level relationships survive exactly, mono
  fold-down is preserved by construction, and the expensive per-bin
  work runs once. Blend toward independent processing where L/R are
  uncorrelated (wide reverbs, hard-panned hats). Concrete candidate for
  the linked-stereo bullet above; the élastique V3 SDK docs describe
  linked analysis (RESEARCH.md §5 item 7). Extends to stems linearly.
- **Sample-rate-scaled bass window.** At 96 kHz the fixed FFT 2048 gives
  47 Hz bins and a 21 ms window in the bass — Halo's operating point.
  Scale the window by sample rate at minimum; a longer window confined
  to the lowest band only (not full-band 4096, which LEARNINGS records
  as smearing real mixes) is the content-adaptive-resolution starting
  point, with its lookahead and cost recorded.

**Gates and falsifier.** Compare the selected Stage 25 PV before/after
on DJ excerpts, and the wide head before/after at ±30/±50%, with both
references. Test each change independently before combining survivors.
Extend `wide_stereo_coherence` with hard-panned transients, overlapping
instruments, stereo reverbs, phase-offset tones, and mono fold-down.
Measure image movement, channel phase/level relationships, bass
stability, attack timing, and modulation sidebands at 44.1/48/96 kHz.
Kill changes that fail to improve the targeted blind artifact class or
introduce new attack/spatial regressions.

**Existing gates to retain.** `tests/pv_null.rs` already exercises the
PV directly at unity and checks long-render purity (Stage 13); do not
schedule a duplicate null test as missing work. Extend these checks to
new resolutions and retain determinism, tonal-purity, stereo, and
callback-budget gates. Re-derive explained sub-bass/two-tone baselines
without weakening unrelated bounds.

**Fallback and exit.** Keep the simpler surviving locking/resolution/
stereo policy if an upgrade loses. Archive per-change ablations and the
combined blind verdict, then freeze the selected steady-state path for
Stage 26. Wide-path work remains useful even if the DJ candidate fails.

### Stage 26 — Quality Lane and Gesture Lane (after a winning DSP candidate)

Promote the candidate selected in Stages 25/24 to the deck's steady-state
quality path only after its sound is established. Retain Keylock as the
12.7 ms gesture path. Share source identity and timeline accounting;
do not force the direct-ratio PV behind a varispeed prepass merely to
share a head. Stage 19's rejected prepass/PV/post-resampler topology
must not return without its own controlled evidence.

**Kill questions.**

- Can the two paths switch without an audible seam, combing, duplicated
  attacks, dropped content, or a discontinuous audible source position?
- Can a nudge, bend, scratch, or hot cue retain the existing gesture
  response while entering/leaving the quality path? Aligning two
  delayed signals does not by itself prove the shorter response.
- Can slow fader rides remain on the quality path without pitch/envelope
  instability or repeated switching?

**Prototype/build-out.** First prove a switch between the selected
paths under recorded rate/seek schedules. Derive alignment from both
paths' measured delays and source positions, not from an assumed
46 ms offset. Measure control-to-audio response, startup, seek recovery,
and transition duration separately from steady-state pipeline delay.
Use the ride/seam harnesses and a blind nudge comparison with Keylock
alone to choose thresholds, hold times, and fades.

Then implement the surviving transition under the zero-allocation RT
contract with bounded working storage and online guidance fallback.
Gate 44.1/48/96 kHz quality and callback budgets, including the periods
when both paths run, the smallest supported callbacks, warm starts,
loop wraps, and rapid retargets. Sample-rate-scaled FFT/hop choices
change transform cost and hop cadence; measure the resulting cost.
Require streaming/offline determinism for the combined path, honest
latency/position reporting through switches, and a desktop A/B mode.

**Fallback.** Offer the proven quality path as an opt-in `EngineProfile`
with a seek-priced switch. Keylock stays the default if automatic
switching cannot meet both sound and gesture contracts. Record that
Stage 26's automatic/default integration goal remains unmet.

**Exit.** Sealed-key steady-state and transition sessions against
Elastique, plus a blind fader-nudge comparison with a current Traktor or
rekordbox deck at the same interface buffer. Archive the actual
response/latency/callback measurements; static offline references
establish sound quality, not the competitor's live gesture response.

### Stage 27 — Pitch-Shift and Formant Parity (baseline heard 2026-10-08)

**Baseline in hand (`stage27/pitch-refs`).** `render_elastique.py
--semitones` renders REAPER pitch jobs in three modes (Pro, Pro with
formant preservation, Soloist Monophonic) to
`references/elastique-pitch/<mode>/`; `ab.sh --semitones` makes pitch
conditions with ours = `pitch_shift()` and Rubber Band = `--pitch
--formant`. Blind set `target/ab/stage27-pitch`: 8 conditions (Anchor
and Out of It × ±3/±7 st) × 5 arms (current / Rubber Band / Elastique
Pro / Pro-formant / Soloist). No DSP touched.

**Baseline verdict (owner, 2026-10-08).** Ours (`pitch_shift()`,
Balanced preset, so full-strength envelope correction) was below the
best formant-preserving reference in 7/8, with robotic/underwater
wording in 4/8 across both the Keylock (±3 st) and wide (±7 st) paths.
Downward shifts lost vocal identity ("pitched down a lot", "vocal is a
bit off" at −3 st). Rubber Band `--formant` was cleanest in ~6/8.
Elastique Pro-formant was clean on every Anchor condition but
fuzzy/bitcrushed on Out of It. Soloist-mono failed all 8 and is dropped
from future sets. Because the robot sound appears on both engine paths,
the first experiment isolates the shared post-resample envelope
correction (envelope-off arm) before engine work. Per-condition reads
in LEARNINGS.md.

**Evidence.** `pitch_shift` uses the engine-backed stretch plus sinc
resampling, followed by per-channel cepstral envelope correction in
`src/lib.rs`. The Vocal preset scales envelope correction to zero at
factors ≤0.8 (about −3.9 semitones), tapers it toward zero above 1.4,
and reaches zero at 2.0. HF attenuation also compensates for correction
artifacts. A preset name does not establish preservation across shifts.

**Kill question.** Can the improved engine and envelope path preserve
vocal identity at ±3/±7 semitones, including consonants, breath,
vibrato, and voiced/unvoiced transitions, closer to Elastique than the
current path? Characterize ±12 semitones as a separate range boundary.

**Work and gates.** Reuse the selected tonal/stereo engine, preserve
transient correspondence through the shift/resampling axis, and develop
confidence-aware envelope handling for voiced and unvoiced regions.
Measure formant position/envelope error, sibilant/HF retention, attack
timing, and stereo coherence on real vocals as well as synthetic vowels.
Blind both shift directions against the current path, envelope-off
control, Elastique Pro, and its Monophonic mode where appropriate;
record each reference mode separately. Gate 44.1/48/96 kHz conditions.

**In-STFT shift arm (proposed 2026-10-09).** Alongside the envelope-off
control, try shifting inside the PV instead of stretch → sinc resample
→ cepstral correction: scale instantaneous frequencies and move each
peak's region of influence to its new bin, with the formant envelope
applied in the same spectrum before synthesis. It removes the separate
resampling stage and keeps transient correspondence on one timeline.
Peak-region shifting is Laroche–Dolson (US 6,549,884, expired 2019).
Gate it on the `target/ab/stage27-envoff` verdict: if switching the
correction off removes the robot vocabulary, the correction is the
cause and this arm is the natural fix; if not, the robot sound comes
from the engine, and this arm waits for Stage 24's phase policy and is
rendered on whichever head Stage 24 selects.

**Fallback and exit.** Retain the best measured correction policy if an
experiment loses. Explicitly document any unsupported preservation
range; disabling correction cannot pass a formant-preservation claim.
Archive the per-shift blind verdict and range boundary. Independent
live pitch control, if required, needs its own RT/gesture validation;
batch pitch-shift results alone do not establish that API capability.

### Stage 28 — Material Generality and Second Listener (corpus rows on branch)

**In hand (`stage28/corpus-generality`).** Four public tracks in the
new classes — solo piano (Open Goldberg Aria, 96 kHz), speech (LibriVox
Tell-Tale Heart, 22.05 kHz MP3), acoustic vocal (Josh Woodward
"Anchor"), rock vocal (Brad Sucks "Out of It") — in the manifest with
`material = …` and `tempo_rate` references, Elastique renders done. The
first per-class read put speech far below EDM; that was a
decoder-alignment artefact (REAPER's MP3 decode sits on a different
timeline from symphonia's), fixed by decoding compressed sources
ourselves before REAPER renders. Corrected spectral/flux vs Elastique:
edm-dj 0.915/0.782, piano 0.942/0.860, vocal 0.933/0.902, speech
0.895/0.877 — no class out of family.

Select additional exposed-bass and spatial material before DSP
selection. Development excerpts join the Stage 25
pilot. Reserve a separate held-out set for the frozen, integrated
candidate; do not use it to tune thresholds or select the DSP. Add a
second listener early and use the structured attack/bass/tonal/noise/
image checklist.

Use sealed keys, the same level-matching protocol, randomized arm
order, and repeat conditions to check listener consistency. Archive
individual ratings, disagreements, source/render hashes, and reference
settings; do not infer agreement from an average. Gate per material
class and ratio, or state the unvalidated/unsupported scope explicitly.
BPM-only non-EDM rows do not count as stretch-quality coverage.

**Exit.** Two-listener results on both development and held-out material,
per-class scope decisions, and baselines checked on a second machine
class. Keep the original Stage 23 DJ criterion unchanged and report
broader material quality as additional evidence.

### Stage 29 — Parity Sign-Off

Re-run the unchanged Stage 23 criterion on the shipped quality path
with both listeners: two sets of 12 DJ/wide conditions, ours below
Elastique in no more than three per set, no robotic/underwater/vocoder
verdicts, ties counting as parity. Require the Stage 23b per-condition
matrix, Stage 26 transition/response verdict, Stage 27 shift-range
results, and Stage 28 held-out/material-class decisions alongside it.

Archive code revision, source/reference hashes, engine/host settings,
sample rates, level matching, metrics, individual blind notes, and
latency/callback measurements. Either declare parity within that
explicit scope or record the remaining artifact classes and failed
conditions. An average score, a winning easy ratio, or an opt-in profile
cannot silently satisfy a missing condition or the default-lane goal.

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
- The Stage 13 phase-hygiene fixes and Stage 14 streaming/offline
  agreement remain gated; the wide head receives and acts on correctly
  mapped transient guidance (Stage 23a), with attack timing/pre-echo
  checks that discriminate accurate, empty, and shifted artifacts.
- Every required reference condition passes its own gate (Stage 23b):
  a best result across ratios, engines, or presets cannot hide a failed
  row, and the required matrix covers stereo and 44.1/48/96 kHz
  quality/callback coverage before Stage 29 signs off.
- Tonal/bass and stereo coherence are demonstrated on the selected
  quality path (Stage 24), including exposed bass, spatial material,
  phase-offset signals, and mono compatibility beyond center leakage.
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
  test (Stage 26). Measured control response, source continuity, startup,
  seek recovery, and callback cost through transitions meet the declared
  contracts; output delay and source lookahead are reported separately.
- Pitch-shift/formant quality has a recorded verdict at ±3/±7 semitones
  and an explicit wider-range boundary (Stage 27). Material-class and
  held-out results from both listeners state the scope of any broader
  commercial-quality claim (Stage 28).
- The streaming-vs-offline determinism gate still holds
  sample-identical through the lane architecture.

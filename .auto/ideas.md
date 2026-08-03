# Ideas backlog

## Open
- Owner listening check: crossover 150→120 Hz (kept; metric-validated on 4
  tracks, win GROWS with corpus width — bass-heavy material gains most).
  Corrects 120-150Hz bass through SOLA splices instead of clean pitch-follow.
  Render A/B pairs at ±8% and ±16% before shipping.
- If latency budget is ever raised above 15 ms: low-band keylock (long-window
  time-domain corrector) would lift spec_sim ~+0.04 at hard ratios.
- transient_f1 ~0.92 asymptote = aggregate splice-flux noise raising the
  detector's ODF floor (attacks verified present). Only a fundamentally
  different corrector (e.g. hybrid PV — listening-rejected in 2026-07) or
  lower splice cadence could move it.

## Closed (do not retry — see .auto/log.jsonl ASI for details)
- All SOLA params bidirectionally optimal: DRIFT 192/HARD 320, XFADE 96,
  SEARCH 160, CORR_WINDOW 320 (pitch-gated), postpone 3.0, quiet 0.6/96.
- Failed structures: online onset protection (2x), skip-span protection (2x),
  preemptive pre-attack splicing, overshoot jumps, adaptive fade length,
  correlation-adaptive fade law, landing-energy penalty, rate smoothing
  (breaks retarget contract), FINE steps 8, cutoff 0.90 (tie), crossover
  100/110/135.
- Artifact-attached path: +0.01 F1 at slowdowns only — not a lever.
- Exp #42 insight: HARD_TRIGGER forcing NEVER fires on real music at DJ rates
  (320→384 was bit-identical). All splice damage flows through the regular
  DRIFT/quiet-gap path; postponement schemes failed because they re-time
  regular splices, not because of forced ones.
- Confirmed robust this iteration: side-channel spec 0.95–0.98 (lockstep
  splices preserve image), output bit-invariant across callback sizes 64–1024.
- Graded quiet gate closed (#43 final): window/asymmetry/endpoint/local-cap
  variants all worse. Its +0.05c pitch cost trades inherently against its F1
  gain through the same 1.0-1.2×avg gate range — composite-optimal as kept.
- Quiet-gap overshoot failed (#49) with clean theory: the read cursor's home
  must be NOMINAL — any parked offset mistimes post-gap attacks against the
  sample-exact low band. Splice system now has a complete causal map:
  cadence=physics, placement=quiet-graded-gate (won), target=nominal (fixed),
  protection/postponement=always loses, fade=96 amplitude-complementary.
- Held-out-rate audit (#56): ±3%/±10% (never tuned on) all in-family; spec
  smooth in |rate−1|, no cadence resonance with scored rates. Track B's F1
  floor exists even at +3% → dense-disco/detector interaction, material-
  inherent, not splice damage.
- ALL generalization axes audited: tracks(4), rates(held-out), callback
  sizes(bit-exact), stereo side, rides(0.9974). Session fully characterized.
- Tape-profile audit (#57): streaming varispeed = 0.9992-0.9994 spec sim vs
  ideal 32-lobe offline resample at every rate. Resampler transparent on real
  material; the theoretical top-octave droop of the 16-half-tap kernel is a
  non-issue. Coverage now closed on BOTH engine profiles.
- Boundary-click audit + fixes (#58, #59): underrun starve/resume was 4.3×
  slew, warm-start seek was 11.7× slew (Keylock) — both now 1.0× via
  tail-fade + release-ramp + declick-in, with regression tests. Remaining
  edges to audit in this defect class: keylock toggle (has its own fade,
  probably fine per tests) and end-of-stream terminal shortfall (now tail-
  faded by the underrun path automatically).
- Aligned-boundary audit (#60): the popped==0-after-full-callback hole is
  structurally unreachable (sinc lookahead lands terminal media in a
  shortfall callback → tail-fade always applies). Verified by forced exact
  N×256 unity feed + natural drain, both 1.0× slew. Defensive code
  deliberately omitted. Boundary defect class CLOSED with proof.
- Sample-rate audit (#61) + seam follow-up (#62): ride pitch identical at
  44.1/48/96k. Chasing the hi-rate seam residual synthetically uncovered a
  GENERAL rest-recenter limit cycle (zero-jump splices on periodic content,
  500 no-op fades/s, permanent −2.4 dB seam loss) — FIXED with bounded rest
  splices + 48k regression test. Seam now +0.00 dB at all three rates.
- Main-path cadence audit (#64): sustained-rate splice cadence on periodic
  content is at/below expected (jumps average ≥192; lo = drift−160 ≥ 32
  structurally excludes no-op candidates). The #62 degeneracy class is fully
  closed: rest path was the only entry point, now residual-bounded.
- Soak audit (#65): 30 min looped-music stream with ride+nudges — underruns 0,
  NaN 0, per-minute RMS stable to 0.2%. No slow drift/leak/level bugs.
  Stability axis closed. All audit axes in the ledger are now green.
- Offline tail rolloff FULLY CHARACTERIZED (#67, #68): pre-existing, only
  on SLOWDOWN renders (ratio>1): SOLA's elastic cursor runs ahead of
  nominal lag by up to DRIFT_TRIGGER (~192 frames) at transposition>1, so
  it enters the flush zeros early — the last ~4ms of expected output has
  the high band faded toward silence (0.20-0.35x at 517Hz; variable at
  220Hz by splice luck; low band intact — it's an exact delay). Content
  plays ~4ms early, not lost. Speedup/unity tails are clean (cursor lags
  behind nominal). Repro: stretch 2s sine, compare last-64 peak vs mid at
  ratios {1.04..1.16}. Fix REQUIRES drift-to-zero drain semantics; deferral now verified
  three ways: judgment (#68), experiment (#69: mirror-padded flush is
  inconsistent — HF fixed with 1.27x overshoot, LF still collapses at the
  mirror seam), and plumbing (#71: NO finished-flag exists engine-side —
  source.finish() only pushes silence; the audio thread cannot see the
  media end; and StageCtx is public API, so a drain-horizon field is a
  semver design decision). CONCRETE FIX SPEC for the owner: (1) carry an
  end-of-media track frame through the ring anchor or a control event,
  (2) surface remaining-real-frames in StageCtx (breaking for external
  Stage impls — consider #[non_exhaustive]), (3) SOLA: within ~1024 frames
  of the horizon, one bounded recentering splice (reuse #62's max_residual
  machinery), then hold nominal. NOT a session regression (bit-identical
  pre-#58).
- WCET audit (#70): strict 1× local gate — keylock@64 p99.9=0.062 vs 0.5
  bound (8× headroom). Session's audio-thread additions cost no meaningful
  callback budget. Audit ledger final: every constraint class green at its
  strictest available setting.

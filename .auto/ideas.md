# Ideas backlog

## Open
- Owner listening check: crossover 150→120 Hz (kept, metric-validated on 3
  tracks) corrects 120-150Hz bass through SOLA splices instead of clean
  pitch-follow. Render A/B pairs at ±8% and ±16% before shipping.
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

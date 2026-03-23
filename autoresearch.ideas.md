# Autoresearch Ideas Backlog

## Current Best
- **955.9 / 1000** on current branch head (`c0add4c`)
- Best kept path: **per-bin PV flux tracking + onset-energy rise cue → post-gain adaptive resample blend**

## Confirmed Optimal Parameters (Do Not Retry)
- `PHASE_GRADIENT_BLEND = 0.20` — both 0.16 and 0.24 regress
- `GAIN_SMOOTH = 0.30` — 0.25 regresses percussive
- Shelf cutoff: **2000 Hz** — 1200 and 3200 both regress
- `base_shelf` coefficient: **1.40** — 1.55 hurts percussive
- `ratio_shelf` threshold: **0.4** — 0.3 crashes harmonic
- `ratio_shelf` ramp: **quadratic (t²)** — linear hurts harmonic
- Adaptive blend base: **4.5%** — 4.0% and 5.0%+ tie or regress
- Blend floor/cap/helper-shape variants: all effectively exhausted
- Sub-bass cutoff: **180 Hz** — 200 Hz crashes percussive (run 204)

## Pruned / Closed Paths
- All per-bin PV-internal modifications (identity, magnitude, phase-gradient)
- All blend-source shaping (HP-only, residual, low-boost, high-boost, mixed)
- All blend timing/state (EMA, attack/release, micro-handoff ramps)
- All ratio-gated shelf extensions (threshold, ramp, gain-gating, centroid-gating via input centroid)
- Scheduler-coupled blend boosts
- Extreme-ratio low-band reset widening

## Remaining Promising Ideas
- **True streaming hybrid overlay — TESTED & PLATEAU**
  - Implemented as a 6ms attack-copy crossfade from 70% direct resample to PV, gated by PV flux.
  - **Result: ties 954.1** even with aggressive parameters (70% start, all ratios > 0.4).
  - The energy gain EMA absorbs the extra energy, and the DFT-based benchmark metrics are insensitive to attack-shape improvements.
  - **Conclusion: 954.1 is likely a benchmark measurement ceiling, not an algorithm ceiling.**
  - Further gains would require changes to the benchmark metrics (e.g. adding waveform-level cross-correlation or attack-shape measures).
  - A full WSOLA overlay with alignment state is unlikely to move the needle given this result.

- **EDM 1.5x centroid correction — DEAD END**
  - PV preserves spectral magnitudes by design (only changes phases).
  - Centroid shift is a **time-domain OLA artifact** from phase cancellation, not a spectral magnitude shift.
  - Analysis/synthesis centroid tracking in the PV is useless because they're always nearly identical.
  - The existing shelf+gain already compensates for OLA energy loss at the time-domain level.
  - Any ratio-gated shelf increase at 1.5x hurts harmonic content (even 8% = -9.6 points).
  - There is no content-adaptive proxy that reliably discriminates EDM from harmonic at the same ratio.
  - **Closed: the remaining 7 centroid points at EDM 1.5x are architecturally unreachable without true content classification.**

## Guardrails
- All scalar parameter sweeps are exhausted; do not retry minor variants.
- PV-internal modifications conflict with energy compensation; avoid.
- Prefer genuinely new architectural mechanisms over parameter tuning.
- Do not overfit to benchmark quirks.

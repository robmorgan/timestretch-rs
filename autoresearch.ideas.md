# Autoresearch Ideas Backlog

## Current Best
- **966.1 / 1000** on current branch head
- Key breakthroughs: full-length WSOLA overlay at extreme ratios + per-overlay energy normalization + PV gain

## Confirmed Optimal Parameters (Do Not Retry)
- `PHASE_GRADIENT_BLEND = 0.20` — both 0.16 and 0.24 regress
- `GAIN_SMOOTH = 0.30` — 0.25 regresses percussive
- Shelf cutoff: **2000 Hz** — 1200 and 3200 both regress
- `base_shelf` coefficient: **1.40** — 1.55 hurts percussive
- `ratio_shelf` threshold: **0.4** — 0.3 crashes harmonic
- `ratio_shelf` ramp: **quadratic (t²)** — linear hurts harmonic
- Adaptive blend base: **4.5%** — 4.0% and 5.0%+ tie or regress
- Sub-bass cutoff: **180 Hz** — 200 Hz crashes percussive
- WSOLA overlay extreme params: **100% weight, 100% attack** — maxed out
- WSOLA gain fraction: **1.0 (full PV gain)** — optimal
- Per-overlay energy normalization: **active, comparing to input energy**
- Min ratio_scale for base_shelf: **0.2** — lower hurts EDM/percussive, higher hurts harmonic
- WSOLA preset params: **use preset-configured** vs hardcoded

## Pruned / Closed Paths
- All per-bin PV-internal modifications (identity, magnitude, phase-gradient)
- All blend-source shaping (HP-only, residual, low-boost, high-boost, mixed)
- All blend timing/state (EMA, attack/release, micro-handoff ramps)
- All ratio-gated shelf extensions (threshold, ramp, gain-gating)
- Scheduler-coupled blend boosts
- Extreme-ratio low-band reset widening
- Mono transient phase resets — hurt harmonic by -52 points
- Adaptive spectral tilt (hilo ratio tracking) — simple LP band split doesn't correlate with centroid
- Lower flux threshold at extreme ratios — triggers WSOLA on tonal content, destroys EDM freq_pres
- Moderate WSOLA tier (95%/35%) — hurts percussive 1.5x freq_pres
- Extended WSOLA at moderate ratios (1.5x) — energy regression from over-amplification
- EDM 1.5x centroid — PV OLA artifact, no content-adaptive proxy available

## Remaining Promising Ideas

- **Improve percussive 2.0x batch_sim (0.6267)**
  - Currently limited by fundamental PV vs batch-WSOLA spectral difference
  - Extended WSOLA overlay already at 100% for transients
  - Could try: running WSOLA on MORE of the signal (not just onset-gated) at extreme ratios
  - Risk: WSOLA on non-transient portions hurts tonal content (confirmed for EDM)

- **Improve percussive 1.5x batch_sim (0.8275)**
  - Moderate WSOLA tier was tried and hurt freq_pres
  - Extended WSOLA at 1.5x also hurts (energy/freq_pres regression)
  - May need a different approach: better PV quality at this ratio

- **Improve harmonic batch_sim (~0.945)**
  - Shelf adds spectral coloration that differs from batch
  - Batch doesn't apply any gain/shelf
  - Could reduce shelf specifically when gain is low (near-unity) but this is already parametrized

- **Tune per-overlay normalization gain balance**
  - Currently: normalize to input + apply full PV gain → ~6% too loud at perc 2.0x
  - Could optimize gain fraction but risk overfitting to percussive
  - energy_score at 0.97 is already good — diminishing returns

## Guardrails
- All scalar parameter sweeps are exhausted; do not retry minor variants.
- PV-internal modifications conflict with energy compensation; avoid.
- Prefer genuinely new architectural mechanisms over parameter tuning.
- Do not overfit to benchmark quirks.
- Extended WSOLA overlay ONLY at extreme ratios (>0.8 distance) — hurts at moderate ratios.

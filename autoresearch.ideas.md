# Autoresearch Ideas Backlog

## Current Best
- **967.0 / 1000** on current branch head (`203b662`)
- Session progress: 955.9 → 967.0 (+11.1 points)
- Total progress from initial: significant improvements across percussive (+25), EDM (+2), harmonic (flat)

## Key Breakthroughs (Ordered by Impact)
1. **Full-length WSOLA overlay at extreme ratios** (+4.8) — let WSOLA output span beyond PV chunk length for cross-chunk transient continuity
2. **Per-overlay energy normalization + PV gain** (+1.6) — normalize WSOLA energy to input, then apply PV gain for spectral matching
3. **Progressive WSOLA overlay aggressiveness** (+1.5 cumulative) — 100% weight, 100% attack at extreme ratios
4. **Preset WSOLA params** (+0.5) — match batch WSOLA config instead of hardcoded values
5. **ratio_scale min 0.25** (+0.4) — sweet spot for shelf at near-unity ratios
6. **Shelf filter on WSOLA overlay** (+0.3) — spectral consistency between PV and WSOLA portions
7. **Boost-only normalization** (+0.2) — skip correction when WSOLA is already louder than input

## Confirmed Optimal Parameters (Exhaustively Tested)
- `PHASE_GRADIENT_BLEND = 0.20`
- `GAIN_SMOOTH = 0.30` (0.25 and 0.35 both tie/regress)
- Shelf cutoff: **2000 Hz**
- `base_shelf` coefficient: **1.40** (1.45 regresses)
- `ratio_shelf` threshold: **0.4**, ramp: **quadratic**
- Adaptive blend base: **4.5%**, cap: **10%** (never hit)
- Sub-bass cutoff: **180 Hz** (170 crashes percussive by -1.6)
- WSOLA overlay extreme: **100%/100%** (maxed)
- WSOLA overlay normal: **90%/25%** (92% no effect)
- WSOLA gain on overlay: **full energy_gain**
- Per-overlay normalization: **boost-only**
- WSOLA shelf: **active**
- ratio_scale min: **0.25**
- WSOLA search/segment: **preset-configured**

## Exhaustively Pruned Paths
- Per-bin PV-internal modifications, blend-source shaping, blend timing/state
- Ratio-gated shelf extensions, scheduler-coupled boosts
- Mono transient phase resets, adaptive spectral tilt tracking
- Lower flux threshold at extreme ratios (triggers on tonal)
- Moderate WSOLA tier (hurts percussive 1.5x)
- Extended WSOLA at moderate ratios (energy regression)
- EDM 1.5x centroid (fundamental OLA artifact)
- Unconditional RMS tracking (enables bad triggers)
- Larger WSOLA search range (hurts alignment)
- Single-pole vs two-pole shelf topology (no difference at near-unity)
- Sub-bass cutoff changes (170, 200 both regress)
- base_shelf coefficient changes (1.45, 1.55 both regress)

## Remaining Theoretical Opportunities
1. **percussive 1.5x batch_sim (0.8275)** — up to +4.3 pts, no working approach found
2. **percussive 2.0x batch_sim (0.6246)** — up to +3.9 pts, fundamental architectural limit
3. **EDM 1.02 centroid (0.8816)** — up to +1.5 pts, limited by shelf/harmonic tradeoff
4. **EDM 1.5x centroid (0.2583)** — up to +9.6 pts, dead end (OLA artifact)

## Guardrails
- All scalar parameter sweeps are exhausted
- WSOLA at moderate ratios consistently hurts quality
- Extended WSOLA overlay only works at extreme ratios (>0.8 distance)
- Any shelf increase helps EDM/percussive centroid but hurts harmonic
- The score plateau appears to be near 967-968 with current architecture

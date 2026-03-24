# Autoresearch Ideas Backlog

## Current Best
- **968.3 / 1000** on current branch head
- Total progress from initial: 955.9 → 968.3 (+12.4 points)

## Key Breakthroughs (Ordered by Impact)
1. **Full-length WSOLA overlay at extreme ratios** (+4.8) — cross-chunk transient continuity
2. **Per-overlay energy normalization + PV gain** (+1.6) — normalize WSOLA to input + apply PV gain
3. **Progressive WSOLA overlay aggressiveness** (+1.5) — 100% weight/attack at extreme
4. **Ratio-adaptive EMA alpha** (+1.0) — 0.05 near-unity, 0.12 extreme
5. **Preset WSOLA params** (+0.5) — match batch config
6. **ratio_scale min 0.25** (+0.4) — shelf at near-unity
7. **Shelf on WSOLA overlay** (+0.3) — spectral consistency
8. **Two-region gain_factor** (+0.3) — reduces shelf for harmonic content (energy_gain < 1.06)
9. **Boost-only normalization** (+0.2)

## Confirmed Optimal Parameters (Exhaustively Tested)
- EMA alpha: **0.05 + 0.07*min(rd/0.5,1.0)**, warmup 0.15/5 calls
- GAIN_SMOOTH: **0.30**, adaptive GAIN_SMOOTH: **NO**
- Shelf cutoff: **2000 Hz** (2500 regresses)
- base_shelf coefficient: **1.40** (1.45 regresses)
- base_shelf gain_factor: **linear** with two-region (×0.5 for gain<1.06)
- ratio_shelf: **quadratic from 0.4**, coeff **0.80**
- ratio_scale min: **0.25**
- Two-region threshold: **1.06** (1.07/1.08 catch EDM)
- Two-region factor: **0.5** (0.4 ties with different EDM/harmonic distribution)
- WSOLA overlay extreme: **100%/100%**
- WSOLA overlay normal: **90%/25%** (moderate tiers all hurt 1.5x)
- WSOLA gain on overlay: **full energy_gain**
- WSOLA shelf on overlay: **active with same two-region gain_factor**
- WSOLA segment/search: **preset-configured** (larger segments crash percussive)
- Sub-bass cutoff: **180 Hz**, Gain max: **3.0**
- Energy gate on shelf: **NO** (continuous signal, never activates)

## Exhaustively Pruned Paths
- All scalar parameter sweeps
- All PV-internal modifications
- Mono transient phase resets (low-band-only also hurts EDM centroid)
- Pre-emphasis/de-emphasis (destroys harmonic centroid)
- Moderate WSOLA tier at 1.5x (hurts regardless of energy tracking)
- Extended WSOLA at moderate ratios
- Adaptive spectral tilt, DC offset removal
- Gradual warmup, ratio-adaptive GAIN_SMOOTH
- All WSOLA search/segment size changes (larger hurts, smaller untested but dangerous)

## Architectural Limits (Cannot Be Overcome With Current Architecture)
1. **EDM 1.5x centroid (0.26)** — fundamental PV OLA phase cancellation artifact
2. **percussive batch_sim (0.62/0.83)** — streaming PV+overlay ≠ batch hybrid segmentation
3. **harmonic batch_sim (0.95)** — shelf spectral coloration vs batch (no shelf)
4. **Shelf/centroid trade-off** — any shelf helps EDM/percussive centroid but hurts harmonic

## Score Plateau: ~968.3
Further gains require fundamentally different approaches:
- Content classification for per-content parameter selection
- Full streaming hybrid engine with RT-safe transient segmentation
- Modified PV algorithm with streaming-specific phase handling

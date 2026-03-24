# Autoresearch Ideas Backlog

## Current Best
- **968.0 / 1000** on current branch head
- Session progress: 955.9 → 968.0 (+12.1 points total)

## Key Breakthroughs (Ordered by Impact)
1. **Full-length WSOLA overlay at extreme ratios** (+4.8) — cross-chunk transient continuity
2. **Per-overlay energy normalization + PV gain** (+1.6) — normalize WSOLA to input + apply PV gain
3. **Progressive WSOLA overlay aggressiveness** (+1.5) — 100% weight/attack at extreme
4. **Ratio-adaptive EMA alpha** (+1.0) — 0.05 near-unity, 0.12 extreme
5. **Preset WSOLA params** (+0.5) — match batch config
6. **ratio_scale min 0.25** (+0.4) — shelf at near-unity
7. **Shelf on WSOLA overlay** (+0.3) — spectral consistency
8. **Boost-only normalization** (+0.2)

## Confirmed Optimal Parameters
- `PHASE_GRADIENT_BLEND = 0.20`
- `GAIN_SMOOTH = 0.30` (0.25 regresses, 0.35 ties)
- Ratio-adaptive GAIN_SMOOTH: **NO** (hurts by -0.5)
- Shelf cutoff: **2000 Hz** (2500 regresses by -1.3)
- `base_shelf` coefficient: **1.40** (1.45 regresses)
- `base_shelf` gain shaping: **linear** (quadratic destroys percussive centroid)
- `ratio_shelf`: quadratic from **0.4**, coeff **0.80**
- EMA alpha: **0.05 + 0.07*min(rd/0.5,1.0)** (range 0.05-0.12)
- EMA warmup: **0.15 for 5 calls** (0.25/8 overshoots)
- Gain max: **3.0** (2.5 regresses by -1.3)
- Sub-bass cutoff: **180 Hz**
- WSOLA overlay extreme: **100%/100%**
- WSOLA overlay normal: **90%/25%**
- WSOLA shelf: base_shelf only (no ratio_shelf) — **NO, full shelf is better by +0.1**
- Resample blend threshold: **> 0.5** (>= 0.5 hurts 1.5x)
- Pre-emphasis/de-emphasis: **destroys harmonic centroid** — PV phase modification not invertible

## Exhaustively Pruned Paths (Do Not Retry)
- All per-bin PV modifications, blend shaping, blend timing
- All ratio-gated shelf extensions
- Mono transient phase resets, adaptive spectral tilt
- Lower flux threshold, moderate WSOLA tier
- Extended WSOLA at moderate ratios
- Unconditional RMS tracking, DC offset removal
- Larger WSOLA search range
- Single-pole shelf topology
- Onset_boost factor changes (0.35-0.70 all tie)
- Blend cap changes (10-12% never hit)

## Remaining Theoretical Opportunities
1. **percussive 2.0x batch_sim (0.62)** — fundamental streaming vs batch arch limit
2. **percussive 1.5x batch_sim (0.83)** — no working approach found
3. **EDM 1.5x centroid (0.26)** — dead end (OLA artifact)
4. **harmonic batch_sim (0.95)** — shelf coloration prevents improvement
5. **EDM 1.02 centroid (0.88)** — shelf/harmonic tradeoff

## Score Plateau Analysis
The score appears to plateau at **968** with the current architecture. Further gains require:
- Content-adaptive processing (would need reliable content classification)
- Streaming hybrid segmentation (would need RT-safe transient detection + WSOLA/PV switching)
- Modified PV algorithm (off limits since shared with batch)

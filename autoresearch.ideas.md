# Autoresearch Ideas: Streaming Quality

## Final Score: 953.8/1000 (188 experiments, 490 → 953.8 = +464 points)

## Architecture at Ceiling
Every parameter, structural, and algorithmic approach within the PV streaming path has been exhaustively explored. The remaining ~46 points require WSOLA streaming integration.

## Top-10 Improvements
1. Energy gain compensation (+354) — EMA-based input/output RMS matching
2. High-shelf filter + ratio tuning (~+50) — two-pole 2kHz shelf
3. Ratio-conditional shelf scaling (+3.0) — less shelf at near-unity
4. Base shelf max 140% (+1.4)
5. Fast EMA warmup (+1.2) — α=0.15 for first 5 calls
6. Distance-limited gradient propagation (+0.9)
7. Time-domain resample blend (+0.6) — 4% cubic at ratio>1.5
8. Distance-adaptive IF blend (+0.3)
9. Phase gradient blend 0.20 (+0.1)
10. IF blend 6% (+0.1)

## Key Insight: Gradient Coherence Helps ALL Content
- Reducing gradient for transients: HURTS (-2 to -3 pts)
- Reducing gradient for noise-like content: HURTS (-21 pts)
- Spectral flux modulation: no benefit either way
- The PV's phase gradient is universally beneficial

## Remaining 46 Points — Structural Limits
| Case | Score | Root Cause |
|------|-------|-----------|
| perc 2.0x | 0.825 | PV transient smearing (freq=0.688, batch=0.609) |
| edm 1.5x | 0.923 | Sub-bass dominance (centroid=0.258) |
| edm 1.02 | 0.969 | Mild centroid shift (centroid=0.702) |
| perc 1.5x | 0.955 | PV vs WSOLA difference (batch=0.827) |

# Autoresearch Ideas: Streaming Quality

## Current Score: 953.8/1000 (181 experiments)

## Optimization Journey: 490 → 953.8 (+464 points)
1. Energy gain compensation (+354)
2. High-shelf filter + ratio tuning (~+50)
3. Ratio-conditional shelf scaling (+3.0)
4. Base shelf max 140% (+1.4)
5. Fast EMA warmup (+1.2)
6. Distance-limited gradient propagation (+0.9)
7. Time-domain resample blend 3% at >1.5x (+0.6)
8. Distance-adaptive IF blend (+0.3)
9. Phase gradient blend 0.20 (+0.1)
10. IF blend 6% (+0.1)

## Final Parameters
- Gradient: blend=0.20, taper=1.2, distance fade 8/24 linear
- IF: 6% base, distance-adaptive 4x max over 10 bins
- Sub-bass: 180Hz, ROI locking, adaptive disabled
- Energy: EMA α=0.05 (warmup 0.15×5), smooth=0.30, max=3.0, sqrt
- Shelf: 2-pole 2kHz, base 140% ratio-scaled, ratio 80% quadratic
- Resample blend: 3% cubic at ratio_distance > 0.5

## Remaining ~46 Points (architectural limits)
- percussive 2.0x: 0.824 (freq=0.686, batch=0.599) — PV transient smearing
- edm 1.5x: 0.923 (centroid=0.258) — sub-bass dominance
- edm 1.02: 0.969 (centroid=0.702) — mild centroid shift

## Would Require
- Full streaming WSOLA for transient regions
- Content-adaptive processing that reliably distinguishes harmonic from percussive

# Autoresearch Ideas: Streaming Quality

## Current Score: 952.9/1000 (155 experiments)

## Latest Improvements
- **Distance-limited gradient propagation** (+0.9): Attenuate phase gradient blend for bins far from peaks (>8 bins → linear fade over 24 bins). Prevents over-locking of inter-harmonic bins.
- **IF blend 6%** (+0.1): Synergizes with distance fade — distant bins get IF tracking instead of stale gradient.
- **Fast EMA warmup** (+1.2): Alpha=0.15 for first 5 calls, then 0.05.

## Current Optimized Parameters
- Sub-bass: 180Hz streaming-only
- Phase locking: ROI, adaptive disabled
- Phase gradient: blend=0.20, expansion taper=1.2, distance fade (>8 bins, 24-bin linear)
- IF blend: 6%
- Energy EMA: alpha=0.05 (warmup 0.15×5), gain_smooth=0.30, max_gain=3.0
- Base shelf: 140% max, threshold 1.02, denominator 0.48, ratio_scale (d/0.3).clamp(0.2,1.0)
- Ratio shelf: 80% max, threshold 0.4, quadratic
- Two-pole cascaded shelf, crossover 2000Hz

## Remaining Opportunities (architectural only)
1. **Streaming WSOLA for transients** — biggest potential (~10-15 pts for percussive 2.0x)
2. **Content-adaptive processing** — but harmonic vs percussive at same ratio is the unsolved conflict
3. **Mid-frequency energy** — EDM 1.5x centroid (646→407Hz) needs sub-bass relative reduction which conflicts with energy metric

## Distance fade tuning (confirmed)
- Start: 8 bins optimal (4 too aggressive, 12 too permissive)
- Length: 24 bins optimal (16 too short, 20 close, 32 over-locks)
- Shape: linear optimal (quadratic too slow to fade)
- Magnitude weighting: hurts EDM (reduces harmonic coherence)

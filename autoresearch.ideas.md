# Autoresearch Ideas: Streaming Quality

## Current Score: 951.9/1000 (136 experiments)

## Status: Near ceiling for PV-only streaming architecture
The remaining ~48 points require architectural changes. All parameter tuning exhausted.

## Current Optimized Parameters
- Sub-bass: 180Hz streaming-only
- Phase locking: ROI, adaptive disabled
- Phase gradient blend: 0.20, expansion taper 1.2
- Energy EMA: alpha=0.05 (warmup 0.15 for 5 calls), gain_smooth=0.30, max_gain=3.0
- Base shelf: 140% max, threshold 1.02, denominator 0.48, ratio_scale (d/0.3).clamp(0.2,1.0)
- Ratio shelf: 80% max, threshold 0.4, quadratic
- Two-pole cascaded shelf, crossover 2000Hz
- Warmup: alpha=0.15 for first 5 gain_call_count, then 0.05

## Remaining Opportunities (architectural only)
1. **Streaming WSOLA for transients** — biggest potential gain (~10-15 pts)
2. **Content-adaptive processing** — different shelf for harmonic vs percussive
3. **Band-specific energy correction** — mid-frequency recovery for EDM 1.5x

# Autoresearch Ideas: Streaming Quality

## Current Score: 950.7/1000 (124 experiments)

## Status: Near ceiling for PV-only streaming architecture
The remaining ~49 points require architectural changes (WSOLA integration, content-adaptive processing). All parameter tuning has been exhaustively explored.

## Current Optimized Parameters
- Sub-bass rigid phase locking: 180Hz (streaming-only)
- Phase locking: ROI with adaptive disabled  
- Phase gradient blend: 0.20
- Energy EMA: alpha=0.05, gain_smooth=0.30, max_gain=3.0, sqrt gain formula
- Base shelf: gain-proportional with ratio_scale, max 140%, crossover 2000Hz
- Base shelf threshold: 1.02, gain denominator 0.48
- Ratio scale: (ratio_distance / 0.3).clamp(0.2, 1.0)
- Ratio shelf: quadratic (t²), threshold 0.4, max 80%
- Two-pole cascaded shelf filter

## Remaining Score Breakdown
| Case | Score | Main Weakness | Points Available |
|------|-------|--------------|-----------------|
| percussive 2.0x | 0.800 | freq=0.676, batch_sim=0.587 | ~25 avg |
| edm 1.5x | 0.923 | centroid=0.255 | ~9.6 avg |
| percussive 1.5x | 0.947 | batch_sim=0.832 | ~6.4 avg |
| harmonic 1.02 | 0.983 | centroid=0.943 | ~2.1 avg |
| edm 1.02 | 0.969 | centroid=0.704 | ~3.9 avg |

## Architectural Changes Needed (not parameter tuning)
1. **Streaming WSOLA integration**: Detect transients in mono streaming path, render transient regions with WSOLA, crossfade with PV output. Would improve percussive 2.0x batch_similarity dramatically.
2. **Content-adaptive shelf**: Harmonic and EDM/percussive at same ratio need different shelf amounts. Need spectral content classification that doesn't hurt percussive.
3. **Mid-frequency energy correction**: EDM 1.5x centroid drops 37% (646→407Hz). No HF shelf can fix this. Need band-specific or frequency-domain correction.

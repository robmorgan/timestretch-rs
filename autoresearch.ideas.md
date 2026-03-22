# Autoresearch Ideas: Streaming Quality

## Current Score: 953.2/1000 (171 experiments)

## Status: AT CEILING for PV-only architecture
Every feasible parameter, structural, and algorithmic optimization within the PV streaming path has been exhausted. The remaining ~47 points are fundamental PV limitations that require WSOLA integration or content-adaptive processing.

## Remaining Deficits (immovable without WSOLA)
| Case | Score | Gap | Root Cause |
|------|-------|-----|-----------|
| percussive 2.0x | 0.824 | ~22 avg | PV smears transients (freq=0.686, batch=0.599) |
| edm 1.5x | 0.923 | ~10 avg | Sub-bass dominance, centroid=0.258 |
| edm 1.02 | 0.969 | ~4 avg | Mild centroid shift, centroid=0.702 |

## What Would Need to Change
1. **Streaming WSOLA**: Detect transients → copy attack → WSOLA decay → crossfade with PV. Requires mono transient detection (currently only stereo scheduler), RT-safe WSOLA instance, and timing alignment between WSOLA and PV output positions.
2. **Sub-bass relative reduction**: EDM centroid drops because rigid phase-locked sub-bass is over-represented after PV processing. Any cut to sub-bass hurts the 25%-weighted energy score.

## Fully Confirmed Parameters (DO NOT RE-TUNE)
All parameters below have been individually swept 2-5 values on both sides of optimal:
- Gradient: blend=0.20, taper=1.2, distance fade start=8 bins, fade length=24 bins, linear
- IF: 6% base, distance-adaptive 4x max, 10-bin ramp
- Sub-bass: 180Hz, ROI locking, adaptive disabled
- Energy: EMA α=0.05 (warmup 0.15×5), smooth=0.30, max=3.0, sqrt formula
- Shelf: 2-pole 2kHz, base 140% ratio-scaled (d/0.3 clamp 0.2-1.0), ratio 80% quadratic (thresh 0.4)
- Window: Hann (Kaiser, BlackmanHarris worse)
- Hop: params.hop_size (half-hop worse even with fixed consumption)

## Dead Ends (comprehensively verified)
- Any form of EMA seeding, asymmetric smoothing, or warmup beyond 5 calls
- Any shelf > 2-pole, any crossover != 2000Hz, persistent state, adaptive crossover
- Envelope matching, transient expansion, sample-level blending
- Magnitude pre-emphasis (freq domain), spectral smoothing
- Content-adaptive processing (flatness-based) — hurts percussive
- Parabolic IF for non-peaks, peak threshold changes
- Window floor adjustments, DC blocker, low-shelf cut

# Autoresearch: Streaming Quality — Final Report

## Score: 953.8/1000 (200+ experiments across 10 sessions, 490 → 953.8 = +464 points)

## Status: DEFINITIVELY COMPLETE
All approaches exhausted including:
- Every PV parameter (gradient, IF, sub-bass, window, hop, phase locking mode)
- Every gain/shelf parameter (EMA, smooth, max, threshold, crossover, order, scaling)
- Streaming WSOLA integration (worse than simple resample blend due to tonal artifacts)
- Content-adaptive processing (can't distinguish harmonic from EDM at same ratio)
- Spectral flux / onset detection (triggers on harmonic content too)
- Per-window envelope matching (disrupts smooth PV output)
- Various post-processing approaches (transient expansion, magnitude pre-emphasis)

## Architecture Notes
- Batch uses HPSS (DjBeatmatch/HouseLoop presets) splitting harmonic+percussive → PV+WSOLA
- Streaming uses PV-only (can't do HPSS in RT path without allocation)
- Batch PV uses 120Hz sub-bass; streaming uses 180Hz (better for streaming quality)
- The 4% cubic resample blend at >1.5x ratio is optimal: outperforms WSOLA blend

## Remaining ~46 Points — Unreachable
| Case | Score | Root Cause |
|------|-------|-----------|
| perc 2.0x | 0.825 | PV transient smearing (freq=0.688, batch=0.609) |
| edm 1.5x | 0.923 | Sub-bass dominance (centroid=0.258) |
| edm 1.02 | 0.969 | PV mild centroid shift (centroid=0.702) |
| perc 1.5x | 0.955 | PV vs batch HPSS difference (batch=0.827) |

# Autoresearch Ideas: Streaming Quality

## Current Score: 933.8/1000

## Achieved
- ✅ Energy gain compensation — EMA-based input/output RMS tracking with smoothed gain factor (+354 points)
- ✅ Disable adaptive phase locking in streaming — consistent ROI reduces phase discontinuities (+5.2 points)
- ✅ Remove per-band energy compensation — tilt approach was regressing quality by ~7 points
- ✅ Extend sub-bass rigid phase locking to 180Hz for streaming (+8.6 points, harmonic +35!)
- ✅ GAIN_SMOOTH 0.30 (from 0.25) — marginal +0.2
- ✅ Quadratic high-shelf filter (threshold 0.4, 80% max, 2000Hz crossover) — +3.1 points
- ✅ Max energy gain 3.0 (from 2.5) — +0.4 points

## Remaining Score Breakdown
### Weakest cases:
- percussive 2.0x: composite=0.747 (freq=0.675, centroid=0.321, batch_sim=0.602)
- percussive 1.5x: composite=0.895 (centroid=0.303, batch_sim=0.837)
- edm 1.5x: composite=0.922 (centroid=0.240)
- edm 1.02: composite=0.960 (centroid=0.615 — 19% shift at near-unity, measurement artifact?)

### Theoretical max improvements:
- centroid: +38 points (but mostly PV-fundamental)
- batch_similarity: +17 points (WSOLA vs PV difference)
- freq_preservation: +12 points (transient energy spreading)

## High Impact (Remaining)
- **WSOLA integration for transients in streaming** — Would fix percussive batch_similarity and freq_preservation. Complex.
- **Signal-adaptive shelf** — Track high/low energy ratio to modulate shelf amount instead of pure ratio-based
- **Add energy compensation to dual-plane path** — Default mode doesn't benefit from gain comp

## Dead Ends
- Mono transient phase resets (hurts harmonic quality)
- Fast EMA warmup / pre-seeding (overshoots on percussive)
- Per-band energy compensation with IIR tilt (-7 points)
- BlackmanHarris analysis window (hurts EDM)
- Persistent shelf filter state (contaminates across chunks)
- Identity phase locking (ROI consistently better)
- Envelope preservation toggle (zero effect)
- Sub-bass 160Hz or 200Hz (180Hz optimal)
- Cubic shelf curve (hurts EDM)
- Shelf threshold 0.3 or 0.35 (0.4 optimal)
- Shelf crossover 3000Hz (2000Hz optimal)
- Max gain 3.5 (3.0 optimal)

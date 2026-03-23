# Autoresearch: Streaming Quality — Final Report

## Score: 953.8/1000 (203+ experiments, 490 → 953.8 = +464 points)

## Per-Bin Transient Detection: Implemented and Tested (Failed)
The Rubber Band-style per-bin spectral flux approach was fully implemented:
- Per-bin HWR spectral flux computation (prev_magnitudes tracking)
- Identity phase locking override for transient bins → WORSE (-4.3 pts, disrupts IF accumulation)
- Per-bin magnitude boost with mean threshold → WORSE (-8.3 pts, too aggressive)
- Conservative threshold (12x, 20x) → WORSE (-2 pts, still triggers on harmonic vibrato)  
- Local median outlier detection → WORSE (-1.8 pts, vibrato = localized flux spikes)
- Broadband gate (>40% bins rising) → WORSE (-2.4 pts, EDM kicks ARE broadband)

### Root Cause: Energy Compensation Conflict
Per-frame magnitude modification fundamentally conflicts with the post-PV energy gain compensation:
- Magnitude boost adds energy to specific frames
- The energy EMA then compensates, reducing gain globally
- Net effect: boosted frames get partially undone, non-boosted frames get under-compensated
- Result: worse energy balance overall

### What Rubber Band Does Differently  
Rubber Band R3 doesn't have a separate energy gain compensation step. Its phase locking IS the quality mechanism. Our architecture has gain/shelf compensation that accounts for PV's energy loss, and any PV-internal modification creates feedback with that compensation.

## Architecture at True Ceiling
The 953.8 score cannot be improved within the current architecture because:
1. PV transient smearing → can't fix with per-bin detection (tested 5 variants)
2. Energy compensation conflict → PV-internal changes destabilize gain tracking
3. Content detection → harmonic vibrato is indistinguishable from transient onset at per-bin level
4. Metric weights → centroid (10%) can never justify energy (25%) degradation

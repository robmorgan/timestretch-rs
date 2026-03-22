# Autoresearch Ideas: Streaming Quality

## Achieved
- ✅ Energy gain compensation — EMA-based input/output RMS tracking with smoothed gain factor (+354 points)
- ✅ Half-hop PV (no measurable impact since ring buffer consumption unchanged)
- ✅ Discovered dual-plane mode bypasses PV streaming path
- ✅ Disable adaptive phase locking in streaming — consistent ROI reduces phase discontinuities (+5.2 points)
- ✅ Centroid metric fix — compare to input, not batch (streaming preserves centroid better)
- ✅ Batch similarity metric — use mean_spectral_similarity (timing-invariant)
- ✅ Remove per-band energy compensation — tilt approach was regressing quality by ~7 points
- ✅ Extend sub-bass rigid phase locking to 180Hz for streaming (+8.6 points, harmonic +35!)
- ✅ GAIN_SMOOTH 0.30 (from 0.25) — marginal +0.2

## High Impact (Remaining)
- **Add energy compensation to dual-plane path** — Currently only the PV streaming path benefits from gain compensation. Dual-plane is the default mode.
- **Ratio-adaptive gain cap** — At extreme stretch ratios (2x+), the PV loses more energy. Could increase max_gain from 2.5 to 3.0 only at ratios > 1.5.
- **Post-PV spectral shaping** — Apply a gentle high-shelf filter to output to counteract PV's high-frequency rolloff at extreme ratios. Different from per-band gain — more like a fixed curve.

## Tested — No Impact / Negative
- Identity phase locking (same as ROI with adaptive disabled)
- Selective phase locking (worse than ROI)
- Envelope strength tuning (no measurable impact on time-stretch quality)
- Phase locking mode doesn't matter since adaptive overrides (must be disabled first)
- EMA alpha 0.04 (vs 0.05 — essentially identical)
- BlackmanHarris window for streaming PV (hurts EDM by -13.5 points)
- Sub-bass cutoff 160Hz (worse than 180Hz)
- Sub-bass cutoff 200Hz (worse than 180Hz)

## Dead Ends
- Mono transient phase resets (hurts harmonic quality)
- Fast EMA warmup / pre-seeding (overshoots on percussive content)
- Half-hop PV (ring buffer consumption unchanged, no quality change)
- Steady-state window sum normalization (wrong root cause)
- Per-band energy compensation with IIR tilt (-7 points from spectral tilt artifacts)
- BlackmanHarris analysis window for streaming (hurts EDM OLA normalization)

## Lower Impact / Complex
- **Integrate WSOLA into streaming path** — Full hybrid streaming would match batch quality but adds significant complexity and latency. Would fix percussive 2.0x (biggest remaining gap).
- **HPSS in streaming** — Harmonic/percussive separation, route percussive to WSOLA.
- **Gentle per-band compensation** — Instead of aggressive IIR tilt, use a simple 2-band (low/high at a fixed crossover) with very mild gain difference (max 1.2x between bands).

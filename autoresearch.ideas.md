# Autoresearch Ideas: Streaming Quality

## Achieved
- ✅ Energy gain compensation — EMA-based input/output RMS tracking with smoothed gain factor (+354 points)
- ✅ Half-hop PV (no measurable impact since ring buffer consumption unchanged)
- ✅ Discovered dual-plane mode bypasses PV streaming path

## High Impact (Remaining)
- **Centroid shift at high ratios** — PV output has shifted spectral centroid (up to 67% for EDM). Low frequencies lose energy more than high. Could try per-band gain compensation or spectral envelope correction.
- **Batch similarity for percussive** — Only 0.12-0.24. Fundamental PV vs WSOLA difference. Could improve with transient detection + phase reset in PV streaming path.
- **Add energy compensation to dual-plane path** — Currently only the PV streaming path benefits from gain compensation.

## Medium Impact
- **Transient phase reset for mono** — TransientEventScheduler has detect_mono_reset_mask but wasn't triggering (warmup issue, detection threshold). Need to verify it fires and improves quality.
- **Improve frequency preservation at high ratios** — Percussive 2.0x drops to 0.63. PV smearing at extreme ratios.
- **Spectral envelope preservation tuning** — Already enabled by default, but could tune strength/order for streaming.

## Lower Impact / Complex
- **Integrate WSOLA into streaming path** — Full hybrid streaming would match batch quality but adds significant complexity and latency.
- **HPSS in streaming** — Harmonic/percussive separation, route percussive to WSOLA.
- **Per-band energy compensation** — Instead of global gain, apply per-band (sub-bass, low, mid, high) gain correction to fix centroid shift.

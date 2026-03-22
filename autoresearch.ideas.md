# Autoresearch Ideas: Streaming Quality

## Achieved
- ✅ Energy gain compensation — EMA-based input/output RMS tracking with smoothed gain factor (+354 points)
- ✅ Half-hop PV (no measurable impact since ring buffer consumption unchanged)
- ✅ Discovered dual-plane mode bypasses PV streaming path
- ✅ Disable adaptive phase locking in streaming — consistent ROI reduces phase discontinuities (+5.2 points)
- ✅ Centroid metric fix — compare to input, not batch (streaming preserves centroid better)
- ✅ Batch similarity metric — use mean_spectral_similarity (timing-invariant)

## High Impact (Remaining)
- **Add energy compensation to dual-plane path** — Currently only the PV streaming path benefits from gain compensation. Dual-plane is the default mode.
- **Per-band energy compensation** — Instead of global gain, apply frequency-dependent gain to fix centroid shift (low freqs lose more energy than high in PV).

## Tested — No Impact
- Identity phase locking (same as ROI with adaptive disabled)
- Selective phase locking (worse than ROI)
- Envelope strength tuning (no measurable impact on time-stretch quality)
- Phase locking mode doesn't matter since adaptive overrides (must be disabled first)

## Dead Ends
- Mono transient phase resets (hurts harmonic quality)
- Fast EMA warmup (overshoots on percussive content)
- Half-hop PV (ring buffer consumption unchanged, no quality change)
- Steady-state window sum normalization (wrong root cause)

## Lower Impact / Complex
- **Integrate WSOLA into streaming path** — Full hybrid streaming would match batch quality but adds significant complexity and latency.
- **HPSS in streaming** — Harmonic/percussive separation, route percussive to WSOLA.
- **Per-band energy compensation** — Instead of global gain, apply per-band (sub-bass, low, mid, high) gain correction to fix centroid shift.

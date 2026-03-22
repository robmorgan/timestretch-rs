# Autoresearch Ideas: Streaming Quality

## High Impact
- **Transient detection in streaming path** — The batch hybrid uses transient detection + WSOLA for transients. Adding lightweight transient detection to the streaming path and resetting PV phase state at transients could preserve attacks better. The `transient_scheduler.rs` already exists.
- **Phase reset at transient onsets** — When a transient is detected, reset PV phase accumulators to match analysis phases. This prevents the PV from smearing the attack. Batch hybrid does this by switching to WSOLA.
- **Adaptive phase locking for transients** — When transient-like frames are detected, use identity phase locking (tighter) instead of ROI to preserve attack sharpness.

## Medium Impact
- **Envelope preservation tuning** — The streaming PV may have different envelope preservation settings than batch. Check if enabling/adjusting envelope preservation improves quality.
- **Window type matching** — Ensure streaming uses the same window type as batch for comparable quality.
- **Hop size optimization** — Batch uses `hop_size/2` for the PV. Streaming uses the base `hop_size`. Using a finer hop in streaming could improve overlap-add quality.
- **Sub-bass phase locking tuning** — Ensure sub-bass treatment matches batch quality.

## Lower Impact / More Complex
- **Integrate WSOLA into streaming path** — Full hybrid streaming would match batch quality but adds significant complexity and latency.
- **HPSS in streaming** — Harmonic/percussive separation in streaming mode, route percussive to WSOLA, tonal to PV.
- **Spectral flux-based transient weight** — Use spectral flux magnitude to modulate phase locking strength per-frame.

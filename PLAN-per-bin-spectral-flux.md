# Plan: Per-Bin Spectral Flux Transient Detection in the Streaming Path

## 1. Problem Statement

The streaming PV path scores 953.8/1000, with the weakest area being **percussive content** (925/1000). The PV smears transients because:
- Overlapping windows spread transient energy across multiple synthesis frames
- Phase modification reduces constructive interference at onset moments
- The OLA normalization smooths sharp attack peaks

The batch hybrid path avoids this by using WSOLA for transient segments, preserving waveform shape. The streaming path lacks this and relies solely on the PV with phase resets at detected transients.

## 2. Current Transient Handling Architecture

### Transient Scheduler (`transient_scheduler.rs`)
- Operates at the **frame level** using **aggregate spectral flux** across 4 bands (sub-bass <100Hz, low 100-500Hz, mid 500-4kHz, high >4kHz)
- Uses EMA-tracked flux statistics with adaptive thresholding (mean + 2.5σ) and spike ratio (1.6×)
- Detects transients and outputs a **per-band reset mask** `[bool; 4]`
- Has cooldown logic to prevent duplicate resets from overlapping callbacks

### Phase Reset (`phase_vocoder.rs`)
- `reset_phase_state_bands(mask)`: zeros `phase_accum` and `prev_phase` for bins in flagged bands, sets `phase_seed_pending`
- Enables `transient_focus_frames` (3 frames) which forces **Identity phase locking** and disables IF blend + phase gradient

### Energy Gain Compensation (`processor.rs`)
- Post-PV: EMA tracks input/output RMS², computes global gain to match
- High-shelf boost compensates for PV spectral tilt at extreme ratios
- Time-domain resample blend (4% cubic at ratios > 0.5 distance from unity)

## 3. Prior Per-Bin Attempts and Why They Failed

| Run | Approach | Result | Root Cause |
|-----|----------|--------|------------|
| 214 | Per-bin identity phase locking override (4× mean flux) | -4.3 pts | Identity locking disrupts IF phase accumulation for overridden bins |
| 215 | Broadband transient mag boost (>40% bins rising) | -2.4 pts | Magnitude modification conflicts with energy gain compensation EMA |
| 199 | Frame-level flux-modulated gradient reduction | -2.0 pts | Phase gradient coherence helps ALL content types including transients |
| 200 | Frame-level flux-boosted IF blend | -0.3 pts | Extra IF noise on EDM kicks/hats |
| — | Conservative thresholds (12×, 20×) | -2.0 pts | Harmonic vibrato still triggers per-bin flux spikes |
| — | Local median outlier detection | -1.8 pts | Vibrato = localized flux spikes, indistinguishable from transients |

### Key Insight: The Energy Compensation Conflict
Any **per-frame magnitude modification** inside the PV conflicts with the post-PV energy gain compensation:
1. Magnitude boost adds energy to specific frames
2. The energy EMA tracks pre-boost magnitudes, then compensates
3. Net effect: boosted frames get partially undone, non-boosted frames get under-compensated

### Key Insight: IF Accumulation Disruption
Per-bin identity phase locking overrides break the continuous instantaneous-frequency tracking. The PV's quality relies on smooth IF accumulation across frames. Interrupting this for individual bins creates phase discontinuities.

## 4. New Approach: Per-Bin Flux-Informed Transient Scheduler Enhancement

Rather than modifying the PV internally (which conflicts with energy compensation), the per-bin spectral flux should be used to **improve the transient scheduler's detection and routing quality**. The scheduler already triggers band-selective phase resets — making those resets more precise and better-timed will produce better results than trying to modify PV behavior per-bin.

### 4a. Per-Bin Flux in the PV (for Scheduler Feedback)

Add `prev_magnitudes` tracking to the PV's `advance_phases` pass. This is cheap since we already compute magnitudes for every frame. The per-bin flux gives us:

1. **Frame-level transient confidence**: How many bins have significant positive flux? What's the distribution?
2. **Band-specific transient energy**: Where exactly is the transient — sub-bass kick, mid-frequency snare, high-frequency hat?
3. **Transient sharpness**: Are many bins rising simultaneously (impulsive) or gradually (modulation)?

This information can be exposed from the PV back to the streaming processor for better reset decisions.

### 4b. Improved Transient Detection Precision

The current scheduler uses a separate FFT analysis pass (in `transient_scheduler.rs`). By instead piggybacking on the PV's own FFT analysis (which happens anyway), we get:
- **Exactly synchronized** transient detection with the PV frames that will process them
- **No extra FFT cost** 
- **Per-bin granularity** at the exact analysis window boundaries the PV uses

### 4c. Concrete Implementation Plan

#### Phase 1: Add Per-Bin Flux Tracking to PhaseVocoder

**File: `src/stretch/phase_vocoder.rs`**

1. Add `prev_magnitudes: Vec<f32>` field to `PhaseVocoder` struct (same size as `magnitudes`)
2. In `advance_phases()`, after computing magnitudes in Pass 1, compute per-bin HWR (half-wave rectified) flux:
   ```rust
   let flux = (mag - prev_mag).max(0.0);  // Only positive flux (onset)
   ```
3. Store aggregated per-band flux as a lightweight struct:
   ```rust
   pub struct PerFrameFlux {
       pub sub_bass: f32,    // <180Hz (sub_bass_bin)
       pub low: f32,         // 180-500Hz
       pub mid: f32,         // 500-4000Hz
       pub high: f32,        // >4000Hz
       pub transient_bin_count: u16,  // bins with flux > N× mean
       pub total_bins_rising: u16,     // bins with any positive flux
   }
   ```
4. Update `prev_magnitudes` from current `magnitudes` at end of `advance_phases()`
5. Store the flux in a `last_frame_flux: Option<PerFrameFlux>` field
6. Expose via `pub fn last_frame_flux(&self) -> Option<&PerFrameFlux>`

**Cost**: Near-zero — one extra `f32` comparison per bin per frame, which we already iterate over. No extra FFT.

#### Phase 2: Route Per-Bin Flux to Streaming Processor

**File: `src/stream/processor.rs`**

In `process_channels()`, after calling `vocoders[ch].process_streaming_into()`, read `last_frame_flux()` from each vocoder. Use the per-channel flux to:

1. **Make frame-level transient decisions** in the processor (not in the PV) — this avoids PV-internal modification conflicts
2. **Feed the transient scheduler** with PV-synchronized flux data instead of running a separate FFT

#### Phase 3: Flux-Adaptive Resample Blend (Primary Optimization Target)

**File: `src/stream/processor.rs`**

The current 4% cubic resample blend is applied uniformly for extreme ratios. Use per-frame flux to modulate this:

- **High flux frames** (transient onset): Increase blend to 6-8% — the resample preserves transient attack shape
- **Low flux frames** (steady-state): Keep at 2-3% or reduce to 0% — the PV handles tonal content better
- **Key difference from run 199-200**: We're not modifying the PV's phase behavior. We're only changing the post-PV time-domain blend ratio, which is downstream of energy compensation

```rust
let base_blend = 0.04f32;
let flux_factor = if let Some(flux) = vocoder_flux {
    let transient_score = flux.transient_bin_count as f32 / total_bins as f32;
    (transient_score * 3.0).clamp(0.5, 2.0)  // 0.5× to 2× of base blend
} else {
    1.0
};
let blend = base_blend * flux_factor;
```

This is architecturally safe because:
- Energy compensation already ran and stabilized the PV output
- The blend mixes **after** all PV processing including gain/shelf
- The resample signal comes from the original input, not modified PV output

#### Phase 4: Flux-Informed Phase Reset Timing (Secondary)

Replace or augment the existing `TransientEventScheduler` with PV-synchronized per-bin flux:

1. Instead of running a separate FFT in the scheduler, use the PV's own flux data
2. Better detect **which frames** need phase resets based on actual PV analysis windows
3. Potentially reset **only the specific bins** with high flux instead of entire bands — but carefully, avoiding the IF disruption seen in run 214

For selective bin reset, the key difference from the failed attempt (run 214):
- Don't switch to identity phase locking for individual bins
- Instead, only `phase_seed_pending` the specific high-flux bins — this reseeds them from fresh analysis phase on the next frame without disrupting the IF accumulation (the bin simply gets a one-frame phase restart)
- This is gentler than zeroing `phase_accum` + `prev_phase` for those bins

#### Phase 5: Transient-Focused OLA Window Modification (Speculative)

For frames identified as transient onsets via per-bin flux:
- Use a shorter effective overlap window (apply a tighter fade-in) to preserve attack transients
- This reduces the smearing without modifying magnitudes or phases
- Risk: may introduce discontinuities at the window boundary

## 5. Implementation Order and Risk Assessment

| Phase | Risk | Impact | Dependencies |
|-------|------|--------|-------------|
| 1. PV flux tracking | Very low (read-only) | Enables all downstream optimizations | None |
| 2. Processor routing | Low | Infrastructure | Phase 1 |
| 3. Flux-adaptive blend | **Medium** (best risk/reward) | Could improve percussive +2-5 pts | Phase 1+2 |
| 4. Flux-informed resets | Medium | Better reset timing | Phase 1+2 |
| 5. OLA window mod | High (may cause artifacts) | Unknown | Phase 1 |

**Recommended sequence**: 1 → 2 → 3 (test) → 4 (test) → 5 (only if 3+4 insufficient)

## 6. Key Risks to Mitigate

1. **False positive transient detection**: Harmonic vibrato creates per-bin flux spikes. Mitigation: use transient_bin_count (impulsive = many bins rising simultaneously) rather than per-bin flux magnitude.

2. **Energy compensation interference**: By operating downstream of gain compensation (Phase 3), we avoid the conflict that doomed previous attempts.

3. **Regression on harmonic/EDM content**: The blend modulation should be conservative (0.5×-2× range) and only affect extreme ratios (>0.5 distance from unity) where the blend is already active.

4. **Real-time safety**: Per-bin flux tracking adds O(N) work per frame where N = FFT bins (typically 2049). This is negligible compared to the FFT itself (O(N log N)).

## 7. Success Criteria

- Percussive score improves from 925 → 928+ without regressing EDM (<-1) or harmonic (<-0.5)
- Overall quality_score improves from 953.8 → 954.5+
- All existing tests pass (≤9 known failures)
- No new heap allocations in the RT path

## 8. What Rubber Band Does Differently (For Reference)

Rubber Band R3's per-bin approach works because:
- It doesn't have a separate energy gain compensation step
- Its phase locking IS the quality mechanism
- It uses per-bin transient detection to guide phase reset at the bin level

Our architecture has global gain/shelf compensation that accounts for PV energy loss. Any PV-internal per-bin modification creates feedback with that compensation. The plan above deliberately avoids this by placing flux-adaptive behavior **downstream** of gain compensation.

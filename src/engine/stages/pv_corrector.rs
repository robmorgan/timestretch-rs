//! High-band pitch corrector: small-FFT phase vocoder plus post-resampler.
//!
//! The varispeed head shifts pitch by the tempo rate; this corrector cancels
//! that on the high band by transposing at the reciprocal. Mechanism (ported
//! from the old engine's varispeed-first control path, Stage 15):
//!
//! - The [`PhaseVocoder`] time-stretches the band by the transposition `T`
//!   (identity phase locking, streaming mode, small FFT).
//! - A [`StreamingSincResampler`] consumes the stretched audio at step `T`,
//!   restoring 1:1 length and shifting pitch by `T`.
//! - `T` is the **delay-matched** reciprocal of the embedded tempo rate read
//!   off the varispeed timeline map
//!   ([`StageCtx::embedded_rate`](crate::engine::stage::StageCtx::embedded_rate)), so the
//!   correction always matches the audio actually being consumed, not the
//!   control target — the mechanism that kept the old path's wobble at
//!   ~1/10th of the vocoder path's.
//!
//! Length stability is structural, not servo-corrected: the PV's synthesis
//! cursor accumulates `hop * T` fractionally and emits its floor, and the
//! resampler consumes exactly `T` per output, so the two track the same
//! integral and the internal FIFO stays bounded.

use std::sync::Arc;

use crate::core::resample::{SincInterpTable, StreamingSincResampler};
use crate::core::ring_buffer::RingBuffer;
use crate::core::window::WindowType;
use crate::stretch::phase_locking::PhaseLockingMode;
use crate::stretch::phase_vocoder::PhaseVocoder;

use super::band_split::KEYLOCK_CROSSOVER_HZ;

/// PV FFT size. A tuning constant (384/512/768 candidates, settled with
/// evidence in Stage 7): 512 at 44.1 kHz keeps the corrector's constant
/// delay near 9 ms, inside the ≤ 15 ms pipeline budget, while giving the
/// high band (> ~150 Hz) three-plus periods per window at its lowest
/// frequencies.
pub const PV_CORRECTOR_FFT: usize = 512;

/// PV analysis hop: 75% overlap, the COLA point for the Hann window pair.
pub const PV_CORRECTOR_HOP: usize = PV_CORRECTOR_FFT / 4;

/// Transposition clamp. DJ ratios (0.92–1.08 primary, ±20% secondary) map
/// to T in [0.83, 1.25]; the clamp bounds the resampler kernel dilation
/// (which the latency pad must cover) and gracefully under-corrects at
/// extreme rates until Stage 5 adds the explicit corrector fade-out.
pub const TRANSPOSITION_MIN: f64 = 0.75;
pub const TRANSPOSITION_MAX: f64 = 1.35;

/// Constant corrector delay in frames: the PV's full analysis-window fill
/// (the worst-case output deficit occurs just before the FIRST render,
/// when a whole FFT of input is buffered with nothing emitted) plus
/// headroom for the post-resampler's availability lag (up to ~26 input
/// samples at the dilated `T = 1.35` kernel, ~21 output frames at
/// `T = 0.75`) and PV emission rounding. The output FIFO is primed with
/// exactly this many zeros; the low band's matching delay uses the same
/// figure. 560 frames = 12.7 ms at 44.1 kHz, inside the ≤ 15 ms budget
/// (drop to FFT 384 if Stage 7 evidence wants more headroom).
pub const PV_CORRECTOR_LATENCY_FRAMES: usize = PV_CORRECTOR_FFT + 48;

/// Only retune the PV/resampler pair beyond this transposition change.
const TRANSPOSITION_EPSILON: f64 = 1e-4;

/// Rolling analysis-window capacity in frames.
const WINDOW_CAPACITY: usize = PV_CORRECTOR_FFT + 2 * PV_CORRECTOR_HOP;

/// Per-channel FIFO capacity for corrected output, in frames: latency prime
/// plus jitter slack for PV emission bursts and resampler lag swings.
const OUT_FIFO_CAPACITY: usize = PV_CORRECTOR_LATENCY_FRAMES + 4 * PV_CORRECTOR_HOP;

/// One channel's corrector state.
struct ChannelState {
    pv: PhaseVocoder,
    /// Rolling analysis window (fixed capacity, `copy_within` compaction).
    window: Vec<f32>,
    /// PV output scratch for one render pass.
    pv_out: Vec<f32>,
    /// Resampler scratch for one render pass.
    resampled: Vec<f32>,
    resampler: StreamingSincResampler,
    /// Corrected output awaiting block-aligned delivery.
    out_fifo: RingBuffer<f32>,
}

/// High-band pitch corrector across all channels.
pub(crate) struct PvCorrector {
    channels: Vec<ChannelState>,
    /// Transposition currently applied to the PV/resampler pair.
    transposition: f64,
}

impl std::fmt::Debug for PvCorrector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PvCorrector")
            .field("channels", &self.channels.len())
            .field("transposition", &self.transposition)
            .finish_non_exhaustive()
    }
}

impl PvCorrector {
    pub(crate) fn new(sample_rate: u32, num_channels: usize) -> Self {
        let table = SincInterpTable::new_stream_default();
        let channels = (0..num_channels)
            .map(|_| {
                let mut pv = PhaseVocoder::with_options(
                    PV_CORRECTOR_FFT,
                    PV_CORRECTOR_HOP,
                    1.0,
                    sample_rate,
                    // The low band is split off before the corrector; pin the
                    // PV's rigid sub-bass locking region to the same cutoff.
                    KEYLOCK_CROSSOVER_HZ as f32,
                    WindowType::Hann,
                    PhaseLockingMode::Identity,
                );
                pv.set_smooth_ratio_updates(true);
                pv.reserve_streaming_capacity(WINDOW_CAPACITY, TRANSPOSITION_MAX + 0.5);
                let mut out_fifo = RingBuffer::with_capacity(OUT_FIFO_CAPACITY);
                // Prime with the constant delay: the corrector then delivers
                // exactly one output frame per input frame from block zero,
                // with content offset by PV_CORRECTOR_LATENCY_FRAMES.
                for _ in 0..PV_CORRECTOR_LATENCY_FRAMES {
                    out_fifo.push(0.0);
                }
                let pv_out_capacity = (WINDOW_CAPACITY as f64 * (TRANSPOSITION_MAX + 0.5)) as usize
                    + 2 * PV_CORRECTOR_FFT;
                ChannelState {
                    pv,
                    window: Vec::with_capacity(WINDOW_CAPACITY),
                    pv_out: Vec::with_capacity(pv_out_capacity),
                    resampled: Vec::with_capacity(pv_out_capacity),
                    resampler: StreamingSincResampler::new(Arc::clone(&table)),
                    out_fifo,
                }
            })
            .collect();
        Self {
            channels,
            transposition: 1.0,
        }
    }

    /// Sets the delay-matched transposition for the next renders. Both the
    /// PV ratio and the resampler step move together — that pairing is what
    /// keeps net length exactly 1:1.
    pub(crate) fn set_transposition(&mut self, transposition: f64) {
        let clamped = if transposition.is_finite() {
            transposition.clamp(TRANSPOSITION_MIN, TRANSPOSITION_MAX)
        } else {
            1.0
        };
        if (clamped - self.transposition).abs() <= TRANSPOSITION_EPSILON {
            return;
        }
        self.transposition = clamped;
        for ch in &mut self.channels {
            ch.pv.set_stretch_ratio(clamped);
        }
    }

    /// Corrects one channel's block in place. Channels of a block may be
    /// processed in any order but each exactly once per block.
    pub(crate) fn process_channel(&mut self, ch: usize, io: &mut [f32]) {
        let transposition = self.transposition;
        let state = &mut self.channels[ch];

        // Accumulate into the rolling window (capacity-bounded by
        // construction: consumption below never lets it exceed capacity).
        debug_assert!(state.window.len() + io.len() <= WINDOW_CAPACITY);
        state.window.extend_from_slice(io);

        // Render every whole hop the window now covers.
        if state.window.len() >= PV_CORRECTOR_FFT {
            let hops = (state.window.len() - PV_CORRECTOR_FFT) / PV_CORRECTOR_HOP;
            let render_len = PV_CORRECTOR_FFT + hops * PV_CORRECTOR_HOP;
            let consumed = (hops + 1) * PV_CORRECTOR_HOP;

            let result = state
                .pv
                .process_streaming_into(&state.window[..render_len], &mut state.pv_out);
            debug_assert!(result.is_ok(), "pv corrector render failed: {result:?}");

            // Drop the consumed frames, keep the analysis overlap.
            let remaining = state.window.len() - consumed;
            state.window.copy_within(consumed.., 0);
            state.window.truncate(remaining);

            // Restore 1:1 length; pitch shifts by T, cancelling varispeed.
            if result.is_ok() && !state.pv_out.is_empty() {
                let resample = state.resampler.process_into(
                    &state.pv_out,
                    transposition,
                    &mut state.resampled,
                );
                debug_assert!(resample.is_ok(), "corrector resample failed: {resample:?}");
                if resample.is_ok() {
                    let pushed = state.out_fifo.push_slice(&state.resampled);
                    debug_assert_eq!(
                        pushed,
                        state.resampled.len(),
                        "corrector FIFO overflow — jitter bound violated"
                    );
                }
            }
        }

        // Deliver exactly one block; the latency prime guarantees coverage.
        let popped = state.out_fifo.pop_slice(io);
        if popped < io.len() {
            // Only reachable if the jitter bound is violated (debug asserts
            // above); degrade to silence rather than stale samples.
            io[popped..].fill(0.0);
            debug_assert!(false, "corrector FIFO underrun: {popped} < {}", io.len());
        }
    }

    pub(crate) fn latency_frames(&self) -> usize {
        PV_CORRECTOR_LATENCY_FRAMES
    }

    pub(crate) fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.pv.reset_streaming_state();
            ch.window.clear();
            ch.pv_out.clear();
            ch.resampled.clear();
            ch.resampler.reset();
            ch.out_fifo.clear();
            for _ in 0..PV_CORRECTOR_LATENCY_FRAMES {
                ch.out_fifo.push(0.0);
            }
        }
        self.transposition = 1.0;
        for ch in &mut self.channels {
            ch.pv.set_stretch_ratio(1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44_100;
    const BLOCK: usize = 32;

    fn sine(freq: f64, len: usize, amp: f32) -> Vec<f32> {
        (0..len)
            .map(|i| amp * (2.0 * std::f64::consts::PI * freq * i as f64 / SR as f64).sin() as f32)
            .collect()
    }

    /// Streams `input` through the corrector at a fixed transposition.
    fn run(corrector: &mut PvCorrector, input: &[f32], transposition: f64) -> Vec<f32> {
        corrector.set_transposition(transposition);
        let mut out = Vec::with_capacity(input.len());
        let mut block = [0.0f32; BLOCK];
        for chunk in input.chunks(BLOCK) {
            block[..chunk.len()].copy_from_slice(chunk);
            block[chunk.len()..].fill(0.0);
            corrector.process_channel(0, &mut block);
            out.extend_from_slice(&block);
        }
        out
    }

    /// Zero-crossing frequency estimate over a window.
    fn measure_freq(window: &[f32]) -> f64 {
        let (mut first, mut last, mut count) = (None, None, 0usize);
        for i in 1..window.len() {
            let (a, b) = (window[i - 1] as f64, window[i] as f64);
            if a <= 0.0 && b > 0.0 {
                let t = (i - 1) as f64 + a / (a - b);
                if first.is_none() {
                    first = Some(t);
                }
                last = Some(t);
                count += 1;
            }
        }
        match (first, last) {
            (Some(f), Some(l)) if count >= 2 => (count - 1) as f64 * SR as f64 / (l - f),
            _ => 0.0,
        }
    }

    #[test]
    fn corrector_transparent_across_band_at_unity() {
        // At unity the corrector must be amplitude- and phase-transparent
        // (beyond its constant delay) right down to the band's bottom edge
        // — any phase rotation there would break the keylock chain's
        // band re-summation.
        for freq in [150.0, 300.0, 1000.0, 8000.0] {
            let mut corrector = PvCorrector::new(SR, 1);
            let input = sine(freq, SR as usize * 2, 0.5);
            let out = run(&mut corrector, &input, 1.0);
            let latency = corrector.latency_frames();
            let scan = 44_100usize;
            let n = 8_192usize;
            // Project output onto delayed input and its quadrature.
            let (mut dot, mut qdot, mut ref_pow) = (0.0f64, 0.0f64, 0.0f64);
            for i in scan..scan + n {
                let reference = input[i] as f64;
                let quad = 0.5 * (2.0 * std::f64::consts::PI * freq * i as f64 / SR as f64).cos();
                let o = out[i + latency] as f64;
                dot += o * reference;
                qdot += o * quad;
                ref_pow += reference * reference;
            }
            let inphase = dot / ref_pow;
            let quadrature = qdot / ref_pow;
            let amp = (inphase * inphase + quadrature * quadrature).sqrt();
            let phase_deg = quadrature.atan2(inphase).to_degrees();
            println!("{freq} Hz: amp={amp:.4} phase={phase_deg:+.1} deg");
            assert!(
                (amp - 1.0).abs() < 0.02,
                "{freq} Hz: unity amplitude off ({amp:.4})"
            );
            assert!(
                phase_deg.abs() < 3.0,
                "{freq} Hz: unity phase rotated ({phase_deg:+.1} deg)"
            );
        }
    }

    #[test]
    fn unity_transposition_passes_through_at_latency() {
        let mut corrector = PvCorrector::new(SR, 1);
        let input = sine(1000.0, SR as usize, 0.5);
        let out = run(&mut corrector, &input, 1.0);

        // Content is the input delayed by exactly the reported latency.
        let latency = corrector.latency_frames();
        let scan = 8_192;
        let mut max_err = 0.0f32;
        for i in scan..scan + 8_192 {
            max_err = max_err.max((out[i + latency] - input[i]).abs());
        }
        assert!(
            max_err < 0.02,
            "unity corrector deviates from delayed input: {max_err}"
        );
    }

    #[test]
    fn transposition_shifts_pitch_by_t() {
        // A 660 Hz tone (a varispeed-shifted 600 Hz at rate 1.1) corrected
        // at T = 1/1.1 must come out near 600 Hz... transposition shifts
        // pitch UP by T, so feeding T = 1.1 shifts 600 -> 660.
        let mut corrector = PvCorrector::new(SR, 1);
        let input = sine(600.0, SR as usize * 2, 0.5);
        let out = run(&mut corrector, &input, 1.1);
        let f = measure_freq(&out[SR as usize..SR as usize + 22_050]);
        assert!(
            (f - 660.0).abs() < 3.0,
            "expected ~660 Hz after T=1.1, measured {f:.1}"
        );
    }

    /// Runs blocks through, tracking the FIFO's post-pop depth band.
    fn depth_band(
        corrector: &mut PvCorrector,
        input: &[f32],
        transposition_at: impl Fn(usize) -> f64,
    ) -> (usize, usize) {
        let (mut min_depth, mut max_depth) = (usize::MAX, 0usize);
        let mut block = [0.0f32; BLOCK];
        for (bi, chunk) in input.chunks_exact(BLOCK).enumerate() {
            corrector.set_transposition(transposition_at(bi));
            block.copy_from_slice(chunk);
            corrector.process_channel(0, &mut block);
            let depth = corrector.channels[0].out_fifo.len();
            min_depth = min_depth.min(depth);
            max_depth = max_depth.max(depth);
        }
        (min_depth, max_depth)
    }

    #[test]
    fn length_is_one_to_one_with_bounded_fifo() {
        // 30 seconds at a non-representable transposition: the FIFO must
        // neither drain nor grow (the drift-free pairing claim) — its depth
        // band must stay clear of empty and clear of capacity throughout.
        let mut corrector = PvCorrector::new(SR, 1);
        let input = sine(700.0, SR as usize * 30, 0.4);
        let (min_depth, max_depth) = depth_band(&mut corrector, &input, |_| 1.02);
        println!("steady T=1.02: fifo depth band [{min_depth}, {max_depth}]");
        assert!(min_depth >= 16, "fifo margin too thin: {min_depth}");
        assert!(
            max_depth + 16 <= OUT_FIFO_CAPACITY,
            "fifo near overflow: {max_depth} of {OUT_FIFO_CAPACITY}"
        );
    }

    #[test]
    fn transposition_rides_stay_bounded() {
        // Sweep T continuously across the full clamp range; the FIFO must
        // stay covered (no underrun) and bounded at every point.
        let mut corrector = PvCorrector::new(SR, 1);
        let input = sine(800.0, SR as usize * 8, 0.4);
        let blocks = input.len() / BLOCK;
        let (min_depth, max_depth) = depth_band(&mut corrector, &input, |bi| {
            let t = bi as f64 / blocks as f64;
            1.0 + 0.3 * (2.0 * std::f64::consts::PI * 0.5 * t * 8.0).sin()
        });
        println!("T ride: fifo depth band [{min_depth}, {max_depth}]");
        assert!(min_depth >= 16, "fifo margin too thin on ride: {min_depth}");
        assert!(
            max_depth + 16 <= OUT_FIFO_CAPACITY,
            "fifo near overflow on ride: {max_depth} of {OUT_FIFO_CAPACITY}"
        );
    }
}

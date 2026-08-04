//! Wide-range Master Tempo corrector: big-FFT identity-locked phase
//! vocoder plus post-resampler, full keylock across the engine's whole
//! tempo range (ROADMAP Stage 11).
//!
//! The varispeed head shifts pitch by the tempo rate; this stage cancels
//! that across the FULL spectrum by transposing at the delay-matched
//! reciprocal (owner listening 2026-08-04: the fully keylocked spectrum
//! beat the free-low-band variant — there is no band split here, the
//! PV's rigid sub-bass region below [`WIDE_SUB_BASS_CUTOFF_HZ`] handles
//! the bass). Mechanism, ported from the Stage-9 small-FFT corrector at
//! the falsification-settled wide constants:
//!
//! - The [`PhaseVocoder`] time-stretches by `T` (identity locking,
//!   streaming, FFT 2048, hop 256 — 87.5% overlap is load-bearing: 75%
//!   causes the −75% tempo click/level blowup) with the phase-gradient
//!   coherence blend held at [`WIDE_COHERENCE_BLEND`] (the shipped taper
//!   zeroes it approaching ratio 2.5 — the confirmed "robotic slowdown"
//!   cause).
//! - A [`StreamingSincResampler`] consumes the stretched audio at step
//!   `T`, restoring 1:1 length and shifting pitch by `T`. `T` spans the
//!   full clamp `[0.5, 4.0]` — exactly the resampler's supported step
//!   range — so there is NO correction fade: keylock holds from tempo
//!   rate 0.25 to 2.0.
//! - Length stability is structural: the PV's synthesis cursor
//!   accumulates `hop·T` fractionally, the resampler consumes exactly
//!   `T` per output, so the two track the same integral and the output
//!   FIFO stays bounded.
//!
//! Phase resets are artifact-only (the Stage-9 online spectral-flux
//! fallback relied on a PV hook deleted at cutover; Halo always attaches
//! artifacts, so the fallback is a deliberate scope cut): each non-beat
//! onset re-locks the mid/high bands once, and the low bands too when
//! the onset is strong and its flux actually lives there — gated on a
//! DELAYED copy of `modulation_hold`, because the graph latches the hold
//! at control time while this stage's audio runs ~[`HOLD_DELAY_BLOCKS`]
//! blocks behind.

use std::sync::Arc;

use crate::core::resample::{SincInterpTable, StreamingSincResampler};
use crate::core::ring_buffer::RingBuffer;
use crate::core::window::WindowType;
use crate::engine::stage::{BLOCK_FRAMES, BlockBuf, Stage, StageCtx};
use crate::engine::stages::delay::FixedDelay;
use crate::engine::stages::keylock::KEYLOCK_TOGGLE_FADE_FRAMES;
use crate::stretch::phase_locking::PhaseLockingMode;
use crate::stretch::phase_vocoder::PhaseVocoder;

/// PV FFT size (falsification-settled; 1024 audibly rotates sub-bass
/// phase, 4096 smears transients and busts the latency contract).
pub const WIDE_FFT: usize = 2048;

/// PV analysis hop: 87.5% overlap. MANDATORY — hop = FFT/4 (75%) causes
/// the −75% tempo +7 LUFS / ~2000 clicks/M blowup on every corpus track.
pub const WIDE_HOP: usize = WIDE_FFT / 8;

/// Rigid identity-locked PV region below this frequency (matches the
/// shipped offline wide path).
pub const WIDE_SUB_BASS_CUTOFF_HZ: f32 = 100.0;

/// Phase-gradient coherence blend held at wide ratios (0.20 vs 0.40 is
/// below metric resolution; 0.40 is the owner's round-3 lean — settled
/// by ear at Stage 11 C4).
pub const WIDE_COHERENCE_BLEND: f64 = 0.40;

/// Transposition clamp: `1/rate` over the engine's tempo range
/// [0.25, 2.0] maps to [0.5, 4.0] — exactly the streaming resampler's
/// supported step range (`STREAM_SINC_MAX_STEP`), zero margin at the top.
pub const WIDE_TRANSPOSITION_MIN: f64 = 0.5;
pub const WIDE_TRANSPOSITION_MAX: f64 = 4.0;

/// Constant stage delay: the PV's full analysis-window fill plus margin
/// for the resampler's availability lag and emission rounding. The lag
/// margin is sized off T = 0.5 — half-span 16 input samples is 34
/// OUTPUT frames of lookahead there, the maximum across the clamp range
/// (T = 4's dilated 76-tap kernel is only ~19 output frames) — with
/// headroom over the per-render bookkeeping trough (the step-anchor
/// snap removes the ramp's ln(T) drift term; what remains is cadence
/// plus lookahead). 2144 frames = 48.6 ms at 44.1 kHz — the wide
/// profile's honest contract, reported via `latency_frames()`, never
/// folded into the keylock chain's 15 ms budget.
pub(crate) const WIDE_KEYLOCK_LATENCY_FRAMES: usize = WIDE_FFT + 96;

/// Rolling analysis-window capacity in frames.
const WINDOW_CAPACITY: usize = WIDE_FFT + 2 * WIDE_HOP;

/// Per-channel corrected-output FIFO: latency prime plus slack for PV
/// emission bursts and resampler ramp swings (~4 hops ≈ 4× the observed
/// jitter).
const OUT_FIFO_CAPACITY: usize = WIDE_KEYLOCK_LATENCY_FRAMES + 4 * WIDE_HOP;

/// Artifact onset strength above which the low bands (<500 Hz) re-lock
/// too; upper bands always re-lock on a transient (Stage-9 policy).
const ONSET_LOW_BAND_RESET_STRENGTH: f32 = 0.45;

/// A low band only re-locks when its per-onset flux is a real fraction
/// of the onset's strongest band.
const LOW_BAND_FLUX_FRACTION: f32 = 0.25;

/// Only retune the PV pair beyond this transposition change.
const TRANSPOSITION_EPSILON: f64 = 1e-4;

/// Per-block log-space transposition slew clamp. An instant full-range
/// sync snap (rate 2.0 → 0.5 = two octaves of transposition) stepped
/// straight into the PV tears its overlap-add seam audibly (~4× the
/// tone slew, measured); clamping T movement to this much natural log
/// per 32-frame block bounds the per-render ratio step while settling a
/// full-range snap in ~30 ms — perceptually instant, and continuous
/// rides never engage the clamp. The resampler's step anchor tracks the
/// slewed T, so stream balance stays exact.
const TRANSPOSITION_SLEW_LN_PER_BLOCK: f64 = 0.05;

/// The graph latches `modulation_hold` at CONTROL time and counts it in
/// ingest blocks; this stage's audio runs its own constant delay behind
/// ingest, so low-band reset gating reads the hold through a shift
/// register of this depth.
const HOLD_DELAY_BLOCKS: usize = WIDE_KEYLOCK_LATENCY_FRAMES.div_ceil(BLOCK_FRAMES);

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

/// Wide-range keylock corrector across all channels.
pub(crate) struct WideKeylockStage {
    channels: Vec<ChannelState>,
    /// Delayed copy of the RAW signal, kept warm for the live keylock
    /// toggle crossfade (aligned to the corrector's constant lag).
    raw_delay: FixedDelay,
    /// Per-channel raw (uncorrected) scratch, delayed to alignment.
    raw: Vec<[f32; BLOCK_FRAMES]>,
    /// Per-channel corrected scratch popped from the FIFOs.
    corrected: Vec<[f32; BLOCK_FRAMES]>,
    /// Transposition currently applied to the PV/resampler pair.
    transposition: f64,
    sample_rate: u32,
    /// Stage-timeline frames ingested (the axis artifact events are
    /// scheduled on).
    ingested: f64,
    /// Newest onset stage-frame already fired (once-only under
    /// re-mapping).
    last_onset_fired: f64,
    /// Shift register of recent `ctx.modulation_hold` bits; low-band
    /// reset gating reads the bit [`HOLD_DELAY_BLOCKS`] blocks back.
    hold_bits: u128,
    /// Smoothed keylock-toggle weight chasing `ctx.keylock`. NaN = snap
    /// to the target on the next block (no fade-in from stale state).
    enable: f32,
    /// Resets fired since construction (observability for tests/QA).
    resets_fired: u64,
    /// Band mask of the most recent reset (test observability).
    #[cfg(test)]
    last_reset_mask: [bool; 4],
}

impl std::fmt::Debug for WideKeylockStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WideKeylockStage")
            .field("channels", &self.channels.len())
            .field("transposition", &self.transposition)
            .finish_non_exhaustive()
    }
}

impl WideKeylockStage {
    pub(crate) fn new(sample_rate: u32, num_channels: usize) -> Self {
        let table = SincInterpTable::new_stream_default();
        let channels = (0..num_channels)
            .map(|_| {
                let mut pv = PhaseVocoder::with_options(
                    WIDE_FFT,
                    WIDE_HOP,
                    1.0,
                    sample_rate,
                    WIDE_SUB_BASS_CUTOFF_HZ,
                    WindowType::Hann,
                    PhaseLockingMode::Identity,
                );
                pv.set_smooth_ratio_updates(true);
                pv.set_wide_ratio_coherence_blend(WIDE_COHERENCE_BLEND);
                pv.reserve_streaming_capacity(WINDOW_CAPACITY, WIDE_TRANSPOSITION_MAX + 0.5);
                let mut out_fifo = RingBuffer::with_capacity(OUT_FIFO_CAPACITY);
                // Prime with the constant delay: the stage then delivers
                // exactly one output frame per input frame from block
                // zero, content offset by WIDE_KEYLOCK_LATENCY_FRAMES.
                for _ in 0..WIDE_KEYLOCK_LATENCY_FRAMES {
                    out_fifo.push(0.0);
                }
                let scratch_capacity = (WINDOW_CAPACITY as f64 * (WIDE_TRANSPOSITION_MAX + 0.5))
                    as usize
                    + 2 * WIDE_FFT;
                ChannelState {
                    pv,
                    window: Vec::with_capacity(WINDOW_CAPACITY),
                    pv_out: Vec::with_capacity(scratch_capacity),
                    resampled: Vec::with_capacity(scratch_capacity),
                    resampler: StreamingSincResampler::new(Arc::clone(&table)),
                    out_fifo,
                }
            })
            .collect();
        let raw_delay = FixedDelay::new(WIDE_KEYLOCK_LATENCY_FRAMES, num_channels);
        debug_assert_eq!(raw_delay.latency_frames(), WIDE_KEYLOCK_LATENCY_FRAMES);
        Self {
            channels,
            raw_delay,
            raw: vec![[0.0; BLOCK_FRAMES]; num_channels],
            corrected: vec![[0.0; BLOCK_FRAMES]; num_channels],
            transposition: 1.0,
            sample_rate,
            ingested: 0.0,
            last_onset_fired: f64::NEG_INFINITY,
            hold_bits: 0,
            enable: f32::NAN,
            resets_fired: 0,
            #[cfg(test)]
            last_reset_mask: [false; 4],
        }
    }

    #[cfg(test)]
    fn resets_fired(&self) -> u64 {
        self.resets_fired
    }

    #[cfg(test)]
    fn fifo_len(&self, ch: usize) -> usize {
        self.channels[ch].out_fifo.len()
    }

    /// Per-block event hook (Stage-9 `begin_block` port): fires
    /// strength/flux-gated per-band phase resets for non-beat onsets
    /// whose mapped position entered the ingested span — each exactly
    /// once even as tempo rides re-map positions.
    fn begin_block(&mut self, ctx: &StageCtx<'_>) {
        self.ingested += BLOCK_FRAMES as f64;
        self.hold_bits = (self.hold_bits << 1) | u128::from(ctx.modulation_hold);
        let delayed_hold = (self.hold_bits >> (HOLD_DELAY_BLOCKS - 1)) & 1 == 1;

        let mut mask = [false; 4];
        let mut fire = false;
        for event in ctx.onsets.iter().filter(|event| !event.beat) {
            if event.stage_frame > self.last_onset_fired && event.stage_frame <= self.ingested {
                mask[2] = true;
                mask[3] = true;
                if event.strength >= ONSET_LOW_BAND_RESET_STRENGTH && !delayed_hold {
                    let peak = event
                        .band_flux
                        .iter()
                        .fold(0.0f32, |a, &b| a.max(b))
                        .max(1e-6);
                    mask[0] |= event.band_flux[0] >= LOW_BAND_FLUX_FRACTION * peak;
                    mask[1] |= event.band_flux[1] >= LOW_BAND_FLUX_FRACTION * peak;
                }
                fire = true;
                self.last_onset_fired = self.last_onset_fired.max(event.stage_frame);
            }
        }

        if fire {
            for ch in &mut self.channels {
                ch.pv.reset_phase_state_bands(mask, self.sample_rate);
            }
            self.resets_fired += 1;
            #[cfg(test)]
            {
                self.last_reset_mask = mask;
            }
        }
    }
}

impl Stage for WideKeylockStage {
    fn process(&mut self, block: &mut BlockBuf, ctx: &StageCtx<'_>) {
        // Delay-matched transposition: cancel the pitch shift embedded in
        // THIS audio, not the control target. The resampler additionally
        // ramps its step across each render, so a rate step lands as a
        // sub-hop glide.
        let target = if ctx.embedded_rate.is_finite() && ctx.embedded_rate > 0.0 {
            (1.0 / ctx.embedded_rate).clamp(WIDE_TRANSPOSITION_MIN, WIDE_TRANSPOSITION_MAX)
        } else {
            1.0
        };
        let slew_step = (target.ln() - self.transposition.ln()).clamp(
            -TRANSPOSITION_SLEW_LN_PER_BLOCK,
            TRANSPOSITION_SLEW_LN_PER_BLOCK,
        );
        let transposition = (self.transposition.ln() + slew_step).exp();
        if (transposition - self.transposition).abs() > TRANSPOSITION_EPSILON {
            self.transposition = transposition;
            for ch in &mut self.channels {
                ch.pv.set_stretch_ratio(transposition);
            }
        }

        self.begin_block(ctx);

        // Channels in order 0..n (FixedDelay's shared-cursor contract).
        for ch in 0..block.channels() {
            self.raw[ch].copy_from_slice(block.channel(ch));
            self.raw_delay.process_channel(ch, &mut self.raw[ch]);

            let state = &mut self.channels[ch];
            debug_assert!(state.window.len() + BLOCK_FRAMES <= WINDOW_CAPACITY);
            state.window.extend_from_slice(block.channel(ch));

            // Render at most ONE hop per block: per-callback FFT work is
            // bounded by construction. Steady state hops once per 8
            // blocks; catch-up after a hiccup drains at one hop per
            // block, 8× the input rate.
            if state.window.len() >= WIDE_FFT {
                let result = state
                    .pv
                    .process_streaming_into(&state.window[..WIDE_FFT], &mut state.pv_out);
                debug_assert!(result.is_ok(), "wide pv render failed: {result:?}");

                let remaining = state.window.len() - WIDE_HOP;
                state.window.copy_within(WIDE_HOP.., 0);
                state.window.truncate(remaining);

                if result.is_ok() && !state.pv_out.is_empty() {
                    // The PV rendered this whole chunk at one uniform
                    // transposition; consume it uniformly too. Ramping
                    // from the previous chunk's step (the resampler's
                    // default zipper guard) drifts the stream balance by
                    // hop/2·ln(T₁/T₀) across a sweep — enough to drain
                    // the latency margin over the 8x wide range.
                    state.resampler.set_step_anchor(self.transposition);
                    let resample = state.resampler.process_into(
                        &state.pv_out,
                        self.transposition,
                        &mut state.resampled,
                    );
                    debug_assert!(resample.is_ok(), "wide resample failed: {resample:?}");
                    if resample.is_ok() {
                        let pushed = state.out_fifo.push_slice(&state.resampled);
                        debug_assert_eq!(
                            pushed,
                            state.resampled.len(),
                            "wide corrector FIFO overflow — jitter bound violated"
                        );
                    }
                }
            }

            // Deliver exactly one block; the latency prime guarantees
            // coverage.
            let popped = state.out_fifo.pop_slice(&mut self.corrected[ch]);
            if popped < BLOCK_FRAMES {
                // Only reachable if the bookkeeping margin is violated
                // (debug asserts above); degrade to silence rather than
                // stale samples.
                self.corrected[ch][popped..].fill(0.0);
                debug_assert!(
                    false,
                    "wide corrector FIFO underrun: {popped} < {BLOCK_FRAMES}"
                );
            }
        }

        // Live keylock toggle: chase the control target per sample so a
        // mid-play switch is a click-free crossfade; weights shared across
        // channels so the image stays stable. No deviation-based fade —
        // full keylock across the whole range is this profile's contract.
        let target = (ctx.keylock.clamp(0.0, 1.0)) as f32;
        if self.enable.is_nan() {
            self.enable = target;
        }
        let step = 1.0 / KEYLOCK_TOGGLE_FADE_FRAMES as f32;
        let mut enable_w = [0.0f32; BLOCK_FRAMES];
        let mut enable = self.enable;
        for w in &mut enable_w {
            enable += (target - enable).clamp(-step, step);
            *w = enable;
        }
        self.enable = enable;

        for ch in 0..block.channels() {
            let out = block.channel_mut(ch);
            for (i, sample) in out.iter_mut().enumerate() {
                let w = enable_w[i];
                *sample = w * self.corrected[ch][i] + (1.0 - w) * self.raw[ch][i];
            }
        }
    }

    fn latency_frames(&self) -> usize {
        debug_assert_eq!(self.raw_delay.latency_frames(), WIDE_KEYLOCK_LATENCY_FRAMES);
        WIDE_KEYLOCK_LATENCY_FRAMES
    }

    fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.pv.reset_streaming_state();
            ch.pv.set_stretch_ratio(1.0);
            ch.window.clear();
            ch.pv_out.clear();
            ch.resampled.clear();
            ch.resampler.reset();
            ch.out_fifo.clear();
            for _ in 0..WIDE_KEYLOCK_LATENCY_FRAMES {
                ch.out_fifo.push(0.0);
            }
        }
        self.raw_delay.reset();
        self.transposition = 1.0;
        self.ingested = 0.0;
        self.last_onset_fired = f64::NEG_INFINITY;
        self.hold_bits = 0;
        self.enable = f32::NAN;
    }

    fn warm_start_settle_frames(&self) -> usize {
        // A full analysis window plus one hop of history so the PV's
        // phase state and the resampler lookahead resume converged.
        WIDE_FFT + WIDE_HOP
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::stage::OnsetEvent;

    const SR: u32 = 44_100;

    fn ctx_at(rate: f64) -> StageCtx<'static> {
        StageCtx {
            embedded_rate: rate,
            embedded_rate_slope: 0.0,
            onsets: &[],
            modulation_hold: false,
            has_artifact: false,
            keylock: 1.0,
        }
    }

    fn sine(freq: f64, len: usize, amp: f32) -> Vec<f32> {
        (0..len)
            .map(|i| amp * (2.0 * std::f64::consts::PI * freq * i as f64 / SR as f64).sin() as f32)
            .collect()
    }

    /// Streams mono input through the stage at a fixed embedded rate.
    fn run(stage: &mut WideKeylockStage, input: &[f32], rate: f64) -> Vec<f32> {
        let mut block = BlockBuf::new(1);
        let ctx = ctx_at(rate);
        let mut out = Vec::with_capacity(input.len());
        for chunk in input.chunks_exact(BLOCK_FRAMES) {
            block.channel_mut(0).copy_from_slice(chunk);
            stage.process(&mut block, &ctx);
            out.extend_from_slice(block.channel(0));
        }
        out
    }

    /// Zero-crossing frequency estimate over a slice.
    fn measure_freq(scan: &[f32]) -> f64 {
        let (mut first, mut last, mut count) = (None, None, 0usize);
        for i in 1..scan.len() {
            let (a, b) = (scan[i - 1] as f64, scan[i] as f64);
            if a <= 0.0 && b > 0.0 {
                let t = (i - 1) as f64 + a / (a - b);
                if first.is_none() {
                    first = Some(t);
                }
                last = Some(t);
                count += 1;
            }
        }
        (count - 1) as f64 * SR as f64 / (last.unwrap() - first.unwrap())
    }

    #[test]
    fn unity_transposition_passes_through_at_latency() {
        let mut stage = WideKeylockStage::new(SR, 1);
        let input = sine(1_000.0, SR as usize * 2, 0.5);
        let out = run(&mut stage, &input, 1.0);

        let latency = stage.latency_frames();
        let scan = SR as usize;
        let mut max_err = 0.0f32;
        for i in scan..scan + 8_192 {
            max_err = max_err.max((out[i + latency] - input[i]).abs());
        }
        assert!(
            max_err < 0.05,
            "unity corrector deviates from delayed input: {max_err}"
        );
    }

    #[test]
    fn transposition_corrects_pitch_across_the_wide_range() {
        // The stage receives already-varispeeded audio: a 600 Hz source
        // arrives pitched to 600·rate; the corrector must return it to
        // ~600 Hz at every rate in the wide clamp — including the edges
        // (rate 0.25 → T = 4.0, rate 2.0 → T = 0.5) where the narrow
        // keylock has long since faded to varispeed.
        for rate in [0.25f64, 0.5, 0.8, 1.25, 2.0] {
            let mut stage = WideKeylockStage::new(SR, 1);
            let shifted = sine(600.0 * rate, SR as usize * 3, 0.5);
            let out = run(&mut stage, &shifted, rate);
            let freq = measure_freq(&out[SR as usize..SR as usize * 2]);
            let cents = 1_200.0 * (freq / 600.0).log2();
            assert!(
                cents.abs() < 15.0,
                "rate {rate}: pitch off by {cents:.1} cents ({freq:.2} Hz)"
            );
        }
    }

    #[test]
    fn full_spectrum_correction_reaches_the_sub_band() {
        // The wide profile keylocks the low band too (owner verdict
        // 2026-08-04): a 60 Hz sub arriving varispeeded to 75 Hz must
        // come back near 60 Hz — the narrow keylock would leave it free.
        let rate = 1.25f64;
        let mut stage = WideKeylockStage::new(SR, 1);
        let shifted = sine(60.0 * rate, SR as usize * 4, 0.5);
        let out = run(&mut stage, &shifted, rate);
        let freq = measure_freq(&out[SR as usize * 2..]);
        let cents = 1_200.0 * (freq / 60.0).log2();
        assert!(
            cents.abs() < 25.0,
            "sub band not corrected: {freq:.2} Hz ({cents:+.1} cents)"
        );
    }

    #[test]
    fn length_is_one_to_one_with_bounded_fifo() {
        // 12 s at a non-representable transposition near unity and at a
        // deep expansion: the FIFO must neither drain nor grow — depth
        // clear of empty and of capacity throughout.
        for rate in [0.98f64, 0.27] {
            let mut stage = WideKeylockStage::new(SR, 1);
            let input = sine(700.0, SR as usize * 12, 0.4);
            let mut block = BlockBuf::new(1);
            let ctx = ctx_at(rate);
            let (mut min_depth, mut max_depth) = (usize::MAX, 0usize);
            for chunk in input.chunks_exact(BLOCK_FRAMES) {
                block.channel_mut(0).copy_from_slice(chunk);
                stage.process(&mut block, &ctx);
                let depth = stage.fifo_len(0);
                min_depth = min_depth.min(depth);
                max_depth = max_depth.max(depth);
            }
            println!("rate {rate}: fifo depth band [{min_depth}, {max_depth}]");
            assert!(
                min_depth >= 8,
                "rate {rate}: fifo margin too thin: {min_depth}"
            );
            assert!(
                max_depth + 8 <= OUT_FIFO_CAPACITY,
                "rate {rate}: fifo near overflow: {max_depth} of {OUT_FIFO_CAPACITY}"
            );
        }
    }

    #[test]
    fn transposition_rides_stay_bounded() {
        // Sweep the embedded rate so T covers the full clamp range; the
        // FIFO must stay covered (no underrun) and bounded at every point.
        let mut stage = WideKeylockStage::new(SR, 1);
        let input = sine(800.0, SR as usize * 8, 0.4);
        let mut block = BlockBuf::new(1);
        let blocks = input.len() / BLOCK_FRAMES;
        let (mut min_depth, mut max_depth) = (usize::MAX, 0usize);
        for (bi, chunk) in input.chunks_exact(BLOCK_FRAMES).enumerate() {
            let t = bi as f64 / blocks as f64;
            // rate sweeps 0.25..2.0 (log-centered), so T sweeps 4.0..0.5.
            let rate = 0.708 * (2.0f64.sqrt() * 2.0).powf((2.0 * std::f64::consts::PI * t).sin());
            let ctx = ctx_at(rate.clamp(0.25, 2.0));
            block.channel_mut(0).copy_from_slice(chunk);
            stage.process(&mut block, &ctx);
            let depth = stage.fifo_len(0);
            min_depth = min_depth.min(depth);
            max_depth = max_depth.max(depth);
        }
        println!("T ride: fifo depth band [{min_depth}, {max_depth}]");
        assert!(min_depth >= 8, "fifo margin too thin on ride: {min_depth}");
        assert!(
            max_depth + 8 <= OUT_FIFO_CAPACITY,
            "fifo near overflow on ride: {max_depth} of {OUT_FIFO_CAPACITY}"
        );
    }

    fn step_block(
        stage: &mut WideKeylockStage,
        block: &mut BlockBuf,
        onsets: &[OnsetEvent],
        hold: bool,
    ) {
        let ctx = StageCtx {
            embedded_rate: 1.0,
            embedded_rate_slope: 0.0,
            onsets,
            modulation_hold: hold,
            has_artifact: true,
            keylock: 1.0,
        };
        block.clear();
        stage.process(block, &ctx);
    }

    #[test]
    fn artifact_onsets_fire_resets_exactly_once_with_flux_gating() {
        let mut stage = WideKeylockStage::new(SR, 1);
        let mut block = BlockBuf::new(1);

        let onset = OnsetEvent {
            stage_frame: 40.0,
            strength: 0.9,
            beat: false,
            band_flux: [1.0; 4],
        };
        // Block 1 ingests frames 0..32: onset at 40 not yet reached.
        step_block(&mut stage, &mut block, &[onset], false);
        assert_eq!(stage.resets_fired(), 0);
        // Block 2 ingests 32..64: fires once, all bands (strong, full flux).
        step_block(&mut stage, &mut block, &[onset], false);
        assert_eq!(stage.resets_fired(), 1);
        assert_eq!(stage.last_reset_mask, [true; 4]);
        // Republished (re-mapped) copies must not refire.
        let remapped = OnsetEvent {
            stage_frame: 39.2,
            ..onset
        };
        step_block(&mut stage, &mut block, &[remapped], false);
        assert_eq!(stage.resets_fired(), 1);

        // Beats never fire; a later strong onset with flux only in the
        // upper bands must not re-lock the low bands.
        let beat = OnsetEvent {
            stage_frame: 100.0,
            strength: 1.0,
            beat: true,
            band_flux: [1.0; 4],
        };
        let hats = OnsetEvent {
            stage_frame: 110.0,
            strength: 0.9,
            beat: false,
            band_flux: [0.0, 0.0, 1.0, 1.0],
        };
        step_block(&mut stage, &mut block, &[beat, hats], false);
        step_block(&mut stage, &mut block, &[beat, hats], false);
        step_block(&mut stage, &mut block, &[beat, hats], false);
        assert_eq!(stage.resets_fired(), 2);
        assert_eq!(stage.last_reset_mask, [false, false, true, true]);

        // Weak onsets never touch the low bands regardless of flux.
        let weak = OnsetEvent {
            stage_frame: stage.ingested + 10.0,
            strength: 0.2,
            beat: false,
            band_flux: [1.0; 4],
        };
        step_block(&mut stage, &mut block, &[weak], false);
        assert_eq!(stage.resets_fired(), 3);
        assert_eq!(stage.last_reset_mask, [false, false, true, true]);
    }

    #[test]
    fn modulation_hold_suppresses_low_band_resets_delayed() {
        // The hold is latched at control time but this stage's audio lags
        // HOLD_DELAY_BLOCKS blocks; an onset arriving that much later must
        // still see the hold and keep its low bands locked.
        let mut stage = WideKeylockStage::new(SR, 1);
        let mut block = BlockBuf::new(1);

        // One held block at the start, then quiet.
        step_block(&mut stage, &mut block, &[], true);
        for _ in 0..HOLD_DELAY_BLOCKS - 2 {
            step_block(&mut stage, &mut block, &[], false);
        }
        // A strong full-flux onset lands exactly where the delayed hold
        // bit surfaces: low bands must stay locked.
        let onset = OnsetEvent {
            stage_frame: stage.ingested + 1.0,
            strength: 0.9,
            beat: false,
            band_flux: [1.0; 4],
        };
        step_block(&mut stage, &mut block, &[onset], false);
        assert_eq!(stage.resets_fired(), 1);
        assert_eq!(stage.last_reset_mask, [false, false, true, true]);

        // Once the delayed hold has drained, the same onset shape
        // re-locks everything.
        for _ in 0..HOLD_DELAY_BLOCKS {
            step_block(&mut stage, &mut block, &[], false);
        }
        let later = OnsetEvent {
            stage_frame: stage.ingested + 1.0,
            strength: 0.9,
            beat: false,
            band_flux: [1.0; 4],
        };
        step_block(&mut stage, &mut block, &[later], false);
        assert_eq!(stage.resets_fired(), 2);
        assert_eq!(stage.last_reset_mask, [true; 4]);
    }

    #[test]
    fn keylock_toggle_is_click_free_and_converges() {
        // Toggle off then back on mid-stream. The output must never step
        // harder than the signal's own slew, and each phase must settle
        // on its mode's pitch (corrected vs varispeed).
        let rate = 1.25f64;
        let secs = SR as usize;
        let shifted = sine(440.0 * rate, secs * 6, 0.6);
        let mut stage = WideKeylockStage::new(SR, 1);

        let mut out = Vec::with_capacity(shifted.len());
        let mut block = BlockBuf::new(1);
        for (bi, chunk) in shifted.chunks_exact(BLOCK_FRAMES).enumerate() {
            let start = bi * BLOCK_FRAMES;
            let keylock = if (secs * 2..secs * 4).contains(&start) {
                0.0
            } else {
                1.0
            };
            let ctx = StageCtx {
                embedded_rate: rate,
                embedded_rate_slope: 0.0,
                onsets: &[],
                modulation_hold: false,
                has_artifact: false,
                keylock,
            };
            block.channel_mut(0).copy_from_slice(chunk);
            stage.process(&mut block, &ctx);
            out.extend_from_slice(block.channel(0));
        }

        let max_step = out
            .windows(2)
            .skip(secs / 2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        let signal_slew = 0.6 * (2.0 * std::f64::consts::PI * 440.0 * rate / SR as f64) as f32;
        assert!(
            max_step < signal_slew * 1.5,
            "toggle clicked: max step {max_step:.4} vs signal slew {signal_slew:.4}"
        );

        let cents = |f: f64, target: f64| 1_200.0 * (f / target).log2();
        let corrected = measure_freq(&out[secs..secs * 2]);
        let bypassed = measure_freq(&out[secs * 3..secs * 4]);
        let recorrected = measure_freq(&out[secs * 5..]);
        assert!(
            cents(corrected, 440.0).abs() < 15.0,
            "keylock phase off: {corrected:.2} Hz"
        );
        assert!(
            cents(bypassed, 440.0 * rate).abs() < 15.0,
            "bypass phase not at varispeed pitch: {bypassed:.2} Hz"
        );
        assert!(
            cents(recorrected, 440.0).abs() < 15.0,
            "re-enabled phase off: {recorrected:.2} Hz"
        );
    }

    #[test]
    fn reset_restores_stream_start() {
        let mut stage = WideKeylockStage::new(SR, 1);
        let input = sine(500.0, SR as usize, 0.4);
        let first = run(&mut stage, &input, 0.7);
        stage.reset();
        let second = run(&mut stage, &input, 0.7);
        let mut max_err = 0.0f32;
        for (a, b) in first.iter().zip(second.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 1e-6,
            "reset is not a full stream restart: {max_err}"
        );
    }
}

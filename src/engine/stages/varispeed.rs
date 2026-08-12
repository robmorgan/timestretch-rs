//! The varispeed head: the engine's tempo axis.
//!
//! Sits at the front of every profile chain and converts source frames into
//! output frames at the current tempo rate by driving a
//! `MultiSincResampler` — all channels in lockstep, one kernel weight row
//! per output frame. Tempo retargets apply to the very next fed chunk with
//! no control glide — the resampler's own intra-chunk step ramp (one
//! 32-frame feed chunk, sub-millisecond) is the only smoothing, inherited
//! from the proven varispeed-first control path.
//!
//! This is deliberately *not* a [`Stage`](crate::engine::stage::Stage):
//! stages are 1:1 fixed-block processors, while the head owns the demand
//! inversion — a ratio-dependent, variable amount of source per output
//! block.

use std::sync::Arc;

use crate::core::resample::{
    MultiSincResampler, STREAM_SINC_HALF_TAPS, STREAM_SINC_MAX_HALF_TAPS, SincInterpTable,
};
use crate::engine::control::MIN_TEMPO_RATE;

/// Source frames fed to the resamplers per feed step. Small so that a tempo
/// retarget's ramp spans well under a millisecond and post-retarget output
/// backlog stays inside the control-to-audio budget.
pub(crate) const FEED_CHUNK_FRAMES: usize = 32;

/// Upper bound on output frames one feed step can emit, used to size
/// scratch buffers at construction.
///
/// Two terms beyond the obvious `chunk / min_rate`: emission is gated by
/// the widest kernel in flight, so when the step drops from a dilated
/// kernel (half-span up to [`STREAM_SINC_MAX_HALF_TAPS`]) back toward
/// unity (half-span [`STREAM_SINC_HALF_TAPS`]), the source frames that
/// margin was holding back are released *in one feed* on top of the chunk
/// itself — at the minimum rate that is
/// `(chunk + max_half_span - min_half_span) / min_rate` outputs (Stage 12
/// robustness finding; previously sized without the kernel-release term
/// and overflowing on a 4.0 -> 0.25 retarget).
pub(crate) const MAX_OUT_PER_FEED: usize = ((FEED_CHUNK_FRAMES + STREAM_SINC_MAX_HALF_TAPS
    - STREAM_SINC_HALF_TAPS) as f64
    / MIN_TEMPO_RATE) as usize
    + 2;

/// Lockstep multi-channel varispeed resampling head.
#[derive(Debug)]
pub(crate) struct VarispeedHead {
    resampler: MultiSincResampler,
    channels: usize,
    /// Channel-contiguous scratch for the deinterleaved feed chunk.
    ch_in: Vec<f32>,
    /// Per-channel emitted output for the current feed step.
    ch_out: Vec<Vec<f32>>,
}

impl VarispeedHead {
    pub(crate) fn new(channels: usize) -> Self {
        let table = SincInterpTable::new_stream_default();
        Self {
            resampler: MultiSincResampler::new(Arc::clone(&table), channels),
            channels,
            ch_in: vec![0.0; channels * FEED_CHUNK_FRAMES],
            ch_out: (0..channels)
                .map(|_| Vec::with_capacity(MAX_OUT_PER_FEED))
                .collect(),
        }
    }

    /// Feeds one interleaved source chunk (at most [`FEED_CHUNK_FRAMES`]
    /// frames) through the per-channel resamplers at `rate` source frames
    /// per output frame. Returns the number of output frames now available
    /// in the per-channel buffers (identical across channels because every
    /// channel sees the same step sequence).
    #[cfg(test)]
    pub(crate) fn feed(&mut self, interleaved: &[f32], rate: f64) -> usize {
        self.feed_capped(interleaved, rate, usize::MAX)
    }

    /// [`feed`](Self::feed) with an emission cap (see
    /// `StreamingSincResampler::process_into_capped`): lands timestamped
    /// retargets on an exact emitted frame. Outputs the cap withholds emit
    /// on the next feed, at that feed's rate.
    pub(crate) fn feed_capped(&mut self, interleaved: &[f32], rate: f64, max_out: usize) -> usize {
        let frames = interleaved.len() / self.channels;
        debug_assert_eq!(interleaved.len() % self.channels, 0);
        debug_assert!(frames <= FEED_CHUNK_FRAMES);
        if frames == 0 {
            for out in &mut self.ch_out {
                out.clear();
            }
            return 0;
        }

        for ch in 0..self.channels {
            for f in 0..frames {
                self.ch_in[ch * frames + f] = interleaved[f * self.channels + ch];
            }
        }
        // Capacity is construction-guaranteed for the clamped rate
        // range, so overflow is unreachable; debug builds assert it.
        let result = self.resampler.process_flat_capped(
            &self.ch_in[..self.channels * frames],
            frames,
            rate,
            &mut self.ch_out,
            max_out,
        );
        debug_assert!(result.is_ok(), "varispeed scratch overflow: {result:?}");

        let produced = self.ch_out[0].len();
        debug_assert!(self.ch_out.iter().all(|out| out.len() == produced));
        produced
    }

    /// One channel's output from the most recent [`feed`](Self::feed).
    #[inline]
    pub(crate) fn output(&self, ch: usize) -> &[f32] {
        &self.ch_out[ch]
    }

    /// Fractional source position (frames since reset) of the next output
    /// frame the head will emit.
    pub(crate) fn source_pos(&self) -> f64 {
        self.resampler.next_output_source_pos()
    }

    /// Kernel lookahead of the resampler at unity/pitch-down, in frames.
    pub(crate) fn lookahead_frames(&self) -> usize {
        STREAM_SINC_HALF_TAPS
    }

    /// Clears all resampler state back to stream start.
    pub(crate) fn reset(&mut self) {
        self.resampler.reset();
        for out in &mut self.ch_out {
            out.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unity_rate_is_sample_aligned_passthrough() {
        let mut head = VarispeedHead::new(2);
        let mut input = Vec::new();
        for f in 0..FEED_CHUNK_FRAMES * 4 {
            input.push(f as f32);
            input.push(-(f as f32));
        }

        let mut left = Vec::new();
        let mut right = Vec::new();
        for chunk in input.chunks(FEED_CHUNK_FRAMES * 2) {
            let produced = head.feed(chunk, 1.0);
            left.extend_from_slice(&head.output(0)[..produced]);
            right.extend_from_slice(&head.output(1)[..produced]);
        }

        // Zero pipeline delay: output frame i IS source frame i, with the
        // final `lookahead` frames withheld until more source arrives.
        assert_eq!(left.len(), FEED_CHUNK_FRAMES * 4 - head.lookahead_frames());
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!((l - i as f32).abs() < 1e-3, "left[{i}] = {l}");
            assert!((r + i as f32).abs() < 1e-3, "right[{i}] = {r}");
        }
    }

    #[test]
    fn channels_emit_identical_counts_across_retargets() {
        let mut head = VarispeedHead::new(2);
        let chunk = vec![0.25f32; FEED_CHUNK_FRAMES * 2];
        for step in [1.0, 1.3, 0.8, 4.0, 0.25, 1.0] {
            let produced = head.feed(&chunk, step);
            assert_eq!(head.output(0).len(), produced);
            assert_eq!(head.output(1).len(), produced);
            assert!(produced <= MAX_OUT_PER_FEED);
        }
    }

    #[test]
    fn regression_step_drop_kernel_release_stays_within_capacity() {
        // Found by the Stage 12 robustness harness
        // (qa/robustness.rs::engine_process_with_extreme_configs_stays_finite):
        // sustained fast rates dilate the kernel (half-span up to 80); a
        // retarget down to the minimum rate then narrows it back to 16,
        // releasing the withheld source frames in one feed. With
        // MAX_OUT_PER_FEED sized only as chunk/min_rate + 2 (130), the
        // release emitted 131+ frames: a BufferOverflow debug panic and,
        // in release, silently truncated emission. Minimized: feed at the
        // max rate until the wide kernel is in steady state, then snap to
        // the min rate.
        let mut head = VarispeedHead::new(1);
        let chunk = vec![0.5f32; FEED_CHUNK_FRAMES];
        for _ in 0..16 {
            let produced = head.feed(&chunk, 4.0);
            assert!(produced <= MAX_OUT_PER_FEED);
        }
        let mut max_produced = 0usize;
        for _ in 0..16 {
            let produced = head.feed(&chunk, 0.25);
            max_produced = max_produced.max(produced);
            assert!(produced <= MAX_OUT_PER_FEED, "{produced}");
        }
        // The release burst really happens (this is what overflowed): the
        // first min-rate feeds emit well beyond the naive 130-frame bound.
        assert!(
            max_produced > FEED_CHUNK_FRAMES * 4 + 2,
            "expected a kernel-release burst, got max {max_produced}"
        );
    }

    #[test]
    fn source_pos_tracks_rate_integral() {
        let mut head = VarispeedHead::new(1);
        let chunk = vec![0.5f32; FEED_CHUNK_FRAMES];
        let mut emitted = 0usize;
        for _ in 0..64 {
            emitted += head.feed(&chunk, 1.25);
        }
        // Constant rate from stream start: cursor == emitted * rate.
        assert!((head.source_pos() - emitted as f64 * 1.25).abs() < 1e-6);
    }
}

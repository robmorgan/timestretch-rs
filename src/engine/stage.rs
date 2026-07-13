//! The uniform DSP-stage trait and the fixed-size planar block it processes.

/// Frames per internal fixed block.
///
/// Every [`Stage`] processes exactly this many frames per call. The engine's
/// block scheduler adapts arbitrary caller callback sizes (64–1024 frames) to
/// this granularity, so control changes and stage work are bounded per block.
/// Kept small so the post-retarget output backlog stays well inside the
/// control-to-audio budget (resampler lookahead + one caller block).
pub const BLOCK_FRAMES: usize = 32;

/// A fixed-size planar audio block: `channels` runs of [`BLOCK_FRAMES`]
/// samples each, stored contiguously channel-after-channel.
///
/// Planar layout lets per-channel DSP (resamplers, vocoders) view each
/// channel as a plain slice without per-sample stride arithmetic.
#[derive(Debug, Clone)]
pub struct BlockBuf {
    data: Vec<f32>,
    channels: usize,
}

impl BlockBuf {
    /// Allocates a zeroed block for `channels` channels.
    pub fn new(channels: usize) -> Self {
        assert!(channels > 0, "BlockBuf requires at least one channel");
        Self {
            data: vec![0.0; channels * BLOCK_FRAMES],
            channels,
        }
    }

    /// Number of channels in the block.
    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Immutable view of one channel's [`BLOCK_FRAMES`] samples.
    #[inline]
    pub fn channel(&self, ch: usize) -> &[f32] {
        debug_assert!(ch < self.channels);
        &self.data[ch * BLOCK_FRAMES..(ch + 1) * BLOCK_FRAMES]
    }

    /// Mutable view of one channel's [`BLOCK_FRAMES`] samples.
    #[inline]
    pub fn channel_mut(&mut self, ch: usize) -> &mut [f32] {
        debug_assert!(ch < self.channels);
        &mut self.data[ch * BLOCK_FRAMES..(ch + 1) * BLOCK_FRAMES]
    }

    /// Fills the whole block with silence.
    #[inline]
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    /// Deinterleaves `frames` frames from `input` into the front of the
    /// block, zero-filling the remainder. `input` must hold at least
    /// `frames * channels` samples and `frames <= BLOCK_FRAMES`.
    pub fn fill_deinterleaved(&mut self, input: &[f32], frames: usize) {
        debug_assert!(frames <= BLOCK_FRAMES);
        debug_assert!(input.len() >= frames * self.channels);
        let channels = self.channels;
        for ch in 0..channels {
            let dst = &mut self.data[ch * BLOCK_FRAMES..(ch + 1) * BLOCK_FRAMES];
            for (f, sample) in dst.iter_mut().enumerate().take(frames) {
                *sample = input[f * channels + ch];
            }
            dst[frames..].fill(0.0);
        }
    }

    /// Interleaves the first `frames` frames of the block into `out`, which
    /// must hold at least `frames * channels` samples.
    pub fn write_interleaved(&self, out: &mut [f32], frames: usize) {
        debug_assert!(frames <= BLOCK_FRAMES);
        debug_assert!(out.len() >= frames * self.channels);
        let channels = self.channels;
        for ch in 0..channels {
            let src = &self.data[ch * BLOCK_FRAMES..(ch + 1) * BLOCK_FRAMES];
            for (f, &sample) in src.iter().enumerate().take(frames) {
                out[f * channels + ch] = sample;
            }
        }
    }
}

/// One artifact event (onset or beat) mapped onto the stage timeline.
#[derive(Debug, Clone, Copy, Default)]
pub struct OnsetEvent {
    /// Absolute stage-timeline frame (the axis stage input blocks advance
    /// on) where the event's audio sits — exact under tempo rides because
    /// it is re-mapped through the varispeed timeline every block.
    pub stage_frame: f64,
    /// Artifact onset strength (beats publish 1.0).
    pub strength: f32,
    /// True for beatgrid positions, false for detected onsets.
    pub beat: bool,
}

/// Per-block context the graph hands every stage.
///
/// Carries the control-plane signals a stage may need, computed once per
/// block by the graph.
#[derive(Debug, Clone, Copy)]
pub struct StageCtx<'a> {
    /// Tempo rate embedded in the audio at the END of this block, read off
    /// the varispeed timeline map (the old engine's delay-matched
    /// transposition mechanism). The audio's pitch is scaled by this rate;
    /// a keylock corrector cancels it by transposing at its reciprocal.
    pub embedded_rate: f64,
    /// Artifact events near this block (behind by up to ~1k frames, ahead
    /// by up to ~2k), in stage-timeline coordinates. Empty when no
    /// artifact is attached — stages fall back to online heuristics.
    pub onsets: &'a [OnsetEvent],
    /// True while a fast control gesture is in flight: stages suppress
    /// disruptive maintenance (low-band phase resets, discretionary
    /// splices) until the ride settles — the graph-level re-expression of
    /// the old engine's modulation-hold latches.
    pub modulation_hold: bool,
    /// Whether a pre-analysis artifact is attached at all. Distinguishes
    /// "no events nearby" from "no artifact" so stages know when to run
    /// their online-detection fallbacks.
    pub has_artifact: bool,
}

/// One DSP stage in the engine graph.
///
/// Stages are small, single-purpose processors composed into fixed
/// per-profile chains. The contract is real-time-first:
///
/// - `process` transforms exactly one [`BlockBuf`] in place, never
///   allocates, never fails, and has construction-bounded worst-case cost.
/// - Invariants are enforced when the stage is built; hot-path validation is
///   debug-only.
/// - `latency_frames` reports the stage's constant pipeline delay so the
///   graph can account (and Stage 2's low-band delay can match) exactly.
/// - `prime` is reserved for warm-start seek (Stage 5): run history through
///   the stage, discarding output, so a jump resumes converged.
pub trait Stage: Send {
    /// Processes one fixed block in place.
    fn process(&mut self, block: &mut BlockBuf, ctx: &StageCtx<'_>);

    /// Constant pipeline delay this stage introduces, in frames.
    fn latency_frames(&self) -> usize;

    /// Clears all internal state back to stream start.
    fn reset(&mut self);

    /// Warm-starts internal state from interleaved history samples
    /// (most recent last). Default: cold reset.
    fn prime(&mut self, history: &[f32]) {
        let _ = history;
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_roundtrip_interleave() {
        let mut block = BlockBuf::new(2);
        let input: Vec<f32> = (0..BLOCK_FRAMES * 2).map(|i| i as f32).collect();
        block.fill_deinterleaved(&input, BLOCK_FRAMES);
        assert_eq!(block.channel(0)[0], 0.0);
        assert_eq!(block.channel(1)[0], 1.0);
        assert_eq!(block.channel(0)[1], 2.0);

        let mut out = vec![0.0; BLOCK_FRAMES * 2];
        block.write_interleaved(&mut out, BLOCK_FRAMES);
        assert_eq!(out, input);
    }

    #[test]
    fn partial_fill_zero_pads() {
        let mut block = BlockBuf::new(1);
        block.fill_deinterleaved(&[1.0; 8], 8);
        assert_eq!(&block.channel(0)[..8], &[1.0; 8]);
        assert_eq!(&block.channel(0)[8..], &[0.0; BLOCK_FRAMES - 8]);
    }
}

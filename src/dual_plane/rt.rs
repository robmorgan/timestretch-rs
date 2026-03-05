//! Hard-RT audio plane.
//!
//! The callback-facing API is intentionally small:
//! - [`RtProcessor::prepare`]
//! - [`RtProcessor::process_block`]
//! - [`RtProcessor::flush`]

use crate::core::ring_buffer::RingBuffer;
use crate::core::types::StretchParams;
use crate::dual_plane::hints::RenderHints;
use crate::dual_plane::quality::{LatencyProfile, QualityGovernor, QualityTier, RtGovernorConfig};
use crate::dual_plane::warp_map::TimeWarpMap;
use crate::error::StretchError;
use crate::stretch::phase_vocoder::PhaseVocoder;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::time::Instant;

const CONTROL_QUEUE_CAPACITY: usize = 8;
const RESIDUAL_HPF_ALPHA: f32 = 0.08;
const RATIO_SNAP_EPS: f64 = 1e-6;

/// Configuration for the hard-RT processor.
#[derive(Debug, Clone)]
pub struct RtConfig {
    pub params: StretchParams,
    /// Callback block size in frames.
    pub block_frames: usize,
    /// Fixed analysis window processed per callback pass.
    pub kernel_frames: usize,
    pub latency_profile: LatencyProfile,
    /// Input-ring depth in callback blocks.
    pub input_ring_blocks: usize,
    /// Output-ring depth in callback blocks.
    pub output_ring_blocks: usize,
    /// Lower clamp for warp slope.
    pub min_ratio: f64,
    /// Upper clamp for warp slope.
    pub max_ratio: f64,
    pub governor: RtGovernorConfig,
}

impl RtConfig {
    pub fn new(params: StretchParams, block_frames: usize) -> Self {
        let kernel_frames = (params.fft_size * 2).max(block_frames);
        Self {
            params,
            block_frames,
            kernel_frames,
            latency_profile: LatencyProfile::Mix,
            input_ring_blocks: 24,
            output_ring_blocks: 24,
            min_ratio: 0.25,
            max_ratio: 4.0,
            governor: RtGovernorConfig::default(),
        }
    }

    #[inline]
    fn input_capacity_frames(&self) -> usize {
        self.block_frames
            .saturating_mul(self.input_ring_blocks)
            .saturating_add(self.kernel_frames)
            .saturating_add(self.params.fft_size)
    }

    #[inline]
    fn output_capacity_frames(&self) -> usize {
        self.block_frames
            .saturating_mul(self.output_ring_blocks)
            .saturating_add(self.max_output_frames_per_kernel())
    }

    #[inline]
    fn max_output_frames_per_kernel(&self) -> usize {
        ((self.kernel_frames as f64 * self.max_ratio).ceil() as usize)
            .saturating_add(self.params.fft_size)
    }

    fn validate(&self) -> Result<(), StretchError> {
        if self.block_frames == 0 {
            return Err(StretchError::InvalidFormat(
                "block_frames must be > 0".to_string(),
            ));
        }
        if self.kernel_frames < self.params.fft_size {
            return Err(StretchError::InvalidFormat(format!(
                "kernel_frames {} must be >= fft_size {}",
                self.kernel_frames, self.params.fft_size
            )));
        }
        if self.params.hop_size == 0 {
            return Err(StretchError::InvalidFormat(
                "hop_size must be > 0".to_string(),
            ));
        }
        if self.input_ring_blocks == 0 || self.output_ring_blocks == 0 {
            return Err(StretchError::InvalidFormat(
                "ring block counts must be > 0".to_string(),
            ));
        }
        if !self.min_ratio.is_finite()
            || !self.max_ratio.is_finite()
            || self.min_ratio <= 0.0
            || self.max_ratio <= self.min_ratio
        {
            return Err(StretchError::InvalidRatio(format!(
                "invalid ratio clamps: min={} max={}",
                self.min_ratio, self.max_ratio
            )));
        }
        if self.input_capacity_frames() < self.kernel_frames {
            return Err(StretchError::InvalidFormat(
                "input capacity contract undersized".to_string(),
            ));
        }
        if self.output_capacity_frames() < self.max_output_frames_per_kernel() {
            return Err(StretchError::InvalidFormat(
                "output capacity contract undersized".to_string(),
            ));
        }
        Ok(())
    }
}

/// Sender handle for publishing control-plane snapshots without blocking RT.
#[derive(Clone)]
pub struct RtControlSender {
    warp_tx: SyncSender<Arc<TimeWarpMap>>,
    hints_tx: SyncSender<Arc<RenderHints>>,
}

impl RtControlSender {
    pub fn publish_warp_map(&self, map: Arc<TimeWarpMap>) -> bool {
        match self.warp_tx.try_send(map) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => false,
            Err(TrySendError::Disconnected(_)) => false,
        }
    }

    pub fn publish_hints(&self, hints: Arc<RenderHints>) -> bool {
        match self.hints_tx.try_send(hints) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => false,
            Err(TrySendError::Disconnected(_)) => false,
        }
    }
}

/// Hard-RT processing plane.
pub struct RtProcessor {
    config: RtConfig,
    num_channels: usize,
    block_samples: usize,
    kernel_samples: usize,
    input_ring: RingBuffer<f32>,
    pending_output: RingBuffer<f32>,
    interleaved_scratch: Vec<f32>,
    channel_input: Vec<Vec<f32>>,
    tonal_output: Vec<Vec<f32>>,
    transient_output: Vec<Vec<f32>>,
    residual_output: Vec<Vec<f32>>,
    residual_source: Vec<Vec<f32>>,
    residual_lp_state: Vec<f32>,
    vocoders: Vec<PhaseVocoder>,
    warp_map: Arc<TimeWarpMap>,
    hints: Arc<RenderHints>,
    control: RtControlSender,
    warp_rx: Receiver<Arc<TimeWarpMap>>,
    hints_rx: Receiver<Arc<RenderHints>>,
    governor: QualityGovernor,
    current_tier: QualityTier,
    target_tier: QualityTier,
    blend_weights: [f32; 3],
    target_weights: [f32; 3],
    crossfade_blocks_left: usize,
    input_timeline_frames: f64,
    active_ratio: f64,
}

impl std::fmt::Debug for RtProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RtProcessor")
            .field("block_frames", &self.config.block_frames)
            .field("kernel_frames", &self.config.kernel_frames)
            .field("num_channels", &self.num_channels)
            .field("current_tier", &self.current_tier)
            .field("target_tier", &self.target_tier)
            .field("active_ratio", &self.active_ratio)
            .field("input_ring_len", &self.input_ring.len())
            .field("pending_output_len", &self.pending_output.len())
            .finish()
    }
}

impl RtProcessor {
    /// Prepares the RT plane and allocates all fixed-capacity state.
    pub fn prepare(config: RtConfig) -> Result<Self, StretchError> {
        config.validate()?;
        let num_channels = config.params.channels.count().max(1);
        let block_samples = config.block_frames.saturating_mul(num_channels);
        let kernel_samples = config.kernel_frames.saturating_mul(num_channels);

        let input_capacity_samples = config.input_capacity_frames().saturating_mul(num_channels);
        let output_capacity_samples = config.output_capacity_frames().saturating_mul(num_channels);
        let max_output_frames = config.max_output_frames_per_kernel();

        let initial_tier = config.latency_profile.initial_tier();
        let governor = QualityGovernor::new(initial_tier, config.governor);

        let mut me = Self::new_inner(
            config,
            num_channels,
            block_samples,
            kernel_samples,
            input_capacity_samples,
            output_capacity_samples,
            max_output_frames,
            governor,
            initial_tier,
        );
        me.prewarm_vocoders()?;
        Ok(me)
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        config: RtConfig,
        num_channels: usize,
        block_samples: usize,
        kernel_samples: usize,
        input_capacity_samples: usize,
        output_capacity_samples: usize,
        max_output_frames: usize,
        governor: QualityGovernor,
        initial_tier: QualityTier,
    ) -> Self {
        let mut vocoders = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            let mut pv = PhaseVocoder::with_all_options(
                config.params.fft_size,
                config.params.hop_size,
                config.params.stretch_ratio,
                config.params.sample_rate,
                config.params.sub_bass_cutoff,
                config.params.window_type,
                config.params.phase_locking_mode,
                config.params.envelope_preservation,
                config.params.envelope_order,
            );
            pv.set_adaptive_phase_locking(config.params.adaptive_phase_locking);
            pv.set_envelope_strength(config.params.envelope_strength);
            pv.set_adaptive_envelope_order(config.params.adaptive_envelope_order);
            vocoders.push(pv);
        }

        let (warp_tx, warp_rx) = sync_channel(CONTROL_QUEUE_CAPACITY);
        let (hints_tx, hints_rx) = sync_channel(CONTROL_QUEUE_CAPACITY);
        let control = RtControlSender { warp_tx, hints_tx };
        let warp_map = Arc::new(
            TimeWarpMap::from_ratio(config.params.stretch_ratio, config.kernel_frames)
                .unwrap_or_default(),
        );
        let hints = Arc::new(RenderHints::default());
        let active_ratio = config.params.stretch_ratio;

        let blend = initial_tier.lane_weights();
        Self {
            config,
            num_channels,
            block_samples,
            kernel_samples,
            input_ring: RingBuffer::with_capacity(input_capacity_samples),
            pending_output: RingBuffer::with_capacity(output_capacity_samples),
            interleaved_scratch: vec![0.0; kernel_samples],
            channel_input: (0..num_channels)
                .map(|_| Vec::with_capacity(kernel_samples / num_channels))
                .collect(),
            tonal_output: (0..num_channels)
                .map(|_| Vec::with_capacity(max_output_frames))
                .collect(),
            transient_output: (0..num_channels)
                .map(|_| Vec::with_capacity(max_output_frames))
                .collect(),
            residual_output: (0..num_channels)
                .map(|_| Vec::with_capacity(max_output_frames))
                .collect(),
            residual_source: (0..num_channels)
                .map(|_| Vec::with_capacity(kernel_samples / num_channels))
                .collect(),
            residual_lp_state: vec![0.0; num_channels],
            vocoders,
            warp_map,
            hints,
            control,
            warp_rx,
            hints_rx,
            governor,
            current_tier: initial_tier,
            target_tier: initial_tier,
            blend_weights: blend,
            target_weights: blend,
            crossfade_blocks_left: 0,
            input_timeline_frames: 0.0,
            active_ratio,
        }
    }

    /// Returns a cloneable control sender used by non-RT producers.
    #[inline]
    pub fn control_sender(&self) -> RtControlSender {
        self.control.clone()
    }

    /// Sets warp map directly on this thread.
    #[inline]
    pub fn set_warp_map_snapshot(&mut self, warp_map: Arc<TimeWarpMap>) {
        self.warp_map = warp_map;
    }

    /// Sets hint snapshot directly on this thread.
    #[inline]
    pub fn set_hint_snapshot(&mut self, hints: Arc<RenderHints>) {
        self.hints = hints;
    }

    /// Current active quality tier.
    #[inline]
    pub fn quality_tier(&self) -> QualityTier {
        self.current_tier
    }

    /// Current warp ratio consumed by kernels.
    #[inline]
    pub fn active_ratio(&self) -> f64 {
        self.active_ratio
    }

    #[inline]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// RT-core processing API.
    ///
    /// `input_slices` and `output_slices` are per-channel planar buffers.
    /// Returns `(consumed_frames, produced_frames)`.
    ///
    /// This API is intentionally non-fallible for callback integration.
    /// On invalid arguments or internal failure it returns `(0, 0)`.
    pub fn process(
        &mut self,
        input_slices: &[&[f32]],
        output_slices: &mut [&mut [f32]],
    ) -> (usize, usize) {
        self.process_checked(input_slices, output_slices)
            .unwrap_or((0, 0))
    }

    /// Fallible variant of [`RtProcessor::process`].
    pub fn process_checked(
        &mut self,
        input_slices: &[&[f32]],
        output_slices: &mut [&mut [f32]],
    ) -> Result<(usize, usize), StretchError> {
        let start = Instant::now();
        self.poll_control_updates();
        self.advance_tier_crossfade();

        if input_slices.len() != self.num_channels || output_slices.len() != self.num_channels {
            return Err(StretchError::InvalidFormat(format!(
                "process expects {} input and {} output channel slices, got {} and {}",
                self.num_channels,
                self.num_channels,
                input_slices.len(),
                output_slices.len()
            )));
        }

        let input_frames = input_slices.first().map_or(0, |ch| ch.len());
        if input_frames > self.config.block_frames {
            return Err(StretchError::InvalidFormat(format!(
                "process input frame count {} exceeds configured block_frames {}",
                input_frames, self.config.block_frames
            )));
        }

        for (ch, slice) in input_slices.iter().enumerate() {
            if slice.len() != input_frames {
                return Err(StretchError::InvalidFormat(format!(
                    "channel {} input length {} does not match channel 0 length {}",
                    ch,
                    slice.len(),
                    input_frames
                )));
            }
            if slice.iter().any(|s| !s.is_finite()) {
                return Err(StretchError::NonFiniteInput);
            }
        }

        let output_frames_capacity = output_slices
            .iter()
            .map(|slice| slice.len())
            .min()
            .unwrap_or(0);

        if input_frames > 0 {
            let needed_samples = input_frames.saturating_mul(self.num_channels);
            if needed_samples > self.interleaved_scratch.len() {
                return Err(StretchError::BufferOverflow {
                    buffer: "rt_interleaved_input_scratch",
                    requested: needed_samples,
                    available: self.interleaved_scratch.len(),
                });
            }

            for frame in 0..input_frames {
                for ch in 0..self.num_channels {
                    self.interleaved_scratch[frame * self.num_channels + ch] =
                        input_slices[ch][frame];
                }
            }

            self.push_input_from_scratch_with_overload_policy(needed_samples)?;

            // Fixed-cost callback kernel: render at most one kernel per call.
            if self.input_ring.len() >= self.kernel_samples {
                self.render_fixed_kernel()?;
            }
        }

        let produced_frames =
            self.drain_pending_to_slices(output_slices, output_frames_capacity)?;

        let tier = self.governor.observe_block(start.elapsed());
        self.set_target_tier(tier);
        Ok((input_frames, produced_frames))
    }

    /// Processes one callback block.
    pub fn process_block(
        &mut self,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<(), StretchError> {
        let start = Instant::now();
        self.poll_control_updates();
        self.advance_tier_crossfade();

        if input.len() != self.block_samples {
            return Err(StretchError::InvalidFormat(format!(
                "process_block requires exactly {} samples ({} frames x {} channels), got {}",
                self.block_samples,
                self.config.block_frames,
                self.num_channels,
                input.len()
            )));
        }
        if input.iter().any(|s| !s.is_finite()) {
            return Err(StretchError::NonFiniteInput);
        }

        self.push_input_with_overload_policy(input)?;
        if self.input_ring.len() >= self.kernel_samples {
            self.render_fixed_kernel()?;
        }
        let max_emit = self.max_output_samples_per_callback();
        let _ = self.drain_pending_to_output(output, max_emit)?;

        let tier = self.governor.observe_block(start.elapsed());
        self.set_target_tier(tier);
        Ok(())
    }

    /// Flushes all pending RT state.
    pub fn flush(&mut self, output: &mut Vec<f32>) -> Result<(), StretchError> {
        self.poll_control_updates();

        let fft_samples = self
            .config
            .params
            .fft_size
            .saturating_mul(self.num_channels);
        while self.input_ring.len() >= self.kernel_samples {
            self.render_fixed_kernel()?;
        }

        if self.input_ring.len() >= fft_samples && self.input_ring.len() < self.kernel_samples {
            let need = self.kernel_samples.saturating_sub(self.input_ring.len());
            self.push_zeros(need)?;
            self.render_fixed_kernel()?;
        }

        self.flush_tonal_tails_to_pending()?;
        let _ = self.drain_pending_to_output(output, usize::MAX)?;

        self.input_ring.clear();
        self.pending_output.clear();
        self.input_timeline_frames = 0.0;
        self.residual_lp_state.fill(0.0);
        for vocoder in &mut self.vocoders {
            vocoder.reset_phase_state();
        }

        Ok(())
    }

    fn prewarm_vocoders(&mut self) -> Result<(), StretchError> {
        let zero_kernel = vec![0.0f32; self.config.kernel_frames];
        let max_output = self.config.max_output_frames_per_kernel();
        for ch in 0..self.num_channels {
            if self.tonal_output[ch].capacity() < max_output {
                return Err(StretchError::BufferOverflow {
                    buffer: "rt_tonal_output_capacity",
                    requested: max_output,
                    available: self.tonal_output[ch].capacity(),
                });
            }
            self.tonal_output[ch].clear();
            self.vocoders[ch].process_streaming_into(&zero_kernel, &mut self.tonal_output[ch])?;
            self.tonal_output[ch].clear();
            self.vocoders[ch].flush_streaming_into(&mut self.tonal_output[ch])?;
            self.tonal_output[ch].clear();
            self.vocoders[ch].reset_phase_state();
        }
        Ok(())
    }

    fn poll_control_updates(&mut self) {
        loop {
            match self.warp_rx.try_recv() {
                Ok(map) => self.warp_map = map,
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
        loop {
            match self.hints_rx.try_recv() {
                Ok(hints) => self.hints = hints,
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
    }

    #[inline]
    fn max_output_samples_per_callback(&self) -> usize {
        self.config
            .max_output_frames_per_kernel()
            .max((self.config.block_frames as f64 * self.config.max_ratio).ceil() as usize)
            .saturating_mul(self.num_channels)
    }

    fn set_target_tier(&mut self, tier: QualityTier) {
        if tier == self.target_tier {
            return;
        }
        self.target_tier = tier;
        self.target_weights = tier.lane_weights();
        self.crossfade_blocks_left = self.config.latency_profile.tier_crossfade_blocks();
    }

    fn advance_tier_crossfade(&mut self) {
        if self.crossfade_blocks_left == 0 {
            self.blend_weights = self.target_weights;
            self.current_tier = self.target_tier;
            return;
        }
        let denom = self.crossfade_blocks_left as f32;
        for i in 0..3 {
            self.blend_weights[i] += (self.target_weights[i] - self.blend_weights[i]) / denom;
        }
        self.crossfade_blocks_left = self.crossfade_blocks_left.saturating_sub(1);
        if self.crossfade_blocks_left == 0 {
            self.blend_weights = self.target_weights;
            self.current_tier = self.target_tier;
        }
    }

    fn force_tier_demote(&mut self) {
        let next = self.governor.force_demote_once();
        self.set_target_tier(next);
    }

    fn push_input_with_overload_policy(&mut self, input: &[f32]) -> Result<(), StretchError> {
        let overflow = input.len().saturating_sub(self.input_ring.available());
        if overflow > 0 {
            self.input_ring.discard(overflow);
            self.force_tier_demote();
        }
        let pushed = self.input_ring.push_slice(input);
        if pushed != input.len() {
            return Err(StretchError::BufferOverflow {
                buffer: "rt_input_ring",
                requested: input.len(),
                available: pushed,
            });
        }
        Ok(())
    }

    fn push_input_from_scratch_with_overload_policy(
        &mut self,
        samples: usize,
    ) -> Result<(), StretchError> {
        if samples > self.interleaved_scratch.len() {
            return Err(StretchError::BufferOverflow {
                buffer: "rt_interleaved_input_scratch",
                requested: samples,
                available: self.interleaved_scratch.len(),
            });
        }
        let overflow = samples.saturating_sub(self.input_ring.available());
        if overflow > 0 {
            self.input_ring.discard(overflow);
            self.force_tier_demote();
        }

        let pushed = {
            let input_ring = &mut self.input_ring;
            let scratch = &self.interleaved_scratch;
            input_ring.push_slice(&scratch[..samples])
        };
        if pushed != samples {
            return Err(StretchError::BufferOverflow {
                buffer: "rt_input_ring",
                requested: samples,
                available: pushed,
            });
        }
        Ok(())
    }

    fn render_fixed_kernel(&mut self) -> Result<(), StretchError> {
        let copied = self
            .input_ring
            .peek_slice(&mut self.interleaved_scratch[..self.kernel_samples]);
        if copied != self.kernel_samples {
            return Err(StretchError::InvalidState(
                "failed to snapshot full RT kernel input",
            ));
        }

        let frames = self.config.kernel_frames;
        for ch in 0..self.num_channels {
            if self.channel_input[ch].capacity() < frames {
                return Err(StretchError::BufferOverflow {
                    buffer: "rt_channel_input",
                    requested: frames,
                    available: self.channel_input[ch].capacity(),
                });
            }
            self.channel_input[ch].clear();
            for frame in 0..frames {
                self.channel_input[ch]
                    .push(self.interleaved_scratch[frame * self.num_channels + ch]);
            }
        }

        let ratio = self.current_kernel_ratio(frames);
        if (ratio - self.active_ratio).abs() > RATIO_SNAP_EPS {
            for vocoder in &mut self.vocoders {
                vocoder.set_stretch_ratio(ratio);
            }
            self.active_ratio = ratio;
        }

        let mut min_output_len = usize::MAX;
        for ch in 0..self.num_channels {
            self.tonal_output[ch].clear();
            self.vocoders[ch]
                .process_streaming_into(&self.channel_input[ch], &mut self.tonal_output[ch])?;
            min_output_len = min_output_len.min(self.tonal_output[ch].len());
        }
        if min_output_len == usize::MAX || min_output_len == 0 {
            self.consume_kernel_input();
            return Ok(());
        }

        for ch in 0..self.num_channels {
            resample_linear_to_len(
                &self.channel_input[ch],
                &mut self.transient_output[ch],
                min_output_len,
                "rt_transient_lane",
            )?;
            self.render_residual_lane(ch, min_output_len)?;
        }

        let weights = self.effective_lane_weights();
        self.mix_into_pending(min_output_len, weights)?;
        self.consume_kernel_input();
        Ok(())
    }

    fn render_residual_lane(&mut self, ch: usize, out_len: usize) -> Result<(), StretchError> {
        let src = &self.channel_input[ch];
        let residual_src = &mut self.residual_source[ch];
        if residual_src.capacity() < src.len() {
            return Err(StretchError::BufferOverflow {
                buffer: "rt_residual_source",
                requested: src.len(),
                available: residual_src.capacity(),
            });
        }
        residual_src.clear();

        let mut lp = self.residual_lp_state[ch];
        for &sample in src {
            lp += RESIDUAL_HPF_ALPHA * (sample - lp);
            residual_src.push(sample - lp);
        }
        self.residual_lp_state[ch] = lp;

        resample_linear_to_len(
            residual_src,
            &mut self.residual_output[ch],
            out_len,
            "rt_residual_lane",
        )?;
        for (idx, sample) in self.residual_output[ch].iter_mut().enumerate() {
            if (idx & 1) == 1 {
                *sample = -*sample;
            }
        }
        Ok(())
    }

    fn mix_into_pending(&mut self, frames: usize, weights: [f32; 3]) -> Result<(), StretchError> {
        let needed = frames.saturating_mul(self.num_channels);
        let overflow = needed.saturating_sub(self.pending_output.available());
        if overflow > 0 {
            self.pending_output.discard(overflow);
            self.force_tier_demote();
        }

        for frame in 0..frames {
            for ch in 0..self.num_channels {
                let tonal = self.tonal_output[ch][frame];
                let transient = self.transient_output[ch][frame];
                let residual = self.residual_output[ch][frame];
                let mixed = transient * weights[0] + tonal * weights[1] + residual * weights[2];
                if !self.pending_output.push(mixed) {
                    return Err(StretchError::InvalidState(
                        "rt pending output rejected push after capacity check",
                    ));
                }
            }
        }
        Ok(())
    }

    fn flush_tonal_tails_to_pending(&mut self) -> Result<(), StretchError> {
        let mut min_len = usize::MAX;
        for ch in 0..self.num_channels {
            self.tonal_output[ch].clear();
            self.vocoders[ch].flush_streaming_into(&mut self.tonal_output[ch])?;
            min_len = min_len.min(self.tonal_output[ch].len());
        }
        if min_len == usize::MAX || min_len == 0 {
            return Ok(());
        }

        let needed = min_len.saturating_mul(self.num_channels);
        let overflow = needed.saturating_sub(self.pending_output.available());
        if overflow > 0 {
            self.pending_output.discard(overflow);
        }
        for frame in 0..min_len {
            for ch in 0..self.num_channels {
                if !self.pending_output.push(self.tonal_output[ch][frame]) {
                    return Err(StretchError::InvalidState(
                        "rt tail push failed after capacity check",
                    ));
                }
            }
        }
        Ok(())
    }

    fn drain_pending_to_output(
        &mut self,
        output: &mut Vec<f32>,
        max_samples: usize,
    ) -> Result<usize, StretchError> {
        let to_emit = self.pending_output.len().min(max_samples);
        if to_emit == 0 {
            return Ok(0);
        }

        let available = output.capacity().saturating_sub(output.len());
        if to_emit > available {
            return Err(StretchError::BufferOverflow {
                buffer: "rt_process_output",
                requested: to_emit,
                available,
            });
        }

        let mut emitted = 0usize;
        let mut chunk = [0.0f32; 512];
        while emitted < to_emit {
            let want = (to_emit - emitted).min(chunk.len());
            let n = self.pending_output.pop_slice(&mut chunk[..want]);
            if n == 0 {
                return Err(StretchError::InvalidState(
                    "rt pending drain made zero progress",
                ));
            }
            output.extend_from_slice(&chunk[..n]);
            emitted += n;
        }
        Ok(emitted)
    }

    fn drain_pending_to_slices(
        &mut self,
        output_slices: &mut [&mut [f32]],
        max_frames: usize,
    ) -> Result<usize, StretchError> {
        if output_slices.len() != self.num_channels {
            return Err(StretchError::InvalidFormat(format!(
                "drain_pending_to_slices expects {} channels, got {}",
                self.num_channels,
                output_slices.len()
            )));
        }

        let available_frames = self.pending_output.len() / self.num_channels.max(1);
        let emit_frames = available_frames.min(max_frames);
        if emit_frames == 0 {
            return Ok(0);
        }

        for frame in 0..emit_frames {
            for ch in 0..self.num_channels {
                let Some(sample) = self.pending_output.pop() else {
                    return Err(StretchError::InvalidState(
                        "rt pending drain to slices made zero progress",
                    ));
                };
                output_slices[ch][frame] = sample;
            }
        }

        Ok(emit_frames)
    }

    fn push_zeros(&mut self, count: usize) -> Result<(), StretchError> {
        if count == 0 {
            return Ok(());
        }
        let zeros = [0.0f32; 256];
        let mut remain = count;
        while remain > 0 {
            let take = remain.min(zeros.len());
            let pushed = self.input_ring.push_slice(&zeros[..take]);
            if pushed != take {
                return Err(StretchError::BufferOverflow {
                    buffer: "rt_input_ring",
                    requested: take,
                    available: pushed,
                });
            }
            remain -= take;
        }
        Ok(())
    }

    #[inline]
    fn current_kernel_ratio(&self, frames: usize) -> f64 {
        let start = self.input_timeline_frames;
        let end = start + frames as f64;
        let base = self.warp_map.ratio_over_range(start, end);
        let bias = self.hints.ratio_bias.clamp(-0.25, 0.25);
        let hinted = base * (1.0 + bias);
        hinted.clamp(self.config.min_ratio, self.config.max_ratio)
    }

    fn effective_lane_weights(&self) -> [f32; 3] {
        let hints = &self.hints;
        let bias = hints.normalized_lane_bias();
        let transient = self.blend_weights[0]
            + 0.20 * hints.transient_confidence.clamp(0.0, 1.0)
            + 0.15 * bias[0];
        let tonal = self.blend_weights[1]
            + 0.20 * hints.tonal_confidence.clamp(0.0, 1.0)
            + 0.10 * hints.beat_confidence.clamp(0.0, 1.0)
            + 0.10 * bias[1];
        let residual =
            self.blend_weights[2] + 0.20 * hints.noise_confidence.clamp(0.0, 1.0) + 0.15 * bias[2];

        let sum = (transient + tonal + residual).max(1e-6);
        [transient / sum, tonal / sum, residual / sum]
    }

    fn consume_kernel_input(&mut self) {
        let hop = self.config.params.hop_size;
        let fft = self.config.params.fft_size;
        if self.config.kernel_frames < fft || hop == 0 {
            return;
        }
        let num_frames_processed = (self.config.kernel_frames - fft) / hop + 1;
        let consumed_frames = num_frames_processed
            .saturating_mul(hop)
            .min(self.config.kernel_frames);
        let consumed_samples = consumed_frames.saturating_mul(self.num_channels);
        self.input_ring.discard(consumed_samples);
        self.input_timeline_frames += consumed_frames as f64;
    }
}

fn resample_linear_to_len(
    input: &[f32],
    output: &mut Vec<f32>,
    output_len: usize,
    buffer_name: &'static str,
) -> Result<(), StretchError> {
    output.clear();
    if output_len == 0 || input.is_empty() {
        return Ok(());
    }
    if output.capacity() < output_len {
        return Err(StretchError::BufferOverflow {
            buffer: buffer_name,
            requested: output_len,
            available: output.capacity(),
        });
    }

    if input.len() == 1 {
        output.resize(output_len, input[0]);
        return Ok(());
    }

    let src_max = (input.len() - 1) as f64;
    let denom = (output_len - 1).max(1) as f64;
    for i in 0..output_len {
        let pos = (i as f64 / denom) * src_max;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let a = input[idx];
        let b = input[(idx + 1).min(input.len() - 1)];
        output.push(a + (b - a) * frac);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{LatencyProfile, RtConfig, RtProcessor};
    use crate::core::types::StretchParams;

    fn stereo_sine_block(frames: usize, sample_rate: u32, hz: f32, phase: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * hz * t + phase).sin() * 0.3;
            out.push(sample);
            out.push(sample);
        }
        out
    }

    #[test]
    fn prepare_rejects_undersized_contracts() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut cfg = RtConfig::new(params, 256);
        cfg.input_ring_blocks = 0;
        assert!(RtProcessor::prepare(cfg).is_err());
    }

    #[test]
    fn process_block_emits_audio_without_allocating_internal_structures() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut cfg = RtConfig::new(params, 256);
        cfg.latency_profile = LatencyProfile::Scratch;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let mut out = Vec::with_capacity(256 * 2 * 32);
        for i in 0..16 {
            let block = stereo_sine_block(256, 48_000, 220.0, i as f32 * 0.1);
            rt.process_block(&block, &mut out).unwrap();
        }
        assert!(!out.is_empty());
    }
}

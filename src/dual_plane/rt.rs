//! Hard-RT audio plane.
//!
//! The callback-facing API is intentionally small:
//! - [`RtProcessor::prepare`]
//! - [`RtProcessor::process_block`]
//! - [`RtProcessor::flush`]

use crate::core::crossover::LR4Crossover;
use crate::core::ring_buffer::RingBuffer;
use crate::core::types::StretchParams;
use crate::dual_plane::hints::RenderHints;
use crate::dual_plane::quality::{LatencyProfile, QualityGovernor, QualityTier, RtGovernorConfig};
use crate::dual_plane::warp_map::TimeWarpMap;
use crate::error::StretchError;
use crate::stretch::phase_vocoder::PhaseVocoder;
use crate::stretch::wsola::Wsola;
use arc_swap::ArcSwap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const RATIO_SNAP_EPS: f64 = 1e-6;
const UNITY_BYPASS_RATIO_EPS: f64 = 1e-6;
const ALGORITHMIC_DELAY_FFT_NUMERATOR: usize = 3;
const ALGORITHMIC_DELAY_FFT_DENOMINATOR: usize = 2;

/// Lock-free snapshot mailbox shared between control producers and RT callback.
///
/// Publishers always replace the previously visible snapshot in-place.
/// RT reads are wait-free: one atomic sequence load and one `Arc` load per
/// updated snapshot kind.
struct RtControlMailbox {
    warp_map: ArcSwap<TimeWarpMap>,
    hints: ArcSwap<RenderHints>,
    warp_sequence: AtomicU64,
    hints_sequence: AtomicU64,
}

impl RtControlMailbox {
    fn new(warp_map: Arc<TimeWarpMap>, hints: Arc<RenderHints>) -> Self {
        Self {
            warp_map: ArcSwap::from(warp_map),
            hints: ArcSwap::from(hints),
            warp_sequence: AtomicU64::new(0),
            hints_sequence: AtomicU64::new(0),
        }
    }

    #[inline]
    fn publish_warp_map(&self, map: Arc<TimeWarpMap>) {
        self.warp_map.store(map);
        self.warp_sequence.fetch_add(1, Ordering::Release);
    }

    #[inline]
    fn publish_hints(&self, hints: Arc<RenderHints>) {
        self.hints.store(hints);
        self.hints_sequence.fetch_add(1, Ordering::Release);
    }

    #[inline]
    fn latest_warp_map_if_updated(&self, seen_sequence: u64) -> Option<(u64, Arc<TimeWarpMap>)> {
        let sequence = self.warp_sequence.load(Ordering::Acquire);
        if sequence == seen_sequence {
            return None;
        }
        Some((sequence, self.warp_map.load_full()))
    }

    #[inline]
    fn latest_hints_if_updated(&self, seen_sequence: u64) -> Option<(u64, Arc<RenderHints>)> {
        let sequence = self.hints_sequence.load(Ordering::Acquire);
        if sequence == seen_sequence {
            return None;
        }
        Some((sequence, self.hints.load_full()))
    }
}

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
    /// Enables context-aware profile switching policy.
    pub auto_profile_switching: bool,
    /// Required consecutive blocks before applying an auto profile change.
    pub profile_switch_hysteresis_blocks: usize,
    /// Optional stem-aware lane weighting in RT mixer.
    ///
    /// Off by default for conservative rollout.
    pub stem_aware_lanes: bool,
    /// Strength of stem-aware lane biasing in `[0, 1]`.
    ///
    /// Effective only when `stem_aware_lanes` is enabled.
    pub stem_lane_hint_strength: f32,
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
            auto_profile_switching: false,
            profile_switch_hysteresis_blocks: 8,
            stem_aware_lanes: false,
            stem_lane_hint_strength: 0.65,
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
        if self.profile_switch_hysteresis_blocks == 0 {
            return Err(StretchError::InvalidFormat(
                "profile_switch_hysteresis_blocks must be > 0".to_string(),
            ));
        }
        if !self.stem_lane_hint_strength.is_finite()
            || self.stem_lane_hint_strength < 0.0
            || self.stem_lane_hint_strength > 1.0
        {
            return Err(StretchError::InvalidFormat(format!(
                "stem_lane_hint_strength must be in [0, 1], got {}",
                self.stem_lane_hint_strength
            )));
        }
        Ok(())
    }
}

/// Runtime profile state exported for host integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtProfileTelemetry {
    pub auto_switching_enabled: bool,
    pub current_profile: LatencyProfile,
    pub target_profile: LatencyProfile,
    pub policy_profile: LatencyProfile,
    pub transition_blocks_left: usize,
    pub callback_budget: Duration,
    pub current_tier: QualityTier,
    pub target_tier: QualityTier,
}

/// Realtime runtime telemetry exported for host integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RtRuntimeTelemetry {
    pub input_overflow_events: u64,
    pub input_dropped_samples: u64,
    pub output_overflow_events: u64,
    pub output_dropped_samples: u64,
    pub quality_demotions_due_to_overload: u64,
    pub process_error_count: u64,
}

/// Exact realtime delay telemetry exported for host compensation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RtDelayTelemetry {
    /// Fixed algorithmic delay of the active stretcher core.
    pub algorithmic_frames: usize,
    /// Additional queued input not yet consumed by the RT core.
    pub buffered_input_frames: usize,
    /// Additional rendered output still queued for host delivery.
    pub buffered_output_frames: usize,
    /// Delay contribution from the current latency profile.
    pub profile_frames: usize,
    /// Delay contribution from the current quality tier.
    pub tier_frames: usize,
    /// Exact current total delay visible to the host.
    pub total_frames: usize,
}

/// Sender handle for publishing control-plane snapshots without blocking RT.
#[derive(Clone)]
pub struct RtControlSender {
    mailbox: Arc<RtControlMailbox>,
}

impl RtControlSender {
    pub fn publish_warp_map(&self, map: Arc<TimeWarpMap>) -> bool {
        self.mailbox.publish_warp_map(map);
        true
    }

    pub fn publish_hints(&self, hints: Arc<RenderHints>) -> bool {
        self.mailbox.publish_hints(hints);
        true
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
    pv_channel_input: Vec<Vec<f32>>,
    sub_bass_crossovers: Vec<LR4Crossover>,
    tonal_output: Vec<Vec<f32>>,
    transient_output: Vec<Vec<f32>>,
    transient_mask: Vec<f32>,
    vocoders: Vec<PhaseVocoder>,
    transient_stretchers: Vec<Wsola>,
    warp_map: Arc<TimeWarpMap>,
    constant_ratio_override: Option<f64>,
    hints: Arc<RenderHints>,
    control: RtControlSender,
    control_warp_sequence: u64,
    control_hints_sequence: u64,
    governor: QualityGovernor,
    auto_profile_switching: bool,
    profile_switch_hysteresis_blocks: usize,
    current_profile: LatencyProfile,
    target_profile: LatencyProfile,
    policy_profile: LatencyProfile,
    profile_candidate: LatencyProfile,
    profile_candidate_streak: usize,
    profile_transition_blocks_left: usize,
    current_tier: QualityTier,
    target_tier: QualityTier,
    runtime_telemetry: RtRuntimeTelemetry,
    blend_weights: [f32; 2],
    target_weights: [f32; 2],
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
            .field("auto_profile_switching", &self.auto_profile_switching)
            .field("current_profile", &self.current_profile)
            .field("target_profile", &self.target_profile)
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
    pub fn prepare(mut config: RtConfig) -> Result<Self, StretchError> {
        config
            .latency_profile
            .apply_governor_defaults(&mut config.governor);
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
        let transient_segment = config
            .params
            .hop_size
            .saturating_mul(4)
            .max(64)
            .min(config.kernel_frames.max(64));
        let transient_search = (transient_segment / 2).max(8);
        let transient_stretchers = (0..num_channels)
            .map(|_| {
                let mut wsola = Wsola::new(
                    transient_segment,
                    transient_search,
                    config.params.stretch_ratio,
                );
                wsola.reserve_output_capacity(config.kernel_frames, config.max_ratio);
                wsola.set_equal_power_crossfade();
                wsola
            })
            .collect::<Vec<_>>();

        let warp_map = Arc::new(
            TimeWarpMap::from_ratio(config.params.stretch_ratio, config.kernel_frames)
                .unwrap_or_default(),
        );
        let hints = Arc::new(RenderHints::default());
        let control = RtControlSender {
            mailbox: Arc::new(RtControlMailbox::new(
                Arc::clone(&warp_map),
                Arc::clone(&hints),
            )),
        };
        let active_ratio = config.params.stretch_ratio;
        let initial_profile = config.latency_profile;
        let auto_profile_switching = config.auto_profile_switching;
        let profile_switch_hysteresis_blocks = config.profile_switch_hysteresis_blocks.max(1);
        let sub_bass_cutoff = config.params.sub_bass_cutoff as f64;
        let sample_rate = config.params.sample_rate;

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
            pv_channel_input: (0..num_channels)
                .map(|_| Vec::with_capacity(kernel_samples / num_channels))
                .collect(),
            sub_bass_crossovers: (0..num_channels)
                .map(|_| LR4Crossover::new(sub_bass_cutoff, sample_rate))
                .collect(),
            tonal_output: (0..num_channels)
                .map(|_| Vec::with_capacity(max_output_frames))
                .collect(),
            transient_output: (0..num_channels)
                .map(|_| Vec::with_capacity(max_output_frames))
                .collect(),
            transient_mask: vec![0.0; max_output_frames],
            vocoders,
            transient_stretchers,
            warp_map,
            constant_ratio_override: None,
            hints,
            control,
            control_warp_sequence: 0,
            control_hints_sequence: 0,
            governor,
            auto_profile_switching,
            profile_switch_hysteresis_blocks,
            current_profile: initial_profile,
            target_profile: initial_profile,
            policy_profile: initial_profile,
            profile_candidate: initial_profile,
            profile_candidate_streak: 0,
            profile_transition_blocks_left: 0,
            current_tier: initial_tier,
            target_tier: initial_tier,
            runtime_telemetry: RtRuntimeTelemetry::default(),
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
        self.control.publish_warp_map(Arc::clone(&warp_map));
        self.control_warp_sequence = self.control.mailbox.warp_sequence.load(Ordering::Acquire);
        self.constant_ratio_override = None;
        self.warp_map = warp_map;
    }

    /// Sets a constant-ratio override without publishing a warp-map snapshot.
    ///
    /// This is intended for fast scalar tempo control where callback-time map
    /// allocation would be unnecessary overhead.
    #[inline]
    pub fn set_constant_ratio(&mut self, ratio: f64) {
        self.constant_ratio_override =
            Some(ratio.clamp(self.config.min_ratio, self.config.max_ratio));
    }

    /// Sets hint snapshot directly on this thread.
    #[inline]
    pub fn set_hint_snapshot(&mut self, hints: Arc<RenderHints>) {
        self.control.publish_hints(Arc::clone(&hints));
        self.control_hints_sequence = self.control.mailbox.hints_sequence.load(Ordering::Acquire);
        self.hints = hints;
    }

    /// Current active quality tier.
    #[inline]
    pub fn quality_tier(&self) -> QualityTier {
        self.current_tier
    }

    /// Sets a fixed latency profile and disables auto profile switching.
    pub fn set_latency_profile(&mut self, profile: LatencyProfile) {
        self.auto_profile_switching = false;
        self.policy_profile = profile;
        self.profile_candidate = profile;
        self.profile_candidate_streak = 0;
        self.set_latency_profile_internal(profile);
    }

    /// Enables or disables context-aware profile switching.
    pub fn set_auto_profile_switching(&mut self, enabled: bool) {
        self.auto_profile_switching = enabled;
        if !enabled {
            self.policy_profile = self.target_profile;
            self.profile_candidate = self.target_profile;
            self.profile_candidate_streak = 0;
        }
    }

    /// Returns current profile telemetry for host integration.
    pub fn profile_telemetry(&self) -> RtProfileTelemetry {
        RtProfileTelemetry {
            auto_switching_enabled: self.auto_profile_switching,
            current_profile: self.current_profile,
            target_profile: self.target_profile,
            policy_profile: self.policy_profile,
            transition_blocks_left: self.profile_transition_blocks_left,
            callback_budget: self.config.governor.callback_budget,
            current_tier: self.current_tier,
            target_tier: self.target_tier,
        }
    }

    /// Returns cumulative realtime runtime telemetry.
    pub fn runtime_telemetry(&self) -> RtRuntimeTelemetry {
        self.runtime_telemetry
    }

    /// Returns exact current delay telemetry for host compensation.
    pub fn delay_telemetry(&self) -> RtDelayTelemetry {
        let algorithmic_frames = algorithmic_delay_frames(self.config.params.fft_size);
        let buffered_input_frames = self.input_ring.len() / self.num_channels.max(1);
        let buffered_output_frames = self.pending_output.len() / self.num_channels.max(1);
        let profile_frames = 0;
        let tier_frames = 0;
        let total_frames = algorithmic_frames
            .saturating_add(buffered_input_frames)
            .saturating_add(buffered_output_frames)
            .saturating_add(profile_frames)
            .saturating_add(tier_frames);

        RtDelayTelemetry {
            algorithmic_frames,
            buffered_input_frames,
            buffered_output_frames,
            profile_frames,
            tier_frames,
            total_frames,
        }
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
        let result = (|| -> Result<(usize, usize), StretchError> {
            let start = Instant::now();
            self.poll_control_updates();
            self.update_profile_policy();
            self.advance_profile_transition();
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

            if self.can_unity_passthrough(input_frames, output_frames_capacity) {
                for ch in 0..self.num_channels {
                    output_slices[ch][..input_frames]
                        .copy_from_slice(&input_slices[ch][..input_frames]);
                }
                self.input_timeline_frames += input_frames as f64;
                self.active_ratio = 1.0;
                let tier = self.governor.observe_block(start.elapsed());
                self.set_target_tier(tier);
                return Ok((input_frames, input_frames));
            }

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
        })();
        self.record_rt_result(result)
    }

    /// Processes one callback block.
    pub fn process_block(
        &mut self,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<(), StretchError> {
        let result = (|| -> Result<(), StretchError> {
            let start = Instant::now();
            self.poll_control_updates();
            self.update_profile_policy();
            self.advance_profile_transition();
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

            let available_output_frames = output
                .capacity()
                .saturating_sub(output.len())
                .saturating_div(self.num_channels.max(1));
            if self.can_unity_passthrough(self.config.block_frames, available_output_frames) {
                let available_samples = output.capacity().saturating_sub(output.len());
                if input.len() > available_samples {
                    return Err(StretchError::BufferOverflow {
                        buffer: "rt_process_block_output",
                        requested: input.len(),
                        available: available_samples,
                    });
                }
                output.extend_from_slice(input);
                self.input_timeline_frames += self.config.block_frames as f64;
                self.active_ratio = 1.0;
                let tier = self.governor.observe_block(start.elapsed());
                self.set_target_tier(tier);
                return Ok(());
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
        })();
        self.record_rt_result(result)
    }

    /// Flushes all pending RT state.
    pub fn flush(&mut self, output: &mut Vec<f32>) -> Result<(), StretchError> {
        let result = (|| -> Result<(), StretchError> {
            self.poll_control_updates();
            self.update_profile_policy();
            self.advance_profile_transition();
            self.advance_tier_crossfade();

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
            for vocoder in &mut self.vocoders {
                vocoder.reset_phase_state();
            }

            Ok(())
        })();
        self.record_rt_result(result)
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
            if self.transient_output[ch].capacity() < max_output {
                return Err(StretchError::BufferOverflow {
                    buffer: "rt_transient_output_capacity",
                    requested: max_output,
                    available: self.transient_output[ch].capacity(),
                });
            }
            self.tonal_output[ch].clear();
            self.vocoders[ch].process_streaming_into(&zero_kernel, &mut self.tonal_output[ch])?;
            self.tonal_output[ch].clear();
            self.vocoders[ch].flush_streaming_into(&mut self.tonal_output[ch])?;
            self.tonal_output[ch].clear();
            self.vocoders[ch].reset_phase_state();

            self.transient_output[ch].clear();
            self.transient_stretchers[ch]
                .process_into_no_grow(&zero_kernel, &mut self.transient_output[ch])?;
            self.transient_output[ch].clear();
        }
        Ok(())
    }

    fn poll_control_updates(&mut self) {
        // RT-thread invariant:
        // - callback reads only atomics + `Arc` pointers here.
        // - no locks, no blocking channels, no syscalls.
        // Control-plane invariant:
        // - publishers are non-blocking and overwrite older snapshots.
        // - RT always converges to latest published snapshot.
        if let Some((sequence, map)) = self
            .control
            .mailbox
            .latest_warp_map_if_updated(self.control_warp_sequence)
        {
            self.control_warp_sequence = sequence;
            self.constant_ratio_override = None;
            self.warp_map = map;
        }
        if let Some((sequence, hints)) = self
            .control
            .mailbox
            .latest_hints_if_updated(self.control_hints_sequence)
        {
            self.control_hints_sequence = sequence;
            self.hints = hints;
        }
    }

    #[inline]
    fn max_output_samples_per_callback(&self) -> usize {
        self.config
            .max_output_frames_per_kernel()
            .max((self.config.block_frames as f64 * self.config.max_ratio).ceil() as usize)
            .saturating_mul(self.num_channels)
    }

    fn reset_state_for_unity_passthrough(&mut self) {
        self.input_ring.clear();
        self.pending_output.clear();
        for out in &mut self.tonal_output {
            out.clear();
        }
        for out in &mut self.transient_output {
            out.clear();
        }
        for vocoder in &mut self.vocoders {
            vocoder.reset_phase_state();
        }
    }

    #[inline]
    fn can_unity_passthrough(
        &mut self,
        input_frames: usize,
        output_frames_capacity: usize,
    ) -> bool {
        if input_frames == 0 || output_frames_capacity < input_frames {
            return false;
        }

        let start = self.input_timeline_frames;
        let end = start + input_frames as f64;
        let base_ratio = self.base_ratio_over_range(start, end);
        if (base_ratio - 1.0).abs() > UNITY_BYPASS_RATIO_EPS {
            return false;
        }

        // Re-arm bit-exact passthrough after non-unity runs by dropping any
        // buffered overlap context when the host returns to unity ratio.
        if !self.input_ring.is_empty() || !self.pending_output.is_empty() {
            self.reset_state_for_unity_passthrough();
        }
        true
    }

    fn set_latency_profile_internal(&mut self, profile: LatencyProfile) {
        if self.target_profile == profile && self.current_profile == profile {
            return;
        }
        self.target_profile = profile;
        self.profile_transition_blocks_left = self
            .current_profile
            .tier_crossfade_blocks()
            .max(profile.tier_crossfade_blocks());
        self.config.latency_profile = profile;
        profile.apply_governor_defaults(&mut self.config.governor);
        self.governor.set_config(self.config.governor);
        self.governor.set_tier(profile.initial_tier());
        self.set_target_tier(profile.initial_tier());
    }

    #[inline]
    fn suggest_profile(&self) -> LatencyProfile {
        let ratio_delta = (self.active_ratio - 1.0).abs();
        let transient = self.hints.transient_confidence.clamp(0.0, 1.0);
        let tonal = self.hints.tonal_confidence.clamp(0.0, 1.0);
        let noise = self.hints.noise_confidence.clamp(0.0, 1.0);

        if ratio_delta >= 0.42 || transient >= 0.72 {
            LatencyProfile::Scratch
        } else if ratio_delta <= 0.12 && tonal >= 0.70 && noise <= 0.35 {
            LatencyProfile::Render
        } else {
            LatencyProfile::Mix
        }
    }

    fn update_profile_policy(&mut self) {
        if !self.auto_profile_switching {
            self.policy_profile = self.target_profile;
            return;
        }

        let suggested = self.suggest_profile();
        self.policy_profile = suggested;
        if suggested == self.target_profile {
            self.profile_candidate = suggested;
            self.profile_candidate_streak = 0;
            return;
        }

        if suggested == self.profile_candidate {
            self.profile_candidate_streak = self.profile_candidate_streak.saturating_add(1);
        } else {
            self.profile_candidate = suggested;
            self.profile_candidate_streak = 1;
        }

        if self.profile_candidate_streak >= self.profile_switch_hysteresis_blocks {
            self.profile_candidate_streak = 0;
            self.set_latency_profile_internal(suggested);
        }
    }

    fn advance_profile_transition(&mut self) {
        if self.current_profile == self.target_profile {
            self.profile_transition_blocks_left = 0;
            return;
        }
        if self.profile_transition_blocks_left == 0 {
            self.current_profile = self.target_profile;
            return;
        }
        self.profile_transition_blocks_left = self.profile_transition_blocks_left.saturating_sub(1);
        if self.profile_transition_blocks_left == 0 {
            self.current_profile = self.target_profile;
        }
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
        for i in 0..2 {
            self.blend_weights[i] += (self.target_weights[i] - self.blend_weights[i]) / denom;
        }
        self.crossfade_blocks_left = self.crossfade_blocks_left.saturating_sub(1);
        if self.crossfade_blocks_left == 0 {
            self.blend_weights = self.target_weights;
            self.current_tier = self.target_tier;
        }
    }

    fn force_tier_demote(&mut self) {
        self.runtime_telemetry.quality_demotions_due_to_overload = self
            .runtime_telemetry
            .quality_demotions_due_to_overload
            .saturating_add(1);
        let next = self.governor.force_demote_once();
        self.set_target_tier(next);
    }

    #[inline]
    fn record_input_overflow(&mut self, dropped_samples: usize) {
        if dropped_samples == 0 {
            return;
        }
        self.runtime_telemetry.input_overflow_events = self
            .runtime_telemetry
            .input_overflow_events
            .saturating_add(1);
        self.runtime_telemetry.input_dropped_samples = self
            .runtime_telemetry
            .input_dropped_samples
            .saturating_add(dropped_samples as u64);
    }

    #[inline]
    fn record_output_overflow(&mut self, dropped_samples: usize) {
        if dropped_samples == 0 {
            return;
        }
        self.runtime_telemetry.output_overflow_events = self
            .runtime_telemetry
            .output_overflow_events
            .saturating_add(1);
        self.runtime_telemetry.output_dropped_samples = self
            .runtime_telemetry
            .output_dropped_samples
            .saturating_add(dropped_samples as u64);
    }

    #[inline]
    fn record_rt_result<T>(&mut self, result: Result<T, StretchError>) -> Result<T, StretchError> {
        if result.is_err() {
            self.runtime_telemetry.process_error_count =
                self.runtime_telemetry.process_error_count.saturating_add(1);
        }
        result
    }

    fn push_input_with_overload_policy(&mut self, input: &[f32]) -> Result<(), StretchError> {
        let overflow = input.len().saturating_sub(self.input_ring.available());
        if overflow > 0 {
            self.record_input_overflow(overflow);
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
            self.record_input_overflow(overflow);
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
        let kernel_start_frame = self.input_timeline_frames;
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
        }
        for frame in 0..frames {
            let base = frame * self.num_channels;
            for ch in 0..self.num_channels {
                let sample = self.interleaved_scratch[base + ch];
                self.channel_input[ch].push(sample);
            }
        }

        let ratio = self.current_kernel_ratio(frames);
        if (ratio - self.active_ratio).abs() > RATIO_SNAP_EPS {
            for vocoder in &mut self.vocoders {
                vocoder.set_stretch_ratio(ratio);
            }
            for stretcher in &mut self.transient_stretchers {
                stretcher.set_stretch_ratio(ratio);
            }
            self.active_ratio = ratio;
        }

        // Build PV input: highpass to remove sub-bass, then mask transients.
        // Sub-bass occupies 1-2 FFT bins at typical sizes; PV smears them.
        // WSOLA receives the full signal (including sub-bass) since it's
        // time-domain and handles low frequencies without phase artifacts.
        // Transient masking prevents phase smearing of percussive content.
        for ch in 0..self.num_channels {
            self.pv_channel_input[ch].clear();
            for frame in 0..frames {
                let (_low, high) =
                    self.sub_bass_crossovers[ch].process_sample(self.channel_input[ch][frame]);
                self.pv_channel_input[ch].push(high);
            }
        }
        self.apply_input_domain_transient_mask(frames, kernel_start_frame);

        let mut min_output_len = usize::MAX;
        for ch in 0..self.num_channels {
            self.tonal_output[ch].clear();
            self.vocoders[ch]
                .process_streaming_into(&self.pv_channel_input[ch], &mut self.tonal_output[ch])?;
            min_output_len = min_output_len.min(self.tonal_output[ch].len());

            self.transient_output[ch].clear();
            self.transient_stretchers[ch]
                .process_into_no_grow(&self.channel_input[ch], &mut self.transient_output[ch])?;
            min_output_len = min_output_len.min(self.transient_output[ch].len());
        }
        if min_output_len == usize::MAX || min_output_len == 0 {
            self.consume_kernel_input();
            return Ok(());
        }

        self.build_transient_mask_from_hints(min_output_len, ratio, kernel_start_frame);

        let weights = self.effective_lane_weights();
        self.mix_into_pending(min_output_len, weights)?;
        self.consume_kernel_input();
        Ok(())
    }

    fn build_transient_mask_from_hints(&mut self, out_len: usize, ratio: f64, kernel_start: f64) {
        if out_len == 0 {
            self.transient_mask.clear();
            return;
        }
        if self.transient_mask.len() < out_len {
            self.transient_mask.resize(out_len, 0.0);
        } else {
            for sample in self.transient_mask[..out_len].iter_mut() {
                *sample = 0.0;
            }
        }

        if matches!(self.current_tier, QualityTier::Q0) {
            return;
        }
        let hint_mask = &self.hints.transient_mask;
        if hint_mask.is_empty() {
            return;
        }
        let hint_start = self.hints.at_input_frame as f64;
        let hint_end = hint_start + hint_mask.len() as f64;
        let inv_ratio = 1.0 / ratio.max(1e-6);

        for out_idx in 0..out_len {
            let input_pos = kernel_start + out_idx as f64 * inv_ratio;
            if input_pos < hint_start || input_pos >= hint_end {
                continue;
            }
            let hint_idx = (input_pos - hint_start) as usize;
            self.transient_mask[out_idx] = hint_mask[hint_idx].clamp(0.0, 1.0);
        }
    }

    /// Attenuate transient content in `pv_channel_input` using the hint mask
    /// mapped directly to input-domain frames (no ratio scaling). This prevents
    /// transient energy from entering the phase vocoder where it would cause
    /// phase smearing artifacts.
    fn apply_input_domain_transient_mask(&mut self, input_frames: usize, kernel_start: f64) {
        if matches!(self.current_tier, QualityTier::Q0) {
            return;
        }
        let hint_mask = &self.hints.transient_mask;
        if hint_mask.is_empty() {
            return;
        }
        let hint_start = self.hints.at_input_frame as f64;
        let hint_end = hint_start + hint_mask.len() as f64;

        for frame in 0..input_frames {
            let input_pos = kernel_start + frame as f64;
            if input_pos < hint_start || input_pos >= hint_end {
                continue;
            }
            let hint_idx = (input_pos - hint_start) as usize;
            let mask_val = hint_mask[hint_idx].clamp(0.0, 1.0);
            if mask_val < 1e-6 {
                continue;
            }
            // Attenuate transient content: multiply by (1 - mask)
            let scale = 1.0 - mask_val;
            for ch in 0..self.num_channels {
                self.pv_channel_input[ch][frame] *= scale;
            }
        }
    }

    fn mix_into_pending(&mut self, frames: usize, weights: [f32; 2]) -> Result<(), StretchError> {
        let needed = frames.saturating_mul(self.num_channels);
        let overflow = needed.saturating_sub(self.pending_output.available());
        if overflow > 0 {
            self.record_output_overflow(overflow);
            self.pending_output.discard(overflow);
            self.force_tier_demote();
        }

        for frame in 0..frames {
            let transient_gate = self
                .transient_mask
                .get(frame)
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let transient_w = weights[0] * (0.40 + 0.60 * transient_gate);
            let tonal_w = weights[1] * (1.0 - 0.55 * transient_gate);
            let norm = (transient_w + tonal_w).max(1e-6);
            let tw = transient_w / norm;
            let tow = tonal_w / norm;

            for ch in 0..self.num_channels {
                let mixed =
                    self.transient_output[ch][frame] * tw + self.tonal_output[ch][frame] * tow;
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
            self.record_output_overflow(overflow);
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
        let base = self.base_ratio_over_range(start, end);
        let bias = self.hints.ratio_bias.clamp(-0.25, 0.25);
        let hinted = base * (1.0 + bias);
        hinted.clamp(self.config.min_ratio, self.config.max_ratio)
    }

    #[inline]
    fn base_ratio_over_range(&self, start: f64, end: f64) -> f64 {
        if let Some(ratio) = self.constant_ratio_override {
            ratio
        } else {
            self.warp_map.ratio_over_range(start, end)
        }
    }

    fn effective_lane_weights(&self) -> [f32; 2] {
        let hints = &self.hints;
        let bias = hints.normalized_lane_bias();
        let stem = hints.normalized_stem_lane_confidence();
        let stem_gate = if self.config.stem_aware_lanes {
            1.0
        } else {
            0.0
        };
        let stem_strength = self.config.stem_lane_hint_strength * stem_gate;
        let transient = self.blend_weights[0]
            + 0.20 * hints.transient_confidence.clamp(0.0, 1.0)
            + 0.15 * bias[0]
            + stem_strength * 0.28 * stem[0];
        let tonal = self.blend_weights[1]
            + 0.20 * hints.tonal_confidence.clamp(0.0, 1.0)
            + 0.10 * hints.beat_confidence.clamp(0.0, 1.0)
            + 0.10 * bias[1]
            + stem_strength * 0.30 * stem[1];

        let sum = (transient + tonal).max(1e-6);
        [transient / sum, tonal / sum]
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

#[inline]
const fn algorithmic_delay_frames(fft_size: usize) -> usize {
    fft_size * ALGORITHMIC_DELAY_FFT_NUMERATOR / ALGORITHMIC_DELAY_FFT_DENOMINATOR
}

#[cfg(test)]
mod tests {
    use super::{
        LatencyProfile, QualityTier, RtConfig, RtDelayTelemetry, RtProcessor, RtRuntimeTelemetry,
    };
    use crate::core::types::StretchParams;
    use crate::dual_plane::hints::RenderHints;
    use crate::dual_plane::warp_map::TimeWarpMap;
    use std::sync::Arc;

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

    #[test]
    fn process_slice_unity_ratio_is_bit_exact_passthrough() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut cfg = RtConfig::new(params, 256);
        cfg.latency_profile = LatencyProfile::Scratch;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let frames = 256usize;
        let mut left = vec![0.0f32; frames];
        let mut right = vec![0.0f32; frames];
        for i in 0..frames {
            left[i] = (i as f32 * 0.01).sin() * 0.5;
            right[i] = (i as f32 * 0.02).cos() * 0.4;
        }
        let input_refs = [&left[..], &right[..]];

        let mut out_left = vec![0.0f32; frames];
        let mut out_right = vec![0.0f32; frames];
        let mut output_refs = [&mut out_left[..], &mut out_right[..]];

        let (consumed, produced) = rt.process(&input_refs, &mut output_refs);
        assert_eq!(consumed, frames);
        assert_eq!(produced, frames);
        assert_eq!(out_left, left);
        assert_eq!(out_right, right);
        assert!(rt.input_ring.is_empty());
        assert!(rt.pending_output.is_empty());
    }

    #[test]
    fn process_block_unity_ratio_is_bit_exact_passthrough() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut cfg = RtConfig::new(params, 256);
        cfg.latency_profile = LatencyProfile::Scratch;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = stereo_sine_block(256, 48_000, 330.0, 0.37);
        let mut out = Vec::with_capacity(input.len() + 64);
        rt.process_block(&input, &mut out).unwrap();

        assert_eq!(out, input);
        assert!(rt.input_ring.is_empty());
        assert!(rt.pending_output.is_empty());
    }

    #[test]
    fn unity_passthrough_reengages_after_non_unity_ratio_roundtrip() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut cfg = RtConfig::new(params, 256);
        cfg.latency_profile = LatencyProfile::Scratch;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let frames = 256usize;
        let mut left = vec![0.0f32; frames];
        let mut right = vec![0.0f32; frames];
        for i in 0..frames {
            left[i] = (i as f32 * 0.013).sin() * 0.4;
            right[i] = (i as f32 * 0.017).cos() * 0.3;
        }
        let input_refs = [&left[..], &right[..]];

        rt.set_constant_ratio(1.35);
        for _ in 0..4 {
            let mut out_left = vec![0.0f32; frames * 4];
            let mut out_right = vec![0.0f32; frames * 4];
            let mut output_refs = [&mut out_left[..], &mut out_right[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
        }
        assert!(
            !rt.input_ring.is_empty() || !rt.pending_output.is_empty(),
            "non-unity warmup should leave buffered processing context"
        );

        rt.set_constant_ratio(1.0);
        let mut out_left = vec![0.0f32; frames];
        let mut out_right = vec![0.0f32; frames];
        let mut output_refs = [&mut out_left[..], &mut out_right[..]];
        let (consumed, produced) = rt.process(&input_refs, &mut output_refs);

        assert_eq!(consumed, frames);
        assert_eq!(produced, frames);
        assert_eq!(out_left, left);
        assert_eq!(out_right, right);
        assert!(rt.input_ring.is_empty());
        assert!(rt.pending_output.is_empty());
    }

    fn assert_weights_close(a: [f32; 2], b: [f32; 2], eps: f32) {
        for idx in 0..2 {
            assert!(
                (a[idx] - b[idx]).abs() <= eps,
                "lane weight mismatch at index {idx}: {} vs {} (eps={eps})",
                a[idx],
                b[idx]
            );
        }
    }

    #[test]
    fn control_publish_is_non_blocking_under_bursty_updates() {
        let params = StretchParams::new(1.35)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let cfg = RtConfig::new(params, 128);
        let mut rt = RtProcessor::prepare(cfg).unwrap();
        let control = rt.control_sender();

        let mut expected = 1.0f64;
        for i in 0..4_096usize {
            expected = 0.60 + (i as f64 * 0.0005);
            let map = Arc::new(TimeWarpMap::from_ratio(expected, 256).unwrap());
            assert!(
                control.publish_warp_map(map),
                "control publish must not block or reject under burst load"
            );
        }

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];
        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);

        let actual = rt.active_ratio();
        assert!(
            (actual - expected).abs() < 1e-6,
            "rt should converge to latest published warp ratio (expected {expected}, got {actual})"
        );
    }

    #[test]
    fn control_hints_latest_value_wins_for_rt_ratio_bias() {
        let params = StretchParams::new(1.20)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let cfg = RtConfig::new(params, 128);
        let mut rt = RtProcessor::prepare(cfg).unwrap();
        let control = rt.control_sender();

        let mut final_bias = 0.0f64;
        for i in 0..1_024usize {
            final_bias = if (i & 1) == 0 { -0.20 } else { 0.15 };
            let hints = RenderHints {
                sequence: i as u64,
                ratio_bias: final_bias,
                ..RenderHints::default()
            };
            assert!(control.publish_hints(Arc::new(hints)));
        }

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];
        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);

        let expected = 1.20 * (1.0 + final_bias);
        let actual = rt.active_ratio();
        assert!(
            (actual - expected).abs() < 1e-6,
            "rt should consume latest hint snapshot bias (expected {expected}, got {actual})"
        );
    }

    #[test]
    fn stem_aware_lane_weighting_is_feature_gated_off_by_default() {
        let params = StretchParams::new(1.20)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let cfg = RtConfig::new(params, 128);
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let baseline = RenderHints {
            tonal_confidence: 0.6,
            noise_confidence: 0.1,
            lane_bias: [0.1, 0.2, 0.1],
            ..RenderHints::default()
        };
        let stem_heavy = RenderHints {
            stem_lane_confidence: [1.0, 0.0, 0.0],
            ..baseline.clone()
        };

        rt.set_hint_snapshot(Arc::new(baseline));
        let w0 = rt.effective_lane_weights();
        rt.set_hint_snapshot(Arc::new(stem_heavy));
        let w1 = rt.effective_lane_weights();

        assert_weights_close(w0, w1, 1e-6);
    }

    #[test]
    fn stem_aware_lane_weighting_changes_blend_when_enabled() {
        let params = StretchParams::new(1.20)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.stem_aware_lanes = true;
        cfg.stem_lane_hint_strength = 1.0;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let percussive = RenderHints {
            stem_lane_confidence: [1.0, 0.0, 0.0],
            transient_confidence: 0.8,
            tonal_confidence: 0.1,
            noise_confidence: 0.1,
            ..RenderHints::default()
        };
        let harmonic = RenderHints {
            stem_lane_confidence: [0.0, 1.0, 0.0],
            transient_confidence: 0.1,
            tonal_confidence: 0.8,
            noise_confidence: 0.1,
            ..RenderHints::default()
        };

        rt.set_hint_snapshot(Arc::new(percussive));
        let wp = rt.effective_lane_weights();
        rt.set_hint_snapshot(Arc::new(harmonic));
        let wh = rt.effective_lane_weights();

        assert!(
            wp[0] > wh[0],
            "percussive stem confidence should increase transient lane weight"
        );
        assert!(
            wh[1] > wp[1],
            "harmonic stem confidence should increase tonal lane weight"
        );
    }

    #[test]
    fn lane_transitions_and_tier_crossfades_are_deterministic() {
        let params = StretchParams::new(1.12)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.stem_aware_lanes = true;
        cfg.stem_lane_hint_strength = 0.85;
        cfg.latency_profile = LatencyProfile::Mix;
        let mut a = RtProcessor::prepare(cfg.clone()).unwrap();
        let mut b = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut out_a = [0.0f32; 512];
        let mut out_b = [0.0f32; 512];
        let input_refs = [&input[..]];

        for step in 0..12usize {
            let hints = if step < 4 {
                RenderHints {
                    stem_lane_confidence: [1.0, 0.0, 0.0],
                    transient_confidence: 0.9,
                    tonal_confidence: 0.1,
                    noise_confidence: 0.1,
                    ..RenderHints::default()
                }
            } else if step < 8 {
                RenderHints {
                    stem_lane_confidence: [0.0, 1.0, 0.0],
                    transient_confidence: 0.1,
                    tonal_confidence: 0.9,
                    noise_confidence: 0.1,
                    ..RenderHints::default()
                }
            } else {
                RenderHints {
                    stem_lane_confidence: [0.0, 0.0, 1.0],
                    transient_confidence: 0.2,
                    tonal_confidence: 0.2,
                    noise_confidence: 0.9,
                    ..RenderHints::default()
                }
            };

            a.set_hint_snapshot(Arc::new(hints.clone()));
            b.set_hint_snapshot(Arc::new(hints));
            if step == 3 {
                a.set_target_tier(QualityTier::Q4);
                b.set_target_tier(QualityTier::Q4);
            }

            let mut out_refs_a = [&mut out_a[..]];
            let mut out_refs_b = [&mut out_b[..]];
            let (consumed_a, produced_a) = a.process(&input_refs, &mut out_refs_a);
            let (consumed_b, produced_b) = b.process(&input_refs, &mut out_refs_b);

            assert_eq!(consumed_a, consumed_b);
            assert_eq!(produced_a, produced_b);
            assert_eq!(&out_a[..produced_a], &out_b[..produced_b]);
            assert_weights_close(a.effective_lane_weights(), b.effective_lane_weights(), 1e-6);
            assert_eq!(a.quality_tier(), b.quality_tier());
        }
    }

    #[test]
    fn manual_latency_profile_switch_disables_auto_and_updates_budget() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.auto_profile_switching = true;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        rt.set_latency_profile(LatencyProfile::Scratch);
        let telemetry = rt.profile_telemetry();
        assert!(!telemetry.auto_switching_enabled);
        assert_eq!(telemetry.target_profile, LatencyProfile::Scratch);
        assert_eq!(
            telemetry.callback_budget,
            LatencyProfile::Scratch.callback_budget()
        );
    }

    #[test]
    fn runtime_telemetry_tracks_overflows() {
        let params = StretchParams::new(1.10)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let cfg = RtConfig::new(params, 128);
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input_fill = rt.input_ring.capacity().saturating_sub(8);
        let input_prefill = vec![0.0f32; input_fill];
        assert_eq!(rt.input_ring.push_slice(&input_prefill), input_fill);
        rt.push_input_with_overload_policy(&[0.0; 16]).unwrap();

        let pending_fill = rt.pending_output.capacity().saturating_sub(1);
        let pending_prefill = vec![0.0f32; pending_fill];
        assert_eq!(rt.pending_output.push_slice(&pending_prefill), pending_fill);
        rt.transient_output[0].clear();
        rt.tonal_output[0].clear();
        rt.transient_output[0].extend_from_slice(&[0.25, 0.25]);
        rt.tonal_output[0].extend_from_slice(&[0.5, 0.5]);
        rt.transient_mask[0] = 0.0;
        rt.transient_mask[1] = 0.0;
        rt.mix_into_pending(2, [0.5, 0.5]).unwrap();

        assert_eq!(
            rt.runtime_telemetry(),
            RtRuntimeTelemetry {
                input_overflow_events: 1,
                input_dropped_samples: 8,
                output_overflow_events: 1,
                output_dropped_samples: 1,
                quality_demotions_due_to_overload: 2,
                process_error_count: 0,
            }
        );
    }

    #[test]
    fn runtime_telemetry_tracks_swallowed_process_errors() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let cfg = RtConfig::new(params, 128);
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let bad_inputs: [&[f32]; 0] = [];
        let mut out = [0.0f32; 128];
        let mut outputs = [&mut out[..]];
        assert_eq!(rt.process(&bad_inputs, &mut outputs), (0, 0));
        assert_eq!(rt.runtime_telemetry().process_error_count, 1);
    }

    #[test]
    fn delay_telemetry_tracks_algorithmic_and_buffered_frames() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let cfg = RtConfig::new(params, 128);
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        assert_eq!(
            rt.delay_telemetry(),
            RtDelayTelemetry {
                algorithmic_frames: 96,
                buffered_input_frames: 0,
                buffered_output_frames: 0,
                profile_frames: 0,
                tier_frames: 0,
                total_frames: 96,
            }
        );

        let input_fill = vec![0.0f32; 24];
        let output_fill = vec![0.0f32; 40];
        assert_eq!(rt.input_ring.push_slice(&input_fill), input_fill.len());
        assert_eq!(
            rt.pending_output.push_slice(&output_fill),
            output_fill.len()
        );

        assert_eq!(
            rt.delay_telemetry(),
            RtDelayTelemetry {
                algorithmic_frames: 96,
                buffered_input_frames: 24,
                buffered_output_frames: 40,
                profile_frames: 0,
                tier_frames: 0,
                total_frames: 160,
            }
        );
    }

    #[test]
    fn auto_profile_switching_changes_profiles_with_hysteresis() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];

        // Render-favoring policy: near-unity ratio + strong tonal confidence.
        rt.set_constant_ratio(1.0);
        rt.set_hint_snapshot(Arc::new(RenderHints {
            transient_confidence: 0.05,
            tonal_confidence: 0.95,
            noise_confidence: 0.05,
            ..RenderHints::default()
        }));
        for _ in 0..3 {
            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
        }
        assert_eq!(
            rt.profile_telemetry().target_profile,
            LatencyProfile::Render
        );

        // Scratch-favoring policy: large ratio delta or strong transient confidence.
        rt.set_constant_ratio(1.70);
        rt.set_hint_snapshot(Arc::new(RenderHints {
            transient_confidence: 0.90,
            tonal_confidence: 0.10,
            noise_confidence: 0.20,
            ..RenderHints::default()
        }));
        for _ in 0..3 {
            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
        }
        assert_eq!(
            rt.profile_telemetry().target_profile,
            LatencyProfile::Scratch
        );
    }
}

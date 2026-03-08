//! Hard-RT audio plane.
//!
//! The callback-facing API is intentionally small:
//! - [`RtProcessor::prepare`]
//! - [`RtProcessor::process_block`]
//! - [`RtProcessor::process_block_into`]
//! - [`RtProcessor::flush`]
//! - [`RtProcessor::flush_into`]

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
const RATIO_MOTION_FREEZE_TRIGGER: f64 = 7.5e-4;
const RATIO_MOTION_HISTORY_BLOCKS: usize = 4;

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
    /// Hold profile and tier transitions for this many kernels after a ratio step.
    pub ratio_motion_freeze_blocks: usize,
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
            ratio_motion_freeze_blocks: 3,
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
    unity_history: RingBuffer<f32>,
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
    ratio_motion_freeze_blocks_left: usize,
    post_ratio_motion_profile_hold_blocks_left: usize,
    ratio_motion_history: [f64; RATIO_MOTION_HISTORY_BLOCKS],
    ratio_motion_history_len: usize,
    ratio_motion_history_cursor: usize,
    current_tier: QualityTier,
    target_tier: QualityTier,
    runtime_telemetry: RtRuntimeTelemetry,
    blend_weights: [f32; 2],
    target_weights: [f32; 2],
    crossfade_blocks_left: usize,
    flush_drain_pending: bool,
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
            .field(
                "ratio_motion_freeze_blocks_left",
                &self.ratio_motion_freeze_blocks_left,
            )
            .field("current_tier", &self.current_tier)
            .field("target_tier", &self.target_tier)
            .field("active_ratio", &self.active_ratio)
            .field("input_ring_len", &self.input_ring.len())
            .field("pending_output_len", &self.pending_output.len())
            .field("flush_drain_pending", &self.flush_drain_pending)
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
            unity_history: RingBuffer::with_capacity(kernel_samples),
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
            ratio_motion_freeze_blocks_left: 0,
            post_ratio_motion_profile_hold_blocks_left: 0,
            ratio_motion_history: [1.0; RATIO_MOTION_HISTORY_BLOCKS],
            ratio_motion_history_len: 0,
            ratio_motion_history_cursor: 0,
            current_tier: initial_tier,
            target_tier: initial_tier,
            runtime_telemetry: RtRuntimeTelemetry::default(),
            blend_weights: blend,
            target_weights: blend,
            crossfade_blocks_left: 0,
            flush_drain_pending: false,
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
        self.post_ratio_motion_profile_hold_blocks_left = 0;
        self.set_latency_profile_internal(profile);
    }

    /// Enables or disables context-aware profile switching.
    pub fn set_auto_profile_switching(&mut self, enabled: bool) {
        self.auto_profile_switching = enabled;
        if !enabled {
            self.policy_profile = self.target_profile;
            self.profile_candidate = self.target_profile;
            self.profile_candidate_streak = 0;
            self.post_ratio_motion_profile_hold_blocks_left = 0;
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

    #[inline]
    fn ratio_motion_freeze_active(&self) -> bool {
        self.ratio_motion_freeze_blocks_left > 0
    }

    fn hold_profile_and_tier_transitions(&mut self) {
        self.policy_profile = self.current_profile;
        self.target_profile = self.current_profile;
        self.profile_transition_blocks_left = 0;
        self.config.latency_profile = self.current_profile;
        self.current_profile
            .apply_governor_defaults(&mut self.config.governor);
        self.governor.set_config(self.config.governor);
        self.governor.set_tier(self.current_tier);
        self.target_tier = self.current_tier;
        self.target_weights = self.blend_weights;
        self.crossfade_blocks_left = 0;
    }

    fn bias_ratio_motion_hold_to_scratch_if_needed(&mut self) -> bool {
        if !self.auto_profile_switching || self.current_profile != LatencyProfile::Mix {
            return false;
        }

        let scratch = LatencyProfile::Scratch;
        let scratch_tier = scratch.initial_tier();
        let scratch_weights = scratch_tier.lane_weights();
        self.current_profile = scratch;
        self.policy_profile = scratch;
        self.profile_candidate = scratch;
        self.profile_candidate_streak = 0;
        self.target_profile = scratch;
        self.profile_transition_blocks_left = 0;
        self.config.latency_profile = scratch;
        scratch.apply_governor_defaults(&mut self.config.governor);
        self.governor.set_config(self.config.governor);
        self.governor.set_tier(scratch_tier);
        self.current_tier = scratch_tier;
        self.target_tier = scratch_tier;
        self.blend_weights = scratch_weights;
        self.target_weights = scratch_weights;
        self.crossfade_blocks_left = 0;
        true
    }

    #[inline]
    fn reset_ratio_motion_history(&mut self) {
        self.ratio_motion_history_len = 0;
        self.ratio_motion_history_cursor = 0;
    }

    #[inline]
    fn push_ratio_motion_history(&mut self, ratio: f64) -> f64 {
        let slot = self.ratio_motion_history_cursor;
        self.ratio_motion_history[slot] = ratio;
        self.ratio_motion_history_cursor =
            (self.ratio_motion_history_cursor + 1) % self.ratio_motion_history.len();
        self.ratio_motion_history_len =
            (self.ratio_motion_history_len + 1).min(self.ratio_motion_history.len());

        let mut min_ratio = ratio;
        let mut max_ratio = ratio;
        for &observed in self
            .ratio_motion_history
            .iter()
            .take(self.ratio_motion_history_len)
        {
            min_ratio = min_ratio.min(observed);
            max_ratio = max_ratio.max(observed);
        }
        max_ratio - min_ratio
    }

    fn engage_ratio_motion_freeze_if_needed(&mut self, next_ratio: f64) {
        if self.config.ratio_motion_freeze_blocks == 0 || !next_ratio.is_finite() {
            return;
        }
        if (next_ratio - 1.0).abs() <= UNITY_BYPASS_RATIO_EPS {
            self.reset_ratio_motion_history();
            return;
        }
        let step_delta = (next_ratio - self.active_ratio).abs();
        if self.ratio_motion_freeze_active() {
            self.reset_ratio_motion_history();
            if step_delta < RATIO_MOTION_FREEZE_TRIGGER {
                return;
            }
        } else {
            let recent_span = self.push_ratio_motion_history(next_ratio);
            if step_delta < RATIO_MOTION_FREEZE_TRIGGER && recent_span < RATIO_MOTION_FREEZE_TRIGGER
            {
                return;
            }
        }

        self.ratio_motion_freeze_blocks_left = self.config.ratio_motion_freeze_blocks;
        self.post_ratio_motion_profile_hold_blocks_left = 0;
        if self.bias_ratio_motion_hold_to_scratch_if_needed() {
            return;
        }
        if self.auto_profile_switching && self.current_profile == LatencyProfile::Scratch {
            let scratch = LatencyProfile::Scratch;
            let current_weights = self.current_tier.lane_weights();
            self.policy_profile = scratch;
            self.profile_candidate = scratch;
            self.profile_candidate_streak = 0;
            self.target_profile = scratch;
            self.profile_transition_blocks_left = 0;
            self.config.latency_profile = scratch;
            scratch.apply_governor_defaults(&mut self.config.governor);
            self.governor.set_config(self.config.governor);
            self.governor.set_tier(self.current_tier);
            self.target_tier = self.current_tier;
            self.blend_weights = current_weights;
            self.target_weights = current_weights;
            self.crossfade_blocks_left = 0;
            return;
        }
        self.hold_profile_and_tier_transitions();
    }

    #[inline]
    fn advance_ratio_motion_freeze(&mut self) {
        let was_active = self.ratio_motion_freeze_blocks_left > 0;
        self.ratio_motion_freeze_blocks_left =
            self.ratio_motion_freeze_blocks_left.saturating_sub(1);
        if was_active && self.ratio_motion_freeze_blocks_left == 0 && self.auto_profile_switching {
            self.post_ratio_motion_profile_hold_blocks_left = 1;
        }
    }

    fn prepare_runtime_policy(&mut self) {
        self.poll_control_updates();
    }

    #[inline]
    fn advance_runtime_policy_for_committed_kernel(&mut self) {
        self.update_profile_policy();
        self.advance_profile_transition();
        self.advance_tier_crossfade();
    }

    #[inline]
    fn observe_governor_block(&mut self, elapsed: Duration) {
        let tier = self.governor.observe_block(elapsed);
        if !self.ratio_motion_freeze_active() {
            self.set_target_tier(tier);
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
            self.prepare_runtime_policy();
            self.assert_can_process_new_input()?;

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
                self.rearm_unity_passthrough();
                let needed_samples = input_frames.saturating_mul(self.num_channels);
                debug_assert!(needed_samples <= self.interleaved_scratch.len());
                for frame in 0..input_frames {
                    for ch in 0..self.num_channels {
                        self.interleaved_scratch[frame * self.num_channels + ch] =
                            input_slices[ch][frame];
                    }
                }
                self.record_unity_passthrough_from_scratch(needed_samples);
                for ch in 0..self.num_channels {
                    output_slices[ch][..input_frames]
                        .copy_from_slice(&input_slices[ch][..input_frames]);
                }
                self.input_timeline_frames += input_frames as f64;
                self.reset_ratio_motion_history();
                self.active_ratio = 1.0;
                self.advance_ratio_motion_freeze();
                self.observe_governor_block(start.elapsed());
                return Ok((input_frames, input_frames));
            }

            self.prime_tonal_state_from_unity_history_if_needed(
                self.current_kernel_ratio(self.config.kernel_frames),
            )?;
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

            self.observe_governor_block(start.elapsed());
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
            self.prepare_runtime_policy();
            self.assert_can_process_new_input()?;

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
                self.rearm_unity_passthrough();
                self.record_unity_passthrough_samples(input);
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
                self.reset_ratio_motion_history();
                self.active_ratio = 1.0;
                self.advance_ratio_motion_freeze();
                self.observe_governor_block(start.elapsed());
                return Ok(());
            }

            self.prime_tonal_state_from_unity_history_if_needed(
                self.current_kernel_ratio(self.config.kernel_frames),
            )?;
            self.push_input_with_overload_policy(input)?;
            if self.input_ring.len() >= self.kernel_samples {
                self.render_fixed_kernel()?;
            }
            let max_emit = self.max_output_samples_per_callback();
            let _ = self.drain_pending_to_output(output, max_emit)?;

            self.observe_governor_block(start.elapsed());
            Ok(())
        })();
        self.record_rt_result(result)
    }

    /// Processes one callback block into a fixed interleaved output buffer.
    ///
    /// Returns the number of interleaved samples written to `output`.
    /// Only full frames are written; any trailing partial-frame capacity in
    /// `output` is ignored.
    pub fn process_block_into(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<usize, StretchError> {
        let result = (|| -> Result<usize, StretchError> {
            let start = Instant::now();
            self.prepare_runtime_policy();
            self.assert_can_process_new_input()?;

            if input.len() != self.block_samples {
                return Err(StretchError::InvalidFormat(format!(
                    "process_block_into requires exactly {} samples ({} frames x {} channels), got {}",
                    self.block_samples,
                    self.config.block_frames,
                    self.num_channels,
                    input.len()
                )));
            }
            if input.iter().any(|s| !s.is_finite()) {
                return Err(StretchError::NonFiniteInput);
            }

            let output_samples_capacity = output
                .len()
                .saturating_div(self.num_channels.max(1))
                .saturating_mul(self.num_channels.max(1));
            if self.unity_passthrough_eligible(self.config.block_frames) {
                if output_samples_capacity < input.len() {
                    return Err(StretchError::BufferOverflow {
                        buffer: "rt_process_block_output",
                        requested: input.len(),
                        available: output_samples_capacity,
                    });
                }
                self.rearm_unity_passthrough();
                self.record_unity_passthrough_samples(input);
                output[..input.len()].copy_from_slice(input);
                self.input_timeline_frames += self.config.block_frames as f64;
                self.reset_ratio_motion_history();
                self.active_ratio = 1.0;
                self.advance_ratio_motion_freeze();
                self.observe_governor_block(start.elapsed());
                return Ok(input.len());
            }

            self.prime_tonal_state_from_unity_history_if_needed(
                self.current_kernel_ratio(self.config.kernel_frames),
            )?;
            self.push_input_with_overload_policy(input)?;
            if self.input_ring.len() >= self.kernel_samples {
                self.render_fixed_kernel()?;
            }
            let max_emit = self
                .max_output_samples_per_callback()
                .min(output_samples_capacity);
            let written = self.drain_pending_to_buffer(output, max_emit)?;

            self.observe_governor_block(start.elapsed());
            Ok(written)
        })();
        self.record_rt_result(result)
    }

    /// Flushes all pending RT state.
    pub fn flush(&mut self, output: &mut Vec<f32>) -> Result<(), StretchError> {
        let result = (|| -> Result<(), StretchError> {
            self.prepare_runtime_policy();
            self.prepare_pending_flush_output()?;
            let _ = self.drain_pending_to_output(output, usize::MAX)?;
            if self.pending_output.is_empty() {
                self.finish_flush_drain();
            }
            Ok(())
        })();
        self.record_rt_result(result)
    }

    /// Flushes pending RT state into a fixed interleaved output buffer.
    ///
    /// Returns the number of interleaved samples written to `output`.
    /// Only full frames are written; any trailing partial-frame capacity in
    /// `output` is ignored.
    ///
    /// If more flushed output remains than fits in `output`, subsequent calls
    /// continue draining the already-flushed tail until this method returns `0`.
    /// No new input may be processed until the flush output has been fully
    /// drained.
    pub fn flush_into(&mut self, output: &mut [f32]) -> Result<usize, StretchError> {
        let result = (|| -> Result<usize, StretchError> {
            self.prepare_runtime_policy();
            self.prepare_pending_flush_output()?;

            if self.pending_output.is_empty() {
                self.finish_flush_drain();
                return Ok(0);
            }

            let aligned_capacity = output
                .len()
                .saturating_div(self.num_channels.max(1))
                .saturating_mul(self.num_channels.max(1));
            if aligned_capacity == 0 {
                return Err(StretchError::BufferOverflow {
                    buffer: "rt_flush_output",
                    requested: self.num_channels.max(1),
                    available: aligned_capacity,
                });
            }

            let written = self.drain_pending_to_buffer(output, usize::MAX)?;
            if self.pending_output.is_empty() {
                self.finish_flush_drain();
            }
            Ok(written)
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

    #[inline]
    fn assert_can_process_new_input(&self) -> Result<(), StretchError> {
        if self.flush_drain_pending {
            return Err(StretchError::InvalidState(
                "rt flush output must be fully drained before new input",
            ));
        }
        Ok(())
    }

    fn prepare_pending_flush_output(&mut self) -> Result<(), StretchError> {
        if self.flush_drain_pending {
            return Ok(());
        }

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
        self.input_ring.clear();
        for out in &mut self.tonal_output {
            out.clear();
        }
        for out in &mut self.transient_output {
            out.clear();
        }
        for vocoder in &mut self.vocoders {
            vocoder.reset_phase_state();
        }
        self.flush_drain_pending = true;
        Ok(())
    }

    fn finish_flush_drain(&mut self) {
        self.pending_output.clear();
        self.flush_drain_pending = false;
        self.input_timeline_frames = 0.0;
    }

    fn record_unity_passthrough_samples(&mut self, input: &[f32]) {
        let capacity = self.unity_history.capacity();
        if capacity == 0 || input.is_empty() {
            return;
        }

        let to_store = if input.len() > capacity {
            &input[input.len() - capacity..]
        } else {
            input
        };
        let overflow = to_store
            .len()
            .saturating_sub(self.unity_history.available());
        if overflow > 0 {
            self.unity_history
                .discard(overflow.min(self.unity_history.len()));
        }
        let pushed = self.unity_history.push_slice(to_store);
        debug_assert_eq!(pushed, to_store.len());
    }

    fn record_unity_passthrough_from_scratch(&mut self, samples: usize) {
        let capacity = self.unity_history.capacity();
        if capacity == 0 || samples == 0 {
            return;
        }

        let available = samples.min(self.interleaved_scratch.len());
        let keep = available.min(capacity);
        let start = available.saturating_sub(keep);
        let overflow = keep.saturating_sub(self.unity_history.available());
        if overflow > 0 {
            self.unity_history
                .discard(overflow.min(self.unity_history.len()));
        }

        let pushed = self
            .unity_history
            .push_slice(&self.interleaved_scratch[start..start + keep]);
        debug_assert_eq!(pushed, keep);
    }

    fn prime_tonal_state_from_unity_history_if_needed(
        &mut self,
        ratio: f64,
    ) -> Result<(), StretchError> {
        if (ratio - 1.0).abs() <= UNITY_BYPASS_RATIO_EPS
            || (self.active_ratio - 1.0).abs() > UNITY_BYPASS_RATIO_EPS
            || !self.input_ring.is_empty()
            || !self.pending_output.is_empty()
        {
            return Ok(());
        }

        let history_samples = self.unity_history.len();
        let min_history_samples = self
            .config
            .params
            .fft_size
            .saturating_mul(self.num_channels);
        if history_samples < min_history_samples {
            return Ok(());
        }
        if history_samples > self.interleaved_scratch.len() {
            return Err(StretchError::BufferOverflow {
                buffer: "rt_unity_history_scratch",
                requested: history_samples,
                available: self.interleaved_scratch.len(),
            });
        }

        let copied = self
            .unity_history
            .peek_slice(&mut self.interleaved_scratch[..history_samples]);
        if copied != history_samples {
            return Err(StretchError::InvalidState(
                "failed to snapshot unity passthrough history",
            ));
        }

        let history_frames = history_samples / self.num_channels.max(1);
        for ch in 0..self.num_channels {
            if self.channel_input[ch].capacity() < history_frames {
                return Err(StretchError::BufferOverflow {
                    buffer: "rt_channel_input",
                    requested: history_frames,
                    available: self.channel_input[ch].capacity(),
                });
            }
            if self.pv_channel_input[ch].capacity() < history_frames {
                return Err(StretchError::BufferOverflow {
                    buffer: "rt_pv_channel_input",
                    requested: history_frames,
                    available: self.pv_channel_input[ch].capacity(),
                });
            }
            self.channel_input[ch].clear();
            self.pv_channel_input[ch].clear();
        }

        for frame in 0..history_frames {
            let base = frame * self.num_channels;
            for ch in 0..self.num_channels {
                self.channel_input[ch].push(self.interleaved_scratch[base + ch]);
            }
        }

        for vocoder in &mut self.vocoders {
            vocoder.set_stretch_ratio(ratio);
        }

        for ch in 0..self.num_channels {
            for frame in 0..history_frames {
                let (_low, high) =
                    self.sub_bass_crossovers[ch].process_sample(self.channel_input[ch][frame]);
                self.pv_channel_input[ch].push(high);
            }
        }

        // Keep unity-exit warm-starts aligned with the committed-kernel path:
        // transient-heavy regions should be attenuated before they seed the PV.
        let history_start_frame = (self.input_timeline_frames - history_frames as f64).max(0.0);
        self.apply_input_domain_transient_mask(history_frames, history_start_frame);

        for ch in 0..self.num_channels {
            self.tonal_output[ch].clear();
            self.vocoders[ch]
                .process_streaming_into(&self.pv_channel_input[ch], &mut self.tonal_output[ch])?;
            self.tonal_output[ch].clear();
        }

        self.unity_history.clear();
        Ok(())
    }

    fn reset_state_for_unity_passthrough(&mut self) {
        self.input_ring.clear();
        self.pending_output.clear();
        self.unity_history.clear();
        self.flush_drain_pending = false;
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
    fn unity_passthrough_eligible(&self, input_frames: usize) -> bool {
        if input_frames == 0 {
            return false;
        }

        let start = self.input_timeline_frames;
        let end = start + input_frames as f64;
        let base_ratio = self.base_ratio_over_range(start, end);
        if (base_ratio - 1.0).abs() > UNITY_BYPASS_RATIO_EPS {
            return false;
        }
        if !self.local_base_ratio_stays_near_unity(start, end) {
            return false;
        }

        true
    }

    fn local_base_ratio_stays_near_unity(&self, start: f64, end: f64) -> bool {
        if let Some(ratio) = self.constant_ratio_override {
            return (ratio - 1.0).abs() <= UNITY_BYPASS_RATIO_EPS;
        }

        // TimeWarpMap is piecewise-linear, so checking the segment slope at the
        // interval start and every internal anchor covers the whole block.
        if (self.warp_map.local_ratio_at_input(start) - 1.0).abs() > UNITY_BYPASS_RATIO_EPS {
            return false;
        }

        for anchor in self.warp_map.anchors() {
            if anchor.input_frame <= start || anchor.input_frame >= end {
                continue;
            }
            if (self.warp_map.local_ratio_at_input(anchor.input_frame) - 1.0).abs()
                > UNITY_BYPASS_RATIO_EPS
            {
                return false;
            }
        }

        true
    }

    #[inline]
    fn can_unity_passthrough(&self, input_frames: usize, output_frames_capacity: usize) -> bool {
        output_frames_capacity >= input_frames && self.unity_passthrough_eligible(input_frames)
    }

    #[inline]
    fn rearm_unity_passthrough(&mut self) {
        // Re-arm bit-exact passthrough after non-unity runs by dropping any
        // buffered overlap context when the host returns to unity ratio.
        if !self.input_ring.is_empty() || !self.pending_output.is_empty() {
            self.reset_state_for_unity_passthrough();
        }
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
        if self.ratio_motion_freeze_active() {
            self.policy_profile = self.current_profile;
            return;
        }
        if !self.auto_profile_switching {
            self.policy_profile = self.target_profile;
            return;
        }
        if self.post_ratio_motion_profile_hold_blocks_left > 0 {
            self.policy_profile = self.current_profile;
            self.post_ratio_motion_profile_hold_blocks_left = self
                .post_ratio_motion_profile_hold_blocks_left
                .saturating_sub(1);
            return;
        }
        if self.current_profile != self.target_profile {
            // Keep the committed auto-profile target stable until the
            // crossfade settles instead of reversing direction mid-transition.
            self.policy_profile = self.target_profile;
            self.profile_candidate = self.target_profile;
            self.profile_candidate_streak = 0;
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
        if self.ratio_motion_freeze_active() {
            self.profile_transition_blocks_left = 0;
            return;
        }
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
        // Commit ratio-motion holds only when a kernel actually renders so
        // preview-only callbacks cannot mutate live profile state.
        self.engage_ratio_motion_freeze_if_needed(ratio);
        if (ratio - self.active_ratio).abs() > RATIO_SNAP_EPS {
            for vocoder in &mut self.vocoders {
                vocoder.set_stretch_ratio(ratio);
            }
            for stretcher in &mut self.transient_stretchers {
                stretcher.set_stretch_ratio(ratio);
            }
            self.active_ratio = ratio;
        }
        self.advance_runtime_policy_for_committed_kernel();

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
            self.advance_ratio_motion_freeze();
            return Ok(());
        }

        self.build_transient_mask_from_hints(min_output_len, ratio, kernel_start_frame);

        let weights = self.effective_lane_weights();
        self.mix_into_pending(min_output_len, weights)?;
        self.consume_kernel_input();
        self.advance_ratio_motion_freeze();
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

    fn drain_pending_to_buffer(
        &mut self,
        output: &mut [f32],
        max_samples: usize,
    ) -> Result<usize, StretchError> {
        let aligned_capacity = output
            .len()
            .min(max_samples)
            .saturating_div(self.num_channels.max(1))
            .saturating_mul(self.num_channels.max(1));
        let to_emit = self.pending_output.len().min(aligned_capacity);
        if to_emit == 0 {
            return Ok(0);
        }

        for sample in &mut output[..to_emit] {
            let Some(value) = self.pending_output.pop() else {
                return Err(StretchError::InvalidState(
                    "rt pending drain to buffer made zero progress",
                ));
            };
            *sample = value;
        }
        Ok(to_emit)
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
    use crate::dual_plane::warp_map::{TimeWarpMap, WarpAnchor};
    use crate::error::StretchError;
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
    fn process_block_into_unity_ratio_is_bit_exact_passthrough() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut cfg = RtConfig::new(params, 256);
        cfg.latency_profile = LatencyProfile::Scratch;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = stereo_sine_block(256, 48_000, 330.0, 0.37);
        let mut out = vec![0.0f32; input.len()];
        let written = rt.process_block_into(&input, &mut out).unwrap();

        assert_eq!(written, input.len());
        assert_eq!(out, input);
        assert!(rt.input_ring.is_empty());
        assert!(rt.pending_output.is_empty());
    }

    #[test]
    fn process_block_into_rejects_average_unity_piecewise_warp_for_passthrough() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut cfg = RtConfig::new(params, 256);
        cfg.latency_profile = LatencyProfile::Scratch;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let warp_map = Arc::new(
            TimeWarpMap::from_anchors(vec![
                WarpAnchor::new(0.0, 0.0).unwrap(),
                WarpAnchor::new(128.0, 123.52).unwrap(),
                WarpAnchor::new(256.0, 256.0).unwrap(),
            ])
            .unwrap(),
        );
        rt.set_warp_map_snapshot(warp_map);

        assert!(
            !rt.unity_passthrough_eligible(256),
            "piecewise warp blocks that only average to unity must stay on the non-unity path"
        );

        let input = stereo_sine_block(256, 48_000, 330.0, 0.37);
        let mut out = vec![0.0f32; input.len()];
        let written = rt.process_block_into(&input, &mut out).unwrap();

        assert_eq!(
            written, 0,
            "average-unity piecewise warp should buffer for deterministic processing instead of bypassing"
        );
        assert_eq!(rt.input_ring.len(), input.len());
        assert!(rt.pending_output.is_empty());
    }

    #[test]
    fn process_block_into_unity_ratio_requires_full_block_capacity() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut cfg = RtConfig::new(params, 256);
        cfg.latency_profile = LatencyProfile::Scratch;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = stereo_sine_block(256, 48_000, 330.0, 0.37);
        let mut out = vec![0.0f32; input.len() - 2];
        let err = rt.process_block_into(&input, &mut out).unwrap_err();

        assert!(matches!(
            err,
            StretchError::BufferOverflow {
                buffer: "rt_process_block_output",
                requested,
                available,
            } if requested == input.len() && available == out.len()
        ));
        assert!(rt.input_ring.is_empty());
        assert!(rt.pending_output.is_empty());
    }

    #[test]
    fn process_block_into_allows_non_unity_processing_without_vec_output() {
        let params = StretchParams::new(1.35)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut cfg = RtConfig::new(params, 256);
        cfg.latency_profile = LatencyProfile::Scratch;
        let max_output_frames = cfg.max_output_frames_per_kernel();
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let block = stereo_sine_block(256, 48_000, 220.0, 0.0);
        let mut no_output = [0.0f32; 0];
        for i in 0..8 {
            let phase = i as f32 * 0.17;
            let input = stereo_sine_block(256, 48_000, 220.0, phase);
            assert_eq!(rt.process_block_into(&input, &mut no_output).unwrap(), 0);
        }

        assert_eq!(rt.runtime_telemetry().process_error_count, 0);
        let pending_before = rt.delay_telemetry().buffered_output_frames;
        assert!(
            pending_before > 0,
            "expected pending RT output after warmup"
        );

        let mut out = vec![0.0f32; block.len() * 4];
        let written = rt.process_block_into(&block, &mut out).unwrap();

        assert!(written > 0);
        assert_eq!(written % rt.num_channels(), 0);
        assert!(written <= out.len());
        assert!(
            rt.delay_telemetry().buffered_output_frames <= pending_before + max_output_frames,
            "draining into a fixed buffer should not amplify pending output"
        );
    }

    #[test]
    fn flush_into_matches_vec_flush_when_drained_in_chunks() {
        let params = StretchParams::new(1.35)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut cfg = RtConfig::new(params, 256);
        cfg.latency_profile = LatencyProfile::Scratch;
        let mut rt_vec = RtProcessor::prepare(cfg.clone()).unwrap();
        let mut rt_fixed = RtProcessor::prepare(cfg).unwrap();

        let mut no_output = [0.0f32; 0];
        for i in 0..8 {
            let input = stereo_sine_block(256, 48_000, 220.0, i as f32 * 0.17);
            assert_eq!(
                rt_vec.process_block_into(&input, &mut no_output).unwrap(),
                0
            );
            assert_eq!(
                rt_fixed.process_block_into(&input, &mut no_output).unwrap(),
                0
            );
        }

        let mut expected = Vec::with_capacity(16_384);
        rt_vec.flush(&mut expected).unwrap();
        assert!(
            !expected.is_empty(),
            "flush should emit deferred tail audio"
        );

        let mut actual = Vec::with_capacity(expected.len());
        let mut chunk = vec![0.0f32; 192];
        let mut iterations = 0usize;
        loop {
            let written = rt_fixed.flush_into(&mut chunk).unwrap();
            if written == 0 {
                break;
            }
            iterations += 1;
            actual.extend_from_slice(&chunk[..written]);
        }

        assert!(
            iterations > 1,
            "fixed-buffer flush should support chunked draining"
        );
        assert_eq!(actual, expected);
        assert_eq!(rt_fixed.runtime_telemetry().process_error_count, 0);
        assert_eq!(rt_fixed.delay_telemetry().buffered_input_frames, 0);
        assert_eq!(rt_fixed.delay_telemetry().buffered_output_frames, 0);
        assert!(rt_fixed.input_ring.is_empty());
        assert!(rt_fixed.pending_output.is_empty());
    }

    #[test]
    fn flush_into_requires_pending_output_to_drain_before_new_input() {
        let params = StretchParams::new(1.35)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut cfg = RtConfig::new(params, 256);
        cfg.latency_profile = LatencyProfile::Scratch;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let mut no_output = [0.0f32; 0];
        for i in 0..8 {
            let input = stereo_sine_block(256, 48_000, 220.0, i as f32 * 0.11);
            assert_eq!(rt.process_block_into(&input, &mut no_output).unwrap(), 0);
        }

        let mut chunk = vec![0.0f32; 192];
        let first_written = rt.flush_into(&mut chunk).unwrap();
        assert!(first_written > 0);
        assert!(
            !rt.pending_output.is_empty(),
            "small fixed buffer should leave flushed tail queued"
        );

        let input = stereo_sine_block(256, 48_000, 330.0, 0.37);
        let err = rt.process_block_into(&input, &mut no_output).unwrap_err();
        assert_eq!(
            err,
            StretchError::InvalidState("rt flush output must be fully drained before new input")
        );
        assert_eq!(rt.runtime_telemetry().process_error_count, 1);

        loop {
            if rt.flush_into(&mut chunk).unwrap() == 0 {
                break;
            }
        }

        assert!(rt.process_block_into(&input, &mut no_output).is_ok());
        assert_eq!(rt.runtime_telemetry().process_error_count, 1);
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

    #[test]
    fn non_unity_entry_primes_tonal_history_after_unity_passthrough() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 32);
        cfg.latency_profile = LatencyProfile::Scratch;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let mut unity_output = [0.0f32; 32];
        for block_idx in 0..4 {
            let input: Vec<f32> = (0..32)
                .map(|i| {
                    let phase = (block_idx * 32 + i) as f32 * 0.031;
                    phase.sin() * 0.4
                })
                .collect();
            let written = rt.process_block_into(&input, &mut unity_output).unwrap();
            assert_eq!(written, input.len());
            assert_eq!(&unity_output[..written], &input[..]);
        }

        assert_eq!(rt.unity_history.len(), rt.kernel_samples);

        rt.set_constant_ratio(1.04);
        let input: Vec<f32> = (0..32)
            .map(|i| ((128 + i) as f32 * 0.031).sin() * 0.4)
            .collect();
        let written = rt.process_block_into(&input, &mut []).unwrap();
        assert_eq!(
            written, 0,
            "single warm-start block should not render a full kernel"
        );
        assert_eq!(rt.input_ring.len(), input.len());
        assert!(rt.unity_history.is_empty());

        let tonal_tail = rt.vocoders[0].flush_streaming().unwrap();
        assert!(
            !tonal_tail.is_empty(),
            "unity-exit warm start should leave tonal overlap state primed"
        );
        assert!(
            tonal_tail.iter().all(|sample| sample.is_finite()),
            "primed tonal tail must remain finite"
        );
    }

    #[test]
    fn non_unity_entry_applies_transient_mask_when_priming_unity_history() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 32);
        cfg.latency_profile = LatencyProfile::Scratch;

        let mut baseline = RtProcessor::prepare(cfg.clone()).unwrap();
        let mut masked = RtProcessor::prepare(cfg).unwrap();
        let mut unity_output = [0.0f32; 32];

        for block_idx in 0..4 {
            let input: Vec<f32> = (0..32)
                .map(|i| {
                    let phase = (block_idx * 32 + i) as f32 * 0.031;
                    phase.sin() * 0.4
                })
                .collect();
            let baseline_written = baseline
                .process_block_into(&input, &mut unity_output)
                .unwrap();
            let masked_written = masked
                .process_block_into(&input, &mut unity_output)
                .unwrap();
            assert_eq!(baseline_written, input.len());
            assert_eq!(masked_written, input.len());
        }

        masked.set_hint_snapshot(Arc::new(RenderHints {
            at_input_frame: 0,
            transient_mask: vec![1.0; masked.config.kernel_frames],
            ..RenderHints::default()
        }));

        let input: Vec<f32> = (0..32)
            .map(|i| ((128 + i) as f32 * 0.031).sin() * 0.4)
            .collect();
        baseline.set_constant_ratio(1.04);
        masked.set_constant_ratio(1.04);
        assert_eq!(baseline.process_block_into(&input, &mut []).unwrap(), 0);
        assert_eq!(masked.process_block_into(&input, &mut []).unwrap(), 0);

        let baseline_tail = baseline.vocoders[0].flush_streaming().unwrap();
        let masked_tail = masked.vocoders[0].flush_streaming().unwrap();
        assert!(
            !baseline_tail.is_empty() && !masked_tail.is_empty(),
            "unity-exit warm start should leave a tonal tail to compare"
        );
        assert!(
            baseline_tail.iter().all(|sample| sample.is_finite())
                && masked_tail.iter().all(|sample| sample.is_finite()),
            "warm-start tonal tails must remain finite"
        );

        let baseline_mean_abs = baseline_tail.iter().map(|sample| sample.abs()).sum::<f32>()
            / baseline_tail.len() as f32;
        let masked_mean_abs =
            masked_tail.iter().map(|sample| sample.abs()).sum::<f32>() / masked_tail.len() as f32;
        assert!(
            masked_mean_abs <= baseline_mean_abs * 0.05 + 1e-6,
            "transient-masked unity-history priming should strongly suppress masked tonal carryover (baseline_mean_abs={baseline_mean_abs:.6}, masked_mean_abs={masked_mean_abs:.6})"
        );
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
        cfg.ratio_motion_freeze_blocks = 0;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];

        // Render-favoring policy: near-unity ratio + strong tonal confidence.
        // Keep the ratio slightly non-unity so the test exercises committed
        // kernels instead of the bit-exact unity passthrough path.
        rt.set_constant_ratio(1.01);
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
        let render_transition = rt.profile_telemetry();
        for _ in 0..=render_transition.transition_blocks_left {
            if rt.profile_telemetry().current_profile == LatencyProfile::Render {
                break;
            }
            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
        }
        assert_eq!(
            rt.profile_telemetry().current_profile,
            LatencyProfile::Render,
            "render-favoring policy should eventually settle the active profile onto render"
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

    #[test]
    fn ratio_motion_freeze_holds_auto_profile_churn_for_configured_kernels() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.latency_profile = LatencyProfile::Render;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 1;
        cfg.ratio_motion_freeze_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];

        rt.set_constant_ratio(1.70);
        rt.set_hint_snapshot(Arc::new(RenderHints {
            transient_confidence: 0.90,
            tonal_confidence: 0.10,
            noise_confidence: 0.20,
            ..RenderHints::default()
        }));

        for hold_idx in 0..2 {
            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
            let telemetry = rt.profile_telemetry();
            assert_eq!(
                telemetry.target_profile,
                LatencyProfile::Render,
                "ratio-motion freeze should hold the current profile during kernel {hold_idx}"
            );
            assert_eq!(
                telemetry.current_profile,
                LatencyProfile::Render,
                "current profile should stay fixed while ratio-motion freeze is active"
            );
            assert_eq!(
                telemetry.target_tier,
                QualityTier::Q4,
                "ratio-motion freeze should also hold tier retargeting"
            );
        }

        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);
        let telemetry = rt.profile_telemetry();
        assert_eq!(
            telemetry.target_profile,
            LatencyProfile::Render,
            "the first committed kernel after the ratio-motion freeze should still hold the current render profile"
        );

        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);
        let telemetry = rt.profile_telemetry();
        assert_eq!(
            telemetry.target_profile,
            LatencyProfile::Scratch,
            "auto profile switching should resume on the second committed kernel after the ratio-motion freeze expires"
        );
    }

    #[test]
    fn ratio_motion_freeze_biases_mix_profile_to_scratch_under_fast_modulation() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.latency_profile = LatencyProfile::Mix;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 1;
        cfg.ratio_motion_freeze_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];

        rt.set_constant_ratio(1.035);
        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);

        let telemetry = rt.profile_telemetry();
        assert_eq!(
            telemetry.current_profile,
            LatencyProfile::Scratch,
            "fast ratio motion should move auto-mode mix kernels onto the scratch profile"
        );
        assert_eq!(
            telemetry.target_profile,
            LatencyProfile::Scratch,
            "fast ratio motion should hold the scratch target until modulation settles"
        );
        assert_eq!(
            telemetry.policy_profile,
            LatencyProfile::Scratch,
            "policy telemetry should expose the modulation-biased scratch hold"
        );
        assert_eq!(
            telemetry.target_tier,
            QualityTier::Q1,
            "scratch-biased modulation holds should retarget the scratch tier ladder"
        );
        assert_eq!(
            telemetry.current_tier,
            QualityTier::Q1,
            "scratch-biased modulation holds should snap the active tier onto the scratch ladder"
        );
        assert_weights_close(rt.blend_weights, QualityTier::Q1.lane_weights(), 1e-6);
    }

    #[test]
    fn ratio_motion_freeze_rearms_under_callback_rate_modulation() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.latency_profile = LatencyProfile::Render;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 1;
        cfg.ratio_motion_freeze_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];
        let render_hints = Arc::new(RenderHints {
            transient_confidence: 0.05,
            tonal_confidence: 0.95,
            noise_confidence: 0.05,
            ..RenderHints::default()
        });
        let scratch_hints = Arc::new(RenderHints {
            transient_confidence: 0.90,
            tonal_confidence: 0.10,
            noise_confidence: 0.20,
            ..RenderHints::default()
        });

        for (step_idx, (ratio, hints)) in [
            (1.70, Arc::clone(&scratch_hints)),
            (1.00, Arc::clone(&render_hints)),
            (1.68, Arc::clone(&scratch_hints)),
            (1.02, Arc::clone(&render_hints)),
        ]
        .into_iter()
        .enumerate()
        {
            rt.set_constant_ratio(ratio);
            rt.set_hint_snapshot(hints);

            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
            let telemetry = rt.profile_telemetry();
            assert_eq!(
                telemetry.target_profile,
                LatencyProfile::Render,
                "callback-rate modulation should keep target profile frozen during step {step_idx}"
            );
            assert_eq!(
                telemetry.current_profile,
                LatencyProfile::Render,
                "callback-rate modulation should not flap the active profile during step {step_idx}"
            );
        }

        rt.set_constant_ratio(1.70);
        rt.set_hint_snapshot(Arc::clone(&scratch_hints));
        for _ in 0..4 {
            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
        }
        assert_eq!(
            rt.profile_telemetry().target_profile,
            LatencyProfile::Scratch,
            "auto profile switching should resume once the repeated modulation stops"
        );
    }

    #[test]
    fn ratio_motion_freeze_holds_auto_profile_steady_for_short_interval_plateaus() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.latency_profile = LatencyProfile::Render;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 1;
        cfg.ratio_motion_freeze_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];
        let render_hints = Arc::new(RenderHints {
            transient_confidence: 0.05,
            tonal_confidence: 0.95,
            noise_confidence: 0.05,
            ..RenderHints::default()
        });
        let scratch_hints = Arc::new(RenderHints {
            transient_confidence: 0.90,
            tonal_confidence: 0.10,
            noise_confidence: 0.20,
            ..RenderHints::default()
        });

        for (step_idx, (ratio, hints)) in [
            (1.035, Arc::clone(&scratch_hints)),
            (1.035, Arc::clone(&scratch_hints)),
            (0.975, Arc::clone(&render_hints)),
            (0.975, Arc::clone(&render_hints)),
            (1.025, Arc::clone(&scratch_hints)),
            (1.025, Arc::clone(&scratch_hints)),
            (0.965, Arc::clone(&render_hints)),
            (0.965, Arc::clone(&render_hints)),
        ]
        .into_iter()
        .enumerate()
        {
            rt.set_constant_ratio(ratio);
            rt.set_hint_snapshot(hints);

            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
            let telemetry = rt.profile_telemetry();
            assert_eq!(
                telemetry.target_profile,
                LatencyProfile::Render,
                "two-callback plateau modulation should keep the target profile fixed during step {step_idx}"
            );
            assert_eq!(
                telemetry.current_profile,
                LatencyProfile::Render,
                "two-callback plateau modulation should not flap the active profile during step {step_idx}"
            );
            assert_eq!(
                telemetry.policy_profile,
                LatencyProfile::Render,
                "policy telemetry should remain on the held render profile during step {step_idx}"
            );
        }

        rt.set_constant_ratio(1.035);
        rt.set_hint_snapshot(Arc::clone(&scratch_hints));
        for _ in 0..4 {
            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
        }
        assert_eq!(
            rt.profile_telemetry().target_profile,
            LatencyProfile::Scratch,
            "auto profile switching should resume once the short-interval plateau modulation stops"
        );
    }

    #[test]
    fn ratio_motion_freeze_arms_for_cumulative_subthreshold_modulation() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.latency_profile = LatencyProfile::Render;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 4;
        cfg.ratio_motion_freeze_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];
        let scratch_hints = Arc::new(RenderHints {
            transient_confidence: 0.90,
            tonal_confidence: 0.10,
            noise_confidence: 0.20,
            ..RenderHints::default()
        });

        for (step_idx, ratio) in [1.0003, 1.0006, 1.0009, 1.0012].into_iter().enumerate() {
            rt.set_constant_ratio(ratio);
            rt.set_hint_snapshot(Arc::clone(&scratch_hints));

            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
            let telemetry = rt.profile_telemetry();
            assert_eq!(
                telemetry.current_profile,
                LatencyProfile::Render,
                "sub-threshold modulation should not flap the committed render profile during step {step_idx}"
            );
            assert_eq!(
                telemetry.target_profile,
                LatencyProfile::Render,
                "cumulative sub-threshold modulation should keep the render target held during step {step_idx}"
            );
        }

        assert_eq!(
            rt.profile_candidate,
            LatencyProfile::Scratch,
            "scratch-biased hints should still accumulate the pending scratch candidate"
        );
        assert_eq!(
            rt.profile_candidate_streak, 3,
            "the modulation hold should pause the scratch hysteresis streak before the fourth sub-threshold step can commit"
        );
        assert_eq!(
            rt.ratio_motion_freeze_blocks_left, 1,
            "the fourth sub-threshold step should arm and then advance the configured ratio-motion hold"
        );
        assert_eq!(
            rt.profile_telemetry().policy_profile,
            LatencyProfile::Render,
            "policy telemetry should stay on the held render profile once the cumulative motion hold arms"
        );
    }

    #[test]
    fn ratio_motion_freeze_preview_callbacks_do_not_mutate_mix_profile_before_first_kernel() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 32);
        cfg.latency_profile = LatencyProfile::Mix;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 1;
        cfg.ratio_motion_freeze_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 32];
        let mut output = [0.0f32; 64];
        let input_refs = [&input[..]];

        rt.set_constant_ratio(1.035);

        for preview_idx in 0..3 {
            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
            let telemetry = rt.profile_telemetry();
            assert_eq!(
                telemetry.current_profile,
                LatencyProfile::Mix,
                "preview-only callback {preview_idx} must not snap the committed mix profile to scratch before any kernel renders"
            );
            assert_eq!(
                telemetry.target_profile,
                LatencyProfile::Mix,
                "preview-only callback {preview_idx} must not retarget the mix profile hold before any kernel renders"
            );
            assert_eq!(
                telemetry.policy_profile,
                LatencyProfile::Mix,
                "preview-only callback {preview_idx} must leave policy telemetry on mix before any kernel renders"
            );
            assert_eq!(
                rt.ratio_motion_freeze_blocks_left, 0,
                "preview-only callback {preview_idx} must not arm the ratio-motion freeze before the first committed kernel"
            );
        }

        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);
        let committed = rt.profile_telemetry();
        assert_eq!(
            committed.current_profile,
            LatencyProfile::Scratch,
            "the first committed fast-modulation kernel should still bias mix mode onto scratch"
        );
        assert_eq!(
            committed.target_profile,
            LatencyProfile::Scratch,
            "the first committed fast-modulation kernel should still hold scratch as the target"
        );
        assert_eq!(
            committed.policy_profile,
            LatencyProfile::Scratch,
            "policy telemetry should reflect the scratch-biased hold once a kernel commits"
        );
        assert_eq!(
            rt.ratio_motion_freeze_blocks_left, 1,
            "the first committed kernel should arm and then advance the configured freeze hold"
        );
    }

    #[test]
    fn ratio_motion_freeze_preview_does_not_rearm_before_next_kernel() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 32);
        cfg.latency_profile = LatencyProfile::Render;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 1;
        cfg.ratio_motion_freeze_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 32];
        let mut output = [0.0f32; 64];
        let input_refs = [&input[..]];
        let scratch_hints = Arc::new(RenderHints {
            transient_confidence: 0.90,
            tonal_confidence: 0.10,
            noise_confidence: 0.20,
            ..RenderHints::default()
        });

        rt.set_constant_ratio(1.70);
        rt.set_hint_snapshot(scratch_hints);

        for _ in 0..4 {
            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
        }
        assert_eq!(
            rt.ratio_motion_freeze_blocks_left, 1,
            "first committed kernel should advance the configured freeze hold by exactly one block"
        );

        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);
        assert_eq!(
            rt.ratio_motion_freeze_blocks_left, 1,
            "callback previews without another committed kernel must not rearm the freeze hold"
        );
        assert_eq!(
            rt.profile_telemetry().target_profile,
            LatencyProfile::Render,
            "non-kernel preview callbacks should stay on the held render profile"
        );
    }

    #[test]
    fn ratio_motion_freeze_pauses_inflight_hysteresis_instead_of_restarting_it() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.latency_profile = LatencyProfile::Render;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 4;
        cfg.ratio_motion_freeze_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];
        let scratch_hints = Arc::new(RenderHints {
            transient_confidence: 0.90,
            tonal_confidence: 0.10,
            noise_confidence: 0.20,
            ..RenderHints::default()
        });

        rt.set_constant_ratio(1.0005);
        rt.set_hint_snapshot(Arc::clone(&scratch_hints));
        for build_idx in 0..2 {
            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
            assert_eq!(
                rt.profile_candidate,
                LatencyProfile::Scratch,
                "scratch-biased hints should accumulate a scratch candidate before the freeze at step {build_idx}"
            );
        }
        assert_eq!(
            rt.profile_candidate_streak, 2,
            "pre-freeze scratch evidence should accumulate toward the configured hysteresis threshold"
        );
        assert_eq!(
            rt.profile_telemetry().target_profile,
            LatencyProfile::Render,
            "partial hysteresis progress should not retarget the profile before the threshold is met"
        );

        rt.set_constant_ratio(1.0015);
        for freeze_idx in 0..2 {
            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
            assert_eq!(
                rt.profile_candidate,
                LatencyProfile::Scratch,
                "ratio-motion freeze should preserve the in-flight scratch candidate during held step {freeze_idx}"
            );
            assert_eq!(
                rt.profile_candidate_streak, 2,
                "ratio-motion freeze should pause, not reset, the accumulated hysteresis streak during held step {freeze_idx}"
            );
            assert_eq!(
                rt.profile_telemetry().target_profile,
                LatencyProfile::Render,
                "the held render profile should stay committed until the freeze fully expires"
            );
        }

        for resume_idx in 0..3 {
            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
            if resume_idx == 0 {
                assert_eq!(
                    rt.profile_telemetry().target_profile,
                    LatencyProfile::Render,
                    "the first post-freeze callback should keep the render hold in place without restarting or advancing hysteresis"
                );
                assert_eq!(
                    rt.profile_candidate_streak, 2,
                    "the first post-freeze callback should preserve the pre-freeze scratch streak without advancing it"
                );
            } else if resume_idx == 1 {
                assert_eq!(
                    rt.profile_telemetry().target_profile,
                    LatencyProfile::Render,
                    "the second post-freeze callback should resume the preserved hysteresis streak without committing early"
                );
                assert_eq!(
                    rt.profile_candidate_streak, 3,
                    "the second post-freeze callback should resume the preserved scratch streak"
                );
            }
        }

        assert_eq!(
            rt.profile_telemetry().target_profile,
            LatencyProfile::Scratch,
            "preserved hysteresis progress should let scratch retarget after only the remaining callbacks once the post-freeze hold clears"
        );
    }

    #[test]
    fn post_freeze_profile_hold_blocks_first_calm_kernel_from_retargeting_away_from_scratch() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.latency_profile = LatencyProfile::Mix;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 1;
        cfg.ratio_motion_freeze_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];
        let render_hints = Arc::new(RenderHints {
            transient_confidence: 0.05,
            tonal_confidence: 0.95,
            noise_confidence: 0.05,
            ..RenderHints::default()
        });

        rt.set_constant_ratio(1.035);
        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);
        assert_eq!(
            rt.profile_telemetry().current_profile,
            LatencyProfile::Scratch,
            "fast motion should still bias the auto mix path onto scratch"
        );

        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);
        assert_eq!(
            rt.ratio_motion_freeze_blocks_left, 0,
            "the second kernel should consume the configured ratio-motion freeze hold"
        );

        rt.set_hint_snapshot(Arc::clone(&render_hints));

        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);
        let first_calm = rt.profile_telemetry();
        assert_eq!(
            first_calm.current_profile,
            LatencyProfile::Scratch,
            "the first calm kernel after a fast-modulation burst should keep scratch active"
        );
        assert_eq!(
            first_calm.target_profile,
            LatencyProfile::Scratch,
            "the first calm kernel after a fast-modulation burst should not immediately retarget away from scratch"
        );

        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);
        let second_calm = rt.profile_telemetry();
        assert_eq!(
            second_calm.current_profile,
            LatencyProfile::Scratch,
            "retargeting away from scratch should still occur via a transition"
        );
        assert_eq!(
            second_calm.target_profile,
            LatencyProfile::Render,
            "the second calm kernel should be the first one allowed to retarget away from scratch"
        );
        assert!(
            second_calm.transition_blocks_left > 0,
            "retargeting away from scratch should queue a transition instead of snapping instantly"
        );
    }

    #[test]
    fn ratio_motion_freeze_rearm_from_scratch_cancels_pending_mix_transition() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.latency_profile = LatencyProfile::Mix;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 1;
        cfg.ratio_motion_freeze_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];

        rt.set_constant_ratio(1.035);
        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);
        let scratch_hold = rt.profile_telemetry();
        assert_eq!(scratch_hold.current_profile, LatencyProfile::Scratch);
        assert_eq!(scratch_hold.target_profile, LatencyProfile::Scratch);

        rt.ratio_motion_freeze_blocks_left = 0;
        rt.set_latency_profile_internal(LatencyProfile::Mix);
        rt.advance_profile_transition();
        rt.advance_tier_crossfade();

        let mix_return = rt.profile_telemetry();
        assert_eq!(
            mix_return.current_profile,
            LatencyProfile::Scratch,
            "the scratch profile should still be active while the pending mix transition is in flight"
        );
        assert_eq!(
            mix_return.target_profile,
            LatencyProfile::Mix,
            "test setup should queue a mix retarget while scratch remains active"
        );
        assert_eq!(
            mix_return.target_tier,
            QualityTier::Q2,
            "test setup should also move the target tier back toward mix defaults"
        );
        assert!(
            rt.profile_transition_blocks_left > 0,
            "test setup should leave a profile transition in flight"
        );
        assert!(
            rt.crossfade_blocks_left > 0,
            "test setup should leave a tier crossfade in flight"
        );

        rt.active_ratio = 1.0;
        rt.engage_ratio_motion_freeze_if_needed(1.035);

        let rearmed = rt.profile_telemetry();
        assert_eq!(
            rearmed.current_profile,
            LatencyProfile::Scratch,
            "re-arming the modulation hold should keep scratch as the active profile"
        );
        assert_eq!(
            rearmed.target_profile,
            LatencyProfile::Scratch,
            "re-arming from scratch should cancel the stale mix retarget"
        );
        assert_eq!(
            rearmed.policy_profile,
            LatencyProfile::Scratch,
            "policy telemetry should reflect the renewed scratch hold"
        );
        assert_eq!(
            rearmed.current_tier,
            QualityTier::Q1,
            "re-arming from scratch should keep the active tier on the scratch ladder"
        );
        assert_eq!(
            rearmed.target_tier,
            QualityTier::Q1,
            "re-arming from scratch should cancel the stale mix tier retarget"
        );
        assert_eq!(
            rt.profile_transition_blocks_left, 0,
            "re-arming the modulation hold should clear the pending profile transition"
        );
        assert_eq!(
            rt.crossfade_blocks_left, 0,
            "re-arming the modulation hold should clear the pending tier crossfade"
        );
        assert_weights_close(rt.blend_weights, QualityTier::Q1.lane_weights(), 1e-6);
    }

    #[test]
    fn unity_passthrough_does_not_arm_ratio_motion_freeze_or_bias_mix_profile() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.latency_profile = LatencyProfile::Mix;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 1;
        cfg.ratio_motion_freeze_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];
        let initial = rt.profile_telemetry();
        let initial_weights = rt.blend_weights;

        rt.active_ratio = 1.04;
        rt.set_constant_ratio(1.0);

        let mut output_refs = [&mut output[..]];
        let (consumed, produced) = rt.process(&input_refs, &mut output_refs);
        assert_eq!((consumed, produced), (128, 128));
        assert_eq!(
            rt.ratio_motion_freeze_blocks_left, 0,
            "exact-unity passthrough callbacks must not arm a stale ratio-motion freeze"
        );
        assert_eq!(rt.active_ratio, 1.0);

        let telemetry = rt.profile_telemetry();
        assert_eq!(
            telemetry.current_profile, initial.current_profile,
            "exact-unity passthrough must not bias the committed mix profile"
        );
        assert_eq!(
            telemetry.target_profile, initial.target_profile,
            "exact-unity passthrough must not retarget the profile hold state"
        );
        assert_eq!(
            telemetry.policy_profile, initial.policy_profile,
            "exact-unity passthrough must leave profile policy telemetry unchanged"
        );
        assert_eq!(
            telemetry.current_tier, initial.current_tier,
            "exact-unity passthrough must not retarget the active quality tier"
        );
        assert_eq!(
            telemetry.target_tier, initial.target_tier,
            "exact-unity passthrough must not queue a tier crossfade"
        );
        assert_weights_close(rt.blend_weights, initial_weights, 1e-6);
        assert_weights_close(rt.target_weights, initial_weights, 1e-6);
    }

    #[test]
    fn buffered_unity_kernel_does_not_arm_ratio_motion_freeze_or_bias_mix_profile() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.latency_profile = LatencyProfile::Mix;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 1;
        cfg.ratio_motion_freeze_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let initial = rt.profile_telemetry();
        let initial_weights = rt.blend_weights;
        rt.active_ratio = 1.04;
        rt.set_constant_ratio(1.0);

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 0];
        let input_refs = [&input[..]];
        let mut output_refs = [&mut output[..]];

        let (consumed, produced) = rt.process(&input_refs, &mut output_refs);
        assert_eq!((consumed, produced), (128, 0));
        assert_eq!(
            rt.ratio_motion_freeze_blocks_left, 0,
            "buffered exact-unity kernels must not arm a stale ratio-motion freeze"
        );
        assert_eq!(rt.active_ratio, 1.0);

        let telemetry = rt.profile_telemetry();
        assert_eq!(
            telemetry.current_profile, initial.current_profile,
            "buffered exact-unity kernels must not bias the committed mix profile"
        );
        assert_eq!(
            telemetry.target_profile, initial.target_profile,
            "buffered exact-unity kernels must not retarget the held profile"
        );
        assert_eq!(
            telemetry.policy_profile, initial.policy_profile,
            "buffered exact-unity kernels must leave profile policy telemetry unchanged"
        );
        assert_eq!(
            telemetry.current_tier, initial.current_tier,
            "buffered exact-unity kernels must not retarget the active quality tier"
        );
        assert_eq!(
            telemetry.target_tier, initial.target_tier,
            "buffered exact-unity kernels must not queue a tier crossfade"
        );
        assert_weights_close(rt.blend_weights, initial_weights, 1e-6);
        assert_weights_close(rt.target_weights, initial_weights, 1e-6);
    }

    #[test]
    fn unity_passthrough_advances_existing_ratio_motion_freeze_hold() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.latency_profile = LatencyProfile::Mix;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 1;
        cfg.ratio_motion_freeze_blocks = 2;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];
        let render_hints = Arc::new(RenderHints {
            transient_confidence: 0.05,
            tonal_confidence: 0.95,
            noise_confidence: 0.05,
            ..RenderHints::default()
        });

        rt.set_constant_ratio(1.035);
        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);
        let scratch_hold = rt.profile_telemetry();
        assert_eq!(scratch_hold.current_profile, LatencyProfile::Scratch);
        assert_eq!(scratch_hold.target_profile, LatencyProfile::Scratch);
        assert_eq!(rt.ratio_motion_freeze_blocks_left, 1);

        rt.set_constant_ratio(1.0);
        let mut output_refs = [&mut output[..]];
        let (consumed, produced) = rt.process(&input_refs, &mut output_refs);
        assert_eq!((consumed, produced), (128, 128));
        assert_eq!(
            rt.ratio_motion_freeze_blocks_left, 0,
            "exact-unity passthrough should consume an existing ratio-motion freeze hold"
        );
        let post_unity = rt.profile_telemetry();
        assert_eq!(post_unity.current_profile, LatencyProfile::Scratch);
        assert_eq!(post_unity.target_profile, LatencyProfile::Scratch);

        rt.set_constant_ratio(1.0005);
        rt.set_hint_snapshot(render_hints);
        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);

        let resumed = rt.profile_telemetry();
        assert_eq!(
            resumed.current_profile,
            LatencyProfile::Scratch,
            "the first stable near-unity kernel after a unity callback should keep the scratch profile active"
        );
        assert_eq!(
            resumed.target_profile,
            LatencyProfile::Scratch,
            "the first stable near-unity kernel after a unity callback should not immediately retarget away from scratch"
        );

        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);
        let resumed = rt.profile_telemetry();
        assert_eq!(
            resumed.current_profile,
            LatencyProfile::Scratch,
            "the committed scratch profile should stay active until the queued render transition settles"
        );
        assert_eq!(
            resumed.target_profile,
            LatencyProfile::Render,
            "once the unity callback consumes the hold, the second stable near-unity kernel should be able to retarget away from scratch"
        );
        assert!(
            resumed.transition_blocks_left > 0,
            "retargeting away from scratch should queue a transition instead of snapping instantly"
        );
    }

    #[test]
    fn auto_profile_switching_holds_target_steady_until_transition_settles() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 128);
        cfg.latency_profile = LatencyProfile::Render;
        cfg.auto_profile_switching = true;
        cfg.profile_switch_hysteresis_blocks = 1;
        cfg.ratio_motion_freeze_blocks = 0;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        let input = [0.0f32; 128];
        let mut output = [0.0f32; 256];
        let input_refs = [&input[..]];
        let render_hints = Arc::new(RenderHints {
            transient_confidence: 0.05,
            tonal_confidence: 0.95,
            noise_confidence: 0.05,
            ..RenderHints::default()
        });
        let scratch_hints = Arc::new(RenderHints {
            transient_confidence: 0.90,
            tonal_confidence: 0.10,
            noise_confidence: 0.20,
            ..RenderHints::default()
        });

        rt.set_constant_ratio(1.70);
        rt.set_hint_snapshot(Arc::clone(&scratch_hints));
        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);

        let scratch_transition = rt.profile_telemetry();
        assert_eq!(scratch_transition.current_profile, LatencyProfile::Render);
        assert_eq!(scratch_transition.target_profile, LatencyProfile::Scratch);
        assert_eq!(scratch_transition.policy_profile, LatencyProfile::Scratch);
        assert!(
            scratch_transition.transition_blocks_left > 0,
            "test setup should leave a profile transition in flight"
        );

        rt.set_constant_ratio(1.01);
        rt.set_hint_snapshot(Arc::clone(&render_hints));
        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);

        let held_transition = rt.profile_telemetry();
        assert_eq!(
            held_transition.current_profile,
            LatencyProfile::Render,
            "the active profile should stay on render until the scratch transition completes"
        );
        assert_eq!(
            held_transition.target_profile,
            LatencyProfile::Scratch,
            "opposite suggestions must not immediately reverse an in-flight auto profile transition"
        );
        assert_eq!(
            held_transition.policy_profile,
            LatencyProfile::Scratch,
            "policy telemetry should keep advertising the committed scratch target while it is still crossfading"
        );

        for _ in 0..=held_transition.transition_blocks_left {
            if rt.profile_telemetry().current_profile == LatencyProfile::Scratch {
                break;
            }
            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
        }

        let settled_scratch = rt.profile_telemetry();
        assert_eq!(settled_scratch.current_profile, LatencyProfile::Scratch);
        assert_eq!(settled_scratch.target_profile, LatencyProfile::Scratch);

        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);
        let render_retarget = rt.profile_telemetry();
        assert_eq!(
            render_retarget.target_profile,
            LatencyProfile::Render,
            "once the scratch transition settles, auto mode should be free to retarget back to render"
        );
    }

    #[test]
    fn preview_callbacks_do_not_advance_profile_or_tier_transitions_before_kernel_commit() {
        let params = StretchParams::new(1.05)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut cfg = RtConfig::new(params, 32);
        cfg.latency_profile = LatencyProfile::Render;
        cfg.auto_profile_switching = false;
        let mut rt = RtProcessor::prepare(cfg).unwrap();

        rt.set_latency_profile(LatencyProfile::Mix);
        let initial_transition = rt.profile_transition_blocks_left;
        let initial_crossfade = rt.crossfade_blocks_left;
        assert!(
            initial_transition > 0,
            "test setup should stage a profile transition"
        );
        assert!(
            initial_crossfade > 0,
            "test setup should stage a tier crossfade"
        );

        let input = [0.0f32; 32];
        let mut output = [0.0f32; 64];
        let input_refs = [&input[..]];

        for preview_idx in 0..3 {
            let mut output_refs = [&mut output[..]];
            let _ = rt.process(&input_refs, &mut output_refs);
            assert_eq!(
                rt.profile_transition_blocks_left, initial_transition,
                "preview-only callback {preview_idx} must not advance the pending profile transition"
            );
            assert_eq!(
                rt.crossfade_blocks_left, initial_crossfade,
                "preview-only callback {preview_idx} must not advance the pending tier crossfade"
            );
        }

        let mut output_refs = [&mut output[..]];
        let _ = rt.process(&input_refs, &mut output_refs);
        assert_eq!(
            rt.profile_transition_blocks_left,
            initial_transition - 1,
            "the first committed kernel should advance the profile transition by exactly one block"
        );
        assert_eq!(
            rt.crossfade_blocks_left,
            initial_crossfade - 1,
            "the first committed kernel should advance the tier crossfade by exactly one block"
        );
    }
}

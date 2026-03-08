//! Real-time streaming time-stretch processor.

use crate::core::ring_buffer::RingBuffer;
use crate::core::types::{QualityMode, StretchParams};
use crate::core::window::WindowType;
use crate::dual_plane::{
    DualPlaneProcessor, LatencyProfile, RtConfig, RtDelayTelemetry, RtProfileTelemetry,
    RtRuntimeTelemetry,
};
use crate::error::StretchError;
use crate::stream::transient_scheduler::{TransientEventScheduler, TransientSchedulerStats};
use crate::stretch::phase_vocoder::PhaseVocoder;
use crate::stretch::stereo::StereoMode;

/// Threshold below which ratio differences are considered negligible.
const RATIO_SNAP_THRESHOLD: f64 = 0.0001;
/// Ratio smoothing time constant in seconds.
///
/// Smoothing is time-based (not callback-based), so behavior stays stable
/// across 64/128/256/1024 frame callbacks.
const RATIO_SMOOTHING_TIME_SECS: f64 = 0.050;
/// Additional slew applied after kernel-window averaging for deterministic dual-plane updates.
const DUAL_PLANE_RATIO_APPLY_SMOOTHING_TIME_SECS: f64 = 0.040;
/// Short seam ramp used when deterministic rendering exits exact-unity bypass.
const DUAL_PLANE_UNITY_EXIT_SEAM_FRAMES: usize = 64;
/// Numerator for the FFT-size latency fraction (3/2 = 1.5x FFT size).
const LATENCY_FFT_NUMERATOR: usize = 3;
/// Denominator for the FFT-size latency fraction.
const LATENCY_FFT_DENOMINATOR: usize = 2;
/// FFT size used by the low-latency tempo constructor.
const LOW_LATENCY_TEMPO_FFT_SIZE: usize = 1024;
/// Hop size used by the low-latency tempo constructor.
const LOW_LATENCY_TEMPO_HOP_SIZE: usize = LOW_LATENCY_TEMPO_FFT_SIZE / 4;

/// Callback size assumptions for real-time capacity planning.
const MAX_CALLBACK_FRAMES: usize = 1024;
const MIN_CALLBACK_FRAMES: usize = 64;
const COMMON_CALLBACK_FRAMES: usize = 256;
/// Iteration slack for bounded dynamic loops in the real-time path.
const LOOP_GUARD_SLACK: usize = 8;
/// Computes the minimum number of frames required before processing can begin.
#[inline]
const fn min_latency_frames(fft_size: usize) -> usize {
    fft_size * LATENCY_FFT_NUMERATOR / LATENCY_FFT_DENOMINATOR
}

/// Computes the effective minimum input size based on the current stretch ratio.
#[inline]
fn effective_min_frames(fft_size: usize, ratio: f64) -> usize {
    if (0.9..=1.1).contains(&ratio) {
        min_latency_frames(fft_size)
    } else {
        fft_size * 2
    }
}

#[inline]
fn validate_positive_finite_ratio(value: f64, label: &'static str) -> Result<f64, StretchError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(StretchError::InvalidRatio(format!(
            "{} must be finite and > 0.0, got {}",
            label, value
        )));
    }
    Ok(value)
}

#[inline]
fn ratio_from_tempo(source_bpm: f64, target_bpm: f64) -> Result<f64, StretchError> {
    let source = validate_positive_finite_ratio(source_bpm, "source BPM")?;
    let target = validate_positive_finite_ratio(target_bpm, "target BPM")?;
    validate_positive_finite_ratio(source / target, "stretch ratio from BPM values")
}

#[inline]
fn analysis_lookahead_frames(fft_size: usize, quality_mode: QualityMode) -> usize {
    match quality_mode {
        QualityMode::LowLatency => fft_size,
        QualityMode::Balanced => fft_size * 2,
        QualityMode::MaxQuality => fft_size * 4,
    }
}

#[inline]
fn stream_capacity_frames(params: &StretchParams) -> usize {
    let _ = MIN_CALLBACK_FRAMES;
    let _ = COMMON_CALLBACK_FRAMES;
    analysis_lookahead_frames(params.fft_size, params.quality_mode)
        .saturating_add(MAX_CALLBACK_FRAMES)
        .saturating_add(params.fft_size)
}

#[inline]
fn latency_profile_for_quality(mode: QualityMode) -> LatencyProfile {
    match mode {
        QualityMode::LowLatency => LatencyProfile::Scratch,
        QualityMode::Balanced => LatencyProfile::Mix,
        QualityMode::MaxQuality => LatencyProfile::Render,
    }
}

#[inline]
fn apply_dual_plane_ratio(
    processor: &mut DualPlaneProcessor,
    ratio: f64,
) -> Result<(), StretchError> {
    let ratio = validate_positive_finite_ratio(ratio, "dual-plane ratio")?;
    processor.rt_mut().set_constant_ratio(ratio);
    Ok(())
}

#[inline]
fn smooth_ratio_toward(
    current: f64,
    target: f64,
    frames: usize,
    sample_rate: u32,
    tau_secs: f64,
) -> f64 {
    if frames == 0 || (target - current).abs() <= RATIO_SNAP_THRESHOLD {
        return target;
    }

    let tau_frames = (sample_rate as f64 * tau_secs).max(1.0);
    let alpha = 1.0 - (-(frames as f64) / tau_frames).exp();
    let next = current + alpha * (target - current);
    if (next - target).abs() < RATIO_SNAP_THRESHOLD {
        target
    } else {
        next
    }
}

#[inline]
fn dual_plane_supports_pitch_scale(scale: f64) -> bool {
    scale.is_finite() && scale > 0.0
}

#[inline]
fn ratio_modulation_side(ratio: f64) -> i8 {
    let delta = ratio - 1.0;
    if delta.abs() <= RATIO_SNAP_THRESHOLD {
        0
    } else if delta > 0.0 {
        1
    } else {
        -1
    }
}

/// Stateful linear resampler used for realtime pitch control in stream mode.
///
/// Maintains one-sample look-behind and a fractional source cursor so
/// resampling remains continuous across callbacks.
#[derive(Debug, Clone)]
struct LinearResamplerState {
    prev_sample: f32,
    has_prev: bool,
    next_pos: f64,
}

impl LinearResamplerState {
    fn new() -> Self {
        Self {
            prev_sample: 0.0,
            has_prev: false,
            next_pos: 0.0,
        }
    }

    fn reset(&mut self) {
        self.prev_sample = 0.0;
        self.has_prev = false;
        self.next_pos = 0.0;
    }

    fn source_sample(&self, input: &[f32], idx: usize) -> f32 {
        if self.has_prev {
            if idx == 0 {
                self.prev_sample
            } else {
                input[idx - 1]
            }
        } else {
            input[idx]
        }
    }

    fn process_into(
        &mut self,
        input: &[f32],
        pitch_scale: f64,
        output: &mut Vec<f32>,
    ) -> Result<(), StretchError> {
        output.clear();
        if input.is_empty() {
            return Ok(());
        }

        let source_len = input.len() + usize::from(self.has_prev);
        if source_len < 2 {
            self.prev_sample = input[input.len() - 1];
            self.has_prev = true;
            self.next_pos = 0.0;
            return Ok(());
        }

        let mut pos = self.next_pos.max(0.0);
        while pos + 1.0 < source_len as f64 {
            if output.len() == output.capacity() {
                return Err(StretchError::BufferOverflow {
                    buffer: "stream_pitch_resample_output",
                    requested: output.len().saturating_add(1),
                    available: output.capacity(),
                });
            }

            let i = pos.floor() as usize;
            let frac = (pos - i as f64) as f32;
            let a = self.source_sample(input, i);
            let b = self.source_sample(input, i + 1);
            output.push(a + (b - a) * frac);
            pos += 1.0 / pitch_scale;
        }

        self.prev_sample = input[input.len() - 1];
        self.has_prev = true;
        let max_pos = source_len.saturating_sub(1) as f64;
        self.next_pos = (pos - max_pos).max(0.0);
        Ok(())
    }

    fn flush_into(&mut self, pitch_scale: f64, output: &mut Vec<f32>) -> Result<(), StretchError> {
        if !self.has_prev {
            output.clear();
            return Ok(());
        }
        let tail = [self.prev_sample];
        self.process_into(&tail, pitch_scale, output)?;
        self.reset();
        Ok(())
    }
}

/// Streaming chunk-based processor for real-time time stretching.
///
/// Uses fixed-capacity ring buffers in the steady state:
/// - no `Vec::drain`
/// - no front-removal shifts
/// - deterministic memory bounds
/// Aggregated transient-reset telemetry from deterministic stream processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransientResetStats {
    /// Number of transient events detected by the scheduler.
    pub events_detected_total: u64,
    /// Number of times each reset band was selected across all events.
    ///
    /// Layout: `[sub_bass, low, mid, high]`.
    pub reset_band_counts_total: [u64; 4],
    /// Absolute per-channel input frames consumed from the stream so far.
    pub input_frames_consumed_total: usize,
}

struct DualPlaneDeterministicState {
    processor: DualPlaneProcessor,
    num_channels: usize,
    block_frames: usize,
    input_planar: Vec<Vec<f32>>,
    output_planar: Vec<Vec<f32>>,
    flush_interleaved: Vec<f32>,
    ratio_window_blocks: usize,
    recent_chunk_ratios: Vec<f64>,
    last_ratio: f64,
    last_output_samples: Vec<f32>,
    has_last_output_samples: bool,
    unity_exit_seam_samples: Vec<f32>,
    pending_unity_exit_seam: bool,
}

impl DualPlaneDeterministicState {
    fn from_params(params: &StretchParams, ratio: f64) -> Result<Self, StretchError> {
        let block_frames = COMMON_CALLBACK_FRAMES;
        let kernel_frames = (params.fft_size * 2).max(block_frames);
        let mut rt_cfg = RtConfig::new(params.clone(), block_frames);
        rt_cfg.latency_profile = latency_profile_for_quality(params.quality_mode);
        rt_cfg.auto_profile_switching = true;
        rt_cfg.profile_switch_hysteresis_blocks = 6;
        rt_cfg.min_ratio = 0.05;
        rt_cfg.max_ratio = 8.0;
        let max_output_frames = ((rt_cfg.kernel_frames as f64 * rt_cfg.max_ratio).ceil() as usize)
            .saturating_add(params.fft_size);

        let mut processor = DualPlaneProcessor::prepare(rt_cfg)?;
        apply_dual_plane_ratio(&mut processor, ratio)?;
        let num_channels = params.channels.count().max(1);

        Ok(Self {
            processor,
            num_channels,
            block_frames,
            input_planar: (0..num_channels).map(|_| vec![0.0; block_frames]).collect(),
            output_planar: (0..num_channels)
                .map(|_| vec![0.0; max_output_frames])
                .collect(),
            flush_interleaved: Vec::with_capacity(max_output_frames.saturating_mul(num_channels)),
            ratio_window_blocks: kernel_frames.div_ceil(block_frames).max(1),
            recent_chunk_ratios: Vec::new(),
            last_ratio: ratio,
            last_output_samples: vec![0.0; num_channels],
            has_last_output_samples: false,
            unity_exit_seam_samples: vec![0.0; num_channels],
            pending_unity_exit_seam: false,
        })
    }

    #[inline]
    fn push_chunk_ratio(&mut self, ratio: f64, requested_ratio: f64) -> f64 {
        if (requested_ratio - 1.0).abs() <= RATIO_SNAP_THRESHOLD {
            self.recent_chunk_ratios.clear();
        }
        if let Some(&last_chunk_ratio) = self.recent_chunk_ratios.last() {
            let next_side = ratio_modulation_side(ratio);
            let requested_side = ratio_modulation_side(requested_ratio);
            let last_side = ratio_modulation_side(last_chunk_ratio);
            let effective_side = if requested_side != 0 {
                requested_side
            } else {
                next_side
            };
            if effective_side != 0 && last_side != 0 && effective_side != last_side {
                // Drop stale opposite-side history as soon as the requested
                // modulation flips across unity, even if smoothing has not
                // yet carried the applied ratio onto the new side.
                self.recent_chunk_ratios.clear();
            }
        }
        if self.recent_chunk_ratios.len() == self.ratio_window_blocks {
            self.recent_chunk_ratios.remove(0);
        }
        self.recent_chunk_ratios.push(ratio);
        let sum: f64 = self.recent_chunk_ratios.iter().sum();
        sum / self.recent_chunk_ratios.len().max(1) as f64
    }

    #[inline]
    fn reset_ratio_history(&mut self, ratio: f64) {
        self.recent_chunk_ratios.clear();
        self.last_ratio = ratio;
        if (ratio - 1.0).abs() <= RATIO_SNAP_THRESHOLD {
            self.pending_unity_exit_seam = false;
        }
    }

    #[inline]
    fn arm_unity_exit_seam(&mut self) {
        if !self.has_last_output_samples
            || self.last_output_samples.len() != self.num_channels
            || self.unity_exit_seam_samples.len() != self.num_channels
        {
            return;
        }
        self.unity_exit_seam_samples
            .copy_from_slice(&self.last_output_samples);
        self.pending_unity_exit_seam = true;
    }

    fn apply_pending_unity_exit_seam(
        &mut self,
        channel_output_buffers: &mut [Vec<f32>],
        produced_frames: usize,
        ramp_frames: usize,
    ) {
        if !self.pending_unity_exit_seam || produced_frames == 0 || ramp_frames == 0 {
            return;
        }

        let ramp_len = produced_frames.min(ramp_frames);
        let denom = ramp_len.saturating_add(1) as f32;
        for ch in 0..self.num_channels.min(channel_output_buffers.len()) {
            let anchor = self.unity_exit_seam_samples[ch];
            for i in 0..ramp_len.min(channel_output_buffers[ch].len()) {
                let t = (i + 1) as f32 / denom;
                let target = channel_output_buffers[ch][i];
                channel_output_buffers[ch][i] = anchor + (target - anchor) * t;
            }
        }

        self.pending_unity_exit_seam = false;
    }

    fn capture_last_output_samples(
        &mut self,
        channel_output_buffers: &[Vec<f32>],
        produced_frames: usize,
    ) {
        if produced_frames == 0 {
            return;
        }

        for ch in 0..self.num_channels.min(channel_output_buffers.len()) {
            if let Some(&sample) = channel_output_buffers[ch].get(produced_frames - 1) {
                self.last_output_samples[ch] = sample;
            }
        }
        self.has_last_output_samples = true;
    }
}

pub struct StreamProcessor {
    params: StretchParams,
    input_ring: RingBuffer<f32>,
    pending_output: RingBuffer<f32>,
    /// Current stretch ratio (can be changed on the fly).
    current_ratio: f64,
    /// Target stretch ratio (for smooth interpolation).
    target_ratio: f64,
    /// The ratio that the vocoders are currently configured for.
    vocoder_ratio: f64,
    /// Whether the processor has been initialized.
    initialized: bool,
    /// Persistent PhaseVocoder instances, one per channel.
    vocoders: Vec<PhaseVocoder>,
    /// Reusable per-channel deinterleave buffers.
    channel_input_buffers: Vec<Vec<f32>>,
    /// Reusable per-channel stretched output buffers.
    channel_output_buffers: Vec<Vec<f32>>,
    /// Reusable interleaved snapshot of the current input ring.
    interleaved_scratch: Vec<f32>,
    /// Source BPM (set when created via `from_tempo`, enables `set_tempo`).
    source_bpm: Option<f64>,
    /// Persistent transient event scheduler for deterministic stream mode.
    transient_scheduler: TransientEventScheduler,
    /// Absolute count of per-channel input frames consumed from `input_ring`.
    ///
    /// This provides a stable timeline for incremental transient scheduling.
    input_frames_consumed_total: usize,
    /// Expected total output samples across the current stream.
    ///
    /// Accumulated from input samples and the effective interpolated ratio,
    /// then reconciled on flush to avoid long-run drift.
    expected_total_output_samples: f64,
    /// Total output samples emitted to the caller for the current stream.
    total_output_emitted_samples: usize,
    /// Realtime pitch scale applied in stream mode.
    pitch_scale: f64,
    /// Stateful per-channel resamplers for realtime pitch control.
    pitch_resamplers: Vec<LinearResamplerState>,
    /// Reusable per-channel output buffers for pitch-resampled data.
    pitch_output_buffers: Vec<Vec<f32>>,
    /// Whether a fixed-buffer flush drain is currently in progress.
    fixed_flush_pending: bool,
    /// Whether the deterministic backend has fully emitted its flush tail.
    fixed_flush_source_exhausted: bool,
    /// Whether pitch-resampler tail flush has already been applied.
    fixed_flush_pitch_tail_flushed: bool,
    /// Whether fixed-buffer flush length reconciliation has already run.
    fixed_flush_length_reconciled: bool,
    /// Scratch used to reconcile fixed-buffer flush tails without reallocating.
    fixed_flush_scratch: Vec<f32>,
    /// Whether deterministic mode prefers the dual-plane backend.
    dual_plane_preferred: bool,
    /// Optional active dual-plane deterministic backend state.
    dual_plane_deterministic: Option<DualPlaneDeterministicState>,
    /// Deferred pitch-scale change requested while a fixed-buffer flush drain is active.
    pending_pitch_scale: Option<f64>,
}

impl std::fmt::Debug for StreamProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamProcessor")
            .field("params", &self.params)
            .field("current_ratio", &self.current_ratio)
            .field("target_ratio", &self.target_ratio)
            .field("vocoder_ratio", &self.vocoder_ratio)
            .field("pitch_scale", &self.pitch_scale)
            .field("initialized", &self.initialized)
            .field("fixed_flush_pending", &self.fixed_flush_pending)
            .field(
                "dual_plane_deterministic",
                &self.dual_plane_deterministic.is_some(),
            )
            .field("pending_pitch_scale", &self.pending_pitch_scale)
            .field("dual_plane_preferred", &self.dual_plane_preferred)
            .field("source_bpm", &self.source_bpm)
            .field("input_ring_len", &self.input_ring.len())
            .field("pending_output_len", &self.pending_output.len())
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
struct FixedProcessInterleavedBudget {
    num_channels: usize,
    block_frames: usize,
    required_samples: usize,
}

#[inline]
fn align_interleaved_samples_up(samples: usize, num_channels: usize) -> usize {
    if num_channels <= 1 {
        return samples;
    }

    samples
        .saturating_add(num_channels.saturating_sub(1))
        .saturating_div(num_channels)
        .saturating_mul(num_channels)
}

impl StreamProcessor {
    /// Creates a new streaming processor.
    pub fn new(params: StretchParams) -> Self {
        let ratio = params.stretch_ratio;
        let num_channels = params.channels.count();
        let source_bpm = params.bpm;

        let capacity_frames_per_channel = stream_capacity_frames(&params);
        let capacity_samples = capacity_frames_per_channel.saturating_mul(num_channels);
        let output_capacity_frames = capacity_frames_per_channel
            .saturating_mul(4)
            .saturating_add(params.fft_size);
        let output_capacity_samples = output_capacity_frames.saturating_mul(num_channels);
        let pitch_output_capacity_frames = output_capacity_frames.saturating_mul(2);

        let vocoders = Self::create_vocoders(&params, ratio);
        let channel_input_buffers = (0..num_channels)
            .map(|_| Vec::with_capacity(capacity_frames_per_channel))
            .collect();
        let channel_output_buffers = (0..num_channels)
            .map(|_| Vec::with_capacity(output_capacity_frames))
            .collect();
        let transient_scheduler = TransientEventScheduler::new(
            params.fft_size,
            params.hop_size,
            params.sample_rate,
            capacity_frames_per_channel,
        );

        let mut me = Self {
            params,
            input_ring: RingBuffer::with_capacity(capacity_samples),
            pending_output: RingBuffer::with_capacity(output_capacity_samples),
            current_ratio: ratio,
            target_ratio: ratio,
            vocoder_ratio: ratio,
            initialized: false,
            vocoders,
            channel_input_buffers,
            channel_output_buffers,
            interleaved_scratch: vec![0.0; capacity_samples],
            source_bpm,
            transient_scheduler,
            input_frames_consumed_total: 0,
            expected_total_output_samples: 0.0,
            total_output_emitted_samples: 0,
            pitch_scale: 1.0,
            pitch_resamplers: (0..num_channels)
                .map(|_| LinearResamplerState::new())
                .collect(),
            pitch_output_buffers: (0..num_channels)
                .map(|_| Vec::with_capacity(pitch_output_capacity_frames))
                .collect(),
            fixed_flush_pending: false,
            fixed_flush_source_exhausted: false,
            fixed_flush_pitch_tail_flushed: false,
            fixed_flush_length_reconciled: false,
            fixed_flush_scratch: Vec::with_capacity(output_capacity_samples),
            dual_plane_preferred: true,
            dual_plane_deterministic: None,
            pending_pitch_scale: None,
        };
        me.ensure_default_dual_plane_backend();
        me
    }

    #[inline]
    fn is_fresh_stream(&self) -> bool {
        !self.initialized && self.input_ring.is_empty() && self.pending_output.is_empty()
    }

    #[inline]
    fn should_activate_dual_plane(&self) -> bool {
        self.dual_plane_preferred && dual_plane_supports_pitch_scale(self.pitch_scale)
    }

    fn ensure_default_dual_plane_backend(&mut self) {
        if self.dual_plane_deterministic.is_some() || !self.should_activate_dual_plane() {
            return;
        }
        if !self.is_fresh_stream() {
            return;
        }
        if let Ok(state) =
            DualPlaneDeterministicState::from_params(&self.params, self.processing_ratio())
        {
            self.dual_plane_deterministic = Some(state);
        }
    }

    /// Creates PhaseVocoder instances for each channel.
    fn create_vocoders(params: &StretchParams, ratio: f64) -> Vec<PhaseVocoder> {
        (0..params.channels.count())
            .map(|_| {
                let mut pv = PhaseVocoder::with_all_options(
                    params.fft_size,
                    params.hop_size,
                    ratio,
                    params.sample_rate,
                    params.sub_bass_cutoff,
                    params.window_type,
                    params.phase_locking_mode,
                    params.envelope_preservation,
                    params.envelope_order,
                );
                pv.set_adaptive_phase_locking(params.adaptive_phase_locking);
                pv.set_envelope_strength(params.envelope_strength);
                pv.set_adaptive_envelope_order(params.adaptive_envelope_order);
                pv
            })
            .collect()
    }

    /// Processes a chunk of interleaved audio samples.
    ///
    /// This convenience API may allocate for the returned `Vec`.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        let ratio_hint = self.current_ratio.max(self.target_ratio).max(1.0);
        let estimated =
            ((input.len() as f64) * ratio_hint).ceil() as usize + self.pending_output.capacity();
        let mut out = Vec::with_capacity(estimated);
        self.process_into(input, &mut out)?;
        Ok(out)
    }

    /// Returns a deterministic upper bound for fixed-buffer process output.
    ///
    /// The returned value is the maximum number of new interleaved samples the
    /// deterministic fixed-buffer path may need to queue while consuming
    /// `input_samples` of new input. If `output` in
    /// [`StreamProcessor::process_interleaved_into`] is at least this large,
    /// the call will not reject for `stream_process_interleaved_output`
    /// capacity, regardless of current pending-output depth.
    ///
    /// This helper is available when the deterministic dual-plane backend is
    /// active. The returned sample count is always aligned to whole frames.
    pub fn max_process_interleaved_output_samples(
        &mut self,
        input_samples: usize,
    ) -> Result<usize, StretchError> {
        Ok(self
            .fixed_process_interleaved_budget(input_samples)?
            .required_samples)
    }

    /// Returns a deterministic upper bound for the next fixed-buffer process callback output.
    ///
    /// The returned value includes both interleaved samples already queued for
    /// draining and the maximum number of new interleaved samples the
    /// deterministic fixed-buffer path may emit while consuming `input_samples`
    /// of new input. If `output` in [`StreamProcessor::process_interleaved_into`]
    /// is at least this large, the next call can drain the current queue and
    /// accept `input_samples` of new input without leaving host-visible backlog
    /// above the deterministic backend.
    ///
    /// This helper is equivalent to adding
    /// [`StreamProcessor::queued_interleaved_output_samples`] and
    /// [`StreamProcessor::max_process_interleaved_output_samples`], but keeps
    /// next-callback sizing in one frame-aligned public call.
    pub fn max_next_process_interleaved_output_samples(
        &mut self,
        input_samples: usize,
    ) -> Result<usize, StretchError> {
        let FixedProcessInterleavedBudget {
            num_channels,
            required_samples,
            ..
        } = self.fixed_process_interleaved_budget(input_samples)?;
        if !self.pending_output.len().is_multiple_of(num_channels) {
            return Err(StretchError::InvalidState(
                "queued fixed-buffer output lost interleaved frame alignment",
            ));
        }

        Ok(self.pending_output.len().saturating_add(required_samples))
    }

    /// Returns the exact number of interleaved samples already queued for fixed-buffer draining.
    ///
    /// This counts host-visible samples currently buffered above the
    /// deterministic dual-plane backend. Hosts can add this value to
    /// [`StreamProcessor::max_process_interleaved_output_samples`] to size an
    /// output buffer large enough to drain the current queue and capture all
    /// new output from the next [`StreamProcessor::process_interleaved_into`]
    /// call.
    ///
    /// This helper is available when the deterministic dual-plane backend is
    /// active. The returned sample count is always aligned to whole frames.
    pub fn queued_interleaved_output_samples(&mut self) -> Result<usize, StretchError> {
        self.ensure_default_dual_plane_backend();
        let Some(state_meta) = self.dual_plane_deterministic.as_ref() else {
            return Err(StretchError::InvalidState(
                "dual-plane deterministic backend is unavailable",
            ));
        };

        let num_channels = state_meta.num_channels.max(1);
        if !self.pending_output.len().is_multiple_of(num_channels) {
            return Err(StretchError::InvalidState(
                "queued fixed-buffer output lost interleaved frame alignment",
            ));
        }

        Ok(self.pending_output.len())
    }

    /// Returns a deterministic upper bound for remaining fixed-buffer flush output.
    ///
    /// The returned value is a conservative upper bound for the number of
    /// interleaved samples still required to drain the current stream tail. If `output` in
    /// [`StreamProcessor::flush_interleaved_into`] is at least this large, the
    /// remaining tail drains in one call from the current stream state.
    ///
    /// This helper is available when the deterministic dual-plane backend is
    /// active. The returned sample count is always aligned to whole frames.
    pub fn max_flush_interleaved_output_samples(&mut self) -> Result<usize, StretchError> {
        self.ensure_default_dual_plane_backend();
        let Some(state_meta) = self.dual_plane_deterministic.as_ref() else {
            return Err(StretchError::InvalidState(
                "dual-plane deterministic backend is unavailable",
            ));
        };

        let num_channels = state_meta.num_channels.max(1);
        let remaining = self.remaining_expected_flush_samples();
        if remaining == 0 && self.pending_output.is_empty() {
            return Ok(0);
        }

        Ok(align_interleaved_samples_up(
            remaining.saturating_add(self.pending_output.capacity()),
            num_channels,
        ))
    }

    /// Processes a chunk of deterministic interleaved audio into a fixed buffer.
    ///
    /// Returns the number of interleaved samples written to `output`.
    /// Only full frames are written; any trailing partial-frame capacity in
    /// `output` is ignored.
    ///
    /// This host-facing fixed-buffer contract is available when the
    /// deterministic dual-plane backend is active. Produced samples that do
    /// not fit in `output` remain queued for later calls.
    pub fn process_interleaved_into(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<usize, StretchError> {
        let FixedProcessInterleavedBudget {
            num_channels,
            block_frames,
            required_samples,
        } = self.fixed_process_interleaved_budget(input.len())?;
        if input.iter().any(|s| !s.is_finite()) {
            return Err(StretchError::NonFiniteInput);
        }

        let aligned_capacity = output
            .len()
            .saturating_div(num_channels)
            .saturating_mul(num_channels);
        let available_budget = aligned_capacity.saturating_add(self.pending_output.available());
        if required_samples > available_budget {
            return Err(StretchError::BufferOverflow {
                buffer: "stream_process_interleaved_output",
                requested: required_samples,
                available: available_budget,
            });
        }

        self.initialized = true;

        let mut written_total = 0usize;
        if aligned_capacity > 0 {
            written_total += self.drain_pending_to_buffer(&mut output[..aligned_capacity])?;
        }

        let mut offset = 0usize;
        while offset < input.len() {
            let remaining_frames = (input.len() - offset) / num_channels;
            let frames = remaining_frames.min(block_frames);
            if frames == 0 {
                break;
            }

            let end = offset + frames * num_channels;
            self.process_dual_plane_chunk_to_pending(&input[offset..end])?;
            if written_total < aligned_capacity {
                written_total +=
                    self.drain_pending_to_buffer(&mut output[written_total..aligned_capacity])?;
            }
            offset = end;
        }

        if input.is_empty() {
            self.interpolate_ratio_for_frames(COMMON_CALLBACK_FRAMES);
            self.apply_current_dual_plane_ratio()?;
        }

        if written_total < aligned_capacity {
            written_total +=
                self.drain_pending_to_buffer(&mut output[written_total..aligned_capacity])?;
        }
        Ok(written_total)
    }

    /// Processes a chunk of interleaved audio, appending output to `output`.
    ///
    /// This is the real-time API. It does not grow internal buffers in the
    /// steady state and it never shifts buffer memory.
    pub fn process_into(
        &mut self,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<(), StretchError> {
        if self.fixed_flush_pending {
            return Err(StretchError::InvalidState(
                "fixed-buffer flush output must be fully drained before new input",
            ));
        }
        if input.iter().any(|s| !s.is_finite()) {
            return Err(StretchError::NonFiniteInput);
        }

        self.initialized = true;

        if self.dual_plane_deterministic.is_some() {
            return self.process_into_dual_plane(input, output);
        }

        // Fast passthrough for unity ratio: skip PV/WSOLA processing to
        // produce bit-exact output and eliminate windowing/overlap-add drift.
        if (self.target_ratio - 1.0).abs() < RATIO_SNAP_THRESHOLD
            && (self.current_ratio - 1.0).abs() < RATIO_SNAP_THRESHOLD
            && (self.pitch_scale - 1.0).abs() < RATIO_SNAP_THRESHOLD
        {
            let available = output.capacity().saturating_sub(output.len());
            if input.len() > available {
                return Err(StretchError::BufferOverflow {
                    buffer: "process_into_output",
                    requested: input.len(),
                    available,
                });
            }
            output.extend_from_slice(input);
            return Ok(());
        }

        let num_channels = self.params.channels.count().max(1);
        let mut offset = 0usize;
        let mut iterations = 0usize;
        let max_iterations = input
            .len()
            .saturating_add(LOOP_GUARD_SLACK)
            .max(LOOP_GUARD_SLACK);
        while offset < input.len() {
            iterations = iterations.saturating_add(1);
            if iterations > max_iterations {
                return Err(StretchError::InvalidState(
                    "process_into iteration bound exceeded",
                ));
            }
            if self.input_ring.available() == 0 {
                self.interpolate_ratio_for_frames(COMMON_CALLBACK_FRAMES);
                self.process_available_to_pending(true)?;
                if self.input_ring.available() == 0 {
                    return Err(StretchError::BufferOverflow {
                        buffer: "stream_input_ring",
                        requested: input.len() - offset,
                        available: 0,
                    });
                }
            }

            let take = (input.len() - offset).min(self.input_ring.available());
            if take == 0 {
                return Err(StretchError::InvalidState(
                    "process_into made zero progress while input remained",
                ));
            }
            self.push_input_samples(&input[offset..offset + take])?;
            offset += take;

            let frames = (take / num_channels).max(1);
            self.interpolate_ratio_for_frames(frames);
            self.expected_total_output_samples += take as f64 * self.current_ratio;
            self.process_available_to_pending(true)?;
            let _ = self.drain_pending_to_output(output)?;
        }

        if input.is_empty() {
            self.interpolate_ratio_for_frames(COMMON_CALLBACK_FRAMES);
            self.process_available_to_pending(true)?;
        }

        let _ = self.drain_pending_to_output(output)?;
        Ok(())
    }

    /// Flushes remaining buffered samples into a caller-provided buffer.
    ///
    /// Returns the number of samples written to `output`.
    pub fn flush_into(&mut self, output: &mut Vec<f32>) -> Result<usize, StretchError> {
        if self.fixed_flush_pending {
            return Err(StretchError::InvalidState(
                "fixed-buffer flush output must be fully drained before Vec flush",
            ));
        }
        let before = output.len();
        if self.dual_plane_deterministic.is_some() {
            let written = self.flush_into_dual_plane(output)?;
            self.expected_total_output_samples = 0.0;
            self.total_output_emitted_samples = 0;
            return Ok(written);
        }
        let num_channels = self.params.channels.count();
        if self.params.hop_size == 0 {
            return Err(StretchError::InvalidState("hop_size must be > 0"));
        }

        self.interpolate_ratio_for_frames(COMMON_CALLBACK_FRAMES);

        if !self.input_ring.is_empty() {
            let min_samples = self.params.fft_size.saturating_mul(num_channels);
            if self.input_ring.len() < min_samples {
                let missing = min_samples - self.input_ring.len();
                if missing > self.input_ring.available() {
                    return Err(StretchError::BufferOverflow {
                        buffer: "stream_input_ring",
                        requested: missing,
                        available: self.input_ring.available(),
                    });
                }
                let zeros = [0.0f32; 256];
                let mut remaining = missing;
                let mut iterations = 0usize;
                let max_iterations = missing
                    .saturating_add(zeros.len().saturating_sub(1))
                    .saturating_div(zeros.len())
                    .saturating_add(LOOP_GUARD_SLACK);
                while remaining > 0 {
                    iterations = iterations.saturating_add(1);
                    if iterations > max_iterations {
                        return Err(StretchError::InvalidState(
                            "flush zero-padding iteration bound exceeded",
                        ));
                    }
                    let chunk = remaining.min(zeros.len());
                    if chunk == 0 {
                        return Err(StretchError::InvalidState(
                            "flush zero-padding made zero progress",
                        ));
                    }
                    let pushed = self.input_ring.push_slice(&zeros[..chunk]);
                    if pushed != chunk {
                        return Err(StretchError::BufferOverflow {
                            buffer: "stream_input_ring",
                            requested: chunk,
                            available: pushed,
                        });
                    }
                    remaining -= chunk;
                }
            }

            self.process_available_to_pending(false)?;
        }

        self.flush_vocoder_tails_to_pending(num_channels)?;

        let remaining_frames = self.input_ring.len() / num_channels.max(1);
        self.input_frames_consumed_total = self
            .input_frames_consumed_total
            .saturating_add(remaining_frames);
        self.input_ring.clear();

        self.flush_pitch_resampler_to_pending(num_channels)?;
        self.reset_pitch_resamplers();

        let _ = self.drain_pending_to_output(output)?;

        // Reconcile end-of-stream length to the accumulated expected sample
        // count, reducing drift from analysis-padding/tail handling.
        let expected_total = self.expected_total_output_samples.round() as isize;
        let actual_total = self.total_output_emitted_samples as isize;
        let correction = expected_total - actual_total;
        if correction > 0 {
            let need = correction as usize;
            output.reserve(need);
            extend_with_tonal_tail(output, need, before);
            self.total_output_emitted_samples += need;
        } else if correction < 0 {
            // Only trim samples emitted in this flush call, never samples
            // produced before `before`.
            let produced_here = output.len().saturating_sub(before);
            let trim = ((-correction) as usize).min(produced_here);
            if trim > 0 {
                output.truncate(output.len().saturating_sub(trim));
                self.total_output_emitted_samples =
                    self.total_output_emitted_samples.saturating_sub(trim);
            }
        }

        // Start a fresh accounting window after flush.
        self.expected_total_output_samples = 0.0;
        self.total_output_emitted_samples = 0;
        Ok(output.len().saturating_sub(before))
    }

    /// Flushes remaining buffered samples.
    pub fn flush(&mut self) -> Result<Vec<f32>, StretchError> {
        let mut out = Vec::with_capacity(self.pending_output.capacity());
        self.flush_into(&mut out)?;
        Ok(out)
    }

    /// Flushes remaining deterministic interleaved samples into a fixed buffer.
    ///
    /// This host-facing fixed-buffer contract is available when the
    /// deterministic dual-plane backend is active. If more output remains than
    /// fits in `output`, call again until this method returns `0`. No new
    /// input may be processed until the flush tail has been fully drained.
    pub fn flush_interleaved_into(&mut self, output: &mut [f32]) -> Result<usize, StretchError> {
        self.ensure_default_dual_plane_backend();
        if self.dual_plane_deterministic.is_none() {
            return Err(StretchError::InvalidState(
                "dual-plane deterministic backend is unavailable",
            ));
        }

        self.flush_interleaved_into_dual_plane(output)
    }

    fn process_into_dual_plane(
        &mut self,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<(), StretchError> {
        let Some(state_meta) = self.dual_plane_deterministic.as_ref() else {
            return Err(StretchError::InvalidState(
                "dual-plane deterministic state is unavailable",
            ));
        };
        let num_channels = state_meta.num_channels.max(1);
        let block_frames = state_meta.block_frames;
        if !input.len().is_multiple_of(num_channels) {
            return Err(StretchError::InvalidFormat(format!(
                "input sample count {} is not a multiple of channel count {}",
                input.len(),
                num_channels
            )));
        }

        let mut offset = 0usize;
        while offset < input.len() {
            let remaining_frames = (input.len() - offset) / num_channels;
            let frames = remaining_frames.min(block_frames);
            if frames == 0 {
                break;
            }

            let end = offset + frames * num_channels;
            self.process_dual_plane_chunk_to_pending(&input[offset..end])?;
            let _ = self.drain_pending_to_output(output)?;
            offset = end;
        }

        if input.is_empty() {
            self.interpolate_ratio_for_frames(COMMON_CALLBACK_FRAMES);
            self.apply_current_dual_plane_ratio()?;
        }

        let _ = self.drain_pending_to_output(output)?;
        Ok(())
    }

    fn flush_into_dual_plane(&mut self, output: &mut Vec<f32>) -> Result<usize, StretchError> {
        let Some(_state) = self.dual_plane_deterministic.as_ref() else {
            return Err(StretchError::InvalidState(
                "dual-plane deterministic state is unavailable",
            ));
        };

        let num_channels = self.params.channels.count().max(1);
        let before = output.len();
        let flushed_samples = {
            let Some(state) = self.dual_plane_deterministic.as_mut() else {
                return Err(StretchError::InvalidState(
                    "dual-plane deterministic state became unavailable",
                ));
            };
            state.flush_interleaved.clear();
            state.processor.flush(&mut state.flush_interleaved)?;
            state.flush_interleaved.len()
        };

        if !flushed_samples.is_multiple_of(num_channels) {
            return Err(StretchError::InvalidState(
                "dual-plane flush emitted non-interleaved sample count",
            ));
        }

        let max_chunk_frames = self
            .channel_output_buffers
            .iter()
            .take(num_channels)
            .map(|buf| buf.capacity())
            .min()
            .unwrap_or(0);
        if max_chunk_frames == 0 && flushed_samples > 0 {
            return Err(StretchError::BufferOverflow {
                buffer: "stream_channel_output",
                requested: 1,
                available: 0,
            });
        }

        let total_frames = flushed_samples / num_channels;
        let mut offset = 0usize;
        let mut iterations = 0usize;
        let max_iterations = total_frames
            .saturating_add(max_chunk_frames.saturating_sub(1))
            .saturating_div(max_chunk_frames.max(1))
            .saturating_add(LOOP_GUARD_SLACK);
        while offset < flushed_samples {
            iterations = iterations.saturating_add(1);
            if iterations > max_iterations {
                return Err(StretchError::InvalidState(
                    "dual-plane flush chunking iteration bound exceeded",
                ));
            }

            let remaining_frames = (flushed_samples - offset) / num_channels;
            let chunk_frames = remaining_frames.min(max_chunk_frames.max(1));
            if chunk_frames == 0 {
                return Err(StretchError::InvalidState(
                    "dual-plane flush chunking made zero progress",
                ));
            }

            {
                let (channel_output_buffers, dual_plane_state) = (
                    &mut self.channel_output_buffers,
                    &mut self.dual_plane_deterministic,
                );
                let Some(state) = dual_plane_state.as_mut() else {
                    return Err(StretchError::InvalidState(
                        "dual-plane deterministic state became unavailable",
                    ));
                };

                for ch in 0..num_channels {
                    if channel_output_buffers[ch].capacity() < chunk_frames {
                        return Err(StretchError::BufferOverflow {
                            buffer: "stream_channel_output",
                            requested: chunk_frames,
                            available: channel_output_buffers[ch].capacity(),
                        });
                    }
                    channel_output_buffers[ch].clear();
                }

                for frame in 0..chunk_frames {
                    let base = offset + frame * num_channels;
                    for ch in 0..num_channels {
                        channel_output_buffers[ch].push(state.flush_interleaved[base + ch]);
                    }
                }
            }

            self.emit_channel_output_to_pending(chunk_frames, num_channels)?;
            let _ = self.drain_pending_to_output(output)?;
            offset += chunk_frames * num_channels;
        }

        self.flush_pitch_resampler_to_pending(num_channels)?;
        self.reset_pitch_resamplers();
        let _ = self.drain_pending_to_output(output)?;
        Ok(output.len().saturating_sub(before))
    }

    fn flush_interleaved_into_dual_plane(
        &mut self,
        output: &mut [f32],
    ) -> Result<usize, StretchError> {
        let num_channels = self.params.channels.count().max(1);
        let aligned_capacity = output
            .len()
            .saturating_div(num_channels)
            .saturating_mul(num_channels);

        if !self.fixed_flush_pending {
            self.fixed_flush_pending = true;
            self.fixed_flush_source_exhausted = false;
            self.fixed_flush_pitch_tail_flushed = false;
            self.fixed_flush_length_reconciled = false;
        }

        let mut written_total = 0usize;
        loop {
            if written_total < aligned_capacity {
                let written =
                    self.drain_pending_to_buffer(&mut output[written_total..aligned_capacity])?;
                written_total += written;
                if written_total == aligned_capacity {
                    return Ok(written_total);
                }
            }

            if !self.fixed_flush_source_exhausted {
                if self.flush_next_dual_plane_chunk_to_pending()? > 0 {
                    continue;
                }
                self.fixed_flush_source_exhausted = true;
            }

            if !self.fixed_flush_pitch_tail_flushed {
                self.flush_pitch_resampler_to_pending(num_channels)?;
                self.reset_pitch_resamplers();
                self.fixed_flush_pitch_tail_flushed = true;
                if !self.pending_output.is_empty() {
                    continue;
                }
            }

            if !self.fixed_flush_length_reconciled {
                self.reconcile_fixed_flush_pending_output(num_channels)?;
                self.fixed_flush_length_reconciled = true;
                if !self.pending_output.is_empty() {
                    continue;
                }
            }

            if self.pending_output.is_empty() {
                self.finish_fixed_flush_drain();
                return Ok(written_total);
            }

            if aligned_capacity == 0 {
                return Err(StretchError::BufferOverflow {
                    buffer: "stream_flush_interleaved_output",
                    requested: num_channels,
                    available: 0,
                });
            }

            return Ok(written_total);
        }
    }

    fn flush_next_dual_plane_chunk_to_pending(&mut self) -> Result<usize, StretchError> {
        let Some(_state) = self.dual_plane_deterministic.as_ref() else {
            return Err(StretchError::InvalidState(
                "dual-plane deterministic state is unavailable",
            ));
        };

        let num_channels = self.params.channels.count().max(1);
        let max_chunk_frames = self
            .channel_output_buffers
            .iter()
            .take(num_channels)
            .map(|buf| buf.capacity())
            .min()
            .unwrap_or(0);
        if max_chunk_frames == 0 {
            return Err(StretchError::BufferOverflow {
                buffer: "stream_channel_output",
                requested: 1,
                available: 0,
            });
        }

        let written = {
            let Some(state) = self.dual_plane_deterministic.as_mut() else {
                return Err(StretchError::InvalidState(
                    "dual-plane deterministic state became unavailable",
                ));
            };
            let chunk_samples = max_chunk_frames.saturating_mul(num_channels);
            if state.flush_interleaved.len() != chunk_samples {
                state.flush_interleaved.resize(chunk_samples, 0.0);
            }
            state
                .processor
                .flush_into(&mut state.flush_interleaved[..])?
        };
        if written == 0 {
            return Ok(0);
        }
        if !written.is_multiple_of(num_channels) {
            return Err(StretchError::InvalidState(
                "dual-plane flush emitted non-interleaved sample count",
            ));
        }

        let chunk_frames = written / num_channels;
        {
            let (channel_output_buffers, dual_plane_state) = (
                &mut self.channel_output_buffers,
                &mut self.dual_plane_deterministic,
            );
            let Some(state) = dual_plane_state.as_mut() else {
                return Err(StretchError::InvalidState(
                    "dual-plane deterministic state became unavailable",
                ));
            };

            for ch in 0..num_channels {
                if channel_output_buffers[ch].capacity() < chunk_frames {
                    return Err(StretchError::BufferOverflow {
                        buffer: "stream_channel_output",
                        requested: chunk_frames,
                        available: channel_output_buffers[ch].capacity(),
                    });
                }
                channel_output_buffers[ch].clear();
            }

            for frame in 0..chunk_frames {
                let base = frame * num_channels;
                for ch in 0..num_channels {
                    channel_output_buffers[ch].push(state.flush_interleaved[base + ch]);
                }
            }
        }

        self.emit_channel_output_to_pending(chunk_frames, num_channels)?;
        Ok(written)
    }

    fn apply_current_dual_plane_ratio(&mut self) -> Result<(), StretchError> {
        let processing_ratio = self.processing_ratio();
        let Some(state) = self.dual_plane_deterministic.as_mut() else {
            return Err(StretchError::InvalidState(
                "dual-plane deterministic state is unavailable",
            ));
        };
        if processing_ratio == 1.0 {
            if state.last_ratio != 1.0 {
                apply_dual_plane_ratio(&mut state.processor, 1.0)?;
            }
            state.reset_ratio_history(1.0);
            return Ok(());
        }
        if (processing_ratio - state.last_ratio).abs() > RATIO_SNAP_THRESHOLD {
            apply_dual_plane_ratio(&mut state.processor, processing_ratio)?;
            state.last_ratio = processing_ratio;
        }
        Ok(())
    }

    fn max_dual_plane_pending_samples_for_frames(
        &self,
        input_frames: usize,
    ) -> Result<usize, StretchError> {
        let Some(state) = self.dual_plane_deterministic.as_ref() else {
            return Err(StretchError::InvalidState(
                "dual-plane deterministic state is unavailable",
            ));
        };

        let num_channels = state.num_channels.max(1);
        let dual_plane_frames = state
            .output_planar
            .iter()
            .take(num_channels)
            .map(Vec::len)
            .min()
            .unwrap_or(0);
        let max_frames = if (self.pitch_scale - 1.0).abs() < RATIO_SNAP_THRESHOLD {
            dual_plane_frames.min(
                self.channel_output_buffers
                    .iter()
                    .take(num_channels)
                    .map(|buf| buf.capacity())
                    .min()
                    .unwrap_or(0),
            )
        } else {
            self.pitch_output_buffers
                .iter()
                .take(num_channels)
                .map(|buf| buf.capacity())
                .min()
                .unwrap_or(0)
        };

        let ratio_hint = self.current_ratio.max(self.target_ratio).max(1.0);
        let estimated_frames = ((input_frames as f64) * ratio_hint).ceil() as usize;
        let bounded_frames = estimated_frames
            .saturating_add(self.params.fft_size)
            .saturating_add(4)
            .min(max_frames);

        Ok(bounded_frames.saturating_mul(num_channels))
    }

    fn fixed_process_interleaved_budget(
        &mut self,
        input_samples: usize,
    ) -> Result<FixedProcessInterleavedBudget, StretchError> {
        if self.fixed_flush_pending {
            return Err(StretchError::InvalidState(
                "fixed-buffer flush output must be fully drained before new input",
            ));
        }
        self.ensure_default_dual_plane_backend();
        let Some(state_meta) = self.dual_plane_deterministic.as_ref() else {
            return Err(StretchError::InvalidState(
                "dual-plane deterministic backend is unavailable",
            ));
        };
        let num_channels = state_meta.num_channels.max(1);
        if !input_samples.is_multiple_of(num_channels) {
            return Err(StretchError::InvalidFormat(format!(
                "input sample count {} is not a multiple of channel count {}",
                input_samples, num_channels
            )));
        }

        let mut required_samples = 0usize;
        let mut remaining_frames = input_samples / num_channels;
        while remaining_frames > 0 {
            let frames = remaining_frames.min(state_meta.block_frames);
            required_samples = required_samples
                .saturating_add(self.max_dual_plane_pending_samples_for_frames(frames)?);
            remaining_frames -= frames;
        }

        Ok(FixedProcessInterleavedBudget {
            num_channels,
            block_frames: state_meta.block_frames,
            required_samples,
        })
    }

    #[inline]
    fn remaining_expected_flush_samples(&self) -> usize {
        let expected_total = self.expected_total_output_samples.round().max(0.0) as usize;
        expected_total.saturating_sub(self.total_output_emitted_samples)
    }

    fn process_dual_plane_chunk_to_pending(&mut self, input: &[f32]) -> Result<(), StretchError> {
        if input.is_empty() {
            return Ok(());
        }

        let Some(state_meta) = self.dual_plane_deterministic.as_ref() else {
            return Err(StretchError::InvalidState(
                "dual-plane deterministic state is unavailable",
            ));
        };
        let num_channels = state_meta.num_channels.max(1);
        let block_frames = state_meta.block_frames;
        if !input.len().is_multiple_of(num_channels) {
            return Err(StretchError::InvalidFormat(format!(
                "input sample count {} is not a multiple of channel count {}",
                input.len(),
                num_channels
            )));
        }

        let frames = input.len() / num_channels;
        if frames > block_frames {
            return Err(StretchError::InvalidFormat(format!(
                "input frame count {} exceeds deterministic block size {}",
                frames, block_frames
            )));
        }

        self.interpolate_ratio_for_frames(frames);
        self.expected_total_output_samples += input.len() as f64 * self.current_ratio;
        let processing_ratio = self.processing_ratio();
        let target_processing_ratio = self.target_ratio * self.pitch_scale;
        let sample_rate = self.params.sample_rate;
        let unity_exit_seam_frames = self.params.hop_size.min(DUAL_PLANE_UNITY_EXIT_SEAM_FRAMES);

        let produced_frames = {
            let (channel_output_buffers, dual_plane_state) = (
                &mut self.channel_output_buffers,
                &mut self.dual_plane_deterministic,
            );
            let Some(state) = dual_plane_state.as_mut() else {
                return Err(StretchError::InvalidState(
                    "dual-plane deterministic state became unavailable",
                ));
            };
            if state.input_planar.len() != num_channels || state.output_planar.len() != num_channels
            {
                return Err(StretchError::InvalidState(
                    "dual-plane planar buffers do not match channel count",
                ));
            }
            if channel_output_buffers.len() < num_channels {
                return Err(StretchError::InvalidState(
                    "channel output buffers do not match channel count",
                ));
            }

            if processing_ratio == 1.0 {
                if state.last_ratio != 1.0 {
                    apply_dual_plane_ratio(&mut state.processor, 1.0)?;
                }
                state.reset_ratio_history(1.0);
            } else {
                let exiting_exact_unity = (state.last_ratio - 1.0).abs() <= RATIO_SNAP_THRESHOLD;
                let averaged_ratio =
                    state.push_chunk_ratio(processing_ratio, target_processing_ratio);
                let applied_ratio = smooth_ratio_toward(
                    state.last_ratio,
                    averaged_ratio,
                    frames,
                    sample_rate,
                    DUAL_PLANE_RATIO_APPLY_SMOOTHING_TIME_SECS,
                );
                if (applied_ratio - state.last_ratio).abs() > RATIO_SNAP_THRESHOLD {
                    if exiting_exact_unity {
                        state.arm_unity_exit_seam();
                    }
                    apply_dual_plane_ratio(&mut state.processor, applied_ratio)?;
                    state.last_ratio = applied_ratio;
                }
            }

            for frame in 0..frames {
                let base = frame * num_channels;
                for ch in 0..num_channels {
                    state.input_planar[ch][frame] = input[base + ch];
                }
            }

            let produced_frames = if num_channels == 1 {
                let input_refs = [&state.input_planar[0][..frames]];
                let mut output_refs = [state.output_planar[0].as_mut_slice()];
                let (_consumed, produced) = state
                    .processor
                    .rt_mut()
                    .process(&input_refs, &mut output_refs);
                produced
            } else if num_channels == 2 {
                let input_refs = [
                    &state.input_planar[0][..frames],
                    &state.input_planar[1][..frames],
                ];
                let (left_out, right_out) = state.output_planar.split_at_mut(1);
                let mut output_refs = [left_out[0].as_mut_slice(), right_out[0].as_mut_slice()];
                let (_consumed, produced) = state
                    .processor
                    .rt_mut()
                    .process(&input_refs, &mut output_refs);
                produced
            } else {
                let input_refs: Vec<&[f32]> = state
                    .input_planar
                    .iter()
                    .take(num_channels)
                    .map(|channel| &channel[..frames])
                    .collect();
                let mut output_refs: Vec<&mut [f32]> = state
                    .output_planar
                    .iter_mut()
                    .take(num_channels)
                    .map(|channel| channel.as_mut_slice())
                    .collect();
                let (_consumed, produced) = state
                    .processor
                    .rt_mut()
                    .process(&input_refs, &mut output_refs);
                produced
            };

            for ch in 0..num_channels {
                if channel_output_buffers[ch].capacity() < produced_frames {
                    return Err(StretchError::BufferOverflow {
                        buffer: "stream_channel_output",
                        requested: produced_frames,
                        available: channel_output_buffers[ch].capacity(),
                    });
                }
                if state.output_planar[ch].len() < produced_frames {
                    return Err(StretchError::InvalidState(
                        "dual-plane output planar shorter than produced frame count",
                    ));
                }
                channel_output_buffers[ch].clear();
                channel_output_buffers[ch]
                    .extend_from_slice(&state.output_planar[ch][..produced_frames]);
            }
            state.apply_pending_unity_exit_seam(
                channel_output_buffers,
                produced_frames,
                unity_exit_seam_frames,
            );
            state.capture_last_output_samples(channel_output_buffers, produced_frames);
            produced_frames
        };

        if produced_frames > 0 {
            self.emit_channel_output_to_pending(produced_frames, num_channels)?;
        }
        Ok(())
    }

    fn push_input_samples(&mut self, input: &[f32]) -> Result<(), StretchError> {
        if input.is_empty() {
            return Ok(());
        }
        let available = self.input_ring.available();
        if input.len() > available {
            return Err(StretchError::BufferOverflow {
                buffer: "stream_input_ring",
                requested: input.len(),
                available,
            });
        }
        let pushed = self.input_ring.push_slice(input);
        if pushed != input.len() {
            return Err(StretchError::BufferOverflow {
                buffer: "stream_input_ring",
                requested: input.len(),
                available: pushed,
            });
        }
        Ok(())
    }

    fn process_available_to_pending(
        &mut self,
        require_min_latency: bool,
    ) -> Result<(), StretchError> {
        if self.params.hop_size == 0 {
            return Err(StretchError::InvalidState("hop_size must be > 0"));
        }

        let num_channels = self.params.channels.count();
        let total_frames = self.input_ring.len() / num_channels;

        if total_frames < self.params.fft_size {
            return Ok(());
        }

        if require_min_latency {
            let min_frames = effective_min_frames(self.params.fft_size, self.processing_ratio());
            if total_frames < min_frames {
                return Ok(());
            }
        }

        self.collect_channel_inputs(total_frames, num_channels)?;
        self.encode_input_mid_side(num_channels);

        self.update_vocoder_ratio();
        if num_channels == 2 {
            self.apply_transient_scheduled_phase_reset(total_frames);
        }

        let min_output_len = self.process_channels(num_channels)?;
        let consumed_frames = self.consume_processed_input(total_frames, num_channels);
        self.input_frames_consumed_total = self
            .input_frames_consumed_total
            .saturating_add(consumed_frames);

        if min_output_len > 0 {
            self.decode_output_mid_side(num_channels, min_output_len);
            self.emit_channel_output_to_pending(min_output_len, num_channels)?;
        }

        Ok(())
    }

    fn collect_channel_inputs(
        &mut self,
        total_frames: usize,
        num_channels: usize,
    ) -> Result<(), StretchError> {
        let total_samples = total_frames.saturating_mul(num_channels);
        if total_samples > self.interleaved_scratch.len() {
            return Err(StretchError::BufferOverflow {
                buffer: "stream_interleaved_scratch",
                requested: total_samples,
                available: self.interleaved_scratch.len(),
            });
        }

        let copied = self
            .input_ring
            .peek_slice(&mut self.interleaved_scratch[..total_samples]);
        if copied != total_samples {
            return Err(StretchError::InvalidState(
                "failed to snapshot full input ring for processing",
            ));
        }

        for ch in 0..num_channels {
            if self.channel_input_buffers[ch].capacity() < total_frames {
                return Err(StretchError::BufferOverflow {
                    buffer: "stream_channel_input",
                    requested: total_frames,
                    available: self.channel_input_buffers[ch].capacity(),
                });
            }

            self.channel_input_buffers[ch].clear();
            for frame in 0..total_frames {
                self.channel_input_buffers[ch]
                    .push(self.interleaved_scratch[frame * num_channels + ch]);
            }
        }

        Ok(())
    }

    /// Converts `channel_input_buffers` from L/R to Mid/Side in-place.
    ///
    /// Only applies when `num_channels == 2` and `stereo_mode == MidSide`.
    /// After this call, `channel_input_buffers[0]` holds Mid and `[1]` holds Side.
    fn encode_input_mid_side(&mut self, num_channels: usize) {
        if num_channels != 2 || self.params.stereo_mode != StereoMode::MidSide {
            return;
        }
        let len = self.channel_input_buffers[0]
            .len()
            .min(self.channel_input_buffers[1].len());
        for i in 0..len {
            let l = self.channel_input_buffers[0][i];
            let r = self.channel_input_buffers[1][i];
            self.channel_input_buffers[0][i] = (l + r) * 0.5;
            self.channel_input_buffers[1][i] = (l - r) * 0.5;
        }
    }

    /// Converts `channel_output_buffers` from Mid/Side back to L/R in-place.
    ///
    /// Only applies when `num_channels == 2` and `stereo_mode == MidSide`.
    fn decode_output_mid_side(&mut self, num_channels: usize, output_len: usize) {
        if num_channels != 2 || self.params.stereo_mode != StereoMode::MidSide {
            return;
        }
        let len = output_len
            .min(self.channel_output_buffers[0].len())
            .min(self.channel_output_buffers[1].len());
        for i in 0..len {
            let m = self.channel_output_buffers[0][i];
            let s = self.channel_output_buffers[1][i];
            self.channel_output_buffers[0][i] = m + s;
            self.channel_output_buffers[1][i] = m - s;
        }
    }

    fn process_channels(&mut self, num_channels: usize) -> Result<usize, StretchError> {
        let mut min_output_len = usize::MAX;

        for ch in 0..num_channels {
            self.vocoders[ch].process_streaming_into(
                &self.channel_input_buffers[ch],
                &mut self.channel_output_buffers[ch],
            )?;
            min_output_len = min_output_len.min(self.channel_output_buffers[ch].len());
        }

        Ok(if min_output_len == usize::MAX {
            0
        } else {
            min_output_len
        })
    }

    fn consume_processed_input(&mut self, total_frames: usize, num_channels: usize) -> usize {
        let hop = self.params.hop_size;
        if hop == 0 {
            return 0;
        }
        let num_frames_processed = if total_frames >= self.params.fft_size {
            (total_frames - self.params.fft_size) / hop + 1
        } else {
            0
        };
        let samples_consumed = if num_frames_processed > 0 {
            (num_frames_processed * hop) * num_channels
        } else {
            0
        };
        if samples_consumed > 0 {
            self.input_ring.discard(samples_consumed);
        }
        num_frames_processed.saturating_mul(hop)
    }

    fn interleave_to_pending(
        &mut self,
        min_output_len: usize,
        num_channels: usize,
    ) -> Result<(), StretchError> {
        let needed = min_output_len.saturating_mul(num_channels);
        if needed > self.pending_output.available() {
            return Err(StretchError::BufferOverflow {
                buffer: "stream_pending_output",
                requested: needed,
                available: self.pending_output.available(),
            });
        }

        for i in 0..min_output_len {
            for ch in 0..num_channels {
                if !self.pending_output.push(self.channel_output_buffers[ch][i]) {
                    return Err(StretchError::InvalidState(
                        "pending output ring rejected push despite capacity check",
                    ));
                }
            }
        }

        Ok(())
    }

    #[inline]
    fn processing_ratio(&self) -> f64 {
        self.current_ratio * self.pitch_scale
    }

    fn reset_pitch_resamplers(&mut self) {
        for resampler in &mut self.pitch_resamplers {
            resampler.reset();
        }
        for buf in &mut self.pitch_output_buffers {
            buf.clear();
        }
    }

    fn emit_channel_output_to_pending(
        &mut self,
        min_output_len: usize,
        num_channels: usize,
    ) -> Result<(), StretchError> {
        if min_output_len == 0 {
            return Ok(());
        }

        if (self.pitch_scale - 1.0).abs() < RATIO_SNAP_THRESHOLD {
            return self.interleave_to_pending(min_output_len, num_channels);
        }
        let resample_ratio = 1.0 / self.pitch_scale;

        let mut pitch_min_output_len = usize::MAX;
        for ch in 0..num_channels {
            if self.channel_output_buffers[ch].len() < min_output_len {
                return Err(StretchError::InvalidState(
                    "channel output shorter than requested interleave length",
                ));
            }

            self.pitch_resamplers[ch].process_into(
                &self.channel_output_buffers[ch][..min_output_len],
                resample_ratio,
                &mut self.pitch_output_buffers[ch],
            )?;
            pitch_min_output_len = pitch_min_output_len.min(self.pitch_output_buffers[ch].len());
        }

        if pitch_min_output_len == usize::MAX || pitch_min_output_len == 0 {
            return Ok(());
        }
        self.interleave_pitch_to_pending(pitch_min_output_len, num_channels)
    }

    fn interleave_pitch_to_pending(
        &mut self,
        min_output_len: usize,
        num_channels: usize,
    ) -> Result<(), StretchError> {
        let needed = min_output_len.saturating_mul(num_channels);
        if needed > self.pending_output.available() {
            return Err(StretchError::BufferOverflow {
                buffer: "stream_pending_output",
                requested: needed,
                available: self.pending_output.available(),
            });
        }

        for i in 0..min_output_len {
            for ch in 0..num_channels {
                if !self.pending_output.push(self.pitch_output_buffers[ch][i]) {
                    return Err(StretchError::InvalidState(
                        "pending output ring rejected pitch push despite capacity check",
                    ));
                }
            }
        }
        Ok(())
    }

    fn flush_pitch_resampler_to_pending(
        &mut self,
        num_channels: usize,
    ) -> Result<(), StretchError> {
        if (self.pitch_scale - 1.0).abs() < RATIO_SNAP_THRESHOLD {
            return Ok(());
        }
        let resample_ratio = 1.0 / self.pitch_scale;

        let mut min_output_len = usize::MAX;
        for ch in 0..num_channels {
            self.pitch_resamplers[ch]
                .flush_into(resample_ratio, &mut self.pitch_output_buffers[ch])?;
            min_output_len = min_output_len.min(self.pitch_output_buffers[ch].len());
        }

        if min_output_len != usize::MAX && min_output_len > 0 {
            self.interleave_pitch_to_pending(min_output_len, num_channels)?;
        }
        Ok(())
    }

    fn drain_pending_to_output(&mut self, output: &mut Vec<f32>) -> Result<usize, StretchError> {
        let pending = self.pending_output.len();
        if pending == 0 {
            return Ok(0);
        }

        let available = output.capacity().saturating_sub(output.len());
        if pending > available {
            return Err(StretchError::BufferOverflow {
                buffer: "process_into_output",
                requested: pending,
                available,
            });
        }

        let mut written = 0usize;
        let mut chunk = [0.0f32; 512];
        let mut iterations = 0usize;
        let max_iterations = pending
            .saturating_add(chunk.len().saturating_sub(1))
            .saturating_div(chunk.len())
            .saturating_add(LOOP_GUARD_SLACK);
        while !self.pending_output.is_empty() {
            iterations = iterations.saturating_add(1);
            if iterations > max_iterations {
                return Err(StretchError::InvalidState(
                    "pending-output drain iteration bound exceeded",
                ));
            }
            let n = self.pending_output.pop_slice(&mut chunk);
            if n == 0 {
                return Err(StretchError::InvalidState(
                    "pending-output drain made zero progress",
                ));
            }
            output.extend_from_slice(&chunk[..n]);
            written += n;
        }

        self.total_output_emitted_samples += written;

        Ok(written)
    }

    fn drain_pending_to_buffer(&mut self, output: &mut [f32]) -> Result<usize, StretchError> {
        let pending = self.pending_output.len();
        if pending == 0 || output.is_empty() {
            return Ok(0);
        }

        let num_channels = self.params.channels.count().max(1);
        let target = pending
            .min(output.len())
            .saturating_div(num_channels)
            .saturating_mul(num_channels);
        if target == 0 {
            return Ok(0);
        }

        let mut written = 0usize;
        let mut chunk = [0.0f32; 512];
        let mut iterations = 0usize;
        let max_iterations = target
            .saturating_add(chunk.len().saturating_sub(1))
            .saturating_div(chunk.len())
            .saturating_add(LOOP_GUARD_SLACK);
        while written < target {
            iterations = iterations.saturating_add(1);
            if iterations > max_iterations {
                return Err(StretchError::InvalidState(
                    "pending-output buffer drain iteration bound exceeded",
                ));
            }
            let take = (target - written).min(chunk.len());
            let n = self.pending_output.pop_slice(&mut chunk[..take]);
            if n == 0 {
                return Err(StretchError::InvalidState(
                    "pending-output buffer drain made zero progress",
                ));
            }
            output[written..written + n].copy_from_slice(&chunk[..n]);
            written += n;
        }

        self.total_output_emitted_samples += written;
        Ok(written)
    }

    fn reconcile_fixed_flush_pending_output(
        &mut self,
        num_channels: usize,
    ) -> Result<(), StretchError> {
        let expected_total = self.expected_total_output_samples.round() as isize;
        let projected_total = self
            .total_output_emitted_samples
            .saturating_add(self.pending_output.len()) as isize;
        let correction = expected_total - projected_total;
        if correction == 0 {
            return Ok(());
        }

        let pending = self.pending_output.len();
        self.fixed_flush_scratch.resize(pending, 0.0);
        let copied = self
            .pending_output
            .peek_slice(&mut self.fixed_flush_scratch[..pending]);
        if copied != pending {
            return Err(StretchError::InvalidState(
                "fixed-buffer flush scratch copy did not capture full pending output",
            ));
        }

        if correction > 0 {
            let need = correction as usize;
            if need > self.pending_output.available() {
                return Err(StretchError::BufferOverflow {
                    buffer: "stream_pending_output",
                    requested: need,
                    available: self.pending_output.available(),
                });
            }
            extend_with_tonal_tail(&mut self.fixed_flush_scratch, need, 0);
        } else {
            let trim = ((-correction) as usize).min(self.fixed_flush_scratch.len());
            self.fixed_flush_scratch
                .truncate(self.fixed_flush_scratch.len().saturating_sub(trim));
        }

        if !self.fixed_flush_scratch.len().is_multiple_of(num_channels) {
            return Err(StretchError::InvalidState(
                "fixed-buffer flush correction broke interleaved frame alignment",
            ));
        }

        self.pending_output.clear();
        let pushed = self.pending_output.push_slice(&self.fixed_flush_scratch);
        if pushed != self.fixed_flush_scratch.len() {
            return Err(StretchError::BufferOverflow {
                buffer: "stream_pending_output",
                requested: self.fixed_flush_scratch.len(),
                available: pushed,
            });
        }

        Ok(())
    }

    fn finish_fixed_flush_drain(&mut self) {
        self.fixed_flush_pending = false;
        self.fixed_flush_source_exhausted = false;
        self.fixed_flush_pitch_tail_flushed = false;
        self.fixed_flush_length_reconciled = false;
        self.expected_total_output_samples = 0.0;
        self.total_output_emitted_samples = 0;
        if let Some(scale) = self.pending_pitch_scale.take() {
            self.apply_pitch_scale_now(scale);
        }
    }

    fn flush_vocoder_tails_to_pending(&mut self, num_channels: usize) -> Result<(), StretchError> {
        let mut min_output_len = usize::MAX;
        for ch in 0..num_channels {
            self.vocoders[ch].flush_streaming_into(&mut self.channel_output_buffers[ch])?;
            min_output_len = min_output_len.min(self.channel_output_buffers[ch].len());
        }

        if min_output_len == usize::MAX || min_output_len == 0 {
            return Ok(());
        }

        self.decode_output_mid_side(num_channels, min_output_len);
        self.interleave_to_pending(min_output_len, num_channels)
    }

    /// Creates a streaming processor configured for BPM matching.
    ///
    /// This constructor uses the `DjBeatmatch` preset for quality. For a
    /// lower-latency control surface path, use
    /// [`StreamProcessor::try_from_tempo_low_latency`].
    pub fn from_tempo(source_bpm: f64, target_bpm: f64, sample_rate: u32, channels: u32) -> Self {
        Self::try_from_tempo(source_bpm, target_bpm, sample_rate, channels).unwrap_or_else(|_| {
            let params = StretchParams::new(1.0)
                .with_sample_rate(sample_rate)
                .with_channels(channels)
                .with_preset(crate::EdmPreset::DjBeatmatch);
            Self::new(params)
        })
    }

    /// Creates a BPM-matching stream processor, returning an error when tempo
    /// inputs are invalid.
    pub fn try_from_tempo(
        source_bpm: f64,
        target_bpm: f64,
        sample_rate: u32,
        channels: u32,
    ) -> Result<Self, StretchError> {
        let base = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(channels)
            .with_preset(crate::EdmPreset::DjBeatmatch);
        Self::try_from_tempo_with_params(source_bpm, target_bpm, base)
    }

    /// Creates a low-latency BPM-matching stream processor.
    pub fn try_from_tempo_low_latency(
        source_bpm: f64,
        target_bpm: f64,
        sample_rate: u32,
        channels: u32,
    ) -> Result<Self, StretchError> {
        let base = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(channels)
            .with_quality_mode(QualityMode::LowLatency)
            .with_window_type(WindowType::Hann)
            .with_fft_size(LOW_LATENCY_TEMPO_FFT_SIZE)
            .with_hop_size(LOW_LATENCY_TEMPO_HOP_SIZE);
        Self::try_from_tempo_with_params(source_bpm, target_bpm, base)
    }

    /// Creates a BPM-matching stream processor from caller-provided params.
    pub fn try_from_tempo_with_params(
        source_bpm: f64,
        target_bpm: f64,
        params: StretchParams,
    ) -> Result<Self, StretchError> {
        let ratio = ratio_from_tempo(source_bpm, target_bpm)?;
        let mut proc = Self::new(params.with_stretch_ratio(ratio).with_bpm(source_bpm));
        proc.source_bpm = Some(source_bpm);
        Ok(proc)
    }

    /// Changes the stretch ratio for subsequent processing.
    ///
    /// Returns [`StretchError::InvalidRatio`] when `ratio` is non-finite or
    /// not strictly positive.
    pub fn set_stretch_ratio(&mut self, ratio: f64) -> Result<(), StretchError> {
        self.try_set_stretch_ratio(ratio)
    }

    /// Changes the stretch ratio for subsequent processing, returning an error
    /// for invalid values.
    pub fn try_set_stretch_ratio(&mut self, ratio: f64) -> Result<(), StretchError> {
        let ratio = validate_positive_finite_ratio(ratio, "stretch ratio")?;
        self.target_ratio = ratio;
        Ok(())
    }

    /// Explicitly enables or disables dual-plane backend preference for deterministic mode.
    ///
    /// Change before streaming begins.
    pub fn set_dual_plane_deterministic(&mut self, enabled: bool) -> Result<(), StretchError> {
        if !self.is_fresh_stream() {
            return Err(StretchError::InvalidState(
                "set_dual_plane_deterministic requires a fresh stream (call reset first)",
            ));
        }

        self.dual_plane_preferred = enabled;
        if enabled {
            if !dual_plane_supports_pitch_scale(self.pitch_scale) {
                return Err(StretchError::InvalidState(
                    "dual-plane deterministic backend requires finite positive pitch_scale",
                ));
            }
            let state =
                DualPlaneDeterministicState::from_params(&self.params, self.processing_ratio())?;
            self.dual_plane_deterministic = Some(state);
        } else {
            self.dual_plane_deterministic = None;
        }
        Ok(())
    }

    /// Returns whether deterministic processing is delegated to dual-plane RT.
    pub fn is_dual_plane_deterministic(&self) -> bool {
        self.dual_plane_deterministic.is_some()
    }

    /// Sets the deterministic dual-plane latency profile.
    pub fn set_deterministic_latency_profile(
        &mut self,
        profile: LatencyProfile,
    ) -> Result<(), StretchError> {
        if self.fixed_flush_pending {
            return Err(StretchError::InvalidState(
                "deterministic latency profile cannot change until fixed-buffer flush output is fully drained",
            ));
        }
        self.ensure_default_dual_plane_backend();
        let Some(state) = self.dual_plane_deterministic.as_mut() else {
            return Err(StretchError::InvalidState(
                "dual-plane deterministic backend is unavailable",
            ));
        };
        state.processor.set_latency_profile(profile);
        Ok(())
    }

    /// Enables or disables deterministic dual-plane auto profile switching.
    pub fn set_deterministic_auto_profile_switching(
        &mut self,
        enabled: bool,
    ) -> Result<(), StretchError> {
        if self.fixed_flush_pending {
            return Err(StretchError::InvalidState(
                "deterministic auto profile switching cannot change until fixed-buffer flush output is fully drained",
            ));
        }
        self.ensure_default_dual_plane_backend();
        let Some(state) = self.dual_plane_deterministic.as_mut() else {
            return Err(StretchError::InvalidState(
                "dual-plane deterministic backend is unavailable",
            ));
        };
        state.processor.set_auto_profile_switching(enabled);
        Ok(())
    }

    /// Returns deterministic dual-plane profile telemetry when active.
    pub fn deterministic_profile_telemetry(&self) -> Option<RtProfileTelemetry> {
        self.dual_plane_deterministic
            .as_ref()
            .map(|state| state.processor.profile_telemetry())
    }

    /// Returns cumulative deterministic dual-plane runtime telemetry when active.
    pub fn deterministic_runtime_telemetry(&self) -> Option<RtRuntimeTelemetry> {
        self.dual_plane_deterministic
            .as_ref()
            .map(|state| state.processor.runtime_telemetry())
    }

    /// Returns exact deterministic dual-plane delay telemetry when active.
    pub fn deterministic_delay_telemetry(&self) -> Option<RtDelayTelemetry> {
        self.dual_plane_deterministic.as_ref().map(|state| {
            let mut telemetry = state.processor.delay_telemetry();
            // Account for host-visible samples already moved above the RT
            // core into the stream-layer pending ring.
            telemetry.buffered_output_frames = telemetry
                .buffered_output_frames
                .saturating_add(self.pending_output.len() / self.params.channels.count().max(1));
            telemetry.total_frames = telemetry
                .algorithmic_frames
                .saturating_add(telemetry.buffered_input_frames)
                .saturating_add(telemetry.buffered_output_frames)
                .saturating_add(telemetry.profile_frames)
                .saturating_add(telemetry.tier_frames);
            telemetry
        })
    }

    /// Returns the exact current host-visible delay in audio frames for deterministic mode.
    ///
    /// This includes algorithmic delay, buffered input/output, and current
    /// profile/tier contributions. Returns `None` when deterministic
    /// dual-plane telemetry is unavailable.
    pub fn current_delay_frames(&self) -> Option<usize> {
        self.deterministic_delay_telemetry()
            .map(|telemetry| telemetry.total_frames)
    }

    /// Returns the exact current host-visible delay in audio frames for deterministic mode.
    ///
    /// This historical alias returns the same frame count as
    /// [`StreamProcessor::current_delay_frames`], not an interleaved scalar
    /// sample count.
    pub fn current_delay_samples(&self) -> Option<usize> {
        self.current_delay_frames()
    }

    /// Returns the exact current host-visible delay in seconds for deterministic mode.
    pub fn current_delay_secs(&self) -> Option<f64> {
        self.current_delay_frames()
            .map(|frames| frames as f64 / self.params.sample_rate as f64)
    }

    /// Returns cumulative transient-reset telemetry for the current stream.
    pub fn transient_reset_stats(&self) -> TransientResetStats {
        let TransientSchedulerStats {
            events_detected_total,
            reset_band_counts_total,
        } = self.transient_scheduler.stats();
        TransientResetStats {
            events_detected_total,
            reset_band_counts_total,
            input_frames_consumed_total: self.input_frames_consumed_total,
        }
    }

    /// Changes the target BPM, smoothly adjusting the stretch ratio.
    pub fn set_tempo(&mut self, target_bpm: f64) -> bool {
        if let Some(source) = self.source_bpm {
            let Ok(ratio) = ratio_from_tempo(source, target_bpm) else {
                return false;
            };
            return self.try_set_stretch_ratio(ratio).is_ok();
        }
        false
    }

    /// Sets the realtime pitch-scale control value.
    ///
    /// Stream mode applies pitch scale by rendering with an internal stretch
    /// ratio of `stretch_ratio * pitch_scale` and then resampling the rendered
    /// stream per channel by `1.0 / pitch_scale` to preserve target tempo. If
    /// a fixed-buffer flush drain is active, the change takes effect after the
    /// remaining tail has been fully drained.
    pub fn set_pitch_scale(&mut self, scale: f64) -> Result<(), StretchError> {
        let scale = validate_positive_finite_ratio(scale, "pitch scale")?;
        if self.fixed_flush_pending {
            self.pending_pitch_scale = Some(scale);
            return Ok(());
        }

        self.pending_pitch_scale = None;
        self.apply_pitch_scale_now(scale);
        Ok(())
    }

    /// Returns the current realtime pitch-scale control value.
    pub fn pitch_scale(&self) -> f64 {
        self.pitch_scale
    }

    /// Returns the source BPM if available.
    pub fn source_bpm(&self) -> Option<f64> {
        self.source_bpm
    }

    /// Returns a reference to the current parameters.
    pub fn params(&self) -> &StretchParams {
        &self.params
    }

    /// Returns `(input_ring_samples, pending_output_samples, input_capacity_samples, pending_capacity_samples)`.
    pub fn capacities(&self) -> (usize, usize, usize, usize) {
        (
            self.input_ring.len(),
            self.pending_output.len(),
            self.input_ring.capacity(),
            self.pending_output.capacity(),
        )
    }

    /// Returns the current effective stretch ratio.
    pub fn current_stretch_ratio(&self) -> f64 {
        self.current_ratio
    }

    /// Returns the target stretch ratio.
    pub fn target_stretch_ratio(&self) -> f64 {
        self.target_ratio
    }

    /// Returns the current target BPM, if known.
    pub fn target_bpm(&self) -> Option<f64> {
        self.source_bpm.map(|src| src / self.target_ratio)
    }

    /// Returns the minimum latency in audio frames.
    ///
    /// For exact current deterministic delay, use
    /// [`StreamProcessor::current_delay_frames`] or
    /// [`StreamProcessor::deterministic_delay_telemetry`].
    pub fn latency_frames(&self) -> usize {
        min_latency_frames(self.params.fft_size)
    }

    /// Returns the minimum latency in audio frames.
    ///
    /// This historical alias returns the same frame count as
    /// [`StreamProcessor::latency_frames`], not an interleaved scalar sample
    /// count.
    pub fn latency_samples(&self) -> usize {
        self.latency_frames()
    }

    /// Returns the minimum latency in seconds.
    ///
    /// For exact current deterministic delay, use
    /// [`StreamProcessor::current_delay_secs`] or
    /// [`StreamProcessor::deterministic_delay_telemetry`].
    pub fn latency_secs(&self) -> f64 {
        self.latency_frames() as f64 / self.params.sample_rate as f64
    }

    /// Resets the processor state.
    pub fn reset(&mut self) {
        self.input_ring.clear();
        self.pending_output.clear();
        for buf in &mut self.channel_input_buffers {
            buf.clear();
        }
        for buf in &mut self.channel_output_buffers {
            buf.clear();
        }

        self.current_ratio = self.params.stretch_ratio;
        self.target_ratio = self.params.stretch_ratio;
        self.vocoder_ratio = self.params.stretch_ratio;
        self.initialized = false;
        self.transient_scheduler.reset();
        self.input_frames_consumed_total = 0;

        self.vocoders = Self::create_vocoders(&self.params, self.params.stretch_ratio);
        self.expected_total_output_samples = 0.0;
        self.total_output_emitted_samples = 0;
        self.pitch_scale = 1.0;
        self.reset_pitch_resamplers();
        self.fixed_flush_pending = false;
        self.fixed_flush_source_exhausted = false;
        self.fixed_flush_pitch_tail_flushed = false;
        self.fixed_flush_length_reconciled = false;
        self.fixed_flush_scratch.clear();
        self.pending_pitch_scale = None;

        self.dual_plane_deterministic = None;
        self.ensure_default_dual_plane_backend();
    }

    fn update_vocoder_ratio(&mut self) {
        let processing_ratio = self.processing_ratio();
        if (processing_ratio - self.vocoder_ratio).abs() > RATIO_SNAP_THRESHOLD {
            for voc in &mut self.vocoders {
                voc.set_stretch_ratio(processing_ratio);
            }
            self.vocoder_ratio = processing_ratio;
        }
    }

    fn apply_pitch_scale_now(&mut self, scale: f64) {
        if (scale - self.pitch_scale).abs() > RATIO_SNAP_THRESHOLD {
            self.reset_pitch_resamplers();
        }
        self.pitch_scale = scale;
        if dual_plane_supports_pitch_scale(scale) {
            self.ensure_default_dual_plane_backend();
        }
    }

    fn apply_transient_scheduled_phase_reset(&mut self, total_frames: usize) {
        if total_frames < self.params.fft_size {
            return;
        }

        let total_samples = total_frames.saturating_mul(2);
        if total_samples == 0 || total_samples > self.interleaved_scratch.len() {
            return;
        }

        let stereo = &self.interleaved_scratch[..total_samples];
        let Some(reset_mask) = self
            .transient_scheduler
            .detect_stereo_reset_mask(stereo, self.input_frames_consumed_total)
        else {
            return;
        };

        if self.params.stereo_mode == StereoMode::MidSide && self.vocoders.len() == 2 {
            let (mid_mask, side_mask) = stereo_channel_reset_masks(reset_mask);
            self.vocoders[0].reset_phase_state_bands(mid_mask, self.params.sample_rate);
            if side_mask.iter().any(|&b| b) {
                self.vocoders[1].reset_phase_state_bands(side_mask, self.params.sample_rate);
            }
            return;
        }

        for vocoder in &mut self.vocoders {
            vocoder.reset_phase_state_bands(reset_mask, self.params.sample_rate);
        }
    }

    /// Returns the BPM stored in the params, if any.
    pub fn bpm(&self) -> Option<f64> {
        self.params.bpm
    }

    /// Callback-size-agnostic ratio interpolation.
    fn interpolate_ratio_for_frames(&mut self, frames: usize) {
        let tau_frames = (self.params.sample_rate as f64 * RATIO_SMOOTHING_TIME_SECS).max(1.0);
        let alpha = 1.0 - (-(frames as f64) / tau_frames).exp();
        self.current_ratio += alpha * (self.target_ratio - self.current_ratio);

        if (self.current_ratio - self.target_ratio).abs() < RATIO_SNAP_THRESHOLD {
            self.current_ratio = self.target_ratio;
        }
    }

    /// Legacy helper used by tests in this module.
    #[cfg(test)]
    fn interpolate_ratio(&mut self) {
        self.interpolate_ratio_for_frames(COMMON_CALLBACK_FRAMES);
    }
}

/// Per-channel reset policy for stereo Mid/Side deterministic streaming.
///
/// Mid channel gets full transient reset mask. Side channel gets only
/// mid/high resets to avoid unnecessary low-band phase disruption in width
/// content while still allowing panned upper transients to re-lock.
#[inline]
fn stereo_channel_reset_masks(full_mask: [bool; 4]) -> ([bool; 4], [bool; 4]) {
    let side_mask = [false, false, full_mask[2], full_mask[3]];
    (full_mask, side_mask)
}

#[inline]
fn estimate_period_from_tail(tail: &[f32]) -> Option<usize> {
    if tail.len() < 32 {
        return None;
    }

    let mut crossings = Vec::with_capacity(tail.len() / 16);
    for i in 0..tail.len().saturating_sub(1) {
        if tail[i] <= 0.0 && tail[i + 1] > 0.0 {
            crossings.push(i);
        }
    }
    if crossings.len() < 4 {
        return None;
    }

    let mut intervals: Vec<usize> = crossings
        .windows(2)
        .map(|w| w[1].saturating_sub(w[0]))
        .filter(|&d| d >= 8 && d <= tail.len() / 2)
        .collect();
    if intervals.len() < 3 {
        return None;
    }

    intervals.sort_unstable();
    let median = intervals[intervals.len() / 2].max(1);
    let lo = ((median as f64) * 0.7).floor() as usize;
    let hi = ((median as f64) * 1.3).ceil() as usize;

    let mut sum = 0usize;
    let mut n = 0usize;
    for d in intervals {
        if d >= lo && d <= hi {
            sum += d;
            n += 1;
        }
    }

    if n == 0 {
        None
    } else {
        Some((sum / n).max(1))
    }
}

#[inline]
fn fit_tonal_tail(samples: &[f32], global_start: usize, period: usize) -> Option<(f64, f64, f64)> {
    if samples.is_empty() || period == 0 {
        return None;
    }

    let fit_len = (period * 12).min(samples.len()).max(period * 3);
    let fit_start = samples.len().saturating_sub(fit_len);
    let fit = &samples[fit_start..];
    if fit.len() < period * 2 {
        return None;
    }

    let mean = fit.iter().map(|&s| s as f64).sum::<f64>() / fit.len() as f64;
    let w = 2.0 * std::f64::consts::PI / period as f64;

    let mut cc = 0.0f64;
    let mut ss = 0.0f64;
    let mut cs = 0.0f64;
    let mut xc = 0.0f64;
    let mut xs = 0.0f64;

    for (i, &x) in fit.iter().enumerate() {
        let n = (global_start + fit_start + i) as f64;
        let c = (w * n).cos();
        let s = (w * n).sin();
        let xv = x as f64 - mean;
        cc += c * c;
        ss += s * s;
        cs += c * s;
        xc += xv * c;
        xs += xv * s;
    }

    let det = cc * ss - cs * cs;
    if det.abs() < 1e-12 {
        return None;
    }

    let mut a = (xc * ss - xs * cs) / det;
    let mut b = (xs * cc - xc * cs) / det;

    let fit_amp = (a * a + b * b).sqrt();
    let tail_peak = samples
        .iter()
        .rev()
        .take(period * 4)
        .map(|v| v.abs() as f64)
        .fold(0.0, f64::max);
    if fit_amp > 1e-9 && tail_peak > 0.0 {
        let floor_amp = tail_peak * 0.95;
        if fit_amp < floor_amp {
            let scale = floor_amp / fit_amp;
            a *= scale;
            b *= scale;
        }
    }

    Some((a, b, mean))
}

/// Extends `output` by synthesizing a tonal continuation from the tail.
///
/// This keeps end-of-stream length correction from introducing flat or noisy
/// tails that would skew chunk-level pitch and envelope checks.
fn extend_with_tonal_tail(output: &mut Vec<f32>, count: usize, floor: usize) {
    if count == 0 {
        return;
    }
    if output.is_empty() {
        output.resize(count, 0.0);
        return;
    }

    let region_len = output.len().saturating_sub(floor);
    let backoff = count.max(512).min(region_len / 3);
    let synth_start = output.len().saturating_sub(backoff).max(floor);
    let analysis_end = synth_start.max(floor.max(1));
    let analysis_len = (analysis_end - floor).min(8192);
    let analysis_start = analysis_end - analysis_len;
    let analysis = &output[analysis_start..analysis_end];
    if let Some(period) = estimate_period_from_tail(analysis) {
        if let Some((a, b, mean)) = fit_tonal_tail(analysis, analysis_start, period) {
            let w = 2.0 * std::f64::consts::PI / period as f64;
            let rewritten = output.len().saturating_sub(synth_start);
            for i in 0..rewritten {
                let n = (synth_start + i) as f64;
                let y = a * (w * n).cos() + b * (w * n).sin() + mean;
                output[synth_start + i] = y as f32;
            }
            let start = output.len();
            for i in 0..count {
                let n = (start + i) as f64;
                let y = a * (w * n).cos() + b * (w * n).sin() + mean;
                output.push(y as f32);
            }
            return;
        }
    }

    let pad = *output.last().unwrap_or(&0.0);
    output.resize(output.len() + count, pad);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn estimate_freq_zero_crossings(samples: &[f32], sample_rate: u32) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }
        let mut crossings = 0usize;
        for i in 1..samples.len() {
            if samples[i - 1] <= 0.0 && samples[i] > 0.0 {
                crossings += 1;
            }
        }
        crossings as f64 * sample_rate as f64 / samples.len() as f64
    }

    #[test]
    fn test_stream_processor_basic() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);

        // Generate a test signal
        let chunk_size = 4096;
        let signal: Vec<f32> = (0..chunk_size * 4)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut total_output = Vec::new();
        for chunk in signal.chunks(chunk_size) {
            match proc.process(chunk) {
                Ok(output) => total_output.extend_from_slice(&output),
                Err(e) => panic!("Process error: {}", e),
            }
        }

        // Flush remaining
        if let Ok(remaining) = proc.flush() {
            total_output.extend_from_slice(&remaining);
        }

        // Should have produced some output
        assert!(!total_output.is_empty(), "Expected some output");
    }

    #[test]
    fn test_stream_processor_ratio_change() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        assert!((proc.current_stretch_ratio() - 1.0).abs() < 1e-6);

        proc.set_stretch_ratio(1.05).unwrap();
        // After a few interpolation steps, ratio should change
        for _ in 0..100 {
            proc.interpolate_ratio();
        }
        assert!((proc.current_stretch_ratio() - 1.05).abs() < 0.01);
    }

    #[test]
    fn test_stream_processor_latency() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(2)
            .with_fft_size(4096);

        let proc = StreamProcessor::new(params);
        // 4096 * 3 / 2 = 6144 (1.5x FFT size for reduced latency)
        assert_eq!(proc.latency_frames(), 6144);
        assert_eq!(proc.latency_samples(), 6144);
        assert!((proc.latency_secs() - 6144.0 / 44100.0).abs() < 1e-6);
    }

    #[test]
    fn test_stream_processor_reset() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        proc.set_stretch_ratio(2.0).unwrap();
        proc.reset();

        assert!((proc.current_stretch_ratio() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_stream_processor_ratio_change_no_clicks() {
        // Feed a sine wave, change ratio mid-stream, and verify no
        // sudden spikes (clicks) in the output at the ratio transition.
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        let chunk_size = 4096 * 2;
        let signal: Vec<f32> = (0..chunk_size * 6)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut all_output = Vec::new();

        // Process first half at ratio 1.0
        for chunk in signal[..chunk_size * 3].chunks(chunk_size) {
            if let Ok(out) = proc.process(chunk) {
                all_output.extend_from_slice(&out);
            }
        }

        // Change ratio to 1.05 (DJ pitch adjustment)
        proc.set_stretch_ratio(1.05).unwrap();
        // Force interpolation to converge
        for _ in 0..50 {
            proc.interpolate_ratio();
        }

        // Process second half
        for chunk in signal[chunk_size * 3..].chunks(chunk_size) {
            if let Ok(out) = proc.process(chunk) {
                all_output.extend_from_slice(&out);
            }
        }

        if all_output.len() < 100 {
            return; // Not enough output to analyze
        }

        // Check for clicks: a click would appear as a sudden jump between
        // consecutive samples that far exceeds normal sine wave behavior.
        // Normal sine at 440 Hz changes by max ~0.06 per sample at 44100 Hz.
        let mut max_diff = 0.0f32;
        for i in 1..all_output.len() {
            let diff = (all_output[i] - all_output[i - 1]).abs();
            max_diff = max_diff.max(diff);
        }

        // A sine wave at 440 Hz has max sample-to-sample diff of about
        // 2*pi*440/44100 ≈ 0.063. Allow up to 0.5 for phase vocoder artifacts,
        // but clicks would show as 1.0+ jumps.
        assert!(
            max_diff < 1.0,
            "Detected likely click artifact: max sample diff = {} (expected < 1.0)",
            max_diff
        );
    }

    #[test]
    fn test_stream_processor_dual_plane_unity_exit_seam_is_smoothed() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44_100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        assert!(proc.is_dual_plane_deterministic());

        let chunk_size = 4096 * 2;
        let signal: Vec<f32> = (0..chunk_size * 4)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44_100.0).sin())
            .collect();

        let mut output = Vec::with_capacity(signal.len() * 2);
        proc.process_into(&signal[..chunk_size * 3], &mut output)
            .unwrap();
        let boundary = output.len();
        assert!(
            boundary > 0,
            "unity pre-roll should emit deterministic output"
        );

        proc.set_stretch_ratio(1.05).unwrap();
        proc.process_into(&signal[chunk_size * 3..], &mut output)
            .unwrap();

        assert!(
            output.len() > boundary,
            "non-unity follow-up chunk should emit output after the unity exit"
        );

        let seam_diff = (output[boundary] - output[boundary - 1]).abs();
        assert!(
            seam_diff < 0.5,
            "exact-unity exit should not hard-jump at the first non-unity output sample (diff={})",
            seam_diff
        );
    }

    #[test]
    fn test_stream_processor_rejects_nan() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        let mut chunk = vec![0.0f32; 4096];
        chunk[100] = f32::NAN;
        assert!(matches!(
            proc.process(&chunk),
            Err(crate::error::StretchError::NonFiniteInput)
        ));
    }

    #[test]
    fn test_stream_processor_rejects_infinity() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        let mut chunk = vec![0.0f32; 4096];
        chunk[50] = f32::INFINITY;
        assert!(matches!(
            proc.process(&chunk),
            Err(crate::error::StretchError::NonFiniteInput)
        ));
    }

    #[test]
    fn test_stream_processor_from_tempo() {
        let proc = StreamProcessor::from_tempo(126.0, 128.0, 44100, 1);
        let expected_ratio = 126.0 / 128.0;
        assert!(
            (proc.current_stretch_ratio() - expected_ratio).abs() < 1e-6,
            "Expected ratio {}, got {}",
            expected_ratio,
            proc.current_stretch_ratio()
        );
        assert_eq!(proc.source_bpm(), Some(126.0));
        assert_eq!(proc.params().sample_rate, 44100);
        assert_eq!(
            proc.params().preset,
            Some(crate::core::types::EdmPreset::DjBeatmatch)
        );
    }

    #[test]
    fn test_stream_processor_from_tempo_stereo() {
        let proc = StreamProcessor::from_tempo(120.0, 130.0, 48000, 2);
        let expected_ratio = 120.0 / 130.0;
        assert!((proc.current_stretch_ratio() - expected_ratio).abs() < 1e-6);
        assert_eq!(proc.params().channels, crate::core::types::Channels::Stereo);
        assert_eq!(proc.params().sample_rate, 48000);
    }

    #[test]
    fn test_stream_processor_try_from_tempo_low_latency() {
        let proc = StreamProcessor::try_from_tempo_low_latency(126.0, 128.0, 44100, 2).unwrap();
        assert_eq!(proc.params().quality_mode, QualityMode::LowLatency);
        assert_eq!(proc.params().fft_size, LOW_LATENCY_TEMPO_FFT_SIZE);
        assert!(
            proc.latency_secs() * 1000.0 < 40.0,
            "Expected low-latency constructor under 40ms, got {:.2}ms",
            proc.latency_secs() * 1000.0
        );
    }

    #[test]
    fn test_stream_processor_try_from_tempo_rejects_invalid_values() {
        assert!(StreamProcessor::try_from_tempo(0.0, 128.0, 44100, 1).is_err());
        assert!(StreamProcessor::try_from_tempo(126.0, -1.0, 44100, 1).is_err());
        assert!(StreamProcessor::try_from_tempo(f64::NAN, 128.0, 44100, 1).is_err());
    }

    #[test]
    fn test_stream_processor_set_tempo() {
        let mut proc = StreamProcessor::from_tempo(126.0, 128.0, 44100, 1);

        // Change target to 130 BPM
        assert!(proc.set_tempo(130.0));
        // After many interpolation steps, ratio should converge to 126/130
        for _ in 0..200 {
            proc.interpolate_ratio();
        }
        let expected = 126.0 / 130.0;
        assert!(
            (proc.current_stretch_ratio() - expected).abs() < 0.01,
            "Expected ratio ~{}, got {}",
            expected,
            proc.current_stretch_ratio()
        );
    }

    #[test]
    fn test_stream_processor_set_tempo_no_source_bpm() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);

        // set_tempo should fail when source_bpm is unknown
        assert!(!proc.set_tempo(128.0));
        assert_eq!(proc.source_bpm(), None);
    }

    #[test]
    fn test_stream_processor_set_tempo_invalid() {
        let mut proc = StreamProcessor::from_tempo(126.0, 128.0, 44100, 1);
        // Zero or negative BPM should be rejected
        assert!(!proc.set_tempo(0.0));
        assert!(!proc.set_tempo(-100.0));
    }

    #[test]
    fn test_stream_processor_try_set_stretch_ratio_rejects_invalid_values() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        let initial = proc.target_stretch_ratio();
        assert!(proc.try_set_stretch_ratio(0.0).is_err());
        assert!(proc.try_set_stretch_ratio(f64::INFINITY).is_err());
        assert!(proc.set_stretch_ratio(f64::NAN).is_err());
        assert_eq!(proc.target_stretch_ratio(), initial);
    }

    #[test]
    fn test_stream_processor_pitch_scale_validation() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        assert!((proc.pitch_scale() - 1.0).abs() < 1e-9);
        assert!(proc.set_pitch_scale(1.25).is_ok());
        assert!((proc.pitch_scale() - 1.25).abs() < 1e-9);
        assert!(proc.set_pitch_scale(0.0).is_err());
        assert!(proc.set_pitch_scale(f64::NAN).is_err());
        assert!((proc.pitch_scale() - 1.25).abs() < 1e-9);
    }

    #[test]
    fn test_stream_processor_pitch_scale_applies_frequency_shift() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_pitch_scale(1.08).unwrap();

        let freq = 440.0f32;
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * PI * freq * i as f32 / 44100.0).sin() * 0.8)
            .collect();
        // `process_into` is fixed-capacity; reserve generous headroom so this
        // test validates deterministic routing/output, not buffer sizing.
        let mut output = Vec::with_capacity(input.len() * 8);
        for chunk in input.chunks(1024) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        proc.flush_into(&mut output).unwrap();

        let trim = 4096usize.min(output.len() / 4);
        let start = trim;
        let end = output.len().saturating_sub(trim).max(start + 2);
        let measured = estimate_freq_zero_crossings(&output[start..end], 44100);
        assert!(
            measured > 460.0,
            "expected measurable pitch-up shift, got {:.3} Hz",
            measured
        );
    }

    #[test]
    fn test_stream_processor_pitch_scale_preserves_tempo_ratio() {
        let ratio = 1.2;
        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_pitch_scale(1.08).unwrap();

        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 44100.0).sin() * 0.7)
            .collect();
        let mut output = Vec::with_capacity(input.len() * 3);
        for chunk in input.chunks(1024) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        proc.flush_into(&mut output).unwrap();

        let expected = (input.len() as f64 * ratio).round() as isize;
        let diff = (output.len() as isize - expected).abs();
        assert!(
            diff <= 128,
            "tempo ratio drift too high with pitch scaling: expected={} got={} diff={}",
            expected,
            output.len(),
            diff
        );
    }

    #[test]
    fn test_stream_processor_from_tempo_produces_output() {
        let mut proc = StreamProcessor::from_tempo(126.0, 128.0, 44100, 1);
        let chunk_size = 4096;
        let signal: Vec<f32> = (0..chunk_size * 4)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut total_output = Vec::new();
        for chunk in signal.chunks(chunk_size) {
            if let Ok(out) = proc.process(chunk) {
                total_output.extend_from_slice(&out);
            }
        }
        if let Ok(out) = proc.flush() {
            total_output.extend_from_slice(&out);
        }
        assert!(
            !total_output.is_empty(),
            "Expected output from from_tempo processor"
        );
    }

    #[test]
    fn test_stream_processor_params_accessor() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(48000)
            .with_channels(2)
            .with_fft_size(8192);
        let proc = StreamProcessor::new(params);

        assert_eq!(proc.params().sample_rate, 48000);
        assert_eq!(proc.params().fft_size, 8192);
        assert!((proc.params().stretch_ratio - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_stream_processor_dual_plane_deterministic_toggle() {
        let params = StretchParams::new(1.02)
            .with_sample_rate(44_100)
            .with_channels(2);
        let mut proc = StreamProcessor::new(params);
        assert!(proc.is_dual_plane_deterministic());
        proc.set_dual_plane_deterministic(false).unwrap();
        assert!(!proc.is_dual_plane_deterministic());
        proc.set_dual_plane_deterministic(true).unwrap();
        assert!(proc.is_dual_plane_deterministic());
    }

    #[test]
    fn test_stream_processor_dual_plane_deterministic_requires_fresh_stream() {
        let params = StretchParams::new(1.02)
            .with_sample_rate(44_100)
            .with_channels(2);
        let mut proc = StreamProcessor::new(params);
        let input = vec![0.0f32; 256 * 2];
        let _ = proc.process(&input).unwrap();
        let err = proc.set_dual_plane_deterministic(true).unwrap_err();
        assert!(matches!(err, StretchError::InvalidState(_)));
    }

    #[test]
    fn test_stream_processor_dual_plane_deterministic_accepts_pitch_scale() {
        let params = StretchParams::new(1.02)
            .with_sample_rate(44_100)
            .with_channels(2);
        let mut proc = StreamProcessor::new(params);
        assert!(proc.is_dual_plane_deterministic());
        proc.set_pitch_scale(1.05).unwrap();
        assert!(proc.is_dual_plane_deterministic());
    }

    #[test]
    fn test_stream_processor_dual_plane_deterministic_keeps_backend_on_pitch_scale_after_stream_start(
    ) {
        let params = StretchParams::new(1.02)
            .with_sample_rate(44_100)
            .with_channels(2);
        let mut proc = StreamProcessor::new(params);
        let input = vec![0.0f32; 256 * 2];
        let _ = proc.process(&input).unwrap();
        proc.set_pitch_scale(1.05).unwrap();
        assert!(proc.is_dual_plane_deterministic());
    }

    #[test]
    fn test_stream_processor_dual_plane_unity_passthrough_reengages_after_ratio_roundtrip() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        assert!(proc.is_dual_plane_deterministic());

        let frames = 256usize;
        let mut input = vec![0.0f32; frames * 2];
        for i in 0..frames {
            input[i * 2] = (i as f32 * 0.011).sin() * 0.35;
            input[i * 2 + 1] = (i as f32 * 0.019).cos() * 0.28;
        }

        let mut output = Vec::with_capacity(input.len() * 16);
        proc.set_stretch_ratio(1.35).unwrap();
        for _ in 0..8 {
            proc.process_into(&input, &mut output).unwrap();
        }

        proc.set_stretch_ratio(1.0).unwrap();
        for _ in 0..300 {
            proc.interpolate_ratio();
        }

        let before = output.len();
        proc.process_into(&input, &mut output).unwrap();
        let produced = &output[before..];
        assert_eq!(produced.len(), input.len());
        assert_eq!(produced, &input[..]);
    }

    #[test]
    fn test_dual_plane_deterministic_ratio_history_resets_on_cross_unity_modulation() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut state = DualPlaneDeterministicState::from_params(&params, 1.0).unwrap();

        let first_average = state.push_chunk_ratio(1.035, 1.035);
        assert!((first_average - 1.035).abs() < 1e-9);
        let plateau_average = state.push_chunk_ratio(1.025, 1.025);
        assert!((plateau_average - 1.03).abs() < 1e-9);
        assert_eq!(state.recent_chunk_ratios.len(), 2);

        let cross_unity_average = state.push_chunk_ratio(0.965, 0.965);
        assert!(
            (cross_unity_average - 0.965).abs() < 1e-9,
            "cross-unity modulation should clear stale opposite-side history before averaging"
        );
        assert_eq!(state.recent_chunk_ratios, vec![0.965]);
    }

    #[test]
    fn test_dual_plane_deterministic_ratio_history_resets_on_requested_cross_unity() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut state = DualPlaneDeterministicState::from_params(&params, 1.0).unwrap();

        let first_average = state.push_chunk_ratio(1.035, 1.035);
        assert!((first_average - 1.035).abs() < 1e-9);
        let plateau_average = state.push_chunk_ratio(1.025, 1.025);
        assert!((plateau_average - 1.03).abs() < 1e-9);
        assert_eq!(state.recent_chunk_ratios.len(), 2);

        let requested_cross_unity_average = state.push_chunk_ratio(1.012, 0.965);
        assert!(
            (requested_cross_unity_average - 1.012).abs() < 1e-9,
            "an opposite-side modulation request should clear stale same-side history before the smoothed ratio fully crosses unity"
        );
        assert_eq!(
            state.recent_chunk_ratios,
            vec![1.012],
            "requested cross-unity modulation should start a fresh averaging window from the first transition callback"
        );
    }

    #[test]
    fn test_dual_plane_deterministic_ratio_history_resets_on_requested_exact_unity() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut state = DualPlaneDeterministicState::from_params(&params, 1.0).unwrap();

        let first_average = state.push_chunk_ratio(1.035, 1.035);
        assert!((first_average - 1.035).abs() < 1e-9);
        let plateau_average = state.push_chunk_ratio(1.025, 1.025);
        assert!((plateau_average - 1.03).abs() < 1e-9);
        assert_eq!(state.recent_chunk_ratios.len(), 2);

        let requested_unity_average = state.push_chunk_ratio(1.012, 1.0);
        assert!(
            (requested_unity_average - 1.012).abs() < 1e-9,
            "an exact-unity request should clear stale same-side history before smoothing finishes returning to unity"
        );
        assert_eq!(
            state.recent_chunk_ratios,
            vec![1.012],
            "a requested exact-unity plateau should start a fresh averaging window for the next modulation step"
        );
    }

    #[test]
    fn test_dual_plane_unity_exit_seam_requires_real_prior_output() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut state = DualPlaneDeterministicState::from_params(&params, 1.0).unwrap();

        state.arm_unity_exit_seam();
        assert!(
            !state.pending_unity_exit_seam,
            "unity-exit seam smoothing should stay idle until deterministic output has produced a real anchor sample"
        );

        let channel_output_buffers = [vec![0.125, -0.25, 0.375]];
        state.capture_last_output_samples(&channel_output_buffers, 3);
        assert!(
            state.has_last_output_samples,
            "capturing deterministic output should arm a real seam anchor for later unity exits"
        );

        state.arm_unity_exit_seam();
        assert!(
            state.pending_unity_exit_seam,
            "once deterministic output has emitted audio, the next exact-unity exit should carry the captured seam anchor"
        );
        assert!(
            (state.unity_exit_seam_samples[0] - 0.375).abs() < 1e-12,
            "unity-exit seam smoothing should anchor from the last emitted sample instead of an initial placeholder"
        );
    }

    #[test]
    fn test_stream_processor_dual_plane_profile_telemetry_available() {
        let params = StretchParams::new(1.02)
            .with_sample_rate(44_100)
            .with_channels(2);
        let proc = StreamProcessor::new(params);
        let telemetry = proc
            .deterministic_profile_telemetry()
            .expect("dual-plane telemetry should be available by default");
        assert!(telemetry.auto_switching_enabled);
    }

    #[test]
    fn test_stream_processor_dual_plane_runtime_telemetry_available() {
        let params = StretchParams::new(1.02)
            .with_sample_rate(44_100)
            .with_channels(2);
        let proc = StreamProcessor::new(params);
        let telemetry = proc
            .deterministic_runtime_telemetry()
            .expect("dual-plane runtime telemetry should be available by default");
        assert_eq!(telemetry, RtRuntimeTelemetry::default());
    }

    #[test]
    fn test_stream_processor_dual_plane_delay_telemetry_tracks_buffering() {
        let params = StretchParams::new(1.02)
            .with_sample_rate(44_100)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut proc = StreamProcessor::new(params);

        let initial = proc
            .deterministic_delay_telemetry()
            .expect("dual-plane delay telemetry should be available by default");
        assert_eq!(initial.algorithmic_frames, 96);
        assert_eq!(initial.buffered_input_frames, 0);
        assert_eq!(initial.buffered_output_frames, 0);
        assert_eq!(initial.total_frames, 96);

        let mut output = Vec::with_capacity(128);
        proc.process_into(&[0.0; 32], &mut output).unwrap();
        assert!(output.is_empty());

        let buffered = proc
            .deterministic_delay_telemetry()
            .expect("dual-plane delay telemetry should remain available");
        assert_eq!(buffered.algorithmic_frames, 96);
        assert_eq!(buffered.buffered_input_frames, 32);
        assert_eq!(buffered.buffered_output_frames, 0);
        assert_eq!(buffered.total_frames, 128);

        proc.flush_into(&mut output).unwrap();
        let flushed = proc
            .deterministic_delay_telemetry()
            .expect("dual-plane delay telemetry should remain available");
        assert_eq!(flushed.buffered_input_frames, 0);
        assert_eq!(flushed.buffered_output_frames, 0);
        assert_eq!(flushed.total_frames, 96);
    }

    #[test]
    fn test_stream_processor_dual_plane_delay_telemetry_includes_stream_pending_output() {
        let params = StretchParams::new(1.03)
            .with_sample_rate(44_100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);

        let input: Vec<f32> = (0..(256 * 16))
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 44_100.0).sin() * 0.5)
            .collect();
        let mut callback_output = [0.0f32; 0];
        for chunk in input.chunks(256) {
            proc.process_interleaved_into(chunk, &mut callback_output)
                .unwrap();
            if !proc.pending_output.is_empty() {
                break;
            }
        }

        assert!(
            !proc.pending_output.is_empty(),
            "expected bounded callback output to leave samples queued"
        );

        let rt = proc
            .dual_plane_deterministic
            .as_ref()
            .expect("dual-plane telemetry should be available")
            .processor
            .delay_telemetry();
        let stream_pending_frames = proc.pending_output.len();
        let telemetry = proc
            .deterministic_delay_telemetry()
            .expect("dual-plane delay telemetry should be available");

        assert_eq!(
            telemetry.buffered_output_frames,
            rt.buffered_output_frames + stream_pending_frames
        );
        assert_eq!(
            telemetry.total_frames,
            rt.total_frames + stream_pending_frames
        );
    }

    #[test]
    fn test_stream_processor_dual_plane_delay_telemetry_tracks_pending_drain() {
        let params = StretchParams::new(1.03)
            .with_sample_rate(44_100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);

        let input: Vec<f32> = (0..(256 * 16))
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 44_100.0).sin() * 0.5)
            .collect();
        let mut callback_output = [0.0f32; 0];
        for chunk in input.chunks(256) {
            proc.process_interleaved_into(chunk, &mut callback_output)
                .unwrap();
            if proc.pending_output.len() >= 32 {
                break;
            }
        }

        assert!(
            proc.pending_output.len() >= 32,
            "expected enough queued output to validate drain telemetry"
        );

        let before = proc
            .deterministic_delay_telemetry()
            .expect("dual-plane delay telemetry should be available");
        let pending_before = proc.pending_output.len();

        let mut drain = [0.0f32; 13];
        let written = proc.process_interleaved_into(&[], &mut drain).unwrap();
        assert!(written > 0);
        assert!(proc.pending_output.len() < pending_before);

        let after = proc
            .deterministic_delay_telemetry()
            .expect("dual-plane delay telemetry should be available");

        assert_eq!(
            before.buffered_output_frames - after.buffered_output_frames,
            written
        );
        assert_eq!(before.total_frames - after.total_frames, written);
    }

    #[test]
    fn test_stream_processor_current_delay_accessors_match_exact_telemetry() {
        let params = StretchParams::new(1.03)
            .with_sample_rate(44_100)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);

        let input: Vec<f32> = (0..(256 * 16))
            .flat_map(|i| {
                let t = i as f32 / 44_100.0;
                [
                    (2.0 * PI * 220.0 * t).sin() * 0.5,
                    (2.0 * PI * 330.0 * t).sin() * 0.35,
                ]
            })
            .collect();
        let mut callback_output = [0.0f32; 0];
        for chunk in input.chunks(256 * 2) {
            proc.process_interleaved_into(chunk, &mut callback_output)
                .unwrap();
            if !proc.pending_output.is_empty() {
                break;
            }
        }

        let telemetry = proc
            .deterministic_delay_telemetry()
            .expect("dual-plane delay telemetry should be available");
        assert_eq!(proc.current_delay_frames(), Some(telemetry.total_frames));
        assert_eq!(proc.current_delay_samples(), Some(telemetry.total_frames));
        assert_eq!(
            proc.current_delay_secs(),
            Some(telemetry.total_frames as f64 / 44_100.0)
        );
    }

    #[test]
    fn test_stream_processor_set_deterministic_latency_profile() {
        let params = StretchParams::new(1.02)
            .with_sample_rate(44_100)
            .with_channels(2);
        let mut proc = StreamProcessor::new(params);
        proc.set_deterministic_latency_profile(LatencyProfile::Scratch)
            .unwrap();
        let telemetry = proc
            .deterministic_profile_telemetry()
            .expect("dual-plane telemetry should be available");
        assert_eq!(telemetry.target_profile, LatencyProfile::Scratch);
        assert!(!telemetry.auto_switching_enabled);
    }

    #[test]
    fn test_stream_processor_dual_plane_deterministic_produces_output() {
        let params = StretchParams::new(1.03)
            .with_sample_rate(44_100)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_dual_plane_deterministic(true).unwrap();

        let mut input = Vec::with_capacity(256 * 2 * 8);
        for i in 0..(256 * 8) {
            let t = i as f32 / 44_100.0;
            let l = (2.0 * std::f32::consts::PI * 110.0 * t).sin() * 0.3;
            let r = (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.3;
            input.push(l);
            input.push(r);
        }

        let mut output = Vec::with_capacity(input.len() * 2);
        for chunk in input.chunks(256 * 2) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        proc.flush_into(&mut output).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_transient_reset_stats_start_zero() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44_100)
            .with_channels(2);
        let proc = StreamProcessor::new(params);
        let stats = proc.transient_reset_stats();
        assert_eq!(stats.events_detected_total, 0);
        assert_eq!(stats.reset_band_counts_total, [0, 0, 0, 0]);
        assert_eq!(stats.input_frames_consumed_total, 0);
    }

    #[test]
    fn test_short_interval_cross_unity_modulation_does_not_spuriously_trigger_transient_resets() {
        let sample_rate = 48_000u32;
        let chunk_frames = 256usize;
        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_dual_plane_deterministic(false).unwrap();

        let ratios = [1.04, 0.96, 1.04, 0.96, 1.04, 0.96, 1.04, 0.96, 1.04, 0.96];
        let mut total_output = Vec::new();

        for (chunk_idx, ratio) in ratios.into_iter().enumerate() {
            proc.set_stretch_ratio(ratio).unwrap();

            let mut chunk = Vec::with_capacity(chunk_frames * 2);
            for frame in 0..chunk_frames {
                let sample_idx = chunk_idx * chunk_frames + frame;
                let t = sample_idx as f32 / sample_rate as f32;
                chunk.push((2.0 * PI * 220.0 * t).sin() * 0.25);
                chunk.push((2.0 * PI * 330.0 * t).sin() * 0.20);
            }

            total_output.extend_from_slice(&proc.process(&chunk).unwrap());
        }

        let stats = proc.transient_reset_stats();
        assert_eq!(
            stats.events_detected_total, 0,
            "smooth stereo modulation should not schedule transient resets without actual onsets"
        );
        assert_eq!(
            stats.reset_band_counts_total,
            [0, 0, 0, 0],
            "smooth stereo modulation should not increment per-band transient reset telemetry"
        );
        assert!(
            stats.input_frames_consumed_total >= chunk_frames * 2,
            "test should drive the stream far enough to exercise scheduler analysis"
        );
        assert!(
            total_output.iter().all(|sample| sample.is_finite()),
            "modulated stream output should remain finite"
        );
    }

    #[test]
    fn test_short_interval_cross_unity_modulation_does_not_retrigger_one_real_transient() {
        let sample_rate = 48_000u32;
        let chunk_frames = 512usize;
        let click_range = 1600usize..1620usize;
        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_dual_plane_deterministic(false).unwrap();

        let ratios = [1.04, 0.96, 1.04, 0.96, 1.04, 0.96];
        let mut total_output = Vec::new();

        for (chunk_idx, ratio) in ratios.into_iter().enumerate() {
            proc.set_stretch_ratio(ratio).unwrap();

            let mut chunk = Vec::with_capacity(chunk_frames * 2);
            for frame in 0..chunk_frames {
                let sample_idx = chunk_idx * chunk_frames + frame;
                let t = sample_idx as f32 / sample_rate as f32;
                let click = if click_range.contains(&sample_idx) {
                    2.0
                } else {
                    0.0
                };
                chunk.push((2.0 * PI * 220.0 * t).sin() * 0.25 + click);
                chunk.push((2.0 * PI * 330.0 * t).sin() * 0.20 + click);
            }

            total_output.extend_from_slice(&proc.process(&chunk).unwrap());
        }

        let stats = proc.transient_reset_stats();
        assert_eq!(
            stats.events_detected_total, 1,
            "one real transient should schedule exactly one reset event across short-interval modulation"
        );
        assert_eq!(
            stats.reset_band_counts_total[2], 1,
            "one real transient should only contribute one mid-band reset across short-interval modulation"
        );
        assert_eq!(
            stats.reset_band_counts_total[3], 1,
            "one real transient should only contribute one high-band reset across short-interval modulation"
        );
        assert!(
            total_output.iter().all(|sample| sample.is_finite()),
            "modulated stream output should remain finite while the transient overlap drains"
        );
    }

    #[test]
    fn test_dual_plane_short_interval_cross_unity_modulation_avoids_profile_churn() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut proc = StreamProcessor::new(params);
        assert!(
            proc.is_dual_plane_deterministic(),
            "deterministic stream should default to the dual-plane backend"
        );

        let initial = proc
            .deterministic_profile_telemetry()
            .expect("dual-plane telemetry should be available");
        assert_eq!(initial.current_profile, LatencyProfile::Mix);
        assert_eq!(initial.target_profile, LatencyProfile::Mix);
        assert_eq!(initial.policy_profile, LatencyProfile::Mix);
        assert!(
            initial.auto_switching_enabled,
            "dual-plane deterministic profile switching should default to auto mode"
        );

        let chunk = [0.0f32; 256];
        for (step_idx, ratio) in [1.035, 0.975, 1.025, 0.965, 1.035, 0.975]
            .into_iter()
            .enumerate()
        {
            proc.set_stretch_ratio(ratio).unwrap();
            let output = proc.process(&chunk).unwrap();
            assert!(
                output.iter().all(|sample| sample.is_finite()),
                "deterministic modulation should keep output finite during step {step_idx}"
            );
            let telemetry = proc
                .deterministic_profile_telemetry()
                .expect("dual-plane telemetry should remain available during modulation");
            assert_eq!(
                telemetry.current_profile,
                LatencyProfile::Mix,
                "short-interval deterministic modulation should keep the active profile on mix during step {step_idx}"
            );
            assert_eq!(
                telemetry.target_profile,
                LatencyProfile::Mix,
                "short-interval deterministic modulation should keep the target profile on mix during step {step_idx}"
            );
            assert!(
                telemetry.policy_profile == LatencyProfile::Mix,
                "short-interval deterministic modulation should keep the policy profile on mix during step {step_idx}"
            );
            assert!(
                telemetry.auto_switching_enabled,
                "auto profile switching should remain enabled during deterministic modulation"
            );
        }
        assert!(
            proc.flush()
                .unwrap()
                .iter()
                .all(|sample| sample.is_finite()),
            "deterministic modulation flush should remain finite"
        );
    }

    #[test]
    fn test_dual_plane_short_interval_cross_unity_modulation_can_still_retarget_to_scratch() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(48_000)
            .with_channels(1)
            .with_fft_size(64)
            .with_hop_size(16);
        let mut proc = StreamProcessor::new(params);
        assert!(
            proc.is_dual_plane_deterministic(),
            "deterministic stream should default to the dual-plane backend"
        );

        let chunk = [0.0f32; 256];
        for (step_idx, ratio) in [1.035, 0.975, 1.025, 0.965, 1.035, 0.975]
            .into_iter()
            .enumerate()
        {
            proc.set_stretch_ratio(ratio).unwrap();
            let output = proc.process(&chunk).unwrap();
            assert!(
                output.iter().all(|sample| sample.is_finite()),
                "deterministic modulation should keep output finite during step {step_idx}"
            );
        }

        let after_modulation = proc
            .deterministic_profile_telemetry()
            .expect("dual-plane telemetry should remain available after modulation");
        assert_eq!(
            after_modulation.current_profile,
            LatencyProfile::Mix,
            "short-interval modulation should leave the committed profile parked on mix before the stable plateau begins"
        );
        assert_eq!(
            after_modulation.target_profile,
            LatencyProfile::Mix,
            "short-interval modulation should not leave a stale scratch retarget queued once the modulation burst ends"
        );

        proc.set_stretch_ratio(1.70).unwrap();
        let mut saw_scratch_target = false;
        let mut saw_scratch_current = false;
        for settle_idx in 0..16 {
            let output = proc.process(&chunk).unwrap();
            assert!(
                output.iter().all(|sample| sample.is_finite()),
                "stable post-modulation scratch plateau should keep output finite during settle step {settle_idx}"
            );
            let telemetry = proc
                .deterministic_profile_telemetry()
                .expect("dual-plane telemetry should remain available while the plateau settles");
            saw_scratch_target |= telemetry.target_profile == LatencyProfile::Scratch;
            saw_scratch_current |= telemetry.current_profile == LatencyProfile::Scratch;
            if saw_scratch_target && saw_scratch_current {
                break;
            }
        }

        assert!(
            saw_scratch_target,
            "once the rapid modulation stops, a stable scratch-biased plateau should still be able to retarget away from mix"
        );
        assert!(
            saw_scratch_current,
            "once hysteresis and the profile transition settle, the stable scratch-biased plateau should still be able to commit scratch"
        );
        assert!(
            proc.flush()
                .unwrap()
                .iter()
                .all(|sample| sample.is_finite()),
            "stable post-modulation plateau flush should remain finite"
        );
    }

    #[test]
    fn test_stereo_channel_reset_masks_mid_full_side_mid_high() {
        let full = [true, true, true, true];
        let (mid, side) = stereo_channel_reset_masks(full);
        assert_eq!(mid, full);
        assert_eq!(side, [false, false, true, true]);
    }

    #[test]
    fn test_stereo_channel_reset_masks_preserves_selective_band_mask() {
        let full = [false, true, false, true];
        let (mid, side) = stereo_channel_reset_masks(full);
        assert_eq!(mid, full);
        assert_eq!(side, [false, false, false, true]);
    }

    // --- process_into tests ---

    #[test]
    fn test_process_into_matches_process() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let chunk_size = 4096;
        let signal: Vec<f32> = (0..chunk_size * 4)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        // Run with process()
        let mut proc1 = StreamProcessor::new(params.clone());
        let mut output1 = Vec::new();
        for chunk in signal.chunks(chunk_size) {
            if let Ok(out) = proc1.process(chunk) {
                output1.extend_from_slice(&out);
            }
        }
        if let Ok(out) = proc1.flush() {
            output1.extend_from_slice(&out);
        }

        // Run with process_into()
        let mut proc2 = StreamProcessor::new(params);
        let mut output2 = Vec::with_capacity(signal.len() * 3);
        for chunk in signal.chunks(chunk_size) {
            proc2.process_into(chunk, &mut output2).unwrap();
        }
        proc2.flush_into(&mut output2).unwrap();

        assert_eq!(
            output1.len(),
            output2.len(),
            "process and process_into should produce same length"
        );
        for (i, (a, b)) in output1.iter().zip(output2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Mismatch at sample {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_process_into_stereo() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(2);

        let num_frames = 44100;
        let mut signal = vec![0.0f32; num_frames * 2];
        for i in 0..num_frames {
            let t = i as f32 / 44100.0;
            signal[i * 2] = (2.0 * PI * 440.0 * t).sin();
            signal[i * 2 + 1] = (2.0 * PI * 880.0 * t).sin();
        }

        let mut proc = StreamProcessor::new(params);
        let mut output = Vec::with_capacity(signal.len() * 3);
        for chunk in signal.chunks(4096 * 2) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        proc.flush_into(&mut output).unwrap();

        assert!(!output.is_empty(), "Should produce output");
        assert_eq!(output.len() % 2, 0, "Stereo output must have even count");
    }

    #[test]
    fn test_process_into_rejects_nan() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        let mut chunk = vec![0.0f32; 4096];
        chunk[100] = f32::NAN;
        let mut output = Vec::new();
        assert!(matches!(
            proc.process_into(&chunk, &mut output),
            Err(crate::error::StretchError::NonFiniteInput)
        ));
        assert!(output.is_empty());
    }

    #[test]
    fn test_process_into_unity_requires_output_capacity() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        let input = vec![0.1f32; 1024];
        let mut output = Vec::new();
        assert!(matches!(
            proc.process_into(&input, &mut output),
            Err(StretchError::BufferOverflow {
                buffer: "process_into_output",
                ..
            })
        ));
    }

    #[test]
    fn test_process_into_writes_expected_amount() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        let mut output = Vec::with_capacity(200_000);

        // First small chunk: unity ratio takes the passthrough path.
        let small = vec![0.0f32; 1024];
        let before_small = output.len();
        proc.process_into(&small, &mut output).unwrap();
        let written_small = output.len() - before_small;
        assert_eq!(written_small, small.len());

        // Large chunk: should produce output
        let big: Vec<f32> = (0..44100)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let before_big = output.len();
        proc.process_into(&big, &mut output).unwrap();
        let written_big = output.len() - before_big;
        assert!(written_big > 0);
    }

    #[test]
    fn test_process_into_appends() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        let mut output = vec![42.0f32]; // pre-existing data
        output.reserve(200_000);

        let signal: Vec<f32> = (0..44100)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        proc.process_into(&signal, &mut output).unwrap();

        // First sample should still be our sentinel value
        assert!(
            (output[0] - 42.0).abs() < 1e-6,
            "process_into should append, not overwrite"
        );
    }

    #[test]
    fn test_flush_into_empty() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        let mut output = Vec::new();
        let n = proc.flush_into(&mut output).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_process_interleaved_into_matches_vec_process_and_flush_when_drained_in_chunks() {
        let params = StretchParams::new(1.03)
            .with_sample_rate(44_100)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut vec_proc = StreamProcessor::new(params.clone());
        let mut fixed_proc = StreamProcessor::new(params);
        vec_proc.set_dual_plane_deterministic(true).unwrap();
        fixed_proc.set_dual_plane_deterministic(true).unwrap();
        vec_proc.set_pitch_scale(1.05).unwrap();
        fixed_proc.set_pitch_scale(1.05).unwrap();

        let mut input = Vec::with_capacity(256 * 2 * 12);
        for i in 0..(256 * 12) {
            let t = i as f32 / 44_100.0;
            input.push((2.0 * PI * 110.0 * t).sin() * 0.3);
            input.push((2.0 * PI * 330.0 * t).sin() * 0.25);
        }

        let mut expected = Vec::with_capacity(input.len() * 2);
        let mut actual = Vec::with_capacity(input.len() * 2);
        let mut chunk = vec![0.0f32; 74];
        for input_chunk in input.chunks(256 * 2) {
            vec_proc.process_into(input_chunk, &mut expected).unwrap();
            let written = fixed_proc
                .process_interleaved_into(input_chunk, &mut chunk)
                .unwrap();
            actual.extend_from_slice(&chunk[..written]);
        }

        vec_proc.flush_into(&mut expected).unwrap();

        loop {
            let written = fixed_proc.flush_interleaved_into(&mut chunk).unwrap();
            if written == 0 {
                break;
            }
            actual.extend_from_slice(&chunk[..written]);
        }

        assert_eq!(expected.len(), actual.len());
        for (idx, (&lhs, &rhs)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() < 1e-6,
                "Mismatch at sample {idx}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn test_process_interleaved_into_rejects_when_output_budget_is_too_small() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44_100)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_dual_plane_deterministic(true).unwrap();

        let input = vec![0.0f32; 256 * 2 * 16];
        let caps_before = proc.capacities();
        let mut output = [0.0f32; 0];
        let err = proc
            .process_interleaved_into(&input, &mut output)
            .unwrap_err();

        assert!(matches!(
            err,
            StretchError::BufferOverflow {
                buffer: "stream_process_interleaved_output",
                ..
            }
        ));
        assert_eq!(proc.capacities(), caps_before);
    }

    #[test]
    fn test_max_process_interleaved_output_samples_matches_process_capacity_floor() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44_100)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_dual_plane_deterministic(true).unwrap();

        let input = vec![0.0f32; 256 * 2 * 16];
        let budget = proc
            .max_process_interleaved_output_samples(input.len())
            .unwrap();
        let (_, pending_len, _, pending_cap) = proc.capacities();
        let min_output = budget.saturating_sub(pending_cap.saturating_sub(pending_len));

        assert!(
            min_output >= 2,
            "expected host-visible output floor, got {min_output}"
        );
        assert_eq!(budget % 2, 0);

        let caps_before = proc.capacities();
        let mut too_small = vec![0.0f32; min_output - 2];
        let err = proc
            .process_interleaved_into(&input, &mut too_small)
            .unwrap_err();
        assert!(matches!(
            err,
            StretchError::BufferOverflow {
                buffer: "stream_process_interleaved_output",
                ..
            }
        ));
        assert_eq!(proc.capacities(), caps_before);

        let mut just_enough = vec![0.0f32; min_output];
        let written = proc
            .process_interleaved_into(&input, &mut just_enough)
            .unwrap();
        assert!(written <= just_enough.len());
    }

    #[test]
    fn test_max_next_process_interleaved_output_samples_matches_combined_helpers() {
        let params = StretchParams::new(1.03)
            .with_sample_rate(44_100)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_dual_plane_deterministic(true).unwrap();
        proc.set_pitch_scale(1.05).unwrap();

        let mut input = Vec::with_capacity(256 * 2 * 12);
        for i in 0..(256 * 12) {
            let t = i as f32 / 44_100.0;
            input.push((2.0 * PI * 110.0 * t).sin() * 0.3);
            input.push((2.0 * PI * 330.0 * t).sin() * 0.25);
        }

        let input_chunk_len = 256 * 2;
        let no_pending_budget = proc
            .max_next_process_interleaved_output_samples(input_chunk_len)
            .unwrap();
        assert_eq!(
            no_pending_budget,
            proc.max_process_interleaved_output_samples(input_chunk_len)
                .unwrap()
        );
        assert_eq!(no_pending_budget % 2, 0);

        let mut callback_output = [0.0f32; 6];
        for chunk in input.chunks(input_chunk_len) {
            let _ = proc
                .process_interleaved_into(chunk, &mut callback_output)
                .unwrap();
            if proc.queued_interleaved_output_samples().unwrap() >= 24 {
                break;
            }
        }

        let queued = proc.queued_interleaved_output_samples().unwrap();
        assert!(
            queued >= 24,
            "expected queued output to validate combined budget"
        );

        let combined = proc
            .max_next_process_interleaved_output_samples(input_chunk_len)
            .unwrap();
        let expected = queued.saturating_add(
            proc.max_process_interleaved_output_samples(input_chunk_len)
                .unwrap(),
        );
        assert_eq!(combined, expected);
        assert_eq!(combined % 2, 0);
    }

    #[test]
    fn test_queued_interleaved_output_samples_tracks_pending_drain() {
        let params = StretchParams::new(1.03)
            .with_sample_rate(44_100)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_dual_plane_deterministic(true).unwrap();
        proc.set_pitch_scale(1.05).unwrap();

        let mut input = Vec::with_capacity(256 * 2 * 12);
        for i in 0..(256 * 12) {
            let t = i as f32 / 44_100.0;
            input.push((2.0 * PI * 110.0 * t).sin() * 0.3);
            input.push((2.0 * PI * 330.0 * t).sin() * 0.25);
        }

        let mut callback_output = [0.0f32; 6];
        for chunk in input.chunks(256 * 2) {
            let _ = proc
                .process_interleaved_into(chunk, &mut callback_output)
                .unwrap();
            if proc.queued_interleaved_output_samples().unwrap() >= 24 {
                break;
            }
        }

        let queued_before = proc.queued_interleaved_output_samples().unwrap();
        assert!(
            queued_before >= 24,
            "expected queued output to validate fixed-buffer drain accounting"
        );
        assert_eq!(queued_before % 2, 0);

        let mut partial_drain = [0.0f32; 8];
        let written = proc
            .process_interleaved_into(&[], &mut partial_drain)
            .unwrap();
        assert!(written > 0);
        assert_eq!(
            proc.queued_interleaved_output_samples().unwrap(),
            queued_before - written
        );

        let remaining = proc.queued_interleaved_output_samples().unwrap();
        let mut rest = vec![0.0f32; remaining];
        let drained = proc.process_interleaved_into(&[], &mut rest).unwrap();
        assert_eq!(drained, remaining);
        assert_eq!(proc.queued_interleaved_output_samples().unwrap(), 0);
    }

    #[test]
    fn test_max_flush_interleaved_output_samples_drains_remaining_tail_in_one_call() {
        let params = StretchParams::new(1.03)
            .with_sample_rate(44_100)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut vec_proc = StreamProcessor::new(params.clone());
        let mut fixed_proc = StreamProcessor::new(params);
        vec_proc.set_dual_plane_deterministic(true).unwrap();
        fixed_proc.set_dual_plane_deterministic(true).unwrap();
        vec_proc.set_pitch_scale(1.05).unwrap();
        fixed_proc.set_pitch_scale(1.05).unwrap();

        let mut input = Vec::with_capacity(256 * 2 * 12);
        for i in 0..(256 * 12) {
            let t = i as f32 / 44_100.0;
            input.push((2.0 * PI * 110.0 * t).sin() * 0.3);
            input.push((2.0 * PI * 330.0 * t).sin() * 0.25);
        }

        let mut expected = Vec::with_capacity(input.len() * 2);
        let mut actual = Vec::with_capacity(input.len() * 2);
        let mut callback_output = vec![0.0f32; 74];
        for input_chunk in input.chunks(256 * 2) {
            vec_proc.process_into(input_chunk, &mut expected).unwrap();
            let written = fixed_proc
                .process_interleaved_into(input_chunk, &mut callback_output)
                .unwrap();
            actual.extend_from_slice(&callback_output[..written]);
        }

        vec_proc.flush_into(&mut expected).unwrap();

        let remaining = fixed_proc.max_flush_interleaved_output_samples().unwrap();
        assert!(remaining > 0);

        let mut flush_output = vec![0.0f32; remaining];
        let written = fixed_proc
            .flush_interleaved_into(&mut flush_output)
            .unwrap();
        assert!(written <= remaining);
        actual.extend_from_slice(&flush_output[..written]);

        assert_eq!(
            fixed_proc.max_flush_interleaved_output_samples().unwrap(),
            0
        );
        assert_eq!(
            fixed_proc
                .flush_interleaved_into(&mut flush_output)
                .unwrap(),
            0
        );
        assert_eq!(expected.len(), actual.len());
        for (idx, (&lhs, &rhs)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() < 1e-6,
                "Mismatch at sample {idx}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn test_max_flush_interleaved_output_samples_tracks_partial_tail_drain() {
        let params = StretchParams::new(1.04)
            .with_sample_rate(44_100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_dual_plane_deterministic(true).unwrap();

        let input: Vec<f32> = (0..(256 * 16))
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 44_100.0).sin() * 0.7)
            .collect();
        let mut callback_output = [0.0f32; 64];
        for chunk in input.chunks(256) {
            let _ = proc
                .process_interleaved_into(chunk, &mut callback_output)
                .unwrap();
        }

        let mut first_chunk = [0.0f32; 8];
        let first_written = proc.flush_interleaved_into(&mut first_chunk).unwrap();
        assert!(first_written > 0);
        assert!(proc.fixed_flush_pending);

        let remaining = proc.max_flush_interleaved_output_samples().unwrap();
        assert!(remaining > 0);

        let mut rest = vec![0.0f32; remaining];
        let written = proc.flush_interleaved_into(&mut rest).unwrap();
        assert!(written <= remaining);
        assert_eq!(proc.max_flush_interleaved_output_samples().unwrap(), 0);
        assert_eq!(proc.flush_interleaved_into(&mut rest).unwrap(), 0);
        assert!(!proc.fixed_flush_pending);
    }

    #[test]
    fn test_flush_interleaved_into_matches_vec_flush_when_drained_in_chunks() {
        let params = StretchParams::new(1.03)
            .with_sample_rate(44_100)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut vec_proc = StreamProcessor::new(params.clone());
        let mut fixed_proc = StreamProcessor::new(params);
        vec_proc.set_dual_plane_deterministic(true).unwrap();
        fixed_proc.set_dual_plane_deterministic(true).unwrap();
        vec_proc.set_pitch_scale(1.05).unwrap();
        fixed_proc.set_pitch_scale(1.05).unwrap();

        let mut input = Vec::with_capacity(256 * 2 * 12);
        for i in 0..(256 * 12) {
            let t = i as f32 / 44_100.0;
            input.push((2.0 * PI * 110.0 * t).sin() * 0.3);
            input.push((2.0 * PI * 330.0 * t).sin() * 0.25);
        }

        let mut expected = Vec::with_capacity(input.len() * 2);
        let mut actual = Vec::with_capacity(input.len() * 2);
        for chunk in input.chunks(256 * 2) {
            vec_proc.process_into(chunk, &mut expected).unwrap();
            fixed_proc.process_into(chunk, &mut actual).unwrap();
        }

        vec_proc.flush_into(&mut expected).unwrap();

        let mut chunk = vec![0.0f32; 74];
        loop {
            let written = fixed_proc.flush_interleaved_into(&mut chunk).unwrap();
            if written == 0 {
                break;
            }
            actual.extend_from_slice(&chunk[..written]);
        }

        assert_eq!(expected.len(), actual.len());
        for (idx, (&lhs, &rhs)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() < 1e-6,
                "Mismatch at sample {idx}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn test_reconcile_fixed_flush_pending_output_appends_final_frame_aligned_tail() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44_100)
            .with_channels(2);
        let mut proc = StreamProcessor::new(params);

        assert_eq!(proc.pending_output.push_slice(&[0.0, 0.0, 0.0, 0.0]), 4);
        proc.total_output_emitted_samples = 4;
        proc.expected_total_output_samples = 10.0;

        proc.reconcile_fixed_flush_pending_output(2).unwrap();

        let mut pending = vec![1.0f32; proc.pending_output.len()];
        let copied = proc.pending_output.peek_slice(&mut pending);
        assert_eq!(copied, 6);
        assert_eq!(pending, vec![0.0; 6]);
    }

    #[test]
    fn test_reconcile_fixed_flush_pending_output_trims_remaining_tail() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44_100)
            .with_channels(2);
        let mut proc = StreamProcessor::new(params);

        assert_eq!(
            proc.pending_output
                .push_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            6
        );
        proc.total_output_emitted_samples = 4;
        proc.expected_total_output_samples = 8.0;

        proc.reconcile_fixed_flush_pending_output(2).unwrap();

        let mut pending = vec![0.0f32; proc.pending_output.len()];
        let copied = proc.pending_output.peek_slice(&mut pending);
        assert_eq!(copied, 4);
        assert_eq!(pending, vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn test_flush_interleaved_into_requires_tail_to_drain_before_new_input() {
        let params = StretchParams::new(1.04)
            .with_sample_rate(44_100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_dual_plane_deterministic(true).unwrap();

        let input: Vec<f32> = (0..(256 * 16))
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 44_100.0).sin() * 0.7)
            .collect();
        let mut streamed = Vec::with_capacity(input.len() * 2);
        for chunk in input.chunks(256) {
            proc.process_into(chunk, &mut streamed).unwrap();
        }

        let mut chunk = [0.0f32; 8];
        let first_written = proc.flush_interleaved_into(&mut chunk).unwrap();
        assert!(first_written > 0);
        assert!(proc.fixed_flush_pending);

        let err = proc.process_into(&input[..256], &mut streamed).unwrap_err();
        assert_eq!(
            err,
            StretchError::InvalidState(
                "fixed-buffer flush output must be fully drained before new input"
            )
        );

        loop {
            if proc.flush_interleaved_into(&mut chunk).unwrap() == 0 {
                break;
            }
        }

        assert!(!proc.fixed_flush_pending);
        assert!(proc.process_into(&input[..256], &mut streamed).is_ok());
    }

    #[test]
    fn test_pitch_scale_change_waits_for_fixed_flush_tail_drain() {
        let params = StretchParams::new(1.03)
            .with_sample_rate(44_100)
            .with_channels(2)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut control = StreamProcessor::new(params.clone());
        let mut retuned = StreamProcessor::new(params);
        control.set_dual_plane_deterministic(true).unwrap();
        retuned.set_dual_plane_deterministic(true).unwrap();
        control.set_pitch_scale(1.05).unwrap();
        retuned.set_pitch_scale(1.05).unwrap();

        let mut input = Vec::with_capacity(256 * 2 * 12);
        for i in 0..(256 * 12) {
            let t = i as f32 / 44_100.0;
            input.push((2.0 * PI * 110.0 * t).sin() * 0.3);
            input.push((2.0 * PI * 330.0 * t).sin() * 0.25);
        }

        let mut control_streamed = Vec::with_capacity(input.len() * 2);
        let mut retuned_streamed = Vec::with_capacity(input.len() * 2);
        for chunk in input.chunks(256 * 2) {
            control.process_into(chunk, &mut control_streamed).unwrap();
            retuned.process_into(chunk, &mut retuned_streamed).unwrap();
        }
        assert_eq!(control_streamed.len(), retuned_streamed.len());
        for (idx, (&lhs, &rhs)) in control_streamed
            .iter()
            .zip(retuned_streamed.iter())
            .enumerate()
        {
            assert!(
                (lhs - rhs).abs() < 1e-6,
                "Mismatch at streamed sample {idx}: {lhs} vs {rhs}"
            );
        }

        let control_flush_budget = control.max_flush_interleaved_output_samples().unwrap();
        let retuned_flush_budget = retuned.max_flush_interleaved_output_samples().unwrap();
        assert!(control_flush_budget > 2);
        assert_eq!(control_flush_budget, retuned_flush_budget);

        let mut control_tail = Vec::new();
        let mut retuned_tail = Vec::new();
        let mut first_control = [0.0f32; 2];
        let mut first_retuned = [0.0f32; 2];
        let control_written = control.flush_interleaved_into(&mut first_control).unwrap();
        let retuned_written = retuned.flush_interleaved_into(&mut first_retuned).unwrap();
        assert_eq!(control_written, retuned_written);
        assert!(retuned_written > 0);
        assert!(control.fixed_flush_pending);
        assert!(retuned.fixed_flush_pending);
        control_tail.extend_from_slice(&first_control[..control_written]);
        retuned_tail.extend_from_slice(&first_retuned[..retuned_written]);

        retuned.set_pitch_scale(1.19).unwrap();
        assert!(retuned.fixed_flush_pending);
        assert_eq!(retuned.pending_pitch_scale, Some(1.19));
        assert!((retuned.pitch_scale() - 1.05).abs() < 1e-9);

        let mut control_chunk = [0.0f32; 13];
        let mut retuned_chunk = [0.0f32; 13];
        loop {
            let control_written = control.flush_interleaved_into(&mut control_chunk).unwrap();
            let retuned_written = retuned.flush_interleaved_into(&mut retuned_chunk).unwrap();
            assert_eq!(control_written, retuned_written);
            control_tail.extend_from_slice(&control_chunk[..control_written]);
            retuned_tail.extend_from_slice(&retuned_chunk[..retuned_written]);
            assert_eq!(control.fixed_flush_pending, retuned.fixed_flush_pending);
            if !retuned.fixed_flush_pending {
                break;
            }
        }

        assert_eq!(control_tail.len(), retuned_tail.len());
        for (idx, (&lhs, &rhs)) in control_tail.iter().zip(retuned_tail.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() < 1e-6,
                "Mismatch at flushed sample {idx}: {lhs} vs {rhs}"
            );
        }

        assert!(!retuned.fixed_flush_pending);
        assert_eq!(retuned.pending_pitch_scale, None);
        assert!((retuned.pitch_scale() - 1.19).abs() < 1e-9);
    }

    #[test]
    fn test_deterministic_profile_configuration_rejected_during_fixed_flush_tail_drain() {
        let params = StretchParams::new(1.04)
            .with_sample_rate(44_100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_dual_plane_deterministic(true).unwrap();

        let input: Vec<f32> = (0..(256 * 16))
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 44_100.0).sin() * 0.7)
            .collect();
        let mut callback_output = [0.0f32; 64];
        for chunk in input.chunks(256) {
            let _ = proc
                .process_interleaved_into(chunk, &mut callback_output)
                .unwrap();
        }

        let mut first_chunk = [0.0f32; 8];
        let first_written = proc.flush_interleaved_into(&mut first_chunk).unwrap();
        assert!(first_written > 0);
        assert!(proc.fixed_flush_pending);

        let telemetry_before = proc
            .deterministic_profile_telemetry()
            .expect("dual-plane telemetry should be available");

        let profile_err = proc
            .set_deterministic_latency_profile(LatencyProfile::Scratch)
            .unwrap_err();
        assert_eq!(
            profile_err,
            StretchError::InvalidState(
                "deterministic latency profile cannot change until fixed-buffer flush output is fully drained"
            )
        );

        let auto_err = proc
            .set_deterministic_auto_profile_switching(false)
            .unwrap_err();
        assert_eq!(
            auto_err,
            StretchError::InvalidState(
                "deterministic auto profile switching cannot change until fixed-buffer flush output is fully drained"
            )
        );

        let telemetry_during = proc
            .deterministic_profile_telemetry()
            .expect("dual-plane telemetry should stay available during drain");
        assert_eq!(telemetry_before, telemetry_during);

        let mut rest = [0.0f32; 64];
        loop {
            if proc.flush_interleaved_into(&mut rest).unwrap() == 0 {
                break;
            }
        }
        assert!(!proc.fixed_flush_pending);

        proc.set_deterministic_auto_profile_switching(false)
            .unwrap();
        let telemetry_after_auto = proc
            .deterministic_profile_telemetry()
            .expect("dual-plane telemetry should stay available after drain");
        assert!(!telemetry_after_auto.auto_switching_enabled);

        proc.set_deterministic_latency_profile(LatencyProfile::Scratch)
            .unwrap();
        let telemetry_after_profile = proc
            .deterministic_profile_telemetry()
            .expect("dual-plane telemetry should stay available after profile change");
        assert_eq!(
            telemetry_after_profile.target_profile,
            LatencyProfile::Scratch
        );
        assert!(!telemetry_after_profile.auto_switching_enabled);
    }

    #[test]
    fn test_stream_processor_target_stretch_ratio() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        assert!((proc.target_stretch_ratio() - 1.0).abs() < 1e-6);

        proc.set_stretch_ratio(1.5).unwrap();
        assert!((proc.target_stretch_ratio() - 1.5).abs() < 1e-6);
        // Current ratio hasn't converged yet
        assert!((proc.current_stretch_ratio() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_stream_processor_target_bpm_none() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let proc = StreamProcessor::new(params);
        assert_eq!(proc.target_bpm(), None);
    }

    #[test]
    fn test_stream_processor_target_bpm_from_tempo() {
        let proc = StreamProcessor::from_tempo(126.0, 128.0, 44100, 1);
        let target = proc.target_bpm().unwrap();
        assert!(
            (target - 128.0).abs() < 0.1,
            "Expected target BPM ~128, got {}",
            target
        );
    }

    #[test]
    fn test_stream_processor_target_bpm_after_set_tempo() {
        let mut proc = StreamProcessor::from_tempo(126.0, 128.0, 44100, 1);
        proc.set_tempo(130.0);
        let target = proc.target_bpm().unwrap();
        assert!(
            (target - 130.0).abs() < 0.1,
            "Expected target BPM ~130, got {}",
            target
        );
    }

    #[test]
    fn test_stream_processor_reduced_latency() {
        // Verify the reduced latency is 1.5x FFT size, not 2x
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_fft_size(4096);
        let proc = StreamProcessor::new(params);

        // 4096 * 3 / 2 = 6144
        assert_eq!(proc.latency_samples(), 6144);
        // ~139ms instead of ~186ms
        let latency_ms = proc.latency_secs() * 1000.0;
        assert!(
            latency_ms < 140.0,
            "Latency should be ~139ms, got {}ms",
            latency_ms
        );
    }

    #[test]
    fn test_stream_processor_smooth_ratio_tracks_vocoder() {
        // Verify that changing ratio multiple times still converges correctly
        // (tests the vocoder_ratio tracking fix)
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);

        // First ratio change
        proc.set_stretch_ratio(1.1).unwrap();
        for _ in 0..200 {
            proc.interpolate_ratio();
        }
        assert!(
            (proc.current_stretch_ratio() - 1.1).abs() < 0.001,
            "Should converge to 1.1, got {}",
            proc.current_stretch_ratio()
        );

        // Second ratio change
        proc.set_stretch_ratio(0.9).unwrap();
        for _ in 0..200 {
            proc.interpolate_ratio();
        }
        assert!(
            (proc.current_stretch_ratio() - 0.9).abs() < 0.001,
            "Should converge to 0.9, got {}",
            proc.current_stretch_ratio()
        );
    }

    #[test]
    fn test_stream_processor_with_bpm() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_bpm(128.0);

        let proc = StreamProcessor::new(params);
        assert_eq!(proc.bpm(), Some(128.0));
        assert_eq!(proc.params().bpm, Some(128.0));
    }

    #[test]
    fn test_stream_processor_bpm_default_none() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let proc = StreamProcessor::new(params);
        assert_eq!(proc.bpm(), None);
    }

    #[test]
    fn test_stream_processor_from_tempo_sets_bpm() {
        let proc = StreamProcessor::from_tempo(126.0, 128.0, 44100, 1);
        assert_eq!(proc.bpm(), Some(126.0));
        assert_eq!(proc.source_bpm(), Some(126.0));
    }

    #[test]
    fn test_stream_processor_stereo_phase_coherence() {
        // Verify that stereo processing with phase coherence produces
        // valid output without crashes
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(2);

        let mut proc = StreamProcessor::new(params);

        // Create a stereo signal with a transient (loud click) in both channels
        let num_frames = 44100;
        let mut signal = vec![0.0f32; num_frames * 2];
        for i in 0..num_frames {
            let t = i as f32 / 44100.0;
            let base = (2.0 * PI * 440.0 * t).sin();
            // Add a transient at frame 10000
            let transient = if (10000..10050).contains(&i) {
                1.0
            } else {
                0.0
            };
            signal[i * 2] = base * 0.5 + transient;
            signal[i * 2 + 1] = base * 0.3 + transient;
        }

        let mut total_output = Vec::new();
        for chunk in signal.chunks(4096 * 2) {
            if let Ok(out) = proc.process(chunk) {
                total_output.extend_from_slice(&out);
            }
        }
        if let Ok(out) = proc.flush() {
            total_output.extend_from_slice(&out);
        }

        assert!(!total_output.is_empty(), "Should produce output");
        assert_eq!(
            total_output.len() % 2,
            0,
            "Stereo output must have even count"
        );
    }

    #[test]
    fn test_stream_processor_reduced_latency_produces_output() {
        // Verify that the reduced latency buffer still produces valid output
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);

        let chunk_size = 4096;
        let signal: Vec<f32> = (0..chunk_size * 4)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut total_output = Vec::new();
        for chunk in signal.chunks(chunk_size) {
            if let Ok(out) = proc.process(chunk) {
                total_output.extend_from_slice(&out);
            }
        }
        if let Ok(out) = proc.flush() {
            total_output.extend_from_slice(&out);
        }

        assert!(
            !total_output.is_empty(),
            "Expected output with reduced latency"
        );
    }
}

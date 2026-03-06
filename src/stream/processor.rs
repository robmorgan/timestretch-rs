//! Real-time streaming time-stretch processor.

use crate::analysis::transient::{detect_transients_with_options, TransientDetectionOptions};
use crate::core::ring_buffer::RingBuffer;
use crate::core::types::{QualityMode, StretchParams};
use crate::core::window::WindowType;
use crate::dual_plane::{
    DualPlaneProcessor, LatencyProfile, RtConfig, RtDelayTelemetry, RtProfileTelemetry,
    RtRuntimeTelemetry,
};
use crate::error::StretchError;
use crate::stream::transient_scheduler::{TransientEventScheduler, TransientSchedulerStats};
use crate::stretch::hybrid::HybridStretcher;
use crate::stretch::phase_vocoder::PhaseVocoder;
use crate::stretch::stereo::StereoMode;

/// Threshold below which ratio differences are considered negligible.
const RATIO_SNAP_THRESHOLD: f64 = 0.0001;
/// Ratio smoothing time constant in seconds.
///
/// Smoothing is time-based (not callback-based), so behavior stays stable
/// across 64/128/256/1024 frame callbacks.
const RATIO_SMOOTHING_TIME_SECS: f64 = 0.050;
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
/// Cross-fade length (in samples) at hybrid streaming chunk boundaries.
///
/// Smooths phase discontinuities caused by re-rendering overlapping audio
/// with fresh PV phase state on each call.
const HYBRID_STREAM_CROSSFADE_SAMPLES: usize = 3072;

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
fn dual_plane_supports_pitch_scale(scale: f64) -> bool {
    scale.is_finite() && scale > 0.0
}

/// Persistent hybrid-streaming state.
///
/// Keeps a bounded per-channel rolling tail and emits only the newly rendered
/// region on each call.
struct HybridStreamingState {
    stretchers: Vec<HybridStretcher>,
    rolling_inputs: Vec<RingBuffer<f32>>,
    rolling_scratch: Vec<Vec<f32>>,
    tail_output_lens: Vec<usize>,
    last_ratio: f64,
    max_tail_frames: usize,
    /// Per-channel held-back samples from the previous delta's tail,
    /// used for cross-fading at chunk boundaries to smooth phase
    /// discontinuities from fresh PV state on each re-render.
    crossfade_held: Vec<Vec<f32>>,
    /// Input samples accumulated (per channel) since the last hybrid render.
    ///
    /// Starts at `usize::MAX` so the very first render triggers immediately
    /// once the minimum-latency threshold is met. After each render it resets
    /// to zero, and subsequent renders are deferred until at least `fft_size`
    /// new samples have accumulated. This prevents tiny per-chunk deltas
    /// whose crossfade regions dominate the output and create spectral-flux
    /// artifacts (false onsets).
    input_accumulated: usize,
    /// Reused scratch for pre-trim input lengths per channel.
    pre_trim_lens: Vec<usize>,
    /// Reused scratch for rendered output lengths per channel.
    rendered_lens: Vec<usize>,
}

impl HybridStreamingState {
    fn new(params: &StretchParams, ratio: f64, capacity_frames: usize) -> Self {
        let num_channels = params.channels.count();
        let mut per_channel = params.clone();
        per_channel.stretch_ratio = ratio;
        // Disable elastic timing in streaming: the re-rendering approach
        // snapshots a rolling window and extracts a delta.  Elastic timing
        // redistributes stretch ratios across beat-anchored segments, so
        // shifting the rolling window changes the per-segment ratios for
        // ALL segments, not just the new ones.  This makes the skip
        // estimate (which assumes uniform stretch) unreliable, causing
        // catastrophic spectral degradation for far-from-unity ratios.
        per_channel.elastic_timing = false;
        // Keep a generous tail so that transient detection and HPSS have
        // enough context to produce results consistent with full-batch
        // processing.  Fifty-six FFT windows (~5.2 s at 4096/44100) gives
        // the PV enough warmup frames and the transient detector enough
        // beat-level context for stable segmentation across chunks.
        // The larger window also ensures full signal context is
        // available for short clips (≤5 s), closing the quality gap
        // between streaming and batch rendering.
        let max_tail_frames = params.fft_size * 56;
        // The rolling buffer must hold the retained tail context PLUS a full
        // input batch so that tail samples are not discarded prematurely.
        let rolling_capacity = capacity_frames + max_tail_frames;
        let crossfade_capacity =
            (params.fft_size.saturating_mul(8)).max(HYBRID_STREAM_CROSSFADE_SAMPLES);

        Self {
            stretchers: (0..num_channels)
                .map(|_| HybridStretcher::new(per_channel.clone()))
                .collect(),
            rolling_inputs: (0..num_channels)
                .map(|_| RingBuffer::with_capacity(rolling_capacity))
                .collect(),
            rolling_scratch: (0..num_channels)
                .map(|_| Vec::with_capacity(rolling_capacity))
                .collect(),
            tail_output_lens: vec![0; num_channels],
            last_ratio: ratio,
            max_tail_frames,
            crossfade_held: (0..num_channels)
                .map(|_| Vec::with_capacity(crossfade_capacity))
                .collect(),
            input_accumulated: usize::MAX,
            pre_trim_lens: vec![0; num_channels],
            rendered_lens: vec![0; num_channels],
        }
    }

    fn reset(&mut self, params: &StretchParams, ratio: f64, capacity_frames: usize) {
        *self = Self::new(params, ratio, capacity_frames);
    }

    fn update_ratio(&mut self, ratio: f64) {
        if (ratio - self.last_ratio).abs() <= RATIO_SNAP_THRESHOLD {
            return;
        }
        for stretcher in &mut self.stretchers {
            stretcher.set_stretch_ratio(ratio);
        }
        self.last_ratio = ratio;
    }

    fn retain_tail(&mut self) {
        for input in &mut self.rolling_inputs {
            if input.len() > self.max_tail_frames {
                input.discard(input.len() - self.max_tail_frames);
            }
        }
    }

    /// Rebase rolling buffers when ratio changes so already-emitted history
    /// remains immutable while preserving a small bounded analysis tail.
    fn rebase_after_ratio_change(&mut self) {
        self.retain_tail();
        self.tail_output_lens.fill(0);
        self.input_accumulated = usize::MAX;
    }

    fn update_tail_output_estimates_from_rendered(&mut self) {
        for (idx, input) in self.rolling_inputs.iter().enumerate() {
            let tail_len = input.len();
            if self.pre_trim_lens[idx] > 0 {
                // Scale the actual rendered length by the proportion of input
                // retained as tail — more accurate than `tail_len * ratio`
                // because it reflects real PV hop quantisation.
                self.tail_output_lens[idx] = ((self.rendered_lens[idx] as f64) * tail_len as f64
                    / self.pre_trim_lens[idx] as f64)
                    .round() as usize;
            } else {
                self.tail_output_lens[idx] = 0;
            }
        }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingEngine {
    /// Default deterministic stream engine with bounded per-callback work.
    ///
    /// This engine uses persistent phase-vocoder state and avoids full rolling
    /// re-renders of historical context in callback paths.
    Deterministic,
    /// Legacy rolling-window hybrid re-render engine.
    ///
    /// This mode can provide stronger transient handling for selected content,
    /// but has higher callback-cost variability and is intended as opt-in.
    LegacyHybridRerender,
}

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
    last_ratio: f64,
}

impl DualPlaneDeterministicState {
    fn from_params(params: &StretchParams, ratio: f64) -> Result<Self, StretchError> {
        let block_frames = COMMON_CALLBACK_FRAMES;
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
            last_ratio: ratio,
        })
    }
}

pub struct StreamProcessor {
    params: StretchParams,
    capacity_frames_per_channel: usize,
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
    /// Enables the legacy rolling-window hybrid re-render engine.
    ///
    /// Prefer [`StreamingEngine::Deterministic`] for real-time callback
    /// stability. Keep this flag only as a compatibility bridge for callers
    /// still opting into historical hybrid streaming behavior.
    use_hybrid: bool,
    /// Persistent hybrid streaming state (rolling bounded tail + incremental output).
    hybrid_state: HybridStreamingState,
    /// Indicates that hybrid rolling buffers should rebase on the next process call.
    hybrid_pending_rebase: bool,
    /// When enabled, hybrid mode uses the allocation-free realtime-safe path.
    ///
    /// This trades hybrid transient rendering quality for hard-RT callback
    /// behavior by routing through the preallocated PV streaming path.
    hybrid_realtime_strict: bool,
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
    /// Whether deterministic mode prefers the dual-plane backend.
    dual_plane_preferred: bool,
    /// Optional active dual-plane deterministic backend state.
    dual_plane_deterministic: Option<DualPlaneDeterministicState>,
}

impl std::fmt::Debug for StreamProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamProcessor")
            .field("params", &self.params)
            .field("current_ratio", &self.current_ratio)
            .field("target_ratio", &self.target_ratio)
            .field("vocoder_ratio", &self.vocoder_ratio)
            .field("pitch_scale", &self.pitch_scale)
            .field("hybrid_realtime_strict", &self.hybrid_realtime_strict)
            .field("initialized", &self.initialized)
            .field("fixed_flush_pending", &self.fixed_flush_pending)
            .field(
                "dual_plane_deterministic",
                &self.dual_plane_deterministic.is_some(),
            )
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
        let hybrid_state = HybridStreamingState::new(&params, ratio, capacity_frames_per_channel);
        let transient_scheduler = TransientEventScheduler::new(
            params.fft_size,
            params.hop_size,
            params.sample_rate,
            capacity_frames_per_channel,
        );

        let mut me = Self {
            params,
            capacity_frames_per_channel,
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
            use_hybrid: false,
            hybrid_state,
            hybrid_pending_rebase: false,
            hybrid_realtime_strict: false,
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
            dual_plane_preferred: true,
            dual_plane_deterministic: None,
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
        self.dual_plane_preferred
            && !self.use_hybrid
            && dual_plane_supports_pitch_scale(self.pitch_scale)
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
        if self.use_hybrid {
            return Err(StretchError::InvalidState(
                "fixed-buffer flush is unavailable in legacy hybrid mode",
            ));
        }

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

        if self.dual_plane_deterministic.is_some() && !self.use_hybrid {
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
        if self.dual_plane_deterministic.is_some() && !self.use_hybrid {
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

        if !self.use_hybrid {
            self.flush_vocoder_tails_to_pending(num_channels)?;
        }

        let remaining_frames = self.input_ring.len() / num_channels.max(1);
        self.input_frames_consumed_total = self
            .input_frames_consumed_total
            .saturating_add(remaining_frames);
        self.input_ring.clear();
        if self.use_hybrid {
            // Emit any held-back cross-fade tails before resetting state.
            // These tails are in M/S space and need decoding to L/R.
            let mut held_min_len = usize::MAX;
            for ch in 0..num_channels {
                let held = &self.hybrid_state.crossfade_held[ch];
                if !held.is_empty() {
                    self.channel_output_buffers[ch].clear();
                    self.channel_output_buffers[ch].extend_from_slice(held);
                    held_min_len = held_min_len.min(held.len());
                }
            }
            if held_min_len != usize::MAX && held_min_len > 0 {
                self.decode_output_mid_side(num_channels, held_min_len);
                self.emit_channel_output_to_pending(held_min_len, num_channels)?;
            }

            self.hybrid_state.reset(
                &self.params,
                self.current_ratio,
                self.capacity_frames_per_channel,
            );
            self.hybrid_pending_rebase = false;
        }

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
        if self.use_hybrid {
            return Err(StretchError::InvalidState(
                "fixed-buffer flush is unavailable in legacy hybrid mode",
            ));
        }

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
        if self.use_hybrid {
            return Err(StretchError::InvalidState(
                "fixed-buffer processing is unavailable in legacy hybrid mode",
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
        self.apply_current_dual_plane_ratio()?;

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

        if self.use_hybrid && !self.hybrid_realtime_strict {
            let min_output_len =
                self.process_hybrid_persistent_channels(num_channels, require_min_latency)?;
            let consumed = total_frames * num_channels;
            self.input_ring.discard(consumed);
            self.input_frames_consumed_total = self
                .input_frames_consumed_total
                .saturating_add(total_frames);

            if min_output_len > 0 {
                self.decode_output_mid_side(num_channels, min_output_len);
                self.emit_channel_output_to_pending(min_output_len, num_channels)?;
            }
            return Ok(());
        }

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

    fn finish_fixed_flush_drain(&mut self) {
        self.fixed_flush_pending = false;
        self.fixed_flush_source_exhausted = false;
        self.fixed_flush_pitch_tail_flushed = false;
        self.expected_total_output_samples = 0.0;
        self.total_output_emitted_samples = 0;
    }

    fn append_hybrid_input(&mut self, num_channels: usize) -> Result<(), StretchError> {
        let mut first_ch_pushed = 0;
        for ch in 0..num_channels {
            let input = &self.channel_input_buffers[ch];
            let rb = &mut self.hybrid_state.rolling_inputs[ch];
            if input.len() > rb.available() {
                rb.discard(input.len() - rb.available());
            }
            let pushed = rb.push_slice(input);
            if pushed != input.len() {
                return Err(StretchError::BufferOverflow {
                    buffer: "stream_hybrid_input",
                    requested: input.len(),
                    available: pushed,
                });
            }
            if ch == 0 {
                first_ch_pushed = pushed;
            }
        }
        self.hybrid_state.input_accumulated = self
            .hybrid_state
            .input_accumulated
            .saturating_add(first_ch_pushed);
        Ok(())
    }

    fn process_hybrid_persistent_channels(
        &mut self,
        num_channels: usize,
        allow_defer: bool,
    ) -> Result<usize, StretchError> {
        if self.hybrid_pending_rebase {
            self.hybrid_state.rebase_after_ratio_change();
            self.hybrid_pending_rebase = false;
        }
        self.hybrid_state.update_ratio(self.processing_ratio());

        self.append_hybrid_input(num_channels)?;

        // Accumulate enough new input before re-rendering.  With small
        // input chunks (e.g. 1024 samples) the output delta per render is
        // tiny and almost entirely consumed by the crossfade, producing
        // spectral-flux spikes at chunk boundaries that manifest as false
        // onsets.  Batching input to at least fft_size new samples per
        // render makes each delta large enough for the crossfade to be a
        // minor fraction, eliminating these artifacts.
        // Use 2× the FFT size for ratios far from unity (|r-1| > 0.1).
        // Higher thresholds reduce crossfade fraction but accumulate more
        // context change between renders, causing the skip estimate to
        // drift (HPSS segmentation shifts as the rolling window moves).
        // 2× balances crossfade fraction (~35%) against skip accuracy.
        // Using 2× for ALL ratios: near-unity ratios had 71% crossfade
        // at 1× which destroyed transient timing in house tracks;
        // at 2× the crossfade fraction drops to ~36%, preserving
        // more of each render's original onset positions.  Skip drift
        // is minimal for near-unity because the output-to-input ratio
        // is nearly constant across segments.
        let accum_threshold = self.params.fft_size * 2;
        if allow_defer && self.hybrid_state.input_accumulated < accum_threshold {
            return Ok(0);
        }

        let mut min_output_len = usize::MAX;
        self.hybrid_state.pre_trim_lens.fill(0);
        self.hybrid_state.rendered_lens.fill(0);

        // Phase 1: Snapshot all channels from rolling buffers.
        for ch in 0..num_channels {
            let len = self.hybrid_state.rolling_inputs[ch].len();
            self.hybrid_state.rolling_scratch[ch].resize(len, 0.0);
            let copied = self.hybrid_state.rolling_inputs[ch]
                .peek_slice(&mut self.hybrid_state.rolling_scratch[ch]);
            if copied != len {
                return Err(StretchError::InvalidState(
                    "failed to snapshot hybrid rolling ring",
                ));
            }
        }

        // Phase 2: For stereo M/S, detect shared transients from mid channel
        // so both channels use identical segmentation. This prevents phase
        // misalignment when decoded back to L/R, matching the batch path's
        // shared onset detection in stretch_mid_side().
        let shared_onsets: Option<(Vec<usize>, Vec<f32>)> = if num_channels == 2
            && self.params.stereo_mode == StereoMode::MidSide
            && !self.hybrid_state.rolling_scratch[0].is_empty()
        {
            let mid = &self.hybrid_state.rolling_scratch[0];
            let fft = self.params.fft_size.min(2048);
            let hop = self.params.hop_size.min(512);
            let map = detect_transients_with_options(
                mid,
                self.params.sample_rate,
                fft,
                hop,
                self.params.transient_sensitivity,
                TransientDetectionOptions::from_stretch_params(&self.params),
            );
            let onsets = map.onsets.clone();
            let strengths = if map.strengths.len() == onsets.len() {
                map.strengths.clone()
            } else {
                vec![1.0; onsets.len()]
            };
            Some((onsets, strengths))
        } else {
            None
        };

        // Phase 3: Process each channel and extract deltas.
        for ch in 0..num_channels {
            let rendered = if let Some((ref onsets, ref strengths)) = shared_onsets {
                self.hybrid_state.stretchers[ch].process_with_onsets(
                    &self.hybrid_state.rolling_scratch[ch],
                    onsets,
                    strengths,
                )?
            } else {
                self.hybrid_state.stretchers[ch].process(&self.hybrid_state.rolling_scratch[ch])?
            };
            let skip = self.hybrid_state.tail_output_lens[ch].min(rendered.len());
            let delta_len = rendered.len().saturating_sub(skip);

            if self.channel_output_buffers[ch].capacity() < delta_len {
                return Err(StretchError::BufferOverflow {
                    buffer: "stream_hybrid_output",
                    requested: delta_len,
                    available: self.channel_output_buffers[ch].capacity(),
                });
            }

            self.hybrid_state.pre_trim_lens[ch] = self.hybrid_state.rolling_scratch[ch].len();
            self.hybrid_state.rendered_lens[ch] = rendered.len();

            self.channel_output_buffers[ch].clear();

            // Cross-fade at the chunk boundary to smooth phase discontinuities.
            // The hybrid stretcher creates a fresh PV on each call, so the
            // absolute phase of the rendered output may differ between
            // consecutive calls for the overlapping tail region. Without
            // cross-fading, this creates clicks at chunk boundaries.
            //
            // Scale crossfade with the stretch ratio: larger ratios produce
            // synthesis frames farther apart, amplifying phase divergence
            // between consecutive PV renderings.
            let ratio_scale = self.hybrid_state.last_ratio.max(1.0);
            let xfade_base =
                (HYBRID_STREAM_CROSSFADE_SAMPLES as f64 * ratio_scale).round() as usize;
            let xfade = xfade_base.min(skip).min(delta_len * 7 / 8);
            let held = &self.hybrid_state.crossfade_held[ch];
            if !held.is_empty() && xfade > 0 {
                // Cross-fade: blend the held-back samples (previous delta end)
                // with the current rendering's prediction of that region
                // (rendered[skip-xfade..skip]).
                let overlap = &rendered[skip - xfade..skip];
                let n = held.len().min(xfade).min(overlap.len());

                // Adaptive crossfade: when held and overlap have very
                // different content (low correlation), a transient likely
                // appeared in one region but not the other.  A long
                // crossfade would smear the transient's attack, hurting
                // TP.  Shorten the crossfade to preserve sharpness while
                // still preventing clicks.
                let actual_xfade = if n >= 128 && (self.hybrid_state.last_ratio - 1.0).abs() > 0.1 {
                    let check = n.min(256);
                    let (mut dot, mut he, mut oe) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..check {
                        let h = held[i] as f64;
                        let o = overlap[i] as f64;
                        dot += h * o;
                        he += h * h;
                        oe += o * o;
                    }
                    let denom = (he * oe).sqrt();
                    let corr = if denom > 1e-12 {
                        (dot / denom).clamp(-1.0, 1.0)
                    } else {
                        1.0
                    };

                    if corr > 0.8 {
                        // High-correlation tonal content: the two renderings
                        // have similar magnitudes but potentially different PV
                        // phases. A long crossfade blends two phase-mismatched
                        // signals, creating FM artifacts (amplitude modulation
                        // that broadens spectral peaks and increases LSD).
                        // A shorter crossfade reduces the affected region while
                        // remaining smooth enough to avoid clicks.
                        (n / 2).max(128)
                    } else if corr < 0.3 {
                        // Also require an energy imbalance to distinguish
                        // a genuine transient onset (one region loud, the
                        // other quiet) from normal PV phase divergence on
                        // tonal content (both regions similarly loud but
                        // phase-shifted).
                        let h_rms = (he / check as f64).sqrt();
                        let o_rms = (oe / check as f64).sqrt();
                        let imbalance = h_rms.max(o_rms) / (h_rms.min(o_rms) + 1e-12);
                        if imbalance > 4.0 {
                            // Transient onset — shorten crossfade
                            (n / 4).max(64)
                        } else {
                            n
                        }
                    } else {
                        n
                    }
                } else {
                    n
                };

                for i in 0..actual_xfade {
                    let t = (i as f32 + 0.5) / actual_xfade as f32;
                    let s = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
                    self.channel_output_buffers[ch].push(held[i] * (1.0 - s) + overlap[i] * s);
                }
                // If crossfade was shortened, emit the remaining held
                // samples directly to avoid dropping the transient tail.
                if actual_xfade < n {
                    self.channel_output_buffers[ch].extend_from_slice(&overlap[actual_xfade..n]);
                }
            }

            // Always hold back a crossfade-sized tail, even on the first
            // render (when skip=0 and xfade=0).  Without this, the first
            // render emits all output with no holdback, creating a raw
            // waveform splice at the render-1→render-2 boundary.  That
            // discontinuity triggers false onset detection in spectral-
            // flux-based metrics.  By holding back on every render, the
            // next render always has crossfade material available.
            let holdback = xfade_base.min(delta_len * 7 / 8);

            // Emit the new delta, holding back the tail for next cross-fade.
            let emit_end = delta_len.saturating_sub(holdback);
            self.channel_output_buffers[ch].extend_from_slice(&rendered[skip..skip + emit_end]);

            // Save the tail for the next cross-fade.
            let held_tail = &mut self.hybrid_state.crossfade_held[ch];
            held_tail.clear();
            held_tail.extend_from_slice(&rendered[skip + emit_end..skip + delta_len]);

            min_output_len = min_output_len.min(self.channel_output_buffers[ch].len());
        }

        self.hybrid_state.input_accumulated = 0;
        self.hybrid_state.retain_tail();
        self.hybrid_state
            .update_tail_output_estimates_from_rendered();

        Ok(if min_output_len == usize::MAX {
            0
        } else {
            min_output_len
        })
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
        if (ratio - self.target_ratio).abs() > RATIO_SNAP_THRESHOLD {
            self.hybrid_pending_rebase = true;
        }
        self.target_ratio = ratio;
        Ok(())
    }

    /// Enables or disables the legacy rolling-window hybrid re-render mode.
    ///
    /// This is equivalent to selecting
    /// [`StreamingEngine::LegacyHybridRerender`] when enabled and
    /// [`StreamingEngine::Deterministic`] when disabled.
    pub fn set_hybrid_mode(&mut self, enabled: bool) {
        if enabled && self.dual_plane_deterministic.is_some() {
            self.dual_plane_deterministic = None;
        }
        if enabled && !self.use_hybrid {
            self.hybrid_state.reset(
                &self.params,
                self.current_ratio,
                self.capacity_frames_per_channel,
            );
            self.hybrid_pending_rebase = false;
        }
        if self.use_hybrid != enabled {
            self.reset_pitch_resamplers();
        }
        self.use_hybrid = enabled;
        if !enabled {
            self.ensure_default_dual_plane_backend();
        }
    }

    /// Returns whether hybrid processing mode is enabled.
    pub fn is_hybrid_mode(&self) -> bool {
        self.use_hybrid
    }

    /// Selects the streaming engine implementation.
    ///
    /// [`StreamingEngine::Deterministic`] is the recommended real-time path.
    /// [`StreamingEngine::LegacyHybridRerender`] keeps the previous
    /// rolling-window hybrid behavior as an explicit opt-in mode.
    pub fn set_streaming_engine(&mut self, engine: StreamingEngine) {
        match engine {
            StreamingEngine::Deterministic => self.set_hybrid_mode(false),
            StreamingEngine::LegacyHybridRerender => self.set_hybrid_mode(true),
        }
    }

    /// Returns the currently selected streaming engine.
    pub fn streaming_engine(&self) -> StreamingEngine {
        if self.use_hybrid {
            StreamingEngine::LegacyHybridRerender
        } else {
            StreamingEngine::Deterministic
        }
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
        if self.use_hybrid {
            return Err(StretchError::InvalidState(
                "deterministic latency profile is unavailable in legacy hybrid mode",
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
        if self.use_hybrid {
            return Err(StretchError::InvalidState(
                "deterministic auto profile switching is unavailable in legacy hybrid mode",
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

    /// Returns the exact current host-visible delay in samples for deterministic mode.
    ///
    /// This includes algorithmic delay, buffered input/output, and current
    /// profile/tier contributions. Returns `None` when deterministic
    /// dual-plane telemetry is unavailable.
    pub fn current_delay_samples(&self) -> Option<usize> {
        self.deterministic_delay_telemetry()
            .map(|telemetry| telemetry.total_frames)
    }

    /// Returns the exact current host-visible delay in seconds for deterministic mode.
    pub fn current_delay_secs(&self) -> Option<f64> {
        self.current_delay_samples()
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

    /// Enables or disables strict realtime-safe behavior while hybrid mode is on.
    ///
    /// Strict mode routes processing through the preallocated PV stream path to
    /// guarantee no heap growth in callbacks. This flag only matters when
    /// [`StreamingEngine::LegacyHybridRerender`] is selected.
    pub fn set_hybrid_realtime_strict(&mut self, enabled: bool) {
        if self.hybrid_realtime_strict != enabled {
            self.hybrid_pending_rebase = true;
            self.reset_pitch_resamplers();
        }
        self.hybrid_realtime_strict = enabled;
    }

    /// Returns whether strict realtime-safe hybrid mode is enabled.
    pub fn is_hybrid_realtime_strict(&self) -> bool {
        self.hybrid_realtime_strict
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
    /// stream per channel by `1.0 / pitch_scale` to preserve target tempo.
    pub fn set_pitch_scale(&mut self, scale: f64) -> Result<(), StretchError> {
        let scale = validate_positive_finite_ratio(scale, "pitch scale")?;
        if (scale - self.pitch_scale).abs() > RATIO_SNAP_THRESHOLD {
            self.hybrid_pending_rebase = true;
            self.reset_pitch_resamplers();
        }
        self.pitch_scale = scale;
        if dual_plane_supports_pitch_scale(scale) {
            self.ensure_default_dual_plane_backend();
        }
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

    /// Returns the minimum latency in samples.
    ///
    /// For exact current deterministic delay, use
    /// [`StreamProcessor::current_delay_samples`] or
    /// [`StreamProcessor::deterministic_delay_telemetry`].
    pub fn latency_samples(&self) -> usize {
        min_latency_frames(self.params.fft_size)
    }

    /// Returns the minimum latency in seconds.
    ///
    /// For exact current deterministic delay, use
    /// [`StreamProcessor::current_delay_secs`] or
    /// [`StreamProcessor::deterministic_delay_telemetry`].
    pub fn latency_secs(&self) -> f64 {
        self.latency_samples() as f64 / self.params.sample_rate as f64
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
        self.hybrid_state.reset(
            &self.params,
            self.params.stretch_ratio,
            self.capacity_frames_per_channel,
        );
        self.hybrid_pending_rebase = false;
        self.expected_total_output_samples = 0.0;
        self.total_output_emitted_samples = 0;
        self.pitch_scale = 1.0;
        self.reset_pitch_resamplers();
        self.fixed_flush_pending = false;
        self.fixed_flush_source_exhausted = false;
        self.fixed_flush_pitch_tail_flushed = false;

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
            .with_fft_size(4096);

        let proc = StreamProcessor::new(params);
        // 4096 * 3 / 2 = 6144 (1.5x FFT size for reduced latency)
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
    fn test_stream_processor_hybrid_mode_default() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let proc = StreamProcessor::new(params);
        assert!(!proc.is_hybrid_mode());
    }

    #[test]
    fn test_stream_processor_hybrid_mode_toggle() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);

        proc.set_hybrid_mode(true);
        assert!(proc.is_hybrid_mode());

        proc.set_hybrid_mode(false);
        assert!(!proc.is_hybrid_mode());
    }

    #[test]
    fn test_stream_processor_streaming_engine_default() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let proc = StreamProcessor::new(params);
        assert_eq!(proc.streaming_engine(), StreamingEngine::Deterministic);
        assert!(proc.is_dual_plane_deterministic());
    }

    #[test]
    fn test_stream_processor_streaming_engine_toggle() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);

        proc.set_streaming_engine(StreamingEngine::LegacyHybridRerender);
        assert_eq!(
            proc.streaming_engine(),
            StreamingEngine::LegacyHybridRerender
        );
        assert!(proc.is_hybrid_mode());

        proc.set_streaming_engine(StreamingEngine::Deterministic);
        assert_eq!(proc.streaming_engine(), StreamingEngine::Deterministic);
        assert!(!proc.is_hybrid_mode());
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

        let telemetry = proc
            .deterministic_delay_telemetry()
            .expect("dual-plane delay telemetry should be available");
        assert_eq!(proc.current_delay_samples(), Some(telemetry.total_frames));
        assert_eq!(
            proc.current_delay_secs(),
            Some(telemetry.total_frames as f64 / 44_100.0)
        );
    }

    #[test]
    fn test_stream_processor_current_delay_accessors_unavailable_in_hybrid_mode() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44_100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        proc.set_hybrid_mode(true);

        assert_eq!(proc.current_delay_samples(), None);
        assert_eq!(proc.current_delay_secs(), None);
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

    #[test]
    fn test_stream_processor_hybrid_realtime_strict_toggle() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        assert!(!proc.is_hybrid_realtime_strict());

        proc.set_hybrid_realtime_strict(true);
        assert!(proc.is_hybrid_realtime_strict());

        proc.set_hybrid_realtime_strict(false);
        assert!(!proc.is_hybrid_realtime_strict());
    }

    #[test]
    fn test_stream_processor_hybrid_realtime_strict_produces_output() {
        let params = StretchParams::new(1.15)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        proc.set_hybrid_mode(true);
        proc.set_hybrid_realtime_strict(true);

        let signal: Vec<f32> = (0..44100)
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 44100.0).sin() * 0.8)
            .collect();

        let mut out = Vec::with_capacity(signal.len() * 2);
        for chunk in signal.chunks(1024) {
            proc.process_into(chunk, &mut out).unwrap();
        }
        proc.flush_into(&mut out).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_stream_processor_hybrid_produces_output() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(crate::core::types::EdmPreset::HouseLoop);

        let mut proc = StreamProcessor::new(params);
        proc.set_hybrid_mode(true);

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
            "Hybrid mode should produce output"
        );
    }

    #[test]
    fn test_stream_processor_hybrid_stretch_ratio() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        proc.set_hybrid_mode(true);

        // Feed enough data in one go for reliable ratio measurement
        let num_samples = 44100 * 2;
        let signal: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut total_output = Vec::new();
        if let Ok(out) = proc.process(&signal) {
            total_output.extend_from_slice(&out);
        }
        if let Ok(out) = proc.flush() {
            total_output.extend_from_slice(&out);
        }

        if !total_output.is_empty() {
            let ratio = total_output.len() as f64 / signal.len() as f64;
            assert!(
                (0.4..=2.2).contains(&ratio),
                "Hybrid stretch ratio {} out of expected real-time range",
                ratio
            );
            assert!(total_output.iter().all(|s| s.is_finite()));
        }
    }

    #[test]
    fn test_stream_processor_hybrid_state_persists_across_calls() {
        let params = StretchParams::new(1.25)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(crate::core::types::EdmPreset::HouseLoop);

        let mut proc = StreamProcessor::new(params);
        proc.set_hybrid_mode(true);

        let signal: Vec<f32> = (0..44100 * 3)
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 44100.0).sin())
            .collect();

        let _ = proc.process(&signal[..16384]).unwrap();
        let emitted_after_first = proc.hybrid_state.tail_output_lens[0];
        assert!(
            emitted_after_first > 0,
            "Expected hybrid state to emit output after first call"
        );

        let _ = proc.process(&signal[16384..32768]).unwrap();
        let emitted_after_second = proc.hybrid_state.tail_output_lens[0];
        assert!(
            emitted_after_second > 0,
            "Hybrid emitted estimate should remain valid across calls ({} -> {})",
            emitted_after_first,
            emitted_after_second
        );
    }

    #[test]
    fn test_stream_processor_hybrid_rejects_nan() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        proc.set_hybrid_mode(true);

        let mut chunk = vec![0.0f32; 4096];
        chunk[100] = f32::NAN;
        assert!(matches!(
            proc.process(&chunk),
            Err(crate::error::StretchError::NonFiniteInput)
        ));
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
    fn test_max_process_interleaved_output_samples_rejects_hybrid_mode() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44_100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        proc.set_hybrid_mode(true);

        let err = proc.max_process_interleaved_output_samples(16).unwrap_err();
        assert_eq!(
            err,
            StretchError::InvalidState(
                "fixed-buffer processing is unavailable in legacy hybrid mode"
            )
        );
    }

    #[test]
    fn test_process_interleaved_into_rejects_hybrid_mode() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44_100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        proc.set_hybrid_mode(true);

        let input = [0.0f32; 16];
        let mut output = [0.0f32; 32];
        let err = proc
            .process_interleaved_into(&input, &mut output)
            .unwrap_err();
        assert_eq!(
            err,
            StretchError::InvalidState(
                "fixed-buffer processing is unavailable in legacy hybrid mode"
            )
        );
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
    fn test_max_flush_interleaved_output_samples_rejects_hybrid_mode() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44_100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        proc.set_hybrid_mode(true);

        let err = proc.max_flush_interleaved_output_samples().unwrap_err();
        assert_eq!(
            err,
            StretchError::InvalidState("fixed-buffer flush is unavailable in legacy hybrid mode")
        );
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
    fn test_flush_interleaved_into_rejects_hybrid_mode() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44_100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        proc.set_hybrid_mode(true);

        let mut output = [0.0f32; 32];
        let err = proc.flush_interleaved_into(&mut output).unwrap_err();
        assert_eq!(
            err,
            StretchError::InvalidState("fixed-buffer flush is unavailable in legacy hybrid mode")
        );
    }

    #[test]
    fn test_process_into_hybrid_mode() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(crate::core::types::EdmPreset::HouseLoop);

        let mut proc = StreamProcessor::new(params);
        proc.set_hybrid_mode(true);

        let signal: Vec<f32> = (0..44100)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut output = Vec::with_capacity(signal.len() * 3);
        proc.process_into(&signal, &mut output).unwrap();
        proc.flush_into(&mut output).unwrap();

        assert!(
            !output.is_empty(),
            "Hybrid process_into should produce output"
        );
    }

    #[test]
    fn test_stream_processor_hybrid_stereo() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(2);

        let mut proc = StreamProcessor::new(params);
        proc.set_hybrid_mode(true);

        let num_frames = 44100;
        let mut signal = vec![0.0f32; num_frames * 2];
        for i in 0..num_frames {
            let t = i as f32 / 44100.0;
            signal[i * 2] = (2.0 * PI * 440.0 * t).sin();
            signal[i * 2 + 1] = (2.0 * PI * 880.0 * t).sin();
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

        assert!(
            !total_output.is_empty(),
            "Hybrid stereo should produce output"
        );
        assert_eq!(
            total_output.len() % 2,
            0,
            "Stereo output must have even sample count"
        );
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

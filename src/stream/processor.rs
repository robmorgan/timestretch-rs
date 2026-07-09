//! Real-time streaming time-stretch processor.

use std::sync::Arc;

use crate::analysis::adaptive_snapshot::merge_onsets_and_beats;
use crate::analysis::transient::{detect_transients_with_options, TransientDetectionOptions};
use crate::core::preanalysis::PreAnalysisArtifact;
use crate::core::resample::{SincInterpTable, StreamingSincResampler};
use crate::core::ring_buffer::RingBuffer;
use crate::core::types::{QualityMode, StreamProfile, StretchParams};
use crate::error::StretchError;
use crate::stream::transient_scheduler::{TransientEventScheduler, TransientSchedulerStats};
use crate::stretch::hybrid::HybridStretcher;
use crate::stretch::phase_vocoder::PhaseVocoder;
use crate::stretch::stereo::StereoMode;
use crate::stretch::wsola::Wsola;

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
/// Frames over which end-of-stream padding fades from the last real input
/// frame to silence (~5.8 ms at 44.1 kHz) instead of cutting hard to zero.
const FLUSH_PAD_FADE_FRAMES: usize = 256;
/// Samples over which a freshly armed WSOLA overlay ramps in (~2.9 ms).
///
/// The overlay splices WSOLA-rendered audio (arbitrary phase relative to the
/// PV output) over the stream; starting it at full weight is an audible
/// click whenever the two signals disagree at the splice point.
const WSOLA_OVERLAY_FADE_IN_SAMPLES: usize = 128;
/// In-flight processing-ratio delta above which the transient scheduler and
/// WSOLA overlay arming enter modulation-hold. ~0.2% ratio ≈ 3.5 cents;
/// below this a ratio seam is inaudible.
const MODULATION_HOLD_MIN_RATIO_DELTA: f64 = 0.002;
/// Time cap on the modulation-hold budget: how long (in absolute audio time)
/// trigger desensitization may persist after a control change. Chosen so the
/// window cap equals the legacy 4 windows at the reference 4096-FFT /
/// 44.1 kHz configuration; smaller FFT sizes get proportionally more (and
/// shorter) windows covering the same time span. The scheduler scales its
/// per-window desensitization steps by `fft/4096`, keeping the ceilings
/// constant (threshold scale <= 1.32, spike ratio <= 1.2x). Long gestures
/// stay protected anyway because the in-flight delta is re-evaluated on
/// every scheduler pass.
const MODULATION_HOLD_MAX_SECS: f64 = 0.371;
/// Hard ceiling for the derived modulation-hold window cap.
const MODULATION_HOLD_MAX_WINDOWS_CEILING: usize = 16;

/// Multiplier on the per-channel input capacity used to size the streaming
/// output buffers. A single render over the retained input window emits up
/// to ~`stretch_ratio` times its frame count, so this factor sets the
/// largest ratio that will not overflow the PV output buffer (8x keeps
/// ratios up to ~10x, i.e. down to quarter-... tenth-speed, safe).
const OUTPUT_CAPACITY_MULTIPLIER: usize = 8;

/// Declick fade-in applied to the first output samples after a warm-start
/// seek (interleaved samples; ~3 ms stereo at 44.1 kHz). The caller cut the
/// previous stream mid-waveform, so the resumed stream ramps in briefly.
const WARM_START_FADE_SAMPLES: usize = 256;

/// Minimum artifact onset strength for a low-band (100-500 Hz) phase reset.
const ARTIFACT_LOW_BAND_RESET_STRENGTH: f32 = 0.45;
/// Minimum artifact onset strength for a sub-bass (<100 Hz) phase reset.
const ARTIFACT_SUB_BASS_RESET_STRENGTH: f32 = 0.7;

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
    /// Absolute source frame of the rolling window's first retained frame.
    ///
    /// Advanced on every front-discard (overflow and tail retention) so
    /// pre-analysis artifact positions can be mapped into window-relative
    /// onsets. All channels discard in lockstep; channel 0 is authoritative.
    window_base_abs: usize,
    /// True once `window_base_abs` has been anchored by the first append.
    window_base_valid: bool,
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
        // The rolling window is window-relative while artifact positions are
        // absolute source frames; the streaming path maps them explicitly
        // (see `window_base_abs`), so the stretchers must never consume the
        // artifact through their own batch-oriented analysis.
        per_channel.pre_analysis = None;
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
            window_base_abs: 0,
            window_base_valid: false,
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
        for (ch, input) in self.rolling_inputs.iter_mut().enumerate() {
            if input.len() > self.max_tail_frames {
                let discarded = input.len() - self.max_tail_frames;
                input.discard(discarded);
                if ch == 0 {
                    self.window_base_abs = self.window_base_abs.saturating_add(discarded);
                }
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

/// Quality of the realtime pitch resampler used when `pitch_scale != 1.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPitchQuality {
    /// Kaiser-windowed sinc resampling with ratio-adaptive anti-aliasing
    /// (default).
    ///
    /// Keeps pitch-up material (hats, cymbals, vocals) alias-free at the cost
    /// of a few dozen multiply-adds per output sample and up to
    /// [`crate::core::resample::STREAM_SINC_MAX_HALF_TAPS`] samples of extra
    /// buffering while pitch is engaged.
    Sinc,
    /// Legacy linear interpolation.
    ///
    /// Cheapest possible pitch control; audibly aliases bright material when
    /// pitching up. Kept as an explicit low-CPU/emergency fallback.
    Linear,
}

/// Breakdown of a [`StreamProcessor`]'s current effective latency.
///
/// Produced by [`StreamProcessor::latency_report`]. All figures reflect the
/// *current control targets*, so a ratio or pitch change updates the report
/// at the control call, not after the ~50 ms glide.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamLatencyReport {
    /// Input-buffering floor in frames: `fft_size * 3/2`.
    pub base_gate_frames: usize,
    /// Actual buffering gate in frames at the current target processing
    /// ratio: equals `base_gate_frames` inside the `[0.9, 1.1]` ratio band,
    /// `fft_size * 2` outside it.
    pub effective_gate_frames: usize,
    /// Kernel lookahead of the realtime pitch resampler in samples: 0 when
    /// pitch is inactive; 16 (sinc, unity/pitch-down) up to 80 (sinc,
    /// pitch-up); 1 for the linear fallback.
    pub pitch_lookahead_samples: usize,
    /// Time constant of the ratio/pitch control glide in seconds (~0.050).
    /// Control changes become fully audible a few time constants after the
    /// buffering delay, not instantly.
    pub control_smoothing_secs: f64,
    /// Sample rate the frame figures are expressed against.
    pub sample_rate: u32,
    /// Total effective latency in frames:
    /// `effective_gate_frames + pitch_lookahead_samples`.
    pub total_frames: usize,
}

impl StreamLatencyReport {
    /// Total effective latency in seconds.
    pub fn total_secs(&self) -> f64 {
        self.total_frames as f64 / self.sample_rate.max(1) as f64
    }
}

/// Aggregated transient-reset telemetry from deterministic stream processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransientResetStats {
    /// Number of transient events detected by the online scheduler.
    pub events_detected_total: u64,
    /// Number of phase resets scheduled from pre-analysis artifact onsets.
    ///
    /// When a usable artifact is attached, artifact scheduling replaces the
    /// online scheduler, so this and `events_detected_total` are mutually
    /// exclusive within one stream.
    pub artifact_events_scheduled_total: u64,
    /// Number of times each reset band was selected across all events
    /// (online-scheduled and artifact-scheduled combined).
    ///
    /// Layout: `[sub_bass, low, mid, high]`.
    pub reset_band_counts_total: [u64; 4],
    /// Absolute per-channel input frames consumed from the stream so far.
    pub input_frames_consumed_total: usize,
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
    /// Absolute source frame of the first frame pushed after creation/reset.
    ///
    /// Set via [`StreamProcessor::set_source_position`] so pre-analysis
    /// artifact positions stay aligned when streaming starts mid-file
    /// (seek/cue rebuild flows).
    source_start_frames: usize,
    /// Frames emitted by the unity passthrough fast path, which bypasses
    /// `input_ring` and never advances `input_frames_consumed_total`.
    passthrough_frames_total: usize,
    /// Cached result of the artifact usability gate (recomputed whenever the
    /// artifact changes; never in the audio callback).
    artifact_active: bool,
    /// Monotonic cursor into the artifact's `transient_onsets` for the
    /// deterministic engine's scheduled phase resets.
    artifact_onset_cursor: usize,
    /// Count of phase resets scheduled from artifact onsets.
    artifact_events_scheduled_total: u64,
    /// Per-band counts of artifact-scheduled resets `[sub, low, mid, high]`.
    artifact_reset_band_counts_total: [u64; 4],
    /// Artifact onsets merged with beat anchors (absolute source frames),
    /// precomputed for the legacy hybrid re-render engine.
    artifact_merged_anchors: (Vec<usize>, Vec<f32>),
    /// Remaining interleaved samples of the post-warm-start declick fade.
    warm_start_fade_remaining: usize,
    /// Expected total output samples across the current stream.
    ///
    /// Accumulated from input samples and the effective interpolated ratio,
    /// then reconciled on flush to avoid long-run drift.
    expected_total_output_samples: f64,
    /// Total output samples emitted to the caller for the current stream.
    total_output_emitted_samples: usize,
    /// Realtime pitch scale applied in stream mode (smoothed toward
    /// `target_pitch_scale` alongside the stretch-ratio glide).
    pitch_scale: f64,
    /// Target pitch scale set by [`StreamProcessor::set_pitch_scale`].
    target_pitch_scale: f64,
    /// Selected realtime pitch resampler quality.
    pitch_quality: StreamPitchQuality,
    /// Stateful per-channel sinc resamplers for realtime pitch control.
    ///
    /// Each holds an `Arc` of the shared Kaiser-sinc prototype table.
    sinc_pitch_resamplers: Vec<StreamingSincResampler>,
    /// True once the pitch resampler has processed any samples this stream.
    ///
    /// While engaged, the resampler stays in the signal path even at unity
    /// pitch (a bit-clean passthrough) so returning to `pitch_scale == 1.0`
    /// never splices its held lookahead out of the stream.
    pitch_resampler_engaged: bool,
    /// True once the PV processing path has consumed input this stream.
    ///
    /// While engaged, the bit-exact unity passthrough stays disabled even if
    /// the ratio settles back to exactly 1.0: switching between the raw
    /// input and the PV's rendered stream mid-playback is a phase-arbitrary
    /// splice (a full-scale click on tonal content) plus a re-warmup gap.
    dsp_engaged: bool,
    /// Stateful per-channel linear resamplers (legacy pitch fallback).
    pitch_resamplers: Vec<LinearResamplerState>,
    /// Reusable per-channel output buffers for pitch-resampled data.
    pitch_output_buffers: Vec<Vec<f32>>,
    /// EMA-tracked input energy (RMS²) for gain compensation.
    input_energy_ema: f64,
    /// EMA-tracked output energy (RMS²) for gain compensation.
    output_energy_ema: f64,
    /// Smoothed energy gain factor applied to output.
    energy_gain: f64,
    /// Count of gain compensation calls for warmup tracking.
    gain_call_count: usize,
    /// EMA-tracked high-frequency input energy for spectral shape correction.
    input_hf_energy_ema: f64,
    /// EMA-tracked high-frequency output energy for spectral shape correction.
    output_hf_energy_ema: f64,
    /// High-pass filter state for input HF energy measurement.
    hf_input_hp_state: f64,
    /// High-pass filter state for output HF energy measurement.
    hf_output_hp_state: f64,
    /// EMA-tracked mid-band input energy (500-2000 Hz) for centroid correction.
    input_mid_energy_ema: f64,
    /// EMA-tracked mid-band output energy (500-2000 Hz) for centroid correction.
    output_mid_energy_ema: f64,
    /// Bandpass filter states for input mid-band measurement [hp_state, lp_state].
    mid_input_bp_state: [f64; 2],
    /// Bandpass filter states for output mid-band measurement [hp_state, lp_state].
    mid_output_bp_state: [f64; 2],
    /// Previous input-frame RMS for simple onset-energy tracking on the helper path.
    prev_blend_input_rms: f32,
    /// Per-channel WSOLA instances for transient overlay processing.
    wsola_instances: Vec<Wsola>,
    /// Per-channel WSOLA output scratch buffers.
    wsola_output_buffers: Vec<Vec<f32>>,
    /// Remaining output samples where WSOLA overlay is crossfaded over PV.
    wsola_overlay_remaining: usize,
    /// Total length of the current WSOLA overlay crossfade window.
    wsola_overlay_total: usize,
    /// Per-channel WSOLA overlay samples (pre-rendered).
    wsola_overlay_buffers: Vec<Vec<f32>>,
    /// Current read position within the WSOLA overlay buffers.
    wsola_overlay_pos: usize,
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
            .field("source_bpm", &self.source_bpm)
            .field("input_ring_len", &self.input_ring.len())
            .field("pending_output_len", &self.pending_output.len())
            .finish()
    }
}

impl StreamProcessor {
    /// Creates a new streaming processor.
    pub fn new(params: StretchParams) -> Self {
        let ratio = params.stretch_ratio;
        let num_channels = params.channels.count();
        let source_bpm = params.bpm;
        let sample_rate = params.sample_rate;
        let wsola_seg_size = params.wsola_segment_size;
        let wsola_search = params.wsola_search_range;

        let capacity_frames_per_channel = stream_capacity_frames(&params);
        let capacity_samples = capacity_frames_per_channel.saturating_mul(num_channels);
        // Output headroom is sized for the widest supported stretch (a render
        // over the retained input emits up to ~ratio times as many frames).
        // The multiplier bounds the maximum usable stretch ratio: 8x the
        // input-capacity window keeps ratios up to ~10x from overflowing the
        // per-render PV output buffer (see `OUTPUT_CAPACITY_MULTIPLIER`).
        let output_capacity_frames = capacity_frames_per_channel
            .saturating_mul(OUTPUT_CAPACITY_MULTIPLIER)
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
        let sinc_table = SincInterpTable::new_stream_default();

        let mut processor = Self {
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
            source_start_frames: 0,
            passthrough_frames_total: 0,
            artifact_active: false,
            artifact_onset_cursor: 0,
            artifact_events_scheduled_total: 0,
            artifact_reset_band_counts_total: [0; 4],
            artifact_merged_anchors: (Vec::new(), Vec::new()),
            warm_start_fade_remaining: 0,
            expected_total_output_samples: 0.0,
            total_output_emitted_samples: 0,
            pitch_scale: 1.0,
            target_pitch_scale: 1.0,
            pitch_quality: StreamPitchQuality::Sinc,
            sinc_pitch_resamplers: (0..num_channels)
                .map(|_| StreamingSincResampler::new(Arc::clone(&sinc_table)))
                .collect(),
            pitch_resampler_engaged: false,
            dsp_engaged: false,
            pitch_resamplers: (0..num_channels)
                .map(|_| LinearResamplerState::new())
                .collect(),
            pitch_output_buffers: (0..num_channels)
                .map(|_| Vec::with_capacity(pitch_output_capacity_frames))
                .collect(),
            input_energy_ema: 0.0,
            output_energy_ema: 0.0,
            energy_gain: 1.0,
            gain_call_count: 0,
            input_hf_energy_ema: 0.0,
            output_hf_energy_ema: 0.0,
            hf_input_hp_state: 0.0,
            hf_output_hp_state: 0.0,
            input_mid_energy_ema: 0.0,
            output_mid_energy_ema: 0.0,
            mid_input_bp_state: [0.0; 2],
            mid_output_bp_state: [0.0; 2],
            prev_blend_input_rms: 0.0,
            wsola_instances: (0..num_channels)
                .map(|_| {
                    // Use preset-configured WSOLA parameters for consistency with batch path.
                    let seg = if wsola_seg_size > 0 {
                        wsola_seg_size
                    } else {
                        (sample_rate as f64 * 0.030).round() as usize
                    };
                    let search = if wsola_search > 0 {
                        wsola_search
                    } else {
                        (sample_rate as f64 * 0.015).round() as usize
                    };
                    let mut w = Wsola::new(seg, search, ratio);
                    w.set_equal_power_crossfade();
                    w.reserve_output_capacity(capacity_frames_per_channel, ratio.max(2.5));
                    w
                })
                .collect(),
            wsola_output_buffers: (0..num_channels)
                .map(|_| Vec::with_capacity(output_capacity_frames))
                .collect(),
            wsola_overlay_remaining: 0,
            wsola_overlay_total: 0,
            wsola_overlay_buffers: (0..num_channels)
                .map(|_| Vec::with_capacity(output_capacity_frames))
                .collect(),
            wsola_overlay_pos: 0,
        };
        processor.refresh_artifact_state();
        processor
    }

    /// Creates PhaseVocoder instances for each channel.
    fn create_vocoders(params: &StretchParams, ratio: f64) -> Vec<PhaseVocoder> {
        (0..params.channels.count())
            .map(|_| {
                // Use a wider sub-bass rigid phase locking range for streaming
                // to improve phase coherence for low-mid frequencies. The streaming
                // PV doesn't benefit from WSOLA fallback, so tighter phase control
                // at low frequencies helps preserve energy at those bins.
                let streaming_sub_bass_cutoff = params.sub_bass_cutoff.max(180.0);
                let mut pv = PhaseVocoder::with_all_options(
                    params.fft_size,
                    params.hop_size,
                    ratio,
                    params.sample_rate,
                    streaming_sub_bass_cutoff,
                    params.window_type,
                    params.phase_locking_mode,
                    params.envelope_preservation,
                    params.envelope_order,
                );
                // Disable adaptive phase locking to use the configured mode consistently.
                pv.set_adaptive_phase_locking(false);
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

    /// Processes a chunk of interleaved audio, appending output to `output`.
    ///
    /// This is the real-time API. It does not grow internal buffers in the
    /// steady state and it never shifts buffer memory.
    pub fn process_into(
        &mut self,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<(), StretchError> {
        if input.iter().any(|s| !s.is_finite()) {
            return Err(StretchError::NonFiniteInput);
        }

        self.initialized = true;

        // Fast passthrough for unity ratio: skip PV/WSOLA processing to
        // produce bit-exact output and eliminate windowing/overlap-add drift.
        if (self.target_ratio - 1.0).abs() < RATIO_SNAP_THRESHOLD
            && (self.current_ratio - 1.0).abs() < RATIO_SNAP_THRESHOLD
            && (self.pitch_scale - 1.0).abs() < RATIO_SNAP_THRESHOLD
            && (self.target_pitch_scale - 1.0).abs() < RATIO_SNAP_THRESHOLD
            && !self.pitch_resampler_engaged
            && !self.dsp_engaged
        {
            let available = output.capacity().saturating_sub(output.len());
            if input.len() > available {
                return Err(StretchError::BufferOverflow {
                    buffer: "process_into_output",
                    requested: input.len(),
                    available,
                });
            }
            if self.warm_start_fade_remaining > 0 {
                // Declick ramp after a unity warm-start seek; degrades the
                // bit-exact guarantee only for the first few ms post-seek.
                let total = WARM_START_FADE_SAMPLES as f32;
                for &sample in input {
                    let scaled = if self.warm_start_fade_remaining > 0 {
                        let progress = 1.0 - self.warm_start_fade_remaining as f32 / total;
                        self.warm_start_fade_remaining -= 1;
                        sample * progress
                    } else {
                        sample
                    };
                    output.push(scaled);
                }
            } else {
                output.extend_from_slice(input);
            }
            self.passthrough_frames_total = self
                .passthrough_frames_total
                .saturating_add(input.len() / self.params.channels.count().max(1));
            return Ok(());
        }

        if !input.is_empty() {
            self.dsp_engaged = true;
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
        let before = output.len();
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
                // Pad with a short fade-out continuation of the last input
                // frame instead of raw zeros: a hard cut to zero mid-waveform
                // is an input discontinuity that the vocoder faithfully
                // reproduces as an audible click in the flush tail. (Flush is
                // not the realtime callback; allocation here is fine.)
                let existing = self.input_ring.len();
                let mut last_frame = vec![0.0f32; num_channels.max(1)];
                if existing >= num_channels && num_channels > 0 {
                    let mut ring_copy = vec![0.0f32; existing];
                    let copied = self.input_ring.peek_slice(&mut ring_copy);
                    if copied == existing {
                        last_frame.copy_from_slice(&ring_copy[existing - num_channels..]);
                    }
                }
                let channel_phase = if num_channels > 0 {
                    existing % num_channels
                } else {
                    0
                };
                let fade_frames = (missing / num_channels.max(1)).min(FLUSH_PAD_FADE_FRAMES);
                let mut pad = Vec::with_capacity(missing);
                for i in 0..missing {
                    let frame_idx = i / num_channels.max(1);
                    let ch = (channel_phase + i) % num_channels.max(1);
                    let sample = if frame_idx < fade_frames {
                        let t = 1.0 - (frame_idx + 1) as f32 / fade_frames.max(1) as f32;
                        last_frame[ch] * t
                    } else {
                        0.0
                    };
                    pad.push(sample);
                }
                let pushed = self.input_ring.push_slice(&pad);
                if pushed != missing {
                    return Err(StretchError::BufferOverflow {
                        buffer: "stream_input_ring",
                        requested: missing,
                        available: pushed,
                    });
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
        self.prev_blend_input_rms = 0.0;
        self.wsola_overlay_remaining = 0;
        self.wsola_overlay_total = 0;
        self.wsola_overlay_pos = 0;
        for buf in &mut self.wsola_overlay_buffers {
            buf.clear();
        }

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
                // Truncation can cut mid-waveform; fade the new end so the
                // stream does not click against the silence that follows.
                let remaining_here = output.len().saturating_sub(before);
                fade_out_tail(output, 128.min(remaining_here));
            }
        }

        // Start a fresh accounting window after flush.
        self.expected_total_output_samples = 0.0;
        self.total_output_emitted_samples = 0;
        self.dsp_engaged = false;
        Ok(output.len().saturating_sub(before))
    }

    /// Flushes remaining buffered samples.
    pub fn flush(&mut self) -> Result<Vec<f32>, StretchError> {
        let mut out = Vec::with_capacity(self.pending_output.capacity());
        self.flush_into(&mut out)?;
        Ok(out)
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
            let min_frames =
                effective_min_frames(self.params.fft_size, self.target_processing_ratio());
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
        // Exclusive: artifact-scheduled resets replace the online scheduler
        // when a usable artifact is attached (never both, or transients
        // would double-reset).
        if self.artifact_active {
            self.apply_artifact_scheduled_phase_reset(total_frames, num_channels);
        } else {
            self.apply_transient_scheduled_phase_reset(total_frames, num_channels);
        }

        // Track input energy for gain compensation.
        {
            let input_energy = self.channel_input_buffers[0]
                .iter()
                .map(|&s| (s as f64) * (s as f64))
                .sum::<f64>()
                / self.channel_input_buffers[0].len().max(1) as f64;
            // Fast warmup for the first 10 calls to converge quickly,
            // then settle to stable tracking.
            // Ratio-adaptive EMA: extreme ratios get faster convergence
            // because PV energy loss is larger and needs quicker correction.
            let rd = (self.current_ratio - 1.0).abs();
            let ema_alpha = if self.gain_call_count < 5 {
                0.15
            } else {
                0.05 + 0.07 * (rd / 0.5).min(1.0)
            };
            self.input_energy_ema += ema_alpha * (input_energy - self.input_energy_ema);
            self.gain_call_count = self.gain_call_count.saturating_add(1);

            // Track high-frequency input energy using a one-pole high-pass.
            let hp_coeff = (2.0 * std::f64::consts::PI * 2000.0
                / self.params.sample_rate.max(1) as f64)
                .min(0.5);
            let mut hp_state = self.hf_input_hp_state;
            let mut hf_energy = 0.0f64;
            for &s in &self.channel_input_buffers[0] {
                hp_state += hp_coeff * (s as f64 - hp_state);
                let hp = s as f64 - hp_state;
                hf_energy += hp * hp;
            }
            self.hf_input_hp_state = hp_state;
            hf_energy /= self.channel_input_buffers[0].len().max(1) as f64;
            // HF energy uses faster EMA for quicker convergence since spectral
            // shape changes are often rapid (transient onsets).
            let hf_alpha = (ema_alpha * 1.5).min(0.20);
            self.input_hf_energy_ema += hf_alpha * (hf_energy - self.input_hf_energy_ema);

            // Track mid-band input energy (500-2000 Hz) via bandpass.
            let mid_hp_c = (2.0 * std::f64::consts::PI * 500.0
                / self.params.sample_rate.max(1) as f64)
                .min(0.5);
            let mid_lp_c = (2.0 * std::f64::consts::PI * 2000.0
                / self.params.sample_rate.max(1) as f64)
                .min(0.5);
            let mut m_hp = self.mid_input_bp_state[0];
            let mut m_lp = self.mid_input_bp_state[1];
            let mut mid_e = 0.0f64;
            for &s in &self.channel_input_buffers[0] {
                m_hp += mid_hp_c * (s as f64 - m_hp);
                let hp_s = s as f64 - m_hp;
                m_lp += mid_lp_c * (hp_s - m_lp);
                mid_e += m_lp * m_lp;
            }
            self.mid_input_bp_state = [m_hp, m_lp];
            mid_e /= self.channel_input_buffers[0].len().max(1) as f64;
            self.input_mid_energy_ema += hf_alpha * (mid_e - self.input_mid_energy_ema);
        }

        let min_output_len = self.process_channels(num_channels)?;
        let consumed_frames = self.consume_processed_input(total_frames, num_channels);
        self.input_frames_consumed_total = self
            .input_frames_consumed_total
            .saturating_add(consumed_frames);

        if min_output_len > 0 {
            self.decode_output_mid_side(num_channels, min_output_len);

            // Track output energy and compute gain compensation.
            {
                let output_energy = self.channel_output_buffers[0][..min_output_len]
                    .iter()
                    .map(|&s| (s as f64) * (s as f64))
                    .sum::<f64>()
                    / min_output_len.max(1) as f64;
                let ema_alpha_out = if self.gain_call_count < 5 {
                    0.15
                } else {
                    let rd = (self.current_ratio - 1.0).abs();
                    0.05 + 0.07 * (rd / 0.5).min(1.0)
                };
                self.output_energy_ema += ema_alpha_out * (output_energy - self.output_energy_ema);

                // Track high-frequency output energy.
                let hp_coeff_out = (2.0 * std::f64::consts::PI * 2000.0
                    / self.params.sample_rate.max(1) as f64)
                    .min(0.5);
                let mut hp_state_out = self.hf_output_hp_state;
                let mut hf_energy_out = 0.0f64;
                for &s in &self.channel_output_buffers[0][..min_output_len] {
                    hp_state_out += hp_coeff_out * (s as f64 - hp_state_out);
                    let hp = s as f64 - hp_state_out;
                    hf_energy_out += hp * hp;
                }
                self.hf_output_hp_state = hp_state_out;
                hf_energy_out /= min_output_len.max(1) as f64;
                let hf_alpha_out = (ema_alpha_out * 1.5).min(0.20);
                self.output_hf_energy_ema +=
                    hf_alpha_out * (hf_energy_out - self.output_hf_energy_ema);

                // Track mid-band output energy (500-2000 Hz).
                let mid_hp_c2 = (2.0 * std::f64::consts::PI * 500.0
                    / self.params.sample_rate.max(1) as f64)
                    .min(0.5);
                let mid_lp_c2 = (2.0 * std::f64::consts::PI * 2000.0
                    / self.params.sample_rate.max(1) as f64)
                    .min(0.5);
                let mut m_hp2 = self.mid_output_bp_state[0];
                let mut m_lp2 = self.mid_output_bp_state[1];
                let mut mid_e2 = 0.0f64;
                for &s in &self.channel_output_buffers[0][..min_output_len] {
                    m_hp2 += mid_hp_c2 * (s as f64 - m_hp2);
                    let hp_s = s as f64 - m_hp2;
                    m_lp2 += mid_lp_c2 * (hp_s - m_lp2);
                    mid_e2 += m_lp2 * m_lp2;
                }
                self.mid_output_bp_state = [m_hp2, m_lp2];
                mid_e2 /= min_output_len.max(1) as f64;
                self.output_mid_energy_ema += hf_alpha_out * (mid_e2 - self.output_mid_energy_ema);

                // Compute global gain to match input energy.
                const GAIN_SMOOTH: f64 = 0.30;
                if self.output_energy_ema > 1e-12 && self.input_energy_ema > 1e-12 {
                    let target_gain = (self.input_energy_ema / self.output_energy_ema)
                        .sqrt()
                        .min(3.0);
                    self.energy_gain += GAIN_SMOOTH * (target_gain - self.energy_gain);
                }

                // Apply gain compensation to output buffers.
                // At extreme stretch ratios, the PV naturally loses high-frequency
                // energy due to phase modifications. Apply a ratio-dependent
                // high-shelf boost to counteract centroid shift.
                let ratio_distance = (self.current_ratio - 1.0).abs();
                // Two-tier shelf: base correction for inherent PV spectral
                // tilt (always active when gain compensation is active), plus
                // ratio-dependent boost for centroid shift at extreme ratios.
                // Scale base shelf proportionally to energy gain: more gain
                // means more PV energy loss, which correlates with more spectral
                // tilt. At low gain (harmonic near-unity), shelf is minimal.
                let base_shelf = if self.energy_gain > 1.02 {
                    // Two-region gain_factor: dead zone below 1.06 (harmonic
                    // barely needs shelf), then standard scaling above.
                    let raw_gf = ((self.energy_gain - 1.02) / 0.48).clamp(0.0, 1.0);
                    let gain_factor = if self.energy_gain < 1.06 {
                        raw_gf * 0.5
                    } else {
                        raw_gf
                    };
                    let ratio_scale = (ratio_distance / 0.3).clamp(0.25, 1.0);
                    (1.0 + 1.40 * gain_factor * ratio_scale) as f32
                } else {
                    1.0f32
                };
                // HF-energy-driven shelf: directly measured HF loss drives
                // additional correction. Only active when total energy_gain is
                // low (< 1.10) — high energy_gain means broadband loss from
                // transient smearing where shelf hurts batch_sim.
                let hf_shelf = if self.output_hf_energy_ema > 1e-12
                    && self.input_hf_energy_ema > 1e-12
                    && self.gain_call_count > 3
                    && self.energy_gain < 1.20
                {
                    let hf_ratio = (self.input_hf_energy_ema / self.output_hf_energy_ema).sqrt();
                    if hf_ratio > 1.08 {
                        ((hf_ratio - 1.0) * 0.8 + 1.0).min(1.6) as f32
                    } else {
                        1.0f32
                    }
                } else {
                    1.0f32
                };
                let ratio_shelf = if ratio_distance > 0.4 {
                    let t = ((ratio_distance - 0.4) / 0.6).min(1.0);
                    let fixed = (1.0 + 0.80 * t * t) as f32;
                    fixed.max(hf_shelf)
                } else {
                    hf_shelf
                };
                let shelf_amount = base_shelf * ratio_shelf;
                let use_shelf = shelf_amount > 1.001;

                if (self.energy_gain - 1.0).abs() > 0.01 || use_shelf {
                    let gain = self.energy_gain as f32;
                    if use_shelf {
                        // Two-pole high-shelf via cascaded one-pole filters
                        // for steeper transition and more targeted boost.
                        let lp_coeff = (2.0 * std::f64::consts::PI * 2000.0
                            / self.params.sample_rate.max(1) as f64)
                            .min(0.5) as f32;
                        // Use sqrt of shelf for each stage (cascaded = product)
                        let stage_shelf = shelf_amount.sqrt();
                        for ch in 0..num_channels {
                            let mut lp1 = 0.0f32;
                            let mut lp2 = 0.0f32;
                            for s in self.channel_output_buffers[ch][..min_output_len].iter_mut() {
                                // First stage
                                lp1 += lp_coeff * (*s - lp1);
                                let hp1 = *s - lp1;
                                let mid = lp1 + hp1 * stage_shelf;
                                // Second stage
                                lp2 += lp_coeff * (mid - lp2);
                                let hp2 = mid - lp2;
                                *s = (lp2 + hp2 * stage_shelf) * gain;
                            }
                        }
                    } else {
                        for ch in 0..num_channels {
                            for s in self.channel_output_buffers[ch][..min_output_len].iter_mut() {
                                *s *= gain;
                            }
                        }
                    }
                }
            }

            // Time-domain blend: mix a small fraction of linearly-resampled
            // input into the PV output. Linear resampling preserves transient
            // shape (at the cost of aliasing), partially restoring kick impact.
            // Only active at extreme ratios where transient smearing is worst.
            // Blend amount is modulated by per-bin spectral flux: transient
            // frames get more blend (up to 8%) to preserve attack shape, while
            // steady-state frames get less (down to 2%) to let the PV shine.
            // Mid-band spectral correction: when the 500-2000 Hz range loses
            // energy (measured via bandpass tracking), apply a gentle additive
            // boost to the mid-band content. This addresses centroid drop in
            // the frequency range that high-shelf at 2kHz misses.
            if self.output_mid_energy_ema > 1e-12
                && self.input_mid_energy_ema > 1e-12
                && self.gain_call_count > 5
                && self.energy_gain < 1.20
                && min_output_len > 0
            {
                let mid_ratio = (self.input_mid_energy_ema / self.output_mid_energy_ema).sqrt();
                if mid_ratio > 1.10 {
                    // Extract mid-band and add a fraction back for gentle boost.
                    // Amount proportional to measured loss, clamped to prevent
                    // over-correction.
                    let boost = ((mid_ratio - 1.0) * 0.25).min(0.12) as f32;
                    let bp_hp = (2.0 * std::f64::consts::PI * 500.0
                        / self.params.sample_rate.max(1) as f64)
                        .min(0.5) as f32;
                    let bp_lp = (2.0 * std::f64::consts::PI * 2000.0
                        / self.params.sample_rate.max(1) as f64)
                        .min(0.5) as f32;
                    for ch in 0..num_channels {
                        let mut hp_st = 0.0f32;
                        let mut lp_st = 0.0f32;
                        for s in self.channel_output_buffers[ch][..min_output_len].iter_mut() {
                            hp_st += bp_hp * (*s - hp_st);
                            let hp_out = *s - hp_st;
                            lp_st += bp_lp * (hp_out - lp_st);
                            *s += lp_st * boost;
                        }
                    }
                }
            }

            if (self.current_ratio - 1.0).abs() > 0.5 {
                let base_blend = 0.045f32;
                let mut flux_factor = self.compute_flux_blend_factor();
                let input_rms = (self.channel_input_buffers[0]
                    .iter()
                    .map(|&s| s * s)
                    .sum::<f32>()
                    / self.channel_input_buffers[0].len().max(1) as f32)
                    .sqrt();
                let prev_input_rms = self.prev_blend_input_rms;
                let onset_rise = (input_rms - prev_input_rms).max(0.0);
                self.prev_blend_input_rms = input_rms;
                let onset_boost = if prev_input_rms > 1e-6 {
                    (onset_rise / prev_input_rms.max(1e-6)).min(1.0)
                } else {
                    0.0
                };
                flux_factor *= 1.0 + 0.35 * onset_boost;
                let blend = (base_blend * flux_factor).clamp(0.01, 0.10);
                let ratio = self.current_ratio;
                for ch in 0..num_channels {
                    let in_buf = &self.channel_input_buffers[ch];
                    let out_buf = &mut self.channel_output_buffers[ch][..min_output_len];
                    let in_len = in_buf.len();
                    if in_len < 2 {
                        continue;
                    }
                    for (i, sample) in out_buf.iter_mut().enumerate().take(min_output_len) {
                        let in_pos = i as f64 / ratio;
                        let idx = in_pos as usize;
                        if idx + 2 >= in_len || idx == 0 {
                            // Fall back to linear at boundaries
                            if idx + 1 < in_len {
                                let frac = (in_pos - idx as f64) as f32;
                                let interp = in_buf[idx] * (1.0 - frac) + in_buf[idx + 1] * frac;
                                *sample = *sample * (1.0 - blend) + interp * blend;
                            }
                            continue;
                        }
                        // Cubic (Catmull-Rom) interpolation for less aliasing
                        let t = (in_pos - idx as f64) as f32;
                        let p0 = in_buf[idx - 1];
                        let p1 = in_buf[idx];
                        let p2 = in_buf[idx + 1];
                        let p3 = in_buf[idx + 2];
                        let interp = p1
                            + 0.5
                                * t
                                * (p2 - p0
                                    + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                                        + t * (-p0 + 3.0 * p1 - 3.0 * p2 + p3)));
                        *sample = *sample * (1.0 - blend) + interp * blend;
                    }
                }
            }

            // Streaming WSOLA-PV hybrid overlay: when a strong transient is
            // detected at extreme ratios, run WSOLA on the input and crossfade
            // its output over the PV output. WSOLA preserves waveform shape
            // for kicks/snares while the PV handles tonal content.
            let ratio_distance = (self.current_ratio - 1.0).abs();
            if ratio_distance > 0.01 {
                // Arm a new WSOLA overlay on strong onsets.
                let flux_factor = self.compute_flux_blend_factor();
                let input_rms = (self.channel_input_buffers[0]
                    .iter()
                    .map(|&s| s * s)
                    .sum::<f32>()
                    / self.channel_input_buffers[0].len().max(1) as f32)
                    .sqrt();
                let prev_rms = self.prev_blend_input_rms;
                let onset_rise = (input_rms - prev_rms).max(0.0);
                let onset_strength = if prev_rms > 1e-6 {
                    (onset_rise / prev_rms.max(1e-6)).min(1.0)
                } else {
                    0.0
                };

                // PV-side flux spikes whenever the stretch ratio steps, so
                // during an in-flight ratio/pitch slew it is a false onset
                // signal: arming the overlay on it splices phase-arbitrary
                // WSOLA audio over steady tonal content. While modulating,
                // require an input-domain onset (RMS rise), which is
                // ratio-independent.
                let modulating = self.modulation_hold_overlap_windows() > 0;
                let should_arm_wsola = self.wsola_overlay_remaining == 0
                    && ((flux_factor > 0.8 && !modulating) || onset_strength > 0.10);

                if should_arm_wsola {
                    // Run WSOLA on channel input buffers and store results.
                    let ratio = self.current_ratio;
                    let mut wsola_min_len = usize::MAX;
                    for ch in 0..num_channels {
                        let in_buf = &self.channel_input_buffers[ch];
                        if in_buf.len() < self.wsola_instances[ch].segment_size() {
                            wsola_min_len = 0;
                            break;
                        }
                        self.wsola_instances[ch].set_stretch_ratio(ratio);
                        self.wsola_output_buffers[ch].clear();
                        if self.wsola_instances[ch]
                            .process_into(in_buf, &mut self.wsola_output_buffers[ch])
                            .is_ok()
                        {
                            wsola_min_len = wsola_min_len.min(self.wsola_output_buffers[ch].len());
                        } else {
                            wsola_min_len = 0;
                            break;
                        }
                    }

                    if wsola_min_len > 0 {
                        // At extreme ratios, use full WSOLA output for cross-chunk
                        // transient continuity. At normal ratios, limit to PV output.
                        let overlay_len = if ratio_distance > 0.8 {
                            wsola_min_len
                        } else {
                            wsola_min_len.min(min_output_len)
                        };
                        for ch in 0..num_channels {
                            self.wsola_overlay_buffers[ch].clear();
                            self.wsola_overlay_buffers[ch]
                                .extend_from_slice(&self.wsola_output_buffers[ch][..overlay_len]);
                        }
                        self.wsola_overlay_remaining = overlay_len;
                        self.wsola_overlay_total = overlay_len;
                        self.wsola_overlay_pos = 0;
                    }

                    // Pre-normalize WSOLA overlay energy to match input energy.
                    // The WSOLA output may have slight energy loss from segment
                    // crossfade dips. Compare actual WSOLA energy to input energy
                    // and apply per-overlay correction for accurate energy matching.
                    if self.wsola_overlay_pos == 0 && !self.wsola_overlay_buffers[0].is_empty() {
                        let in_buf = &self.channel_input_buffers[0];
                        let ws_buf = &self.wsola_overlay_buffers[0];
                        let in_rms_sq =
                            in_buf.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>()
                                / in_buf.len().max(1) as f64;
                        // Compare WSOLA energy to expected output energy (input * ratio_factor)
                        let ws_rms_sq =
                            ws_buf.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>()
                                / ws_buf.len().max(1) as f64;
                        if ws_rms_sq > 1e-12 && in_rms_sq > 1e-12 {
                            let correction = (in_rms_sq / ws_rms_sq).sqrt().min(3.0) as f32;
                            // Only boost if WSOLA is quieter than input (correction > 1).
                            // If WSOLA is already louder, leave it to avoid over-amplification
                            // when combined with the PV gain applied in the overlay loop.
                            if correction > 1.0 {
                                for ch_buf in &mut self.wsola_overlay_buffers {
                                    for s in ch_buf.iter_mut() {
                                        *s *= correction;
                                    }
                                }
                            }
                        }
                    }
                }

                // Apply the same high-shelf filter to WSOLA overlay for spectral
                // consistency with the shelf-boosted PV output.
                let wsola_shelf = {
                    let rd = (self.current_ratio - 1.0).abs();
                    let bs = if self.energy_gain > 1.02 {
                        let raw_gf = ((self.energy_gain - 1.02) / 0.48).clamp(0.0, 1.0);
                        let gf = if self.energy_gain < 1.06 {
                            raw_gf * 0.5
                        } else {
                            raw_gf
                        };
                        let rs = (rd / 0.3).clamp(0.25, 1.0);
                        (1.0 + 1.40 * gf * rs) as f32
                    } else {
                        1.0f32
                    };
                    let rsh = if rd > 0.4 {
                        let t = ((rd - 0.4) / 0.6).min(1.0);
                        (1.0 + 0.80 * t * t) as f32
                    } else {
                        1.0f32
                    };
                    bs * rsh
                };
                if self.wsola_overlay_pos == 0
                    && !self.wsola_overlay_buffers[0].is_empty()
                    && wsola_shelf > 1.001
                {
                    let lp_coeff = (2.0 * std::f64::consts::PI * 2000.0
                        / self.params.sample_rate.max(1) as f64)
                        .min(0.5) as f32;
                    let stage_shelf = wsola_shelf.sqrt();
                    for ch_buf in &mut self.wsola_overlay_buffers {
                        let mut lp1 = 0.0f32;
                        let mut lp2 = 0.0f32;
                        for s in ch_buf.iter_mut() {
                            lp1 += lp_coeff * (*s - lp1);
                            let hp1 = *s - lp1;
                            let mid = lp1 + hp1 * stage_shelf;
                            lp2 += lp_coeff * (mid - lp2);
                            let hp2 = mid - lp2;
                            *s = lp2 + hp2 * stage_shelf;
                        }
                    }
                }

                // Apply active WSOLA overlay: crossfade WSOLA over PV output.
                if self.wsola_overlay_remaining > 0 {
                    let total = self.wsola_overlay_total.max(1);
                    let apply_len = self.wsola_overlay_remaining.min(min_output_len);
                    for ch in 0..num_channels {
                        let overlay = &self.wsola_overlay_buffers[ch];
                        let out = &mut self.channel_output_buffers[ch][..min_output_len];
                        for (i, out_sample) in out[..apply_len].iter_mut().enumerate() {
                            let pos = self.wsola_overlay_pos + i;
                            if pos >= overlay.len() {
                                break;
                            }
                            // Crossfade: start with WSOLA dominant, transition to PV.
                            // At extreme ratios, extend WSOLA dominance to preserve
                            // more of the transient waveform shape.
                            let progress = pos as f32 / total as f32;
                            let (peak_weight, attack_end) = if ratio_distance > 0.8 {
                                (1.0f32, 1.0f32) // extreme: pure WSOLA for entire overlay
                            } else {
                                (0.90f32, 0.25f32) // normal
                            };
                            let wsola_weight = if progress < attack_end {
                                peak_weight
                            } else {
                                let t = (progress - attack_end) / (1.0 - attack_end);
                                peak_weight * (1.0 - t * t)
                            };
                            // Ramp the overlay in over a few ms: WSOLA output
                            // has arbitrary phase relative to the PV stream,
                            // so switching to it at full weight in one sample
                            // is an audible click at the splice.
                            let fade_in = ((pos + 1) as f32
                                / WSOLA_OVERLAY_FADE_IN_SAMPLES.min(total / 4).max(1) as f32)
                                .min(1.0);
                            let wsola_weight = wsola_weight * fade_in;
                            // Apply PV gain on top of per-overlay normalization
                            // for spectral consistency with shelf-boosted PV output.
                            let wsola_sample = overlay[pos] * self.energy_gain as f32;
                            *out_sample =
                                *out_sample * (1.0 - wsola_weight) + wsola_sample * wsola_weight;
                        }
                    }
                    self.wsola_overlay_pos += apply_len;
                    self.wsola_overlay_remaining =
                        self.wsola_overlay_remaining.saturating_sub(apply_len);
                }
            }

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

    /// Per-channel frames a render pass over `total_frames` will consume.
    ///
    /// Shared between `consume_processed_input` and artifact reset span math
    /// so the two cannot drift.
    #[inline]
    fn frames_consumed_for(&self, total_frames: usize) -> usize {
        let hop = self.params.hop_size;
        if hop == 0 || total_frames < self.params.fft_size {
            return 0;
        }
        ((total_frames - self.params.fft_size) / hop + 1) * hop
    }

    fn consume_processed_input(&mut self, total_frames: usize, num_channels: usize) -> usize {
        let frames_consumed = self.frames_consumed_for(total_frames);
        let samples_consumed = frames_consumed * num_channels;
        if samples_consumed > 0 {
            self.input_ring.discard(samples_consumed);
        }
        frames_consumed
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

    /// Processing ratio at the current control targets.
    ///
    /// The buffering gate is computed from this rather than the glided
    /// [`Self::processing_ratio`]: targets change only at control calls, so
    /// the gate cannot flip while a 50 ms glide passes through the ratio
    /// band boundary.
    #[inline]
    fn target_processing_ratio(&self) -> f64 {
        self.target_ratio * self.target_pitch_scale
    }

    /// Returns how many FFT-window footprints the in-flight ratio/pitch slew
    /// will span before settling to `RATIO_SNAP_THRESHOLD`, capped at
    /// `MODULATION_HOLD_MAX_OVERLAP_WINDOWS`. Zero means steady state.
    ///
    /// "Windows" is the unit the transient scheduler's modulation-hold
    /// machinery scales its cooldowns and trigger thresholds by: how many
    /// overlapping analysis footprints the seam disturbance persists. The
    /// control EMA reaches the snap threshold after `tau * ln(delta/snap)`
    /// samples, which this converts to FFT footprints.
    fn modulation_hold_overlap_windows(&self) -> usize {
        let delta = (self.target_processing_ratio() - self.processing_ratio()).abs();
        if delta < MODULATION_HOLD_MIN_RATIO_DELTA {
            return 0;
        }
        let tau = self.params.sample_rate.max(1) as f64 * RATIO_SMOOTHING_TIME_SECS;
        let settle_samples = tau * (delta / RATIO_SNAP_THRESHOLD).ln().max(0.0);
        ((settle_samples / self.params.fft_size.max(1) as f64).ceil() as usize)
            .clamp(1, self.modulation_hold_max_overlap_windows())
    }

    /// Time-based cap on the modulation-hold window budget: the same
    /// absolute hold time regardless of FFT size (4 windows at the reference
    /// 4096-FFT / 44.1 kHz configuration, 16 at 1024).
    #[inline]
    fn modulation_hold_max_overlap_windows(&self) -> usize {
        let window_secs =
            self.params.fft_size.max(1) as f64 / self.params.sample_rate.max(1) as f64;
        ((MODULATION_HOLD_MAX_SECS / window_secs).ceil() as usize)
            .clamp(1, MODULATION_HOLD_MAX_WINDOWS_CEILING)
    }

    fn reset_pitch_resamplers(&mut self) {
        for resampler in &mut self.pitch_resamplers {
            resampler.reset();
        }
        for resampler in &mut self.sinc_pitch_resamplers {
            resampler.reset();
        }
        for buf in &mut self.pitch_output_buffers {
            buf.clear();
        }
        self.pitch_resampler_engaged = false;
    }

    fn emit_channel_output_to_pending(
        &mut self,
        min_output_len: usize,
        num_channels: usize,
    ) -> Result<(), StretchError> {
        if min_output_len == 0 {
            return Ok(());
        }

        // The resampler stays engaged once pitch has been used this stream:
        // at unity it degenerates to a bit-clean passthrough, which avoids
        // splicing its held lookahead out of the stream when a pitch sweep
        // returns to 1.0.
        let pitch_active = self.pitch_resampler_engaged
            || (self.pitch_scale - 1.0).abs() >= RATIO_SNAP_THRESHOLD
            || (self.target_pitch_scale - 1.0).abs() >= RATIO_SNAP_THRESHOLD;
        if !pitch_active {
            return self.interleave_to_pending(min_output_len, num_channels);
        }
        self.pitch_resampler_engaged = true;

        let mut pitch_min_output_len = usize::MAX;
        for ch in 0..num_channels {
            if self.channel_output_buffers[ch].len() < min_output_len {
                return Err(StretchError::InvalidState(
                    "channel output shorter than requested interleave length",
                ));
            }

            match self.pitch_quality {
                StreamPitchQuality::Sinc => {
                    // The sinc resampler consumes the source at `pitch_scale`
                    // samples per output sample and ramps toward it from the
                    // previous block's step internally.
                    self.sinc_pitch_resamplers[ch].process_into(
                        &self.channel_output_buffers[ch][..min_output_len],
                        self.pitch_scale,
                        &mut self.pitch_output_buffers[ch],
                    )?;
                }
                StreamPitchQuality::Linear => {
                    self.pitch_resamplers[ch].process_into(
                        &self.channel_output_buffers[ch][..min_output_len],
                        1.0 / self.pitch_scale,
                        &mut self.pitch_output_buffers[ch],
                    )?;
                }
            }
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
        if !self.pitch_resampler_engaged {
            return Ok(());
        }

        let mut min_output_len = usize::MAX;
        for ch in 0..num_channels {
            match self.pitch_quality {
                StreamPitchQuality::Sinc => {
                    self.sinc_pitch_resamplers[ch]
                        .flush_into(self.pitch_scale, &mut self.pitch_output_buffers[ch])?;
                }
                StreamPitchQuality::Linear => {
                    self.pitch_resamplers[ch]
                        .flush_into(1.0 / self.pitch_scale, &mut self.pitch_output_buffers[ch])?;
                }
            }
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
            // Declick ramp on the first output after a warm-start seek: the
            // caller cut the old stream mid-waveform (that is what a cue
            // jump is), so the resumed stream fades in over a few ms.
            if self.warm_start_fade_remaining > 0 {
                let total = WARM_START_FADE_SAMPLES as f32;
                for sample in chunk[..n].iter_mut() {
                    if self.warm_start_fade_remaining == 0 {
                        break;
                    }
                    let progress = 1.0 - self.warm_start_fade_remaining as f32 / total;
                    *sample *= progress;
                    self.warm_start_fade_remaining -= 1;
                }
            }
            output.extend_from_slice(&chunk[..n]);
            written += n;
        }

        self.total_output_emitted_samples += written;

        Ok(written)
    }

    fn append_hybrid_input(&mut self, num_channels: usize) -> Result<(), StretchError> {
        // The frames being appended start at the input ring's current
        // absolute position (the hybrid branch consumes the whole ring
        // right after this call, so appends and consumption stay 1:1).
        let append_base = self.ring_start_abs();
        let mut first_ch_pushed = 0;
        for ch in 0..num_channels {
            let input = &self.channel_input_buffers[ch];
            let rb = &mut self.hybrid_state.rolling_inputs[ch];
            if ch == 0 {
                if rb.is_empty() && !self.hybrid_state.window_base_valid {
                    self.hybrid_state.window_base_abs = append_base;
                    self.hybrid_state.window_base_valid = true;
                } else if input.len() > rb.available() {
                    self.hybrid_state.window_base_abs = self
                        .hybrid_state
                        .window_base_abs
                        .saturating_add(input.len() - rb.available());
                }
            }
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

        // Phase 2: Build shared segmentation anchors for all channels.
        //
        // With a usable pre-analysis artifact, the precomputed onset/beat
        // anchors (absolute source frames) are mapped into the rolling
        // window — no online detection, and the mono path below skips its
        // per-render full adaptive analysis. Without an artifact, stereo
        // M/S falls back to detecting shared transients from the mid
        // channel so both channels use identical segmentation (preventing
        // phase misalignment on L/R decode, matching stretch_mid_side()).
        let rolling_len = self.hybrid_state.rolling_scratch[0].len();
        let shared_onsets: Option<(Vec<usize>, Vec<f32>)> =
            if self.artifact_active && self.hybrid_state.window_base_valid && rolling_len > 0 {
                let base = self.hybrid_state.window_base_abs;
                let (positions, strengths) = &self.artifact_merged_anchors;
                let start = positions.partition_point(|&p| p < base);
                let end = positions.partition_point(|&p| p < base + rolling_len);
                let window_onsets: Vec<usize> =
                    positions[start..end].iter().map(|&p| p - base).collect();
                let window_strengths: Vec<f32> = strengths[start..end].to_vec();
                Some((window_onsets, window_strengths))
            } else if num_channels == 2
                && self.params.stereo_mode == StereoMode::MidSide
                && rolling_len > 0
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

    // TODO(stage-1 follow-up): tails drained here (and by the pitch
    // resampler flush) skip the energy-gain/shelf corrections that normal
    // chunks receive in `process_available_to_pending`, leaving a small gain
    // seam at the flush boundary at far-from-unity ratios.
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
        // Route through the pitch resampler like every other emission: with
        // pitch engaged, the held PV overlap was rendered at the pre-resample
        // rate, and splicing it in raw is a full-scale pitch discontinuity.
        self.emit_channel_output_to_pending(min_output_len, num_channels)
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
    ///
    /// Equivalent to [`Self::try_from_tempo_with_profile`] with
    /// [`StreamProfile::Live`] (~35 ms at 44.1 kHz). Unlike earlier
    /// versions, this now carries the full `DjBeatmatch` tuning bundle
    /// instead of silently dropping it.
    pub fn try_from_tempo_low_latency(
        source_bpm: f64,
        target_bpm: f64,
        sample_rate: u32,
        channels: u32,
    ) -> Result<Self, StretchError> {
        Self::try_from_tempo_with_profile(
            source_bpm,
            target_bpm,
            sample_rate,
            channels,
            StreamProfile::Live,
        )
    }

    /// Creates a BPM-matching stream processor with an explicit streaming
    /// latency/quality profile (see [`StreamProfile`]).
    pub fn try_from_tempo_with_profile(
        source_bpm: f64,
        target_bpm: f64,
        sample_rate: u32,
        channels: u32,
        profile: StreamProfile,
    ) -> Result<Self, StretchError> {
        let base = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(channels)
            .with_stream_profile(profile);
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

    /// Returns cumulative transient-reset telemetry for the current stream.
    pub fn transient_reset_stats(&self) -> TransientResetStats {
        let TransientSchedulerStats {
            events_detected_total,
            mut reset_band_counts_total,
        } = self.transient_scheduler.stats();
        for (band, count) in reset_band_counts_total.iter_mut().enumerate() {
            *count = count.saturating_add(self.artifact_reset_band_counts_total[band]);
        }
        TransientResetStats {
            events_detected_total,
            artifact_events_scheduled_total: self.artifact_events_scheduled_total,
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
    ///
    /// The applied pitch glides toward the new value over ~50 ms so nudges
    /// and sweeps stay click-free. With the default
    /// [`StreamPitchQuality::Sinc`] resampler, engaging pitch adds a small
    /// amount of buffering (8–[`crate::core::resample::STREAM_SINC_MAX_HALF_TAPS`]
    /// samples of kernel lookahead) on top of [`Self::latency_samples`]; the
    /// resampled output itself stays sample-aligned.
    pub fn set_pitch_scale(&mut self, scale: f64) -> Result<(), StretchError> {
        let scale = validate_positive_finite_ratio(scale, "pitch scale")?;
        if (scale - self.target_pitch_scale).abs() > RATIO_SNAP_THRESHOLD {
            self.hybrid_pending_rebase = true;
        }
        // The applied pitch glides toward the target with the same time
        // constant as stretch-ratio changes (see
        // `interpolate_ratio_for_frames`), so pitch nudges and sweeps stay
        // click-free instead of splicing a resampler discontinuity.
        self.target_pitch_scale = scale;
        Ok(())
    }

    /// Returns the current realtime pitch-scale control value.
    ///
    /// This is the most recently set target; the internally applied pitch
    /// glides toward it over the smoothing window.
    #[allow(clippy::misnamed_getters)]
    pub fn pitch_scale(&self) -> f64 {
        self.target_pitch_scale
    }

    /// Selects the realtime pitch resampler quality.
    ///
    /// Defaults to [`StreamPitchQuality::Sinc`]. Switching mid-stream flushes
    /// the active resampler's held lookahead into the output (a short splice
    /// may be audible), so prefer selecting the quality before streaming.
    pub fn set_pitch_resampler_quality(
        &mut self,
        quality: StreamPitchQuality,
    ) -> Result<(), StretchError> {
        if quality == self.pitch_quality {
            return Ok(());
        }
        let num_channels = self.params.channels.count();
        self.flush_pitch_resampler_to_pending(num_channels)?;
        self.reset_pitch_resamplers();
        self.pitch_quality = quality;
        Ok(())
    }

    /// Returns the selected realtime pitch resampler quality.
    pub fn pitch_resampler_quality(&self) -> StreamPitchQuality {
        self.pitch_quality
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

    /// Returns the current effective input-buffering latency in frames.
    ///
    /// Honest reporting: this is the real gate before output emerges — the
    /// `fft * 3/2` floor, widened to `fft * 2` when the *target* processing
    /// ratio (stretch ratio × pitch scale) sits outside `[0.9, 1.1]`, plus
    /// the engaged sinc pitch resampler's kernel lookahead. Equals
    /// `fft * 3/2` at construction with an in-band ratio and no pitch.
    ///
    /// See [`Self::latency_report`] for the breakdown, including the ~50 ms
    /// control-glide time constant that governs how fast ratio/pitch changes
    /// become audible.
    pub fn latency_samples(&self) -> usize {
        self.latency_report().total_frames
    }

    /// Returns the current effective input-buffering latency in seconds.
    pub fn latency_secs(&self) -> f64 {
        self.latency_samples() as f64 / self.params.sample_rate.max(1) as f64
    }

    /// Returns a breakdown of the processor's current effective latency.
    ///
    /// Pure arithmetic over current control targets — allocation-free and
    /// safe to call from the audio thread.
    pub fn latency_report(&self) -> StreamLatencyReport {
        let base_gate_frames = min_latency_frames(self.params.fft_size);
        let effective_gate_frames =
            effective_min_frames(self.params.fft_size, self.target_processing_ratio());

        // Pitch lookahead applies once the resampler is (or is about to be)
        // in the signal path: engaged resamplers stay engaged even at unity
        // pitch, and a non-unity target engages on the next process call.
        let pitch_active = self.pitch_resampler_engaged
            || (self.target_pitch_scale - 1.0).abs() > RATIO_SNAP_THRESHOLD
            || (self.pitch_scale - 1.0).abs() > RATIO_SNAP_THRESHOLD;
        let pitch_lookahead_samples = if !pitch_active {
            0
        } else {
            match self.pitch_quality {
                StreamPitchQuality::Sinc => self
                    .sinc_pitch_resamplers
                    .first()
                    .map(|r| r.current_half_span())
                    .unwrap_or(0),
                StreamPitchQuality::Linear => 1,
            }
        };

        StreamLatencyReport {
            base_gate_frames,
            effective_gate_frames,
            pitch_lookahead_samples,
            control_smoothing_secs: RATIO_SMOOTHING_TIME_SECS,
            sample_rate: self.params.sample_rate,
            total_frames: effective_gate_frames + pitch_lookahead_samples,
        }
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
        // The timeline returns to source frame 0; seek flows call
        // `set_source_position` again after reset.
        self.source_start_frames = 0;
        self.passthrough_frames_total = 0;
        self.artifact_events_scheduled_total = 0;
        self.artifact_reset_band_counts_total = [0; 4];
        self.warm_start_fade_remaining = 0;
        self.reposition_artifact_cursor();

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
        self.target_pitch_scale = 1.0;
        self.reset_pitch_resamplers();
        self.dsp_engaged = false;
        self.input_energy_ema = 0.0;
        self.output_energy_ema = 0.0;
        self.energy_gain = 1.0;
        self.gain_call_count = 0;
        self.prev_blend_input_rms = 0.0;
        self.wsola_overlay_remaining = 0;
        self.wsola_overlay_total = 0;
        self.wsola_overlay_pos = 0;
        for buf in &mut self.wsola_overlay_buffers {
            buf.clear();
        }
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

    /// Computes a blend multiplier (0.5×–2.0×) from the PV's per-frame flux.
    ///
    /// Transient frames (many bins rising) get more time-domain blend to
    /// preserve attack shape; steady-state frames get less to let the PV's
    /// tonal quality dominate.
    fn compute_flux_blend_factor(&self) -> f32 {
        // Use channel 0's vocoder flux (mid channel for stereo M/S).
        let flux = match self.vocoders.first().and_then(|v| v.last_frame_flux()) {
            Some(f) => f,
            None => return 1.0,
        };

        let num_bins = self.params.fft_size / 2 + 1;
        if num_bins == 0 {
            return 1.0;
        }

        // Fraction of bins that are rising — indicator of impulsive onset.
        let rising_fraction = flux.total_bins_rising as f32 / num_bins as f32;
        // Fraction of bins with significant flux (4× mean threshold).
        let transient_fraction = flux.transient_bin_count as f32 / num_bins as f32;

        // Combine: high rising_fraction AND high transient_fraction = impulsive.
        // Pure vibrato has high rising_fraction but low transient_fraction.
        let impulsiveness = (rising_fraction * 2.0).min(1.0) * (transient_fraction * 8.0).min(1.0);

        // Map to 0.3–2.5 range: lower for steady state (PV quality),
        // higher for strong transients (preserve attack shape).
        0.3 + 2.2 * impulsiveness
    }

    fn apply_transient_scheduled_phase_reset(&mut self, total_frames: usize, num_channels: usize) {
        if total_frames < self.params.fft_size {
            return;
        }

        // The scheduler analyzes mono or stereo snapshots; other channel
        // counts fall outside the deterministic reset path.
        if num_channels != 1 && num_channels != 2 {
            return;
        }

        let total_samples = total_frames.saturating_mul(num_channels);
        if total_samples == 0 || total_samples > self.interleaved_scratch.len() {
            return;
        }

        // While a ratio/pitch slew is in flight, engage the scheduler's
        // modulation-hold: low bands stay phase-locked (a low-band reset on
        // top of an in-flight seam compounds it) and triggers tighten in
        // proportion to how long the slew will persist.
        let modulation_overlap_windows = self.modulation_hold_overlap_windows();
        let suppress_low_bands = modulation_overlap_windows > 0;

        // `interleaved_scratch` still holds the raw input snapshot here (the
        // M/S encode only rewrites the per-channel buffers).
        let snapshot = &self.interleaved_scratch[..total_samples];
        let reset_mask = if num_channels == 1 {
            self.transient_scheduler.detect_mono_reset_mask(
                snapshot,
                self.input_frames_consumed_total,
                suppress_low_bands,
                modulation_overlap_windows,
            )
        } else {
            self.transient_scheduler.detect_stereo_reset_mask(
                snapshot,
                self.input_frames_consumed_total,
                suppress_low_bands,
                modulation_overlap_windows,
            )
        };
        let Some(reset_mask) = reset_mask else {
            return;
        };

        if num_channels == 2
            && self.params.stereo_mode == StereoMode::MidSide
            && self.vocoders.len() == 2
        {
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

    /// Schedules per-band phase resets from pre-analysis artifact onsets.
    ///
    /// Replaces the online [`TransientEventScheduler`] when a usable artifact
    /// is attached: the artifact's offline onsets are ground truth, so no
    /// detection heuristics run here. Allocation-free — a monotonic cursor
    /// over the onset slice plus band-mask accumulation.
    ///
    /// Reset granularity matches the online scheduler: one combined mask per
    /// render pass, applied at the current render rather than sample-timed.
    fn apply_artifact_scheduled_phase_reset(&mut self, total_frames: usize, num_channels: usize) {
        if total_frames < self.params.fft_size {
            return;
        }
        // Mirror the deterministic reset path's channel-count support.
        if num_channels != 1 && num_channels != 2 {
            return;
        }

        // The absolute source span this render pass will consume. Computed
        // before consumption via the same arithmetic as
        // `consume_processed_input`.
        let span_len = self.frames_consumed_for(total_frames);
        if span_len == 0 {
            return;
        }
        let span_start = self.ring_start_abs();
        let span_end = span_start.saturating_add(span_len);

        let Some(artifact) = self.params.pre_analysis.as_ref() else {
            return;
        };
        let onsets = &artifact.transient_onsets;
        let mut cursor = self.artifact_onset_cursor;
        let mut mask = [false; 4];
        let mut events = 0u64;
        while cursor < onsets.len() && onsets[cursor] < span_end {
            // Onsets behind the span (possible after underfull warmup
            // passes) are skipped silently.
            if onsets[cursor] >= span_start {
                let strength = artifact.strength_at(cursor);
                // Upper bands always re-lock on a transient (mirrors the
                // online scheduler's deterministic upper-band policy); low
                // and sub-bass resets are reserved for strong hits.
                mask[2] = true;
                mask[3] = true;
                if strength >= ARTIFACT_LOW_BAND_RESET_STRENGTH {
                    mask[1] = true;
                }
                if strength >= ARTIFACT_SUB_BASS_RESET_STRENGTH {
                    mask[0] = true;
                }
                events += 1;
            }
            cursor += 1;
        }
        self.artifact_onset_cursor = cursor;
        if events == 0 {
            return;
        }

        // While a ratio/pitch slew is in flight, keep low bands phase-locked
        // (same low-band suppression the online scheduler applies).
        if self.modulation_hold_overlap_windows() > 0 {
            mask[0] = false;
            mask[1] = false;
        }

        self.artifact_events_scheduled_total =
            self.artifact_events_scheduled_total.saturating_add(events);
        for (band, &selected) in mask.iter().enumerate() {
            if selected {
                self.artifact_reset_band_counts_total[band] =
                    self.artifact_reset_band_counts_total[band].saturating_add(1);
            }
        }

        if num_channels == 2
            && self.params.stereo_mode == StereoMode::MidSide
            && self.vocoders.len() == 2
        {
            let (mid_mask, side_mask) = stereo_channel_reset_masks(mask);
            self.vocoders[0].reset_phase_state_bands(mid_mask, self.params.sample_rate);
            if side_mask.iter().any(|&b| b) {
                self.vocoders[1].reset_phase_state_bands(side_mask, self.params.sample_rate);
            }
            return;
        }

        for vocoder in &mut self.vocoders {
            vocoder.reset_phase_state_bands(mask, self.params.sample_rate);
        }
    }

    /// Returns the BPM stored in the params, if any.
    pub fn bpm(&self) -> Option<f64> {
        self.params.bpm
    }

    /// Absolute source frame of the next input frame awaiting consumption.
    ///
    /// This is the timeline pre-analysis artifact positions are compared
    /// against: `source start + passthrough frames + DSP-consumed frames`.
    #[inline]
    fn ring_start_abs(&self) -> usize {
        self.source_start_frames
            .saturating_add(self.passthrough_frames_total)
            .saturating_add(self.input_frames_consumed_total)
    }

    /// Tells the processor the absolute source frame of the next pushed
    /// input frame.
    ///
    /// Call this on a fresh or freshly-[`reset`](Self::reset) processor,
    /// before any input, when streaming starts mid-file (seek/cue rebuild
    /// flows) so pre-analysis artifact positions stay aligned. `reset()`
    /// returns the timeline to source frame 0; call this again afterwards
    /// if needed.
    ///
    /// # Errors
    ///
    /// Returns [`StretchError::InvalidState`] once any input has been
    /// pushed or processed.
    pub fn set_source_position(&mut self, source_frame: usize) -> Result<(), StretchError> {
        if self.input_frames_consumed_total > 0
            || self.passthrough_frames_total > 0
            || !self.input_ring.is_empty()
            || self.dsp_engaged
        {
            return Err(StretchError::InvalidState(
                "set_source_position requires a fresh or freshly-reset processor",
            ));
        }
        self.source_start_frames = source_frame;
        self.reposition_artifact_cursor();
        Ok(())
    }

    /// Returns the absolute source frame of the next frame to be consumed.
    pub fn source_position(&self) -> usize {
        self.ring_start_abs()
    }

    /// Declares a gapless jump in the SOURCE timeline (a loop wrap or
    /// beat-jump) without touching any DSP state.
    ///
    /// Use this when the caller splices its input feed — e.g. after pushing
    /// audio up to a loop end, subsequent pushes come from the loop start.
    /// The processor keeps rendering continuously (output stays as seamless
    /// as the source splice itself); this call only re-anchors the absolute
    /// source position so pre-analysis artifact onsets keep firing at the
    /// right places. Frames already buffered from before the jump keep
    /// their old positions until consumed.
    ///
    /// For jumps where the output pipeline is flushed (cue jumps, scrub
    /// seeks), use [`Self::warm_start_seek`] instead.
    pub fn notify_source_jump(&mut self, next_frame: usize) {
        // Frames still buffered belong to pre-jump material; position the
        // timeline so the NEXT pushed frame maps to `next_frame`.
        let num_channels = self.params.channels.count().max(1);
        let in_flight = self.input_ring.len() / num_channels;
        let already_counted = self
            .passthrough_frames_total
            .saturating_add(self.input_frames_consumed_total)
            .saturating_add(in_flight);
        self.source_start_frames = next_frame.saturating_sub(already_counted);
        self.reposition_artifact_cursor();
    }

    /// Frames of preceding audio [`Self::warm_start_seek`] wants as preroll.
    ///
    /// Enough to clear the input-buffering gate plus one extra FFT window
    /// of convergence margin for the phase vocoder's overlap and the pitch
    /// resampler's history. Bounded, so rapid cue jumps stay cheap.
    pub fn warm_start_preroll_frames(&self) -> usize {
        effective_min_frames(self.params.fft_size, self.target_processing_ratio())
            .saturating_add(self.params.fft_size)
    }

    /// Re-primes the processor at a new source position from the audio
    /// immediately preceding it, so playback resumes converged: no cold
    /// input-buffering gate refill from silence, no phase-vocoder warm-up
    /// transient. This is the seek/cue/loop path — commercial decks re-prime
    /// from surrounding audio the same way.
    ///
    /// `target_frame` is the absolute source frame playback resumes from.
    /// `preroll` is interleaved audio ending exactly at the target (its
    /// last frame is source frame `target_frame - 1`). Pass
    /// [`Self::warm_start_preroll_frames`] frames; longer prerolls are
    /// trimmed from the front, shorter ones degrade gracefully toward a
    /// cold start (an empty preroll is equivalent to a state clear plus
    /// [`Self::set_source_position`]).
    ///
    /// Stretch-ratio and pitch control state — targets **and** in-flight
    /// glides — is preserved; there is no re-glide from unity. Allocation-
    /// free in the deterministic engine (the legacy hybrid re-render engine
    /// rebuilds its rolling state, which allocates; that engine is not
    /// RT-safe anyway). CPU cost is bounded by the preroll length.
    ///
    /// # Errors
    ///
    /// Returns an error if `preroll` is not whole frames, or if an internal
    /// buffer overflows while priming. After an error the processor state
    /// is cleared but unprimed; callers should fall back to [`Self::reset`].
    pub fn warm_start_seek(
        &mut self,
        target_frame: usize,
        preroll: &[f32],
    ) -> Result<(), StretchError> {
        let num_channels = self.params.channels.count().max(1);
        if preroll.len() % num_channels != 0 {
            return Err(StretchError::InvalidState(
                "warm-start preroll must contain whole frames",
            ));
        }

        // Bound CPU: keep only the most recent preroll frames, and never
        // more than actually precede the target.
        let max_frames = self.warm_start_preroll_frames().min(target_frame);
        let preroll_frames = (preroll.len() / num_channels).min(max_frames);
        let preroll = &preroll[preroll.len() - preroll_frames * num_channels..];
        let preroll_start = target_frame - preroll_frames;

        // Clear per-stream DSP state without deallocating. Energy/gain EMAs
        // and their filter states are deliberately preserved: a seek stays
        // within the same track, so the levels they track remain valid and
        // the preroll refines them exactly as a continuous stream would.
        self.input_ring.clear();
        self.pending_output.clear();
        for buf in &mut self.channel_input_buffers {
            buf.clear();
        }
        for buf in &mut self.channel_output_buffers {
            buf.clear();
        }
        for vocoder in &mut self.vocoders {
            vocoder.reset_streaming_state();
        }
        self.transient_scheduler.reset();
        self.reset_pitch_resamplers();
        self.wsola_overlay_remaining = 0;
        self.wsola_overlay_total = 0;
        self.wsola_overlay_pos = 0;
        for buf in &mut self.wsola_overlay_buffers {
            buf.clear();
        }
        if self.use_hybrid {
            // The rolling re-render window cannot survive a jump; rebuilding
            // it allocates, which the legacy hybrid engine already does per
            // render (documented non-RT).
            self.hybrid_state.reset(
                &self.params,
                self.current_ratio,
                self.capacity_frames_per_channel,
            );
            self.hybrid_pending_rebase = false;
        }

        // Timeline and accounting: the stream resumes at the preroll start;
        // the preroll's rendered output is discarded below, and post-target
        // output begins a fresh accounting window. The first audible output
        // fades in briefly — the caller cut the old stream mid-waveform.
        self.input_frames_consumed_total = 0;
        self.passthrough_frames_total = 0;
        self.source_start_frames = preroll_start;
        self.reposition_artifact_cursor();
        self.expected_total_output_samples = 0.0;
        self.total_output_emitted_samples = 0;
        self.initialized = true;
        self.warm_start_fade_remaining = WARM_START_FADE_SAMPLES;

        // Unity fast path: with no stretch or pitch in play, the bit-exact
        // passthrough resumes instantly and needs no priming.
        let unity = (self.target_ratio - 1.0).abs() < RATIO_SNAP_THRESHOLD
            && (self.current_ratio - 1.0).abs() < RATIO_SNAP_THRESHOLD
            && (self.pitch_scale - 1.0).abs() < RATIO_SNAP_THRESHOLD
            && (self.target_pitch_scale - 1.0).abs() < RATIO_SNAP_THRESHOLD;
        if unity || preroll_frames == 0 {
            self.dsp_engaged = false;
            self.source_start_frames = target_frame;
            self.reposition_artifact_cursor();
            return Ok(());
        }

        // Prime: run the preroll through the full DSP path — phase vocoder,
        // schedulers, pitch resamplers, gain tracking — discarding the
        // rendered output. Control glides do NOT advance: the jump is
        // instantaneous, so no wall-clock time passes.
        self.dsp_engaged = true;
        let chunk_samples = MAX_CALLBACK_FRAMES * num_channels;
        for chunk in preroll.chunks(chunk_samples) {
            self.push_input_samples(chunk)?;
            self.process_available_to_pending(true)?;
            self.pending_output.clear();
        }
        self.pending_output.clear();
        self.expected_total_output_samples = 0.0;
        self.total_output_emitted_samples = 0;

        Ok(())
    }

    /// Attaches or clears an offline pre-analysis artifact.
    ///
    /// When the artifact is usable (sample-rate match, confident, non-empty),
    /// its beat/onset data drives transient handling instead of online
    /// detection. This precomputes derived views and is **not** RT-safe:
    /// call it at build/rebuild time, never from an audio callback.
    pub fn set_pre_analysis(&mut self, artifact: Option<PreAnalysisArtifact>) {
        self.params.pre_analysis = artifact;
        self.refresh_artifact_state();
    }

    /// Recomputes the cached artifact gate and derived anchor views from
    /// `params.pre_analysis`.
    fn refresh_artifact_state(&mut self) {
        let usable = self.params.pre_analysis.as_ref().filter(|artifact| {
            artifact.is_usable(
                self.params.sample_rate,
                self.params.beat_snap_confidence_threshold,
            )
            // The onset cursor relies on ascending positions; artifacts from
            // `analyze_for_dj` always satisfy this.
            && artifact.transient_onsets.is_sorted()
            && artifact.beat_positions.is_sorted()
        });

        match usable {
            Some(artifact) => {
                let strengths: Vec<f32> = (0..artifact.transient_onsets.len())
                    .map(|i| artifact.strength_at(i))
                    .collect();
                self.artifact_merged_anchors = merge_onsets_and_beats(
                    &artifact.transient_onsets,
                    &strengths,
                    &artifact.beat_positions,
                    usize::MAX,
                );
                self.artifact_active = true;
            }
            None => {
                self.artifact_merged_anchors = (Vec::new(), Vec::new());
                self.artifact_active = false;
            }
        }
        self.reposition_artifact_cursor();
    }

    /// Repositions the onset cursor to the first artifact onset at or after
    /// the current absolute source position.
    fn reposition_artifact_cursor(&mut self) {
        let position = self.ring_start_abs();
        self.artifact_onset_cursor = match self.params.pre_analysis.as_ref() {
            Some(artifact) => artifact
                .transient_onsets
                .partition_point(|&onset| onset < position),
            None => 0,
        };
    }

    /// Callback-size-agnostic ratio and pitch interpolation.
    fn interpolate_ratio_for_frames(&mut self, frames: usize) {
        let tau_frames = (self.params.sample_rate as f64 * RATIO_SMOOTHING_TIME_SECS).max(1.0);
        let alpha = 1.0 - (-(frames as f64) / tau_frames).exp();
        self.current_ratio += alpha * (self.target_ratio - self.current_ratio);

        if (self.current_ratio - self.target_ratio).abs() < RATIO_SNAP_THRESHOLD {
            self.current_ratio = self.target_ratio;
        }

        self.pitch_scale += alpha * (self.target_pitch_scale - self.pitch_scale);
        if (self.pitch_scale - self.target_pitch_scale).abs() < RATIO_SNAP_THRESHOLD {
            self.pitch_scale = self.target_pitch_scale;
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
fn estimate_period_from_tail(tail: &[f32]) -> Option<f64> {
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
        return None;
    }
    // Fractional period: integer rounding here drifts the synthesized phase
    // by up to half a sample per cycle across the splice region, which is
    // audible as a click at the splice boundary.
    Some((sum as f64 / n as f64).max(1.0))
}

#[inline]
fn fit_tonal_tail(samples: &[f32], global_start: usize, period: f64) -> Option<(f64, f64, f64)> {
    if samples.is_empty() || period < 1.0 {
        return None;
    }

    let period_len = period.round().max(1.0) as usize;
    let fit_len = (period_len * 12).min(samples.len()).max(period_len * 3);
    let fit_start = samples.len().saturating_sub(fit_len);
    let fit = &samples[fit_start..];
    if fit.len() < period_len * 2 {
        return None;
    }

    let mean = fit.iter().map(|&s| s as f64).sum::<f64>() / fit.len() as f64;
    let w = 2.0 * std::f64::consts::PI / period;

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

    // Raw least-squares amplitude: the synthesized tail must continue the
    // *actual* (possibly decayed) tail level. Flooring the amplitude toward
    // the tail peak used to force a level jump at the splice point.
    let a = (xc * ss - xs * cs) / det;
    let b = (xs * cc - xc * cs) / det;

    Some((a, b, mean))
}

/// Extends `output` by synthesizing a tonal continuation from the tail.
///
/// This keeps end-of-stream length correction from introducing flat or noisy
/// tails that would skew chunk-level pitch and envelope checks.
///
/// Continuity at the splice: the fitted sinusoid is gain-matched to the RMS
/// of the real tail region it replaces and blended in with a linear
/// crossfade (both signals are phase-aligned tonal content via the LSQ fit,
/// so they sum coherently), instead of hard-overwriting from `synth_start` —
/// the hard rewrite used to click at the splice boundary.
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
            let w = 2.0 * std::f64::consts::PI / period;
            let rewritten = output.len().saturating_sub(synth_start);

            // Evaluate the fitted sinusoid over the blend region.
            let synth: Vec<f64> = (0..rewritten)
                .map(|i| {
                    let n = (synth_start + i) as f64;
                    a * (w * n).cos() + b * (w * n).sin() + mean
                })
                .collect();

            // Gain-match the synthetic tail to the level of the real region
            // it blends over, so it continues the decayed tail rather than
            // the pre-decay fit amplitude.
            let real_rms = {
                let sum: f64 = output[synth_start..]
                    .iter()
                    .map(|&s| (s as f64) * (s as f64))
                    .sum();
                (sum / rewritten.max(1) as f64).sqrt()
            };
            let synth_rms = {
                let sum: f64 = synth.iter().map(|&s| s * s).sum();
                (sum / synth.len().max(1) as f64).sqrt()
            };
            let gain = if synth_rms > 1e-9 && real_rms > 1e-9 {
                (real_rms / synth_rms).clamp(0.25, 4.0)
            } else {
                1.0
            };

            // Linear crossfade real -> synth across the blend region. The
            // fade starts near 0 (C0-continuous at synth_start) and ends
            // near 1 (the appended region continues from pure synth).
            for (i, &s) in synth.iter().enumerate() {
                let t = (i + 1) as f64 / (rewritten + 1) as f64;
                let real = output[synth_start + i] as f64;
                output[synth_start + i] = (real * (1.0 - t) + s * gain * t) as f32;
            }

            let start = output.len();
            for i in 0..count {
                let n = (start + i) as f64;
                let y = (a * (w * n).cos() + b * (w * n).sin() + mean) * gain;
                output.push(y as f32);
            }
            return;
        }
    }

    // No tonal fit: decay linearly from the last sample instead of holding
    // it as DC (a DC hold ends the stream with a step to silence).
    let pad = *output.last().unwrap_or(&0.0);
    let start = output.len();
    output.reserve(count);
    for i in 0..count {
        let t = 1.0 - (i + 1) as f32 / count as f32;
        output.push(pad * t);
    }
    debug_assert_eq!(output.len(), start + count);
}

/// Applies a linear fade-out over the last `fade_len` samples.
fn fade_out_tail(output: &mut [f32], fade_len: usize) {
    let len = output.len();
    let fade = fade_len.min(len);
    if fade == 0 {
        return;
    }
    let start = len - fade;
    for (i, s) in output[start..].iter_mut().enumerate() {
        let t = 1.0 - (i + 1) as f32 / fade as f32;
        *s *= t;
    }
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
    fn test_hybrid_window_base_tracks_discards() {
        let params = StretchParams::new(1.1)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut state = HybridStreamingState::new(&params, 1.1, 4096);
        assert!(!state.window_base_valid);

        // Anchor the base as append_hybrid_input would on first append.
        state.window_base_abs = 500;
        state.window_base_valid = true;
        let fill = vec![0.0f32; state.max_tail_frames + 300];
        state.rolling_inputs[0].push_slice(&fill);

        // Tail retention discards from the front and must advance the base.
        state.retain_tail();
        assert_eq!(state.window_base_abs, 800);
        assert_eq!(state.rolling_inputs[0].len(), state.max_tail_frames);

        // A no-op retention leaves the base untouched.
        state.retain_tail();
        assert_eq!(state.window_base_abs, 800);

        // Rebase after a ratio change goes through the same path.
        let drain = state.rolling_inputs[0].len();
        state.rolling_inputs[0].discard(drain);
        state.rolling_inputs[0].push_slice(&fill);
        state.rebase_after_ratio_change();
        assert_eq!(state.window_base_abs, 1100);

        // A full reset invalidates the anchor.
        state.reset(&params, 1.1, 4096);
        assert!(!state.window_base_valid);
        assert_eq!(state.window_base_abs, 0);
    }

    #[test]
    fn test_append_hybrid_input_anchors_and_advances_window_base() {
        let params = StretchParams::new(1.1)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256)
            .with_beat_snap_confidence_threshold(0.1);
        let mut proc = StreamProcessor::new(params);
        proc.set_source_position(10_000).expect("fresh position");

        // First append anchors the base at the ring's absolute position.
        proc.channel_input_buffers[0].clear();
        proc.channel_input_buffers[0].extend_from_slice(&vec![0.0f32; 2048]);
        proc.append_hybrid_input(1).expect("append");
        assert!(proc.hybrid_state.window_base_valid);
        assert_eq!(proc.hybrid_state.window_base_abs, 10_000);

        // Overflow the rolling ring: the front-discard advances the base.
        let capacity = proc.hybrid_state.rolling_inputs[0].capacity();
        let available = proc.hybrid_state.rolling_inputs[0].available();
        let overflow = 64usize;
        proc.channel_input_buffers[0].clear();
        proc.channel_input_buffers[0].extend_from_slice(&vec![0.0f32; available + overflow]);
        // Grow the scratch buffer for this synthetic oversized append.
        proc.channel_input_buffers[0].reserve(capacity);
        proc.append_hybrid_input(1).expect("append with overflow");
        assert_eq!(proc.hybrid_state.window_base_abs, 10_000 + overflow);
    }

    #[test]
    fn test_modulation_hold_zero_at_steady_state() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let proc = StreamProcessor::new(params);
        assert_eq!(proc.modulation_hold_overlap_windows(), 0);
    }

    #[test]
    fn test_modulation_hold_engages_and_scales_with_step() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);

        proc.set_stretch_ratio(1.004).unwrap();
        let small = proc.modulation_hold_overlap_windows();
        assert!(small >= 1, "small step should engage hold, got {}", small);

        proc.set_stretch_ratio(1.08).unwrap();
        let large = proc.modulation_hold_overlap_windows();
        assert!(
            large >= small,
            "hold should be monotone in step size: {} < {}",
            large,
            small
        );
        assert!(
            large <= proc.modulation_hold_max_overlap_windows(),
            "hold must stay within the time-based cap: {} > {}",
            large,
            proc.modulation_hold_max_overlap_windows()
        );
    }

    #[test]
    fn test_modulation_hold_cap_is_time_based() {
        // Reference configuration reproduces the legacy 4-window cap.
        let reference = StreamProcessor::new(
            StretchParams::new(1.0)
                .with_sample_rate(44100)
                .with_channels(1)
                .with_fft_size(4096)
                .with_hop_size(1024),
        );
        assert_eq!(reference.modulation_hold_max_overlap_windows(), 4);

        // Smaller windows get a proportionally larger budget covering the
        // same absolute time (~0.37 s).
        let low_latency = StreamProcessor::new(
            StretchParams::new(1.0)
                .with_sample_rate(44100)
                .with_channels(1)
                .with_fft_size(1024)
                .with_hop_size(256),
        );
        assert_eq!(low_latency.modulation_hold_max_overlap_windows(), 16);

        // A 1.0 -> 1.08 snap at the reference config still hits the cap
        // (the legacy behavior this budget was tuned for).
        let mut proc = StreamProcessor::new(
            StretchParams::new(1.0)
                .with_sample_rate(44100)
                .with_channels(1)
                .with_fft_size(4096)
                .with_hop_size(1024),
        );
        proc.set_stretch_ratio(1.08).unwrap();
        assert_eq!(proc.modulation_hold_overlap_windows(), 4);
    }

    #[test]
    fn test_modulation_hold_engages_for_pitch_slew() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_pitch_scale(1.05).unwrap();
        assert!(proc.modulation_hold_overlap_windows() >= 1);
    }

    #[test]
    fn test_modulation_hold_decays_after_settling() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_stretch_ratio(1.04).unwrap();
        assert!(proc.modulation_hold_overlap_windows() >= 1);

        // ~700 ms of audio lets the 50 ms control EMA settle to the target.
        let input: Vec<f32> = (0..30_000)
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let mut output = Vec::with_capacity(input.len() * 2 + 65_536);
        for chunk in input.chunks(256) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        assert_eq!(
            proc.modulation_hold_overlap_windows(),
            0,
            "hold should release once the slew settles"
        );
    }

    #[test]
    fn test_extend_with_tonal_tail_splice_continuity() {
        // Decaying 220 Hz sine: the splice into the synthesized tail and the
        // append boundary must both stay within the tone's natural slew.
        let sr = 44_100.0f32;
        let freq = 220.0f32;
        let n = 8192usize;
        let mut output: Vec<f32> = (0..n)
            .map(|i| {
                let decay = 1.0 - 0.5 * i as f32 / n as f32;
                0.5 * decay * (2.0 * PI * freq * i as f32 / sr).sin()
            })
            .collect();

        let count = 700usize;
        extend_with_tonal_tail(&mut output, count, 0);
        assert_eq!(output.len(), n + count);

        let natural_slew = 0.5 * 2.0 * PI * freq / sr;
        let mut worst = (0usize, 0.0f32);
        for (i, w) in output.windows(2).enumerate() {
            let d = (w[1] - w[0]).abs();
            if d > worst.1 {
                worst = (i, d);
            }
        }
        assert!(
            worst.1 <= natural_slew * 1.5,
            "tonal-tail splice discontinuity at {}: |delta|={:.4} > {:.4}",
            worst.0,
            worst.1,
            natural_slew * 1.5
        );
    }

    #[test]
    fn test_extend_with_tonal_tail_fallback_decays() {
        // Non-tonal tail (no zero crossings): fallback must decay to zero
        // instead of holding the last sample as DC.
        let mut output = vec![0.3f32; 100];
        extend_with_tonal_tail(&mut output, 64, 0);
        assert_eq!(output.len(), 164);
        assert!(output[100..].windows(2).all(|w| w[1] <= w[0] + 1e-6));
        assert!(output.last().unwrap().abs() < 1e-6);
    }

    #[test]
    fn test_fade_out_tail() {
        let mut output = vec![1.0f32; 256];
        fade_out_tail(&mut output, 128);
        assert!((output[127] - 1.0).abs() < 1e-6);
        assert!(output[128] < 1.0);
        assert!(output.last().unwrap().abs() < 1e-6);
        for w in output[128..].windows(2) {
            assert!(w[1] <= w[0]);
        }

        // Degenerate cases must not panic.
        fade_out_tail(&mut [], 16);
        fade_out_tail(&mut output, 0);
        let mut short = vec![0.5f32; 4];
        fade_out_tail(&mut short, 128);
        assert!(short.last().unwrap().abs() < 1e-6);
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
        // 2*pi*440/44100 ≈ 0.063. Plain streaming can produce somewhat larger
        // seam deltas during live ratio changes than the removed deterministic
        // path, but obvious clicks still show up as near-full-scale jumps.
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
        assert_eq!(proc.params().fft_size, 1024);
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
    fn test_stream_processor_pitch_quality_selection() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        assert_eq!(proc.pitch_resampler_quality(), StreamPitchQuality::Sinc);
        proc.set_pitch_resampler_quality(StreamPitchQuality::Linear)
            .unwrap();
        assert_eq!(proc.pitch_resampler_quality(), StreamPitchQuality::Linear);

        // The linear fallback still applies a measurable pitch shift.
        proc.set_pitch_scale(1.08).unwrap();
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin() * 0.8)
            .collect();
        let mut output = Vec::with_capacity(input.len() * 2);
        for chunk in input.chunks(1024) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        proc.flush_into(&mut output).unwrap();
        let trim = 4096usize.min(output.len() / 4);
        let end = output.len().saturating_sub(trim).max(trim + 2);
        let measured = estimate_freq_zero_crossings(&output[trim..end], 44100);
        assert!(
            measured > 460.0,
            "linear fallback should still shift pitch, got {:.3} Hz",
            measured
        );
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
        let mut output = Vec::with_capacity(input.len() * 2);
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

        // Small chunks may or may not emit immediately depending on the
        // current streaming priming strategy, but they must never shrink output.
        let small = vec![0.0f32; 1024];
        let before_small = output.len();
        proc.process_into(&small, &mut output).unwrap();
        let written_small = output.len() - before_small;
        assert!(written_small <= small.len());

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

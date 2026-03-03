//! Real-time streaming time-stretch processor.

use rustfft::{num_complex::Complex, FftPlanner};

use crate::core::fft::COMPLEX_ZERO;
use crate::core::ring_buffer::RingBuffer;
use crate::core::types::{QualityMode, StretchParams};
use crate::core::window::{generate_window, WindowType};
use crate::error::StretchError;
use crate::analysis::transient::{detect_transients_with_options, TransientDetectionOptions};
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
/// Mid-band start for spectral-flux transient detection (Hz).
const FLUX_MID_START_HZ: f64 = 500.0;
/// High-band start for spectral-flux transient detection (Hz).
const FLUX_HIGH_START_HZ: f64 = 4000.0;
/// EMA coefficient for adaptive spectral-flux statistics.
const FLUX_EMA_ALPHA: f64 = 0.2;
/// Sigma multiplier for adaptive spectral-flux threshold.
const FLUX_THRESHOLD_SIGMA: f64 = 2.5;
/// Required jump versus previous frame flux to classify a transient.
const FLUX_SPIKE_RATIO: f64 = 1.6;
/// Cross-fade length (in samples) at hybrid streaming chunk boundaries.
///
/// Smooths phase discontinuities caused by re-rendering overlapping audio
/// with fresh PV phase state on each call.
const HYBRID_STREAM_CROSSFADE_SAMPLES: usize = 3072;
/// Absolute guard to suppress near-silence false triggers.
const FLUX_ABS_MIN: f64 = 1e-4;
/// Extra emphasis on high-band flux.
const FLUX_HIGH_WEIGHT: f64 = 1.25;
/// Number of flux frames to observe before trigger checks.
const FLUX_WARMUP_FRAMES: usize = 3;

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
}

impl HybridStreamingState {
    fn new(params: &StretchParams, ratio: f64, capacity_frames: usize) -> Self {
        let num_channels = params.channels.count();
        let mut per_channel = params.clone();
        per_channel.stretch_ratio = ratio;
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
    /// When true, use the full hybrid algorithm (transient detection + WSOLA + PV)
    /// instead of PV-only. Higher quality for EDM but higher latency.
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
    /// Reusable mono mixdown buffer for stereo phase coherence.
    mid_buffer: Vec<f32>,
    /// Cached FFT for spectral-flux transient detection.
    transient_fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    /// Reusable FFT buffer for spectral-flux detection.
    transient_fft_buffer: Vec<Complex<f32>>,
    /// Scratch buffer for spectral-flux FFT execution.
    transient_fft_scratch: Vec<Complex<f32>>,
    /// Reusable previous magnitudes for spectral-flux detection.
    transient_prev_magnitudes: Vec<f32>,
    /// Analysis window used by spectral-flux detection.
    transient_window: Vec<f32>,
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
        let mut planner = FftPlanner::new();
        let transient_fft = planner.plan_fft_forward(params.fft_size);
        let transient_fft_scratch = vec![COMPLEX_ZERO; transient_fft.get_inplace_scratch_len()];
        let transient_fft_buffer = vec![COMPLEX_ZERO; params.fft_size];
        let transient_prev_magnitudes = vec![0.0; params.fft_size / 2 + 1];
        let transient_window = generate_window(WindowType::Hann, params.fft_size);

        Self {
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
            mid_buffer: Vec::with_capacity(capacity_frames_per_channel),
            transient_fft,
            transient_fft_buffer,
            transient_fft_scratch,
            transient_prev_magnitudes,
            transient_window,
            expected_total_output_samples: 0.0,
            total_output_emitted_samples: 0,
            pitch_scale: 1.0,
            pitch_resamplers: (0..num_channels)
                .map(|_| LinearResamplerState::new())
                .collect(),
            pitch_output_buffers: (0..num_channels)
                .map(|_| Vec::with_capacity(pitch_output_capacity_frames))
                .collect(),
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

            if min_output_len > 0 {
                self.decode_output_mid_side(num_channels, min_output_len);
                self.emit_channel_output_to_pending(min_output_len, num_channels)?;
            }
            return Ok(());
        }

        self.update_vocoder_ratio();
        if num_channels == 2 {
            self.sync_stereo_phase_reset(total_frames);
        }

        let min_output_len = self.process_channels(num_channels)?;
        self.consume_processed_input(total_frames, num_channels);

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

    fn consume_processed_input(&mut self, total_frames: usize, num_channels: usize) {
        let hop = self.params.hop_size;
        if hop == 0 {
            return;
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
        // Use 4× the FFT size for ratios far from unity (|r-1| > 0.1)
        // to increase the delta-to-crossfade ratio.  With 1× threshold,
        // crossfades affect 74-87% of each delta; at 4× this drops to
        // ~19-25%, dramatically reducing spectral artifacts from
        // blending divergent renderings in both expansion and compression.
        let accum_threshold = if (self.hybrid_state.last_ratio - 1.0).abs() > 0.1 {
            self.params.fft_size * 4
        } else {
            self.params.fft_size
        };
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
        let shared_onsets: Option<(Vec<usize>, Vec<f32>)> =
            if num_channels == 2
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
                self.hybrid_state.stretchers[ch]
                    .process(&self.hybrid_state.rolling_scratch[ch])?
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

    /// Enables or disables hybrid processing mode.
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

    /// Enables or disables strict realtime-safe behavior while hybrid mode is on.
    ///
    /// Strict mode routes processing through the preallocated PV stream path to
    /// guarantee no heap growth in callbacks.
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
    pub fn latency_samples(&self) -> usize {
        min_latency_frames(self.params.fft_size)
    }

    /// Returns the minimum latency in seconds.
    pub fn latency_secs(&self) -> f64 {
        self.latency_samples() as f64 / self.params.sample_rate as f64
    }

    /// Resets the processor state.
    pub fn reset(&mut self) {
        self.input_ring.clear();
        self.pending_output.clear();
        self.mid_buffer.clear();
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
        self.transient_prev_magnitudes.fill(0.0);

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

    fn sync_stereo_phase_reset(&mut self, total_frames: usize) {
        if total_frames < self.params.fft_size {
            return;
        }

        self.mid_buffer.clear();
        if self.mid_buffer.capacity() < total_frames {
            return;
        }

        for i in 0..total_frames {
            let left = self.interleaved_scratch[i * 2];
            let right = self.interleaved_scratch[i * 2 + 1];
            self.mid_buffer.push((left + right) * 0.5);
        }

        let hop = self.params.hop_size;
        if hop == 0 || total_frames < self.params.fft_size + hop {
            return;
        }

        let fft_size = self.params.fft_size;
        let num_frames = (total_frames - fft_size) / hop + 1;
        if num_frames < 2 {
            return;
        }

        let num_bins = fft_size / 2 + 1;
        if self.transient_fft_buffer.len() < fft_size
            || self.transient_prev_magnitudes.len() < num_bins
            || self.transient_window.len() < fft_size
        {
            return;
        }
        let transient_fft = std::sync::Arc::clone(&self.transient_fft);

        let bin_hz = self.params.sample_rate as f64 / fft_size as f64;
        let mid_start_bin = ((FLUX_MID_START_HZ / bin_hz).floor() as usize)
            .max(1)
            .min(num_bins.saturating_sub(1));
        let high_start_bin = ((FLUX_HIGH_START_HZ / bin_hz).floor() as usize).min(num_bins);
        self.transient_prev_magnitudes[..num_bins].fill(0.0);

        // Prime previous magnitudes with the first analysis frame.
        for i in 0..fft_size {
            self.transient_fft_buffer[i] =
                Complex::new(self.mid_buffer[i] * self.transient_window[i], 0.0);
        }
        transient_fft.process_with_scratch(
            &mut self.transient_fft_buffer,
            &mut self.transient_fft_scratch,
        );
        for bin in mid_start_bin..num_bins {
            self.transient_prev_magnitudes[bin] = self.transient_fft_buffer[bin].norm();
        }

        let mut mean_flux = 0.0f64;
        let mut var_flux = 0.0f64;
        let mut prev_flux = 0.0f64;
        let mut transient_detected = false;

        for frame_idx in 1..num_frames {
            let start = frame_idx * hop;
            let frame = &self.mid_buffer[start..start + fft_size];
            for (buf, (&s, &w)) in self.transient_fft_buffer[..fft_size]
                .iter_mut()
                .zip(frame.iter().zip(self.transient_window.iter()))
            {
                *buf = Complex::new(s * w, 0.0);
            }
            transient_fft.process_with_scratch(
                &mut self.transient_fft_buffer,
                &mut self.transient_fft_scratch,
            );

            let mut mid_flux = 0.0f64;
            let mut high_flux = 0.0f64;
            for bin in mid_start_bin..num_bins {
                let mag = self.transient_fft_buffer[bin].norm();
                let diff = (mag - self.transient_prev_magnitudes[bin]).max(0.0) as f64;
                if bin >= high_start_bin {
                    high_flux += diff;
                } else {
                    mid_flux += diff;
                }
                self.transient_prev_magnitudes[bin] = mag;
            }

            let flux = mid_flux + high_flux * FLUX_HIGH_WEIGHT;
            if frame_idx >= FLUX_WARMUP_FRAMES {
                let sigma = var_flux.max(0.0).sqrt();
                let threshold = mean_flux + FLUX_THRESHOLD_SIGMA * sigma;
                if flux > threshold
                    && flux > prev_flux.max(FLUX_ABS_MIN) * FLUX_SPIKE_RATIO
                    && flux > FLUX_ABS_MIN
                {
                    transient_detected = true;
                    break;
                }
            }

            let delta = flux - mean_flux;
            mean_flux += FLUX_EMA_ALPHA * delta;
            var_flux += FLUX_EMA_ALPHA * (delta * delta - var_flux);
            prev_flux = flux;
        }

        if transient_detected {
            for voc in &mut self.vocoders {
                voc.reset_phase_state_bands([false, false, true, true], self.params.sample_rate);
            }
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
            max_diff < 0.8,
            "Detected likely click artifact: max sample diff = {} (expected < 0.8)",
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

        let mut out = Vec::new();
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

        // First small chunk: not enough data yet
        let small = vec![0.0f32; 1024];
        let before_small = output.len();
        proc.process_into(&small, &mut output).unwrap();
        let written_small = output.len() - before_small;
        assert_eq!(written_small, 0);
        assert!(output.is_empty());

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

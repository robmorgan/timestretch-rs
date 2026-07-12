//! Real-time streaming time-stretch processor.

use std::sync::Arc;

use crate::core::preanalysis::PreAnalysisArtifact;
use crate::core::resample::{
    SincInterpTable, StreamingSincResampler, STREAM_SINC_MAX_HALF_TAPS, STREAM_SINC_MAX_STEP,
};
use crate::core::ring_buffer::RingBuffer;
use crate::core::types::{QualityMode, StreamProfile, StretchParams};
use crate::error::StretchError;
use crate::stream::transient_scheduler::{TransientEventScheduler, TransientSchedulerStats};
use crate::stretch::multi_resolution::MultiResolutionStretcher;
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

/// Smallest mid-band FFT the multi-resolution streaming engine supports.
///
/// The engine's buffering gate is set by the sub-bass band (`mid * 4`), so
/// at the Live profile's 1024-point FFT the gate would still be ~139 ms —
/// the whole point of Live is defeated. Selecting
/// [`StreamingEngine::MultiResolution`] below this mid FFT is an error.
const MULTI_RES_MIN_MID_FFT: usize = 2048;
/// Ceiling on the multi-resolution engine's derived sub-bass FFT
/// (`mid * 4`, so Club/2048 lands on 8192 and Quality/4096 on the offline
/// default of 16384; larger custom mid FFTs are capped here).
const MULTI_RES_SUB_BASS_FFT_CAP: usize = 16384;

/// Sub-bass FFT size the multi-resolution streaming engine derives from a
/// mid-band FFT size (used for gate/capacity math before construction).
#[inline]
const fn multi_res_sub_bass_fft(mid_fft: usize) -> usize {
    let derived = mid_fft * 4;
    if derived > MULTI_RES_SUB_BASS_FFT_CAP {
        MULTI_RES_SUB_BASS_FFT_CAP
    } else {
        derived
    }
}

/// Widest tempo ratio the varispeed-first control path accepts.
///
/// The varispeed input stage runs the streaming sinc resampler at
/// `step = 1 / ratio`; the floor keeps the step within the kernel's
/// anti-aliased range (`STREAM_SINC_MAX_STEP`), and the ceiling bounds the
/// preallocated per-callback emission headroom (a 1024-frame callback emits
/// at most `1024 * VARISPEED_MAX_RATIO` resampled frames).
const VARISPEED_MAX_RATIO: f64 = VARISPEED_MAX_RATIO_INT as f64;
/// Integer form of [`VARISPEED_MAX_RATIO`] for const capacity math.
const VARISPEED_MAX_RATIO_INT: usize = 4;
/// Narrowest tempo ratio the varispeed-first control path accepts
/// (`1 / STREAM_SINC_MAX_STEP`).
const VARISPEED_MIN_RATIO: f64 = 1.0 / STREAM_SINC_MAX_STEP;

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
fn validate_varispeed_ratio(ratio: f64) -> Result<f64, StretchError> {
    if !(VARISPEED_MIN_RATIO..=VARISPEED_MAX_RATIO).contains(&ratio) {
        return Err(StretchError::InvalidRatio(format!(
            "varispeed-first control path needs a stretch ratio in [{}, {}], got {}",
            VARISPEED_MIN_RATIO, VARISPEED_MAX_RATIO, ratio
        )));
    }
    Ok(ratio)
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

/// FFT size the active engine's buffering gate and capacities are keyed to:
/// the configured FFT for the deterministic engine, the derived sub-bass
/// FFT (the largest band) for the multi-resolution engine.
#[inline]
fn effective_stream_fft(params: &StretchParams, engine: StreamingEngine) -> usize {
    match engine {
        StreamingEngine::Deterministic => params.fft_size,
        StreamingEngine::MultiResolution => multi_res_sub_bass_fft(params.fft_size),
    }
}

#[inline]
fn stream_capacity_frames(
    params: &StretchParams,
    engine: StreamingEngine,
    control_path: ControlPath,
) -> usize {
    let _ = MIN_CALLBACK_FRAMES;
    let _ = COMMON_CALLBACK_FRAMES;
    let fft = effective_stream_fft(params, engine);
    // The varispeed-first path feeds the input ring through the varispeed
    // resampler, whose emission per callback expands by up to the maximum
    // tempo ratio (plus the released kernel lookahead).
    let callback_headroom = match control_path {
        ControlPath::VocoderTempo => MAX_CALLBACK_FRAMES,
        ControlPath::VarispeedFirst => varispeed_output_capacity_frames(),
    };
    analysis_lookahead_frames(fft, params.quality_mode)
        .saturating_add(callback_headroom)
        .saturating_add(fft)
}

/// Per-channel capacity of the varispeed stage's output scratch: the widest
/// emission a `MAX_CALLBACK_FRAMES` source block can produce at the minimum
/// step, including the cursor slack and released lookahead an extreme
/// mid-block retarget can add.
const fn varispeed_output_capacity_frames() -> usize {
    (MAX_CALLBACK_FRAMES + STREAM_SINC_MAX_HALF_TAPS + 2) * VARISPEED_MAX_RATIO_INT + 2
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

/// One resampled↔source timeline checkpoint recorded per varispeed block.
#[derive(Debug, Clone, Copy, Default)]
struct RatioMapEntry {
    /// Ring-timeline frame count (resampled frames emitted by the varispeed
    /// stage, resynced to the consumption cursor at engagement).
    resampled_abs: u64,
    /// Fractional source frames consumed by the varispeed stage at the
    /// checkpoint.
    source_pos: f64,
    /// Instantaneous tempo ratio in effect AT the checkpoint (the block's
    /// retarget value — the resampler's ramp lands on it at block end).
    ratio: f64,
}

/// Fixed-size FIFO mapping the ring (resampled) timeline back to source
/// frames on the varispeed-first control path.
///
/// The varispeed resampler's step is piecewise-linear per fed block, so one
/// checkpoint per block plus linear interpolation reconstructs the source
/// position of any in-flight ring frame to well under a hop of error. The
/// buffer is preallocated and never grows: pushes beyond capacity merge into
/// the newest entry (bounded extra interpolation error over one span), and
/// eviction always retains one anchor at or behind the consumption cursor.
#[derive(Debug)]
struct RatioMapFifo {
    entries: Box<[RatioMapEntry]>,
    head: usize,
    len: usize,
}

impl RatioMapFifo {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: vec![RatioMapEntry::default(); capacity].into_boxed_slice(),
            head: 0,
            len: 0,
        }
    }

    fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    #[inline]
    fn at(&self, i: usize) -> RatioMapEntry {
        self.entries[(self.head + i) % self.entries.len()]
    }

    /// Appends a checkpoint, deduping zero-advance blocks and merging into
    /// the newest entry when full.
    fn push(&mut self, resampled_abs: u64, source_pos: f64, ratio: f64) {
        if self.entries.is_empty() {
            return;
        }
        if self.len > 0 {
            let last_idx = (self.head + self.len - 1) % self.entries.len();
            let last = self.entries[last_idx];
            if resampled_abs <= last.resampled_abs || self.len == self.entries.len() {
                self.entries[last_idx] = RatioMapEntry {
                    resampled_abs: resampled_abs.max(last.resampled_abs),
                    source_pos: source_pos.max(last.source_pos),
                    ratio,
                };
                return;
            }
        }
        let idx = (self.head + self.len) % self.entries.len();
        self.entries[idx] = RatioMapEntry {
            resampled_abs,
            source_pos,
            ratio,
        };
        self.len += 1;
    }

    /// Drops checkpoints strictly behind `consumed`, always retaining one
    /// anchor entry at or behind it.
    fn evict_before(&mut self, consumed: u64) {
        while self.len >= 2 && self.at(1).resampled_abs <= consumed {
            self.head = (self.head + 1) % self.entries.len();
            self.len -= 1;
        }
    }

    /// Maps a ring-timeline frame count to fractional source frames.
    ///
    /// Clamps outside the checkpointed range (behind the anchor, or into
    /// flush padding beyond the final checkpoint). Returns `None` when no
    /// checkpoint exists yet.
    fn map_to_source(&self, resampled: f64) -> Option<f64> {
        if self.len == 0 {
            return None;
        }
        let first = self.at(0);
        if resampled <= first.resampled_abs as f64 {
            return Some(first.source_pos);
        }
        for i in 1..self.len {
            let a = self.at(i - 1);
            let b = self.at(i);
            if resampled <= b.resampled_abs as f64 {
                let span = (b.resampled_abs - a.resampled_abs) as f64;
                if span <= 0.0 {
                    return Some(b.source_pos);
                }
                let t = (resampled - a.resampled_abs as f64) / span;
                return Some(a.source_pos + t * (b.source_pos - a.source_pos));
            }
        }
        Some(self.at(self.len - 1).source_pos)
    }

    /// Instantaneous embedded tempo ratio at a ring-timeline frame count.
    ///
    /// Interpolates the per-checkpoint retarget values, matching the
    /// varispeed resampler's linear intra-block step ramp — unlike a
    /// position-gradient over a whole block, this resolves the ramp itself.
    /// Clamps outside the checkpointed range; `None` when empty.
    fn ratio_at(&self, resampled: f64) -> Option<f64> {
        if self.len == 0 {
            return None;
        }
        let first = self.at(0);
        if resampled <= first.resampled_abs as f64 {
            return Some(first.ratio);
        }
        for i in 1..self.len {
            let a = self.at(i - 1);
            let b = self.at(i);
            if resampled <= b.resampled_abs as f64 {
                let span = (b.resampled_abs - a.resampled_abs) as f64;
                if span <= 0.0 {
                    return Some(b.ratio);
                }
                let t = (resampled - a.resampled_abs as f64) / span;
                return Some(a.ratio + t * (b.ratio - a.ratio));
            }
        }
        Some(self.at(self.len - 1).ratio)
    }
}

/// Engine used by [`StreamProcessor`] for stream-mode rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingEngine {
    /// Default deterministic stream engine with bounded per-callback work.
    ///
    /// Uses persistent per-channel phase-vocoder state plus scheduled
    /// per-band transient phase resets, and never re-renders historical
    /// context in callback paths.
    Deterministic,
    /// Multi-resolution filterbank engine: a three-band Linkwitz-Riley
    /// split (sub-bass/mid/high) with a per-band phase vocoder whose FFT
    /// size is tuned to its frequency range (sub-bass `mid * 4`, high
    /// `mid / 4`).
    ///
    /// Trades latency for low-end quality: the buffering gate is set by the
    /// sub-bass FFT, so this engine needs the Club profile or larger
    /// (`fft_size >= 2048`); selecting it on the Live profile returns an
    /// error from [`StreamProcessor::set_streaming_engine`].
    MultiResolution,
}

/// Control-path architecture used by [`StreamProcessor`] to implement tempo
/// changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlPath {
    /// The phase vocoder implements the tempo change directly (default).
    ///
    /// Tempo control latency is the buffering gate plus the ~50 ms control
    /// glide. This is the right path for tape-mode/no-keylock use, where the
    /// caller wants pitch to follow tempo.
    VocoderTempo,
    /// A varispeed sinc resampler at the input implements the tempo change
    /// instantly; the phase vocoder only pitch-corrects at a slowly-varying
    /// transposition (keylock).
    ///
    /// Tempo retargets are sample-accurate at the resampler — control-to-
    /// audio latency drops to the resampler kernel lookahead plus one
    /// callback, while the vocoder's buffering gate becomes a constant
    /// pipeline delay the host can compensate in its timeline (see
    /// [`StreamProcessor::latency_report`]). Tempo ratios are bounded to
    /// `[0.25, 4.0]` on this path.
    VarispeedFirst,
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
///
/// The two headline figures separate concerns a deck integration must treat
/// differently:
///
/// - [`pipeline_delay_frames`](Self::pipeline_delay_frames) — constant
///   content delay from input to output. A host compensates it once in its
///   timeline (playhead offset), like any plugin delay.
/// - [`control_to_audio_frames`](Self::control_to_audio_frames) — how long a
///   *tempo change* takes to reach the output. On
///   [`ControlPath::VocoderTempo`] this is the whole pipeline delay (the PV
///   implements tempo behind its gate); on
///   [`ControlPath::VarispeedFirst`] it collapses to the varispeed kernel
///   lookahead (tens of samples) plus the caller's own callback size, while
///   the pipeline delay stays constant and host-compensated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamLatencyReport {
    /// Input-buffering floor in frames: `fft_size * 3/2`.
    pub base_gate_frames: usize,
    /// Actual buffering gate in frames at the current target processing
    /// ratio: equals `base_gate_frames` inside the `[0.9, 1.1]` ratio band,
    /// `fft_size * 2` outside it.
    pub effective_gate_frames: usize,
    /// Kernel lookahead of the realtime output (pitch/keylock-correction)
    /// resampler in samples: 0 when inactive; 16 (sinc, unity/down) up to
    /// 80 (sinc, up); 1 for the linear fallback.
    pub pitch_lookahead_samples: usize,
    /// Kernel lookahead of the varispeed input resampler in samples: 0 on
    /// [`ControlPath::VocoderTempo`] and while disengaged; 16–80 otherwise
    /// (16 at DJ tempo ratios).
    pub varispeed_lookahead_samples: usize,
    /// Time constant of the control glide in seconds (~0.050). On
    /// [`ControlPath::VocoderTempo`] it smooths ratio and pitch alike; on
    /// [`ControlPath::VarispeedFirst`] only the user-pitch axis — tempo
    /// retargets sample-accurately and its correction is delay-matched.
    pub control_smoothing_secs: f64,
    /// Sample rate the frame figures are expressed against.
    pub sample_rate: u32,
    /// Frames between a tempo-control change and its effect on the output,
    /// excluding the caller's own callback buffering.
    pub control_to_audio_frames: usize,
    /// Constant input-to-output content delay in frames, for host timeline
    /// compensation:
    /// `effective_gate_frames + pitch_lookahead_samples + varispeed_lookahead_samples`.
    pub pipeline_delay_frames: usize,
    /// Control path the figures describe.
    pub control_path: ControlPath,
    /// Total effective content latency in frames; equals
    /// [`pipeline_delay_frames`](Self::pipeline_delay_frames).
    pub total_frames: usize,
}

impl StreamLatencyReport {
    /// Total effective latency in seconds.
    pub fn total_secs(&self) -> f64 {
        self.total_frames as f64 / self.sample_rate.max(1) as f64
    }

    /// [`Self::control_to_audio_frames`] in seconds.
    pub fn control_to_audio_secs(&self) -> f64 {
        self.control_to_audio_frames as f64 / self.sample_rate.max(1) as f64
    }

    /// [`Self::pipeline_delay_frames`] in seconds.
    pub fn pipeline_delay_secs(&self) -> f64 {
        self.pipeline_delay_frames as f64 / self.sample_rate.max(1) as f64
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
    ///
    /// On [`ControlPath::VarispeedFirst`] these are resampled (ring
    /// timeline) frames, not source frames — use
    /// [`StreamProcessor::source_position`] for the source timeline.
    pub input_frames_consumed_total: usize,
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
    /// Active stream-mode rendering engine.
    engine: StreamingEngine,
    /// Active tempo control path.
    control_path: ControlPath,
    /// Per-channel varispeed input resamplers (varispeed-first path only;
    /// empty on the vocoder-tempo path).
    varispeed_resamplers: Vec<StreamingSincResampler>,
    /// Reusable per-channel deinterleave scratch feeding the varispeed stage.
    varispeed_input_buffers: Vec<Vec<f32>>,
    /// Reusable per-channel varispeed resampler output buffers.
    varispeed_output_buffers: Vec<Vec<f32>>,
    /// Reusable interleave scratch for varispeed output entering the ring.
    varispeed_interleave_scratch: Vec<f32>,
    /// True once the varispeed stage has resampled any input this stream.
    ///
    /// While engaged, the stage stays in the signal path even at unity ratio
    /// (a bit-transparent passthrough) so returning to `target_ratio == 1.0`
    /// never splices its held lookahead out of the stream.
    varispeed_engaged: bool,
    /// Cumulative real source frames fed to the varispeed stage (flush drain
    /// zeros are never counted).
    varispeed_source_fed_total: usize,
    /// Ring-timeline frame count of varispeed emissions: resampled frames
    /// pushed into `input_ring`, resynced to `input_frames_consumed_total`
    /// at engagement so both share one timeline.
    varispeed_resampled_emitted_total: u64,
    /// Source-fed total folded down at each varispeed resampler reset;
    /// checkpoint source positions are `base + next_output_source_pos()`.
    varispeed_map_base_source: f64,
    /// Tempo ratio applied at the previous varispeed block; bounds the next
    /// block's worst-case emission (the resampler ramps between the two).
    varispeed_prev_ratio: f64,
    /// Delay-matched pitch-correction ratio for the varispeed-first path:
    /// the embedded ratio at the end of the span the PV is about to consume
    /// (via the ratio map); the post-PV resampler's per-block step ramp
    /// smooths between consecutive targets. Unused (held at the build
    /// ratio) on the vocoder-tempo path.
    transposition: f64,
    /// Resampled↔source timeline checkpoints for in-flight ring content.
    ratio_map: RatioMapFifo,
    /// Persistent PhaseVocoder instances, one per channel (deterministic
    /// engine; empty when the multi-resolution engine is active).
    vocoders: Vec<PhaseVocoder>,
    /// Persistent multi-resolution stretchers, one per channel
    /// (multi-resolution engine; empty for the deterministic engine).
    multi_res: Vec<MultiResolutionStretcher>,
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
            .field("control_path", &self.control_path)
            .field("initialized", &self.initialized)
            .field("source_bpm", &self.source_bpm)
            .field("input_ring_len", &self.input_ring.len())
            .field("pending_output_len", &self.pending_output.len())
            .finish()
    }
}

impl StreamProcessor {
    /// Creates a new streaming processor (deterministic engine).
    pub fn new(params: StretchParams) -> Self {
        Self::new_with_engine(params, StreamingEngine::Deterministic)
    }

    /// Creates a new streaming processor for a specific engine.
    fn new_with_engine(params: StretchParams, engine: StreamingEngine) -> Self {
        Self::new_with_engine_and_path(params, engine, ControlPath::VocoderTempo)
    }

    /// Creates a new streaming processor for a specific engine and control
    /// path.
    ///
    /// Callers must have validated engine availability first (see
    /// [`Self::set_streaming_engine`]); buffer capacities and gates are
    /// sized for the engine's effective FFT and the control path's
    /// per-callback emission headroom.
    fn new_with_engine_and_path(
        params: StretchParams,
        engine: StreamingEngine,
        control_path: ControlPath,
    ) -> Self {
        let ratio = params.stretch_ratio;
        let num_channels = params.channels.count();
        let source_bpm = params.bpm;
        let sample_rate = params.sample_rate;
        let wsola_seg_size = params.wsola_segment_size;
        let wsola_search = params.wsola_search_range;

        let capacity_frames_per_channel = stream_capacity_frames(&params, engine, control_path);
        let capacity_samples = capacity_frames_per_channel.saturating_mul(num_channels);
        // Output headroom is sized for the widest supported stretch (a render
        // over the retained input emits up to ~ratio times as many frames).
        // The multiplier bounds the maximum usable stretch ratio: 8x the
        // input-capacity window keeps ratios up to ~10x from overflowing the
        // per-render PV output buffer (see `OUTPUT_CAPACITY_MULTIPLIER`).
        let output_capacity_frames = capacity_frames_per_channel
            .saturating_mul(OUTPUT_CAPACITY_MULTIPLIER)
            .saturating_add(effective_stream_fft(&params, engine));
        let output_capacity_samples = output_capacity_frames.saturating_mul(num_channels);
        let pitch_output_capacity_frames = output_capacity_frames.saturating_mul(2);

        let vocoders = match engine {
            StreamingEngine::Deterministic => {
                Self::create_vocoders(&params, ratio, capacity_frames_per_channel)
            }
            StreamingEngine::MultiResolution => Vec::new(),
        };
        let multi_res = match engine {
            StreamingEngine::Deterministic => Vec::new(),
            StreamingEngine::MultiResolution => {
                Self::create_multi_res(&params, ratio, capacity_frames_per_channel)
            }
        };
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
        let sinc_table = SincInterpTable::new_stream_default();

        // Varispeed stage state exists only on the varispeed-first path so
        // the default path carries no extra memory.
        let (
            varispeed_resamplers,
            varispeed_input_buffers,
            varispeed_output_buffers,
            varispeed_interleave_scratch,
        ) = match control_path {
            ControlPath::VocoderTempo => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            ControlPath::VarispeedFirst => (
                (0..num_channels)
                    .map(|_| StreamingSincResampler::new(Arc::clone(&sinc_table)))
                    .collect::<Vec<_>>(),
                (0..num_channels)
                    .map(|_| Vec::with_capacity(MAX_CALLBACK_FRAMES))
                    .collect::<Vec<_>>(),
                (0..num_channels)
                    .map(|_| Vec::with_capacity(varispeed_output_capacity_frames()))
                    .collect::<Vec<_>>(),
                vec![0.0; varispeed_output_capacity_frames().saturating_mul(num_channels)],
            ),
        };
        let ratio_map = match control_path {
            ControlPath::VocoderTempo => RatioMapFifo::with_capacity(0),
            ControlPath::VarispeedFirst => {
                RatioMapFifo::with_capacity(capacity_frames_per_channel / MIN_CALLBACK_FRAMES + 16)
            }
        };

        let (mut vocoders, mut multi_res) = (vocoders, multi_res);
        if control_path == ControlPath::VarispeedFirst {
            // The PV ratio on this path is the delay-matched transposition —
            // a smooth small-step stream, not discrete automation jumps.
            for pv in &mut vocoders {
                pv.set_smooth_ratio_updates(true);
            }
            for stretcher in &mut multi_res {
                stretcher.set_smooth_ratio_updates(true);
            }
        }

        let mut processor = Self {
            params,
            input_ring: RingBuffer::with_capacity(capacity_samples),
            pending_output: RingBuffer::with_capacity(output_capacity_samples),
            current_ratio: ratio,
            target_ratio: ratio,
            vocoder_ratio: ratio,
            initialized: false,
            engine,
            control_path,
            varispeed_resamplers,
            varispeed_input_buffers,
            varispeed_output_buffers,
            varispeed_interleave_scratch,
            varispeed_engaged: false,
            varispeed_source_fed_total: 0,
            varispeed_resampled_emitted_total: 0,
            varispeed_map_base_source: 0.0,
            varispeed_prev_ratio: ratio,
            transposition: ratio,
            ratio_map,
            vocoders,
            multi_res,
            channel_input_buffers,
            channel_output_buffers,
            interleaved_scratch: vec![0.0; capacity_samples],
            source_bpm,
            transient_scheduler,
            input_frames_consumed_total: 0,
            source_start_frames: 0,
            passthrough_frames_total: 0,
            artifact_active: false,
            artifact_onset_cursor: 0,
            artifact_events_scheduled_total: 0,
            artifact_reset_band_counts_total: [0; 4],
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

    /// Creates PhaseVocoder instances for each channel, pre-reserving their
    /// streaming buffers for the ring capacity so callback-path renders stay
    /// allocation-free as the ratio and render-window size move.
    fn create_vocoders(
        params: &StretchParams,
        ratio: f64,
        capacity_frames_per_channel: usize,
    ) -> Vec<PhaseVocoder> {
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
                pv.reserve_streaming_capacity(
                    capacity_frames_per_channel,
                    OUTPUT_CAPACITY_MULTIPLIER as f64,
                );
                pv
            })
            .collect()
    }

    /// Creates per-channel multi-resolution stretchers for streaming.
    fn create_multi_res(
        params: &StretchParams,
        ratio: f64,
        capacity_frames_per_channel: usize,
    ) -> Vec<MultiResolutionStretcher> {
        (0..params.channels.count())
            .map(|_| {
                // Match the deterministic engine's streaming tuning: wider
                // sub-bass rigid phase locking and no adaptive mode flips.
                let streaming_sub_bass_cutoff = params.sub_bass_cutoff.max(180.0);
                let mut stretcher = MultiResolutionStretcher::with_sub_bass_fft_cap(
                    params.fft_size,
                    ratio,
                    params.sample_rate,
                    streaming_sub_bass_cutoff,
                    MULTI_RES_SUB_BASS_FFT_CAP,
                );
                stretcher.set_adaptive_phase_locking(false);
                stretcher.set_envelope_strength(params.envelope_strength);
                stretcher.set_adaptive_envelope_order(params.adaptive_envelope_order);
                stretcher.reserve_streaming_capacity(
                    capacity_frames_per_channel,
                    OUTPUT_CAPACITY_MULTIPLIER as f64,
                );
                stretcher
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
            && !self.varispeed_engaged
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
        if self.control_path == ControlPath::VarispeedFirst && input.len() % num_channels != 0 {
            return Err(StretchError::InvalidState(
                "varispeed-first streaming requires whole interleaved frames",
            ));
        }
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

            let remaining = input.len() - offset;
            let take = match self.control_path {
                ControlPath::VocoderTempo => remaining.min(self.input_ring.available()),
                ControlPath::VarispeedFirst => self.varispeed_take_samples(remaining, num_channels),
            };
            let take = if take == 0 {
                // No room for this feed: render buffered input to make some,
                // advancing the control glide by a callback of wall time.
                self.interpolate_ratio_for_frames(COMMON_CALLBACK_FRAMES);
                self.process_available_to_pending(true)?;
                let retry = match self.control_path {
                    ControlPath::VocoderTempo => remaining.min(self.input_ring.available()),
                    ControlPath::VarispeedFirst => {
                        self.varispeed_take_samples(remaining, num_channels)
                    }
                };
                if retry == 0 {
                    return Err(StretchError::BufferOverflow {
                        buffer: "stream_input_ring",
                        requested: remaining,
                        available: self.input_ring.available(),
                    });
                }
                retry
            } else {
                take
            };

            let samples_pushed = match self.control_path {
                ControlPath::VocoderTempo => {
                    self.push_input_samples(&input[offset..offset + take])?;
                    take
                }
                ControlPath::VarispeedFirst => {
                    self.feed_varispeed_to_ring(&input[offset..offset + take], num_channels)?
                }
            };
            offset += take;

            let frames = (take / num_channels).max(1);
            self.interpolate_ratio_for_frames(frames);
            // Expected-length accounting: the vocoder path stretches ring
            // content by the glided ratio; the varispeed path already
            // resampled to output length (the PV plus post-resample nets
            // 1:1), so the emission itself is the expectation.
            self.expected_total_output_samples += match self.control_path {
                ControlPath::VocoderTempo => samples_pushed as f64 * self.current_ratio,
                ControlPath::VarispeedFirst => samples_pushed as f64,
            };
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

        // Release the varispeed stage's held lookahead into the ring first so
        // the tail below renders over real content, symmetric to the post-PV
        // pitch resampler flush at the other end of the chain.
        self.flush_varispeed_to_ring(num_channels)?;

        if !self.input_ring.is_empty() {
            let min_samples = self.effective_fft().saturating_mul(num_channels);
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

        self.flush_vocoder_tails_to_pending(num_channels)?;

        let remaining_frames = self.input_ring.len() / num_channels.max(1);
        self.input_frames_consumed_total = self
            .input_frames_consumed_total
            .saturating_add(remaining_frames);
        self.input_ring.clear();

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

    /// Largest interleaved sample count the varispeed stage can accept right
    /// now: whole frames, bounded by the per-block scratch and by ring
    /// availability for the block's worst-case emission.
    fn varispeed_take_samples(&self, remaining: usize, num_channels: usize) -> usize {
        let avail_frames = self.input_ring.available() / num_channels;
        let ratio_hint = self.target_ratio.max(self.varispeed_prev_ratio);
        // Invert the emission bound `frames * ratio + overhead <= avail`,
        // where the overhead covers cursor slack and lookahead release.
        let overhead = ((STREAM_SINC_MAX_HALF_TAPS + 2) as f64 * ratio_hint).ceil() as usize + 2;
        if avail_frames <= overhead {
            return 0;
        }
        let max_feed_frames = ((avail_frames - overhead) as f64 / ratio_hint).floor() as usize;
        (remaining / num_channels)
            .min(MAX_CALLBACK_FRAMES)
            .min(max_feed_frames)
            .saturating_mul(num_channels)
    }

    /// Feeds interleaved source samples through the varispeed input stage
    /// into `input_ring`, recording a resampled↔source checkpoint for the
    /// block. Returns the number of interleaved resampled samples pushed.
    ///
    /// The caller must have bounded `input` via
    /// [`Self::varispeed_take_samples`] so the worst-case emission fits both
    /// the scratch buffers and ring availability.
    fn feed_varispeed_to_ring(
        &mut self,
        input: &[f32],
        num_channels: usize,
    ) -> Result<usize, StretchError> {
        let frames = input.len() / num_channels;
        if frames == 0 {
            return Ok(0);
        }
        if !self.varispeed_engaged {
            // The ring is empty at engagement: resync the emission timeline
            // to the consumption cursor (it may have advanced across earlier
            // streams or flush padding) and drop an anchor checkpoint.
            self.varispeed_engaged = true;
            self.varispeed_resampled_emitted_total = self.input_frames_consumed_total as u64;
            self.ratio_map.push(
                self.varispeed_resampled_emitted_total,
                self.varispeed_map_base_source,
                self.varispeed_prev_ratio,
            );
        }

        for ch in 0..num_channels {
            let buf = &mut self.varispeed_input_buffers[ch];
            buf.clear();
            buf.extend(input.iter().skip(ch).step_by(num_channels));
        }

        // The tempo axis has no control glide: the resampler retargets to
        // the current target each block (its internal one-block ramp only
        // suppresses zipper noise).
        let step = 1.0 / self.target_ratio;
        let mut emitted = usize::MAX;
        for ch in 0..num_channels {
            self.varispeed_resamplers[ch].process_into(
                &self.varispeed_input_buffers[ch],
                step,
                &mut self.varispeed_output_buffers[ch],
            )?;
            emitted = emitted.min(self.varispeed_output_buffers[ch].len());
        }
        if emitted == usize::MAX {
            emitted = 0;
        }
        debug_assert!(self.varispeed_output_buffers[..num_channels]
            .iter()
            .all(|b| b.len() == emitted));
        self.varispeed_prev_ratio = self.target_ratio;
        self.varispeed_source_fed_total = self.varispeed_source_fed_total.saturating_add(frames);

        self.push_varispeed_emission(emitted, num_channels)?;

        self.varispeed_resampled_emitted_total += emitted as u64;
        self.ratio_map
            .evict_before(self.input_frames_consumed_total as u64);
        self.ratio_map.push(
            self.varispeed_resampled_emitted_total,
            self.varispeed_map_base_source + self.varispeed_resamplers[0].next_output_source_pos(),
            self.target_ratio,
        );
        Ok(emitted * num_channels)
    }

    /// Interleaves the varispeed output buffers into `input_ring`.
    fn push_varispeed_emission(
        &mut self,
        emitted: usize,
        num_channels: usize,
    ) -> Result<(), StretchError> {
        let pushed_samples = emitted.saturating_mul(num_channels);
        if pushed_samples == 0 {
            return Ok(());
        }
        if pushed_samples > self.input_ring.available() {
            return Err(StretchError::BufferOverflow {
                buffer: "stream_input_ring",
                requested: pushed_samples,
                available: self.input_ring.available(),
            });
        }
        for i in 0..emitted {
            for ch in 0..num_channels {
                self.varispeed_interleave_scratch[i * num_channels + ch] =
                    self.varispeed_output_buffers[ch][i];
            }
        }
        let pushed = self
            .input_ring
            .push_slice(&self.varispeed_interleave_scratch[..pushed_samples]);
        if pushed != pushed_samples {
            return Err(StretchError::BufferOverflow {
                buffer: "stream_input_ring",
                requested: pushed_samples,
                available: pushed,
            });
        }
        Ok(())
    }

    /// Drains the varispeed stage's held lookahead into the ring as real
    /// content and drops a final source-exact checkpoint, disengaging the
    /// stage (the per-resampler flush resets it).
    fn flush_varispeed_to_ring(&mut self, num_channels: usize) -> Result<(), StretchError> {
        if !self.varispeed_engaged {
            return Ok(());
        }
        let num_channels = num_channels.max(1);
        let step = 1.0 / self.target_ratio;
        let mut emitted = usize::MAX;
        for ch in 0..num_channels {
            self.varispeed_resamplers[ch]
                .flush_into(step, &mut self.varispeed_output_buffers[ch])?;
            emitted = emitted.min(self.varispeed_output_buffers[ch].len());
        }
        if emitted == usize::MAX {
            emitted = 0;
        }
        self.push_varispeed_emission(emitted, num_channels)?;
        // The drained tail becomes rendered output like any other emission.
        self.expected_total_output_samples += (emitted * num_channels) as f64;

        self.varispeed_resampled_emitted_total += emitted as u64;
        // The drain feeds synthetic zeros, so clamp the final checkpoint to
        // the real source-fed total and fold it into the base the next
        // stream's checkpoints build on.
        self.varispeed_map_base_source = self.varispeed_source_fed_total as f64;
        self.ratio_map
            .evict_before(self.input_frames_consumed_total as u64);
        self.ratio_map.push(
            self.varispeed_resampled_emitted_total,
            self.varispeed_map_base_source,
            self.target_ratio,
        );
        self.varispeed_engaged = false;
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

        if total_frames < self.effective_fft() {
            return Ok(());
        }

        if require_min_latency {
            let min_frames =
                effective_min_frames(self.effective_fft(), self.target_processing_ratio());
            if total_frames < min_frames {
                return Ok(());
            }
        }

        self.collect_channel_inputs(total_frames, num_channels)?;
        self.encode_input_mid_side(num_channels);

        if self.control_path == ControlPath::VarispeedFirst {
            self.update_varispeed_transposition(total_frames);
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
                let ratio_distance = self.pv_stretch_distance();
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
            let ratio_distance = self.pv_stretch_distance();
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
                    // The overlay is spliced over PV output, so it must run
                    // at the PV's tempo-axis stretch: the glided tempo on
                    // the vocoder path, the transposition on the varispeed
                    // path (ring content is already tempo-resampled there).
                    let ratio = match self.control_path {
                        ControlPath::VocoderTempo => self.current_ratio,
                        ControlPath::VarispeedFirst => self.transposition,
                    };
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
            match self.engine {
                StreamingEngine::Deterministic => {
                    self.vocoders[ch].process_streaming_into(
                        &self.channel_input_buffers[ch],
                        &mut self.channel_output_buffers[ch],
                    )?;
                }
                StreamingEngine::MultiResolution => {
                    self.multi_res[ch].process_streaming_into(
                        &self.channel_input_buffers[ch],
                        &mut self.channel_output_buffers[ch],
                    )?;
                }
            }
            min_output_len = min_output_len.min(self.channel_output_buffers[ch].len());
        }

        Ok(if min_output_len == usize::MAX {
            0
        } else {
            min_output_len
        })
    }

    /// FFT size the active engine's gate and consumption math are keyed to.
    #[inline]
    fn effective_fft(&self) -> usize {
        effective_stream_fft(&self.params, self.engine)
    }

    /// Per-channel frames a render pass over `total_frames` will consume.
    ///
    /// Shared between `consume_processed_input` and artifact reset span math
    /// so the two cannot drift. Engine-aware: the multi-resolution engine's
    /// cadence is keyed to its sub-bass band.
    #[inline]
    fn frames_consumed_for(&self, total_frames: usize) -> usize {
        match self.engine {
            StreamingEngine::Deterministic => {
                let hop = self.params.hop_size;
                if hop == 0 || total_frames < self.params.fft_size {
                    return 0;
                }
                ((total_frames - self.params.fft_size) / hop + 1) * hop
            }
            StreamingEngine::MultiResolution => self
                .multi_res
                .first()
                .map(|m| m.streaming_frames_consumed(total_frames))
                .unwrap_or(0),
        }
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

    /// Ratio the PV renders at: the glided tempo on the vocoder-tempo path,
    /// the delay-matched transposition correction on the varispeed-first
    /// path (the tempo itself lives on the input resampler there), times the
    /// glided user pitch on both.
    #[inline]
    fn processing_ratio(&self) -> f64 {
        match self.control_path {
            ControlPath::VocoderTempo => self.current_ratio * self.pitch_scale,
            ControlPath::VarispeedFirst => self.transposition * self.pitch_scale,
        }
    }

    /// Distance of the PV's tempo-axis stretch from unity; drives corrective
    /// heuristics (spectral shelves, blend scaling, overlay sizing) that
    /// track PV artifact severity.
    #[inline]
    fn pv_stretch_distance(&self) -> f64 {
        match self.control_path {
            ControlPath::VocoderTempo => (self.current_ratio - 1.0).abs(),
            ControlPath::VarispeedFirst => (self.transposition - 1.0).abs(),
        }
    }

    /// Step of the post-PV output resampler: how many PV output samples are
    /// consumed per emitted sample.
    ///
    /// The vocoder-tempo path resamples away only the pitch over-stretch;
    /// the varispeed-first path resamples away the PV's entire over-stretch
    /// (transposition correction times user pitch), restoring the varispeed
    /// emission's length 1:1.
    #[inline]
    fn post_resample_step(&self) -> f64 {
        match self.control_path {
            ControlPath::VocoderTempo => self.pitch_scale,
            ControlPath::VarispeedFirst => self.processing_ratio(),
        }
    }

    /// [`Self::post_resample_step`] at the current control targets.
    #[inline]
    fn target_post_resample_step(&self) -> f64 {
        match self.control_path {
            ControlPath::VocoderTempo => self.target_pitch_scale,
            ControlPath::VarispeedFirst => self.target_processing_ratio(),
        }
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
    ///
    /// On the varispeed-first path the in-flight delta is the gap between
    /// the delay-matched transposition and its target, which settles after
    /// the gate transit rather than the 50 ms EMA; the τ-based estimate is
    /// then an approximation (slightly conservative for a hold heuristic,
    /// and the delta is re-evaluated on every scheduler pass anyway).
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

    /// Clears the varispeed input stage back to disengaged, zeroing its
    /// timeline bookkeeping (used by seek/reset flows that restart the
    /// source timeline).
    fn reset_varispeed_stage(&mut self) {
        for resampler in &mut self.varispeed_resamplers {
            resampler.reset();
        }
        for buf in &mut self.varispeed_output_buffers {
            buf.clear();
        }
        self.ratio_map.clear();
        self.varispeed_engaged = false;
        self.varispeed_source_fed_total = 0;
        self.varispeed_resampled_emitted_total = 0;
        self.varispeed_map_base_source = 0.0;
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

        // The resampler stays engaged once its step has been off unity this
        // stream: at unity it degenerates to a bit-clean passthrough, which
        // avoids splicing its held lookahead out of the stream when the step
        // returns to 1.0.
        let step = self.post_resample_step();
        let pitch_active = self.pitch_resampler_engaged
            || (step - 1.0).abs() >= RATIO_SNAP_THRESHOLD
            || (self.target_post_resample_step() - 1.0).abs() >= RATIO_SNAP_THRESHOLD;
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
                    // The sinc resampler consumes the source at `step`
                    // samples per output sample and ramps toward it from the
                    // previous block's step internally.
                    self.sinc_pitch_resamplers[ch].process_into(
                        &self.channel_output_buffers[ch][..min_output_len],
                        step,
                        &mut self.pitch_output_buffers[ch],
                    )?;
                }
                StreamPitchQuality::Linear => {
                    self.pitch_resamplers[ch].process_into(
                        &self.channel_output_buffers[ch][..min_output_len],
                        1.0 / step,
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

        let step = self.post_resample_step();
        let mut min_output_len = usize::MAX;
        for ch in 0..num_channels {
            match self.pitch_quality {
                StreamPitchQuality::Sinc => {
                    self.sinc_pitch_resamplers[ch]
                        .flush_into(step, &mut self.pitch_output_buffers[ch])?;
                }
                StreamPitchQuality::Linear => {
                    self.pitch_resamplers[ch]
                        .flush_into(1.0 / step, &mut self.pitch_output_buffers[ch])?;
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

    // TODO(stage-1 follow-up): tails drained here (and by the pitch
    // resampler flush) skip the energy-gain/shelf corrections that normal
    // chunks receive in `process_available_to_pending`, leaving a small gain
    // seam at the flush boundary at far-from-unity ratios.
    fn flush_vocoder_tails_to_pending(&mut self, num_channels: usize) -> Result<(), StretchError> {
        let mut min_output_len = usize::MAX;
        for ch in 0..num_channels {
            match self.engine {
                StreamingEngine::Deterministic => {
                    self.vocoders[ch].flush_streaming_into(&mut self.channel_output_buffers[ch])?;
                }
                StreamingEngine::MultiResolution => {
                    self.multi_res[ch]
                        .flush_streaming_into(&mut self.channel_output_buffers[ch])?;
                }
            }
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
    ///
    /// On the [`ControlPath::VarispeedFirst`] path the ratio is additionally
    /// bounded to the varispeed range `[0.25, 4.0]`.
    pub fn try_set_stretch_ratio(&mut self, ratio: f64) -> Result<(), StretchError> {
        let ratio = validate_positive_finite_ratio(ratio, "stretch ratio")?;
        if self.control_path == ControlPath::VarispeedFirst {
            validate_varispeed_ratio(ratio)?;
        }
        self.target_ratio = ratio;
        Ok(())
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

    /// Selects the stream-mode rendering engine.
    ///
    /// Not RT-safe: switching engines re-sizes buffers for the new engine's
    /// buffering gate and starts a fresh stream (buffered input/output is
    /// dropped; ratio/pitch targets, resampler quality, tempo state, and
    /// the pre-analysis artifact carry over). Select the engine at
    /// build/rebuild time, before streaming — never from an audio callback.
    ///
    /// # Errors
    ///
    /// Returns [`StretchError::InvalidFormat`] when
    /// [`StreamingEngine::MultiResolution`] is selected with a mid-band FFT
    /// below 2048 (e.g. the Live profile): the engine's sub-bass band would
    /// still gate at ~139 ms, defeating the profile's latency budget. Use
    /// the Club or Quality profile instead.
    pub fn set_streaming_engine(&mut self, engine: StreamingEngine) -> Result<(), StretchError> {
        if engine == self.engine {
            return Ok(());
        }
        if engine == StreamingEngine::MultiResolution
            && self.params.fft_size < MULTI_RES_MIN_MID_FFT
        {
            return Err(StretchError::InvalidFormat(format!(
                "multi-resolution streaming needs fft_size >= {} (Club/Quality profiles), got {}",
                MULTI_RES_MIN_MID_FFT, self.params.fft_size
            )));
        }

        // Rebuild at the current control targets so engine, buffers, and
        // stretcher ratios all agree from the first render.
        let params = self.params.clone().with_stretch_ratio(self.target_ratio);
        let mut rebuilt = Self::new_with_engine_and_path(params, engine, self.control_path);
        rebuilt.source_bpm = self.source_bpm;
        rebuilt.target_pitch_scale = self.target_pitch_scale;
        rebuilt.pitch_scale = self.target_pitch_scale;
        rebuilt.pitch_quality = self.pitch_quality;
        *self = rebuilt;
        Ok(())
    }

    /// Returns the active stream-mode rendering engine.
    pub fn streaming_engine(&self) -> StreamingEngine {
        self.engine
    }

    /// Selects the tempo control path.
    ///
    /// Not RT-safe: switching paths re-sizes buffers for the new path's
    /// per-callback emission headroom and starts a fresh stream (buffered
    /// input/output is dropped; ratio/pitch targets, resampler quality,
    /// engine, tempo state, and the pre-analysis artifact carry over).
    /// Select the path at build/rebuild time — never from an audio callback.
    ///
    /// # Errors
    ///
    /// Returns [`StretchError::InvalidRatio`] when switching to
    /// [`ControlPath::VarispeedFirst`] while the target stretch ratio is
    /// outside the varispeed range `[0.25, 4.0]`.
    pub fn set_control_path(&mut self, control_path: ControlPath) -> Result<(), StretchError> {
        if control_path == self.control_path {
            return Ok(());
        }
        if control_path == ControlPath::VarispeedFirst {
            validate_varispeed_ratio(self.target_ratio)?;
        }

        // Rebuild at the current control targets so path, buffers, and
        // stretcher ratios all agree from the first render.
        let params = self.params.clone().with_stretch_ratio(self.target_ratio);
        let mut rebuilt = Self::new_with_engine_and_path(params, self.engine, control_path);
        rebuilt.source_bpm = self.source_bpm;
        rebuilt.target_pitch_scale = self.target_pitch_scale;
        rebuilt.pitch_scale = self.target_pitch_scale;
        rebuilt.pitch_quality = self.pitch_quality;
        *self = rebuilt;
        Ok(())
    }

    /// Returns the active tempo control path.
    pub fn control_path(&self) -> ControlPath {
        self.control_path
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
        let base_gate_frames = min_latency_frames(self.effective_fft());
        let effective_gate_frames =
            effective_min_frames(self.effective_fft(), self.target_processing_ratio());

        // Output-resampler lookahead applies once the resampler is (or is
        // about to be) in the signal path: engaged resamplers stay engaged
        // even at a unity step, and a non-unity target engages on the next
        // process call. On the varispeed-first path the step is the whole
        // keylock correction, so any off-unity tempo target engages it.
        let pitch_active = self.pitch_resampler_engaged
            || (self.target_post_resample_step() - 1.0).abs() > RATIO_SNAP_THRESHOLD
            || (self.post_resample_step() - 1.0).abs() > RATIO_SNAP_THRESHOLD;
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

        let varispeed_active = self.varispeed_engaged
            || (self.control_path == ControlPath::VarispeedFirst
                && ((self.target_ratio - 1.0).abs() > RATIO_SNAP_THRESHOLD
                    || (self.target_pitch_scale - 1.0).abs() > RATIO_SNAP_THRESHOLD));
        let varispeed_lookahead_samples = if !varispeed_active {
            0
        } else {
            self.varispeed_resamplers
                .first()
                .map(|r| r.current_half_span())
                .unwrap_or(0)
        };

        let pipeline_delay_frames =
            effective_gate_frames + pitch_lookahead_samples + varispeed_lookahead_samples;
        let control_to_audio_frames = match self.control_path {
            // The PV implements tempo: a change becomes audible only after
            // the gate (plus the glide, reported separately).
            ControlPath::VocoderTempo => pipeline_delay_frames,
            // The varispeed input stage implements tempo sample-accurately;
            // only its kernel lookahead (and the caller's callback size,
            // which the caller owns) stands between control and audio.
            ControlPath::VarispeedFirst => varispeed_lookahead_samples,
        };

        StreamLatencyReport {
            base_gate_frames,
            effective_gate_frames,
            pitch_lookahead_samples,
            varispeed_lookahead_samples,
            control_smoothing_secs: RATIO_SMOOTHING_TIME_SECS,
            sample_rate: self.params.sample_rate,
            control_to_audio_frames,
            pipeline_delay_frames,
            control_path: self.control_path,
            total_frames: pipeline_delay_frames,
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
        self.transposition = self.params.stretch_ratio;
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

        match self.engine {
            StreamingEngine::Deterministic => {
                self.vocoders = Self::create_vocoders(
                    &self.params,
                    self.params.stretch_ratio,
                    stream_capacity_frames(&self.params, self.engine, self.control_path),
                );
                if self.control_path == ControlPath::VarispeedFirst {
                    for pv in &mut self.vocoders {
                        pv.set_smooth_ratio_updates(true);
                    }
                }
            }
            StreamingEngine::MultiResolution => {
                for stretcher in &mut self.multi_res {
                    stretcher.reset_streaming_state();
                    stretcher.set_stretch_ratio(self.params.stretch_ratio);
                }
            }
        }
        self.expected_total_output_samples = 0.0;
        self.total_output_emitted_samples = 0;
        self.pitch_scale = 1.0;
        self.target_pitch_scale = 1.0;
        self.reset_pitch_resamplers();
        self.reset_varispeed_stage();
        self.varispeed_prev_ratio = self.params.stretch_ratio;
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

    /// Delay-matched transposition tracking for the varispeed-first path.
    ///
    /// The varispeed stage changes pitch instantly at the INPUT, but the PV
    /// consumes that audio a buffering gate later. Correcting at the current
    /// control ratio would therefore mis-pitch every render while a tempo
    /// change transits the gate (audible wobble under fast rides). Instead,
    /// the correction target is the embedded ratio at the END of the span
    /// the PV is about to consume, read off the ratio map. The post-PV sinc
    /// resampler ramps its step from the previous target across each block,
    /// so consecutive end-of-span targets make the applied correction trace
    /// the embedded piecewise-linear ratio with no average lag.
    ///
    /// Residual wobble under fast rides (~10 cents p95 on a pure tone at
    /// the qa ±8%/2 s torture ride, versus ~100 on the vocoder path) comes
    /// from the PV's own overlap-add under a moving ratio, not from this
    /// estimate — better instantaneous-frequency tracking (ROADMAP Stage
    /// 16) is the lever for it.
    fn update_varispeed_transposition(&mut self, total_frames: usize) {
        let span_len = self.frames_consumed_for(total_frames);
        if span_len == 0 {
            return;
        }
        let end = self.input_frames_consumed_total as f64 + span_len as f64;
        let Some(ratio) = self.ratio_map.ratio_at(end) else {
            return;
        };
        self.transposition = ratio.clamp(VARISPEED_MIN_RATIO, VARISPEED_MAX_RATIO);
    }

    fn update_vocoder_ratio(&mut self) {
        let processing_ratio = self.processing_ratio();
        if (processing_ratio - self.vocoder_ratio).abs() > RATIO_SNAP_THRESHOLD {
            for voc in &mut self.vocoders {
                voc.set_stretch_ratio(processing_ratio);
            }
            for stretcher in &mut self.multi_res {
                stretcher.set_stretch_ratio(processing_ratio);
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
        // Use channel 0's flux (mid channel for stereo M/S). For the
        // multi-resolution engine this is the combined three-band report,
        // normalized against the summed band spectra.
        let (flux, num_bins) = match self.engine {
            StreamingEngine::Deterministic => (
                self.vocoders
                    .first()
                    .and_then(|v| v.last_frame_flux().copied()),
                self.params.fft_size / 2 + 1,
            ),
            StreamingEngine::MultiResolution => (
                self.multi_res.first().and_then(|m| m.last_frame_flux()),
                self.multi_res
                    .first()
                    .map(|m| m.total_spectral_bins())
                    .unwrap_or(0),
            ),
        };
        let flux = match flux {
            Some(f) => f,
            None => return 1.0,
        };

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
        if total_frames < self.effective_fft() {
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

        self.apply_reset_mask_to_engine(reset_mask, num_channels);
    }

    /// Routes a scheduler reset mask to the active engine's per-channel
    /// stretchers, applying the Mid/Side split policy when applicable.
    ///
    /// The multi-resolution path forwards the mask through
    /// [`MultiResolutionStretcher::reset_phase_state_bands`], which applies
    /// it to every band vocoder by absolute bin frequency — the same
    /// 100/500/4000 Hz band edges the scheduler routes by, independent of
    /// the filterbank's 200/4000 Hz crossover points.
    fn apply_reset_mask_to_engine(&mut self, reset_mask: [bool; 4], num_channels: usize) {
        let sample_rate = self.params.sample_rate;
        let mid_side = num_channels == 2 && self.params.stereo_mode == StereoMode::MidSide;
        match self.engine {
            StreamingEngine::Deterministic => {
                if mid_side && self.vocoders.len() == 2 {
                    let (mid_mask, side_mask) = stereo_channel_reset_masks(reset_mask);
                    self.vocoders[0].reset_phase_state_bands(mid_mask, sample_rate);
                    if side_mask.iter().any(|&b| b) {
                        self.vocoders[1].reset_phase_state_bands(side_mask, sample_rate);
                    }
                    return;
                }
                for vocoder in &mut self.vocoders {
                    vocoder.reset_phase_state_bands(reset_mask, sample_rate);
                }
            }
            StreamingEngine::MultiResolution => {
                if mid_side && self.multi_res.len() == 2 {
                    let (mid_mask, side_mask) = stereo_channel_reset_masks(reset_mask);
                    self.multi_res[0].reset_phase_state_bands(mid_mask, sample_rate);
                    if side_mask.iter().any(|&b| b) {
                        self.multi_res[1].reset_phase_state_bands(side_mask, sample_rate);
                    }
                    return;
                }
                for stretcher in &mut self.multi_res {
                    stretcher.reset_phase_state_bands(reset_mask, sample_rate);
                }
            }
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
        if total_frames < self.effective_fft() {
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
        // `span_len` is in ring (resampled) frames; onset positions are
        // source frames, so the span end maps through the varispeed
        // timeline on the varispeed-first path.
        let span_end = match self.control_path {
            ControlPath::VocoderTempo => span_start.saturating_add(span_len),
            ControlPath::VarispeedFirst => {
                let end_resampled = (self.input_frames_consumed_total + span_len) as f64;
                let end_source = self
                    .ratio_map
                    .map_to_source(end_resampled)
                    .unwrap_or(end_resampled);
                self.source_start_frames
                    .saturating_add(self.passthrough_frames_total)
                    .saturating_add(end_source.floor() as usize)
            }
        };

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

        self.apply_reset_mask_to_engine(mask, num_channels);
    }

    /// Returns the BPM stored in the params, if any.
    pub fn bpm(&self) -> Option<f64> {
        self.params.bpm
    }

    /// Absolute source frame of the next input frame awaiting consumption.
    ///
    /// This is the timeline pre-analysis artifact positions are compared
    /// against: `source start + passthrough frames + DSP-consumed frames`,
    /// with the consumption cursor mapped from ring (resampled) frames back
    /// to source frames when the varispeed stage leads the PV.
    #[inline]
    fn ring_start_abs(&self) -> usize {
        self.source_start_frames
            .saturating_add(self.passthrough_frames_total)
            .saturating_add(self.consumed_source_frames())
    }

    /// Source frames the DSP path has consumed: the ring consumption cursor,
    /// mapped through the varispeed timeline on the varispeed-first path.
    #[inline]
    fn consumed_source_frames(&self) -> usize {
        match self.control_path {
            ControlPath::VocoderTempo => self.input_frames_consumed_total,
            ControlPath::VarispeedFirst => self
                .ratio_map
                .map_to_source(self.input_frames_consumed_total as f64)
                .unwrap_or(self.input_frames_consumed_total as f64)
                .floor() as usize,
        }
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
        // timeline so the NEXT pushed frame maps to `next_frame`. Everything
        // already accepted must be counted in SOURCE frames: on the
        // varispeed-first path that is the source-fed total (ring content is
        // resampled and the resampler holds fractional lookahead).
        let already_counted = match self.control_path {
            ControlPath::VocoderTempo => {
                let num_channels = self.params.channels.count().max(1);
                let in_flight = self.input_ring.len() / num_channels;
                self.passthrough_frames_total
                    .saturating_add(self.input_frames_consumed_total)
                    .saturating_add(in_flight)
            }
            ControlPath::VarispeedFirst => self
                .passthrough_frames_total
                .saturating_add(self.varispeed_source_fed_total),
        };
        self.source_start_frames = next_frame.saturating_sub(already_counted);
        self.reposition_artifact_cursor();
    }

    /// Frames of preceding audio [`Self::warm_start_seek`] wants as preroll.
    ///
    /// Enough to clear the input-buffering gate plus one extra FFT window
    /// of convergence margin for the phase vocoder's overlap and the pitch
    /// resampler's history. Bounded, so rapid cue jumps stay cheap.
    pub fn warm_start_preroll_frames(&self) -> usize {
        let ring_frames =
            effective_min_frames(self.effective_fft(), self.target_processing_ratio())
                .saturating_add(self.effective_fft());
        match self.control_path {
            ControlPath::VocoderTempo => ring_frames,
            ControlPath::VarispeedFirst => {
                // The gate is in resampled frames: speeding up (ratio < 1)
                // needs proportionally more source to fill it, and the
                // varispeed kernel holds back its lookahead.
                let scale = (1.0 / self.target_ratio).max(1.0);
                (ring_frames as f64 * scale).ceil() as usize + STREAM_SINC_MAX_HALF_TAPS
            }
        }
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
    /// free after warmup. CPU cost is bounded by the preroll length.
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
        for stretcher in &mut self.multi_res {
            stretcher.reset_streaming_state();
        }
        self.transient_scheduler.reset();
        self.reset_pitch_resamplers();
        self.reset_varispeed_stage();
        self.wsola_overlay_remaining = 0;
        self.wsola_overlay_total = 0;
        self.wsola_overlay_pos = 0;
        for buf in &mut self.wsola_overlay_buffers {
            buf.clear();
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

        // Prime: run the preroll through the full DSP path — varispeed
        // stage, phase vocoder, schedulers, pitch resamplers, gain tracking
        // — discarding the rendered output. Control glides do NOT advance:
        // the jump is instantaneous, so no wall-clock time passes.
        self.dsp_engaged = true;
        let chunk_samples = MAX_CALLBACK_FRAMES * num_channels;
        for chunk in preroll.chunks(chunk_samples) {
            match self.control_path {
                ControlPath::VocoderTempo => self.push_input_samples(chunk)?,
                ControlPath::VarispeedFirst => {
                    self.feed_varispeed_to_ring(chunk, num_channels)?;
                }
            }
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

        self.artifact_active = usable.is_some();
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
    ///
    /// On the varispeed-first path the tempo axis has no glide: the ratio
    /// snaps to its target (the input resampler retargets sample-accurately
    /// and the PV correction is delay-matched separately); only user pitch
    /// keeps the smoothing.
    fn interpolate_ratio_for_frames(&mut self, frames: usize) {
        let tau_frames = (self.params.sample_rate as f64 * RATIO_SMOOTHING_TIME_SECS).max(1.0);
        let alpha = 1.0 - (-(frames as f64) / tau_frames).exp();
        match self.control_path {
            ControlPath::VocoderTempo => {
                self.current_ratio += alpha * (self.target_ratio - self.current_ratio);
                if (self.current_ratio - self.target_ratio).abs() < RATIO_SNAP_THRESHOLD {
                    self.current_ratio = self.target_ratio;
                }
            }
            ControlPath::VarispeedFirst => {
                self.current_ratio = self.target_ratio;
            }
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
    fn test_streaming_engine_round_trip() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        assert_eq!(proc.streaming_engine(), StreamingEngine::Deterministic);
        proc.set_streaming_engine(StreamingEngine::Deterministic)
            .expect("selecting the active engine is a no-op");
        assert_eq!(proc.streaming_engine(), StreamingEngine::Deterministic);

        proc.set_streaming_engine(StreamingEngine::MultiResolution)
            .expect("default 4096 FFT supports multi-resolution");
        assert_eq!(proc.streaming_engine(), StreamingEngine::MultiResolution);
        proc.set_streaming_engine(StreamingEngine::Deterministic)
            .expect("switch back");
        assert_eq!(proc.streaming_engine(), StreamingEngine::Deterministic);
    }

    #[test]
    fn test_streaming_engine_multi_res_rejected_at_live_profile() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(2)
            .with_stream_profile(crate::core::types::StreamProfile::Live);

        let mut proc = StreamProcessor::new(params);
        let err = proc
            .set_streaming_engine(StreamingEngine::MultiResolution)
            .expect_err("Live profile (1024 FFT) must reject multi-resolution");
        assert!(matches!(err, StretchError::InvalidFormat(_)));
        // The failed selection must leave the active engine untouched.
        assert_eq!(proc.streaming_engine(), StreamingEngine::Deterministic);
    }

    #[test]
    fn test_stream_processor_multi_res_basic() {
        let params = StretchParams::new(1.2)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_stream_profile(crate::core::types::StreamProfile::Club);

        let mut proc = StreamProcessor::new(params);
        proc.set_streaming_engine(StreamingEngine::MultiResolution)
            .expect("Club profile supports multi-resolution");

        // Long enough to clear the sub-bass gate several times over.
        let total = 44100 * 4;
        let signal: Vec<f32> = (0..total)
            .map(|i| {
                let t = i as f32 / 44100.0;
                0.4 * (2.0 * PI * 55.0 * t).sin() + 0.3 * (2.0 * PI * 880.0 * t).sin()
            })
            .collect();

        let mut output = Vec::new();
        for chunk in signal.chunks(1024) {
            let rendered = proc.process(chunk).expect("multi-res stream chunk");
            output.extend_from_slice(&rendered);
        }
        let tail = proc.flush().expect("multi-res flush");
        output.extend_from_slice(&tail);

        let expected = (total as f64 * 1.2) as usize;
        assert!(
            output.len().abs_diff(expected) < expected / 10,
            "multi-res output length {} too far from expected {}",
            output.len(),
            expected
        );
        assert!(
            output.iter().all(|s| s.is_finite()),
            "multi-res output contains non-finite samples"
        );
        let peak = output.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(
            (0.1..4.0).contains(&peak),
            "multi-res output peak {peak} outside sane range"
        );
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

    #[test]
    fn test_ratio_map_fifo_interpolation_and_eviction() {
        let mut map = RatioMapFifo::with_capacity(4);
        assert_eq!(map.map_to_source(10.0), None);
        assert_eq!(map.ratio_at(10.0), None);

        map.push(0, 0.0, 1.25);
        map.push(100, 80.0, 1.25); // ratio 1.25 block
        map.push(200, 180.0, 1.0); // block ramping down to unity
        assert_eq!(map.map_to_source(0.0), Some(0.0));
        assert_eq!(map.map_to_source(50.0), Some(40.0));
        assert_eq!(map.map_to_source(150.0), Some(130.0));
        // Beyond the final checkpoint clamps (flush padding).
        assert_eq!(map.map_to_source(500.0), Some(180.0));
        // Instantaneous ratio interpolates the per-block retarget ramp.
        assert_eq!(map.ratio_at(50.0), Some(1.25));
        assert_eq!(map.ratio_at(150.0), Some(1.125));
        assert_eq!(map.ratio_at(500.0), Some(1.0));

        // Eviction retains one anchor at or behind the cursor.
        map.evict_before(120);
        assert_eq!(map.map_to_source(150.0), Some(130.0));
        assert_eq!(map.map_to_source(0.0), Some(80.0), "behind anchor clamps");

        // Overflow merges into the newest entry instead of growing.
        let mut small = RatioMapFifo::with_capacity(2);
        small.push(0, 0.0, 1.0);
        small.push(100, 100.0, 1.0);
        small.push(200, 150.0, 1.1);
        assert_eq!(small.map_to_source(200.0), Some(150.0));
        assert_eq!(small.ratio_at(200.0), Some(1.1));
    }

    #[test]
    fn test_varispeed_control_path_selection_and_ratio_bounds() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params.clone());
        assert_eq!(proc.control_path(), ControlPath::VocoderTempo);

        proc.set_control_path(ControlPath::VarispeedFirst).unwrap();
        assert_eq!(proc.control_path(), ControlPath::VarispeedFirst);

        // The varispeed range is enforced on the ratio setter...
        assert!(proc.set_stretch_ratio(5.0).is_err());
        assert!(proc.set_stretch_ratio(0.2).is_err());
        assert!(proc.set_stretch_ratio(1.08).is_ok());

        // ...and on the path switch itself.
        let mut wide = StreamProcessor::new(params.with_stretch_ratio(6.0));
        wide.set_stretch_ratio(6.0).unwrap();
        assert!(wide.set_control_path(ControlPath::VarispeedFirst).is_err());
        assert_eq!(wide.control_path(), ControlPath::VocoderTempo);
    }

    #[test]
    fn test_varispeed_unity_stays_bit_exact_passthrough() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_control_path(ControlPath::VarispeedFirst).unwrap();

        let input: Vec<f32> = (0..4096)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let mut output = Vec::with_capacity(8192);
        for chunk in input.chunks(512) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        assert_eq!(output, input, "unity varispeed must stay bit-exact");
        assert_eq!(proc.source_position(), 4096);
    }

    #[test]
    fn test_varispeed_round_trip_length_and_keylock_pitch() {
        let ratio = 1.25;
        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_control_path(ControlPath::VarispeedFirst).unwrap();

        let n = 44100usize;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let mut output: Vec<f32> = Vec::with_capacity(n * 2 + 65_536);
        for chunk in input.chunks(512) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        proc.flush_into(&mut output).unwrap();

        let expected = (n as f64 * ratio) as usize;
        let delta = output.len().abs_diff(expected);
        assert!(
            delta <= 128,
            "round-trip length {} not within 128 of {}",
            output.len(),
            expected
        );

        // Keylock: the varispeed drop (440 -> 352 Hz) must be corrected back.
        let mid = &output[output.len() / 4..output.len() * 3 / 4];
        let freq = estimate_freq_zero_crossings(mid, 44100);
        assert!(
            (freq - 440.0).abs() < 15.0,
            "keylock pitch off: {:.1} Hz vs 440 Hz",
            freq
        );

        // Post-flush the source timeline lands exactly on the fed total.
        assert_eq!(proc.source_position(), n);
    }

    #[test]
    fn test_varispeed_speedup_round_trip_length() {
        let ratio = 0.8;
        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_control_path(ControlPath::VarispeedFirst).unwrap();

        let n = 44100usize;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 330.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let mut output: Vec<f32> = Vec::with_capacity(n * 2 + 65_536);
        for chunk in input.chunks(512) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        proc.flush_into(&mut output).unwrap();

        let expected = (n as f64 * ratio) as usize;
        let delta = output.len().abs_diff(expected);
        assert!(
            delta <= 128,
            "speed-up length {} not within 128 of {}",
            output.len(),
            expected
        );
        let mid = &output[output.len() / 4..output.len() * 3 / 4];
        let freq = estimate_freq_zero_crossings(mid, 44100);
        assert!(
            (freq - 330.0).abs() < 12.0,
            "keylock pitch off: {:.1} Hz vs 330 Hz",
            freq
        );
    }

    #[test]
    fn test_varispeed_source_jump_keeps_source_position_aligned() {
        let params = StretchParams::new(1.05)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_control_path(ControlPath::VarispeedFirst).unwrap();

        let sine = |i: usize| (2.0 * PI * 220.0 * i as f32 / 44100.0).sin() * 0.5;
        let a: Vec<f32> = (0..8192).map(sine).collect();
        let b: Vec<f32> = (0..4096).map(sine).collect();
        let mut output: Vec<f32> = Vec::with_capacity(65_536);
        for chunk in a.chunks(512) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        proc.notify_source_jump(100_000);
        for chunk in b.chunks(512) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        proc.flush_into(&mut output).unwrap();
        assert_eq!(
            proc.source_position(),
            100_000 + 4096,
            "post-jump source position must land on jump target + fed frames"
        );
    }

    #[test]
    fn test_varispeed_transposition_converges_to_tempo_ratio() {
        let ratio = 1.08;
        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_control_path(ControlPath::VarispeedFirst).unwrap();

        let input: Vec<f32> = (0..66_150)
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let mut output = Vec::with_capacity(input.len() * 2 + 65_536);
        for chunk in input.chunks(512) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        assert!(
            (proc.transposition - ratio).abs() < 0.002,
            "transposition {} should converge to the tempo ratio {}",
            proc.transposition,
            ratio
        );
    }

    #[test]
    fn test_varispeed_modulation_hold_tracks_tempo_step() {
        let params = StretchParams::new(1.02)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_control_path(ControlPath::VarispeedFirst).unwrap();

        let sine = |i: usize| (2.0 * PI * 220.0 * i as f32 / 44100.0).sin() * 0.5;
        let warmup: Vec<f32> = (0..44_100).map(sine).collect();
        let mut output = Vec::with_capacity(warmup.len() * 2 + 65_536);
        for chunk in warmup.chunks(512) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        assert_eq!(
            proc.modulation_hold_overlap_windows(),
            0,
            "steady state must not hold"
        );

        // A tempo step engages the hold until the retargeted audio has
        // transited the gate and the transposition catches up.
        proc.set_stretch_ratio(1.08).unwrap();
        assert!(
            proc.modulation_hold_overlap_windows() >= 1,
            "tempo step must engage modulation hold on the varispeed path"
        );

        let settle: Vec<f32> = (0..44_100).map(sine).collect();
        for chunk in settle.chunks(512) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        assert_eq!(
            proc.modulation_hold_overlap_windows(),
            0,
            "hold must release after the step settles"
        );
    }

    #[test]
    fn test_varispeed_warm_start_seek_resumes_converged() {
        let params = StretchParams::new(1.05)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut proc = StreamProcessor::new(params);
        proc.set_control_path(ControlPath::VarispeedFirst).unwrap();

        let sine: Vec<f32> = (0..131_072)
            .map(|i| (2.0 * PI * 220.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let target = 88_200usize;
        let preroll_frames = proc.warm_start_preroll_frames().min(target);
        let preroll = &sine[target - preroll_frames..target];
        proc.warm_start_seek(target, preroll).unwrap();

        let mut output: Vec<f32> = Vec::with_capacity(65_536);
        for chunk in sine[target..target + 8192].chunks(512) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        assert!(
            !output.is_empty(),
            "warm-started varispeed stream must produce output promptly"
        );
        let pos = proc.source_position();
        assert!(
            pos >= target - preroll_frames && pos <= target + 8192,
            "source position {} outside plausible window around target {}",
            pos,
            target
        );
    }
}

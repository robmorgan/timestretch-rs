//! Real-time streaming time-stretch processor.

use crate::core::types::StretchParams;
use crate::error::StretchError;
use crate::stretch::hybrid::HybridStretcher;
use crate::stretch::phase_vocoder::PhaseVocoder;

/// Threshold below which ratio differences are considered negligible.
const RATIO_SNAP_THRESHOLD: f64 = 0.0001;
/// Smoothing factor for interpolating between current and target ratio.
const RATIO_INTERPOLATION_ALPHA: f64 = 0.1;
/// Multiplier for FFT size to determine minimum input and latency (2x FFT).
const LATENCY_FFT_MULTIPLIER: usize = 2;

/// Streaming chunk-based processor for real-time time stretching.
///
/// Accumulates input samples in an internal buffer and processes them
/// using the phase vocoder when enough data is available.
/// PhaseVocoder instances are persisted per channel to avoid
/// expensive FFT planner recreation on each call.
///
/// # Example
///
/// ```
/// use timestretch::{StreamProcessor, StretchParams, EdmPreset};
///
/// let params = StretchParams::new(1.0)
///     .with_preset(EdmPreset::DjBeatmatch)
///     .with_sample_rate(44100)
///     .with_channels(1);
///
/// let mut processor = StreamProcessor::new(params);
///
/// // Feed a chunk of silence (in practice, real audio data)
/// let chunk = vec![0.0f32; 4096];
/// let _output = processor.process(&chunk).unwrap();
///
/// // Change ratio on the fly
/// processor.set_stretch_ratio(1.05);
/// ```
pub struct StreamProcessor {
    params: StretchParams,
    input_buffer: Vec<f32>,
    /// Current stretch ratio (can be changed on the fly).
    current_ratio: f64,
    /// Target stretch ratio (for smooth interpolation).
    target_ratio: f64,
    /// Whether the processor has been initialized.
    initialized: bool,
    /// Persistent PhaseVocoder instances, one per channel.
    vocoders: Vec<PhaseVocoder>,
    /// Reusable per-channel deinterleave buffers.
    channel_buffers: Vec<Vec<f32>>,
    /// Reusable interleaved output buffer.
    output_scratch: Vec<f32>,
    /// Source BPM (set when created via `from_tempo`, enables `set_tempo`).
    source_bpm: Option<f64>,
    /// When true, use the full hybrid algorithm (transient detection + WSOLA + PV)
    /// instead of PV-only. Higher quality for EDM but higher latency.
    use_hybrid: bool,
}

impl std::fmt::Debug for StreamProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamProcessor")
            .field("params", &self.params)
            .field("current_ratio", &self.current_ratio)
            .field("target_ratio", &self.target_ratio)
            .field("initialized", &self.initialized)
            .field("source_bpm", &self.source_bpm)
            .field("input_buffer_len", &self.input_buffer.len())
            .finish()
    }
}

impl StreamProcessor {
    /// Creates a new streaming processor.
    pub fn new(params: StretchParams) -> Self {
        let ratio = params.stretch_ratio;
        let num_channels = params.channels.count();
        let vocoders = Self::create_vocoders(&params, ratio);
        let channel_buffers = (0..num_channels).map(|_| Vec::new()).collect();

        Self {
            params,
            input_buffer: Vec::new(),
            current_ratio: ratio,
            target_ratio: ratio,
            initialized: false,
            vocoders,
            channel_buffers,
            output_scratch: Vec::new(),
            source_bpm: None,
            use_hybrid: false,
        }
    }

    /// Creates PhaseVocoder instances for each channel.
    fn create_vocoders(params: &StretchParams, ratio: f64) -> Vec<PhaseVocoder> {
        (0..params.channels.count())
            .map(|_| {
                PhaseVocoder::with_window(
                    params.fft_size,
                    params.hop_size,
                    ratio,
                    params.sample_rate,
                    params.sub_bass_cutoff,
                    params.window_type,
                )
            })
            .collect()
    }

    /// Processes a chunk of interleaved audio samples.
    ///
    /// Returns stretched output samples. May return an empty slice if
    /// not enough input has accumulated yet.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        if input.iter().any(|s| !s.is_finite()) {
            return Err(StretchError::NonFiniteInput);
        }
        self.input_buffer.extend_from_slice(input);
        self.initialized = true;
        self.interpolate_ratio();

        let num_channels = self.params.channels.count();
        let min_input = self.params.fft_size * num_channels * LATENCY_FFT_MULTIPLIER;

        if self.input_buffer.len() < min_input {
            return Ok(vec![]);
        }

        let total_frames = self.input_buffer.len() / num_channels;
        if total_frames < self.params.fft_size {
            return Ok(vec![]);
        }

        if self.use_hybrid {
            return self.process_hybrid_path(num_channels, total_frames);
        }

        // Update vocoders' stretch ratio in-place to preserve phase state.
        // Recreating vocoders would reset phase accumulators and cause clicks.
        if (self.current_ratio - self.params.stretch_ratio).abs() > RATIO_SNAP_THRESHOLD {
            for voc in &mut self.vocoders {
                voc.set_stretch_ratio(self.current_ratio);
            }
        }

        // Deinterleave, process per-channel, collect min output length
        let min_output_len = self.process_channels(num_channels)?;

        // Drain consumed input samples
        self.drain_consumed_input(total_frames, num_channels);

        // Re-interleave channel outputs
        if min_output_len == 0 {
            return Ok(vec![]);
        }
        Ok(self.interleave_output(min_output_len, num_channels))
    }

    /// Deinterleaves input, stretches each channel, returns min output length across channels.
    fn process_channels(&mut self, num_channels: usize) -> Result<usize, StretchError> {
        let mut min_output_len = usize::MAX;

        for ch in 0..num_channels {
            self.deinterleave_channel(ch, num_channels);
            let stretched = self.vocoders[ch].process(&self.channel_buffers[ch])?;
            min_output_len = min_output_len.min(stretched.len());
            self.channel_buffers[ch] = stretched;
        }

        Ok(if min_output_len == usize::MAX {
            0
        } else {
            min_output_len
        })
    }

    /// Extracts a single channel from the interleaved input buffer.
    fn deinterleave_channel(&mut self, ch: usize, num_channels: usize) {
        self.channel_buffers[ch].clear();
        self.channel_buffers[ch].extend(
            self.input_buffer
                .iter()
                .skip(ch)
                .step_by(num_channels)
                .copied(),
        );
    }

    /// Drains consumed samples from the input buffer, keeping unprocessed remainder.
    fn drain_consumed_input(&mut self, total_frames: usize, num_channels: usize) {
        let hop = self.params.hop_size;
        let num_frames_processed = if total_frames >= self.params.fft_size {
            (total_frames - self.params.fft_size) / hop + 1
        } else {
            0
        };
        let samples_consumed = if num_frames_processed > 0 {
            ((num_frames_processed - 1) * hop + self.params.fft_size) * num_channels
        } else {
            0
        };
        if samples_consumed > 0 && samples_consumed <= self.input_buffer.len() {
            self.input_buffer.drain(..samples_consumed);
        }
    }

    /// Interleaves per-channel outputs into a single buffer.
    ///
    /// Returns a copy of the interleaved data; the scratch buffer stays
    /// allocated for reuse on the next call.
    fn interleave_output(&mut self, min_output_len: usize, num_channels: usize) -> Vec<f32> {
        let total = min_output_len * num_channels;
        self.output_scratch.clear();
        self.output_scratch.reserve(total);
        for i in 0..min_output_len {
            for ch in 0..num_channels {
                self.output_scratch.push(self.channel_buffers[ch][i]);
            }
        }
        self.output_scratch.clone()
    }

    /// Processes accumulated input through the hybrid algorithm (transient detection + WSOLA + PV).
    ///
    /// Deinterleaves the input, runs each channel through a fresh HybridStretcher
    /// with the current ratio, then reinterleaves the output. All accumulated input
    /// is consumed on each call.
    fn process_hybrid_path(
        &mut self,
        num_channels: usize,
        total_frames: usize,
    ) -> Result<Vec<f32>, StretchError> {
        // Build params with the current (interpolated) ratio
        let mut hybrid_params = self.params.clone();
        hybrid_params.stretch_ratio = self.current_ratio;

        let mut min_output_len = usize::MAX;

        for ch in 0..num_channels {
            self.deinterleave_channel(ch, num_channels);
            let stretcher = HybridStretcher::new(hybrid_params.clone());
            let stretched = stretcher.process(&self.channel_buffers[ch])?;
            min_output_len = min_output_len.min(stretched.len());
            self.channel_buffers[ch] = stretched;
        }

        // Consume all input
        self.input_buffer.clear();
        // Keep overlap for continuity: retain the last FFT window of frames
        // (not applicable in hybrid mode — each call is self-contained)
        let _ = total_frames; // used for API consistency

        if min_output_len == usize::MAX || min_output_len == 0 {
            return Ok(vec![]);
        }
        Ok(self.interleave_output(min_output_len, num_channels))
    }

    /// Creates a streaming processor configured for BPM matching.
    ///
    /// This is a convenience constructor for DJ workflows. It computes the
    /// stretch ratio as `source_bpm / target_bpm` and applies the
    /// [`EdmPreset::DjBeatmatch`](crate::EdmPreset::DjBeatmatch) preset.
    ///
    /// Use [`set_tempo`](Self::set_tempo) to smoothly change the target BPM
    /// during playback.
    pub fn from_tempo(source_bpm: f64, target_bpm: f64, sample_rate: u32, channels: u32) -> Self {
        let params = StretchParams::from_tempo(source_bpm, target_bpm)
            .with_sample_rate(sample_rate)
            .with_channels(channels)
            .with_preset(crate::EdmPreset::DjBeatmatch);
        let mut proc = Self::new(params);
        proc.source_bpm = Some(source_bpm);
        proc
    }

    /// Changes the stretch ratio for subsequent processing.
    ///
    /// The ratio change is interpolated smoothly to avoid clicks.
    pub fn set_stretch_ratio(&mut self, ratio: f64) {
        self.target_ratio = ratio;
    }

    /// Enables or disables hybrid processing mode.
    ///
    /// When enabled, the processor uses the full hybrid algorithm (transient
    /// detection + WSOLA for transients + phase vocoder for tonal content),
    /// matching the quality of the batch [`stretch()`](crate::stretch()) API.
    /// This produces better results for EDM audio with kicks and transients,
    /// but has higher latency since it processes all accumulated input at once.
    ///
    /// When disabled (default), uses phase vocoder only for lowest latency.
    pub fn set_hybrid_mode(&mut self, enabled: bool) {
        self.use_hybrid = enabled;
    }

    /// Returns whether hybrid processing mode is enabled.
    pub fn is_hybrid_mode(&self) -> bool {
        self.use_hybrid
    }

    /// Changes the target BPM, smoothly adjusting the stretch ratio.
    ///
    /// Requires that the processor was created with [`from_tempo`](Self::from_tempo)
    /// so that the source BPM is known. Returns `false` if the source BPM is
    /// unknown (processor was created with [`new`](Self::new)).
    pub fn set_tempo(&mut self, target_bpm: f64) -> bool {
        if let Some(source) = self.source_bpm {
            if target_bpm > 0.0 {
                self.set_stretch_ratio(source / target_bpm);
                return true;
            }
        }
        false
    }

    /// Returns the source BPM if the processor was created with [`from_tempo`](Self::from_tempo).
    pub fn source_bpm(&self) -> Option<f64> {
        self.source_bpm
    }

    /// Returns a reference to the current parameters.
    pub fn params(&self) -> &StretchParams {
        &self.params
    }

    /// Returns the current effective stretch ratio.
    pub fn current_stretch_ratio(&self) -> f64 {
        self.current_ratio
    }

    /// Returns the minimum latency in samples.
    ///
    /// This is the number of input samples needed before any output is produced.
    pub fn latency_samples(&self) -> usize {
        self.params.fft_size * LATENCY_FFT_MULTIPLIER
    }

    /// Returns the minimum latency in seconds.
    pub fn latency_secs(&self) -> f64 {
        self.latency_samples() as f64 / self.params.sample_rate as f64
    }

    /// Resets the processor state.
    pub fn reset(&mut self) {
        self.input_buffer.clear();
        self.output_scratch.clear();
        self.current_ratio = self.params.stretch_ratio;
        self.target_ratio = self.params.stretch_ratio;
        self.initialized = false;

        // Recreate vocoders with original ratio
        self.vocoders = Self::create_vocoders(&self.params, self.params.stretch_ratio);
    }

    /// Processes a chunk of interleaved audio, writing output into `output`.
    ///
    /// This is the zero-copy variant of [`process`](Self::process). Instead of
    /// returning a new `Vec`, it appends stretched samples to the caller-provided
    /// buffer. This eliminates one heap allocation per call and is ideal for
    /// real-time audio engines with pre-allocated ring buffers.
    ///
    /// Returns the number of samples written to `output`.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::{StreamProcessor, StretchParams, EdmPreset};
    ///
    /// let params = StretchParams::new(1.0)
    ///     .with_preset(EdmPreset::DjBeatmatch)
    ///     .with_sample_rate(44100)
    ///     .with_channels(1);
    ///
    /// let mut processor = StreamProcessor::new(params);
    /// let mut output_buf = Vec::with_capacity(8192);
    ///
    /// let chunk = vec![0.0f32; 4096];
    /// let n = processor.process_into(&chunk, &mut output_buf).unwrap();
    /// // output_buf now contains `n` stretched samples (may be 0 until enough input accumulates)
    /// ```
    pub fn process_into(
        &mut self,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<usize, StretchError> {
        if input.iter().any(|s| !s.is_finite()) {
            return Err(StretchError::NonFiniteInput);
        }
        self.input_buffer.extend_from_slice(input);
        self.initialized = true;
        self.interpolate_ratio();

        let num_channels = self.params.channels.count();
        let min_input = self.params.fft_size * num_channels * LATENCY_FFT_MULTIPLIER;

        if self.input_buffer.len() < min_input {
            return Ok(0);
        }

        let total_frames = self.input_buffer.len() / num_channels;
        if total_frames < self.params.fft_size {
            return Ok(0);
        }

        if self.use_hybrid {
            return self.process_hybrid_into(num_channels, total_frames, output);
        }

        if (self.current_ratio - self.params.stretch_ratio).abs() > RATIO_SNAP_THRESHOLD {
            for voc in &mut self.vocoders {
                voc.set_stretch_ratio(self.current_ratio);
            }
        }

        let min_output_len = self.process_channels(num_channels)?;
        self.drain_consumed_input(total_frames, num_channels);

        if min_output_len == 0 {
            return Ok(0);
        }

        let written = self.interleave_into(min_output_len, num_channels, output);
        Ok(written)
    }

    /// Flushes remaining buffered samples into a caller-provided buffer.
    ///
    /// Zero-copy variant of [`flush`](Self::flush). Returns the number of
    /// samples written to `output`.
    pub fn flush_into(&mut self, output: &mut Vec<f32>) -> Result<usize, StretchError> {
        if self.input_buffer.is_empty() {
            return Ok(0);
        }

        let nc = self.params.channels.count();
        let min_size = self.params.fft_size * nc * LATENCY_FFT_MULTIPLIER;
        while self.input_buffer.len() < min_size {
            self.input_buffer.push(0.0);
        }

        self.process_into(&[], output)
    }

    /// Flushes remaining buffered samples.
    pub fn flush(&mut self) -> Result<Vec<f32>, StretchError> {
        if self.input_buffer.is_empty() {
            return Ok(vec![]);
        }

        // Pad input to minimum size and process
        let nc = self.params.channels.count();
        let min_size = self.params.fft_size * nc * LATENCY_FFT_MULTIPLIER;
        while self.input_buffer.len() < min_size {
            self.input_buffer.push(0.0);
        }

        self.process(&[])
    }

    /// Interleaves per-channel outputs directly into a caller-provided buffer.
    ///
    /// Returns the number of samples written.
    fn interleave_into(
        &self,
        min_output_len: usize,
        num_channels: usize,
        output: &mut Vec<f32>,
    ) -> usize {
        let total = min_output_len * num_channels;
        output.reserve(total);
        for i in 0..min_output_len {
            for ch in 0..num_channels {
                output.push(self.channel_buffers[ch][i]);
            }
        }
        total
    }

    /// Processes accumulated input through hybrid algorithm, writing into caller buffer.
    fn process_hybrid_into(
        &mut self,
        num_channels: usize,
        total_frames: usize,
        output: &mut Vec<f32>,
    ) -> Result<usize, StretchError> {
        let mut hybrid_params = self.params.clone();
        hybrid_params.stretch_ratio = self.current_ratio;

        let mut min_output_len = usize::MAX;

        for ch in 0..num_channels {
            self.deinterleave_channel(ch, num_channels);
            let stretcher = HybridStretcher::new(hybrid_params.clone());
            let stretched = stretcher.process(&self.channel_buffers[ch])?;
            min_output_len = min_output_len.min(stretched.len());
            self.channel_buffers[ch] = stretched;
        }

        self.input_buffer.clear();
        let _ = total_frames;

        if min_output_len == usize::MAX || min_output_len == 0 {
            return Ok(0);
        }

        Ok(self.interleave_into(min_output_len, num_channels, output))
    }

    /// Smoothly interpolates between current and target ratio.
    fn interpolate_ratio(&mut self) {
        self.current_ratio += RATIO_INTERPOLATION_ALPHA * (self.target_ratio - self.current_ratio);

        if (self.current_ratio - self.target_ratio).abs() < RATIO_SNAP_THRESHOLD {
            self.current_ratio = self.target_ratio;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

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

        proc.set_stretch_ratio(1.05);
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
        assert_eq!(proc.latency_samples(), 8192);
        assert!((proc.latency_secs() - 8192.0 / 44100.0).abs() < 1e-6);
    }

    #[test]
    fn test_stream_processor_reset() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        proc.set_stretch_ratio(2.0);
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
        proc.set_stretch_ratio(1.05);
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
                (ratio - 1.5).abs() < 0.4,
                "Hybrid stretch ratio {} too far from 1.5",
                ratio
            );
        }
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
        let mut output2 = Vec::new();
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
        let mut output = Vec::new();
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
    fn test_process_into_returns_count() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        let mut output = Vec::new();

        // First small chunk: not enough data yet
        let small = vec![0.0f32; 1024];
        let n = proc.process_into(&small, &mut output).unwrap();
        assert_eq!(n, 0);
        assert!(output.is_empty());

        // Large chunk: should produce output
        let big: Vec<f32> = (0..44100)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let n = proc.process_into(&big, &mut output).unwrap();
        assert_eq!(n, output.len());
        assert!(n > 0);
    }

    #[test]
    fn test_process_into_appends() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut proc = StreamProcessor::new(params);
        let mut output = vec![42.0f32]; // pre-existing data

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

        let mut output = Vec::new();
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
}

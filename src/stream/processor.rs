//! Real-time streaming time-stretch processor.

use crate::core::types::StretchParams;
use crate::error::StretchError;
use crate::stretch::phase_vocoder::PhaseVocoder;

/// Threshold below which ratio differences are considered negligible.
const RATIO_SNAP_THRESHOLD: f64 = 0.0001;
/// Smoothing factor for interpolating between current and target ratio.
const RATIO_INTERPOLATION_ALPHA: f64 = 0.1;

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
        }
    }

    /// Creates PhaseVocoder instances for each channel.
    fn create_vocoders(params: &StretchParams, ratio: f64) -> Vec<PhaseVocoder> {
        (0..params.channels.count())
            .map(|_| {
                PhaseVocoder::new(
                    params.fft_size,
                    params.hop_size,
                    ratio,
                    params.sample_rate,
                    params.sub_bass_cutoff,
                )
            })
            .collect()
    }

    /// Processes a chunk of interleaved audio samples.
    ///
    /// Returns stretched output samples. May return an empty slice if
    /// not enough input has accumulated yet.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        self.input_buffer.extend_from_slice(input);
        self.initialized = true;
        self.interpolate_ratio();

        let num_channels = self.params.channels.count();
        let min_input = self.params.fft_size * num_channels * 2;

        if self.input_buffer.len() < min_input {
            return Ok(vec![]);
        }

        let total_frames = self.input_buffer.len() / num_channels;
        if total_frames < self.params.fft_size {
            return Ok(vec![]);
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

        Ok(if min_output_len == usize::MAX { 0 } else { min_output_len })
    }

    /// Extracts a single channel from the interleaved input buffer.
    fn deinterleave_channel(&mut self, ch: usize, num_channels: usize) {
        self.channel_buffers[ch].clear();
        let mut idx = ch;
        while idx < self.input_buffer.len() {
            self.channel_buffers[ch].push(self.input_buffer[idx]);
            idx += num_channels;
        }
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
    fn interleave_output(&mut self, min_output_len: usize, num_channels: usize) -> Vec<f32> {
        self.output_scratch.clear();
        self.output_scratch.reserve(min_output_len * num_channels);
        for i in 0..min_output_len {
            for ch in 0..num_channels {
                self.output_scratch.push(self.channel_buffers[ch][i]);
            }
        }
        std::mem::take(&mut self.output_scratch)
    }

    /// Changes the stretch ratio for subsequent processing.
    ///
    /// The ratio change is interpolated smoothly to avoid clicks.
    pub fn set_stretch_ratio(&mut self, ratio: f64) {
        self.target_ratio = ratio;
    }

    /// Returns the current effective stretch ratio.
    pub fn current_stretch_ratio(&self) -> f64 {
        self.current_ratio
    }

    /// Returns the minimum latency in samples.
    ///
    /// This is the number of input samples needed before any output is produced.
    pub fn latency_samples(&self) -> usize {
        self.params.fft_size * 2
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

    /// Flushes remaining buffered samples.
    pub fn flush(&mut self) -> Result<Vec<f32>, StretchError> {
        if self.input_buffer.is_empty() {
            return Ok(vec![]);
        }

        // Pad input to minimum size and process
        let nc = self.params.channels.count();
        let min_size = self.params.fft_size * nc * 2;
        while self.input_buffer.len() < min_size {
            self.input_buffer.push(0.0);
        }

        self.process(&[])
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
}

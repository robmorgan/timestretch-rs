use crate::core::types::StretchParams;
use crate::error::StretchError;
use crate::stretch::phase_vocoder::PhaseVocoder;

/// Streaming chunk-based processor for real-time time stretching.
///
/// Accumulates input samples in a ring buffer and processes them
/// using the phase vocoder when enough data is available.
pub struct StreamProcessor {
    params: StretchParams,
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    /// Current stretch ratio (can be changed on the fly).
    current_ratio: f64,
    /// Target stretch ratio (for smooth interpolation).
    target_ratio: f64,
    /// Whether the processor has been initialized.
    initialized: bool,
}

impl StreamProcessor {
    /// Creates a new streaming processor.
    pub fn new(params: StretchParams) -> Self {
        let ratio = params.stretch_ratio;
        Self {
            params,
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            current_ratio: ratio,
            target_ratio: ratio,
            initialized: false,
        }
    }

    /// Processes a chunk of interleaved audio samples.
    ///
    /// Returns stretched output samples. May return an empty slice if
    /// not enough input has accumulated yet.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        // Append input to buffer
        self.input_buffer.extend_from_slice(input);
        self.initialized = true;

        // Smoothly interpolate ratio
        self.interpolate_ratio();

        let nc = self.params.channels.count();
        let min_input = self.params.fft_size * nc * 2;

        // Process when we have enough data
        if self.input_buffer.len() < min_input {
            return Ok(vec![]);
        }

        // Process each channel separately
        let num_channels = self.params.channels.count();
        let total_frames = self.input_buffer.len() / num_channels;

        if total_frames < self.params.fft_size {
            return Ok(vec![]);
        }

        let mut channel_outputs: Vec<Vec<f32>> = Vec::new();

        for ch in 0..num_channels {
            // Deinterleave
            let channel_data: Vec<f32> = self.input_buffer
                .iter()
                .skip(ch)
                .step_by(num_channels)
                .copied()
                .collect();

            // Process with phase vocoder
            let mut pv = PhaseVocoder::new(
                self.params.fft_size,
                self.params.hop_size,
                self.current_ratio,
                self.params.sample_rate,
                self.params.sub_bass_cutoff,
            );

            match pv.process(&channel_data) {
                Ok(stretched) => channel_outputs.push(stretched),
                Err(e) => return Err(e),
            }
        }

        // Clear input buffer (keep a small overlap for continuity)
        let keep = self.params.fft_size * nc;
        if self.input_buffer.len() > keep {
            let drain_to = self.input_buffer.len() - keep;
            self.input_buffer.drain(..drain_to);
        }

        // Re-interleave output
        if channel_outputs.is_empty() {
            return Ok(vec![]);
        }

        let min_len = channel_outputs.iter().map(|c| c.len()).min().unwrap_or(0);
        let mut output = Vec::with_capacity(min_len * num_channels);

        for i in 0..min_len {
            for ch_out in &channel_outputs {
                output.push(ch_out[i]);
            }
        }

        // Append to output buffer and drain
        self.output_buffer.extend_from_slice(&output);
        let result = std::mem::take(&mut self.output_buffer);
        Ok(result)
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
        self.output_buffer.clear();
        self.current_ratio = self.params.stretch_ratio;
        self.target_ratio = self.params.stretch_ratio;
        self.initialized = false;
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
        let alpha = 0.1; // Smoothing factor
        self.current_ratio += alpha * (self.target_ratio - self.current_ratio);

        // Snap when close enough
        if (self.current_ratio - self.target_ratio).abs() < 0.0001 {
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
}

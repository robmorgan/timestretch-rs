use crate::core::types::{StretchError, StretchParams};
use crate::stretch::phase_vocoder::PhaseVocoder;

/// Streaming processor for real-time time stretching.
///
/// Processes audio in chunks, maintaining internal state between calls.
/// Supports dynamic stretch ratio changes for DJ pitch fader use.
pub struct StreamProcessor {
    params: StretchParams,
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    phase_vocoder: PhaseVocoder,
    channels: u16,
}

impl StreamProcessor {
    /// Create a new stream processor.
    ///
    /// # Errors
    /// Returns error if parameters are invalid.
    pub fn new(params: StretchParams) -> Result<Self, StretchError> {
        params.validate()?;
        let hop = params.effective_hop_size();
        let pv = PhaseVocoder::new(
            params.fft_size,
            hop,
            params.stretch_ratio,
            params.sample_rate,
            120.0,
        );
        let channels = params.channels;
        Ok(Self {
            params,
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            phase_vocoder: pv,
            channels,
        })
    }

    /// Process a chunk of interleaved audio samples.
    ///
    /// Returns the stretched output samples (interleaved if stereo).
    /// May return fewer or more samples than input depending on stretch ratio
    /// and internal buffering.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        if self.channels == 1 {
            return Ok(self.process_mono(input));
        }

        // Stereo: deinterleave, process each channel, reinterleave
        let num_frames = input.len() / 2;
        let mut left = Vec::with_capacity(num_frames);
        let mut right = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            left.push(input[i * 2]);
            right.push(input[i * 2 + 1]);
        }

        let left_out = self.process_mono(&left);
        let right_out = self.process_mono(&right);

        let out_frames = left_out.len().min(right_out.len());
        let mut interleaved = Vec::with_capacity(out_frames * 2);
        for i in 0..out_frames {
            interleaved.push(left_out[i]);
            interleaved.push(right_out[i]);
        }

        Ok(interleaved)
    }

    fn process_mono(&mut self, input: &[f32]) -> Vec<f32> {
        self.input_buffer.extend_from_slice(input);

        let fft_size = self.params.fft_size;
        if self.input_buffer.len() < fft_size {
            return Vec::new();
        }

        // Process all complete frames
        let result = self.phase_vocoder.process(&self.input_buffer);

        // Keep a tail for overlap with next chunk
        let keep = fft_size;
        if self.input_buffer.len() > keep {
            let drain_to = self.input_buffer.len() - keep;
            self.input_buffer.drain(..drain_to);
        }

        // Return new output beyond what we've already returned
        if !result.is_empty() {
            let new_output = result;
            self.output_buffer.extend_from_slice(&new_output);
        }

        let output = self.output_buffer.clone();
        self.output_buffer.clear();
        output
    }

    /// Set a new stretch ratio (smooth transition for DJ use).
    pub fn set_stretch_ratio(&mut self, ratio: f64) -> Result<(), StretchError> {
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err(StretchError::InvalidStretchRatio(ratio));
        }
        self.params.stretch_ratio = ratio;
        let hop = self.params.effective_hop_size();
        self.phase_vocoder = PhaseVocoder::new(
            self.params.fft_size,
            hop,
            ratio,
            self.params.sample_rate,
            120.0,
        );
        Ok(())
    }

    /// Get the minimum latency in samples.
    pub fn latency_samples(&self) -> usize {
        self.params.fft_size
    }

    /// Get the minimum latency in seconds.
    pub fn latency_secs(&self) -> f64 {
        self.params.fft_size as f64 / self.params.sample_rate as f64
    }

    /// Flush remaining audio from internal buffers.
    pub fn flush(&mut self) -> Vec<f32> {
        if self.input_buffer.is_empty() {
            return std::mem::take(&mut self.output_buffer);
        }

        // Pad remaining input and process
        let fft_size = self.params.fft_size;
        while self.input_buffer.len() < fft_size {
            self.input_buffer.push(0.0);
        }

        let result = self.phase_vocoder.process(&self.input_buffer);
        self.input_buffer.clear();

        let mut output = std::mem::take(&mut self.output_buffer);
        output.extend_from_slice(&result);
        output
    }

    /// Reset the processor state.
    pub fn reset(&mut self) {
        self.input_buffer.clear();
        self.output_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn generate_sine(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_stream_processor_basic() {
        let params = StretchParams::new(1.0).unwrap().with_channels(1);
        let mut proc = StreamProcessor::new(params).unwrap();
        let input = generate_sine(440.0, 44100, 44100);

        // Process in chunks
        let chunk_size = 1024;
        let mut total_output = Vec::new();
        for chunk in input.chunks(chunk_size) {
            let out = proc.process(chunk).unwrap();
            total_output.extend_from_slice(&out);
        }
        let flushed = proc.flush();
        total_output.extend_from_slice(&flushed);

        assert!(
            !total_output.is_empty(),
            "Streaming processor should produce output"
        );
    }

    #[test]
    fn test_stream_processor_stereo() {
        let params = StretchParams::new(1.0).unwrap().with_channels(2);
        let mut proc = StreamProcessor::new(params).unwrap();

        // Create stereo sine wave (interleaved)
        let num_frames = 22050;
        let mut input = Vec::with_capacity(num_frames * 2);
        for i in 0..num_frames {
            let l = (2.0 * PI * 440.0 * i as f32 / 44100.0).sin();
            let r = (2.0 * PI * 880.0 * i as f32 / 44100.0).sin();
            input.push(l);
            input.push(r);
        }

        let output = proc.process(&input).unwrap();
        // Output should have even number of samples (stereo)
        assert_eq!(output.len() % 2, 0);
    }

    #[test]
    fn test_stream_processor_set_ratio() {
        let params = StretchParams::new(1.0).unwrap();
        let mut proc = StreamProcessor::new(params).unwrap();

        assert!(proc.set_stretch_ratio(1.5).is_ok());
        assert!(proc.set_stretch_ratio(0.0).is_err());
        assert!(proc.set_stretch_ratio(-1.0).is_err());
        assert!(proc.set_stretch_ratio(f64::NAN).is_err());
    }

    #[test]
    fn test_stream_processor_latency() {
        let params = StretchParams::new(1.0).unwrap();
        let proc = StreamProcessor::new(params).unwrap();

        assert_eq!(proc.latency_samples(), 4096);
        assert!((proc.latency_secs() - 4096.0 / 44100.0).abs() < 1e-6);
    }

    #[test]
    fn test_stream_processor_reset() {
        let params = StretchParams::new(1.0).unwrap();
        let mut proc = StreamProcessor::new(params).unwrap();

        let input = generate_sine(440.0, 44100, 8192);
        let _ = proc.process(&input).unwrap();
        proc.reset();

        // After reset, should be able to process again
        let out = proc.process(&input).unwrap();
        let _ = out; // Just verify no panic
    }

    #[test]
    fn test_stream_processor_invalid_params() {
        let mut params = StretchParams::new(1.0).unwrap();
        params.fft_size = 1000; // Not power of two
        assert!(StreamProcessor::new(params).is_err());
    }
}

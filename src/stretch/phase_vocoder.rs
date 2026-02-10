use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

/// Phase vocoder time stretching.
///
/// Uses STFT analysis, phase manipulation, and resynthesis to stretch audio
/// while preserving pitch. Implements identity phase locking for tonal coherence
/// and sub-bass phase locking below 120 Hz.
pub struct PhaseVocoder {
    fft_size: usize,
    hop_size: usize,
    stretch_ratio: f64,
    sample_rate: u32,
    sub_bass_cutoff: f32,
    window: Vec<f32>,
}

impl PhaseVocoder {
    /// Create a new phase vocoder.
    pub fn new(
        fft_size: usize,
        hop_size: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
    ) -> Self {
        let window = crate::core::window::hann_window(fft_size);
        Self {
            fft_size,
            hop_size,
            stretch_ratio,
            sample_rate,
            sub_bass_cutoff,
            window,
        }
    }

    /// Process mono audio samples through the phase vocoder.
    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        if input.len() < self.fft_size {
            // Input too short — pad and process
            let mut padded = vec![0.0f32; self.fft_size];
            for (i, &s) in input.iter().enumerate() {
                padded[i] = s;
            }
            return self.process_internal(&padded);
        }
        self.process_internal(input)
    }

    fn process_internal(&self, input: &[f32]) -> Vec<f32> {
        let analysis_hop = self.hop_size;
        let synthesis_hop = (self.hop_size as f64 * self.stretch_ratio) as usize;
        let synthesis_hop = synthesis_hop.max(1);

        let num_frames = (input.len().saturating_sub(self.fft_size)) / analysis_hop + 1;
        if num_frames == 0 {
            return Vec::new();
        }

        let output_len = (num_frames - 1) * synthesis_hop + self.fft_size;
        let mut output = vec![0.0f32; output_len];
        let mut window_sum = vec![0.0f32; output_len];

        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(self.fft_size);
        let fft_inverse = planner.plan_fft_inverse(self.fft_size);

        let half = self.fft_size / 2 + 1;
        let freq_resolution = self.sample_rate as f32 / self.fft_size as f32;
        let sub_bass_bin = (self.sub_bass_cutoff / freq_resolution).ceil() as usize;

        let mut prev_phase = vec![0.0f32; half];
        let mut synth_phase = vec![0.0f32; half];
        let expected_phase_advance =
            |bin: usize| -> f32 { 2.0 * PI * bin as f32 * analysis_hop as f32 / self.fft_size as f32 };

        for frame_idx in 0..num_frames {
            let start = frame_idx * analysis_hop;
            let end = (start + self.fft_size).min(input.len());

            // Analysis: window and FFT
            let mut buffer: Vec<Complex<f32>> = Vec::with_capacity(self.fft_size);
            for i in 0..self.fft_size {
                let sample = if start + i < end { input[start + i] } else { 0.0 };
                buffer.push(Complex::new(sample * self.window[i], 0.0));
            }
            fft_forward.process(&mut buffer);

            // Phase manipulation
            let mut magnitudes = Vec::with_capacity(half);
            let mut phases = Vec::with_capacity(half);
            for bin in 0..half {
                magnitudes.push(buffer[bin].norm());
                phases.push(buffer[bin].arg());
            }

            if frame_idx == 0 {
                // First frame: use original phases
                synth_phase.copy_from_slice(&phases);
            } else {
                for bin in 0..half {
                    let phase_diff = phases[bin] - prev_phase[bin];
                    let expected = expected_phase_advance(bin);
                    let deviation = phase_diff - expected;
                    // Wrap deviation to [-π, π]
                    let wrapped = deviation - (deviation / (2.0 * PI)).round() * 2.0 * PI;
                    let true_freq = expected + wrapped;

                    if bin < sub_bass_bin {
                        // Sub-bass phase locking: use original phase relationship
                        synth_phase[bin] = phases[bin]
                            + (self.stretch_ratio as f32 - 1.0) * expected;
                    } else {
                        // Standard phase advancement
                        synth_phase[bin] += true_freq * self.stretch_ratio as f32;
                    }
                }
            }

            // Apply identity phase locking for tonal coherence
            let locked_phase = identity_phase_lock(&magnitudes, &synth_phase, half);

            prev_phase.copy_from_slice(&phases);

            // Synthesis: reconstruct spectrum and IFFT
            let mut synth_buffer = vec![Complex::new(0.0f32, 0.0); self.fft_size];
            for bin in 0..half {
                let phase = locked_phase[bin];
                synth_buffer[bin] =
                    Complex::new(magnitudes[bin] * phase.cos(), magnitudes[bin] * phase.sin());
                // Mirror for negative frequencies
                if bin > 0 && bin < self.fft_size / 2 {
                    synth_buffer[self.fft_size - bin] = synth_buffer[bin].conj();
                }
            }

            fft_inverse.process(&mut synth_buffer);

            // Overlap-add
            let out_start = frame_idx * synthesis_hop;
            let scale = 1.0 / self.fft_size as f32;
            for i in 0..self.fft_size {
                let out_idx = out_start + i;
                if out_idx < output.len() {
                    output[out_idx] += synth_buffer[i].re * scale * self.window[i];
                    window_sum[out_idx] += self.window[i] * self.window[i];
                }
            }
        }

        // Normalize by window sum
        for i in 0..output.len() {
            if window_sum[i] > 1e-6 {
                output[i] /= window_sum[i];
            }
        }

        output
    }
}

/// Identity phase locking: adjust phases so that peak bins maintain their
/// phase relationship with neighbors.
fn identity_phase_lock(magnitudes: &[f32], phases: &[f32], half: usize) -> Vec<f32> {
    let mut locked = phases.to_vec();

    // Find local peaks in the magnitude spectrum
    for bin in 1..half.saturating_sub(1) {
        if magnitudes[bin] > magnitudes[bin - 1] && magnitudes[bin] > magnitudes[bin + 1] {
            // This is a peak — its phase stays as is
            // Lock neighboring bins to this peak's phase
            let peak_phase = phases[bin];

            // Lock bins below the peak (until next peak or boundary)
            let mut k = bin.saturating_sub(1);
            loop {
                if magnitudes[k] > magnitudes[k + 1] && k != bin.saturating_sub(1) {
                    break;
                }
                locked[k] = peak_phase + phases[k] - phases[bin];
                if k == 0 {
                    break;
                }
                k -= 1;
            }

            // Lock bins above the peak
            for k in (bin + 1)..half.min(bin + 10) {
                if k + 1 < half && magnitudes[k] > magnitudes[k - 1] && magnitudes[k] > magnitudes[k + 1] {
                    break; // Next peak found
                }
                locked[k] = peak_phase + phases[k] - phases[bin];
            }
        }
    }

    locked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_sine(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_phase_vocoder_identity() {
        let sample_rate = 44100;
        let input = generate_sine(440.0, sample_rate, 44100);
        let pv = PhaseVocoder::new(4096, 1024, 1.0, sample_rate, 120.0);
        let output = pv.process(&input);

        // Output should be approximately the same length
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.0).abs() < 0.1,
            "Identity stretch should preserve length, got ratio {len_ratio}"
        );

        // Check that the output is not all zeros
        let max_val = output.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_val > 0.1, "Output should not be silent, max={max_val}");
    }

    #[test]
    fn test_phase_vocoder_stretch_doubles_length() {
        let sample_rate = 44100;
        let input = generate_sine(440.0, sample_rate, 44100);
        let pv = PhaseVocoder::new(4096, 1024, 2.0, sample_rate, 120.0);
        let output = pv.process(&input);

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 2.0).abs() < 0.2,
            "2x stretch should ~double length, got ratio {len_ratio}"
        );
    }

    #[test]
    fn test_phase_vocoder_compress_halves_length() {
        let sample_rate = 44100;
        let input = generate_sine(440.0, sample_rate, 44100);
        let pv = PhaseVocoder::new(4096, 1024, 0.5, sample_rate, 120.0);
        let output = pv.process(&input);

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 0.5).abs() < 0.2,
            "0.5x stretch should ~halve length, got ratio {len_ratio}"
        );
    }

    #[test]
    fn test_phase_vocoder_preserves_frequency() {
        let sample_rate = 44100;
        let freq = 1000.0;
        let input = generate_sine(freq, sample_rate, 44100);
        let pv = PhaseVocoder::new(4096, 1024, 1.5, sample_rate, 120.0);
        let output = pv.process(&input);

        // Check spectral centroid of output is near the input frequency
        let centroid = crate::analysis::frequency::spectral_centroid(&output, sample_rate, 4096);
        assert!(
            (centroid - freq).abs() < 200.0,
            "Spectral centroid should be near {freq} Hz, got {centroid}"
        );
    }

    #[test]
    fn test_phase_vocoder_short_input() {
        let pv = PhaseVocoder::new(4096, 1024, 1.0, 44100, 120.0);
        let input = vec![0.5; 100]; // Much shorter than FFT size
        let output = pv.process(&input);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_phase_vocoder_silence() {
        let pv = PhaseVocoder::new(4096, 1024, 1.5, 44100, 120.0);
        let input = vec![0.0; 44100];
        let output = pv.process(&input);
        let max_val = output.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(
            max_val < 1e-6,
            "Stretching silence should produce silence, max={max_val}"
        );
    }
}

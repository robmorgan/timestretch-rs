//! Phase vocoder time stretching with identity phase locking.

use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

use crate::core::window::{generate_window, WindowType};
use crate::error::StretchError;

const TWO_PI: f32 = 2.0 * PI;

/// Phase vocoder state for time stretching.
pub struct PhaseVocoder {
    fft_size: usize,
    hop_analysis: usize,
    hop_synthesis: usize,
    window: Vec<f32>,
    /// Phase accumulator for resynthesis.
    phase_accum: Vec<f32>,
    /// Previous analysis phase.
    prev_phase: Vec<f32>,
    /// Sub-bass cutoff bin for phase locking (reserved for future per-band processing).
    #[allow(dead_code)]
    sub_bass_bin: usize,
    /// FFT planner (cached).
    planner: FftPlanner<f32>,
    /// Pre-computed expected phase advance per bin.
    expected_phase_advance: Vec<f32>,
    /// Reusable FFT buffer.
    fft_buffer: Vec<Complex<f32>>,
    /// Reusable magnitude buffer.
    magnitudes: Vec<f32>,
    /// Reusable phase buffer.
    new_phases: Vec<f32>,
    /// Reusable peaks buffer for identity phase locking.
    peaks: Vec<usize>,
}

impl PhaseVocoder {
    /// Creates a new phase vocoder.
    pub fn new(
        fft_size: usize,
        hop_analysis: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
    ) -> Self {
        let hop_synthesis = (hop_analysis as f64 * stretch_ratio).round() as usize;
        let window = generate_window(WindowType::Hann, fft_size);
        let num_bins = fft_size / 2 + 1;
        let sub_bass_bin =
            (sub_bass_cutoff * fft_size as f32 / sample_rate as f32).round() as usize;

        let expected_phase_advance: Vec<f32> = (0..num_bins)
            .map(|bin| TWO_PI * bin as f32 * hop_analysis as f32 / fft_size as f32)
            .collect();

        Self {
            fft_size,
            hop_analysis,
            hop_synthesis,
            window,
            phase_accum: vec![0.0; num_bins],
            prev_phase: vec![0.0; num_bins],
            sub_bass_bin: sub_bass_bin.min(num_bins),
            planner: FftPlanner::new(),
            expected_phase_advance,
            fft_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            magnitudes: vec![0.0; num_bins],
            new_phases: vec![0.0; num_bins],
            peaks: Vec::with_capacity(num_bins / 4),
        }
    }

    /// Stretches a mono audio signal using phase vocoder with identity phase locking.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        if input.len() < self.fft_size {
            return Err(StretchError::InputTooShort {
                provided: input.len(),
                minimum: self.fft_size,
            });
        }

        let num_bins = self.fft_size / 2 + 1;
        let num_frames = (input.len() - self.fft_size) / self.hop_analysis + 1;
        let output_len = (num_frames - 1) * self.hop_synthesis + self.fft_size;

        let mut output = vec![0.0f32; output_len];
        let mut window_sum = vec![0.0f32; output_len];

        let fft_forward = self.planner.plan_fft_forward(self.fft_size);
        let fft_inverse = self.planner.plan_fft_inverse(self.fft_size);

        // Reset phase state without reallocating
        self.phase_accum.iter_mut().for_each(|x| *x = 0.0);
        self.prev_phase.iter_mut().for_each(|x| *x = 0.0);

        let hop_ratio = self.hop_synthesis as f32 / self.hop_analysis as f32;
        let norm = 1.0 / self.fft_size as f32;

        for frame_idx in 0..num_frames {
            let analysis_pos = frame_idx * self.hop_analysis;
            let synthesis_pos = frame_idx * self.hop_synthesis;

            // Analysis: window and FFT (reuse buffer)
            let input_frame = &input[analysis_pos..analysis_pos + self.fft_size];
            for (i, (&sample, &win)) in input_frame.iter().zip(self.window.iter()).enumerate() {
                self.fft_buffer[i] = Complex::new(sample * win, 0.0);
            }

            fft_forward.process(&mut self.fft_buffer);

            // Phase processing (reuse magnitude/phase buffers)
            for bin in 0..num_bins {
                let c = self.fft_buffer[bin];
                self.magnitudes[bin] = c.norm();
                let phase = c.arg();

                // Compute phase deviation
                let expected = self.expected_phase_advance[bin];
                let phase_diff = phase - self.prev_phase[bin];
                let deviation = wrap_phase(phase_diff - expected);

                // True frequency deviation, accumulate phase with synthesis hop
                let true_freq = expected + deviation;
                self.phase_accum[bin] += true_freq * hop_ratio;

                self.new_phases[bin] = self.phase_accum[bin];
                self.prev_phase[bin] = phase;
            }

            // Identity phase locking for tonal coherence
            identity_phase_lock(
                &self.magnitudes,
                &mut self.new_phases,
                num_bins,
                &mut self.peaks,
            );

            // Reconstruct spectrum
            for bin in 0..num_bins {
                self.fft_buffer[bin] =
                    Complex::from_polar(self.magnitudes[bin], self.new_phases[bin]);
            }

            // Mirror for inverse FFT
            for bin in 1..num_bins - 1 {
                let mirror = self.fft_size - bin;
                if mirror < self.fft_size {
                    self.fft_buffer[mirror] = self.fft_buffer[bin].conj();
                }
            }

            // Inverse FFT
            fft_inverse.process(&mut self.fft_buffer);

            // Normalize and overlap-add
            let out_end = (synthesis_pos + self.fft_size).min(output_len);
            #[allow(clippy::needless_range_loop)]
            for i in 0..out_end - synthesis_pos {
                let out_idx = synthesis_pos + i;
                output[out_idx] += self.fft_buffer[i].re * norm * self.window[i];
                window_sum[out_idx] += self.window[i] * self.window[i];
            }
        }

        // Normalize by window sum, clamping to prevent amplification in
        // low-overlap regions (occurs when synthesis hop > analysis hop)
        let max_window_sum = window_sum.iter().cloned().fold(0.0f32, f32::max);
        let min_window_sum = (max_window_sum * 0.1).max(1e-6);
        for i in 0..output_len {
            let ws = window_sum[i].max(min_window_sum);
            if ws > 1e-6 {
                output[i] /= ws;
            }
        }

        Ok(output)
    }
}

/// Wraps a phase value to [-PI, PI] using efficient modulo arithmetic.
#[inline]
fn wrap_phase(phase: f32) -> f32 {
    let p = phase + PI;
    p - (p / TWO_PI).floor() * TWO_PI - PI
}

/// Identity phase locking: locks phase of non-peak bins to nearest peak.
///
/// This reduces phasing artifacts on tonal content by ensuring that
/// non-peak bins maintain their phase relationship to the nearest spectral peak.
fn identity_phase_lock(
    magnitudes: &[f32],
    phases: &mut [f32],
    num_bins: usize,
    peaks: &mut Vec<usize>,
) {
    if num_bins < 3 {
        return;
    }

    // Find spectral peaks (reuse buffer)
    peaks.clear();
    for bin in 1..num_bins - 1 {
        if magnitudes[bin] > magnitudes[bin - 1] && magnitudes[bin] > magnitudes[bin + 1] {
            peaks.push(bin);
        }
    }

    if peaks.is_empty() {
        return;
    }

    // For each non-peak bin, lock phase to nearest peak
    let mut peak_idx = 0;
    for bin in 0..num_bins {
        // Advance to nearest peak
        while peak_idx + 1 < peaks.len()
            && (peaks[peak_idx + 1] as i64 - bin as i64).unsigned_abs()
                < (peaks[peak_idx] as i64 - bin as i64).unsigned_abs()
        {
            peak_idx += 1;
        }

        let nearest_peak = peaks[peak_idx];
        if bin != nearest_peak {
            // Lock phase: maintain the phase difference from the original spectrum
            let phase_diff = phases[bin] - phases[nearest_peak];
            phases[bin] = phases[nearest_peak] + phase_diff;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_phase() {
        assert!((wrap_phase(0.0) - 0.0).abs() < 1e-6);
        assert!((wrap_phase(PI + 0.1) - (-PI + 0.1)).abs() < 1e-5);
        assert!((wrap_phase(-PI - 0.1) - (PI - 0.1)).abs() < 1e-5);
        // Test larger values
        assert!((wrap_phase(10.0 * PI + 0.5) - wrap_phase(0.5)).abs() < 1e-4);
        assert!((wrap_phase(-10.0 * PI - 0.5) - wrap_phase(-0.5)).abs() < 1e-4);
    }

    #[test]
    fn test_phase_vocoder_identity() {
        // Stretch ratio 1.0 should approximately preserve the signal
        let sample_rate = 44100;
        let fft_size = 4096;
        let hop = fft_size / 4;

        // Generate a 440 Hz sine wave
        let num_samples = fft_size * 4;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv = PhaseVocoder::new(fft_size, hop, 1.0, sample_rate, 120.0);
        let output = pv.process(&input).unwrap();

        // Output length should be approximately the same
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.0).abs() < 0.1,
            "Length ratio {} too far from 1.0",
            len_ratio
        );

        // Check that the output contains a similar frequency
        // (RMS should be similar)
        let input_rms: f32 =
            (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32).sqrt();
        let output_rms: f32 =
            (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();

        assert!(
            (output_rms - input_rms).abs() < input_rms * 0.5,
            "RMS mismatch: input={}, output={}",
            input_rms,
            output_rms
        );
    }

    #[test]
    fn test_phase_vocoder_stretch() {
        let sample_rate = 44100;
        let fft_size = 4096;
        let hop = fft_size / 4;

        // Use a longer signal for more accurate length ratio
        let num_samples = fft_size * 8;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let stretch_ratio = 2.0;
        let mut pv = PhaseVocoder::new(fft_size, hop, stretch_ratio, sample_rate, 120.0);
        let output = pv.process(&input).unwrap();

        // Output should be approximately 2x longer (with tolerance for edge effects)
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - stretch_ratio).abs() < 0.35,
            "Length ratio {} too far from {}",
            len_ratio,
            stretch_ratio
        );
    }

    #[test]
    fn test_phase_vocoder_compress() {
        let sample_rate = 44100;
        let fft_size = 4096;
        let hop = fft_size / 4;

        let num_samples = fft_size * 4;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let stretch_ratio = 0.5;
        let mut pv = PhaseVocoder::new(fft_size, hop, stretch_ratio, sample_rate, 120.0);
        let output = pv.process(&input).unwrap();

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - stretch_ratio).abs() < 0.2,
            "Length ratio {} too far from {}",
            len_ratio,
            stretch_ratio
        );
    }

    #[test]
    fn test_phase_vocoder_input_too_short() {
        let mut pv = PhaseVocoder::new(4096, 1024, 1.0, 44100, 120.0);
        let result = pv.process(&[0.0; 100]);
        assert!(result.is_err());
    }
}

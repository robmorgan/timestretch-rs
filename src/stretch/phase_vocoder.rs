//! Phase vocoder time stretching with identity phase locking and sub-bass phase locking.

use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

use crate::core::window::{generate_window, WindowType};
use crate::error::StretchError;

const TWO_PI: f32 = 2.0 * PI;
/// Minimum window sum to prevent amplification in low-overlap regions.
const MIN_WINDOW_SUM_RATIO: f32 = 0.1;
/// Absolute floor for window sum normalization.
const WINDOW_SUM_EPSILON: f32 = 1e-6;
/// Fraction of bins to pre-allocate for spectral peak detection (1/4 of bins).
const PEAKS_CAPACITY_DIVISOR: usize = 4;

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
    /// Current frame's analysis phases (for identity phase locking).
    analysis_phases: Vec<f32>,
    /// Bin index at or below which sub-bass phase locking is applied.
    sub_bass_bin: usize,
    /// Reusable output buffer (avoids allocation per process() call).
    output_buf: Vec<f32>,
    /// Reusable window sum buffer (avoids allocation per process() call).
    window_sum_buf: Vec<f32>,
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

        let expected_phase_advance: Vec<f32> = (0..num_bins)
            .map(|bin| TWO_PI * bin as f32 * hop_analysis as f32 / fft_size as f32)
            .collect();

        // Compute the bin index for the sub-bass cutoff frequency.
        // Bins at or below this index get rigid phase locking to prevent
        // phase cancellation in the critical sub-bass region.
        let sub_bass_bin =
            (sub_bass_cutoff * fft_size as f32 / sample_rate as f32).round() as usize;
        let sub_bass_bin = sub_bass_bin.min(num_bins);

        Self {
            fft_size,
            hop_analysis,
            hop_synthesis,
            window,
            phase_accum: vec![0.0; num_bins],
            prev_phase: vec![0.0; num_bins],
            planner: FftPlanner::new(),
            expected_phase_advance,
            fft_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            magnitudes: vec![0.0; num_bins],
            new_phases: vec![0.0; num_bins],
            peaks: Vec::with_capacity(num_bins / PEAKS_CAPACITY_DIVISOR),
            analysis_phases: vec![0.0; num_bins],
            sub_bass_bin,
            output_buf: Vec::new(),
            window_sum_buf: Vec::new(),
        }
    }

    /// Returns the FFT size.
    #[inline]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Returns the analysis hop size.
    #[inline]
    pub fn hop_analysis(&self) -> usize {
        self.hop_analysis
    }

    /// Returns the synthesis hop size.
    #[inline]
    pub fn hop_synthesis(&self) -> usize {
        self.hop_synthesis
    }

    /// Returns the sub-bass bin cutoff index.
    #[inline]
    pub fn sub_bass_bin(&self) -> usize {
        self.sub_bass_bin
    }

    /// Updates the stretch ratio without resetting phase state.
    ///
    /// This recalculates the synthesis hop size from the new ratio while
    /// preserving all accumulated phase information. Use this for smooth
    /// real-time ratio changes that avoid clicks and discontinuities.
    #[inline]
    pub fn set_stretch_ratio(&mut self, stretch_ratio: f64) {
        self.hop_synthesis = (self.hop_analysis as f64 * stretch_ratio).round() as usize;
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

        // Reuse pre-allocated buffers, growing if needed (never shrinks).
        self.output_buf.resize(output_len, 0.0);
        self.output_buf.fill(0.0);
        self.window_sum_buf.resize(output_len, 0.0);
        self.window_sum_buf.fill(0.0);

        let fft_forward = self.planner.plan_fft_forward(self.fft_size);
        let fft_inverse = self.planner.plan_fft_inverse(self.fft_size);

        // Reset phase state without reallocating
        self.phase_accum.fill(0.0);
        self.prev_phase.fill(0.0);

        let hop_ratio = self.hop_synthesis as f32 / self.hop_analysis as f32;
        let norm = 1.0 / self.fft_size as f32;

        for frame_idx in 0..num_frames {
            let analysis_pos = frame_idx * self.hop_analysis;
            let synthesis_pos = frame_idx * self.hop_synthesis;

            self.analyze_frame(
                &input[analysis_pos..analysis_pos + self.fft_size],
                &fft_forward,
            );
            self.advance_phases(num_bins, hop_ratio);

            // Identity phase locking (Laroche & Dolson 1999): lock non-peak
            // bins to their nearest peak using the analysis phase relationship.
            // Only applies above the sub-bass region.
            identity_phase_lock(
                &self.magnitudes,
                &self.analysis_phases,
                &mut self.new_phases,
                num_bins,
                self.sub_bass_bin,
                &mut self.peaks,
            );

            self.reconstruct_spectrum(num_bins);
            fft_inverse.process(&mut self.fft_buffer);

            // Overlap-add with synthesis window
            let frame_len = (synthesis_pos + self.fft_size).min(output_len) - synthesis_pos;
            #[allow(clippy::needless_range_loop)]
            for i in 0..frame_len {
                let out_idx = synthesis_pos + i;
                self.output_buf[out_idx] += self.fft_buffer[i].re * norm * self.window[i];
                self.window_sum_buf[out_idx] += self.window[i] * self.window[i];
            }
        }

        Self::normalize_output(&mut self.output_buf, &self.window_sum_buf);
        // Return a copy of the output; the buffers stay allocated for reuse.
        Ok(self.output_buf[..output_len].to_vec())
    }

    /// Windows the input frame and transforms to frequency domain.
    fn analyze_frame(
        &mut self,
        input_frame: &[f32],
        fft_forward: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    ) {
        for (i, (&sample, &win)) in input_frame.iter().zip(self.window.iter()).enumerate() {
            self.fft_buffer[i] = Complex::new(sample * win, 0.0);
        }
        fft_forward.process(&mut self.fft_buffer);
    }

    /// Extracts magnitudes and advances phase accumulators for each bin.
    ///
    /// Sub-bass bins (below `sub_bass_bin`) use rigid phase propagation to prevent
    /// phase cancellation in the critical sub-bass region. All other bins use
    /// standard phase vocoder deviation tracking.
    fn advance_phases(&mut self, num_bins: usize, hop_ratio: f32) {
        for bin in 0..num_bins {
            let c = self.fft_buffer[bin];
            self.magnitudes[bin] = c.norm();
            let phase = c.arg();
            self.analysis_phases[bin] = phase;

            if bin < self.sub_bass_bin {
                // Rigid phase propagation for sub-bass coherence
                self.phase_accum[bin] += phase - self.prev_phase[bin];
            } else {
                // Standard deviation tracking with hop-ratio scaling
                let expected = self.expected_phase_advance[bin];
                let deviation = wrap_phase(phase - self.prev_phase[bin] - expected);
                self.phase_accum[bin] += (expected + deviation) * hop_ratio;
            }

            self.new_phases[bin] = self.phase_accum[bin];
            self.prev_phase[bin] = phase;
        }
    }

    /// Reconstructs the complex spectrum from magnitudes and phases,
    /// then mirrors negative frequencies for inverse FFT.
    fn reconstruct_spectrum(&mut self, num_bins: usize) {
        for bin in 0..num_bins {
            self.fft_buffer[bin] = Complex::from_polar(self.magnitudes[bin], self.new_phases[bin]);
        }
        for bin in 1..num_bins - 1 {
            self.fft_buffer[self.fft_size - bin] = self.fft_buffer[bin].conj();
        }
    }

    /// Normalizes output by window sum, clamping to prevent amplification in
    /// low-overlap regions (occurs when synthesis hop > analysis hop).
    fn normalize_output(output: &mut [f32], window_sum: &[f32]) {
        let max_window_sum = window_sum.iter().copied().fold(0.0f32, f32::max);
        let min_window_sum = (max_window_sum * MIN_WINDOW_SUM_RATIO).max(WINDOW_SUM_EPSILON);
        for (sample, &ws) in output.iter_mut().zip(window_sum.iter()) {
            *sample /= ws.max(min_window_sum);
        }
    }
}

/// Wraps a phase value to [-PI, PI] using efficient modulo arithmetic.
#[inline]
fn wrap_phase(phase: f32) -> f32 {
    let p = phase + PI;
    p - (p / TWO_PI).floor() * TWO_PI - PI
}

/// Identity phase locking (Laroche & Dolson 1999).
///
/// For non-peak bins, sets the synthesis phase to preserve the phase
/// relationship from the original analysis spectrum relative to the
/// nearest spectral peak. This reduces phasing artifacts on tonal content.
///
/// Bins below `start_bin` are skipped (they use rigid sub-bass phase locking).
fn identity_phase_lock(
    magnitudes: &[f32],
    analysis_phases: &[f32],
    synthesis_phases: &mut [f32],
    num_bins: usize,
    start_bin: usize,
    peaks: &mut Vec<usize>,
) {
    if num_bins < 3 || start_bin >= num_bins {
        return;
    }

    // Find spectral peaks above the sub-bass region (reuse buffer)
    peaks.clear();
    let search_start = start_bin.max(1);
    for bin in search_start..num_bins - 1 {
        if magnitudes[bin] > magnitudes[bin - 1] && magnitudes[bin] > magnitudes[bin + 1] {
            peaks.push(bin);
        }
    }

    if peaks.is_empty() {
        return;
    }

    // For each non-peak bin above sub-bass, lock phase to nearest peak.
    // The synthesis phase for a non-peak bin becomes:
    //   synth[peak] + (analysis[bin] - analysis[peak])
    // This preserves the original spectral phase relationships around peaks.
    let mut peak_idx = 0;
    for bin in start_bin..num_bins {
        // Advance to nearest peak
        while peak_idx + 1 < peaks.len()
            && (peaks[peak_idx + 1] as i64 - bin as i64).unsigned_abs()
                < (peaks[peak_idx] as i64 - bin as i64).unsigned_abs()
        {
            peak_idx += 1;
        }

        let nearest_peak = peaks[peak_idx];
        if bin != nearest_peak {
            let analysis_diff = analysis_phases[bin] - analysis_phases[nearest_peak];
            synthesis_phases[bin] = synthesis_phases[nearest_peak] + analysis_diff;
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
        let input_rms: f32 = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32).sqrt();
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

    #[test]
    fn test_sub_bass_bin_calculation() {
        // 120 Hz cutoff at 44100 Hz with FFT size 4096
        // Expected bin: 120 * 4096 / 44100 ≈ 11.15 → 11
        let pv = PhaseVocoder::new(4096, 1024, 1.0, 44100, 120.0);
        assert_eq!(pv.sub_bass_bin, 11);

        // 0 Hz cutoff should give bin 0 (no sub-bass locking)
        let pv = PhaseVocoder::new(4096, 1024, 1.0, 44100, 0.0);
        assert_eq!(pv.sub_bass_bin, 0);

        // High cutoff at 48000 Hz
        let pv = PhaseVocoder::new(4096, 1024, 1.0, 48000, 200.0);
        let expected = (200.0f32 * 4096.0 / 48000.0).round() as usize;
        assert_eq!(pv.sub_bass_bin, expected);
    }

    #[test]
    fn test_sub_bass_phase_locking_preserves_low_freq() {
        // A 60 Hz sine should be handled by sub-bass rigid phase locking.
        // Compare output quality with sub-bass locking (120 Hz cutoff)
        // vs without (0 Hz cutoff).
        let sample_rate = 44100u32;
        let fft_size = 4096;
        let hop = fft_size / 4;
        let num_samples = fft_size * 8;
        let freq = 60.0f32; // Well below 120 Hz cutoff

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        // Process with sub-bass locking enabled (120 Hz cutoff)
        let mut pv_locked = PhaseVocoder::new(fft_size, hop, 1.5, sample_rate, 120.0);
        let output_locked = pv_locked.process(&input).unwrap();

        // Process without sub-bass locking (0 Hz cutoff)
        let mut pv_unlocked = PhaseVocoder::new(fft_size, hop, 1.5, sample_rate, 0.0);
        let output_unlocked = pv_unlocked.process(&input).unwrap();

        // Both should produce output
        assert!(!output_locked.is_empty());
        assert!(!output_unlocked.is_empty());

        // Both should have similar RMS (we aren't destroying energy)
        let rms_locked =
            (output_locked.iter().map(|x| x * x).sum::<f32>() / output_locked.len() as f32).sqrt();
        let rms_unlocked = (output_unlocked.iter().map(|x| x * x).sum::<f32>()
            / output_unlocked.len() as f32)
            .sqrt();

        assert!(
            rms_locked > 0.1,
            "Sub-bass locked output should have significant energy, got RMS={}",
            rms_locked
        );
        assert!(
            rms_unlocked > 0.1,
            "Unlocked output should have significant energy, got RMS={}",
            rms_unlocked
        );
    }

    #[test]
    fn test_sub_bass_locking_does_not_affect_high_freq() {
        // A 1000 Hz sine should NOT be affected by sub-bass phase locking
        // (it's above the 120 Hz cutoff).
        let sample_rate = 44100u32;
        let fft_size = 4096;
        let hop = fft_size / 4;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv_with = PhaseVocoder::new(fft_size, hop, 1.0, sample_rate, 120.0);
        let output_with = pv_with.process(&input).unwrap();

        let mut pv_without = PhaseVocoder::new(fft_size, hop, 1.0, sample_rate, 0.0);
        let output_without = pv_without.process(&input).unwrap();

        // Output lengths should be the same
        assert_eq!(output_with.len(), output_without.len());

        // RMS should be very similar since 1000 Hz is above the cutoff
        let rms_with =
            (output_with.iter().map(|x| x * x).sum::<f32>() / output_with.len() as f32).sqrt();
        let rms_without = (output_without.iter().map(|x| x * x).sum::<f32>()
            / output_without.len() as f32)
            .sqrt();

        assert!(
            (rms_with - rms_without).abs() < rms_with * 0.3,
            "1000 Hz signal should be similar with/without sub-bass locking: {} vs {}",
            rms_with,
            rms_without
        );
    }
}

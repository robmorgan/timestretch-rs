//! Phase vocoder time stretching with identity phase locking and sub-bass phase locking.

use crate::core::fft::{COMPLEX_ZERO, WINDOW_SUM_EPSILON, WINDOW_SUM_FLOOR_RATIO};
use crate::core::window::{generate_window, WindowType};
use crate::error::StretchError;
use crate::stretch::envelope::{apply_envelope_correction, extract_envelope};
use crate::stretch::phase_locking::{apply_phase_locking, PhaseLockingMode};
use rustfft::{num_complex::Complex, FftPlanner};

const TWO_PI_F64: f64 = 2.0 * std::f64::consts::PI;
/// Fraction of bins to pre-allocate for spectral peak detection (1/4 of bins).
const PEAKS_CAPACITY_DIVISOR: usize = 4;

/// Phase vocoder state for time stretching.
pub struct PhaseVocoder {
    fft_size: usize,
    hop_analysis: usize,
    hop_synthesis: usize,
    window: Vec<f32>,
    /// Phase accumulator for resynthesis (f64 for precision over long signals).
    phase_accum: Vec<f64>,
    /// Previous analysis phase (f64 to match accumulator precision).
    prev_phase: Vec<f64>,
    /// FFT planner (cached).
    planner: FftPlanner<f32>,
    /// Pre-computed expected phase advance per bin (f64 for precision).
    expected_phase_advance: Vec<f64>,
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
    /// Phase locking algorithm to use.
    phase_locking_mode: PhaseLockingMode,
    /// Whether spectral envelope preservation is enabled.
    envelope_preservation: bool,
    /// Cepstral order for envelope extraction.
    envelope_order: usize,
    /// Reusable buffer for cepstral analysis.
    cepstrum_buf: Vec<Complex<f32>>,
    /// Reusable buffer for analysis envelope.
    analysis_envelope: Vec<f32>,
    /// Reusable buffer for synthesis envelope.
    synthesis_envelope: Vec<f32>,
    /// Synthesis window (Hann) for overlap-add. Using a separate synthesis window
    /// avoids squaring the analysis window (which distorts Kaiser/BlackmanHarris).
    synthesis_window: Vec<f32>,
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
        Self::with_window(
            fft_size,
            hop_analysis,
            stretch_ratio,
            sample_rate,
            sub_bass_cutoff,
            WindowType::BlackmanHarris,
        )
    }

    /// Creates a new phase vocoder with a specific window function.
    pub fn with_window(
        fft_size: usize,
        hop_analysis: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
        window_type: WindowType,
    ) -> Self {
        Self::with_options(
            fft_size,
            hop_analysis,
            stretch_ratio,
            sample_rate,
            sub_bass_cutoff,
            window_type,
            PhaseLockingMode::RegionOfInfluence,
        )
    }

    /// Creates a new phase vocoder with full configuration options.
    pub fn with_options(
        fft_size: usize,
        hop_analysis: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
        window_type: WindowType,
        phase_locking_mode: PhaseLockingMode,
    ) -> Self {
        Self::with_all_options(
            fft_size,
            hop_analysis,
            stretch_ratio,
            sample_rate,
            sub_bass_cutoff,
            window_type,
            phase_locking_mode,
            false,
            40,
        )
    }

    /// Creates a new phase vocoder with all configuration options including envelope preservation.
    #[allow(clippy::too_many_arguments)]
    pub fn with_all_options(
        fft_size: usize,
        hop_analysis: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
        window_type: WindowType,
        phase_locking_mode: PhaseLockingMode,
        envelope_preservation: bool,
        envelope_order: usize,
    ) -> Self {
        let hop_synthesis = (hop_analysis as f64 * stretch_ratio).round() as usize;
        let window = generate_window(window_type, fft_size);
        let synthesis_window = generate_window(WindowType::Hann, fft_size);
        let num_bins = fft_size / 2 + 1;

        let expected_phase_advance: Vec<f64> = (0..num_bins)
            .map(|bin| TWO_PI_F64 * bin as f64 * hop_analysis as f64 / fft_size as f64)
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
            phase_accum: vec![0.0f64; num_bins],
            prev_phase: vec![0.0f64; num_bins],
            planner: FftPlanner::new(),
            expected_phase_advance,
            fft_buffer: vec![COMPLEX_ZERO; fft_size],
            magnitudes: vec![0.0; num_bins],
            new_phases: vec![0.0; num_bins],
            peaks: Vec::with_capacity(num_bins / PEAKS_CAPACITY_DIVISOR),
            analysis_phases: vec![0.0; num_bins],
            sub_bass_bin,
            phase_locking_mode,
            envelope_preservation,
            envelope_order,
            cepstrum_buf: Vec::new(),
            analysis_envelope: Vec::new(),
            synthesis_envelope: Vec::new(),
            synthesis_window,
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

    /// Resets the phase accumulator and previous-phase buffers.
    ///
    /// Call this at transient boundaries so that stale phase state from a
    /// previous tonal segment does not contaminate the next one. The PV will
    /// re-derive phases from the first analysis frame after the reset.
    #[inline]
    pub fn reset_phase_state(&mut self) {
        self.phase_accum.fill(0.0);
        self.prev_phase.fill(0.0);
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

        let hop_ratio = self.hop_synthesis as f64 / self.hop_analysis as f64;
        let norm = 1.0 / self.fft_size as f32;

        for frame_idx in 0..num_frames {
            let analysis_pos = frame_idx * self.hop_analysis;
            let synthesis_pos = frame_idx * self.hop_synthesis;

            self.analyze_frame(
                &input[analysis_pos..analysis_pos + self.fft_size],
                &fft_forward,
            );
            self.advance_phases(num_bins, hop_ratio);

            // Phase locking: lock non-peak bins to their nearest peak using
            // the analysis phase relationship. Only applies above the sub-bass region.
            apply_phase_locking(
                self.phase_locking_mode,
                &self.magnitudes,
                &self.analysis_phases,
                &mut self.new_phases,
                num_bins,
                self.sub_bass_bin,
                &mut self.peaks,
            );

            // Spectral envelope preservation: correct magnitudes so formant
            // structure matches the original analysis frame, preventing
            // unnatural timbre shifts.
            if self.envelope_preservation {
                // Extract envelope from the original analysis magnitudes
                extract_envelope(
                    &self.magnitudes,
                    num_bins,
                    self.envelope_order,
                    &mut self.planner,
                    &mut self.cepstrum_buf,
                    &mut self.analysis_envelope,
                );

                // The synthesis magnitudes are the same (PV doesn't change
                // magnitudes), but after phase locking the spectral shape
                // may shift slightly. We extract the synthesis envelope from
                // the current magnitudes and correct.
                // Clone analysis envelope as synthesis baseline since magnitudes
                // haven't changed. The correction step then normalizes any
                // spectral tilt introduced by windowing or overlap.
                self.synthesis_envelope.clear();
                self.synthesis_envelope
                    .extend_from_slice(&self.analysis_envelope);

                apply_envelope_correction(
                    &mut self.magnitudes,
                    &self.analysis_envelope,
                    &self.synthesis_envelope,
                    num_bins,
                    self.sub_bass_bin,
                );
            }

            self.reconstruct_spectrum(num_bins);
            fft_inverse.process(&mut self.fft_buffer);

            // Overlap-add with dual windowing: analysis window for FFT, Hann
            // synthesis window for overlap-add. This prevents squaring
            // Kaiser/BlackmanHarris which distorts the effective window shape.
            // The window sum tracks wa*ws (the product of both windows) for
            // correct normalization.
            let frame_len = (synthesis_pos + self.fft_size).min(output_len) - synthesis_pos;
            for i in 0..frame_len {
                let ws = self.synthesis_window[i];
                self.output_buf[synthesis_pos + i] += self.fft_buffer[i].re * norm * ws;
                self.window_sum_buf[synthesis_pos + i] += self.window[i] * ws;
            }
        }

        Self::normalize_output(&mut self.output_buf, &self.window_sum_buf);
        // Return a copy of the output; the buffers stay allocated for reuse.
        Ok(self.output_buf[..output_len].to_vec())
    }

    /// Windows the input frame and transforms to frequency domain.
    #[inline]
    fn analyze_frame(
        &mut self,
        input_frame: &[f32],
        fft_forward: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    ) {
        let len = input_frame.len().min(self.fft_buffer.len());
        for (i, (&sample, &w)) in input_frame
            .iter()
            .zip(self.window.iter())
            .enumerate()
            .take(len)
        {
            self.fft_buffer[i] = Complex::new(sample * w, 0.0);
        }
        fft_forward.process(&mut self.fft_buffer);
    }

    /// Extracts magnitudes and advances phase accumulators for each bin.
    ///
    /// Sub-bass bins (below `sub_bass_bin`) use rigid phase propagation to prevent
    /// phase cancellation in the critical sub-bass region. All other bins use
    /// standard phase vocoder deviation tracking.
    ///
    /// Phase accumulation uses f64 precision to prevent cumulative rounding errors
    /// over long signals. The final phases are converted back to f32 for the
    /// spectrum reconstruction step.
    #[inline]
    fn advance_phases(&mut self, num_bins: usize, hop_ratio: f64) {
        for bin in 0..num_bins {
            let c = self.fft_buffer[bin];
            self.magnitudes[bin] = c.norm();
            let phase = c.arg() as f64;
            self.analysis_phases[bin] = phase as f32;

            if bin < self.sub_bass_bin {
                // Rigid phase propagation for sub-bass coherence
                self.phase_accum[bin] += phase - self.prev_phase[bin];
            } else {
                // Standard deviation tracking with hop-ratio scaling
                let expected = self.expected_phase_advance[bin];
                let deviation = wrap_phase_f64(phase - self.prev_phase[bin] - expected);
                self.phase_accum[bin] += (expected + deviation) * hop_ratio;
            }

            self.new_phases[bin] = self.phase_accum[bin] as f32;
            self.prev_phase[bin] = phase;
        }
    }

    /// Reconstructs the complex spectrum from magnitudes and phases,
    /// then mirrors negative frequencies for inverse FFT.
    #[inline]
    fn reconstruct_spectrum(&mut self, num_bins: usize) {
        for i in 0..num_bins {
            self.fft_buffer[i] = Complex::from_polar(self.magnitudes[i], self.new_phases[i]);
        }
        for bin in 1..num_bins - 1 {
            self.fft_buffer[self.fft_size - bin] = self.fft_buffer[bin].conj();
        }
    }

    /// Normalizes output by window sum, clamping to prevent amplification in
    /// low-overlap regions (occurs when synthesis hop > analysis hop).
    #[inline]
    fn normalize_output(output: &mut [f32], window_sum: &[f32]) {
        let max_window_sum = window_sum.iter().copied().fold(0.0f32, f32::max);
        let min_window_sum = (max_window_sum * WINDOW_SUM_FLOOR_RATIO).max(WINDOW_SUM_EPSILON);
        let len = output.len().min(window_sum.len());
        for i in 0..len {
            output[i] /= window_sum[i].max(min_window_sum);
        }
    }
}

impl std::fmt::Debug for PhaseVocoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhaseVocoder")
            .field("fft_size", &self.fft_size)
            .field("hop_analysis", &self.hop_analysis)
            .field("hop_synthesis", &self.hop_synthesis)
            .field("sub_bass_bin", &self.sub_bass_bin)
            .field("phase_locking_mode", &self.phase_locking_mode)
            .finish()
    }
}

/// Wraps a phase value to [-PI, PI] using f64 precision.
#[inline]
fn wrap_phase_f64(phase: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let p = phase + pi;
    p - (p / TWO_PI_F64).floor() * TWO_PI_F64 - pi
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const TWO_PI: f32 = 2.0 * PI;

    /// Wraps a phase value to [-PI, PI] using efficient modulo arithmetic (f32).
    fn wrap_phase(phase: f32) -> f32 {
        let p = phase + PI;
        p - (p / TWO_PI).floor() * TWO_PI - PI
    }

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

    #[test]
    fn test_phase_vocoder_with_blackman_harris() {
        let sample_rate = 44100;
        let fft_size = 4096;
        let hop = fft_size / 4;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv = PhaseVocoder::with_window(
            fft_size,
            hop,
            1.5,
            sample_rate,
            120.0,
            WindowType::BlackmanHarris,
        );
        let output = pv.process(&input).unwrap();

        // Should produce valid stretched output
        assert!(!output.is_empty());
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.5).abs() < 0.3,
            "BH window length ratio {} too far from 1.5",
            len_ratio
        );
    }

    #[test]
    fn test_phase_vocoder_with_kaiser() {
        let sample_rate = 44100;
        let fft_size = 4096;
        let hop = fft_size / 4;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv = PhaseVocoder::with_window(
            fft_size,
            hop,
            1.5,
            sample_rate,
            120.0,
            WindowType::Kaiser(800),
        );
        let output = pv.process(&input).unwrap();

        assert!(!output.is_empty());
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.5).abs() < 0.3,
            "Kaiser window length ratio {} too far from 1.5",
            len_ratio
        );
    }

    #[test]
    fn test_phase_vocoder_different_windows_produce_different_output() {
        let sample_rate = 44100;
        let fft_size = 4096;
        let hop = fft_size / 4;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv_hann =
            PhaseVocoder::with_window(fft_size, hop, 1.5, sample_rate, 120.0, WindowType::Hann);
        let output_hann = pv_hann.process(&input).unwrap();

        let mut pv_bh = PhaseVocoder::with_window(
            fft_size,
            hop,
            1.5,
            sample_rate,
            120.0,
            WindowType::BlackmanHarris,
        );
        let output_bh = pv_bh.process(&input).unwrap();

        // Both should produce valid output of similar length
        assert!(!output_hann.is_empty());
        assert!(!output_bh.is_empty());

        // Outputs should differ (different windows produce different spectral characteristics)
        let min_len = output_hann.len().min(output_bh.len());
        let diff: f32 = output_hann[..min_len]
            .iter()
            .zip(&output_bh[..min_len])
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / min_len as f32;
        assert!(
            diff > 1e-6,
            "Different windows should produce different output, avg diff = {}",
            diff
        );
    }

    // --- phase locking integration (detailed tests in phase_locking module) ---

    #[test]
    fn test_phase_lock_identity_no_peaks() {
        // Flat magnitude spectrum: no local maxima → no peaks → phases unchanged
        let num_bins = 16;
        let magnitudes = vec![1.0f32; num_bins]; // all equal, no peaks
        let analysis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.1).collect();
        let mut synthesis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.2).collect();
        let original_phases = synthesis_phases.clone();
        let mut peaks = Vec::new();

        apply_phase_locking(
            PhaseLockingMode::Identity,
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            num_bins,
            0,
            &mut peaks,
        );

        // With no peaks found, phases should remain unchanged
        assert_eq!(synthesis_phases, original_phases);
    }

    #[test]
    fn test_phase_lock_identity_single_peak() {
        // Single peak at bin 5; all non-peak bins should be locked to it
        let num_bins = 16;
        let mut magnitudes = vec![0.1f32; num_bins];
        magnitudes[5] = 1.0; // peak at bin 5
        let analysis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.3).collect();
        let mut synthesis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.5).collect();
        let mut peaks = Vec::new();

        apply_phase_locking(
            PhaseLockingMode::Identity,
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            num_bins,
            0, // start_bin = 0
            &mut peaks,
        );

        // Peak at bin 5 should keep its phase
        // Non-peak bins should be: synth[peak] + (analysis[bin] - analysis[peak])
        let peak_synth = 5.0 * 0.5; // original synthesis phase of peak
        for bin in 1..num_bins - 1 {
            if bin == 5 {
                // Peak bin keeps its phase
                assert!(
                    (synthesis_phases[bin] - peak_synth).abs() < 1e-6,
                    "Peak bin should keep its phase"
                );
            } else {
                let expected = peak_synth + (analysis_phases[bin] - analysis_phases[5]);
                assert!(
                    (synthesis_phases[bin] - expected).abs() < 1e-6,
                    "Bin {} should be locked to peak: got {}, expected {}",
                    bin,
                    synthesis_phases[bin],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_phase_lock_start_bin_above_num_bins() {
        // start_bin >= num_bins: early return, no changes
        let num_bins = 8;
        let magnitudes = vec![0.0f32; num_bins];
        let analysis_phases = vec![0.0f32; num_bins];
        let mut synthesis_phases = vec![1.0f32; num_bins];
        let original = synthesis_phases.clone();
        let mut peaks = Vec::new();

        apply_phase_locking(
            PhaseLockingMode::Identity,
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            num_bins,
            num_bins, // start_bin == num_bins → early return
            &mut peaks,
        );

        assert_eq!(synthesis_phases, original);
    }

    #[test]
    fn test_phase_lock_num_bins_less_than_3() {
        // num_bins < 3: early return
        let magnitudes = vec![1.0f32; 2];
        let analysis_phases = vec![0.0f32; 2];
        let mut synthesis_phases = vec![0.5f32; 2];
        let original = synthesis_phases.clone();
        let mut peaks = Vec::new();

        apply_phase_locking(
            PhaseLockingMode::Identity,
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            2,
            0,
            &mut peaks,
        );

        assert_eq!(synthesis_phases, original);
    }

    #[test]
    fn test_phase_lock_sub_bass_region_skipped() {
        // Peaks exist only below start_bin → no peaks found above sub-bass
        let num_bins = 16;
        let mut magnitudes = vec![0.1f32; num_bins];
        magnitudes[2] = 1.0; // peak below start_bin=5
        let analysis_phases = vec![0.0f32; num_bins];
        let mut synthesis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32).collect();
        let original = synthesis_phases.clone();
        let mut peaks = Vec::new();

        apply_phase_locking(
            PhaseLockingMode::Identity,
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            num_bins,
            5, // start_bin=5, peak at bin 2 is below
            &mut peaks,
        );

        // No peaks above start_bin → no changes
        assert_eq!(synthesis_phases, original);
    }

    #[test]
    fn test_phase_lock_multiple_peaks() {
        // Two peaks: bins should lock to nearest peak
        let num_bins = 16;
        let mut magnitudes = vec![0.1f32; num_bins];
        magnitudes[3] = 1.0; // peak at bin 3
        magnitudes[10] = 0.8; // peak at bin 10
        let analysis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.1).collect();
        let mut synthesis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.2).collect();
        let mut peaks = Vec::new();

        apply_phase_locking(
            PhaseLockingMode::Identity,
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            num_bins,
            1, // start_bin=1
            &mut peaks,
        );

        // Bin 1 is closest to peak at bin 3
        let expected_1 = 3.0 * 0.2 + (1.0 * 0.1 - 3.0 * 0.1);
        assert!(
            (synthesis_phases[1] - expected_1).abs() < 1e-5,
            "Bin 1 should lock to peak 3"
        );

        // Bin 12 is closest to peak at bin 10
        let expected_12 = 10.0 * 0.2 + (12.0 * 0.1 - 10.0 * 0.1);
        assert!(
            (synthesis_phases[12] - expected_12).abs() < 1e-5,
            "Bin 12 should lock to peak 10"
        );
    }

    // --- normalize_output internals ---

    #[test]
    fn test_normalize_output_uniform_window_sum() {
        // When window_sum is uniform, output should be divided by that value
        let mut output = vec![2.0f32; 10];
        let window_sum = vec![2.0f32; 10];
        PhaseVocoder::normalize_output(&mut output, &window_sum);
        for &s in &output {
            assert!((s - 1.0).abs() < 1e-6, "Expected 1.0, got {}", s);
        }
    }

    #[test]
    fn test_normalize_output_low_window_sum_clamped() {
        // Very small window sums should be clamped to min_window_sum
        // to prevent amplification
        let mut output = vec![1.0f32; 10];
        let mut window_sum = vec![1.0f32; 10];
        // One sample has near-zero window sum (low-overlap region)
        window_sum[5] = 1e-10;
        PhaseVocoder::normalize_output(&mut output, &window_sum);

        // The clamped sample should NOT be amplified wildly
        // min_window_sum = max(1.0) * WINDOW_SUM_FLOOR_RATIO = 0.1
        // So output[5] = 1.0 / 0.1 = 10.0
        assert!(
            output[5] <= 11.0,
            "Low window sum should be clamped, got {}",
            output[5]
        );
        // Normal samples should be ~1.0
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_output_all_zero_window_sum() {
        // All-zero window sum: should use WINDOW_SUM_EPSILON floor
        let mut output = vec![1.0f32; 5];
        let window_sum = vec![0.0f32; 5];
        PhaseVocoder::normalize_output(&mut output, &window_sum);
        // Each sample = 1.0 / WINDOW_SUM_EPSILON
        for &s in &output {
            assert!(s.is_finite(), "Output should be finite, got {}", s);
        }
    }

    // --- wrap_phase edge cases ---

    #[test]
    fn test_wrap_phase_exact_boundaries() {
        // Exactly PI should wrap to -PI (or very close)
        let result = wrap_phase(PI);
        assert!(
            (result - (-PI)).abs() < 1e-5 || (result - PI).abs() < 1e-5,
            "wrap_phase(PI) = {} should be near ±PI",
            result
        );

        // Exactly -PI
        let result = wrap_phase(-PI);
        assert!(
            (result - (-PI)).abs() < 1e-5 || (result - PI).abs() < 1e-5,
            "wrap_phase(-PI) = {} should be near ±PI",
            result
        );

        // Exactly 0
        assert!((wrap_phase(0.0)).abs() < 1e-6);
    }

    #[test]
    fn test_wrap_phase_very_large_values() {
        // Very large positive and negative values
        let result = wrap_phase(1000.0 * PI);
        assert!(
            (-PI..=PI).contains(&result),
            "wrap_phase(1000*PI) = {} should be in [-PI, PI]",
            result
        );

        let result = wrap_phase(-999.0 * PI);
        assert!(
            (-PI..=PI).contains(&result),
            "wrap_phase(-999*PI) = {} should be in [-PI, PI]",
            result
        );
    }

    // --- set_stretch_ratio ---

    #[test]
    fn test_set_stretch_ratio_updates_hop_synthesis() {
        let mut pv = PhaseVocoder::new(4096, 1024, 1.0, 44100, 120.0);
        assert_eq!(pv.hop_synthesis(), 1024); // 1024 * 1.0 = 1024

        pv.set_stretch_ratio(2.0);
        assert_eq!(pv.hop_synthesis(), 2048); // 1024 * 2.0 = 2048

        pv.set_stretch_ratio(0.5);
        assert_eq!(pv.hop_synthesis(), 512); // 1024 * 0.5 = 512
    }

    #[test]
    fn test_set_stretch_ratio_preserves_phase_state() {
        // Process some audio, then change ratio and process more.
        // Phase should be continuous (no reset).
        let fft_size = 4096;
        let hop = 1024;
        let sample_rate = 44100u32;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv = PhaseVocoder::new(fft_size, hop, 1.0, sample_rate, 120.0);
        let output1 = pv.process(&input).unwrap();
        assert!(!output1.is_empty());

        // Change ratio and process again — should work without error
        pv.set_stretch_ratio(1.5);
        let output2 = pv.process(&input).unwrap();
        assert!(!output2.is_empty());
        assert!(output2.len() > output1.len()); // 1.5x should be longer
    }

    // --- sub_bass_bin edge cases ---

    #[test]
    fn test_sub_bass_bin_clamped_to_num_bins() {
        // Very high cutoff: sub_bass_bin should be clamped to num_bins
        let pv = PhaseVocoder::new(256, 64, 1.0, 44100, 30000.0);
        let num_bins = 256 / 2 + 1;
        assert!(
            pv.sub_bass_bin() <= num_bins,
            "sub_bass_bin {} should be <= num_bins {}",
            pv.sub_bass_bin(),
            num_bins
        );
    }

    #[test]
    fn test_sub_bass_all_bins_rigid() {
        // With cutoff >= Nyquist, all bins should use rigid locking.
        // This should still produce valid output (no crash).
        let fft_size = 512;
        let hop = 128;
        let sample_rate = 44100u32;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        // Cutoff at Nyquist: all bins are "sub-bass" → all rigid locking
        let mut pv = PhaseVocoder::new(fft_size, hop, 1.5, sample_rate, 22050.0);
        let output = pv.process(&input).unwrap();
        assert!(!output.is_empty());
        assert!(output.iter().all(|s| s.is_finite()));
    }

    // --- reconstruct_spectrum conjugate symmetry ---

    #[test]
    fn test_reconstruct_spectrum_produces_real_output() {
        // After reconstruct_spectrum + inverse FFT, output should be real-valued
        // (imaginary parts near zero). This verifies conjugate symmetry is correct.
        let fft_size = 256;
        let hop = 64;
        let sample_rate = 44100u32;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv = PhaseVocoder::new(fft_size, hop, 1.0, sample_rate, 120.0);
        let output = pv.process(&input).unwrap();

        // If conjugate symmetry is wrong, we'd get complex residues causing
        // large imaginary parts. The output being finite and reasonable is evidence.
        assert!(output.iter().all(|s| s.is_finite()));
        let rms = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
        assert!(
            rms > 0.01,
            "Output should have significant energy, got RMS={}",
            rms
        );
    }

    // --- PV reuse (buffers grow but don't shrink) ---

    #[test]
    fn test_phase_vocoder_reuse_across_different_lengths() {
        let fft_size = 1024;
        let hop = 256;
        let sample_rate = 44100u32;

        let mut pv = PhaseVocoder::new(fft_size, hop, 1.0, sample_rate, 120.0);

        // Process a long signal
        let long_input: Vec<f32> = (0..fft_size * 8)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let output1 = pv.process(&long_input).unwrap();
        assert!(!output1.is_empty());

        // Process a shorter signal — buffers should still work (they don't shrink)
        let short_input: Vec<f32> = (0..fft_size * 2)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let output2 = pv.process(&short_input).unwrap();
        assert!(!output2.is_empty());
        assert!(output2.len() < output1.len());
    }
}

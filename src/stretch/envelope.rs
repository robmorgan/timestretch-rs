//! Spectral envelope extraction and correction via real cepstrum.
//!
//! Preserves the spectral envelope (formant structure) during time stretching,
//! preventing the unnatural timbre shifts that occur when the phase vocoder
//! modifies magnitude relationships between harmonics.

use rustfft::{num_complex::Complex, FftPlanner};

/// Minimum magnitude floor to avoid log(0) in cepstral analysis.
const LOG_FLOOR: f32 = 1e-10;

/// Extracts the spectral envelope from a magnitude spectrum using real cepstrum.
///
/// The cepstral method:
/// 1. Take log of magnitude spectrum
/// 2. IFFT to get cepstrum
/// 3. Lifter: zero out high-quefrency components (keep only first `order` coefficients)
/// 4. FFT back to get smoothed log-spectrum
/// 5. Exponentiate to get the envelope
///
/// `order` controls the smoothness: lower = smoother envelope, higher = more detail.
/// Typical values: 30-50 for speech/vocals, 20-30 for music.
pub fn extract_envelope(
    magnitudes: &[f32],
    num_bins: usize,
    order: usize,
    planner: &mut FftPlanner<f32>,
    cepstrum_buf: &mut Vec<Complex<f32>>,
    envelope_out: &mut Vec<f32>,
) {
    let fft_size = (num_bins - 1) * 2;

    // Resize buffers
    cepstrum_buf.resize(fft_size, Complex::new(0.0, 0.0));
    envelope_out.resize(num_bins, 1.0);

    // Step 1: Log magnitude spectrum (mirror for full FFT)
    for i in 0..num_bins {
        let log_mag = magnitudes[i].max(LOG_FLOOR).ln();
        cepstrum_buf[i] = Complex::new(log_mag, 0.0);
    }
    // Mirror negative frequencies
    for i in 1..num_bins - 1 {
        cepstrum_buf[fft_size - i] = cepstrum_buf[i];
    }

    // Step 2: IFFT to get cepstrum
    let ifft = planner.plan_fft_inverse(fft_size);
    ifft.process(cepstrum_buf);

    let norm = 1.0 / fft_size as f32;

    // Step 3: Lifter - keep only low-quefrency components
    // Keep bins 0..order and (fft_size-order+1)..fft_size (conjugate mirror)
    let effective_order = order.min(fft_size / 2);
    for i in 0..fft_size {
        if i > effective_order && i < fft_size - effective_order {
            cepstrum_buf[i] = Complex::new(0.0, 0.0);
        } else {
            cepstrum_buf[i] *= norm; // Normalize IFFT
        }
    }

    // Step 4: FFT back to get smoothed log-spectrum
    let fft = planner.plan_fft_forward(fft_size);
    fft.process(cepstrum_buf);

    // Step 5: Exponentiate to get envelope
    for i in 0..num_bins {
        envelope_out[i] = cepstrum_buf[i].re.exp();
    }
}

/// Applies spectral envelope correction to magnitudes.
///
/// Adjusts the output magnitudes so the spectral envelope matches the
/// analysis (input) envelope. This preserves formant structure:
///
/// `corrected[bin] = magnitude[bin] * analysis_envelope[bin] / synthesis_envelope[bin]`
///
/// The `synthesis_envelope` is computed from the current (potentially shifted)
/// magnitudes, and `analysis_envelope` from the original analysis frame.
#[inline]
pub fn apply_envelope_correction(
    magnitudes: &mut [f32],
    analysis_envelope: &[f32],
    synthesis_envelope: &[f32],
    num_bins: usize,
    start_bin: usize,
) {
    for bin in start_bin..num_bins {
        let synth_env = synthesis_envelope[bin].max(LOG_FLOOR);
        let correction = analysis_envelope[bin] / synth_env;
        // Clamp correction to avoid extreme amplification
        let clamped = correction.clamp(0.1, 10.0);
        magnitudes[bin] *= clamped;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_extraction_flat_spectrum() {
        // Flat magnitude spectrum should produce flat envelope
        let num_bins = 129; // FFT size 256
        let magnitudes = vec![1.0f32; num_bins];
        let mut planner = FftPlanner::new();
        let mut cepstrum_buf = Vec::new();
        let mut envelope = Vec::new();

        extract_envelope(&magnitudes, num_bins, 30, &mut planner, &mut cepstrum_buf, &mut envelope);

        // Envelope should be approximately 1.0 everywhere
        for (i, &e) in envelope.iter().enumerate() {
            assert!(
                (e - 1.0).abs() < 0.1,
                "Envelope at bin {} should be ~1.0, got {}",
                i, e
            );
        }
    }

    #[test]
    fn test_envelope_extraction_peaked_spectrum() {
        // Spectrum with a clear peak should have envelope following the peak
        let num_bins = 129;
        let mut magnitudes = vec![0.1f32; num_bins];
        // Create a broad peak around bin 30
        for i in 20..40 {
            magnitudes[i] = 1.0 - ((i as f32 - 30.0) / 10.0).powi(2);
            magnitudes[i] = magnitudes[i].max(0.1);
        }

        let mut planner = FftPlanner::new();
        let mut cepstrum_buf = Vec::new();
        let mut envelope = Vec::new();

        extract_envelope(&magnitudes, num_bins, 20, &mut planner, &mut cepstrum_buf, &mut envelope);

        // Envelope at the peak should be higher than at the edges
        assert!(
            envelope[30] > envelope[0] * 1.5,
            "Peak envelope {} should be higher than edge {}",
            envelope[30],
            envelope[0]
        );
    }

    #[test]
    fn test_envelope_correction_identity() {
        // When analysis and synthesis envelopes are the same, no change
        let num_bins = 64;
        let envelope = vec![2.0f32; num_bins];
        let mut magnitudes = vec![1.0f32; num_bins];
        let original = magnitudes.clone();

        apply_envelope_correction(&mut magnitudes, &envelope, &envelope, num_bins, 0);

        for i in 0..num_bins {
            assert!(
                (magnitudes[i] - original[i]).abs() < 1e-6,
                "Magnitude at bin {} should be unchanged",
                i
            );
        }
    }

    #[test]
    fn test_envelope_correction_scales() {
        // If analysis envelope is 2x synthesis, magnitudes should double
        let num_bins = 64;
        let analysis_env = vec![2.0f32; num_bins];
        let synthesis_env = vec![1.0f32; num_bins];
        let mut magnitudes = vec![1.0f32; num_bins];

        apply_envelope_correction(&mut magnitudes, &analysis_env, &synthesis_env, num_bins, 0);

        for i in 0..num_bins {
            assert!(
                (magnitudes[i] - 2.0).abs() < 1e-6,
                "Magnitude at bin {} should be 2.0, got {}",
                i,
                magnitudes[i]
            );
        }
    }

    #[test]
    fn test_envelope_correction_clamped() {
        // Extreme ratio should be clamped to 10.0
        let num_bins = 16;
        let analysis_env = vec![100.0f32; num_bins];
        let synthesis_env = vec![0.01f32; num_bins];
        let mut magnitudes = vec![1.0f32; num_bins];

        apply_envelope_correction(&mut magnitudes, &analysis_env, &synthesis_env, num_bins, 0);

        for i in 0..num_bins {
            assert!(
                magnitudes[i] <= 10.0 + 1e-6,
                "Magnitude at bin {} should be clamped to 10.0, got {}",
                i,
                magnitudes[i]
            );
        }
    }
}

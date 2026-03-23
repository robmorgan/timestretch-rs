//! Spectral envelope extraction and correction via real cepstrum.
//!
//! Preserves the spectral envelope (formant structure) during time stretching,
//! preventing the unnatural timbre shifts that occur when the phase vocoder
//! modifies magnitude relationships between harmonics.

use rustfft::{num_complex::Complex, FftPlanner};

/// Minimum magnitude floor to avoid log(0) in cepstral analysis.
const LOG_FLOOR: f32 = 1e-10;

/// Computes the spectral centroid from a magnitude spectrum.
///
/// The spectral centroid is the weighted mean of frequencies, where the weights
/// are the magnitudes. A low centroid indicates bass-heavy content, a high
/// centroid indicates bright/noisy content.
///
/// Returns the centroid in Hz. Falls back to 1000 Hz if the spectrum is silent.
pub fn spectral_centroid(magnitudes: &[f32], sample_rate: u32, fft_size: usize) -> f32 {
    let mut weighted_sum = 0.0f64;
    let mut magnitude_sum = 0.0f64;
    let bin_freq = sample_rate as f64 / fft_size as f64;

    for (i, &mag) in magnitudes.iter().enumerate() {
        let freq = i as f64 * bin_freq;
        weighted_sum += freq * mag as f64;
        magnitude_sum += mag as f64;
    }

    if magnitude_sum > 1e-10 {
        (weighted_sum / magnitude_sum) as f32
    } else {
        1000.0 // default fallback
    }
}

/// Chooses a cepstral order based on the spectral centroid of the current frame.
///
/// Low centroid (bass-heavy): fewer coefficients for a smoother envelope that
/// doesn't track individual harmonics. High centroid (vocal/bright): more
/// coefficients to preserve formant structure.
///
/// The result is clamped to `[10, fft_size / 4]`.
pub fn adaptive_cepstral_order(centroid: f32, fft_size: usize) -> usize {
    // Low centroid (bass-heavy): need fewer coefficients (smoother envelope)
    // High centroid (bright/vocal): need more coefficients (preserve formants)
    let order = if centroid < 500.0 {
        25 // Bass-heavy: smooth envelope, don't track individual harmonics
    } else if centroid < 1500.0 {
        35 // Mid-range: moderate detail
    } else if centroid < 4000.0 {
        50 // Vocal/bright: preserve formant structure
    } else {
        40 // Very bright: moderate (noise-like, less structure)
    };

    // Clamp to valid range relative to FFT size
    order.min(fft_size / 4).max(10)
}

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
    for (i, c) in cepstrum_buf.iter_mut().enumerate().take(fft_size) {
        if i > effective_order && i < fft_size - effective_order {
            *c = Complex::new(0.0, 0.0);
        } else {
            *c *= norm; // Normalize IFFT
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

/// Allocation-free envelope extraction variant for real-time paths.
///
/// Uses preplanned FFT handles plus caller-provided scratch buffers, avoiding
/// heap work in steady-state audio callbacks.
#[allow(clippy::too_many_arguments)]
pub fn extract_envelope_with_fft_scratch(
    magnitudes: &[f32],
    num_bins: usize,
    order: usize,
    fft_forward: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    fft_inverse: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    fft_forward_scratch: &mut Vec<Complex<f32>>,
    fft_inverse_scratch: &mut Vec<Complex<f32>>,
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
    let inv_need = fft_inverse.get_inplace_scratch_len();
    if fft_inverse_scratch.len() < inv_need {
        fft_inverse_scratch.resize(inv_need, Complex::new(0.0, 0.0));
    }
    fft_inverse.process_with_scratch(cepstrum_buf, &mut fft_inverse_scratch[..inv_need]);

    let norm = 1.0 / fft_size as f32;

    // Step 3: Lifter - keep only low-quefrency components
    let effective_order = order.min(fft_size / 2);
    for (i, c) in cepstrum_buf.iter_mut().enumerate().take(fft_size) {
        if i > effective_order && i < fft_size - effective_order {
            *c = Complex::new(0.0, 0.0);
        } else {
            *c *= norm;
        }
    }

    // Step 4: FFT back to get smoothed log-spectrum
    let fwd_need = fft_forward.get_inplace_scratch_len();
    if fft_forward_scratch.len() < fwd_need {
        fft_forward_scratch.resize(fwd_need, Complex::new(0.0, 0.0));
    }
    fft_forward.process_with_scratch(cepstrum_buf, &mut fft_forward_scratch[..fwd_need]);

    // Step 5: Exponentiate to get envelope
    for i in 0..num_bins {
        envelope_out[i] = cepstrum_buf[i].re.exp();
    }
}

/// Extracts the spectral envelope using an adaptive cepstral order.
///
/// Computes the spectral centroid of the current frame's magnitudes and selects
/// an appropriate cepstral order: fewer coefficients for bass-heavy content
/// (smoother envelope), more for vocal/bright content (preserves formants).
///
/// If `override_order` is `Some(order)`, the fixed order is used instead of the
/// adaptive one, providing backward compatibility.
#[allow(clippy::too_many_arguments)]
pub fn extract_envelope_adaptive(
    magnitudes: &[f32],
    num_bins: usize,
    override_order: Option<usize>,
    sample_rate: u32,
    fft_size: usize,
    planner: &mut FftPlanner<f32>,
    cepstrum_buf: &mut Vec<Complex<f32>>,
    envelope_out: &mut Vec<f32>,
) {
    let order = match override_order {
        Some(o) => o,
        None => {
            let centroid = spectral_centroid(magnitudes, sample_rate, fft_size);
            adaptive_cepstral_order(centroid, fft_size)
        }
    };

    extract_envelope(
        magnitudes,
        num_bins,
        order,
        planner,
        cepstrum_buf,
        envelope_out,
    );
}

/// Estimates the noise floor of a magnitude spectrum as the 10th percentile value.
///
/// Sorts a copy of the magnitudes and returns the value at the 10th percentile
/// position. This provides a robust noise floor estimate that is not affected
/// by strong spectral peaks.
fn estimate_noise_floor(
    magnitudes: &[f32],
    start_bin: usize,
    num_bins: usize,
    scratch: &mut Vec<f32>,
) -> f32 {
    if start_bin >= num_bins {
        return LOG_FLOOR;
    }
    scratch.clear();
    scratch.extend(
        magnitudes[start_bin..num_bins]
            .iter()
            .copied()
            .filter(|&m| m > LOG_FLOOR),
    );
    if scratch.is_empty() {
        return LOG_FLOOR;
    }
    scratch.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // 10th percentile
    let idx = (scratch.len() as f64 * 0.10) as usize;
    scratch[idx.min(scratch.len() - 1)]
}

/// Computes a per-bin SNR-aware correction clamp.
///
/// Strong signals (high SNR) can tolerate moderate correction (up to 3x).
/// Medium signals allow up to 2x. Weak signals near the noise floor are
/// limited to 1.5x to avoid amplifying noise.
#[inline]
fn clamp_correction(correction: f32, magnitude: f32, noise_floor: f32) -> f32 {
    // Only apply significant correction to bins above noise floor
    let snr = if noise_floor > 1e-10 {
        magnitude / noise_floor
    } else {
        100.0 // assume high SNR if no noise estimate
    };

    let max_correction = if snr > 10.0 {
        3.0 // Strong signal: allow moderate correction
    } else if snr > 3.0 {
        2.0 // Medium signal: conservative correction
    } else {
        1.5 // Weak/near noise: minimal correction to avoid amplifying noise
    };

    correction.clamp(1.0 / max_correction, max_correction)
}

/// Applies spectral envelope correction to magnitudes with SNR-aware clamping.
///
/// Adjusts the output magnitudes so the spectral envelope matches the
/// analysis (input) envelope. This preserves formant structure:
///
/// `corrected[bin] = magnitude[bin] * analysis_envelope[bin] / synthesis_envelope[bin]`
///
/// The `synthesis_envelope` is computed from the current (potentially shifted)
/// magnitudes, and `analysis_envelope` from the original analysis frame.
///
/// Uses per-bin SNR-aware clamping: bins with strong signal allow up to 3x
/// correction, while bins near the noise floor are limited to 1.5x to avoid
/// amplifying noise.
#[inline]
pub fn apply_envelope_correction(
    magnitudes: &mut [f32],
    analysis_envelope: &[f32],
    synthesis_envelope: &[f32],
    num_bins: usize,
    start_bin: usize,
) {
    let mut scratch = Vec::new();
    apply_envelope_correction_with_scratch(
        magnitudes,
        analysis_envelope,
        synthesis_envelope,
        num_bins,
        start_bin,
        &mut scratch,
    );
}

/// Allocation-free variant of [`apply_envelope_correction`] for real-time paths.
///
/// Uses caller-provided scratch storage to estimate the noise floor without
/// heap allocations.
#[inline]
pub fn apply_envelope_correction_with_scratch(
    magnitudes: &mut [f32],
    analysis_envelope: &[f32],
    synthesis_envelope: &[f32],
    num_bins: usize,
    start_bin: usize,
    noise_floor_scratch: &mut Vec<f32>,
) {
    let noise_floor = estimate_noise_floor(magnitudes, start_bin, num_bins, noise_floor_scratch);

    for bin in start_bin..num_bins {
        let synth_env = synthesis_envelope[bin].max(LOG_FLOOR);
        let correction = analysis_envelope[bin] / synth_env;
        // SNR-aware clamping: tighter limits for bins near noise floor
        let clamped = clamp_correction(correction, magnitudes[bin], noise_floor);
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

        extract_envelope(
            &magnitudes,
            num_bins,
            30,
            &mut planner,
            &mut cepstrum_buf,
            &mut envelope,
        );

        // Envelope should be approximately 1.0 everywhere
        for (i, &e) in envelope.iter().enumerate() {
            assert!(
                (e - 1.0).abs() < 0.1,
                "Envelope at bin {} should be ~1.0, got {}",
                i,
                e
            );
        }
    }

    #[test]
    fn test_envelope_extraction_peaked_spectrum() {
        // Spectrum with a clear peak should have envelope following the peak
        let num_bins = 129;
        let mut magnitudes = vec![0.1f32; num_bins];
        // Create a broad peak around bin 30
        for (i, mag) in magnitudes.iter_mut().enumerate().take(40).skip(20) {
            *mag = 1.0 - ((i as f32 - 30.0) / 10.0).powi(2);
            *mag = mag.max(0.1);
        }

        let mut planner = FftPlanner::new();
        let mut cepstrum_buf = Vec::new();
        let mut envelope = Vec::new();

        extract_envelope(
            &magnitudes,
            num_bins,
            20,
            &mut planner,
            &mut cepstrum_buf,
            &mut envelope,
        );

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
        // If analysis envelope is 2x synthesis, magnitudes should scale by 2.0
        // for bins with high SNR. We create a spectrum where most bins are loud
        // (1.0) and a few are very quiet (noise floor), so the loud bins have
        // high SNR and allow the full 2.0x correction.
        let num_bins = 64;
        let analysis_env = vec![2.0f32; num_bins];
        let synthesis_env = vec![1.0f32; num_bins];
        let mut magnitudes = vec![1.0f32; num_bins];
        // Set ~10% of bins to near-zero to establish a low noise floor
        for mag in magnitudes.iter_mut().take(7) {
            *mag = 1e-6;
        }

        apply_envelope_correction(&mut magnitudes, &analysis_env, &synthesis_env, num_bins, 0);

        // The loud bins (index 7+) should have high SNR and get the full 2.0x correction
        for (i, &mag) in magnitudes.iter().enumerate().take(num_bins).skip(7) {
            assert!(
                (mag - 2.0).abs() < 1e-6,
                "Magnitude at bin {} should be 2.0, got {}",
                i,
                mag
            );
        }
    }

    #[test]
    fn test_envelope_correction_clamped() {
        // Extreme ratio should be clamped to 3.0 (strong signal SNR-aware clamp)
        let num_bins = 16;
        let analysis_env = vec![100.0f32; num_bins];
        let synthesis_env = vec![0.01f32; num_bins];
        let mut magnitudes = vec![1.0f32; num_bins];

        apply_envelope_correction(&mut magnitudes, &analysis_env, &synthesis_env, num_bins, 0);

        for (i, &mag) in magnitudes.iter().enumerate().take(num_bins) {
            assert!(
                mag <= 3.0 + 1e-6,
                "Magnitude at bin {} should be clamped to 3.0, got {}",
                i,
                mag
            );
        }
    }

    #[test]
    fn test_spectral_centroid_flat() {
        // Flat spectrum: centroid should be near the center frequency
        let num_bins = 129;
        let magnitudes = vec![1.0f32; num_bins];
        let sample_rate = 44100;
        let fft_size = 256;

        let centroid = spectral_centroid(&magnitudes, sample_rate, fft_size);

        // For a flat spectrum, centroid should be near (num_bins-1)/2 * bin_freq
        let bin_freq = sample_rate as f32 / fft_size as f32;
        let expected = (num_bins - 1) as f32 / 2.0 * bin_freq;
        assert!(
            (centroid - expected).abs() < expected * 0.1,
            "Flat spectrum centroid {} should be near {:.0}",
            centroid,
            expected
        );
    }

    #[test]
    fn test_spectral_centroid_bass_heavy() {
        // Bass-heavy spectrum: centroid should be low
        let num_bins = 129;
        let mut magnitudes = vec![0.001f32; num_bins];
        // Strong energy in first 10 bins
        for mag in magnitudes.iter_mut().take(10) {
            *mag = 1.0;
        }
        let sample_rate = 44100;
        let fft_size = 256;

        let centroid = spectral_centroid(&magnitudes, sample_rate, fft_size);
        assert!(
            centroid < 1000.0,
            "Bass-heavy centroid {} should be < 1000 Hz",
            centroid
        );
    }

    #[test]
    fn test_adaptive_cepstral_order_bass() {
        let order = adaptive_cepstral_order(200.0, 4096);
        assert_eq!(order, 25, "Bass content should get order 25");
    }

    #[test]
    fn test_adaptive_cepstral_order_vocal() {
        let order = adaptive_cepstral_order(2000.0, 4096);
        assert_eq!(order, 50, "Vocal content should get order 50");
    }

    #[test]
    fn test_adaptive_cepstral_order_clamped() {
        // Small FFT: order should be clamped to fft_size/4
        let order = adaptive_cepstral_order(2000.0, 64);
        assert_eq!(order, 16, "Should be clamped to fft_size/4 = 16");
    }

    #[test]
    fn test_extract_envelope_adaptive_override() {
        // With override_order, should behave identically to extract_envelope
        let num_bins = 129;
        let magnitudes = vec![1.0f32; num_bins];
        let mut planner = FftPlanner::new();
        let mut cepstrum_buf1 = Vec::new();
        let mut envelope1 = Vec::new();
        let mut cepstrum_buf2 = Vec::new();
        let mut envelope2 = Vec::new();

        extract_envelope(
            &magnitudes,
            num_bins,
            30,
            &mut planner,
            &mut cepstrum_buf1,
            &mut envelope1,
        );

        extract_envelope_adaptive(
            &magnitudes,
            num_bins,
            Some(30),
            44100,
            256,
            &mut planner,
            &mut cepstrum_buf2,
            &mut envelope2,
        );

        for (i, (&e1, &e2)) in envelope1.iter().zip(envelope2.iter()).enumerate() {
            assert!(
                (e1 - e2).abs() < 1e-6,
                "Envelope mismatch at bin {}: {} vs {}",
                i,
                e1,
                e2
            );
        }
    }

    #[test]
    fn test_snr_aware_clamp_strong_signal() {
        // Strong signal (high SNR) should allow up to 3.0x correction
        let result = clamp_correction(5.0, 1.0, 0.01);
        assert!(
            (result - 3.0).abs() < 1e-6,
            "Strong signal correction should clamp to 3.0, got {}",
            result
        );
    }

    #[test]
    fn test_snr_aware_clamp_weak_signal() {
        // Weak signal (low SNR) should limit to 1.5x correction
        let result = clamp_correction(5.0, 0.02, 0.01);
        assert!(
            (result - 1.5).abs() < 1e-6,
            "Weak signal correction should clamp to 1.5, got {}",
            result
        );
    }

    #[test]
    fn test_snr_aware_clamp_medium_signal() {
        // Medium signal should limit to 2.0x correction
        let result = clamp_correction(5.0, 0.05, 0.01);
        assert!(
            (result - 2.0).abs() < 1e-6,
            "Medium signal correction should clamp to 2.0, got {}",
            result
        );
    }

    #[test]
    fn test_estimate_noise_floor() {
        // Mostly low values with a few peaks: noise floor should be near the low values
        let mut magnitudes = vec![0.01f32; 100];
        magnitudes[50] = 1.0;
        magnitudes[51] = 0.8;
        magnitudes[52] = 0.6;

        let mut scratch = Vec::new();
        let floor = estimate_noise_floor(&magnitudes, 0, 100, &mut scratch);
        assert!(
            floor < 0.05,
            "Noise floor should be near 0.01, got {}",
            floor
        );
    }
}

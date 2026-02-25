//! Harmonic-Percussive Source Separation (HPSS) via median filtering.
//!
//! Separates audio into harmonic (sustained tones) and percussive (transient)
//! components using median filtering on the spectrogram magnitude. Harmonic
//! content is identified by temporal continuity (horizontal median), while
//! percussive content is identified by spectral broadband energy (vertical median).
//!
//! Used as a pre-processing step for the hybrid stretcher: harmonic components
//! are PV-stretched (excellent for sustained tones) while percussive components
//! are WSOLA-stretched (preserves transient detail).

use crate::core::fft::COMPLEX_ZERO;
use crate::core::window::{generate_window, WindowType};
use rustfft::{num_complex::Complex, FftPlanner};

/// Parameters for HPSS processing.
#[derive(Debug, Clone, Copy)]
pub struct HpssParams {
    /// Width of the horizontal (time) median filter in frames.
    /// Larger values better isolate harmonic content but increase latency.
    /// Default: 17 frames (~200ms at typical hop sizes).
    pub harmonic_width: usize,
    /// Width of the vertical (frequency) median filter in bins.
    /// Larger values better isolate percussive content.
    /// Default: 17 bins.
    pub percussive_width: usize,
}

impl Default for HpssParams {
    fn default() -> Self {
        Self {
            harmonic_width: 17,
            percussive_width: 17,
        }
    }
}

/// Separates audio into harmonic and percussive components using HPSS.
///
/// Returns `(harmonic, percussive)` where both vectors have the same length
/// as the input. Uses Wiener-like soft masking for smooth separation.
///
/// # Arguments
///
/// * `input` - Mono audio samples
/// * `fft_size` - FFT window size (e.g., 4096)
/// * `hop_size` - Hop size for STFT (e.g., fft_size / 4)
/// * `params` - HPSS parameters controlling median filter widths
pub fn hpss(
    input: &[f32],
    fft_size: usize,
    hop_size: usize,
    params: &HpssParams,
) -> (Vec<f32>, Vec<f32>) {
    if input.len() < fft_size {
        return (input.to_vec(), vec![0.0; input.len()]);
    }

    let num_bins = fft_size / 2 + 1;
    let num_frames = (input.len() - fft_size) / hop_size + 1;

    if num_frames == 0 {
        return (input.to_vec(), vec![0.0; input.len()]);
    }

    let window = generate_window(WindowType::Hann, fft_size);
    let mut planner = FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(fft_size);
    let fft_inv = planner.plan_fft_inverse(fft_size);
    let norm = 1.0 / fft_size as f32;

    // Step 1: Compute STFT — store complex spectrogram
    let mut spectrogram: Vec<Vec<Complex<f32>>> = Vec::with_capacity(num_frames);
    let mut magnitudes: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
    let mut fft_buf = vec![COMPLEX_ZERO; fft_size];

    for frame_idx in 0..num_frames {
        let pos = frame_idx * hop_size;
        let frame_end = (pos + fft_size).min(input.len());
        let frame_len = frame_end - pos;

        // Window and zero-pad
        for i in 0..fft_size {
            fft_buf[i] = if i < frame_len {
                Complex::new(input[pos + i] * window[i], 0.0)
            } else {
                COMPLEX_ZERO
            };
        }
        fft_fwd.process(&mut fft_buf);

        let mags: Vec<f32> = fft_buf[..num_bins].iter().map(|c| c.norm()).collect();
        magnitudes.push(mags);
        spectrogram.push(fft_buf[..num_bins].to_vec());
    }

    // Step 2: Apply median filters to magnitude spectrogram
    // Horizontal median (across time) → harmonic mask
    let harmonic_mags = median_filter_horizontal(&magnitudes, params.harmonic_width);
    // Vertical median (across frequency) → percussive mask
    let percussive_mags = median_filter_vertical(&magnitudes, params.percussive_width);

    // Step 3: Wiener-like soft masking
    // H_mask = H^2 / (H^2 + P^2 + eps), P_mask = 1 - H_mask
    let eps = 1e-10f32;

    // Step 4: Apply masks and ISTFT
    let output_len = input.len();
    let mut harmonic_out = vec![0.0f32; output_len];
    let mut percussive_out = vec![0.0f32; output_len];
    let mut window_sum = vec![0.0f32; output_len];

    let mut h_buf = vec![COMPLEX_ZERO; fft_size];
    let mut p_buf = vec![COMPLEX_ZERO; fft_size];

    for frame_idx in 0..num_frames {
        let pos = frame_idx * hop_size;
        let frame_end = (pos + fft_size).min(output_len);
        let frame_len = frame_end - pos;

        // Apply Wiener masks to complex spectrum
        for bin in 0..num_bins {
            let h = harmonic_mags[frame_idx][bin];
            let p = percussive_mags[frame_idx][bin];
            let h2 = h * h;
            let p2 = p * p;
            let denom = h2 + p2 + eps;
            let h_mask = h2 / denom;
            let p_mask = p2 / denom;

            h_buf[bin] = spectrogram[frame_idx][bin] * h_mask;
            p_buf[bin] = spectrogram[frame_idx][bin] * p_mask;

            // Mirror negative frequencies
            if bin > 0 && bin < num_bins - 1 {
                h_buf[fft_size - bin] = h_buf[bin].conj();
                p_buf[fft_size - bin] = p_buf[bin].conj();
            }
        }

        // Zero remaining bins
        for i in num_bins..(fft_size - num_bins + 1) {
            h_buf[i] = COMPLEX_ZERO;
            p_buf[i] = COMPLEX_ZERO;
        }

        // ISTFT
        fft_inv.process(&mut h_buf);
        fft_inv.process(&mut p_buf);

        // Overlap-add with synthesis window
        for i in 0..frame_len {
            let out_idx = pos + i;
            harmonic_out[out_idx] += h_buf[i].re * norm * window[i];
            percussive_out[out_idx] += p_buf[i].re * norm * window[i];
            window_sum[out_idx] += window[i] * window[i];
        }
    }

    // Normalize by window sum
    let max_ws = window_sum.iter().copied().fold(0.0f32, f32::max);
    let min_ws = (max_ws * 0.01).max(1e-8);
    for i in 0..output_len {
        let ws = window_sum[i].max(min_ws);
        harmonic_out[i] /= ws;
        percussive_out[i] /= ws;
    }

    (harmonic_out, percussive_out)
}

/// Applies a horizontal (time-axis) median filter to the magnitude spectrogram.
///
/// For each bin, the median is computed over a window of `width` frames centered
/// on the current frame. This enhances temporally stable (harmonic) components.
fn median_filter_horizontal(mags: &[Vec<f32>], width: usize) -> Vec<Vec<f32>> {
    let num_frames = mags.len();
    if num_frames == 0 {
        return vec![];
    }
    let num_bins = mags[0].len();
    let half = width / 2;
    let mut result: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
    let mut scratch = Vec::with_capacity(width);

    for (frame_idx, _) in mags.iter().enumerate().take(num_frames) {
        let start = frame_idx.saturating_sub(half);
        let end = (frame_idx + half + 1).min(num_frames);

        let mut row = Vec::with_capacity(num_bins);
        for bin in 0..num_bins {
            scratch.clear();
            for frame_mags in &mags[start..end] {
                scratch.push(frame_mags[bin]);
            }
            scratch.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            row.push(scratch[scratch.len() / 2]);
        }
        result.push(row);
    }

    result
}

/// Applies a vertical (frequency-axis) median filter to the magnitude spectrogram.
///
/// For each frame, the median is computed over a window of `width` bins centered
/// on the current bin. This enhances spectrally broad (percussive) components.
fn median_filter_vertical(mags: &[Vec<f32>], width: usize) -> Vec<Vec<f32>> {
    let num_frames = mags.len();
    if num_frames == 0 {
        return vec![];
    }
    let num_bins = mags[0].len();
    let half = width / 2;
    let mut result: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
    let mut scratch = Vec::with_capacity(width);

    for frame_mags in mags.iter().take(num_frames) {
        let mut row = Vec::with_capacity(num_bins);
        for bin in 0..num_bins {
            let start = bin.saturating_sub(half);
            let end = (bin + half + 1).min(num_bins);

            scratch.clear();
            scratch.extend_from_slice(&frame_mags[start..end]);
            scratch.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            row.push(scratch[scratch.len() / 2]);
        }
        result.push(row);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hpss_short_input() {
        let input = vec![0.5f32; 100];
        let (h, p) = hpss(&input, 4096, 1024, &HpssParams::default());
        assert_eq!(h.len(), 100);
        assert_eq!(p.len(), 100);
        // Short input: harmonic should be a copy, percussive should be zeros
        assert_eq!(h, input);
        for &v in &p {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_hpss_silence() {
        let input = vec![0.0f32; 8192];
        let (h, p) = hpss(&input, 4096, 1024, &HpssParams::default());
        assert_eq!(h.len(), input.len());
        assert_eq!(p.len(), input.len());
        for &v in &h {
            assert!(v.abs() < 1e-6);
        }
        for &v in &p {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_hpss_sum_preserves_energy() {
        // The sum of harmonic + percussive should approximately equal the input
        let sample_rate = 44100;
        let num_samples = sample_rate;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect();

        let (h, p) = hpss(&input, 4096, 1024, &HpssParams::default());

        // Check that h + p ≈ input (within tolerance due to windowing/overlap-add)
        let sum_rms: f32 = input
            .iter()
            .zip(h.iter().zip(p.iter()))
            .map(|(&inp, (&hi, &pi))| {
                let diff = inp - hi - pi;
                diff * diff
            })
            .sum::<f32>()
            / input.len() as f32;
        let sum_rms = sum_rms.sqrt();

        let input_rms: f32 = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32).sqrt();

        // Reconstruction error should be small relative to signal level
        assert!(
            sum_rms < input_rms * 0.15,
            "HPSS reconstruction error too large: {:.6} vs input RMS {:.6}",
            sum_rms,
            input_rms
        );
    }

    #[test]
    fn test_hpss_tone_is_mostly_harmonic() {
        // A pure sine tone should be mostly in the harmonic component
        let sample_rate = 44100;
        let num_samples = sample_rate * 2;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.8
            })
            .collect();

        let (h, p) = hpss(&input, 4096, 1024, &HpssParams::default());

        let h_energy: f32 = h.iter().map(|x| x * x).sum();
        let p_energy: f32 = p.iter().map(|x| x * x).sum();

        // Harmonic energy should dominate for a pure tone
        assert!(
            h_energy > p_energy,
            "Harmonic energy ({:.2}) should exceed percussive ({:.2}) for a tone",
            h_energy,
            p_energy
        );
    }

    #[test]
    fn test_median_filter_horizontal_identity() {
        // Width 1: each value is its own median
        let mags = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = median_filter_horizontal(&mags, 1);
        assert_eq!(result, mags);
    }

    #[test]
    fn test_median_filter_vertical_identity() {
        let mags = vec![vec![1.0, 2.0, 3.0]];
        let result = median_filter_vertical(&mags, 1);
        assert_eq!(result, mags);
    }
}

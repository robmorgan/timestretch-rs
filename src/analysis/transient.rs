//! Spectral-flux transient detection with adaptive thresholding.

use rustfft::{num_complex::Complex, FftPlanner};

/// Result of transient detection: sample positions of detected onsets.
#[derive(Debug, Clone)]
pub struct TransientMap {
    /// Sample positions of detected transient onsets.
    pub onsets: Vec<usize>,
    /// Spectral flux values at each analysis frame (for debugging/visualization).
    pub flux: Vec<f32>,
    /// Hop size used for analysis.
    pub hop_size: usize,
}

/// Computes the spectral flux for each frame of a mono audio signal.
///
/// Returns a vector of flux values, one per analysis frame. Flux is weighted
/// by frequency band to emphasize the 2-8 kHz transient range.
fn compute_spectral_flux(
    samples: &[f32],
    sample_rate: u32,
    fft_size: usize,
    hop_size: usize,
) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let window =
        crate::core::window::generate_window(crate::core::window::WindowType::Hann, fft_size);

    let bin_weights = compute_bin_weights(fft_size, sample_rate);
    let num_bins = fft_size / 2 + 1;
    let num_frames = (samples.len() - fft_size) / hop_size + 1;
    let mut prev_magnitude = vec![0.0f32; num_bins];
    let mut flux_values = Vec::with_capacity(num_frames);
    let mut fft_buffer = vec![Complex::new(0.0f32, 0.0f32); fft_size];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;

        for i in 0..fft_size {
            fft_buffer[i] = Complex::new(samples[start + i] * window[i], 0.0);
        }

        fft.process(&mut fft_buffer);

        let mut flux = 0.0f32;
        for bin in 0..num_bins {
            let mag = fft_buffer[bin].norm();
            let diff = mag - prev_magnitude[bin];
            if diff > 0.0 {
                flux += diff * bin_weights[bin];
            }
            prev_magnitude[bin] = mag;
        }

        flux_values.push(flux);
    }

    flux_values
}

/// Detects transients in a mono audio signal using spectral flux.
///
/// Uses high-frequency weighted spectral flux with adaptive thresholding,
/// tuned for EDM transient detection (kicks, snares, hi-hats).
pub fn detect_transients(
    samples: &[f32],
    sample_rate: u32,
    fft_size: usize,
    hop_size: usize,
    sensitivity: f32,
) -> TransientMap {
    if samples.len() < fft_size {
        return TransientMap {
            onsets: vec![],
            flux: vec![],
            hop_size,
        };
    }

    let flux_values = compute_spectral_flux(samples, sample_rate, fft_size, hop_size);
    let onsets = adaptive_threshold(&flux_values, sensitivity, hop_size);

    TransientMap {
        onsets,
        flux: flux_values,
        hop_size,
    }
}

/// Computes frequency bin weights for transient detection.
/// Emphasizes the 2-8 kHz range where hi-hats and snare attacks live.
fn compute_bin_weights(fft_size: usize, sample_rate: u32) -> Vec<f32> {
    let num_bins = fft_size / 2 + 1;
    let bin_freq = sample_rate as f32 / fft_size as f32;
    let mut weights = Vec::with_capacity(num_bins);

    for bin in 0..num_bins {
        let freq = bin as f32 * bin_freq;
        let weight = if freq < 100.0 {
            // Sub-bass: low weight for transient detection
            0.3
        } else if freq < 500.0 {
            // Bass/low-mid: moderate weight (kick body)
            0.6
        } else if freq < 2000.0 {
            // Mid: moderate weight
            0.8
        } else if freq < 8000.0 {
            // High-mid: highest weight (transient content)
            1.5
        } else {
            // Very high: moderate (noise)
            0.8
        };
        weights.push(weight);
    }

    weights
}

/// Number of frames in the local median window for adaptive thresholding.
const MEDIAN_WINDOW_FRAMES: usize = 11;
/// Minimum gap between detected onsets in frames (~50ms at typical hop sizes).
const MIN_ONSET_GAP_FRAMES: usize = 4;
/// Floor added to threshold to avoid false positives in near-silence.
const THRESHOLD_FLOOR: f32 = 0.01;

/// Adaptive thresholding for onset detection.
/// Uses a sliding median with multiplicative threshold.
fn adaptive_threshold(flux: &[f32], sensitivity: f32, hop_size: usize) -> Vec<usize> {
    if flux.is_empty() {
        return vec![];
    }

    let half_window = MEDIAN_WINDOW_FRAMES / 2;
    // Higher sensitivity = lower threshold = more detections
    let threshold_multiplier = 1.0 + (1.0 - sensitivity) * 4.0;

    let mut onsets = Vec::new();
    let mut last_onset: Option<usize> = None;
    // Reusable sort buffer to avoid per-frame allocation
    let mut local = Vec::with_capacity(MEDIAN_WINDOW_FRAMES);

    for i in 0..flux.len() {
        // Compute local median
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(flux.len());
        local.clear();
        local.extend_from_slice(&flux[start..end]);
        local.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = local[local.len() / 2];

        let threshold = median * threshold_multiplier + THRESHOLD_FLOOR;

        if flux[i] > threshold {
            // Check minimum gap
            if let Some(last) = last_onset {
                if i - last < MIN_ONSET_GAP_FRAMES {
                    continue;
                }
            }
            onsets.push(i * hop_size);
            last_onset = Some(i);
        }
    }

    onsets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_transients_click_train() {
        // Generate a click train: short impulses every 0.5 seconds
        let sample_rate = 44100u32;
        let duration_secs = 2.0;
        let num_samples = (sample_rate as f64 * duration_secs) as usize;
        let mut samples = vec![0.0f32; num_samples];

        let click_interval = sample_rate as usize / 2; // 0.5 sec
        for i in (0..num_samples).step_by(click_interval) {
            // Short click: 10 samples of impulse
            for j in 0..10.min(num_samples - i) {
                samples[i + j] = if j < 5 { 1.0 } else { -0.5 };
            }
        }

        let result = detect_transients(&samples, sample_rate, 2048, 512, 0.5);
        // Should detect approximately 4 clicks (at 0, 0.5, 1.0, 1.5)
        assert!(
            result.onsets.len() >= 2,
            "Expected at least 2 onsets, got {}",
            result.onsets.len()
        );
    }

    #[test]
    fn test_detect_transients_silence() {
        let samples = vec![0.0f32; 44100];
        let result = detect_transients(&samples, 44100, 2048, 512, 0.5);
        assert!(
            result.onsets.is_empty(),
            "Expected no onsets in silence, got {}",
            result.onsets.len()
        );
    }

    #[test]
    fn test_detect_transients_too_short() {
        let samples = vec![0.0f32; 100];
        let result = detect_transients(&samples, 44100, 2048, 512, 0.5);
        assert!(result.onsets.is_empty());
        assert!(result.flux.is_empty());
    }

    #[test]
    fn test_bin_weights() {
        let weights = compute_bin_weights(4096, 44100);
        assert_eq!(weights.len(), 2049);
        // Sub-bass should have low weight
        assert!(weights[0] < 0.5);
        // 4kHz bin should have high weight
        let bin_4k = (4000.0 / (44100.0 / 4096.0)) as usize;
        assert!(weights[bin_4k] > 1.0);
    }
}

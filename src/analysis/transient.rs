//! Spectral-flux transient detection with adaptive thresholding.

use rustfft::{num_complex::Complex, FftPlanner};

/// Zero-valued complex number, used for FFT buffer initialization.
const COMPLEX_ZERO: Complex<f32> = Complex::new(0.0, 0.0);

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
    let mut fft_buffer = vec![COMPLEX_ZERO; fft_size];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;

        for (buf, (&s, &w)) in fft_buffer
            .iter_mut()
            .zip(samples[start..].iter().zip(window.iter()))
        {
            *buf = Complex::new(s * w, 0.0);
        }

        fft.process(&mut fft_buffer);

        let mut flux = 0.0f32;
        for ((&c, prev), &weight) in fft_buffer[..num_bins]
            .iter()
            .zip(prev_magnitude.iter_mut())
            .zip(bin_weights[..num_bins].iter())
        {
            let mag = c.norm();
            let diff = mag - *prev;
            if diff > 0.0 {
                flux += diff * weight;
            }
            *prev = mag;
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

// Frequency band boundaries for transient weighting (Hz).
const BAND_SUB_BASS_LIMIT: f32 = 100.0;
const BAND_BASS_MID_LIMIT: f32 = 500.0;
const BAND_MID_LIMIT: f32 = 2000.0;
const BAND_HIGH_MID_LIMIT: f32 = 8000.0;

// Spectral flux weights per frequency band.
/// Sub-bass (<100 Hz): low weight — little transient content.
const WEIGHT_SUB_BASS: f32 = 0.3;
/// Bass/low-mid (100–500 Hz): moderate weight — kick body.
const WEIGHT_BASS_MID: f32 = 0.6;
/// Mid (500–2000 Hz): moderate weight.
const WEIGHT_MID: f32 = 0.8;
/// High-mid (2–8 kHz): highest weight — hi-hats, snare attacks.
const WEIGHT_HIGH_MID: f32 = 1.5;
/// Very high (>8 kHz): moderate weight — noise content.
const WEIGHT_VERY_HIGH: f32 = 0.8;

/// Computes frequency bin weights for transient detection.
/// Emphasizes the 2-8 kHz range where hi-hats and snare attacks live.
fn compute_bin_weights(fft_size: usize, sample_rate: u32) -> Vec<f32> {
    let num_bins = fft_size / 2 + 1;
    let bin_freq = sample_rate as f32 / fft_size as f32;

    (0..num_bins)
        .map(|bin| {
            let freq = bin as f32 * bin_freq;
            if freq < BAND_SUB_BASS_LIMIT {
                WEIGHT_SUB_BASS
            } else if freq < BAND_BASS_MID_LIMIT {
                WEIGHT_BASS_MID
            } else if freq < BAND_MID_LIMIT {
                WEIGHT_MID
            } else if freq < BAND_HIGH_MID_LIMIT {
                WEIGHT_HIGH_MID
            } else {
                WEIGHT_VERY_HIGH
            }
        })
        .collect()
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

    for (i, &flux_val) in flux.iter().enumerate() {
        // Compute local median
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(flux.len());
        local.clear();
        local.extend_from_slice(&flux[start..end]);
        local.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = local[local.len() / 2];

        let threshold = median * threshold_multiplier + THRESHOLD_FLOOR;

        if flux_val > threshold {
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

    // --- compute_bin_weights band boundary tests ---

    #[test]
    fn test_bin_weights_all_bands_covered() {
        let fft_size = 4096;
        let sr = 44100u32;
        let weights = compute_bin_weights(fft_size, sr);
        let bin_freq = sr as f32 / fft_size as f32;

        // DC bin: should be sub-bass weight
        assert!(
            (weights[0] - WEIGHT_SUB_BASS).abs() < 1e-6,
            "DC bin weight should be WEIGHT_SUB_BASS"
        );

        // Bin just below 100 Hz boundary
        let bin_99 = (99.0 / bin_freq) as usize;
        assert!(
            (weights[bin_99] - WEIGHT_SUB_BASS).abs() < 1e-6,
            "Bin at ~99Hz should be WEIGHT_SUB_BASS"
        );

        // Bin just above 100 Hz → bass/mid
        let bin_110 = (110.0 / bin_freq) as usize;
        assert!(
            (weights[bin_110] - WEIGHT_BASS_MID).abs() < 1e-6,
            "Bin at ~110Hz should be WEIGHT_BASS_MID"
        );

        // Bin in 500-2000 Hz → mid
        let bin_1000 = (1000.0 / bin_freq) as usize;
        assert!(
            (weights[bin_1000] - WEIGHT_MID).abs() < 1e-6,
            "Bin at ~1000Hz should be WEIGHT_MID"
        );

        // Bin in 2-8 kHz → high-mid (highest weight)
        let bin_4000 = (4000.0 / bin_freq) as usize;
        assert!(
            (weights[bin_4000] - WEIGHT_HIGH_MID).abs() < 1e-6,
            "Bin at ~4000Hz should be WEIGHT_HIGH_MID"
        );

        // Bin above 8 kHz → very high
        let bin_10000 = (10000.0 / bin_freq) as usize;
        assert!(
            (weights[bin_10000] - WEIGHT_VERY_HIGH).abs() < 1e-6,
            "Bin at ~10kHz should be WEIGHT_VERY_HIGH"
        );

        // Nyquist bin
        let nyquist_bin = fft_size / 2;
        assert!(
            (weights[nyquist_bin] - WEIGHT_VERY_HIGH).abs() < 1e-6,
            "Nyquist bin should be WEIGHT_VERY_HIGH"
        );
    }

    #[test]
    fn test_bin_weights_48khz() {
        // Different sample rate should shift band boundaries
        let weights = compute_bin_weights(4096, 48000);
        let bin_freq = 48000.0f32 / 4096.0;

        // 4kHz bin at 48kHz
        let bin_4k = (4000.0 / bin_freq) as usize;
        assert!(
            (weights[bin_4k] - WEIGHT_HIGH_MID).abs() < 1e-6,
            "4kHz at 48kHz should be WEIGHT_HIGH_MID"
        );
    }

    // --- adaptive_threshold internals ---

    #[test]
    fn test_adaptive_threshold_empty_flux() {
        let result = adaptive_threshold(&[], 0.5, 512);
        assert!(result.is_empty());
    }

    #[test]
    fn test_adaptive_threshold_all_below_threshold() {
        // Uniform low flux: all values equal → median = value → threshold = value * multiplier + floor
        // With sensitivity 0.5: multiplier = 1 + (1-0.5)*4 = 3.0
        // threshold = 0.001 * 3.0 + 0.01 = 0.013 > 0.001 → no detections
        let flux = vec![0.001f32; 50];
        let result = adaptive_threshold(&flux, 0.5, 512);
        assert!(
            result.is_empty(),
            "Uniform low flux should produce no onsets"
        );
    }

    #[test]
    fn test_adaptive_threshold_single_spike() {
        // Single large spike in otherwise silent flux
        let mut flux = vec![0.0f32; 50];
        flux[25] = 1.0; // big spike
        let result = adaptive_threshold(&flux, 0.5, 512);
        assert!(
            !result.is_empty(),
            "Large spike should be detected as onset"
        );
        // Onset position should be flux_index * hop_size
        assert_eq!(result[0], 25 * 512);
    }

    #[test]
    fn test_adaptive_threshold_sensitivity_high() {
        // High sensitivity (0.9) → lower threshold → more detections
        let mut flux = vec![0.01f32; 100];
        // Modest spikes
        flux[20] = 0.1;
        flux[50] = 0.1;
        flux[80] = 0.1;

        let high_sens = adaptive_threshold(&flux, 0.9, 512);
        let low_sens = adaptive_threshold(&flux, 0.1, 512);

        assert!(
            high_sens.len() >= low_sens.len(),
            "High sensitivity ({}) should detect >= low sensitivity ({})",
            high_sens.len(),
            low_sens.len()
        );
    }

    #[test]
    fn test_adaptive_threshold_min_onset_gap() {
        // Two spikes within MIN_ONSET_GAP_FRAMES → only first detected
        let mut flux = vec![0.001f32; 50];
        flux[10] = 2.0;
        flux[12] = 2.0; // Only 2 frames apart (< MIN_ONSET_GAP_FRAMES=4)

        let result = adaptive_threshold(&flux, 0.5, 512);

        // Count onsets near frames 10-12
        let onsets_near: Vec<_> = result
            .iter()
            .filter(|&&pos| (10 * 512..=12 * 512).contains(&pos))
            .collect();
        assert!(
            onsets_near.len() <= 1,
            "Close spikes should be deduplicated, got {} onsets",
            onsets_near.len()
        );
    }

    #[test]
    fn test_adaptive_threshold_spikes_beyond_gap() {
        // Two spikes separated by more than MIN_ONSET_GAP_FRAMES → both detected
        let mut flux = vec![0.001f32; 50];
        flux[10] = 2.0;
        flux[20] = 2.0; // 10 frames apart (> MIN_ONSET_GAP_FRAMES=4)

        let result = adaptive_threshold(&flux, 0.5, 512);
        assert!(
            result.len() >= 2,
            "Well-separated spikes should both be detected, got {}",
            result.len()
        );
    }

    // --- compute_spectral_flux ---

    #[test]
    fn test_spectral_flux_silence_is_zero() {
        // Silence should produce zero (or near-zero) flux for all frames
        let samples = vec![0.0f32; 44100];
        let flux = compute_spectral_flux(&samples, 44100, 2048, 512);
        assert!(!flux.is_empty());
        for &f in &flux {
            assert!(f.abs() < 1e-6, "Flux for silence should be ~0, got {}", f);
        }
    }

    #[test]
    fn test_spectral_flux_constant_tone_after_onset() {
        // A constant tone: flux should be high at onset (first frame), then low
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let flux = compute_spectral_flux(&input, sample_rate, 2048, 512);
        assert!(flux.len() > 2);

        // First frame has high flux (transition from zeros in prev_magnitude)
        assert!(
            flux[0] > 0.0,
            "First frame flux should be > 0 (onset from silence)"
        );

        // Later frames should have much lower flux (steady state)
        let late_flux_avg: f32 =
            flux[flux.len() / 2..].iter().sum::<f32>() / (flux.len() / 2) as f32;
        assert!(
            late_flux_avg < flux[0] * 0.5,
            "Late flux avg {} should be much lower than onset flux {}",
            late_flux_avg,
            flux[0]
        );
    }

    #[test]
    fn test_spectral_flux_impulse_detection() {
        // An impulse at a known position should produce a flux spike at that frame
        let sample_rate = 44100u32;
        let fft_size = 2048;
        let hop_size = 512;
        let num_samples = sample_rate as usize;
        let mut input = vec![0.0f32; num_samples];

        // Place impulse at ~0.5 seconds
        let impulse_pos = sample_rate as usize / 2;
        for j in 0..10 {
            if impulse_pos + j < num_samples {
                input[impulse_pos + j] = if j < 3 { 1.0 } else { -0.5 };
            }
        }

        let flux = compute_spectral_flux(&input, sample_rate, fft_size, hop_size);

        // Find frame containing the impulse
        let impulse_frame = impulse_pos / hop_size;
        if impulse_frame < flux.len() {
            // Flux at impulse frame should be above average
            let avg_flux = flux.iter().sum::<f32>() / flux.len() as f32;
            let max_flux_near_impulse = flux
                [impulse_frame.saturating_sub(2)..(impulse_frame + 3).min(flux.len())]
                .iter()
                .copied()
                .fold(0.0f32, f32::max);
            assert!(
                max_flux_near_impulse > avg_flux,
                "Flux near impulse {} should be above average {}",
                max_flux_near_impulse,
                avg_flux
            );
        }
    }
}

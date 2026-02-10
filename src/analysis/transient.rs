use rustfft::{num_complex::Complex, FftPlanner};

/// Result of transient detection: a list of sample positions where transients occur.
#[derive(Debug, Clone)]
pub struct TransientMarkers {
    /// Sample positions of detected transients.
    pub positions: Vec<usize>,
}

/// Detect transients in an audio signal using spectral flux with adaptive threshold.
///
/// # Arguments
/// * `samples` - Mono audio samples
/// * `sample_rate` - Sample rate in Hz
/// * `sensitivity` - Detection sensitivity (0.0 to 1.0, higher = more sensitive)
/// * `fft_size` - FFT size for spectral analysis
///
/// # Returns
/// List of sample positions where transients were detected.
pub fn detect_transients(
    samples: &[f32],
    sample_rate: u32,
    sensitivity: f32,
    fft_size: usize,
) -> TransientMarkers {
    if samples.len() < fft_size {
        return TransientMarkers {
            positions: Vec::new(),
        };
    }

    let hop_size = fft_size / 4;
    let num_frames = (samples.len() - fft_size) / hop_size + 1;

    if num_frames < 2 {
        return TransientMarkers {
            positions: Vec::new(),
        };
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Compute magnitudes for each frame
    let mut magnitudes: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
    let window = crate::core::window::hann_window(fft_size);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let mut buffer: Vec<Complex<f32>> = samples[start..start + fft_size]
            .iter()
            .enumerate()
            .map(|(i, &s)| Complex::new(s * window[i], 0.0))
            .collect();

        fft.process(&mut buffer);

        // Compute magnitude spectrum (only positive frequencies)
        let half = fft_size / 2 + 1;
        let mags: Vec<f32> = buffer[..half].iter().map(|c| c.norm()).collect();
        magnitudes.push(mags);
    }

    // Compute spectral flux with high-frequency emphasis
    let half = fft_size / 2 + 1;
    let freq_resolution = sample_rate as f32 / fft_size as f32;

    // Weight bands: emphasize 2-8 kHz for hi-hats and snares
    let weights: Vec<f32> = (0..half)
        .map(|bin| {
            let freq = bin as f32 * freq_resolution;
            if freq >= 2000.0 && freq <= 8000.0 {
                2.0 // Emphasize this band
            } else if freq < 120.0 {
                0.3 // De-emphasize sub-bass
            } else {
                1.0
            }
        })
        .collect();

    let mut flux: Vec<f32> = Vec::with_capacity(num_frames - 1);
    for i in 1..num_frames {
        let mut sum = 0.0f32;
        for bin in 0..half {
            let diff = magnitudes[i][bin] - magnitudes[i - 1][bin];
            if diff > 0.0 {
                sum += diff * weights[bin];
            }
        }
        flux.push(sum);
    }

    if flux.is_empty() {
        return TransientMarkers {
            positions: Vec::new(),
        };
    }

    // Adaptive threshold: median + sensitivity-scaled deviation
    let mut sorted_flux = flux.clone();
    sorted_flux.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted_flux[sorted_flux.len() / 2];
    let mean_abs_dev: f32 =
        flux.iter().map(|&x| (x - median).abs()).sum::<f32>() / flux.len() as f32;

    // Lower sensitivity = higher threshold = fewer detections
    let threshold_scale = 2.0 - sensitivity * 1.5;
    let threshold = median + mean_abs_dev * threshold_scale;

    // Peak-pick: find local maxima above threshold
    let min_onset_gap = (sample_rate as f32 * 0.03) as usize; // 30ms minimum gap
    let min_hop_gap = min_onset_gap / hop_size;

    let mut positions = Vec::new();
    let mut last_onset: Option<usize> = None;

    for i in 0..flux.len() {
        if flux[i] <= threshold {
            continue;
        }
        let is_peak = (i == 0 || flux[i] >= flux[i - 1])
            && (i == flux.len() - 1 || flux[i] >= flux[i + 1]);
        if !is_peak {
            continue;
        }
        if let Some(last) = last_onset {
            if i - last < min_hop_gap {
                continue;
            }
        }
        let sample_pos = (i + 1) * hop_size; // +1 because flux is computed from frame pairs
        positions.push(sample_pos);
        last_onset = Some(i);
    }

    TransientMarkers { positions }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_transients_click_train() {
        let sample_rate = 44100u32;
        let duration_secs = 1.0;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let mut samples = vec![0.0f32; num_samples];

        // Insert clicks at regular intervals (every 0.25 seconds)
        let click_interval = sample_rate as usize / 4;
        for i in (0..num_samples).step_by(click_interval) {
            let end = (i + 50).min(num_samples);
            for j in i..end {
                samples[j] = 0.9;
            }
        }

        let markers = detect_transients(&samples, sample_rate, 0.5, 2048);
        // Should detect at least some transients
        assert!(
            !markers.positions.is_empty(),
            "Should detect transients in click train"
        );
    }

    #[test]
    fn test_detect_transients_silence() {
        let samples = vec![0.0f32; 44100];
        let markers = detect_transients(&samples, 44100, 0.5, 2048);
        assert!(
            markers.positions.is_empty(),
            "Should not detect transients in silence"
        );
    }

    #[test]
    fn test_detect_transients_short_input() {
        let samples = vec![0.5f32; 100]; // Shorter than FFT size
        let markers = detect_transients(&samples, 44100, 0.5, 2048);
        assert!(markers.positions.is_empty());
    }

    #[test]
    fn test_detect_transients_sensitivity() {
        let sample_rate = 44100u32;
        let mut samples = vec![0.0f32; 44100];
        // One click
        for i in 10000..10050 {
            samples[i] = 0.8;
        }

        let low_sens = detect_transients(&samples, sample_rate, 0.1, 2048);
        let high_sens = detect_transients(&samples, sample_rate, 0.9, 2048);
        // Higher sensitivity should detect at least as many onsets
        assert!(high_sens.positions.len() >= low_sens.positions.len());
    }
}

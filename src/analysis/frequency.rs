use rustfft::{num_complex::Complex, FftPlanner};

/// Frequency band boundaries.
#[derive(Debug, Clone, Copy)]
pub struct FrequencyBands {
    /// Sub-bass boundary (0 to this, default 120 Hz).
    pub sub_bass_cutoff: f32,
    /// Low-mid boundary (sub_bass_cutoff to this, default 500 Hz).
    pub low_cutoff: f32,
    /// Mid-high boundary (low_cutoff to this, default 4000 Hz).
    pub mid_cutoff: f32,
    // Above mid_cutoff is high band
}

impl Default for FrequencyBands {
    fn default() -> Self {
        Self {
            sub_bass_cutoff: 120.0,
            low_cutoff: 500.0,
            mid_cutoff: 4000.0,
        }
    }
}

/// Split a spectrum into frequency bands.
///
/// Returns (sub_bass, low, mid, high) bin index ranges for a given FFT size and sample rate.
pub fn band_bin_ranges(
    fft_size: usize,
    sample_rate: u32,
    bands: &FrequencyBands,
) -> (std::ops::Range<usize>, std::ops::Range<usize>, std::ops::Range<usize>, std::ops::Range<usize>) {
    let freq_resolution = sample_rate as f32 / fft_size as f32;
    let half = fft_size / 2 + 1;

    let sub_bass_end = ((bands.sub_bass_cutoff / freq_resolution).ceil() as usize).min(half);
    let low_end = ((bands.low_cutoff / freq_resolution).ceil() as usize).min(half);
    let mid_end = ((bands.mid_cutoff / freq_resolution).ceil() as usize).min(half);

    (
        0..sub_bass_end,
        sub_bass_end..low_end,
        low_end..mid_end,
        mid_end..half,
    )
}

/// Compute the spectral centroid of a signal.
///
/// The spectral centroid is the "center of mass" of the spectrum,
/// indicating the average frequency weighted by magnitude.
pub fn spectral_centroid(samples: &[f32], sample_rate: u32, fft_size: usize) -> f32 {
    if samples.len() < fft_size {
        return 0.0;
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let window = crate::core::window::hann_window(fft_size);
    let mut buffer: Vec<Complex<f32>> = samples[..fft_size]
        .iter()
        .enumerate()
        .map(|(i, &s)| Complex::new(s * window[i], 0.0))
        .collect();

    fft.process(&mut buffer);

    let freq_resolution = sample_rate as f32 / fft_size as f32;
    let half = fft_size / 2 + 1;

    let mut weighted_sum = 0.0f64;
    let mut magnitude_sum = 0.0f64;

    for bin in 0..half {
        let mag = buffer[bin].norm() as f64;
        let freq = (bin as f32 * freq_resolution) as f64;
        weighted_sum += freq * mag;
        magnitude_sum += mag;
    }

    if magnitude_sum > 0.0 {
        (weighted_sum / magnitude_sum) as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_band_ranges_basic() {
        let (sub, low, mid, high) = band_bin_ranges(4096, 44100, &FrequencyBands::default());
        // Sub-bass: 0-120 Hz
        assert!(sub.start == 0);
        // Bands should be contiguous
        assert_eq!(sub.end, low.start);
        assert_eq!(low.end, mid.start);
        assert_eq!(mid.end, high.start);
        assert_eq!(high.end, 4096 / 2 + 1);
    }

    #[test]
    fn test_spectral_centroid_low_freq() {
        // Generate a 100 Hz sine wave
        let sample_rate = 44100;
        let freq = 100.0;
        let samples: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let centroid = spectral_centroid(&samples, sample_rate, 4096);
        // Centroid should be near 100 Hz (with some spectral leakage)
        assert!(
            centroid > 50.0 && centroid < 200.0,
            "Centroid for 100 Hz sine should be near 100 Hz, got {centroid}"
        );
    }

    #[test]
    fn test_spectral_centroid_high_freq() {
        let sample_rate = 44100;
        let freq = 5000.0;
        let samples: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let centroid = spectral_centroid(&samples, sample_rate, 4096);
        assert!(
            centroid > 4000.0 && centroid < 6000.0,
            "Centroid for 5 kHz sine should be near 5000 Hz, got {centroid}"
        );
    }

    #[test]
    fn test_spectral_centroid_silence() {
        let samples = vec![0.0f32; 4096];
        let centroid = spectral_centroid(&samples, 44100, 4096);
        assert_eq!(centroid, 0.0);
    }

    #[test]
    fn test_spectral_centroid_short_input() {
        let samples = vec![0.5f32; 100];
        let centroid = spectral_centroid(&samples, 44100, 4096);
        assert_eq!(centroid, 0.0);
    }
}

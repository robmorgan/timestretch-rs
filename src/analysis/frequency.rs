//! Frequency-domain analysis: spectral centroid, band splitting, and energy.

use rustfft::{num_complex::Complex, FftPlanner};

/// Result of splitting a spectrum into frequency bands: (sub_bass, low, mid, high).
pub type BandSpectra = (Vec<Complex<f32>>, Vec<Complex<f32>>, Vec<Complex<f32>>, Vec<Complex<f32>>);

/// Frequency band boundaries for EDM processing.
#[derive(Debug, Clone, Copy)]
pub struct FrequencyBands {
    /// Sub-bass upper limit (Hz). Default: 120 Hz.
    pub sub_bass: f32,
    /// Low frequency upper limit (Hz). Default: 500 Hz.
    pub low: f32,
    /// Mid frequency upper limit (Hz). Default: 4000 Hz.
    pub mid: f32,
    // Everything above mid is "high"
}

impl Default for FrequencyBands {
    fn default() -> Self {
        Self {
            sub_bass: 120.0,
            low: 500.0,
            mid: 4000.0,
        }
    }
}

/// Returns the FFT bin index for a given frequency.
#[inline]
pub fn freq_to_bin(freq: f32, fft_size: usize, sample_rate: u32) -> usize {
    let bin = (freq * fft_size as f32 / sample_rate as f32).round() as usize;
    bin.min(fft_size / 2)
}

/// Returns the frequency for a given FFT bin index.
#[inline]
pub fn bin_to_freq(bin: usize, fft_size: usize, sample_rate: u32) -> f32 {
    bin as f32 * sample_rate as f32 / fft_size as f32
}

/// Splits a frequency-domain signal into bands by zeroing out-of-band bins.
///
/// Returns (sub_bass, low, mid, high) spectra, each the same length as input.
pub fn split_spectrum_into_bands(
    spectrum: &[Complex<f32>],
    fft_size: usize,
    sample_rate: u32,
    bands: &FrequencyBands,
) -> BandSpectra {
    let sub_bass_bin = freq_to_bin(bands.sub_bass, fft_size, sample_rate);
    let low_bin = freq_to_bin(bands.low, fft_size, sample_rate);
    let mid_bin = freq_to_bin(bands.mid, fft_size, sample_rate);
    let num_bins = fft_size / 2 + 1;

    let mut sub_bass = vec![Complex::new(0.0f32, 0.0); spectrum.len()];
    let mut low = vec![Complex::new(0.0f32, 0.0); spectrum.len()];
    let mut mid = vec![Complex::new(0.0f32, 0.0); spectrum.len()];
    let mut high = vec![Complex::new(0.0f32, 0.0); spectrum.len()];

    for bin in 0..num_bins.min(spectrum.len()) {
        if bin < sub_bass_bin {
            sub_bass[bin] = spectrum[bin];
        } else if bin < low_bin {
            low[bin] = spectrum[bin];
        } else if bin < mid_bin {
            mid[bin] = spectrum[bin];
        } else {
            high[bin] = spectrum[bin];
        }
    }

    // Mirror for negative frequencies (if full FFT size)
    if spectrum.len() == fft_size {
        for bin in 1..num_bins - 1 {
            let mirror = fft_size - bin;
            if mirror < spectrum.len() {
                if bin < sub_bass_bin {
                    sub_bass[mirror] = spectrum[mirror];
                } else if bin < low_bin {
                    low[mirror] = spectrum[mirror];
                } else if bin < mid_bin {
                    mid[mirror] = spectrum[mirror];
                } else {
                    high[mirror] = spectrum[mirror];
                }
            }
        }
    }

    (sub_bass, low, mid, high)
}

/// Computes the spectral energy in each frequency band for a given frame.
pub fn compute_band_energy(
    samples: &[f32],
    fft_size: usize,
    sample_rate: u32,
    bands: &FrequencyBands,
) -> (f32, f32, f32, f32) {
    if samples.len() < fft_size {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let window = crate::core::window::generate_window(
        crate::core::window::WindowType::Hann,
        fft_size,
    );

    let mut buffer: Vec<Complex<f32>> = samples[..fft_size]
        .iter()
        .enumerate()
        .map(|(i, &s)| Complex::new(s * window[i], 0.0))
        .collect();

    fft.process(&mut buffer);

    let sub_bass_bin = freq_to_bin(bands.sub_bass, fft_size, sample_rate);
    let low_bin = freq_to_bin(bands.low, fft_size, sample_rate);
    let mid_bin = freq_to_bin(bands.mid, fft_size, sample_rate);
    let num_bins = fft_size / 2 + 1;

    let mut sub_e = 0.0f32;
    let mut low_e = 0.0f32;
    let mut mid_e = 0.0f32;
    let mut high_e = 0.0f32;

    for (bin, val) in buffer.iter().enumerate().take(num_bins) {
        let energy = val.norm_sqr();
        if bin < sub_bass_bin {
            sub_e += energy;
        } else if bin < low_bin {
            low_e += energy;
        } else if bin < mid_bin {
            mid_e += energy;
        } else {
            high_e += energy;
        }
    }

    (sub_e, low_e, mid_e, high_e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freq_to_bin() {
        // At 44100 Hz with FFT size 4096:
        // bin resolution = 44100/4096 ≈ 10.77 Hz
        assert_eq!(freq_to_bin(0.0, 4096, 44100), 0);
        let bin_1000 = freq_to_bin(1000.0, 4096, 44100);
        // 1000 / 10.77 ≈ 93
        assert!((92..=94).contains(&bin_1000));
    }

    #[test]
    fn test_bin_to_freq() {
        let freq = bin_to_freq(100, 4096, 44100);
        assert!((freq - 1076.66).abs() < 1.0);
    }

    #[test]
    fn test_band_energy_sine() {
        // Generate a 100 Hz sine wave (sub-bass)
        let sample_rate = 44100u32;
        let fft_size = 4096;
        let samples: Vec<f32> = (0..fft_size)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 100.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let bands = FrequencyBands::default();
        let (sub_e, low_e, mid_e, high_e) = compute_band_energy(&samples, fft_size, sample_rate, &bands);

        // Most energy should be in sub-bass
        assert!(
            sub_e > low_e + mid_e + high_e,
            "Sub-bass energy {} should dominate (low={}, mid={}, high={})",
            sub_e, low_e, mid_e, high_e
        );
    }

    #[test]
    fn test_band_energy_high_freq() {
        // Generate a 5000 Hz sine wave (high)
        let sample_rate = 44100u32;
        let fft_size = 4096;
        let samples: Vec<f32> = (0..fft_size)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 5000.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let bands = FrequencyBands::default();
        let (sub_e, low_e, mid_e, high_e) = compute_band_energy(&samples, fft_size, sample_rate, &bands);

        // Most energy should be in high band (5000 Hz > 4000 Hz)
        assert!(
            high_e > sub_e + low_e + mid_e,
            "High energy {} should dominate (sub={}, low={}, mid={})",
            high_e, sub_e, low_e, mid_e
        );
    }
}

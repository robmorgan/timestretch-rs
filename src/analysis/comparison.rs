//! Audio comparison metrics for benchmarking time-stretch quality.
//!
//! Provides spectral similarity, band-level spectral similarity,
//! cross-correlation, and transient match scoring for comparing
//! library output against professional reference audio.

use rustfft::{num_complex::Complex, FftPlanner};

use crate::analysis::frequency::{freq_to_bin, FrequencyBands};
use crate::analysis::transient::detect_transients;
use crate::core::fft::COMPLEX_ZERO;
use crate::core::window::{generate_window, WindowType};

/// Per-band spectral similarity scores.
#[derive(Debug, Clone)]
pub struct BandSimilarity {
    /// Overall similarity across all bands (0.0-1.0).
    pub overall: f64,
    /// Sub-bass band similarity.
    pub sub_bass: f64,
    /// Low-frequency band similarity.
    pub low: f64,
    /// Mid-frequency band similarity.
    pub mid: f64,
    /// High-frequency band similarity.
    pub high: f64,
}

/// Result of normalized cross-correlation.
#[derive(Debug, Clone)]
pub struct CrossCorrelationResult {
    /// Peak normalized correlation value (0.0-1.0).
    pub peak_value: f64,
    /// Sample offset of peak (positive means `b` is delayed relative to `a`).
    pub peak_offset: isize,
}

/// Result of transient onset matching.
#[derive(Debug, Clone)]
pub struct TransientMatchResult {
    /// Fraction of reference onsets matched (0.0-1.0).
    pub match_rate: f64,
    /// Number of matched onsets.
    pub matched: usize,
    /// Total onsets in reference signal.
    pub total_reference: usize,
    /// Total onsets in test signal.
    pub total_test: usize,
}

/// Computes STFT magnitude cosine similarity averaged across frames.
///
/// Returns a value in 0.0-1.0 where 1.0 means identical magnitude spectra.
/// The two signals are analyzed frame-by-frame and the cosine similarity of
/// each frame's magnitude spectrum is averaged.
pub fn spectral_similarity(a: &[f32], b: &[f32], fft_size: usize, hop_size: usize) -> f64 {
    let window = generate_window(WindowType::Hann, fft_size);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let num_bins = fft_size / 2 + 1;

    let min_len = a.len().min(b.len());
    if min_len < fft_size {
        return 0.0;
    }

    let num_frames = (min_len - fft_size) / hop_size + 1;
    if num_frames == 0 {
        return 0.0;
    }

    let mut buf_a = vec![COMPLEX_ZERO; fft_size];
    let mut buf_b = vec![COMPLEX_ZERO; fft_size];
    let mut similarity_sum = 0.0f64;

    for frame in 0..num_frames {
        let start = frame * hop_size;

        // Fill and window buffers
        for i in 0..fft_size {
            let w = window[i];
            buf_a[i] = Complex::new(a[start + i] * w, 0.0);
            buf_b[i] = Complex::new(b[start + i] * w, 0.0);
        }

        fft.process(&mut buf_a);
        fft.process(&mut buf_b);

        // Cosine similarity of magnitude spectra
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;

        for i in 0..num_bins {
            let ma = buf_a[i].norm() as f64;
            let mb = buf_b[i].norm() as f64;
            dot += ma * mb;
            norm_a += ma * ma;
            norm_b += mb * mb;
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom > 1e-12 {
            similarity_sum += dot / denom;
        }
    }

    similarity_sum / num_frames as f64
}

/// Computes cosine similarity of averaged magnitude spectra.
///
/// Unlike [`spectral_similarity`] which compares frame-by-frame, this computes
/// the mean magnitude spectrum over all frames for each signal, then compares
/// those averages. This is timing-invariant — it measures whether the two
/// signals have the same overall frequency balance regardless of when events
/// occur within the segment.
pub fn mean_spectral_similarity(a: &[f32], b: &[f32], fft_size: usize, hop_size: usize) -> f64 {
    let window = generate_window(WindowType::Hann, fft_size);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let num_bins = fft_size / 2 + 1;

    let len_a = a.len();
    let len_b = b.len();
    if len_a < fft_size || len_b < fft_size {
        return 0.0;
    }

    let frames_a = (len_a - fft_size) / hop_size + 1;
    let frames_b = (len_b - fft_size) / hop_size + 1;
    if frames_a == 0 || frames_b == 0 {
        return 0.0;
    }

    let mut buf = vec![COMPLEX_ZERO; fft_size];

    // Accumulate mean magnitude spectrum for signal A.
    let mut mean_a = vec![0.0f64; num_bins];
    for frame in 0..frames_a {
        let start = frame * hop_size;
        for i in 0..fft_size {
            buf[i] = Complex::new(a[start + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        for i in 0..num_bins {
            mean_a[i] += buf[i].norm() as f64;
        }
    }
    for v in &mut mean_a {
        *v /= frames_a as f64;
    }

    // Accumulate mean magnitude spectrum for signal B.
    let mut mean_b = vec![0.0f64; num_bins];
    for frame in 0..frames_b {
        let start = frame * hop_size;
        for i in 0..fft_size {
            buf[i] = Complex::new(b[start + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        for i in 0..num_bins {
            mean_b[i] += buf[i].norm() as f64;
        }
    }
    for v in &mut mean_b {
        *v /= frames_b as f64;
    }

    // Cosine similarity of the two mean spectra.
    let mut dot = 0.0f64;
    let mut norm_a_sq = 0.0f64;
    let mut norm_b_sq = 0.0f64;
    for i in 0..num_bins {
        dot += mean_a[i] * mean_b[i];
        norm_a_sq += mean_a[i] * mean_a[i];
        norm_b_sq += mean_b[i] * mean_b[i];
    }

    let denom = (norm_a_sq * norm_b_sq).sqrt();
    if denom > 1e-12 {
        dot / denom
    } else {
        0.0
    }
}

/// Computes per-band STFT magnitude cosine similarity.
///
/// Same as [`spectral_similarity`] but split into sub-bass, low, mid, and high
/// bands using the provided sample rate and default [`FrequencyBands`] boundaries.
pub fn band_spectral_similarity(
    a: &[f32],
    b: &[f32],
    fft_size: usize,
    hop_size: usize,
    sample_rate: u32,
) -> BandSimilarity {
    let bands = FrequencyBands::default();
    let window = generate_window(WindowType::Hann, fft_size);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let num_bins = fft_size / 2 + 1;

    let sub_bass_bin = freq_to_bin(bands.sub_bass, fft_size, sample_rate);
    let low_bin = freq_to_bin(bands.low, fft_size, sample_rate);
    let mid_bin = freq_to_bin(bands.mid, fft_size, sample_rate);

    let min_len = a.len().min(b.len());
    if min_len < fft_size {
        return BandSimilarity {
            overall: 0.0,
            sub_bass: 0.0,
            low: 0.0,
            mid: 0.0,
            high: 0.0,
        };
    }

    let num_frames = (min_len - fft_size) / hop_size + 1;
    if num_frames == 0 {
        return BandSimilarity {
            overall: 0.0,
            sub_bass: 0.0,
            low: 0.0,
            mid: 0.0,
            high: 0.0,
        };
    }

    // Band ranges: [0, sub_bass_bin), [sub_bass_bin, low_bin), [low_bin, mid_bin), [mid_bin, num_bins)
    let band_ranges: [(usize, usize); 4] = [
        (0, sub_bass_bin),
        (sub_bass_bin, low_bin),
        (low_bin, mid_bin),
        (mid_bin, num_bins),
    ];

    let mut band_sums = [0.0f64; 4];
    let mut overall_sum = 0.0f64;

    let mut buf_a = vec![COMPLEX_ZERO; fft_size];
    let mut buf_b = vec![COMPLEX_ZERO; fft_size];

    for frame in 0..num_frames {
        let start = frame * hop_size;

        for i in 0..fft_size {
            let w = window[i];
            buf_a[i] = Complex::new(a[start + i] * w, 0.0);
            buf_b[i] = Complex::new(b[start + i] * w, 0.0);
        }

        fft.process(&mut buf_a);
        fft.process(&mut buf_b);

        // Per-band cosine similarity
        for (band_idx, &(lo, hi)) in band_ranges.iter().enumerate() {
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;

            for i in lo..hi.min(num_bins) {
                let ma = buf_a[i].norm() as f64;
                let mb = buf_b[i].norm() as f64;
                dot += ma * mb;
                na += ma * ma;
                nb += mb * mb;
            }

            let denom = (na * nb).sqrt();
            if denom > 1e-12 {
                band_sums[band_idx] += dot / denom;
            }
        }

        // Overall cosine similarity
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..num_bins {
            let ma = buf_a[i].norm() as f64;
            let mb = buf_b[i].norm() as f64;
            dot += ma * mb;
            na += ma * ma;
            nb += mb * mb;
        }
        let denom = (na * nb).sqrt();
        if denom > 1e-12 {
            overall_sum += dot / denom;
        }
    }

    let n = num_frames as f64;
    BandSimilarity {
        overall: overall_sum / n,
        sub_bass: band_sums[0] / n,
        low: band_sums[1] / n,
        mid: band_sums[2] / n,
        high: band_sums[3] / n,
    }
}

/// Computes per-band cosine similarity of averaged magnitude spectra.
///
/// Timing-invariant version of [`band_spectral_similarity`]. Computes the mean
/// magnitude spectrum for each signal, then compares per-band. This measures
/// whether the two signals have the same frequency balance in each band,
/// regardless of when events occur.
pub fn mean_band_spectral_similarity(
    a: &[f32],
    b: &[f32],
    fft_size: usize,
    hop_size: usize,
    sample_rate: u32,
) -> BandSimilarity {
    let bands = FrequencyBands::default();
    let window = generate_window(WindowType::Hann, fft_size);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let num_bins = fft_size / 2 + 1;

    let sub_bass_bin = freq_to_bin(bands.sub_bass, fft_size, sample_rate);
    let low_bin = freq_to_bin(bands.low, fft_size, sample_rate);
    let mid_bin = freq_to_bin(bands.mid, fft_size, sample_rate);

    let len_a = a.len();
    let len_b = b.len();
    let empty = BandSimilarity {
        overall: 0.0,
        sub_bass: 0.0,
        low: 0.0,
        mid: 0.0,
        high: 0.0,
    };
    if len_a < fft_size || len_b < fft_size {
        return empty;
    }
    let frames_a = (len_a - fft_size) / hop_size + 1;
    let frames_b = (len_b - fft_size) / hop_size + 1;
    if frames_a == 0 || frames_b == 0 {
        return empty;
    }

    let mut buf = vec![COMPLEX_ZERO; fft_size];

    // Accumulate mean magnitude spectra.
    let mut mean_a = vec![0.0f64; num_bins];
    for frame in 0..frames_a {
        let start = frame * hop_size;
        for i in 0..fft_size {
            buf[i] = Complex::new(a[start + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        for i in 0..num_bins {
            mean_a[i] += buf[i].norm() as f64;
        }
    }
    for v in &mut mean_a {
        *v /= frames_a as f64;
    }

    let mut mean_b = vec![0.0f64; num_bins];
    for frame in 0..frames_b {
        let start = frame * hop_size;
        for i in 0..fft_size {
            buf[i] = Complex::new(b[start + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        for i in 0..num_bins {
            mean_b[i] += buf[i].norm() as f64;
        }
    }
    for v in &mut mean_b {
        *v /= frames_b as f64;
    }

    // Per-band cosine similarity on mean spectra.
    let band_ranges: [(usize, usize); 4] = [
        (0, sub_bass_bin),
        (sub_bass_bin, low_bin),
        (low_bin, mid_bin),
        (mid_bin, num_bins),
    ];

    let cosine_sim = |lo: usize, hi: usize| -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in lo..hi.min(num_bins) {
            dot += mean_a[i] * mean_b[i];
            na += mean_a[i] * mean_a[i];
            nb += mean_b[i] * mean_b[i];
        }
        let denom = (na * nb).sqrt();
        if denom > 1e-12 {
            dot / denom
        } else {
            0.0
        }
    };

    let band_scores: Vec<f64> = band_ranges
        .iter()
        .map(|&(lo, hi)| cosine_sim(lo, hi))
        .collect();

    BandSimilarity {
        overall: cosine_sim(0, num_bins),
        sub_bass: band_scores[0],
        low: band_scores[1],
        mid: band_scores[2],
        high: band_scores[3],
    }
}

/// Computes normalized cross-correlation between two signals.
///
/// Returns the peak correlation value and the sample offset where it occurs.
/// Uses FFT-based cross-correlation for efficiency.
pub fn cross_correlation(a: &[f32], b: &[f32]) -> CrossCorrelationResult {
    if a.is_empty() || b.is_empty() {
        return CrossCorrelationResult {
            peak_value: 0.0,
            peak_offset: 0,
        };
    }

    // Use FFT-based cross-correlation
    // Pad to next power of 2 >= a.len() + b.len() - 1
    let corr_len = a.len() + b.len() - 1;
    let fft_size = corr_len.next_power_of_two();

    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(fft_size);
    let fft_inv = planner.plan_fft_inverse(fft_size);

    let zero = Complex::new(0.0f64, 0.0);

    // Zero-pad and FFT both signals
    let mut fa: Vec<Complex<f64>> = a
        .iter()
        .map(|&x| Complex::new(x as f64, 0.0))
        .chain(std::iter::repeat(zero))
        .take(fft_size)
        .collect();
    let mut fb: Vec<Complex<f64>> = b
        .iter()
        .map(|&x| Complex::new(x as f64, 0.0))
        .chain(std::iter::repeat(zero))
        .take(fft_size)
        .collect();

    fft_fwd.process(&mut fa);
    fft_fwd.process(&mut fb);

    // Cross-correlation in frequency domain: conj(A) * B
    let mut fc: Vec<Complex<f64>> = fa
        .iter()
        .zip(fb.iter())
        .map(|(&a_val, &b_val)| a_val.conj() * b_val)
        .collect();

    fft_inv.process(&mut fc);

    // Normalize by FFT size
    let inv_n = 1.0 / fft_size as f64;
    for c in fc.iter_mut() {
        *c *= inv_n;
    }

    // Compute energy norms for normalization
    let energy_a: f64 = a.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let energy_b: f64 = b.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let norm = (energy_a * energy_b).sqrt();

    if norm < 1e-12 {
        return CrossCorrelationResult {
            peak_value: 0.0,
            peak_offset: 0,
        };
    }

    // Find peak in the cross-correlation
    // Lags: [0, 1, ..., b.len()-1, -(a.len()-1), ..., -1]
    let mut peak_value = 0.0f64;
    let mut peak_idx = 0usize;

    for (i, c) in fc.iter().enumerate().take(corr_len) {
        let val = c.re.abs();
        if val > peak_value {
            peak_value = val;
            peak_idx = i;
        }
    }

    // Convert index to signed offset
    let peak_offset = if peak_idx < b.len() {
        peak_idx as isize
    } else {
        peak_idx as isize - fft_size as isize
    };

    CrossCorrelationResult {
        peak_value: (peak_value / norm).min(1.0),
        peak_offset,
    }
}

/// Compares transient onset positions between two signals.
///
/// Detects onsets in both signals and counts how many onsets in the reference
/// signal have a matching onset in the test signal within `tolerance_ms`.
/// Uses default detection parameters (fft=2048, hop=512, sensitivity=0.5).
/// For custom detection parameters, use [`transient_match_score_with_params`].
pub fn transient_match_score(
    reference: &[f32],
    test: &[f32],
    sample_rate: u32,
    tolerance_ms: f64,
) -> TransientMatchResult {
    transient_match_score_with_params(reference, test, sample_rate, tolerance_ms, 2048, 512, 0.5)
}

/// Compares transient onset positions between two signals with configurable
/// detection parameters.
///
/// This allows the benchmark to use the same detection settings as the
/// stretch algorithm being tested (e.g., DjBeatmatch uses sensitivity=0.45).
pub fn transient_match_score_with_params(
    reference: &[f32],
    test: &[f32],
    sample_rate: u32,
    tolerance_ms: f64,
    fft_size: usize,
    hop_size: usize,
    sensitivity: f32,
) -> TransientMatchResult {
    let ref_transients = detect_transients(reference, sample_rate, fft_size, hop_size, sensitivity);
    let test_transients = detect_transients(test, sample_rate, fft_size, hop_size, sensitivity);

    let tolerance_samples = (tolerance_ms * sample_rate as f64 / 1000.0) as usize;

    let mut matched = 0usize;
    for &ref_onset in &ref_transients.onsets {
        for &test_onset in &test_transients.onsets {
            let diff = ref_onset.abs_diff(test_onset);
            if diff <= tolerance_samples {
                matched += 1;
                break;
            }
        }
    }

    let total_reference = ref_transients.onsets.len();
    let match_rate = if total_reference > 0 {
        matched as f64 / total_reference as f64
    } else {
        1.0 // No reference onsets means trivial match
    };

    TransientMatchResult {
        match_rate,
        matched,
        total_reference,
        total_test: test_transients.onsets.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_spectral_similarity_identical() {
        let signal = sine_wave(440.0, 44100, 44100);
        let sim = spectral_similarity(&signal, &signal, 2048, 512);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical signals should have similarity ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_spectral_similarity_different_frequencies() {
        let a = sine_wave(440.0, 44100, 44100);
        let b = sine_wave(8000.0, 44100, 44100);
        let sim = spectral_similarity(&a, &b, 2048, 512);
        assert!(
            sim < 0.5,
            "Very different frequencies should have low similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_spectral_similarity_scaled() {
        let a = sine_wave(440.0, 44100, 44100);
        let b: Vec<f32> = a.iter().map(|&x| x * 0.5).collect();
        let sim = spectral_similarity(&a, &b, 2048, 512);
        // Cosine similarity is scale-invariant for magnitude spectra
        assert!(
            (sim - 1.0).abs() < 0.01,
            "Scaled signal should have similarity ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_spectral_similarity_empty() {
        let sim = spectral_similarity(&[], &[], 2048, 512);
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_spectral_similarity_too_short() {
        let a = vec![0.0f32; 100];
        let sim = spectral_similarity(&a, &a, 2048, 512);
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_band_spectral_similarity_identical() {
        let signal = sine_wave(440.0, 44100, 44100);
        let result = band_spectral_similarity(&signal, &signal, 2048, 512, 44100);
        assert!(
            (result.overall - 1.0).abs() < 1e-6,
            "Overall should be ~1.0, got {}",
            result.overall
        );
    }

    #[test]
    fn test_band_spectral_similarity_low_freq() {
        // 100 Hz signal should have high similarity in sub-bass, low elsewhere
        let a = sine_wave(100.0, 44100, 44100);
        let result = band_spectral_similarity(&a, &a, 2048, 512, 44100);
        assert!(
            result.sub_bass > 0.9,
            "Sub-bass self-similarity should be high, got {}",
            result.sub_bass
        );
    }

    #[test]
    fn test_cross_correlation_identical() {
        let signal = sine_wave(440.0, 44100, 4410);
        let result = cross_correlation(&signal, &signal);
        assert!(
            result.peak_value > 0.95,
            "Identical signals should have peak ~1.0, got {}",
            result.peak_value
        );
        assert_eq!(
            result.peak_offset, 0,
            "Identical signals should have zero offset, got {}",
            result.peak_offset
        );
    }

    #[test]
    fn test_cross_correlation_shifted() {
        let signal = sine_wave(440.0, 44100, 4410);
        // Shift by 10 samples
        let mut shifted = vec![0.0f32; 10];
        shifted.extend_from_slice(&signal);
        let result = cross_correlation(&signal, &shifted);
        assert!(
            result.peak_value > 0.9,
            "Shifted signal should have high correlation, got {}",
            result.peak_value
        );
        assert_eq!(
            result.peak_offset, 10,
            "Should detect 10-sample shift, got {}",
            result.peak_offset
        );
    }

    #[test]
    fn test_cross_correlation_empty() {
        let result = cross_correlation(&[], &[]);
        assert!((result.peak_value - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross_correlation_silence() {
        let silence = vec![0.0f32; 1000];
        let result = cross_correlation(&silence, &silence);
        assert!((result.peak_value - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_transient_match_identical() {
        // Click train at known positions
        let sample_rate = 44100u32;
        let mut signal = vec![0.0f32; sample_rate as usize * 2];
        let click_interval = sample_rate as usize / 2;
        for pos in (0..signal.len()).step_by(click_interval) {
            for j in 0..10.min(signal.len() - pos) {
                signal[pos + j] = if j < 5 { 1.0 } else { -0.5 };
            }
        }

        let result = transient_match_score(&signal, &signal, sample_rate, 10.0);
        assert!(
            result.match_rate > 0.9,
            "Identical signals should match well, got {}",
            result.match_rate
        );
    }

    #[test]
    fn test_transient_match_no_transients() {
        let silence = vec![0.0f32; 44100];
        let result = transient_match_score(&silence, &silence, 44100, 10.0);
        // No onsets in reference → trivial match
        assert!(
            (result.match_rate - 1.0).abs() < 1e-6,
            "No reference onsets should give match_rate 1.0, got {}",
            result.match_rate
        );
        assert_eq!(result.total_reference, 0);
    }

    #[test]
    fn test_transient_match_short_signal() {
        let short = vec![0.0f32; 100];
        let result = transient_match_score(&short, &short, 44100, 10.0);
        assert_eq!(result.total_reference, 0);
        assert_eq!(result.total_test, 0);
    }
}

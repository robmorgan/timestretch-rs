//! Audio comparison metrics for benchmarking time-stretch quality.
//!
//! Provides spectral similarity, band-level spectral similarity,
//! cross-correlation, transient match scoring, perceptual weighting,
//! onset timing analysis, LUFS loudness measurement, Bark-scale band
//! similarity, spectral flux comparison, and a comprehensive quality
//! report for comparing library output against professional reference audio.

use rustfft::{num_complex::Complex, FftPlanner};

use crate::analysis::frequency::{bin_to_freq, freq_to_bin, FrequencyBands};
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

// ---------------------------------------------------------------------------
// Perceptual spectral weighting (A-weighting)
// ---------------------------------------------------------------------------

/// A-weighting curve approximation for perceptual frequency weighting.
///
/// Based on IEC 61672, this models human hearing sensitivity across frequencies.
/// Humans are most sensitive around 1-4 kHz and less sensitive at very low and
/// very high frequencies.
fn a_weight(freq_hz: f64) -> f64 {
    let f2 = freq_hz * freq_hz;
    let num = 12194.0_f64.powi(2) * f2 * f2;
    let denom = (f2 + 20.6_f64.powi(2))
        * ((f2 + 107.7_f64.powi(2)) * (f2 + 737.9_f64.powi(2))).sqrt()
        * (f2 + 12194.0_f64.powi(2));
    if denom > 0.0 {
        num / denom
    } else {
        0.0
    }
}

/// Computes perceptually-weighted STFT magnitude cosine similarity.
///
/// Similar to [`spectral_similarity`] but weights each frequency bin by its
/// A-weighting value, emphasizing perceptually important frequencies (1-4 kHz)
/// and de-emphasizing sub-bass and very high frequencies. This gives a metric
/// that better correlates with human perception of spectral quality.
pub fn perceptual_spectral_similarity(
    a: &[f32],
    b: &[f32],
    fft_size: usize,
    hop_size: usize,
    sample_rate: u32,
) -> f64 {
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

    // Precompute A-weights for each bin.
    let weights: Vec<f64> = (0..num_bins)
        .map(|i| {
            let freq = bin_to_freq(i, fft_size, sample_rate) as f64;
            a_weight(freq)
        })
        .collect();

    let mut buf_a = vec![COMPLEX_ZERO; fft_size];
    let mut buf_b = vec![COMPLEX_ZERO; fft_size];
    let mut similarity_sum = 0.0f64;

    for frame in 0..num_frames {
        let start = frame * hop_size;

        for i in 0..fft_size {
            let w = window[i];
            buf_a[i] = Complex::new(a[start + i] * w, 0.0);
            buf_b[i] = Complex::new(b[start + i] * w, 0.0);
        }

        fft.process(&mut buf_a);
        fft.process(&mut buf_b);

        // Weighted cosine similarity of magnitude spectra.
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;

        for i in 0..num_bins {
            let w = weights[i];
            let ma = buf_a[i].norm() as f64 * w;
            let mb = buf_b[i].norm() as f64 * w;
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

// ---------------------------------------------------------------------------
// Onset timing error distribution
// ---------------------------------------------------------------------------

/// Detailed onset timing error distribution between test and reference signals.
///
/// Instead of a binary match/no-match with a fixed tolerance, this provides
/// the full distribution of timing errors, giving much more insight into how
/// well transient timing is preserved.
#[derive(Debug, Clone)]
pub struct OnsetTimingAnalysis {
    /// Average absolute timing offset in milliseconds.
    pub mean_error_ms: f64,
    /// Median absolute timing offset in milliseconds.
    pub median_error_ms: f64,
    /// Standard deviation of timing errors in milliseconds.
    pub std_dev_ms: f64,
    /// Worst-case (maximum) absolute timing error in milliseconds.
    pub max_error_ms: f64,
    /// Count of onsets within +/-5ms of reference.
    pub within_5ms: usize,
    /// Count of onsets within +/-10ms of reference.
    pub within_10ms: usize,
    /// Count of onsets within +/-20ms of reference.
    pub within_20ms: usize,
    /// Total onsets compared.
    pub total_onsets: usize,
}

/// Analyzes onset timing error distribution between test and reference signals.
///
/// For each detected onset in the reference signal, finds the nearest onset in
/// the test signal and computes the signed error in milliseconds. Returns full
/// distribution statistics including mean, median, standard deviation, and
/// counts at various tolerance thresholds.
///
/// Uses default detection parameters (fft=2048, hop=512, sensitivity=0.5).
pub fn onset_timing_analysis(
    reference: &[f32],
    test: &[f32],
    sample_rate: u32,
) -> OnsetTimingAnalysis {
    onset_timing_analysis_with_params(reference, test, sample_rate, 2048, 512, 0.5)
}

/// Analyzes onset timing error distribution with configurable detection parameters.
pub fn onset_timing_analysis_with_params(
    reference: &[f32],
    test: &[f32],
    sample_rate: u32,
    fft_size: usize,
    hop_size: usize,
    sensitivity: f32,
) -> OnsetTimingAnalysis {
    let ref_transients = detect_transients(reference, sample_rate, fft_size, hop_size, sensitivity);
    let test_transients = detect_transients(test, sample_rate, fft_size, hop_size, sensitivity);

    let empty = OnsetTimingAnalysis {
        mean_error_ms: 0.0,
        median_error_ms: 0.0,
        std_dev_ms: 0.0,
        max_error_ms: 0.0,
        within_5ms: 0,
        within_10ms: 0,
        within_20ms: 0,
        total_onsets: 0,
    };

    if ref_transients.onsets.is_empty() || test_transients.onsets.is_empty() {
        return empty;
    }

    let samples_to_ms = 1000.0 / sample_rate as f64;

    // For each reference onset, find the nearest test onset and compute signed error.
    let mut errors_ms: Vec<f64> = Vec::with_capacity(ref_transients.onsets.len());
    for &ref_onset in &ref_transients.onsets {
        let mut best_dist = f64::MAX;
        for &test_onset in &test_transients.onsets {
            let dist_ms = (test_onset as f64 - ref_onset as f64) * samples_to_ms;
            if dist_ms.abs() < best_dist.abs() {
                best_dist = dist_ms;
            }
        }
        errors_ms.push(best_dist);
    }

    let total_onsets = errors_ms.len();
    let abs_errors: Vec<f64> = errors_ms.iter().map(|e| e.abs()).collect();

    // Mean absolute error.
    let mean_error_ms = abs_errors.iter().sum::<f64>() / total_onsets as f64;

    // Median absolute error.
    let mut sorted_abs = abs_errors.clone();
    sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_error_ms = if total_onsets % 2 == 0 && total_onsets >= 2 {
        (sorted_abs[total_onsets / 2 - 1] + sorted_abs[total_onsets / 2]) / 2.0
    } else {
        sorted_abs[total_onsets / 2]
    };

    // Standard deviation of absolute errors.
    let variance = abs_errors
        .iter()
        .map(|e| (e - mean_error_ms).powi(2))
        .sum::<f64>()
        / total_onsets as f64;
    let std_dev_ms = variance.sqrt();

    // Max absolute error.
    let max_error_ms = sorted_abs.last().copied().unwrap_or(0.0);

    // Counts at tolerance thresholds.
    let within_5ms = abs_errors.iter().filter(|&&e| e <= 5.0).count();
    let within_10ms = abs_errors.iter().filter(|&&e| e <= 10.0).count();
    let within_20ms = abs_errors.iter().filter(|&&e| e <= 20.0).count();

    OnsetTimingAnalysis {
        mean_error_ms,
        median_error_ms,
        std_dev_ms,
        max_error_ms,
        within_5ms,
        within_10ms,
        within_20ms,
        total_onsets,
    }
}

// ---------------------------------------------------------------------------
// LUFS loudness measurement
// ---------------------------------------------------------------------------

/// Computes a simplified integrated LUFS (Loudness Units relative to Full Scale).
///
/// This is a simplified estimation based on RMS power with the standard -0.691
/// LUFS offset. A full implementation would include K-weighting (high-shelf at
/// 1500 Hz and high-pass at 38 Hz), but this simplified version provides a
/// useful loudness estimate for comparison purposes.
pub fn estimate_lufs(samples: &[f32], _sample_rate: u32) -> f64 {
    if samples.is_empty() {
        return -70.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let mean_sq = sum_sq / samples.len() as f64;
    if mean_sq > 0.0 {
        -0.691 + 10.0 * mean_sq.log10()
    } else {
        -70.0 // silence floor
    }
}

/// Computes the LUFS loudness difference between test and reference signals.
///
/// A positive value means the test signal is louder than the reference.
/// A value near 0.0 means loudness is well-matched.
pub fn lufs_difference(test: &[f32], reference: &[f32], sample_rate: u32) -> f64 {
    estimate_lufs(test, sample_rate) - estimate_lufs(reference, sample_rate)
}

// ---------------------------------------------------------------------------
// Bark-scale per-band similarity
// ---------------------------------------------------------------------------

/// Number of Bark-scale critical bands used for perceptual band analysis.
pub const BARK_BAND_COUNT: usize = 8;

/// Bark-scale critical band frequency boundaries (Hz).
///
/// These approximate human auditory critical bands:
/// - Band 0: 0-100 Hz (sub-bass)
/// - Band 1: 100-200 Hz (bass)
/// - Band 2: 200-400 Hz (low-mid)
/// - Band 3: 400-840 Hz (mid)
/// - Band 4: 840-1720 Hz (upper-mid)
/// - Band 5: 1720-3400 Hz (presence)
/// - Band 6: 3400-7000 Hz (brilliance)
/// - Band 7: 7000-15000 Hz (air)
const BARK_BAND_EDGES: [f32; 9] = [
    0.0, 100.0, 200.0, 400.0, 840.0, 1720.0, 3400.0, 7000.0, 15000.0,
];

/// Bark-scale band names for display purposes.
pub const BARK_BAND_NAMES: [&str; BARK_BAND_COUNT] = [
    "sub-bass",
    "bass",
    "low-mid",
    "mid",
    "upper-mid",
    "presence",
    "brilliance",
    "air",
];

/// Per-band spectral similarity using Bark-scale critical bands.
#[derive(Debug, Clone)]
pub struct BarkBandSimilarity {
    /// Similarity scores for each of the 8 Bark bands (0.0-1.0).
    pub bands: [f64; BARK_BAND_COUNT],
    /// Overall weighted similarity across all bands (0.0-1.0).
    pub overall: f64,
}

/// Computes per-band spectral similarity using Bark-scale critical bands.
///
/// Unlike [`band_spectral_similarity`] which uses fixed EDM-tuned frequency
/// ranges, this uses 8 Bark-scale critical bands that model human auditory
/// perception. Each band's cosine similarity is computed independently,
/// and the overall score is the mean of all band scores.
pub fn bark_band_similarity(
    a: &[f32],
    b: &[f32],
    fft_size: usize,
    hop_size: usize,
    sample_rate: u32,
) -> BarkBandSimilarity {
    let window = generate_window(WindowType::Hann, fft_size);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let num_bins = fft_size / 2 + 1;

    let empty = BarkBandSimilarity {
        bands: [0.0; BARK_BAND_COUNT],
        overall: 0.0,
    };

    let min_len = a.len().min(b.len());
    if min_len < fft_size {
        return empty;
    }

    let num_frames = (min_len - fft_size) / hop_size + 1;
    if num_frames == 0 {
        return empty;
    }

    // Precompute bin ranges for each Bark band.
    let band_ranges: Vec<(usize, usize)> = (0..BARK_BAND_COUNT)
        .map(|i| {
            let lo = freq_to_bin(BARK_BAND_EDGES[i], fft_size, sample_rate);
            let hi = freq_to_bin(BARK_BAND_EDGES[i + 1], fft_size, sample_rate).min(num_bins);
            (lo, hi)
        })
        .collect();

    let mut band_sums = [0.0f64; BARK_BAND_COUNT];
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

        for (band_idx, &(lo, hi)) in band_ranges.iter().enumerate() {
            if lo >= hi {
                continue;
            }
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;

            for i in lo..hi {
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
    }

    let n = num_frames as f64;
    let mut bands = [0.0f64; BARK_BAND_COUNT];
    for i in 0..BARK_BAND_COUNT {
        bands[i] = band_sums[i] / n;
    }

    let overall = bands.iter().sum::<f64>() / BARK_BAND_COUNT as f64;

    BarkBandSimilarity { bands, overall }
}

// ---------------------------------------------------------------------------
// Spectral flux comparison
// ---------------------------------------------------------------------------

/// Computes frame-by-frame spectral flux (onset strength signal).
///
/// For each frame, sums the positive magnitude differences from the previous
/// frame. This measures how much the spectrum changes between frames -- large
/// values indicate transients.
pub fn compute_spectral_flux(signal: &[f32], fft_size: usize, hop_size: usize) -> Vec<f32> {
    let window = generate_window(WindowType::Hann, fft_size);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let num_bins = fft_size / 2 + 1;

    if signal.len() < fft_size {
        return Vec::new();
    }

    let num_frames = (signal.len() - fft_size) / hop_size + 1;
    if num_frames < 2 {
        return Vec::new();
    }

    let mut buf = vec![COMPLEX_ZERO; fft_size];
    let mut prev_mags = vec![0.0f32; num_bins];
    let mut flux = Vec::with_capacity(num_frames - 1);

    for frame in 0..num_frames {
        let start = frame * hop_size;

        for i in 0..fft_size {
            buf[i] = Complex::new(signal[start + i] * window[i], 0.0);
        }
        fft.process(&mut buf);

        let curr_mags: Vec<f32> = (0..num_bins).map(|i| buf[i].norm()).collect();

        if frame > 0 {
            let frame_flux: f32 = curr_mags
                .iter()
                .zip(prev_mags.iter())
                .map(|(&curr, &prev)| (curr - prev).max(0.0))
                .sum();
            flux.push(frame_flux);
        }

        prev_mags.copy_from_slice(&curr_mags);
    }

    flux
}

/// Compares spectral flux profiles between test and reference signals.
///
/// Computes the normalized cross-correlation of the two spectral flux signals.
/// High similarity means transients are equally sharp and occur at similar
/// relative positions. Returns a value in 0.0-1.0.
pub fn spectral_flux_similarity(a: &[f32], b: &[f32], fft_size: usize, hop_size: usize) -> f64 {
    let flux_a = compute_spectral_flux(a, fft_size, hop_size);
    let flux_b = compute_spectral_flux(b, fft_size, hop_size);

    if flux_a.is_empty() || flux_b.is_empty() {
        return 0.0;
    }

    // Use the shorter length for comparison.
    let len = flux_a.len().min(flux_b.len());
    let fa = &flux_a[..len];
    let fb = &flux_b[..len];

    // Cosine similarity of the two flux signals.
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..len {
        let va = fa[i] as f64;
        let vb = fb[i] as f64;
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 1e-12 {
        (dot / denom).max(0.0)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Comprehensive quality report
// ---------------------------------------------------------------------------

/// Comprehensive quality report combining all available metrics.
///
/// Ties together spectral similarity, perceptual weighting, cross-correlation,
/// onset timing analysis, loudness comparison, Bark-band scores, spectral flux,
/// and an overall letter grade.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Frame-by-frame spectral similarity (0.0-1.0).
    pub spectral_similarity: f64,
    /// A-weighted perceptual spectral similarity (0.0-1.0).
    pub perceptual_spectral_similarity: f64,
    /// Peak normalized cross-correlation (0.0-1.0).
    pub cross_correlation: f64,
    /// Onset timing error distribution.
    pub onset_timing: OnsetTimingAnalysis,
    /// LUFS loudness difference in dB (test minus reference).
    pub lufs_difference: f64,
    /// Per-band similarity using Bark-scale critical bands.
    pub bark_band_scores: [f64; BARK_BAND_COUNT],
    /// Spectral flux similarity (transient sharpness) (0.0-1.0).
    pub spectral_flux_similarity: f64,
    /// Overall letter grade (A through F).
    pub overall_grade: char,
}

/// Generates a comprehensive quality report comparing test audio against a reference.
///
/// Runs all available metrics and combines them into a single [`QualityReport`]
/// with an overall letter grade. The grade is computed as a weighted combination:
/// - 30% perceptual spectral similarity
/// - 20% cross-correlation
/// - 20% onset timing (fraction within 10ms)
/// - 15% spectral flux similarity
/// - 10% Bark band overall similarity
/// - 5% loudness match (penalty for large differences)
pub fn generate_quality_report(
    test: &[f32],
    reference: &[f32],
    sample_rate: u32,
    fft_size: usize,
    hop_size: usize,
) -> QualityReport {
    let spec_sim = spectral_similarity(test, reference, fft_size, hop_size);
    let perc_sim = perceptual_spectral_similarity(test, reference, fft_size, hop_size, sample_rate);

    let max_corr_samples = (sample_rate as usize * 10)
        .min(test.len())
        .min(reference.len());
    let xcorr = if max_corr_samples > 0 {
        cross_correlation(&test[..max_corr_samples], &reference[..max_corr_samples])
    } else {
        CrossCorrelationResult {
            peak_value: 0.0,
            peak_offset: 0,
        }
    };

    let timing = onset_timing_analysis(reference, test, sample_rate);
    let lufs_diff = lufs_difference(test, reference, sample_rate);
    let bark = bark_band_similarity(test, reference, fft_size, hop_size, sample_rate);
    let flux_sim = spectral_flux_similarity(test, reference, fft_size, hop_size);

    // Compute overall score (0.0-1.0) from weighted components.
    let timing_score = if timing.total_onsets > 0 {
        timing.within_10ms as f64 / timing.total_onsets as f64
    } else {
        1.0 // No onsets = trivially perfect timing
    };

    // Loudness score: 1.0 when perfectly matched, decreasing with difference.
    // 3 dB difference -> ~0.5 score, 6 dB -> ~0.25.
    let loudness_score = (-lufs_diff.abs() / 3.0).exp2();

    let overall_score = 0.30 * perc_sim
        + 0.20 * xcorr.peak_value
        + 0.20 * timing_score
        + 0.15 * flux_sim
        + 0.10 * bark.overall
        + 0.05 * loudness_score;

    let overall_grade = score_to_grade(overall_score);

    QualityReport {
        spectral_similarity: spec_sim,
        perceptual_spectral_similarity: perc_sim,
        cross_correlation: xcorr.peak_value,
        onset_timing: timing,
        lufs_difference: lufs_diff,
        bark_band_scores: bark.bands,
        spectral_flux_similarity: flux_sim,
        overall_grade,
    }
}

/// Result of beat grid regularity comparison between two audio signals.
#[derive(Debug, Clone)]
pub struct BeatGridRegularityResult {
    /// Overall regularity match score (0.0-1.0).
    /// High values mean both signals have similar beat regularity.
    pub score: f64,
    /// Beat periodicity of the reference signal (0.0-1.0).
    /// How regular/consistent the reference beats are at the expected tempo.
    pub ref_periodicity: f64,
    /// Beat periodicity of the test signal (0.0-1.0).
    pub test_periodicity: f64,
}

/// Measures beat grid regularity in both reference and test signals and compares them.
///
/// Detects onsets in each signal, then computes autocorrelation of the onset envelope
/// at the expected beat period to measure how regular the beat grid is. The score
/// reflects how well the test signal preserves the reference's rhythmic regularity.
///
/// # Arguments
/// * `ref_signal` - Reference audio (mono)
/// * `test_signal` - Test audio (mono)
/// * `sample_rate` - Sample rate in Hz
/// * `expected_bpm` - Expected tempo in BPM
/// * `fft_size` - FFT size for onset detection
/// * `hop_size` - Hop size for onset detection
/// * `sensitivity` - Onset detection sensitivity
pub fn beat_grid_regularity_with_params(
    ref_signal: &[f32],
    test_signal: &[f32],
    sample_rate: u32,
    expected_bpm: f64,
    fft_size: usize,
    hop_size: usize,
    sensitivity: f32,
) -> BeatGridRegularityResult {
    let ref_periodicity = compute_beat_periodicity(
        ref_signal,
        sample_rate,
        expected_bpm,
        fft_size,
        hop_size,
        sensitivity,
    );
    let test_periodicity = compute_beat_periodicity(
        test_signal,
        sample_rate,
        expected_bpm,
        fft_size,
        hop_size,
        sensitivity,
    );

    // Score: penalize difference in periodicity, but also reward high periodicity in both
    let diff_penalty = 1.0 - (ref_periodicity - test_periodicity).abs();
    let avg_periodicity = (ref_periodicity + test_periodicity) / 2.0;
    let score = (0.5 * diff_penalty + 0.5 * avg_periodicity).clamp(0.0, 1.0);

    BeatGridRegularityResult {
        score,
        ref_periodicity,
        test_periodicity,
    }
}

/// Computes beat periodicity by autocorrelation of the onset envelope at the expected tempo.
fn compute_beat_periodicity(
    signal: &[f32],
    sample_rate: u32,
    expected_bpm: f64,
    fft_size: usize,
    hop_size: usize,
    sensitivity: f32,
) -> f64 {
    if signal.is_empty() || expected_bpm <= 0.0 {
        return 0.0;
    }

    // Detect transients/onsets
    let transients = detect_transients(signal, sample_rate, fft_size, hop_size, sensitivity);
    if transients.onsets.is_empty() {
        return 0.0;
    }

    // Build an onset impulse envelope at hop resolution
    let num_frames = signal.len() / hop_size;
    if num_frames == 0 {
        return 0.0;
    }
    let mut envelope = vec![0.0f64; num_frames];
    for (i, &onset) in transients.onsets.iter().enumerate() {
        let frame = onset / hop_size;
        if frame < num_frames {
            let strength = if i < transients.strengths.len() {
                transients.strengths[i] as f64
            } else {
                1.0
            };
            envelope[frame] = strength;
        }
    }

    // Expected beat period in frames
    let beat_period_samples = 60.0 * sample_rate as f64 / expected_bpm;
    let beat_period_frames = beat_period_samples / hop_size as f64;
    let lag = beat_period_frames.round() as usize;

    if lag == 0 || lag >= num_frames / 2 {
        return 0.0;
    }

    // Compute normalized autocorrelation at the beat period lag
    let mean = envelope.iter().sum::<f64>() / num_frames as f64;
    let mut auto_corr = 0.0;
    let mut energy = 0.0;
    for i in 0..num_frames - lag {
        let a = envelope[i] - mean;
        let b = envelope[i + lag] - mean;
        auto_corr += a * b;
        energy += a * a;
    }

    if energy < 1e-12 {
        return 0.0;
    }

    // Normalized autocorrelation at beat lag, clamped to [0, 1]
    (auto_corr / energy).clamp(0.0, 1.0)
}

/// Converts a 0.0-1.0 score to a letter grade.
fn score_to_grade(score: f64) -> char {
    if score >= 0.9 {
        'A'
    } else if score >= 0.8 {
        'B'
    } else if score >= 0.7 {
        'C'
    } else if score >= 0.6 {
        'D'
    } else {
        'F'
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

    // --- A-weighting tests ---

    #[test]
    fn test_a_weight_peak_around_2khz() {
        // A-weighting peaks around 2-4 kHz; should be higher there than at 100 Hz or 10 kHz.
        let w_100 = a_weight(100.0);
        let w_2500 = a_weight(2500.0);
        let w_10000 = a_weight(10000.0);
        assert!(
            w_2500 > w_100,
            "A-weight at 2500 Hz ({}) should exceed 100 Hz ({})",
            w_2500,
            w_100
        );
        assert!(
            w_2500 > w_10000,
            "A-weight at 2500 Hz ({}) should exceed 10000 Hz ({})",
            w_2500,
            w_10000
        );
    }

    #[test]
    fn test_a_weight_zero_freq() {
        let w = a_weight(0.0);
        assert!(w.abs() < 1e-6, "A-weight at 0 Hz should be ~0, got {}", w);
    }

    #[test]
    fn test_perceptual_spectral_similarity_identical() {
        let signal = sine_wave(1000.0, 44100, 44100);
        let sim = perceptual_spectral_similarity(&signal, &signal, 2048, 512, 44100);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical signals should have perceptual similarity ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_perceptual_spectral_similarity_different_freq() {
        let a = sine_wave(440.0, 44100, 44100);
        let b = sine_wave(8000.0, 44100, 44100);
        let sim = perceptual_spectral_similarity(&a, &b, 2048, 512, 44100);
        assert!(
            sim < 0.5,
            "Very different frequencies should have low perceptual similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_perceptual_spectral_similarity_empty() {
        let sim = perceptual_spectral_similarity(&[], &[], 2048, 512, 44100);
        assert!(
            sim.abs() < 1e-6,
            "Empty signals should give 0.0, got {}",
            sim
        );
    }

    // --- Onset timing analysis tests ---

    #[test]
    fn test_onset_timing_identical_clicks() {
        let sample_rate = 44100u32;
        let mut signal = vec![0.0f32; sample_rate as usize * 2];
        let click_interval = sample_rate as usize / 2;
        for pos in (0..signal.len()).step_by(click_interval) {
            for j in 0..10.min(signal.len() - pos) {
                signal[pos + j] = if j < 5 { 1.0 } else { -0.5 };
            }
        }

        let analysis = onset_timing_analysis(&signal, &signal, sample_rate);
        // Identical signals should have very small timing errors.
        if analysis.total_onsets > 0 {
            assert!(
                analysis.mean_error_ms < 1.0,
                "Identical signals should have near-zero mean error, got {} ms",
                analysis.mean_error_ms
            );
        }
    }

    #[test]
    fn test_onset_timing_empty_signals() {
        let silence = vec![0.0f32; 44100];
        let analysis = onset_timing_analysis(&silence, &silence, 44100);
        assert_eq!(analysis.total_onsets, 0);
        assert!(analysis.mean_error_ms.abs() < 1e-6);
    }

    // --- LUFS tests ---

    #[test]
    fn test_estimate_lufs_silence() {
        let silence = vec![0.0f32; 44100];
        let lufs = estimate_lufs(&silence, 44100);
        assert!(
            lufs <= -70.0 + 1e-6,
            "Silence should be at or below -70 LUFS, got {}",
            lufs
        );
    }

    #[test]
    fn test_estimate_lufs_full_scale_sine() {
        // A full-scale sine wave has RMS = 1/sqrt(2) ≈ 0.707
        // mean_sq = 0.5, so 10*log10(0.5) ≈ -3.01
        // LUFS ≈ -0.691 + (-3.01) = -3.70
        let signal = sine_wave(1000.0, 44100, 44100);
        let lufs = estimate_lufs(&signal, 44100);
        assert!(
            (lufs - (-3.70)).abs() < 0.1,
            "Full-scale sine LUFS should be ~-3.70, got {}",
            lufs
        );
    }

    #[test]
    fn test_estimate_lufs_empty() {
        let lufs = estimate_lufs(&[], 44100);
        assert!(
            (lufs - (-70.0)).abs() < 1e-6,
            "Empty signal should be -70.0 LUFS, got {}",
            lufs
        );
    }

    #[test]
    fn test_lufs_difference_identical() {
        let signal = sine_wave(440.0, 44100, 44100);
        let diff = lufs_difference(&signal, &signal, 44100);
        assert!(
            diff.abs() < 1e-6,
            "Identical signals should have 0.0 LUFS difference, got {}",
            diff
        );
    }

    #[test]
    fn test_lufs_difference_quieter() {
        let signal = sine_wave(440.0, 44100, 44100);
        let quiet: Vec<f32> = signal.iter().map(|&x| x * 0.5).collect();
        let diff = lufs_difference(&quiet, &signal, 44100);
        // Half amplitude = -6 dB in amplitude, which is -6.02 dB in power
        assert!(
            diff < -5.0,
            "Half-amplitude signal should be ~6 dB quieter, got {} dB",
            diff
        );
    }

    // --- Bark band similarity tests ---

    #[test]
    fn test_bark_band_similarity_identical() {
        let signal = sine_wave(1000.0, 44100, 44100);
        let result = bark_band_similarity(&signal, &signal, 2048, 512, 44100);
        assert!(
            result.overall > 0.9,
            "Identical signals should have high Bark band similarity, got {}",
            result.overall
        );
    }

    #[test]
    fn test_bark_band_similarity_low_freq() {
        // A 50 Hz sine should have energy mainly in the sub-bass Bark band.
        let a = sine_wave(50.0, 44100, 44100);
        let result = bark_band_similarity(&a, &a, 2048, 512, 44100);
        assert!(
            result.bands[0] > 0.9,
            "Sub-bass Bark band self-similarity should be high, got {}",
            result.bands[0]
        );
    }

    #[test]
    fn test_bark_band_similarity_empty() {
        let result = bark_band_similarity(&[], &[], 2048, 512, 44100);
        assert!(
            result.overall.abs() < 1e-6,
            "Empty signals should give 0.0, got {}",
            result.overall
        );
    }

    // --- Spectral flux tests ---

    #[test]
    fn test_spectral_flux_steady_signal() {
        // A steady sine wave should have very low spectral flux.
        let signal = sine_wave(440.0, 44100, 44100);
        let flux = compute_spectral_flux(&signal, 2048, 512);
        assert!(!flux.is_empty(), "Should produce flux frames");
        let max_flux = flux.iter().cloned().fold(0.0f32, f32::max);
        // After the initial ramp-up, flux should be small for a steady tone.
        // Check that most frames have low flux.
        let low_flux_count = flux.iter().filter(|&&f| f < max_flux * 0.5).count();
        assert!(
            low_flux_count > flux.len() / 2,
            "Steady signal should have mostly low flux"
        );
    }

    #[test]
    fn test_spectral_flux_empty() {
        let flux = compute_spectral_flux(&[], 2048, 512);
        assert!(flux.is_empty());
    }

    #[test]
    fn test_spectral_flux_similarity_identical() {
        let signal = sine_wave(440.0, 44100, 44100);
        let sim = spectral_flux_similarity(&signal, &signal, 2048, 512);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical signals should have flux similarity 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_spectral_flux_similarity_empty() {
        let sim = spectral_flux_similarity(&[], &[], 2048, 512);
        assert!(sim.abs() < 1e-6);
    }

    // --- Quality report tests ---

    #[test]
    fn test_quality_report_identical() {
        let signal = sine_wave(440.0, 44100, 44100);
        let report = generate_quality_report(&signal, &signal, 44100, 2048, 512);
        assert!(
            (report.spectral_similarity - 1.0).abs() < 1e-6,
            "Spectral similarity should be 1.0 for identical signals"
        );
        assert!(
            (report.perceptual_spectral_similarity - 1.0).abs() < 1e-6,
            "Perceptual spectral similarity should be 1.0 for identical signals"
        );
        assert!(
            report.cross_correlation > 0.95,
            "Cross-correlation should be high for identical signals, got {}",
            report.cross_correlation
        );
        assert!(
            report.lufs_difference.abs() < 1e-6,
            "LUFS difference should be 0.0 for identical signals"
        );
        assert!(
            (report.spectral_flux_similarity - 1.0).abs() < 1e-6,
            "Spectral flux similarity should be 1.0 for identical signals"
        );
        // Grade should be A for identical signals.
        assert_eq!(
            report.overall_grade, 'A',
            "Identical signals should get grade A, got {}",
            report.overall_grade
        );
    }

    #[test]
    fn test_score_to_grade() {
        assert_eq!(score_to_grade(0.95), 'A');
        assert_eq!(score_to_grade(0.90), 'A');
        assert_eq!(score_to_grade(0.85), 'B');
        assert_eq!(score_to_grade(0.75), 'C');
        assert_eq!(score_to_grade(0.65), 'D');
        assert_eq!(score_to_grade(0.50), 'F');
        assert_eq!(score_to_grade(0.0), 'F');
    }
}

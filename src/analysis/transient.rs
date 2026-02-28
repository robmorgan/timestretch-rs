//! Spectral-flux transient detection with adaptive thresholding.
//!
//! Combines spectral flux (frequency-domain onset measure) with an energy
//! envelope detector (time-domain onset measure) and phase deviation analysis
//! for more robust transient detection. Uses mean + stddev adaptive
//! thresholding instead of sliding median for better handling of varying
//! signal dynamics.
//!
//! Phase deviation detection computes the expected phase advance for each FFT
//! bin based on the hop size and compares it to the actual phase, providing
//! ~10ms temporal precision for onset localization (vs ~50ms for magnitude-only
//! spectral flux).

use rustfft::{num_complex::Complex, FftPlanner};

use crate::core::fft::COMPLEX_ZERO;
use crate::core::types::{StretchParams, TransientThresholdPolicy};

/// Weight of spectral flux in the combined onset detection function.
const FLUX_WEIGHT: f32 = 0.6;
/// Weight of onset energy envelope in the combined onset detection function.
const ENERGY_WEIGHT: f32 = 0.4;

/// Weight of spectral flux component in multi-signal onset detection.
const SPECTRAL_FLUX_ONSET_WEIGHT: f32 = 0.6;
/// Weight of phase deviation component in multi-signal onset detection.
const PHASE_DEVIATION_ONSET_WEIGHT: f32 = 0.4;
/// Smoothing coefficient for the onset energy envelope (one-pole lowpass).
/// Higher values = slower response. 0.9 gives ~10-frame smoothing.
const ENERGY_SMOOTH_ALPHA: f32 = 0.9;

/// Result of transient detection: sample positions of detected onsets.
#[derive(Debug, Clone)]
pub struct TransientMap {
    /// Sample positions of detected transient onsets (integer, for backward compatibility).
    pub onsets: Vec<usize>,
    /// Fractional-sample onset positions refined using phase information.
    /// These offer sub-sample precision for beat grid alignment.
    /// Length matches `onsets`; each value is a refinement of the corresponding
    /// integer onset position.
    pub onsets_fractional: Vec<f64>,
    /// Normalized onset strengths in [0, 1], one per onset.
    /// Higher values indicate stronger transients (kicks) vs weaker ones (hi-hats).
    pub strengths: Vec<f32>,
    /// Combined detection function values at each analysis frame.
    pub flux: Vec<f32>,
    /// Hop size used for analysis.
    pub hop_size: usize,
    /// Per-frame spectral flux broken down by frequency band.
    /// Each element is `[sub_bass, low, mid, high]` flux for that analysis frame.
    /// Band boundaries: sub-bass <100Hz, low 100-500Hz, mid 500-4000Hz, high >4000Hz.
    pub per_frame_band_flux: Vec<[f32; 4]>,
}

/// Configuration for onset lookahead confirmation.
///
/// These controls tune how aggressively isolated one-frame spikes are rejected.
#[derive(Debug, Clone, Copy)]
pub struct TransientDetectionOptions {
    /// Number of future frames to inspect before confirming a candidate onset.
    /// `0` disables lookahead confirmation.
    pub lookahead_confirm_frames: usize,
    /// Relaxation factor applied to the local threshold for future-frame checks.
    /// Lower values accept weaker continuation; range: `[0.0, 1.0]`.
    pub lookahead_threshold_relax: f32,
    /// Relative continuation threshold versus the candidate peak.
    /// Range: `[0.0, 1.0]`.
    pub lookahead_peak_retain_ratio: f32,
    /// Multiplier above local threshold that bypasses lookahead checks entirely.
    /// Values `< 1.0` are treated as `1.0` at runtime.
    pub strong_spike_bypass_multiplier: f32,
    /// Adaptive threshold policy (median window, floor, onset-gap profile).
    pub threshold_policy: TransientThresholdPolicy,
}

impl TransientDetectionOptions {
    /// Builds transient-detection options from user-facing stretch parameters.
    #[inline]
    pub fn from_stretch_params(params: &StretchParams) -> Self {
        Self {
            lookahead_confirm_frames: params.transient_lookahead_frames,
            lookahead_threshold_relax: params.transient_lookahead_threshold_relax,
            lookahead_peak_retain_ratio: params.transient_lookahead_peak_retain_ratio,
            strong_spike_bypass_multiplier: params.transient_strong_spike_bypass_multiplier,
            threshold_policy: params.transient_threshold_policy.sanitized(),
        }
    }
}

/// Band boundary frequencies for per-band flux (Hz).
const BAND_FLUX_LOW_LIMIT: f32 = 100.0;
const BAND_FLUX_MID_LIMIT: f32 = 500.0;
const BAND_FLUX_HIGH_LIMIT: f32 = 4000.0;

/// Computes the spectral flux for each frame of a mono audio signal.
///
/// Returns `(flux_values, band_flux)` where:
/// - `flux_values` is a vector of total weighted flux, one per analysis frame
/// - `band_flux` is per-frame `[sub_bass, low, mid, high]` flux values
fn compute_spectral_flux(
    samples: &[f32],
    sample_rate: u32,
    fft_size: usize,
    hop_size: usize,
) -> (Vec<f32>, Vec<[f32; 4]>) {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let window =
        crate::core::window::generate_window(crate::core::window::WindowType::Hann, fft_size);

    let bin_weights = compute_bin_weights(fft_size, sample_rate);
    let bin_freq = sample_rate as f32 / fft_size as f32;
    let num_bins = fft_size / 2 + 1;
    let num_frames = (samples.len() - fft_size) / hop_size + 1;
    let mut prev_magnitude = vec![0.0f32; num_bins];
    let mut flux_values = Vec::with_capacity(num_frames);
    let mut band_flux_values = Vec::with_capacity(num_frames);
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
        let mut band_flux = [0.0f32; 4]; // [sub_bass, low, mid, high]
        for (bin, ((&c, prev), &weight)) in fft_buffer[..num_bins]
            .iter()
            .zip(prev_magnitude.iter_mut())
            .zip(bin_weights[..num_bins].iter())
            .enumerate()
        {
            let mag = c.norm();
            let diff = mag - *prev;
            if diff > 0.0 {
                flux += diff * weight;
                // Accumulate per-band flux
                let freq = bin as f32 * bin_freq;
                let band_idx = if freq < BAND_FLUX_LOW_LIMIT {
                    0 // sub-bass
                } else if freq < BAND_FLUX_MID_LIMIT {
                    1 // low
                } else if freq < BAND_FLUX_HIGH_LIMIT {
                    2 // mid
                } else {
                    3 // high
                };
                band_flux[band_idx] += diff;
            }
            *prev = mag;
        }

        flux_values.push(flux);
        band_flux_values.push(band_flux);
    }

    (flux_values, band_flux_values)
}

/// Computes the onset energy envelope for each analysis frame.
///
/// For each frame, computes the RMS energy, then takes the half-wave rectified
/// first difference (energy increase only) and smooths it with a one-pole
/// lowpass filter. This captures broadband energy onsets that spectral flux
/// might miss (e.g., low-frequency kicks).
fn compute_onset_energy(samples: &[f32], fft_size: usize, hop_size: usize) -> Vec<f32> {
    let num_frames = (samples.len() - fft_size) / hop_size + 1;
    let mut energies = Vec::with_capacity(num_frames);
    let inv_fft = 1.0 / fft_size as f32;

    // Compute RMS energy per frame
    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let end = start + fft_size;
        let rms: f32 = samples[start..end].iter().map(|&s| s * s).sum::<f32>() * inv_fft;
        energies.push(rms.sqrt());
    }

    // Half-wave rectified first difference + smoothing
    let mut envelope = Vec::with_capacity(num_frames);
    let mut smoothed = 0.0f32;

    for i in 0..num_frames {
        let diff = if i > 0 {
            (energies[i] - energies[i - 1]).max(0.0)
        } else {
            energies[0]
        };
        smoothed = ENERGY_SMOOTH_ALPHA * smoothed + (1.0 - ENERGY_SMOOTH_ALPHA) * diff;
        envelope.push(smoothed);
    }

    envelope
}

/// Wraps a phase value to the range [-PI, PI].
#[inline]
fn wrap_phase(phase: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    let p = phase + std::f32::consts::PI;
    p - (p / two_pi).floor() * two_pi - std::f32::consts::PI
}

/// Computes phase deviation (phase vocoder derivative) for each analysis frame.
///
/// For each FFT bin, the expected phase advance between frames is
/// `2 * PI * bin * hop_size / fft_size`. The deviation from this expected
/// phase indicates an onset (phase discontinuity). The result is weighted
/// by the same frequency-band weights used for spectral flux.
///
/// Also returns per-frame information about the strongest phase deviation bin,
/// which is used for fractional-sample onset refinement.
fn compute_phase_deviation(
    samples: &[f32],
    sample_rate: u32,
    fft_size: usize,
    hop_size: usize,
) -> (Vec<f32>, Vec<PhaseDeviationInfo>) {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let window =
        crate::core::window::generate_window(crate::core::window::WindowType::Hann, fft_size);
    let bin_weights = compute_bin_weights(fft_size, sample_rate);
    let num_bins = fft_size / 2 + 1;
    let num_frames = if samples.len() >= fft_size {
        (samples.len() - fft_size) / hop_size + 1
    } else {
        return (vec![], vec![]);
    };

    let two_pi = 2.0 * std::f32::consts::PI;

    // Pre-compute expected phase advance per bin: 2 * PI * bin * hop / fft_size
    let phase_advance: Vec<f32> = (0..num_bins)
        .map(|bin| two_pi * bin as f32 * hop_size as f32 / fft_size as f32)
        .collect();

    let mut prev_phase = vec![0.0f32; num_bins];
    let mut deviation_values = Vec::with_capacity(num_frames);
    let mut deviation_info = Vec::with_capacity(num_frames);
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

        let mut total_deviation = 0.0f32;
        let mut max_deviation = 0.0f32;
        let mut max_deviation_bin: usize = 0;
        let mut max_deviation_phase_diff = 0.0f32;

        for (bin, ((&c, prev), &weight)) in fft_buffer[..num_bins]
            .iter()
            .zip(prev_phase.iter_mut())
            .zip(bin_weights[..num_bins].iter())
            .enumerate()
        {
            let mag = c.norm();
            let current_phase = c.arg();
            let expected = *prev + phase_advance[bin];
            let deviation = wrap_phase(current_phase - expected).abs();

            // Weight by magnitude to gate noise: phase in low-energy bins
            // is unreliable and would amplify noise during normalization.
            let weighted = deviation * weight * mag;
            total_deviation += weighted;

            if weighted > max_deviation {
                max_deviation = weighted;
                max_deviation_bin = bin;
                max_deviation_phase_diff = wrap_phase(current_phase - expected);
            }

            *prev = current_phase;
        }

        deviation_values.push(total_deviation);
        deviation_info.push(PhaseDeviationInfo {
            strongest_bin: max_deviation_bin,
            phase_diff: max_deviation_phase_diff,
        });
    }

    (deviation_values, deviation_info)
}

/// Information about phase deviation at the strongest bin in a given frame,
/// used for fractional-sample onset position refinement.
#[derive(Debug, Clone, Copy)]
struct PhaseDeviationInfo {
    /// FFT bin index with the strongest phase deviation in this frame.
    strongest_bin: usize,
    /// Signed phase difference (wrapped) at the strongest bin.
    phase_diff: f32,
}

/// Computes fractional-sample onset positions from integer onset positions
/// and per-frame phase deviation information.
///
/// Uses the phase deviation at the strongest onset bin to estimate a
/// sub-frame offset:
///   `fractional_offset = -phase_diff / (2 * PI * freq_bin / fft_size)`
///
/// The result is clamped to within one hop of the integer position.
fn compute_fractional_positions(
    onsets: &[usize],
    deviation_info: &[PhaseDeviationInfo],
    hop_size: usize,
    fft_size: usize,
) -> Vec<f64> {
    let two_pi = 2.0 * std::f64::consts::PI;

    onsets
        .iter()
        .map(|&onset_sample| {
            let frame_idx = onset_sample / hop_size;
            if frame_idx >= deviation_info.len() {
                return onset_sample as f64;
            }

            let info = deviation_info[frame_idx];
            if info.strongest_bin == 0 {
                // DC bin — no meaningful phase refinement possible
                return onset_sample as f64;
            }

            // fractional_offset = -phase_diff / (2 * PI * bin / fft_size)
            let bin_freq_factor = two_pi * info.strongest_bin as f64 / fft_size as f64;
            let fractional_offset = -(info.phase_diff as f64) / bin_freq_factor;

            // Clamp to within one hop to prevent wild jumps
            let clamped = fractional_offset.clamp(-(hop_size as f64), hop_size as f64);

            (onset_sample as f64 + clamped).max(0.0)
        })
        .collect()
}

/// Minimum energy envelope max to be considered meaningful.
/// Below this, the energy channel is zeroed to prevent noise amplification.
const ENERGY_GATE_THRESHOLD: f32 = 0.01;

/// Combines spectral flux, onset energy envelope, and phase deviation into a
/// single detection function.
///
/// All three signals are normalized to [0, 1] by their respective maxima before
/// weighting, so neither dominates regardless of absolute scale. The energy
/// channel is gated: if its maximum is below `ENERGY_GATE_THRESHOLD`, it is
/// excluded to prevent noise from being amplified during normalization.
///
/// The spectral flux and phase deviation are combined using a two-level scheme:
/// first they are mixed using `SPECTRAL_FLUX_ONSET_WEIGHT` and
/// `PHASE_DEVIATION_ONSET_WEIGHT`, then the result is blended with the energy
/// envelope using `FLUX_WEIGHT` and `ENERGY_WEIGHT`.
fn combine_detection_functions(flux: &[f32], energy: &[f32], phase_deviation: &[f32]) -> Vec<f32> {
    let max_flux = flux.iter().copied().fold(0.0f32, f32::max);
    let max_energy = energy.iter().copied().fold(0.0f32, f32::max);
    let max_phase = phase_deviation.iter().copied().fold(0.0f32, f32::max);

    let flux_norm = if max_flux > 1e-10 { max_flux } else { 1.0 };
    let phase_norm = if max_phase > 1e-10 { max_phase } else { 1.0 };

    // Gate the energy channel: if peak energy is too low, the signal has no
    // meaningful transients in the time domain and normalizing would amplify noise.
    let use_energy = max_energy > ENERGY_GATE_THRESHOLD;
    let energy_norm = if use_energy { max_energy } else { 1.0 };

    let n = flux.len();
    (0..n)
        .map(|i| {
            let f = flux[i] / flux_norm;
            let pd = if i < phase_deviation.len() {
                phase_deviation[i] / phase_norm
            } else {
                0.0
            };

            // Combine spectral flux and phase deviation
            let spectral_combined =
                SPECTRAL_FLUX_ONSET_WEIGHT * f + PHASE_DEVIATION_ONSET_WEIGHT * pd;

            let e_contrib = if use_energy {
                ENERGY_WEIGHT * (energy[i] / energy_norm)
            } else {
                0.0
            };

            FLUX_WEIGHT * spectral_combined + e_contrib
        })
        .collect()
}

/// Detects transients in a mono audio signal using combined spectral flux,
/// phase deviation, and onset energy envelope.
///
/// Uses high-frequency weighted spectral flux combined with phase vocoder
/// derivative detection and a time-domain onset energy detector, with
/// mean+stddev adaptive thresholding tuned for EDM transient detection
/// (kicks, snares, hi-hats).
///
/// Phase deviation detection provides ~10ms temporal precision compared
/// to ~50ms for magnitude-only spectral flux, significantly improving
/// onset localization accuracy.
pub fn detect_transients(
    samples: &[f32],
    sample_rate: u32,
    fft_size: usize,
    hop_size: usize,
    sensitivity: f32,
) -> TransientMap {
    let default_params = StretchParams::default();
    detect_transients_with_options(
        samples,
        sample_rate,
        fft_size,
        hop_size,
        sensitivity,
        TransientDetectionOptions::from_stretch_params(&default_params),
    )
}

/// Detects transients in a mono signal with configurable lookahead confirmation.
pub fn detect_transients_with_options(
    samples: &[f32],
    sample_rate: u32,
    fft_size: usize,
    hop_size: usize,
    sensitivity: f32,
    options: TransientDetectionOptions,
) -> TransientMap {
    if samples.len() < fft_size {
        return TransientMap {
            onsets: vec![],
            onsets_fractional: vec![],
            strengths: vec![],
            flux: vec![],
            hop_size,
            per_frame_band_flux: vec![],
        };
    }

    let (flux_values, band_flux) = compute_spectral_flux(samples, sample_rate, fft_size, hop_size);
    let energy_envelope = compute_onset_energy(samples, fft_size, hop_size);
    let (phase_deviation, deviation_info) =
        compute_phase_deviation(samples, sample_rate, fft_size, hop_size);
    let combined = combine_detection_functions(&flux_values, &energy_envelope, &phase_deviation);

    // Use policy-configured sensitivity-aware gap to detect rapid patterns.
    let min_gap = options
        .threshold_policy
        .sanitized()
        .min_gap_for_sensitivity(sensitivity);
    let onsets = adaptive_threshold_with_gap(&combined, sensitivity, hop_size, min_gap, options);

    // Compute onset strengths from detection function values
    let strengths = compute_onset_strengths(&combined, &onsets, hop_size);

    // Compute fractional-sample onset positions using phase information
    let onsets_fractional =
        compute_fractional_positions(&onsets, &deviation_info, hop_size, fft_size);

    TransientMap {
        onsets,
        onsets_fractional,
        strengths,
        flux: combined,
        hop_size,
        per_frame_band_flux: band_flux,
    }
}

// Frequency band boundaries for transient weighting (Hz).
const BAND_SUB_BASS_LIMIT: f32 = 100.0;
const BAND_BASS_MID_LIMIT: f32 = 500.0;
const BAND_MID_LIMIT: f32 = 2000.0;
const BAND_HIGH_MID_LIMIT: f32 = 8000.0;

// Spectral flux weights per frequency band.
/// Sub-bass (<100 Hz): high weight — captures kick drum fundamentals.
/// EDM-optimized: equal prominence for kick detection.
const WEIGHT_SUB_BASS: f32 = 1.0;
/// Bass/low-mid (100–500 Hz): high weight — kick body.
/// EDM-optimized: increased to improve kick drum detection.
const WEIGHT_BASS_MID: f32 = 0.9;
/// Mid (500–2000 Hz): moderate weight.
const WEIGHT_MID: f32 = 0.8;
/// High-mid (2–8 kHz): elevated weight — hi-hats, snare attacks.
/// EDM-optimized: reduced from 1.5 to avoid hi-hat false positives.
const WEIGHT_HIGH_MID: f32 = 1.2;
/// Very high (>8 kHz): moderate weight — noise content.
/// EDM-optimized: slightly reduced to limit noise-driven false positives.
const WEIGHT_VERY_HIGH: f32 = 0.7;

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

impl Default for TransientDetectionOptions {
    fn default() -> Self {
        let params = StretchParams::default();
        Self::from_stretch_params(&params)
    }
}

/// Adaptive thresholding with a configurable minimum gap between onsets.
///
/// Uses a sliding median with multiplicative threshold. The median is robust
/// to outliers (unlike mean+stddev), making it well-suited for signals with
/// a mix of strong transients and quiet passages.
fn adaptive_threshold_with_gap(
    flux: &[f32],
    sensitivity: f32,
    hop_size: usize,
    min_gap_frames: usize,
    options: TransientDetectionOptions,
) -> Vec<usize> {
    if flux.is_empty() {
        return vec![];
    }

    let threshold_policy = options.threshold_policy.sanitized();
    let half_window = threshold_policy.median_window_frames / 2;
    // Higher sensitivity = lower threshold = more detections
    let threshold_multiplier =
        1.0 + (1.0 - sensitivity).max(0.0) * threshold_policy.threshold_sensitivity_slope.max(0.0);

    let mut onsets = Vec::new();
    let mut last_onset: Option<usize> = None;
    // Reusable sort buffer to avoid per-frame allocation
    let mut local = Vec::with_capacity(threshold_policy.median_window_frames);

    let lookahead_frames = options.lookahead_confirm_frames;
    let threshold_relax = options.lookahead_threshold_relax.clamp(0.0, 1.0);
    let peak_retain_ratio = options.lookahead_peak_retain_ratio.clamp(0.0, 1.0);
    let strong_bypass_multiplier = options.strong_spike_bypass_multiplier.max(1.0);

    for (i, &flux_val) in flux.iter().enumerate() {
        // Compute local median
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(flux.len());
        local.clear();
        local.extend_from_slice(&flux[start..end]);
        local.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = local[local.len() / 2];

        let threshold = median * threshold_multiplier + threshold_policy.threshold_floor;

        if flux_val > threshold {
            let strong_spike = flux_val > threshold * strong_bypass_multiplier;
            if !strong_spike && lookahead_frames > 0 && i + 1 < flux.len() {
                let lookahead_end = i.saturating_add(lookahead_frames).min(flux.len() - 1);
                let relaxed_threshold = threshold * threshold_relax;
                let retain_level = flux_val * peak_retain_ratio;
                let confirmed = flux[i + 1..=lookahead_end]
                    .iter()
                    .any(|&future| future > relaxed_threshold || future > retain_level);

                if !confirmed {
                    continue;
                }
            }

            // Check minimum gap
            if let Some(last) = last_onset {
                if i - last < min_gap_frames {
                    continue;
                }
            }
            onsets.push(i * hop_size);
            last_onset = Some(i);
        }
    }

    onsets
}

/// Computes normalized onset strengths from the detection function values.
///
/// For each onset sample position, looks up the detection function value at
/// the corresponding frame and normalizes all values to [0, 1].
fn compute_onset_strengths(combined: &[f32], onsets: &[usize], hop_size: usize) -> Vec<f32> {
    if onsets.is_empty() {
        return vec![];
    }

    // Get raw detection values at each onset frame
    let raw: Vec<f32> = onsets
        .iter()
        .map(|&onset_sample| {
            let frame = onset_sample / hop_size;
            if frame < combined.len() {
                combined[frame]
            } else {
                0.0
            }
        })
        .collect();

    // Normalize to [0, 1]
    let max_val = raw.iter().copied().fold(0.0f32, f32::max);
    if max_val < 1e-10 {
        return vec![0.0; onsets.len()];
    }

    raw.iter().map(|&v| v / max_val).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: adaptive_threshold with default gap.
    fn adaptive_threshold(flux: &[f32], sensitivity: f32, hop_size: usize) -> Vec<usize> {
        let options = TransientDetectionOptions::default();
        let min_gap = options
            .threshold_policy
            .min_gap_for_sensitivity(sensitivity);
        adaptive_threshold_with_gap(flux, sensitivity, hop_size, min_gap, options)
    }

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
        assert!(result.strengths.is_empty());
        assert!(result.flux.is_empty());
    }

    #[test]
    fn test_detect_transients_strengths() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let mut samples = vec![0.0f32; num_samples];

        // Strong click and weaker click
        let click_positions = [0, sample_rate as usize / 2, sample_rate as usize];
        let click_amplitudes = [1.0f32, 0.3, 1.0];
        for (&pos, &amp) in click_positions.iter().zip(click_amplitudes.iter()) {
            for j in 0..10.min(num_samples - pos) {
                samples[pos + j] = if j < 5 { amp } else { -amp * 0.5 };
            }
        }

        let result = detect_transients(&samples, sample_rate, 2048, 512, 0.5);
        assert_eq!(
            result.onsets.len(),
            result.strengths.len(),
            "Onsets and strengths should have same length"
        );
        // All strengths should be in [0, 1]
        for &s in &result.strengths {
            assert!(
                (0.0..=1.0).contains(&s),
                "Strength {} out of [0, 1] range",
                s
            );
        }
        // At least one strength should be 1.0 (the maximum)
        if !result.strengths.is_empty() {
            let max_strength = result.strengths.iter().copied().fold(0.0f32, f32::max);
            assert!(
                (max_strength - 1.0).abs() < 1e-6,
                "Max strength should be 1.0, got {}",
                max_strength
            );
        }
    }

    #[test]
    fn test_bin_weights() {
        let weights = compute_bin_weights(4096, 44100);
        assert_eq!(weights.len(), 2049);
        // Sub-bass should have EDM-optimized weight (1.0 for kick prominence)
        assert!(
            (weights[0] - WEIGHT_SUB_BASS).abs() < 1e-6,
            "DC bin weight should be WEIGHT_SUB_BASS ({}), got {}",
            WEIGHT_SUB_BASS,
            weights[0]
        );
        // 4kHz bin should have elevated weight (high-mid band)
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
    fn test_adaptive_threshold_lookahead_rejects_weak_isolated_spike() {
        let mut flux = vec![0.0f32; 50];
        flux[25] = 0.02; // above floor, but below strong-spike bypass

        let result = adaptive_threshold_with_gap(
            &flux,
            0.5,
            512,
            TransientDetectionOptions::default()
                .threshold_policy
                .min_gap_for_sensitivity(0.5),
            TransientDetectionOptions::default(),
        );
        assert!(
            result.is_empty(),
            "Weak isolated spike should be rejected by lookahead confirmation"
        );
    }

    #[test]
    fn test_adaptive_threshold_lookahead_zero_detects_weak_isolated_spike() {
        let mut flux = vec![0.0f32; 50];
        flux[25] = 0.02;

        let result = adaptive_threshold_with_gap(
            &flux,
            0.5,
            512,
            TransientDetectionOptions::default()
                .threshold_policy
                .min_gap_for_sensitivity(0.5),
            TransientDetectionOptions {
                lookahead_confirm_frames: 0,
                ..TransientDetectionOptions::default()
            },
        );
        assert_eq!(result, vec![25 * 512]);
    }

    #[test]
    fn test_default_options_match_stretch_params_defaults() {
        let params = StretchParams::default();
        let options = TransientDetectionOptions::default();
        assert_eq!(
            options.lookahead_confirm_frames,
            params.transient_lookahead_frames
        );
        assert!(
            (options.lookahead_threshold_relax - params.transient_lookahead_threshold_relax).abs()
                < 1e-6
        );
        assert!(
            (options.lookahead_peak_retain_ratio - params.transient_lookahead_peak_retain_ratio)
                .abs()
                < 1e-6
        );
        assert!(
            (options.strong_spike_bypass_multiplier
                - params.transient_strong_spike_bypass_multiplier)
                .abs()
                < 1e-6
        );
        assert_eq!(
            options.threshold_policy,
            params.transient_threshold_policy.sanitized()
        );
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
        let (flux, band_flux) = compute_spectral_flux(&samples, 44100, 2048, 512);
        assert!(!flux.is_empty());
        for &f in &flux {
            assert!(f.abs() < 1e-6, "Flux for silence should be ~0, got {}", f);
        }
        // Band flux should also be all zeros for silence
        assert_eq!(flux.len(), band_flux.len());
        for bf in &band_flux {
            for &v in bf {
                assert!(v.abs() < 1e-6, "Band flux for silence should be ~0");
            }
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

        let (flux, _band_flux) = compute_spectral_flux(&input, sample_rate, 2048, 512);
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

        let (flux, _band_flux) = compute_spectral_flux(&input, sample_rate, fft_size, hop_size);

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

    #[test]
    fn test_per_frame_band_flux_populated() {
        // Verify that detect_transients populates per_frame_band_flux
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize;
        let mut samples = vec![0.0f32; num_samples];
        // Add an impulse
        for j in 0..10 {
            samples[num_samples / 2 + j] = if j < 5 { 1.0 } else { -0.5 };
        }
        let result = detect_transients(&samples, sample_rate, 2048, 512, 0.5);
        assert!(!result.per_frame_band_flux.is_empty());
        assert_eq!(result.flux.len(), result.per_frame_band_flux.len());
    }
}

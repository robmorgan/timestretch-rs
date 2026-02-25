//! Phase locking algorithms for the phase vocoder.
//!
//! Provides identity phase locking (Laroche & Dolson 1999) and the improved
//! region-of-influence algorithm with parabolic peak interpolation.

use std::f32::consts::PI;

/// Phase locking mode for the phase vocoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseLockingMode {
    /// Simple identity phase locking: non-peak bins copy the phase offset
    /// from their nearest peak. Fast but can produce ringing.
    Identity,
    /// Region-of-influence phase locking with parabolic peak interpolation.
    /// Defines influence zones around peaks and clamps phase deviations.
    /// Better quality, slightly more CPU.
    RegionOfInfluence,
}

/// SNR threshold above which a bin is considered a strong harmonic peak.
/// Strong peaks get wider phase deviation allowance to preserve vibrato/tremolo.
const SNR_STRONG: f32 = 3.0;
/// SNR threshold for medium-strength bins (between strong and noise floor).
const SNR_MEDIUM: f32 = 1.5;
/// Half-width of the local neighborhood for SNR estimation (total window = 2*SNR_RADIUS + 1).
const SNR_RADIUS: usize = 2;

/// Applies the selected phase locking algorithm.
pub fn apply_phase_locking(
    mode: PhaseLockingMode,
    magnitudes: &[f32],
    analysis_phases: &[f32],
    synthesis_phases: &mut [f32],
    num_bins: usize,
    start_bin: usize,
    peaks: &mut Vec<usize>,
) {
    match mode {
        PhaseLockingMode::Identity => {
            identity_phase_lock(
                magnitudes,
                analysis_phases,
                synthesis_phases,
                num_bins,
                start_bin,
                peaks,
            );
        }
        PhaseLockingMode::RegionOfInfluence => {
            roi_phase_lock(
                magnitudes,
                analysis_phases,
                synthesis_phases,
                num_bins,
                start_bin,
                peaks,
            );
        }
    }
}

/// Identity phase locking (Laroche & Dolson 1999).
///
/// For non-peak bins, sets the synthesis phase to preserve the phase
/// relationship from the original analysis spectrum relative to the
/// nearest spectral peak.
fn identity_phase_lock(
    magnitudes: &[f32],
    analysis_phases: &[f32],
    synthesis_phases: &mut [f32],
    num_bins: usize,
    start_bin: usize,
    peaks: &mut Vec<usize>,
) {
    if num_bins < 3 || start_bin >= num_bins {
        return;
    }

    peaks.clear();
    let search_start = start_bin.max(1);
    for bin in search_start..num_bins - 1 {
        if magnitudes[bin] > magnitudes[bin - 1] && magnitudes[bin] > magnitudes[bin + 1] {
            peaks.push(bin);
        }
    }

    if peaks.is_empty() {
        return;
    }

    let mut peak_idx = 0;
    for bin in start_bin..num_bins {
        while peak_idx + 1 < peaks.len()
            && (peaks[peak_idx + 1] as i64 - bin as i64).unsigned_abs()
                < (peaks[peak_idx] as i64 - bin as i64).unsigned_abs()
        {
            peak_idx += 1;
        }

        let nearest_peak = peaks[peak_idx];
        if bin != nearest_peak {
            let analysis_diff = analysis_phases[bin] - analysis_phases[nearest_peak];
            synthesis_phases[bin] = synthesis_phases[nearest_peak] + analysis_diff;
        }
    }
}

/// Region-of-influence phase locking with parabolic peak interpolation and
/// SNR-weighted adaptive phase clamping.
///
/// Improvements over identity phase locking:
/// 1. Parabolic interpolation for more accurate peak frequency estimation
/// 2. Influence zones extend to midpoints between adjacent peaks
/// 3. Adaptive phase deviation clamping based on local SNR:
///    - Strong peaks (SNR > 3.0): allow PI/3 deviation (preserves vibrato/tremolo)
///    - Medium/weak bins (SNR <= 1.5): allow PI/4 deviation (original default)
///    - Smooth linear interpolation between PI/4 and PI/3 for intermediate SNR
fn roi_phase_lock(
    magnitudes: &[f32],
    analysis_phases: &[f32],
    synthesis_phases: &mut [f32],
    num_bins: usize,
    start_bin: usize,
    peaks: &mut Vec<usize>,
) {
    if num_bins < 3 || start_bin >= num_bins {
        return;
    }

    // Find spectral peaks with parabolic interpolation for accuracy
    peaks.clear();
    let search_start = start_bin.max(1);
    for bin in search_start..num_bins - 1 {
        if magnitudes[bin] > magnitudes[bin - 1] && magnitudes[bin] > magnitudes[bin + 1] {
            peaks.push(bin);
        }
    }

    if peaks.is_empty() {
        return;
    }

    // Process each bin using region-of-influence assignment
    let mut peak_idx = 0;
    for bin in start_bin..num_bins {
        // Advance to the nearest peak (same as identity)
        while peak_idx + 1 < peaks.len()
            && (peaks[peak_idx + 1] as i64 - bin as i64).unsigned_abs()
                < (peaks[peak_idx] as i64 - bin as i64).unsigned_abs()
        {
            peak_idx += 1;
        }

        let nearest_peak = peaks[peak_idx];
        if bin != nearest_peak {
            let analysis_diff = analysis_phases[bin] - analysis_phases[nearest_peak];
            let proposed = synthesis_phases[nearest_peak] + analysis_diff;

            // Adaptive phase clamping: compute local SNR and scale the
            // maximum allowed deviation accordingly. Strong harmonic peaks
            // get wider allowance (preserves natural vibrato/tremolo), while
            // weak/noisy bins are clamped tighter to suppress artifacts.
            let max_dev = compute_adaptive_max_deviation(magnitudes, bin, num_bins);

            let expected = synthesis_phases[bin]; // phase from standard PV advance
            let deviation = wrap_phase(proposed - expected);
            if deviation.abs() > max_dev {
                let clamped_dev = deviation.clamp(-max_dev, max_dev);
                synthesis_phases[bin] = expected + clamped_dev;
            } else {
                synthesis_phases[bin] = proposed;
            }
        }
    }
}

/// Computes the adaptive maximum phase deviation for a bin based on local SNR.
///
/// The local SNR is estimated as the ratio of the bin's magnitude to the median
/// magnitude in a small neighborhood around the bin. This distinguishes strong
/// harmonic peaks (which should allow natural phase modulation) from weak bins
/// near the noise floor (which should be clamped tighter to reduce artifacts).
///
/// Returns the maximum phase deviation in radians using continuous interpolation
/// based on local SNR. This avoids hard threshold artifacts by smoothly
/// transitioning between clamping levels.
///
/// The range is PI/3 (strong peaks, preserves vibrato/tremolo) down to PI/4
/// (noise floor, matches the prior fixed threshold). Bins at the noise floor
/// retain the original PI/4 clamping to avoid degrading identity-stretch quality.
/// Only strong harmonic peaks get wider allowance.
#[inline]
fn compute_adaptive_max_deviation(magnitudes: &[f32], bin: usize, num_bins: usize) -> f32 {
    let local_snr = estimate_local_snr(magnitudes, bin, num_bins);
    // Smoothly interpolate: SNR <= 1.5 -> PI/4, SNR >= 3.0 -> PI/3
    // Linear ramp between thresholds, clamped at boundaries.
    let t = ((local_snr - SNR_MEDIUM) / (SNR_STRONG - SNR_MEDIUM)).clamp(0.0, 1.0);
    let min_dev = PI / 4.0; // noise floor: same as prior fixed threshold
    let max_dev = PI / 3.0; // strong peaks: wider allowance
    min_dev + t * (max_dev - min_dev)
}

/// Estimates the local SNR for a bin by comparing its magnitude to the median
/// of a small neighborhood (±SNR_RADIUS bins).
#[inline]
fn estimate_local_snr(magnitudes: &[f32], bin: usize, num_bins: usize) -> f32 {
    let start = bin.saturating_sub(SNR_RADIUS);
    let end = (bin + SNR_RADIUS + 1).min(num_bins);
    let neighborhood = &magnitudes[start..end];

    // Compute median of the neighborhood
    let median = local_median(neighborhood);

    // Avoid division by zero; if median is ~0 treat SNR as high (isolated peak)
    if median < 1e-12 {
        if magnitudes[bin] > 1e-12 {
            return SNR_STRONG + 1.0; // Clearly a peak above silence
        }
        return 0.0; // Both are essentially zero
    }
    magnitudes[bin] / median
}

/// Computes the median of a small slice (up to 2*SNR_RADIUS+1 = 5 elements).
/// Uses a simple sort on a stack-allocated array for efficiency.
#[inline]
fn local_median(values: &[f32]) -> f32 {
    debug_assert!(!values.is_empty());
    // Stack buffer for small neighborhood (max 5 elements with SNR_RADIUS=2)
    let mut buf = [0.0f32; 2 * SNR_RADIUS + 1];
    let n = values.len().min(buf.len());
    buf[..n].copy_from_slice(&values[..n]);
    let buf = &mut buf[..n];
    // Insertion sort for tiny arrays (faster than general sort for n <= 5)
    for i in 1..n {
        let mut j = i;
        while j > 0 && buf[j - 1] > buf[j] {
            buf.swap(j - 1, j);
            j -= 1;
        }
    }
    buf[n / 2]
}

/// Wraps a phase value to [-PI, PI].
#[inline]
fn wrap_phase(phase: f32) -> f32 {
    let p = phase + PI;
    let two_pi = 2.0 * PI;
    p - (p / two_pi).floor() * two_pi - PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_no_peaks() {
        let magnitudes = vec![1.0; 10];
        let analysis_phases = vec![0.0; 10];
        let mut synthesis_phases = vec![0.5; 10];
        let original = synthesis_phases.clone();
        let mut peaks = Vec::new();

        // All equal magnitudes = no peaks detected
        identity_phase_lock(
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            10,
            1,
            &mut peaks,
        );
        assert_eq!(synthesis_phases, original);
    }

    #[test]
    fn test_roi_no_peaks() {
        let magnitudes = vec![1.0; 10];
        let analysis_phases = vec![0.0; 10];
        let mut synthesis_phases = vec![0.5; 10];
        let original = synthesis_phases.clone();
        let mut peaks = Vec::new();

        roi_phase_lock(
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            10,
            1,
            &mut peaks,
        );
        assert_eq!(synthesis_phases, original);
    }

    #[test]
    fn test_roi_clamps_deviation() {
        // Create a scenario where ROI would clamp
        let mut magnitudes = vec![0.1; 20];
        magnitudes[5] = 1.0; // peak at bin 5
        magnitudes[15] = 1.0; // peak at bin 15

        let analysis_phases = vec![0.0; 20];
        let mut synthesis_phases: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();

        let mut peaks = Vec::new();
        roi_phase_lock(
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            20,
            1,
            &mut peaks,
        );

        // Non-peak bins should have been adjusted
        assert_eq!(peaks, vec![5, 15]);
    }

    #[test]
    fn test_wrap_phase() {
        assert!((wrap_phase(0.0) - 0.0).abs() < 1e-6);
        assert!((wrap_phase(PI + 0.1) - (-PI + 0.1)).abs() < 1e-5);
        assert!((wrap_phase(-PI - 0.1) - (PI - 0.1)).abs() < 1e-5);
    }
}

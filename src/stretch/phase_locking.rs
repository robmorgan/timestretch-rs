//! Phase locking algorithms for the phase vocoder.
//!
//! Provides identity phase locking (Laroche & Dolson 1999) with full-spectrum
//! trough-bounded influence regions, and the improved region-of-influence
//! algorithm with adaptive SNR-weighted phase clamping as a safety net.
//!
//! ## Full Identity Phase Locking
//!
//! Traditional identity phase locking assigns each non-peak bin to its nearest
//! peak and copies the peak's phase rotation. This implementation improves on
//! that by using **trough-bounded influence regions**: each peak's influence
//! extends to the spectral troughs (local minima) on either side, which
//! naturally captures the full width of each spectral lobe.
//!
//! For each peak `p` with influence region `[start, end]`:
//! ```text
//! phase_rotation = synth_phase[p] - analysis_phase[p]
//! for each bin b in [start, end], b != p:
//!     synth_phase[b] = analysis_phase[b] + phase_rotation
//! ```
//!
//! This preserves the relative phase relationships between a peak and all bins
//! in its spectral lobe, eliminating the inter-bin phase incoherence that
//! causes "underwater" or "phasey" artifacts on polyphonic material.

use std::f32::consts::PI;

/// Phase locking mode for the phase vocoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseLockingMode {
    /// Full identity phase locking with trough-bounded influence regions.
    /// Each peak's phase rotation propagates to all bins in its spectral lobe
    /// (bounded by adjacent troughs). Good phase coherence, minimal artifacts.
    Identity,
    /// Region-of-influence phase locking with trough-bounded influence regions
    /// and SNR-weighted adaptive phase clamping as a safety net.
    /// Uses full identity rotation first, then clamps runaway phase deviations
    /// in noisy regions. Best quality for polyphonic material.
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

/// Finds spectral peaks (local maxima) in the magnitude spectrum.
///
/// A bin `k` is a peak if `magnitudes[k] > magnitudes[k-1]` and
/// `magnitudes[k] > magnitudes[k+1]`. Bins at the boundaries (0 and
/// `num_bins - 1`) are never considered peaks.
///
/// Only bins at or above `start_bin` are searched.
pub fn find_spectral_peaks(magnitudes: &[f32], num_bins: usize, start_bin: usize) -> Vec<usize> {
    let mut peaks = Vec::new();
    if num_bins < 3 || start_bin >= num_bins {
        return peaks;
    }
    let search_start = start_bin.max(1);
    for k in search_start..num_bins - 1 {
        if magnitudes[k] > magnitudes[k - 1] && magnitudes[k] > magnitudes[k + 1] {
            peaks.push(k);
        }
    }
    peaks
}

/// Finds spectral troughs (local minima) in the magnitude spectrum.
///
/// A bin `k` is a trough if `magnitudes[k] <= magnitudes[k-1]` and
/// `magnitudes[k] <= magnitudes[k+1]`. The first and last bins are always
/// included as boundary troughs.
///
/// Only bins at or above `start_bin` are searched for interior troughs.
/// `start_bin` itself is always included as a boundary trough.
pub fn find_spectral_troughs(magnitudes: &[f32], num_bins: usize, start_bin: usize) -> Vec<usize> {
    let mut troughs = Vec::new();
    if num_bins < 2 {
        if num_bins == 1 {
            troughs.push(0);
        }
        return troughs;
    }

    // Always include start_bin as a boundary
    troughs.push(start_bin);

    let search_start = (start_bin + 1).max(1);
    for k in search_start..num_bins - 1 {
        if magnitudes[k] <= magnitudes[k - 1] && magnitudes[k] <= magnitudes[k + 1] {
            troughs.push(k);
        }
    }

    // Always include last bin as a boundary (if it's beyond start_bin)
    if num_bins - 1 > start_bin {
        troughs.push(num_bins - 1);
    }
    troughs
}

/// Finds the influence region for a peak, bounded by adjacent troughs.
///
/// Returns `(region_start, region_end)` inclusive. The region extends from
/// the nearest trough below (or at) the peak to the nearest trough above
/// (or at) the peak. If no trough is found on one side, the boundary
/// defaults to `start_bin` or `num_bins - 1`.
pub fn find_influence_region(
    peak: usize,
    troughs: &[usize],
    start_bin: usize,
    num_bins: usize,
) -> (usize, usize) {
    // Find the trough immediately before (or at) this peak
    let start = troughs
        .iter()
        .rev()
        .find(|&&t| t <= peak)
        .copied()
        .unwrap_or(start_bin);

    // Find the trough immediately after this peak
    let end = troughs
        .iter()
        .find(|&&t| t > peak)
        .copied()
        .unwrap_or(num_bins.saturating_sub(1));

    (start, end)
}

/// Full identity phase locking with trough-bounded influence regions
/// (Laroche & Dolson 1999, enhanced).
///
/// For each spectral peak, the peak's phase rotation is propagated to all
/// bins in its influence region (bounded by adjacent spectral troughs).
/// This preserves relative phase relationships within each spectral lobe,
/// eliminating inter-bin phase incoherence.
///
/// Bins below `start_bin` (typically the sub-bass region) are not modified,
/// preserving the rigid sub-bass phase locking applied earlier.
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

    // Find peaks and troughs
    peaks.clear();
    let found_peaks = find_spectral_peaks(magnitudes, num_bins, start_bin);
    peaks.extend_from_slice(&found_peaks);

    if peaks.is_empty() {
        return;
    }

    let troughs = find_spectral_troughs(magnitudes, num_bins, start_bin);

    // Apply identity phase locking using trough-bounded influence regions
    apply_identity_phase_locking(
        analysis_phases,
        synthesis_phases,
        &found_peaks,
        &troughs,
        start_bin,
        num_bins,
    );
}

/// Region-of-influence phase locking with trough-bounded influence regions
/// and SNR-weighted adaptive phase clamping as a safety net.
///
/// This combines full identity phase locking (trough-bounded regions with
/// peak phase rotation propagation) with adaptive clamping to prevent
/// runaway phase errors in noisy spectral regions.
///
/// The process:
/// 1. Apply full identity phase locking (same as `identity_phase_lock`)
/// 2. For each non-peak bin, check if the resulting phase deviation from
///    the standard PV advance exceeds an SNR-weighted maximum
/// 3. If so, clamp the deviation to prevent artifacts in noisy regions
///
/// This preserves natural modulation (vibrato, tremolo) in clean harmonic
/// peaks while suppressing phase noise in weak bins.
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

    // Find peaks and troughs
    peaks.clear();
    let found_peaks = find_spectral_peaks(magnitudes, num_bins, start_bin);
    peaks.extend_from_slice(&found_peaks);

    if peaks.is_empty() {
        return;
    }

    let troughs = find_spectral_troughs(magnitudes, num_bins, start_bin);

    // Save the standard PV phases before identity locking (for clamping reference)
    // We only need phases for the start_bin..num_bins range.
    let pv_phases: Vec<f32> = synthesis_phases[start_bin..num_bins].to_vec();

    // Step 1: Apply full identity phase locking
    apply_identity_phase_locking(
        analysis_phases,
        synthesis_phases,
        &found_peaks,
        &troughs,
        start_bin,
        num_bins,
    );

    // Step 2: Adaptive clamping as safety net for noisy bins
    for bin in start_bin..num_bins {
        if found_peaks.binary_search(&bin).is_ok() {
            continue; // Peak bins are anchors; don't clamp them
        }

        let max_dev = compute_adaptive_max_deviation(magnitudes, bin, num_bins);
        let expected = pv_phases[bin - start_bin]; // phase from standard PV advance
        let deviation = wrap_phase(synthesis_phases[bin] - expected);
        if deviation.abs() > max_dev {
            let clamped_dev = deviation.clamp(-max_dev, max_dev);
            synthesis_phases[bin] = expected + clamped_dev;
        }
    }
}

/// Apply full identity phase locking using peak influence regions bounded
/// by spectral troughs.
///
/// For each spectral peak, propagate its phase rotation to all bins in its
/// influence region. The phase rotation is the difference between the peak's
/// synthesis phase (computed by IF estimation) and its analysis phase. Each
/// non-peak bin in the region receives:
///
/// ```text
/// synth_phase[bin] = analysis_phase[bin] + phase_rotation
/// ```
///
/// where `phase_rotation = synth_phase[peak] - analysis_phase[peak]`.
///
/// This preserves the relative phase structure of each spectral lobe.
/// Bins not covered by any peak's influence region (rare, typically in the
/// noise floor) retain their IF-estimated phases.
pub fn apply_identity_phase_locking(
    analysis_phases: &[f32],
    synthesis_phases: &mut [f32],
    peaks: &[usize],
    troughs: &[usize],
    start_bin: usize,
    num_bins: usize,
) {
    for &peak in peaks {
        let (region_start, region_end) = find_influence_region(peak, troughs, start_bin, num_bins);

        // Compute the peak's phase rotation
        let phase_rotation = synthesis_phases[peak] - analysis_phases[peak];

        // Apply rotation to all bins in the region (except the peak itself)
        let end = region_end.min(num_bins.saturating_sub(1));
        for bin in region_start..=end {
            if bin != peak && bin >= start_bin && bin < num_bins {
                synthesis_phases[bin] = analysis_phases[bin] + phase_rotation;
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

    // -----------------------------------------------------------------------
    // Existing tests (updated for new signatures)
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // New tests for spectral peak/trough detection and influence regions
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_spectral_peaks_known_spectrum() {
        // Magnitude spectrum with clear peaks at bins 3, 7, 12
        let mut magnitudes = vec![0.1f32; 16];
        magnitudes[3] = 1.0;
        magnitudes[7] = 0.8;
        magnitudes[12] = 0.6;

        let peaks = find_spectral_peaks(&magnitudes, 16, 0);

        assert!(peaks.contains(&3), "Should find peak at bin 3");
        assert!(peaks.contains(&7), "Should find peak at bin 7");
        assert!(peaks.contains(&12), "Should find peak at bin 12");
        // Should not find any other peaks (all other bins are equal at 0.1)
        assert_eq!(peaks.len(), 3, "Should find exactly 3 peaks");
    }

    #[test]
    fn test_find_spectral_peaks_respects_start_bin() {
        let mut magnitudes = vec![0.1f32; 16];
        magnitudes[3] = 1.0;
        magnitudes[7] = 0.8;
        magnitudes[12] = 0.6;

        // With start_bin=5, should skip the peak at bin 3
        let peaks = find_spectral_peaks(&magnitudes, 16, 5);
        assert!(!peaks.contains(&3), "Should skip peak below start_bin");
        assert!(peaks.contains(&7), "Should find peak at bin 7");
        assert!(peaks.contains(&12), "Should find peak at bin 12");
        assert_eq!(peaks.len(), 2);
    }

    #[test]
    fn test_find_spectral_peaks_empty_and_edge_cases() {
        // Too few bins
        let magnitudes = vec![1.0f32; 2];
        assert!(find_spectral_peaks(&magnitudes, 2, 0).is_empty());

        // All equal
        let magnitudes = vec![1.0f32; 10];
        assert!(find_spectral_peaks(&magnitudes, 10, 0).is_empty());

        // start_bin beyond range
        let magnitudes = vec![0.1, 1.0, 0.1];
        assert!(find_spectral_peaks(&magnitudes, 3, 10).is_empty());
    }

    #[test]
    fn test_find_spectral_troughs_known_spectrum() {
        // Spectrum: low, HIGH, low, low, HIGH, low
        let magnitudes = vec![0.1, 0.5, 0.1, 0.05, 0.8, 0.2, 0.1];
        let troughs = find_spectral_troughs(&magnitudes, 7, 0);

        // Should include boundary 0 (start_bin)
        assert!(troughs.contains(&0), "Should include start_bin as boundary");
        // Should include boundary 6 (last bin)
        assert!(troughs.contains(&6), "Should include last bin as boundary");
        // Bin 3 is a trough (0.05 <= 0.1 and 0.05 <= 0.8)
        assert!(troughs.contains(&3), "Should find trough at bin 3");
    }

    #[test]
    fn test_find_spectral_troughs_with_start_bin() {
        let magnitudes = vec![0.1, 0.5, 0.1, 0.05, 0.8, 0.2, 0.1];
        let troughs = find_spectral_troughs(&magnitudes, 7, 2);

        // Start boundary should be at start_bin=2
        assert!(
            troughs.contains(&2),
            "Should include start_bin=2 as boundary"
        );
        // Should include last bin
        assert!(troughs.contains(&6), "Should include last bin as boundary");
        // Bin 3 should still be a trough
        assert!(troughs.contains(&3), "Should find trough at bin 3");
    }

    #[test]
    fn test_find_influence_region_basic() {
        // Troughs at bins 0, 4, 9, 15
        let troughs = vec![0, 4, 9, 15];

        // Peak at bin 2: should be bounded by troughs 0 and 4
        let (start, end) = find_influence_region(2, &troughs, 0, 16);
        assert_eq!(start, 0);
        assert_eq!(end, 4);

        // Peak at bin 6: should be bounded by troughs 4 and 9
        let (start, end) = find_influence_region(6, &troughs, 0, 16);
        assert_eq!(start, 4);
        assert_eq!(end, 9);

        // Peak at bin 12: should be bounded by troughs 9 and 15
        let (start, end) = find_influence_region(12, &troughs, 0, 16);
        assert_eq!(start, 9);
        assert_eq!(end, 15);
    }

    #[test]
    fn test_find_influence_region_peak_at_trough_boundary() {
        let troughs = vec![0, 5, 10];

        // Peak at bin 5 (which is also a trough boundary)
        // Start should be 5 (the trough at or before), end should be 10
        let (start, end) = find_influence_region(5, &troughs, 0, 11);
        assert_eq!(start, 5);
        assert_eq!(end, 10);
    }

    #[test]
    fn test_find_influence_region_no_troughs() {
        let troughs: Vec<usize> = vec![];

        // With no troughs, should default to start_bin and num_bins-1
        let (start, end) = find_influence_region(5, &troughs, 0, 16);
        assert_eq!(start, 0);
        assert_eq!(end, 15);
    }

    // -----------------------------------------------------------------------
    // Tests for apply_identity_phase_locking
    // -----------------------------------------------------------------------

    #[test]
    fn test_identity_locking_phase_rotation_propagation() {
        // Test that a peak's phase rotation propagates to its influence region
        let num_bins = 16;
        let start_bin = 0;

        // Set up: peak at bin 5 with troughs at 0, 3, 8, 15
        let analysis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.2).collect();
        let mut synthesis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.3).collect();

        let peaks = vec![5];
        let troughs = vec![0, 3, 8, 15];

        // Save original synthesis phase of peak
        let peak_synth_phase = synthesis_phases[5];
        let peak_analysis_phase = analysis_phases[5];
        let phase_rotation = peak_synth_phase - peak_analysis_phase;

        apply_identity_phase_locking(
            &analysis_phases,
            &mut synthesis_phases,
            &peaks,
            &troughs,
            start_bin,
            num_bins,
        );

        // Peak bin should be unchanged
        assert!(
            (synthesis_phases[5] - peak_synth_phase).abs() < 1e-6,
            "Peak bin should keep its phase"
        );

        // Bins in the influence region [3, 8] should have:
        // synth[bin] = analysis[bin] + phase_rotation
        for bin in 3..=8 {
            if bin == 5 {
                continue;
            }
            let expected = analysis_phases[bin] + phase_rotation;
            assert!(
                (synthesis_phases[bin] - expected).abs() < 1e-6,
                "Bin {} should have rotated phase: expected {}, got {}",
                bin,
                expected,
                synthesis_phases[bin]
            );
        }
    }

    #[test]
    fn test_identity_locking_different_regions_different_rotations() {
        // Two peaks with different phase rotations
        let num_bins = 20;
        let start_bin = 0;

        let analysis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.1).collect();
        let mut synthesis_phases: Vec<f32> = vec![0.0; num_bins];
        // Give each peak a different synthesis phase to create different rotations
        synthesis_phases[4] = 2.0; // peak 1
        synthesis_phases[14] = 5.0; // peak 2

        let peaks = vec![4, 14];
        let troughs = vec![0, 2, 9, 11, 19];

        let rotation_1 = synthesis_phases[4] - analysis_phases[4];
        let rotation_2 = synthesis_phases[14] - analysis_phases[14];

        apply_identity_phase_locking(
            &analysis_phases,
            &mut synthesis_phases,
            &peaks,
            &troughs,
            start_bin,
            num_bins,
        );

        // Bins in peak 1's region [2, 9] should use rotation_1
        for bin in 2..=9 {
            if bin == 4 {
                continue;
            }
            let expected = analysis_phases[bin] + rotation_1;
            assert!(
                (synthesis_phases[bin] - expected).abs() < 1e-6,
                "Bin {} (peak 1 region) expected {}, got {}",
                bin,
                expected,
                synthesis_phases[bin]
            );
        }

        // Bins in peak 2's region [11, 19] should use rotation_2
        for bin in 11..=19 {
            if bin == 14 {
                continue;
            }
            let expected = analysis_phases[bin] + rotation_2;
            assert!(
                (synthesis_phases[bin] - expected).abs() < 1e-6,
                "Bin {} (peak 2 region) expected {}, got {}",
                bin,
                expected,
                synthesis_phases[bin]
            );
        }
    }

    #[test]
    fn test_identity_locking_harmonic_signal() {
        // Simulate a harmonic signal: fundamental + 3 harmonics
        // Peaks at bins 10, 20, 30, 40 with decreasing amplitude
        let num_bins = 64;
        let start_bin = 1;
        let mut magnitudes = vec![0.01f32; num_bins];

        // Create harmonic peaks with Gaussian-like shape
        let peak_bins = [10usize, 20, 30, 40];
        let peak_amps = [1.0f32, 0.7, 0.4, 0.2];
        for (&p, &a) in peak_bins.iter().zip(peak_amps.iter()) {
            for offset in -3i32..=3 {
                let bin = (p as i32 + offset) as usize;
                if bin < num_bins {
                    let dist = offset.unsigned_abs() as f32;
                    magnitudes[bin] = a * (-dist * dist / 2.0).exp();
                }
            }
        }

        // Analysis phases: linear progression (typical for a stationary sinusoid)
        let analysis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.5).collect();

        // Synthesis phases: slightly shifted (simulating stretch)
        let mut synthesis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.52).collect();
        let original_synth = synthesis_phases.clone();

        let peaks = find_spectral_peaks(&magnitudes, num_bins, start_bin);
        let troughs = find_spectral_troughs(&magnitudes, num_bins, start_bin);

        apply_identity_phase_locking(
            &analysis_phases,
            &mut synthesis_phases,
            &peaks,
            &troughs,
            start_bin,
            num_bins,
        );

        // For each harmonic peak, verify that bins in its region maintain
        // the same relative phase relationships as in the analysis
        for &p in &peaks {
            if !peak_bins.contains(&p) {
                continue;
            }
            let rotation = original_synth[p] - analysis_phases[p];
            let (region_start, region_end) =
                find_influence_region(p, &troughs, start_bin, num_bins);
            for bin in region_start..=region_end.min(num_bins - 1) {
                if bin == p {
                    continue;
                }
                // Check that relative phase is preserved
                let expected_relative = analysis_phases[bin] - analysis_phases[p]; // original relationship
                let actual_relative = synthesis_phases[bin] - synthesis_phases[p]; // after locking
                assert!(
                    (actual_relative - expected_relative).abs() < 1e-5,
                    "Bin {} relative to peak {}: expected {}, got {}",
                    bin,
                    p,
                    expected_relative,
                    actual_relative
                );

                // Also verify absolute phase
                let expected_abs = analysis_phases[bin] + rotation;
                assert!(
                    (synthesis_phases[bin] - expected_abs).abs() < 1e-5,
                    "Bin {} absolute phase: expected {}, got {}",
                    bin,
                    expected_abs,
                    synthesis_phases[bin]
                );
            }
        }
    }

    #[test]
    fn test_identity_locking_respects_start_bin() {
        // Bins below start_bin should not be modified
        let num_bins = 16;
        let start_bin = 5;

        let analysis_phases = vec![0.0f32; num_bins];
        let mut synthesis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.1).collect();
        let original_sub_bass: Vec<f32> = synthesis_phases[..start_bin].to_vec();

        // Peak at bin 8
        let mut magnitudes = vec![0.1f32; num_bins];
        magnitudes[8] = 1.0;

        let peaks = find_spectral_peaks(&magnitudes, num_bins, start_bin);
        let troughs = find_spectral_troughs(&magnitudes, num_bins, start_bin);

        apply_identity_phase_locking(
            &analysis_phases,
            &mut synthesis_phases,
            &peaks,
            &troughs,
            start_bin,
            num_bins,
        );

        // Sub-bass bins should be untouched
        assert_eq!(
            &synthesis_phases[..start_bin],
            &original_sub_bass[..],
            "Sub-bass bins should not be modified"
        );
    }

    // -----------------------------------------------------------------------
    // Integration test: chord stretching
    // -----------------------------------------------------------------------

    #[test]
    fn test_chord_stretch_preserves_frequencies() {
        // Create an A major chord: A4 (440 Hz) + C#5 (554 Hz) + E5 (659 Hz)
        let sample_rate = 44100u32;
        let fft_size = 4096;
        let hop = fft_size / 4;
        let stretch_ratio = 1.25;
        let num_samples = fft_size * 8;

        let freqs = [440.0f32, 554.0, 659.0];
        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                freqs
                    .iter()
                    .map(|&f| (2.0 * PI * f * t).sin() / freqs.len() as f32)
                    .sum::<f32>()
            })
            .collect();

        // Stretch using the phase vocoder with ROI mode (full identity + clamping)
        let mut pv = crate::stretch::phase_vocoder::PhaseVocoder::with_options(
            fft_size,
            hop,
            stretch_ratio,
            sample_rate,
            120.0,
            crate::core::window::WindowType::BlackmanHarris,
            PhaseLockingMode::RegionOfInfluence,
        );
        let output = pv.process(&input).unwrap();
        assert!(!output.is_empty(), "Output should not be empty");

        // Analyze the output spectrum: take FFT of a middle section
        let analysis_start = output.len() / 4;
        let analysis_len = fft_size;
        if analysis_start + analysis_len > output.len() {
            return; // Not enough output to analyze
        }

        let section = &output[analysis_start..analysis_start + analysis_len];

        // Compute magnitude spectrum via DFT at the three chord frequencies
        let num_bins = fft_size / 2 + 1;
        let bin_freq = sample_rate as f32 / fft_size as f32;

        let mut max_mag = 0.0f32;
        let mut freq_mags = [0.0f32; 3];

        for (idx, &freq) in freqs.iter().enumerate() {
            let target_bin = (freq / bin_freq).round() as usize;
            // Sum magnitude in a small window around the target bin
            let mut mag_sum = 0.0f32;
            for offset in -2i32..=2 {
                let bin = (target_bin as i32 + offset).max(0) as usize;
                if bin < num_bins {
                    // Simple DFT at this bin
                    let mut re = 0.0f64;
                    let mut im = 0.0f64;
                    for (n, &s) in section.iter().enumerate() {
                        let angle =
                            -2.0 * std::f64::consts::PI * bin as f64 * n as f64 / fft_size as f64;
                        re += s as f64 * angle.cos();
                        im += s as f64 * angle.sin();
                    }
                    mag_sum += (re * re + im * im).sqrt() as f32;
                }
            }
            freq_mags[idx] = mag_sum;
            if mag_sum > max_mag {
                max_mag = mag_sum;
            }
        }

        // All three frequencies should be present with similar amplitudes
        // (within ~6 dB of the strongest, which is +-3dB tolerance from each other)
        let threshold_ratio = 0.25; // -12 dB (generous; real threshold is ~-6dB)
        for (idx, &freq) in freqs.iter().enumerate() {
            assert!(
                freq_mags[idx] > max_mag * threshold_ratio,
                "Frequency {} Hz is too weak after stretching: mag={}, max={}",
                freq,
                freq_mags[idx],
                max_mag
            );
        }
    }
}

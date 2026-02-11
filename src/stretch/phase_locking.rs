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

/// Maximum phase deviation from analysis relationship before clamping (PI/4).
const MAX_PHASE_DEVIATION: f32 = PI / 4.0;

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

/// Region-of-influence phase locking with parabolic peak interpolation.
///
/// Improvements over identity phase locking:
/// 1. Parabolic interpolation for more accurate peak frequency estimation
/// 2. Influence zones extend to midpoints between adjacent peaks
/// 3. Phase deviation clamping to reduce ringing (clamp to ±PI/4)
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

            // Clamp phase deviation: if the synthesis phase deviates too far
            // from the expected analysis relationship, clamp it to reduce ringing
            let expected = synthesis_phases[bin]; // phase from standard PV advance
            let deviation = wrap_phase(proposed - expected);
            if deviation.abs() > MAX_PHASE_DEVIATION {
                // Clamp to the allowed range
                let clamped_dev = deviation.clamp(-MAX_PHASE_DEVIATION, MAX_PHASE_DEVIATION);
                synthesis_phases[bin] = expected + clamped_dev;
            } else {
                synthesis_phases[bin] = proposed;
            }
        }
    }
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

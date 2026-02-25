//! Beat detection and grid snapping for 4/4 EDM patterns.
//!
//! Uses a phase-locked loop (PLL) to refine the beat grid after initial
//! estimation, allowing the grid to self-correct over time and eliminate
//! systematic offset from the first detected onset.

use crate::analysis::transient::detect_transients;

/// FFT size for beat detection (balances frequency resolution and speed).
const BEAT_FFT_SIZE: usize = 2048;
/// Hop size for beat detection analysis frames.
const BEAT_HOP_SIZE: usize = 512;
/// Transient sensitivity for kick detection (lower = fewer false positives).
const BEAT_SENSITIVITY: f32 = 0.4;
/// Minimum EDM BPM for octave normalization.
const MIN_EDM_BPM: f64 = 100.0;
/// Maximum EDM BPM for octave normalization.
const MAX_EDM_BPM: f64 = 160.0;

/// PLL interval correction gain. Controls how quickly the beat interval
/// adapts to phase errors from detected onsets. Higher values adapt faster
/// but may be less stable.
const PLL_ALPHA: f64 = 0.08;
/// PLL phase (offset) correction gain. Controls how quickly the grid offset
/// adjusts to align with detected onsets.
const PLL_BETA: f64 = 0.03;

/// Beat grid information for a 4/4 track.
#[derive(Debug, Clone)]
pub struct BeatGrid {
    /// Sample positions of detected beat onsets (integer, for backward
    /// compatibility with existing code that indexes into sample buffers).
    pub beats: Vec<usize>,
    /// Fractional-sample beat positions for sub-sample precision.
    /// Length matches `beats`; each value is a refinement of the
    /// corresponding integer beat position.
    pub beats_fractional: Vec<f64>,
    /// Estimated BPM.
    pub bpm: f64,
    /// Sample rate.
    pub sample_rate: u32,
}

impl BeatGrid {
    /// Returns the interval between beats in samples.
    #[inline]
    pub fn beat_interval_samples(&self) -> f64 {
        60.0 * self.sample_rate as f64 / self.bpm
    }

    /// Snaps a sample position to the nearest beat grid position.
    #[inline]
    pub fn snap_to_grid(&self, position: usize) -> usize {
        if self.beats.is_empty() {
            return position;
        }
        let mut closest = self.beats[0];
        let mut min_dist = (position as i64 - closest as i64).unsigned_abs() as usize;
        for &beat in &self.beats[1..] {
            let dist = (position as i64 - beat as i64).unsigned_abs() as usize;
            if dist < min_dist {
                min_dist = dist;
                closest = beat;
            }
        }
        closest
    }

    /// Snaps a sample position to the nearest fractional beat grid position.
    /// Returns a sub-sample-accurate position.
    #[inline]
    pub fn snap_to_grid_fractional(&self, position: f64) -> f64 {
        if self.beats_fractional.is_empty() {
            return position;
        }
        let mut closest = self.beats_fractional[0];
        let mut min_dist = (position - closest).abs();
        for &beat in &self.beats_fractional[1..] {
            let dist = (position - beat).abs();
            if dist < min_dist {
                min_dist = dist;
                closest = beat;
            }
        }
        closest
    }
}

/// Detects beats in a mono audio signal and estimates BPM.
///
/// Optimized for 4/4 EDM (house/techno) with expected BPM range 100-160.
/// Uses a PLL-based beat grid correction to eliminate systematic offset
/// from the initial onset detection.
pub fn detect_beats(samples: &[f32], sample_rate: u32) -> BeatGrid {
    let transients = detect_transients(
        samples,
        sample_rate,
        BEAT_FFT_SIZE,
        BEAT_HOP_SIZE,
        BEAT_SENSITIVITY,
    );

    if transients.onsets.len() < 2 {
        let beats_fractional = transients.onsets_fractional.to_vec();
        let beats_int = if beats_fractional.is_empty() {
            transients.onsets.clone()
        } else {
            beats_fractional
                .iter()
                .map(|&f| f.round() as usize)
                .collect()
        };
        return BeatGrid {
            beats: beats_int,
            beats_fractional,
            bpm: 0.0,
            sample_rate,
        };
    }

    // Use fractional onsets for better precision when available
    let onset_positions_f64: Vec<f64> =
        if transients.onsets_fractional.len() == transients.onsets.len() {
            transients.onsets_fractional.clone()
        } else {
            transients.onsets.iter().map(|&o| o as f64).collect()
        };

    // Compute inter-onset intervals from fractional positions
    let intervals_f64: Vec<f64> = onset_positions_f64
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect();

    // Also compute integer intervals for BPM estimation (backward compatible)
    let intervals: Vec<usize> = transients.onsets.windows(2).map(|w| w[1] - w[0]).collect();

    // Estimate BPM from median interval (robust to outliers)
    let bpm = estimate_bpm_from_intervals(&intervals, sample_rate);

    // Initial beat interval estimate
    let initial_interval = 60.0 * sample_rate as f64 / bpm;

    // Build beat grid using PLL-corrected quantization
    let beats_fractional =
        quantize_to_grid_pll(&onset_positions_f64, &intervals_f64, initial_interval);

    // Derive integer positions from fractional
    let beats: Vec<usize> = beats_fractional
        .iter()
        .map(|&f| f.round() as usize)
        .collect();

    BeatGrid {
        beats,
        beats_fractional,
        bpm,
        sample_rate,
    }
}

/// Estimates BPM from inter-onset intervals.
fn estimate_bpm_from_intervals(intervals: &[usize], sample_rate: u32) -> f64 {
    if intervals.is_empty() {
        return 0.0;
    }

    let mut sorted = intervals.to_vec();
    sorted.sort();
    let median_interval = sorted[sorted.len() / 2];

    let raw_bpm = 60.0 * sample_rate as f64 / median_interval as f64;

    // Snap to reasonable EDM BPM range
    let mut bpm = raw_bpm;
    while bpm > MAX_EDM_BPM {
        bpm /= 2.0;
    }
    while bpm < MIN_EDM_BPM && bpm > 0.0 {
        bpm *= 2.0;
    }

    bpm
}

/// Quantizes onset positions to a regular grid (simple, backward-compatible).
///
/// Kept for test backward compatibility and as a reference implementation.
/// For PLL-corrected grids, use [`quantize_to_grid_pll`].
#[cfg(test)]
fn quantize_to_grid(onsets: &[usize], beat_interval: usize) -> Vec<usize> {
    if onsets.is_empty() || beat_interval == 0 {
        return onsets.to_vec();
    }

    let first = onsets[0];
    let last = *onsets.last().unwrap_or(&first);

    let mut grid = Vec::new();
    let mut pos = first;
    while pos <= last + beat_interval / 2 {
        grid.push(pos);
        pos += beat_interval;
    }

    grid
}

/// Quantizes onset positions to a regular grid using a phase-locked loop (PLL)
/// for self-correcting beat alignment.
///
/// The PLL iterates through detected onsets and adjusts both the beat interval
/// and grid phase (offset) based on the phase error between each detected onset
/// and its nearest grid position. This allows the grid to:
/// - Recover from an inaccurate first onset position
/// - Track gradual tempo drift within a track
/// - Converge to the true beat positions over multiple beats
///
/// # Parameters
/// - `onsets`: Fractional-sample onset positions from transient detection
/// - `_intervals`: Fractional inter-onset intervals (reserved for future use)
/// - `initial_interval`: Initial beat interval estimate from BPM detection
///
/// # Returns
/// Fractional-sample beat grid positions.
fn quantize_to_grid_pll(onsets: &[f64], _intervals: &[f64], initial_interval: f64) -> Vec<f64> {
    if onsets.is_empty() || initial_interval <= 0.0 {
        return onsets.to_vec();
    }

    let first = onsets[0];
    let last = *onsets.last().unwrap_or(&first);

    // Phase 1: Run PLL to refine interval and offset
    let mut beat_interval = initial_interval;
    let mut grid_offset: f64 = 0.0; // cumulative phase correction

    // Iterate through detected onsets and compute corrections
    for &onset in onsets.iter().skip(1) {
        // Find nearest grid position for this onset
        let relative = onset - first - grid_offset;
        if beat_interval <= 0.0 {
            break;
        }
        let nearest_grid_idx = (relative / beat_interval).round();
        let nearest_grid_pos = first + grid_offset + nearest_grid_idx * beat_interval;

        // Phase error: how far is the detected onset from the nearest grid position
        let phase_error = onset - nearest_grid_pos;

        // Apply PLL corrections
        beat_interval += PLL_ALPHA * phase_error / nearest_grid_idx.max(1.0);
        grid_offset += PLL_BETA * phase_error;

        // Ensure interval stays positive and reasonable (within 50% of initial)
        beat_interval = beat_interval.clamp(initial_interval * 0.5, initial_interval * 1.5);
    }

    // Phase 2: Generate corrected grid using refined interval and offset
    let mut grid = Vec::new();
    let mut pos = first + grid_offset;

    // Ensure grid starts at or before the first onset
    while pos > first + beat_interval * 0.5 {
        pos -= beat_interval;
    }
    // Don't start before sample 0
    if pos < 0.0 {
        pos += beat_interval * (((-pos) / beat_interval).ceil());
    }

    let end = last + beat_interval * 0.5;
    while pos <= end {
        grid.push(pos);
        pos += beat_interval;
    }

    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a BeatGrid with matching fractional positions.
    fn make_grid(beats: Vec<usize>, bpm: f64, sample_rate: u32) -> BeatGrid {
        let beats_fractional = beats.iter().map(|&b| b as f64).collect();
        BeatGrid {
            beats,
            beats_fractional,
            bpm,
            sample_rate,
        }
    }

    #[test]
    fn test_beat_grid_snap() {
        let grid = make_grid(vec![0, 22050, 44100, 66150], 120.0, 44100);
        assert_eq!(grid.snap_to_grid(100), 0);
        assert_eq!(grid.snap_to_grid(22000), 22050);
        assert_eq!(grid.snap_to_grid(33000), 22050);
    }

    #[test]
    fn test_estimate_bpm() {
        // 120 BPM at 44100 Hz = 22050 samples per beat
        let intervals = vec![22050, 22050, 22050, 22050];
        let bpm = estimate_bpm_from_intervals(&intervals, 44100);
        assert!((bpm - 120.0).abs() < 1.0, "Expected ~120 BPM, got {}", bpm);
    }

    #[test]
    fn test_estimate_bpm_empty() {
        assert_eq!(estimate_bpm_from_intervals(&[], 44100), 0.0);
    }

    #[test]
    fn test_quantize_grid() {
        let onsets = vec![0, 22000, 44200];
        let grid = quantize_to_grid(&onsets, 22050);
        assert_eq!(grid.len(), 3);
        assert_eq!(grid[0], 0);
        assert_eq!(grid[1], 22050);
        assert_eq!(grid[2], 44100);
    }

    // --- beat_interval_samples ---

    #[test]
    fn test_beat_interval_samples_120bpm() {
        let grid = make_grid(vec![0], 120.0, 44100);
        // 60 * 44100 / 120 = 22050
        assert!((grid.beat_interval_samples() - 22050.0).abs() < 1.0);
    }

    #[test]
    fn test_beat_interval_samples_128bpm_48khz() {
        let grid = make_grid(vec![0], 128.0, 48000);
        // 60 * 48000 / 128 = 22500
        assert!((grid.beat_interval_samples() - 22500.0).abs() < 1.0);
    }

    // --- snap_to_grid edge cases ---

    #[test]
    fn test_snap_to_grid_empty_beats() {
        let grid = make_grid(vec![], 120.0, 44100);
        // Empty grid -> return position unchanged
        assert_eq!(grid.snap_to_grid(1000), 1000);
    }

    #[test]
    fn test_snap_to_grid_before_first_beat() {
        let grid = make_grid(vec![1000, 2000, 3000], 120.0, 44100);
        // Position before any beat -> snaps to first beat
        assert_eq!(grid.snap_to_grid(500), 1000);
    }

    #[test]
    fn test_snap_to_grid_after_last_beat() {
        let grid = make_grid(vec![1000, 2000, 3000], 120.0, 44100);
        // Position after all beats -> snaps to last beat
        assert_eq!(grid.snap_to_grid(10000), 3000);
    }

    #[test]
    fn test_snap_to_grid_equidistant() {
        let grid = make_grid(vec![0, 100, 200], 120.0, 44100);
        // Position exactly between beats 0 and 100 -> snaps to first one found
        // (50 is equidistant from 0 and 100; algorithm picks first as min_dist tie)
        let result = grid.snap_to_grid(50);
        assert!(
            result == 0 || result == 100,
            "Should snap to 0 or 100, got {}",
            result
        );
    }

    #[test]
    fn test_snap_to_grid_exact_beat() {
        let grid = make_grid(vec![0, 22050, 44100], 120.0, 44100);
        assert_eq!(grid.snap_to_grid(22050), 22050);
    }

    // --- estimate_bpm_from_intervals ---

    #[test]
    fn test_estimate_bpm_halving_high_bpm() {
        // Raw BPM 320 -> should halve to 160 (within MAX_EDM_BPM)
        // 320 BPM at 44100 Hz = 60*44100/320 = 8268.75 samples
        let intervals = vec![8269, 8269, 8269];
        let bpm = estimate_bpm_from_intervals(&intervals, 44100);
        assert!(
            (bpm - 160.0).abs() < 2.0,
            "320 BPM should halve to ~160, got {}",
            bpm
        );
    }

    #[test]
    fn test_estimate_bpm_doubling_low_bpm() {
        // Raw BPM 50 -> should double to 100 (at MIN_EDM_BPM)
        // 50 BPM at 44100 Hz = 60*44100/50 = 52920 samples
        let intervals = vec![52920, 52920, 52920];
        let bpm = estimate_bpm_from_intervals(&intervals, 44100);
        assert!(
            (bpm - 100.0).abs() < 2.0,
            "50 BPM should double to ~100, got {}",
            bpm
        );
    }

    #[test]
    fn test_estimate_bpm_already_in_range() {
        // 128 BPM should stay as-is (within 100-160 range)
        let interval = (60.0 * 44100.0 / 128.0) as usize;
        let intervals = vec![interval, interval, interval];
        let bpm = estimate_bpm_from_intervals(&intervals, 44100);
        assert!(
            (bpm - 128.0).abs() < 2.0,
            "128 BPM should stay ~128, got {}",
            bpm
        );
    }

    #[test]
    fn test_estimate_bpm_outlier_robustness() {
        // Median should be robust to one outlier
        // 4 intervals at 120 BPM, 1 outlier
        let normal = (60.0 * 44100.0 / 120.0) as usize; // 22050
        let outlier = normal / 3; // very short interval
        let intervals = vec![normal, normal, outlier, normal, normal];
        let bpm = estimate_bpm_from_intervals(&intervals, 44100);
        // Median of sorted = [outlier, 22050, 22050, 22050, 22050] -> median = 22050
        assert!(
            (bpm - 120.0).abs() < 2.0,
            "BPM should be robust to outlier, got {}",
            bpm
        );
    }

    // --- quantize_to_grid edge cases ---

    #[test]
    fn test_quantize_to_grid_empty_onsets() {
        let result = quantize_to_grid(&[], 22050);
        assert!(result.is_empty());
    }

    #[test]
    fn test_quantize_to_grid_zero_interval() {
        // beat_interval == 0 -> return onsets as-is
        let onsets = vec![100, 200, 300];
        let result = quantize_to_grid(&onsets, 0);
        assert_eq!(result, onsets);
    }

    #[test]
    fn test_quantize_to_grid_single_onset() {
        let result = quantize_to_grid(&[5000], 22050);
        // Grid starts at 5000, extends by beat_interval
        assert_eq!(result, vec![5000]);
    }

    #[test]
    fn test_quantize_to_grid_extension() {
        // Grid should extend up to last + interval/2
        let onsets = vec![0, 44100];
        let result = quantize_to_grid(&onsets, 22050);
        // Grid: 0, 22050, 44100 (44100 <= 44100 + 22050/2 = 55125)
        assert_eq!(result, vec![0, 22050, 44100]);
    }

    // --- PLL quantization tests ---

    #[test]
    fn test_pll_grid_perfect_onsets() {
        // Perfect onsets at exact beat positions -> grid should match closely
        let interval = 22050.0; // 120 BPM at 44100 Hz
        let onsets: Vec<f64> = (0..8).map(|i| i as f64 * interval).collect();
        let intervals: Vec<f64> = onsets.windows(2).map(|w| w[1] - w[0]).collect();
        let grid = quantize_to_grid_pll(&onsets, &intervals, interval);

        assert_eq!(grid.len(), 8, "PLL grid should have 8 beats");
        for (i, &pos) in grid.iter().enumerate() {
            let expected = i as f64 * interval;
            assert!(
                (pos - expected).abs() < interval * 0.1,
                "Beat {} at {:.1} should be near {:.1}",
                i,
                pos,
                expected
            );
        }
    }

    #[test]
    fn test_pll_grid_offset_first_onset() {
        // First onset is 200 samples late, subsequent onsets are correct
        let interval = 22050.0;
        let offset = 200.0;
        let mut onsets: Vec<f64> = (0..8).map(|i| i as f64 * interval).collect();
        onsets[0] += offset; // shift first onset

        let intervals: Vec<f64> = onsets.windows(2).map(|w| w[1] - w[0]).collect();
        let grid = quantize_to_grid_pll(&onsets, &intervals, interval);

        // PLL should correct; later beats should be closer to true positions
        // than the initial offset would suggest
        assert!(
            grid.len() >= 7,
            "PLL grid should produce reasonable number of beats"
        );
    }

    #[test]
    fn test_pll_grid_empty() {
        let result = quantize_to_grid_pll(&[], &[], 22050.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_pll_grid_zero_interval() {
        let onsets = vec![100.0, 200.0, 300.0];
        let result = quantize_to_grid_pll(&onsets, &[], 0.0);
        assert_eq!(result, onsets);
    }

    #[test]
    fn test_pll_grid_single_onset() {
        let result = quantize_to_grid_pll(&[5000.0], &[], 22050.0);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 5000.0).abs() < 1.0);
    }

    // --- snap_to_grid_fractional ---

    #[test]
    fn test_snap_to_grid_fractional_basic() {
        let grid = BeatGrid {
            beats: vec![0, 22050, 44100],
            beats_fractional: vec![0.0, 22050.5, 44100.25],
            bpm: 120.0,
            sample_rate: 44100,
        };
        let snapped = grid.snap_to_grid_fractional(22000.0);
        assert!(
            (snapped - 22050.5).abs() < 1.0,
            "Should snap to 22050.5, got {}",
            snapped
        );
    }

    #[test]
    fn test_snap_to_grid_fractional_empty() {
        let grid = BeatGrid {
            beats: vec![],
            beats_fractional: vec![],
            bpm: 120.0,
            sample_rate: 44100,
        };
        let snapped = grid.snap_to_grid_fractional(1000.0);
        assert!((snapped - 1000.0).abs() < 1e-10);
    }

    // --- beats_fractional populated by detect_beats ---

    #[test]
    fn test_detect_beats_has_fractional() {
        // Generate a click train
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 4;
        let mut samples = vec![0.0f32; num_samples];
        let click_interval = (60.0 * sample_rate as f64 / 120.0) as usize; // 120 BPM
        for i in (0..num_samples).step_by(click_interval) {
            for j in 0..10.min(num_samples - i) {
                samples[i + j] = if j < 5 { 1.0 } else { -0.5 };
            }
        }

        let grid = detect_beats(&samples, sample_rate);
        assert_eq!(
            grid.beats.len(),
            grid.beats_fractional.len(),
            "beats and beats_fractional should have same length"
        );
    }
}

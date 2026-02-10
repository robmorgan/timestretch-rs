//! Beat detection and grid snapping for 4/4 EDM patterns.

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

/// Beat grid information for a 4/4 track.
#[derive(Debug, Clone)]
pub struct BeatGrid {
    /// Sample positions of detected beat onsets.
    pub beats: Vec<usize>,
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
}

/// Detects beats in a mono audio signal and estimates BPM.
///
/// Optimized for 4/4 EDM (house/techno) with expected BPM range 100-160.
pub fn detect_beats(samples: &[f32], sample_rate: u32) -> BeatGrid {
    let transients = detect_transients(
        samples,
        sample_rate,
        BEAT_FFT_SIZE,
        BEAT_HOP_SIZE,
        BEAT_SENSITIVITY,
    );

    if transients.onsets.len() < 2 {
        return BeatGrid {
            beats: transients.onsets,
            bpm: 0.0,
            sample_rate,
        };
    }

    // Compute inter-onset intervals
    let intervals: Vec<usize> = transients.onsets.windows(2).map(|w| w[1] - w[0]).collect();

    // Estimate BPM from median interval (robust to outliers)
    let bpm = estimate_bpm_from_intervals(&intervals, sample_rate);

    // Quantize onsets to beat grid
    let beat_interval = (60.0 * sample_rate as f64 / bpm) as usize;
    let beats = quantize_to_grid(&transients.onsets, beat_interval);

    BeatGrid {
        beats,
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

/// Quantizes onset positions to a regular grid.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beat_grid_snap() {
        let grid = BeatGrid {
            beats: vec![0, 22050, 44100, 66150],
            bpm: 120.0,
            sample_rate: 44100,
        };
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
        let grid = BeatGrid {
            beats: vec![0],
            bpm: 120.0,
            sample_rate: 44100,
        };
        // 60 * 44100 / 120 = 22050
        assert!((grid.beat_interval_samples() - 22050.0).abs() < 1.0);
    }

    #[test]
    fn test_beat_interval_samples_128bpm_48khz() {
        let grid = BeatGrid {
            beats: vec![0],
            bpm: 128.0,
            sample_rate: 48000,
        };
        // 60 * 48000 / 128 = 22500
        assert!((grid.beat_interval_samples() - 22500.0).abs() < 1.0);
    }

    // --- snap_to_grid edge cases ---

    #[test]
    fn test_snap_to_grid_empty_beats() {
        let grid = BeatGrid {
            beats: vec![],
            bpm: 120.0,
            sample_rate: 44100,
        };
        // Empty grid → return position unchanged
        assert_eq!(grid.snap_to_grid(1000), 1000);
    }

    #[test]
    fn test_snap_to_grid_before_first_beat() {
        let grid = BeatGrid {
            beats: vec![1000, 2000, 3000],
            bpm: 120.0,
            sample_rate: 44100,
        };
        // Position before any beat → snaps to first beat
        assert_eq!(grid.snap_to_grid(500), 1000);
    }

    #[test]
    fn test_snap_to_grid_after_last_beat() {
        let grid = BeatGrid {
            beats: vec![1000, 2000, 3000],
            bpm: 120.0,
            sample_rate: 44100,
        };
        // Position after all beats → snaps to last beat
        assert_eq!(grid.snap_to_grid(10000), 3000);
    }

    #[test]
    fn test_snap_to_grid_equidistant() {
        let grid = BeatGrid {
            beats: vec![0, 100, 200],
            bpm: 120.0,
            sample_rate: 44100,
        };
        // Position exactly between beats 0 and 100 → snaps to first one found
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
        let grid = BeatGrid {
            beats: vec![0, 22050, 44100],
            bpm: 120.0,
            sample_rate: 44100,
        };
        assert_eq!(grid.snap_to_grid(22050), 22050);
    }

    // --- estimate_bpm_from_intervals ---

    #[test]
    fn test_estimate_bpm_halving_high_bpm() {
        // Raw BPM 320 → should halve to 160 (within MAX_EDM_BPM)
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
        // Raw BPM 50 → should double to 100 (at MIN_EDM_BPM)
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
        // Median of sorted = [outlier, 22050, 22050, 22050, 22050] → median = 22050
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
        // beat_interval == 0 → return onsets as-is
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
}

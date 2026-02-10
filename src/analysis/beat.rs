//! Beat detection and grid snapping for 4/4 EDM patterns.

use crate::analysis::transient::detect_transients;

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
    let fft_size = 2048;
    let hop_size = 512;

    // Use transient detection focused on low frequencies (kicks)
    let transients = detect_transients(samples, sample_rate, fft_size, hop_size, 0.4);

    if transients.onsets.len() < 2 {
        return BeatGrid {
            beats: transients.onsets,
            bpm: 0.0,
            sample_rate,
        };
    }

    // Compute inter-onset intervals
    let mut intervals: Vec<usize> = Vec::new();
    for i in 1..transients.onsets.len() {
        intervals.push(transients.onsets[i] - transients.onsets[i - 1]);
    }

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

    // Snap to reasonable EDM BPM range (100-160)
    let mut bpm = raw_bpm;
    while bpm > 160.0 {
        bpm /= 2.0;
    }
    while bpm < 100.0 && bpm > 0.0 {
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
}

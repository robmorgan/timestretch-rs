use super::transient::detect_transients;

/// Beat grid representing detected beat positions snapped to a 4/4 grid.
#[derive(Debug, Clone)]
pub struct BeatGrid {
    /// Beat positions in samples.
    pub beat_positions: Vec<usize>,
    /// Estimated BPM.
    pub estimated_bpm: f64,
    /// Samples per beat.
    pub samples_per_beat: f64,
}

/// Detect beats in a 4/4 EDM signal and snap them to a grid.
///
/// # Arguments
/// * `samples` - Mono audio samples
/// * `sample_rate` - Sample rate in Hz
/// * `expected_bpm_range` - Expected BPM range (low, high), e.g., (120.0, 130.0)
pub fn detect_beats(
    samples: &[f32],
    sample_rate: u32,
    expected_bpm_range: (f64, f64),
) -> BeatGrid {
    let markers = detect_transients(samples, sample_rate, 0.6, 2048);

    if markers.positions.len() < 2 {
        return BeatGrid {
            beat_positions: markers.positions,
            estimated_bpm: 0.0,
            samples_per_beat: 0.0,
        };
    }

    // Compute inter-onset intervals
    let mut intervals: Vec<usize> = Vec::new();
    for i in 1..markers.positions.len() {
        intervals.push(markers.positions[i] - markers.positions[i - 1]);
    }

    // Convert expected BPM range to sample intervals
    let min_interval = (60.0 / expected_bpm_range.1 * sample_rate as f64) as usize;
    let max_interval = (60.0 / expected_bpm_range.0 * sample_rate as f64) as usize;

    // Filter intervals within expected BPM range (allow some tolerance)
    let tolerance = 0.3;
    let min_with_tol = (min_interval as f64 * (1.0 - tolerance)) as usize;
    let max_with_tol = (max_interval as f64 * (1.0 + tolerance)) as usize;

    let valid_intervals: Vec<usize> = intervals
        .iter()
        .copied()
        .filter(|&i| i >= min_with_tol && i <= max_with_tol)
        .collect();

    if valid_intervals.is_empty() {
        // Try using all intervals
        let avg_interval =
            intervals.iter().sum::<usize>() as f64 / intervals.len() as f64;
        let bpm = 60.0 * sample_rate as f64 / avg_interval;
        return BeatGrid {
            beat_positions: markers.positions,
            estimated_bpm: bpm,
            samples_per_beat: avg_interval,
        };
    }

    // Estimate tempo from median valid interval
    let mut sorted_intervals = valid_intervals;
    sorted_intervals.sort();
    let median_interval = sorted_intervals[sorted_intervals.len() / 2];
    let samples_per_beat = median_interval as f64;
    let estimated_bpm = 60.0 * sample_rate as f64 / samples_per_beat;

    // Snap beat positions to grid starting from the first detected onset
    let first_beat = markers.positions[0];
    let total_samples = samples.len();
    let mut beat_positions = Vec::new();
    let mut pos = first_beat as f64;
    while (pos as usize) < total_samples {
        beat_positions.push(pos as usize);
        pos += samples_per_beat;
    }

    BeatGrid {
        beat_positions,
        estimated_bpm,
        samples_per_beat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beat_detection_regular_clicks() {
        let sample_rate = 44100u32;
        let bpm = 128.0;
        let samples_per_beat = (60.0 / bpm * sample_rate as f64) as usize;
        let duration_secs = 4.0;
        let total_samples = (sample_rate as f64 * duration_secs) as usize;
        let mut samples = vec![0.0f32; total_samples];

        // Create click train at 128 BPM
        for beat in 0..16 {
            let pos = beat * samples_per_beat;
            if pos + 50 < total_samples {
                for j in 0..50 {
                    samples[pos + j] = 0.9;
                }
            }
        }

        let grid = detect_beats(&samples, sample_rate, (120.0, 135.0));
        assert!(
            !grid.beat_positions.is_empty(),
            "Should detect beats in click train"
        );
        // BPM estimate should be somewhat close to 128
        if grid.estimated_bpm > 0.0 {
            assert!(
                (grid.estimated_bpm - 128.0).abs() < 20.0,
                "BPM estimate {} should be near 128",
                grid.estimated_bpm
            );
        }
    }

    #[test]
    fn test_beat_detection_silence() {
        let samples = vec![0.0f32; 44100];
        let grid = detect_beats(&samples, 44100, (120.0, 130.0));
        assert!(grid.beat_positions.is_empty() || grid.estimated_bpm == 0.0);
    }

    #[test]
    fn test_beat_detection_short_input() {
        let samples = vec![0.5f32; 100];
        let grid = detect_beats(&samples, 44100, (120.0, 130.0));
        // Should handle gracefully without panicking
        let _ = grid;
    }
}

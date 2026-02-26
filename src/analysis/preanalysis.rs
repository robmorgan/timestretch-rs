//! Offline pre-analysis pipeline for DJ beat/onset alignment.

use crate::analysis::beat::detect_beats;
use crate::analysis::transient::detect_transients;
use crate::core::preanalysis::PreAnalysisArtifact;

const PREANALYSIS_FFT_SIZE: usize = 2048;
const PREANALYSIS_HOP_SIZE: usize = 512;
const PREANALYSIS_SENSITIVITY: f32 = 0.4;

/// Produces a reusable beat/onset analysis artifact for runtime snapping.
pub fn analyze_for_dj(samples: &[f32], sample_rate: u32) -> PreAnalysisArtifact {
    let beats = detect_beats(samples, sample_rate);
    let transients = detect_transients(
        samples,
        sample_rate,
        PREANALYSIS_FFT_SIZE,
        PREANALYSIS_HOP_SIZE,
        PREANALYSIS_SENSITIVITY,
    );

    let bpm = if beats.bpm.is_finite() && beats.bpm > 0.0 {
        beats.bpm
    } else {
        0.0
    };

    let downbeat_offset_samples = if bpm > 0.0 && !beats.beats.is_empty() {
        let beat_interval = 60.0 * sample_rate as f64 / bpm;
        (beats.beats[0] as f64).rem_euclid(beat_interval).round() as usize
    } else {
        0
    };

    let confidence = estimate_confidence(&beats.beats, &transients.onsets, sample_rate);

    PreAnalysisArtifact {
        sample_rate,
        bpm,
        downbeat_offset_samples,
        confidence,
        beat_positions: beats.beats,
        transient_onsets: transients.onsets,
    }
}

/// Estimates confidence from beat regularity and onset support.
fn estimate_confidence(beats: &[usize], onsets: &[usize], sample_rate: u32) -> f32 {
    if beats.len() < 3 {
        return 0.0;
    }

    let intervals: Vec<f64> = beats
        .windows(2)
        .map(|w| w[1].saturating_sub(w[0]) as f64)
        .filter(|&v| v > 0.0)
        .collect();
    if intervals.len() < 2 {
        return 0.0;
    }

    let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
    if mean <= 0.0 {
        return 0.0;
    }
    let variance = intervals
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / intervals.len() as f64;
    let std = variance.sqrt();
    let cv = std / mean;

    // Regular interval pattern => high confidence.
    let regularity = (1.0 - (cv / 0.25)).clamp(0.0, 1.0);

    // More onsets around beat regions also boosts confidence.
    let seconds = (beats.last().copied().unwrap_or(0) as f64 / sample_rate as f64).max(1.0);
    let beat_rate = beats.len() as f64 / seconds;
    let onset_rate = onsets.len() as f64 / seconds;

    let beat_density_score = (beat_rate / 2.0).clamp(0.0, 1.0); // ~120 BPM => 2 beats/s
    let onset_support_score = (onset_rate / 8.0).clamp(0.0, 1.0);

    (0.6 * regularity + 0.25 * beat_density_score + 0.15 * onset_support_score) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_for_dj_click_train_has_confidence() {
        let sample_rate = 44100u32;
        let bpm = 120.0;
        let beat_interval = (60.0 * sample_rate as f64 / bpm) as usize;
        let len = sample_rate as usize * 4;
        let mut signal = vec![0.0f32; len];

        for i in (0..len).step_by(beat_interval) {
            for j in 0..10.min(len - i) {
                signal[i + j] = if j < 5 { 1.0 } else { -0.5 };
            }
        }

        let artifact = analyze_for_dj(&signal, sample_rate);
        assert!(artifact.bpm > 0.0);
        assert!(artifact.confidence > 0.2);
        assert!(!artifact.beat_positions.is_empty());
    }
}

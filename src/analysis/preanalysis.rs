//! Offline pre-analysis pipeline for DJ beat/onset alignment.

use crate::analysis::beat::detect_beats_from_transients;
use crate::analysis::transient::detect_transients;
use crate::core::preanalysis::{hash_samples, PreAnalysisArtifact, PREANALYSIS_VERSION};

const PREANALYSIS_FFT_SIZE: usize = 2048;
const PREANALYSIS_HOP_SIZE: usize = 512;
const PREANALYSIS_SENSITIVITY: f32 = 0.4;

/// Downmixes interleaved audio to the mono analysis signal expected by
/// [`analyze_for_dj`]: the input itself for mono, or the mid channel
/// `(L + R) * 0.5` for stereo (matching mid/side encoding). Extra channels
/// beyond the first two are ignored.
pub fn downmix_to_mid(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks(channels)
        .map(|frame| {
            let left = frame.first().copied().unwrap_or(0.0);
            let right = frame.get(1).copied().unwrap_or(left);
            (left + right) * 0.5
        })
        .collect()
}

/// Diagnostic telemetry from an offline analysis run.
///
/// Produced by [`analyze_for_dj_with_report`] alongside the artifact and
/// never serialized — it exists so analysis mistakes can be inspected
/// during tuning (onset detection function statistics, onset rate, timing).
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct AnalysisReport {
    /// Median of the combined onset detection function (ODF).
    pub odf_median: f32,
    /// Median absolute deviation of the ODF around its median.
    pub odf_mad: f32,
    /// Maximum of the ODF.
    pub odf_max: f32,
    /// Number of detected transient onsets.
    pub onset_count: usize,
    /// Detected onsets per second of source audio.
    pub onset_rate_per_sec: f64,
    /// Wall-clock analysis time in seconds.
    pub analysis_elapsed_secs: f64,
}

/// Produces a reusable beat/onset analysis artifact for runtime snapping.
///
/// `samples` must be the mono analysis signal (see [`downmix_to_mid`]).
/// The returned artifact carries content binding (length + hash) so stale
/// artifacts can be rejected via
/// [`PreAnalysisArtifact::matches_source`].
pub fn analyze_for_dj(samples: &[f32], sample_rate: u32) -> PreAnalysisArtifact {
    analyze_for_dj_with_report(samples, sample_rate).0
}

/// [`analyze_for_dj`] plus an [`AnalysisReport`] for tuning/diagnostics.
pub fn analyze_for_dj_with_report(
    samples: &[f32],
    sample_rate: u32,
) -> (PreAnalysisArtifact, AnalysisReport) {
    let started = std::time::Instant::now();
    // One detection pass drives both the onset artifact and beat tracking.
    // Beat detection uses the same 2048/512/0.4 configuration, so a separate
    // pass would recompute exactly this map.
    let transients = detect_transients(
        samples,
        sample_rate,
        PREANALYSIS_FFT_SIZE,
        PREANALYSIS_HOP_SIZE,
        PREANALYSIS_SENSITIVITY,
    );
    let beats = detect_beats_from_transients(&transients, sample_rate);

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

    let transient_strengths = if transients.strengths.len() == transients.onsets.len() {
        transients.strengths.clone()
    } else {
        vec![1.0; transients.onsets.len()]
    };

    let hop = transients.hop_size.max(1);
    let onset_band_flux = transients
        .onsets
        .iter()
        .map(|&onset| {
            transients
                .per_frame_band_flux
                .get(onset / hop)
                .copied()
                .unwrap_or([0.0; 4])
        })
        .collect();

    let report = AnalysisReport {
        odf_median: median_of(&transients.flux),
        odf_mad: mad_of(&transients.flux),
        odf_max: transients.flux.iter().copied().fold(0.0f32, f32::max),
        onset_count: transients.onsets.len(),
        onset_rate_per_sec: transients.onsets.len() as f64
            / (samples.len() as f64 / sample_rate.max(1) as f64).max(1e-9),
        analysis_elapsed_secs: started.elapsed().as_secs_f64(),
    };

    let artifact = PreAnalysisArtifact {
        version: PREANALYSIS_VERSION,
        sample_rate,
        bpm,
        downbeat_offset_samples,
        confidence,
        beat_positions: beats.beats,
        transient_onsets: transients.onsets,
        transient_strengths,
        onset_band_flux,
        analysis_hop_size: hop,
        source_len_samples: samples.len(),
        content_hash: hash_samples(samples),
    };

    (artifact, report)
}

/// Median of a slice (0.0 when empty).
fn median_of(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted[sorted.len() / 2]
}

/// Median absolute deviation around the median (0.0 when empty).
fn mad_of(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let median = median_of(values);
    let deviations: Vec<f32> = values.iter().map(|&v| (v - median).abs()).collect();
    median_of(&deviations)
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

    #[test]
    fn test_analyze_for_dj_fills_v2_fields() {
        let sample_rate = 44100u32;
        let beat_interval = (60.0 * sample_rate as f64 / 120.0) as usize;
        let len = sample_rate as usize * 4;
        let mut signal = vec![0.0f32; len];
        for i in (0..len).step_by(beat_interval) {
            for j in 0..10.min(len - i) {
                signal[i + j] = if j < 5 { 1.0 } else { -0.5 };
            }
        }

        let artifact = analyze_for_dj(&signal, sample_rate);
        assert_eq!(artifact.version, PREANALYSIS_VERSION);
        assert_eq!(artifact.source_len_samples, len);
        assert_eq!(artifact.content_hash, hash_samples(&signal));
        assert_eq!(artifact.analysis_hop_size, PREANALYSIS_HOP_SIZE);
        assert_eq!(
            artifact.transient_strengths.len(),
            artifact.transient_onsets.len()
        );
        assert_eq!(
            artifact.onset_band_flux.len(),
            artifact.transient_onsets.len()
        );
        assert!(artifact.matches_source(&signal, sample_rate));
    }

    #[test]
    fn test_downmix_to_mid() {
        let mono = vec![0.1, 0.2, 0.3];
        assert_eq!(downmix_to_mid(&mono, 1), mono);

        let stereo = vec![1.0, 0.0, 0.5, 0.5, -1.0, 1.0];
        assert_eq!(downmix_to_mid(&stereo, 2), vec![0.5, 0.5, 0.0]);
    }
}

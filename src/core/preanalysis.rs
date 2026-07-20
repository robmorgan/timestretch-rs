//! Offline pre-analysis artifact for DJ beat/onset alignment.

use crate::error::StretchError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Current schema version written by [`crate::analyze_for_dj`].
///
/// v4: beat/onset positions are latency-compensated (centered on the
/// audible attack instead of the analysis-window start; ~29 ms later at
/// the 2048/512 configuration).
pub const PREANALYSIS_VERSION: u32 = 4;

/// Oldest schema version whose positions match the current analysis.
/// Artifacts below this carry the pre-v4 window-start bias and fail
/// [`PreAnalysisArtifact::matches_source`], so cached sidecars regenerate.
const MIN_COMPATIBLE_VERSION: u32 = 4;

fn default_artifact_version() -> u32 {
    1
}

/// A stretch of consecutive beats at (locally) constant tempo.
///
/// Serialized in the artifact (schema v3+) and used as the tempo model of
/// [`crate::BeatGrid`]: `start_beat` indexes the grid's beat sequence, and
/// the segment runs until the next segment's `start_beat`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TempoSegment {
    /// Index of the first beat of this segment in the beat sequence.
    pub start_beat: usize,
    /// Tempo within the segment in BPM.
    pub bpm: f64,
}

/// Serializable beat/onset analysis artifact produced offline and reused at runtime.
///
/// All positions are absolute source frames (per-channel sample indices) at
/// [`sample_rate`](Self::sample_rate), measured on the mono analysis signal:
/// the file itself for mono audio, or the mid downmix `(L + R) * 0.5` for
/// stereo (see [`crate::downmix_to_mid`]). Batch consumers assume their input
/// is the entire analyzed file starting at source frame 0; positions past the
/// end of the input are ignored.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreAnalysisArtifact {
    /// Schema version. Version 1 artifacts (no strengths, no content binding)
    /// remain usable; content validation is skipped when unknown.
    #[serde(default = "default_artifact_version")]
    pub version: u32,
    /// Sample rate used during analysis.
    pub sample_rate: u32,
    /// Estimated BPM.
    pub bpm: f64,
    /// Downbeat phase offset in samples.
    pub downbeat_offset_samples: usize,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f32,
    /// Beat positions in samples.
    #[serde(default)]
    pub beat_positions: Vec<usize>,
    /// Fractional-sample beat positions, parallel to `beat_positions`.
    /// Empty for artifacts older than schema v3.
    #[serde(default)]
    pub beat_positions_fractional: Vec<f64>,
    /// Indices into `beat_positions` marking downbeats (bar starts).
    /// Empty for artifacts older than schema v3.
    #[serde(default)]
    pub downbeat_beat_indices: Vec<usize>,
    /// Piecewise-constant tempo segments over the beat sequence.
    /// Empty for artifacts older than schema v3 (treat as one segment at
    /// [`bpm`](Self::bpm)).
    #[serde(default)]
    pub tempo_segments: Vec<TempoSegment>,
    /// Detected transient onset positions in samples.
    #[serde(default)]
    pub transient_onsets: Vec<usize>,
    /// Normalized onset strengths in [0, 1], parallel to `transient_onsets`.
    /// May be empty for version 1 artifacts (treated as 1.0 per onset).
    #[serde(default)]
    pub transient_strengths: Vec<f32>,
    /// Per-onset band flux `[sub_bass, low, mid, high]`, parallel to
    /// `transient_onsets`. May be empty for version 1 artifacts.
    #[serde(default)]
    pub onset_band_flux: Vec<[f32; 4]>,
    /// Hop size used during analysis (0 = unknown, version 1 artifacts).
    #[serde(default)]
    pub analysis_hop_size: usize,
    /// Length in frames of the mono analysis signal (0 = unknown).
    #[serde(default)]
    pub source_len_samples: usize,
    /// FNV-1a 64 hash of the mono analysis signal (0 = unknown).
    /// See [`hash_samples`].
    #[serde(default)]
    pub content_hash: u64,
}

impl PreAnalysisArtifact {
    /// Returns true when artifact confidence passes the provided threshold.
    #[inline]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold.clamp(0.0, 1.0)
    }

    /// Runtime gate: true when the artifact can drive stretching decisions
    /// for audio at `sample_rate`.
    ///
    /// Requires a sample-rate match, confidence at or above
    /// `confidence_threshold`, and at least one beat or transient position.
    /// This intentionally does not hash audio; use [`Self::matches_source`]
    /// at load boundaries instead.
    #[inline]
    pub fn is_usable(&self, sample_rate: u32, confidence_threshold: f32) -> bool {
        self.sample_rate == sample_rate
            && self.is_confident(confidence_threshold)
            && (!self.beat_positions.is_empty() || !self.transient_onsets.is_empty())
    }

    /// Load-boundary gate: true when the artifact was produced from exactly
    /// this mono analysis signal by a compatible analysis version.
    ///
    /// Checks schema version, sample rate, source length, and content hash.
    /// Length and hash checks are skipped when the artifact predates them
    /// (version 1). Artifacts older than `MIN_COMPATIBLE_VERSION` are
    /// rejected outright: their positions carry the pre-v4 window-start
    /// bias, so a cached sidecar must be regenerated, not reused.
    pub fn matches_source(&self, samples: &[f32], sample_rate: u32) -> bool {
        if self.version < MIN_COMPATIBLE_VERSION {
            return false;
        }
        if self.sample_rate != sample_rate {
            return false;
        }
        if self.source_len_samples != 0 && self.source_len_samples != samples.len() {
            return false;
        }
        if self.content_hash != 0 && self.content_hash != hash_samples(samples) {
            return false;
        }
        true
    }

    /// Returns the strength for onset `idx`, defaulting to 1.0 when the
    /// artifact carries no strengths (version 1).
    #[inline]
    pub fn strength_at(&self, idx: usize) -> f32 {
        self.transient_strengths.get(idx).copied().unwrap_or(1.0)
    }

    /// Rescales every frame-domain position to `sample_rate`, so a track can
    /// be analyzed once at its native rate and reused at any playback rate.
    ///
    /// Positions (beats, onsets, downbeat offset), the analysis hop, and the
    /// source length scale by the rate ratio; BPM, confidence, indices,
    /// strengths, and band flux are rate-invariant. The returned artifact's
    /// content binding is cleared (`source_len_samples`/`content_hash` = 0):
    /// it no longer corresponds to any concrete signal, so
    /// [`Self::matches_source`] must be run against the *original* artifact
    /// at the native rate, never against a resampled copy.
    ///
    /// Returns a plain clone when `sample_rate` already matches.
    pub fn resample_to(&self, sample_rate: u32) -> Self {
        if sample_rate == self.sample_rate || self.sample_rate == 0 {
            return self.clone();
        }
        let ratio = sample_rate as f64 / self.sample_rate as f64;
        let scale = |v: usize| (v as f64 * ratio).round() as usize;
        Self {
            sample_rate,
            downbeat_offset_samples: scale(self.downbeat_offset_samples),
            beat_positions: self.beat_positions.iter().map(|&p| scale(p)).collect(),
            beat_positions_fractional: self
                .beat_positions_fractional
                .iter()
                .map(|&p| p * ratio)
                .collect(),
            transient_onsets: self.transient_onsets.iter().map(|&p| scale(p)).collect(),
            analysis_hop_size: scale(self.analysis_hop_size),
            source_len_samples: 0,
            content_hash: 0,
            ..self.clone()
        }
    }
}

/// Hashes a mono analysis signal with FNV-1a 64 over each sample's bit
/// pattern. Used to bind a [`PreAnalysisArtifact`] to its source audio.
pub fn hash_samples(samples: &[f32]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for sample in samples {
        for byte in sample.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

/// Writes a pre-analysis artifact as JSON.
pub fn write_preanalysis_json(
    path: &Path,
    artifact: &PreAnalysisArtifact,
) -> Result<(), StretchError> {
    let json = serde_json::to_string_pretty(artifact).map_err(|e| {
        StretchError::InvalidFormat(format!("failed to serialize pre-analysis artifact: {}", e))
    })?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Reads a pre-analysis artifact from JSON.
pub fn read_preanalysis_json(path: &Path) -> Result<PreAnalysisArtifact, StretchError> {
    let data = std::fs::read_to_string(path)?;
    serde_json::from_str(&data).map_err(|e| {
        StretchError::InvalidFormat(format!(
            "failed to parse pre-analysis artifact from {}: {}",
            path.display(),
            e
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_artifact() -> PreAnalysisArtifact {
        PreAnalysisArtifact {
            version: PREANALYSIS_VERSION,
            sample_rate: 44100,
            bpm: 128.0,
            downbeat_offset_samples: 100,
            confidence: 0.8,
            beat_positions: vec![0, 22050],
            beat_positions_fractional: vec![0.0, 22050.0],
            downbeat_beat_indices: vec![0],
            tempo_segments: vec![TempoSegment {
                start_beat: 0,
                bpm: 128.0,
            }],
            transient_onsets: vec![0, 22050],
            transient_strengths: vec![1.0, 0.5],
            onset_band_flux: vec![[1.0, 0.5, 0.2, 0.1], [0.2, 0.3, 0.4, 0.5]],
            analysis_hop_size: 512,
            source_len_samples: 44100,
            content_hash: 0,
        }
    }

    #[test]
    fn test_preanalysis_confidence_threshold() {
        let artifact = test_artifact();
        assert!(artifact.is_confident(0.5));
        assert!(!artifact.is_confident(0.9));
    }

    #[test]
    fn test_is_usable_gates() {
        let artifact = test_artifact();
        assert!(artifact.is_usable(44100, 0.5));
        assert!(!artifact.is_usable(48000, 0.5), "sample-rate mismatch");
        assert!(!artifact.is_usable(44100, 0.9), "confidence too low");

        let empty = PreAnalysisArtifact {
            beat_positions: Vec::new(),
            transient_onsets: Vec::new(),
            ..test_artifact()
        };
        assert!(!empty.is_usable(44100, 0.5), "no positions at all");
    }

    #[test]
    fn test_matches_source_binding() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut artifact = test_artifact();
        artifact.source_len_samples = samples.len();
        artifact.content_hash = hash_samples(&samples);

        assert!(artifact.matches_source(&samples, 44100));
        assert!(!artifact.matches_source(&samples, 48000), "rate mismatch");
        assert!(
            !artifact.matches_source(&samples[..999], 44100),
            "length mismatch"
        );

        let mut altered = samples.clone();
        altered[500] += 0.25;
        assert!(!artifact.matches_source(&altered, 44100), "hash mismatch");

        // Current-version artifacts without binding skip content validation.
        artifact.source_len_samples = 0;
        artifact.content_hash = 0;
        assert!(artifact.matches_source(&altered, 44100));

        // Pre-v4 artifacts carry window-start-biased positions: never
        // reused at load boundaries, even when otherwise matching.
        artifact.version = 3;
        assert!(
            !artifact.matches_source(&altered, 44100),
            "stale version must be regenerated"
        );
    }

    #[test]
    fn test_strength_at_v1_default() {
        let mut artifact = test_artifact();
        assert_eq!(artifact.strength_at(1), 0.5);
        artifact.transient_strengths.clear();
        assert_eq!(artifact.strength_at(0), 1.0);
        assert_eq!(artifact.strength_at(999), 1.0);
    }

    #[test]
    fn test_resample_to_scales_positions() {
        let artifact = test_artifact(); // 44.1k, beats at 0 / 22050
        let resampled = artifact.resample_to(88_200);
        assert_eq!(resampled.sample_rate, 88_200);
        assert_eq!(resampled.beat_positions, vec![0, 44_100]);
        assert_eq!(resampled.beat_positions_fractional, vec![0.0, 44_100.0]);
        assert_eq!(resampled.transient_onsets, vec![0, 44_100]);
        assert_eq!(resampled.downbeat_offset_samples, 200);
        assert_eq!(resampled.analysis_hop_size, 1024);
        // Rate-invariant fields survive untouched.
        assert_eq!(resampled.bpm, artifact.bpm);
        assert_eq!(resampled.confidence, artifact.confidence);
        assert_eq!(
            resampled.downbeat_beat_indices,
            artifact.downbeat_beat_indices
        );
        assert_eq!(resampled.tempo_segments, artifact.tempo_segments);
        assert_eq!(resampled.transient_strengths, artifact.transient_strengths);
        assert_eq!(resampled.onset_band_flux, artifact.onset_band_flux);
        assert_eq!(resampled.version, artifact.version);
    }

    #[test]
    fn test_resample_to_clears_content_binding() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut artifact = test_artifact();
        artifact.source_len_samples = samples.len();
        artifact.content_hash = hash_samples(&samples);

        let resampled = artifact.resample_to(48_000);
        assert_eq!(resampled.source_len_samples, 0);
        assert_eq!(resampled.content_hash, 0);
        // Identity keeps the binding.
        let same = artifact.resample_to(44_100);
        assert_eq!(same.content_hash, artifact.content_hash);
        assert_eq!(same.source_len_samples, artifact.source_len_samples);
    }

    #[test]
    fn test_resample_round_trip_is_close() {
        let artifact = test_artifact();
        let round = artifact.resample_to(48_000).resample_to(44_100);
        for (a, b) in round.beat_positions.iter().zip(&artifact.beat_positions) {
            assert!((*a as i64 - *b as i64).abs() <= 1, "{a} vs {b}");
        }
    }

    #[test]
    fn test_v1_json_parses_with_defaults() {
        let v1_json = r#"{
            "sample_rate": 44100,
            "bpm": 128.0,
            "downbeat_offset_samples": 100,
            "confidence": 0.8,
            "beat_positions": [0, 22050],
            "transient_onsets": [0, 22050]
        }"#;
        let artifact: PreAnalysisArtifact =
            serde_json::from_str(v1_json).expect("v1 JSON should parse");
        assert_eq!(artifact.version, 1);
        assert!(artifact.transient_strengths.is_empty());
        assert!(artifact.onset_band_flux.is_empty());
        assert_eq!(artifact.analysis_hop_size, 0);
        assert_eq!(artifact.source_len_samples, 0);
        assert_eq!(artifact.content_hash, 0);
        assert!(artifact.is_usable(44100, 0.5));
    }
}

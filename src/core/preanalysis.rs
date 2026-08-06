//! Offline pre-analysis artifact for DJ beat/onset alignment.

use crate::error::StretchError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Current schema version written by [`crate::analyze_for_dj`].
///
/// v4: beat/onset positions are latency-compensated (centered on the
/// audible attack instead of the analysis-window start; ~29 ms later at
/// the 2048/512 configuration).
///
/// v5: adds the optional [`key`](PreAnalysisArtifact::key) estimate. Purely
/// additive — v4 sidecars stay compatible, they just carry no key.
///
/// v6: adds the optional [`loudness`](PreAnalysisArtifact::loudness)
/// measurement. Purely additive — v4/v5 sidecars stay compatible, they
/// just carry no loudness.
///
/// v7: adds [`tempo_candidates`](PreAnalysisArtifact::tempo_candidates).
/// Purely additive — older sidecars stay compatible with an empty list.
///
/// v8: no schema change — bumped (with `MIN_COMPATIBLE_VERSION`) because
/// the rigid-grid beat fit (v0.10.0) materially changed beat positions on
/// quantized material without a version bump, so v7 artifacts are
/// ambiguous: they may carry either the old wandering grids or the new
/// rigid ones. Forcing regeneration disambiguates.
///
/// POLICY (learned from the v7 ambiguity, see LEARNINGS.md): bump
/// `PREANALYSIS_VERSION` on any release whose analysis output changes
/// materially, and raise `MIN_COMPATIBLE_VERSION` with it whenever cached
/// results from the previous version would be *worse* than re-analysis —
/// schema compatibility alone is not the bar. Checked at release time via
/// RELEASE_CHECKLIST.md.
pub const PREANALYSIS_VERSION: u32 = 8;

/// Oldest schema version whose *analysis results* match the current
/// detector. Artifacts below this fail
/// [`PreAnalysisArtifact::matches_source`], so cached sidecars regenerate:
/// pre-v4 carried the window-start bias; v4–v7 predate (or are ambiguous
/// about) the rigid-grid beat fit.
const MIN_COMPATIBLE_VERSION: u32 = 8;

fn default_artifact_version() -> u32 {
    1
}

/// Mode of a detected musical key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyMode {
    /// Major mode.
    Major,
    /// Minor mode.
    Minor,
}

/// A detected musical key (schema v5+), produced by
/// [`crate::analysis::key::detect_key`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KeyEstimate {
    /// Root pitch class: 0 = C, 1 = C#, ... 11 = B.
    pub root: u8,
    /// Major or minor.
    pub mode: KeyMode,
    /// Margin of the winning key over the runner-up, in [0.0, 1.0]. The
    /// runner-up is often the relative major/minor, so values are modest
    /// even on clearly tonal material.
    pub confidence: f32,
}

impl KeyEstimate {
    /// Note names using sharps (`"C#"`, not `"Db"`).
    const NOTE_NAMES: [&'static str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];

    /// Conventional name, e.g. `"A minor"` or `"F# major"`. Sharps are used
    /// for all accidentals.
    pub fn name(&self) -> String {
        let mode = match self.mode {
            KeyMode::Major => "major",
            KeyMode::Minor => "minor",
        };
        format!("{} {}", Self::NOTE_NAMES[usize::from(self.root) % 12], mode)
    }

    /// Camelot wheel notation for harmonic mixing, e.g. `"8B"` for C major
    /// and `"8A"` for A minor.
    pub fn camelot(&self) -> String {
        // Position on the circle of fifths (C = 0, G = 1, ...).
        let fifth = (usize::from(self.root) * 7) % 12;
        let (number, letter) = match self.mode {
            KeyMode::Major => ((fifth + 7) % 12 + 1, 'B'),
            KeyMode::Minor => ((fifth + 4) % 12 + 1, 'A'),
        };
        format!("{number}{letter}")
    }
}

/// ITU-R BS.1770-4 / EBU R128 loudness measurement (schema v6+), produced
/// by [`crate::analysis::loudness::measure_loudness`].
///
/// Measured on the original interleaved audio, not the mono analysis
/// downmix: BS.1770 sums per-channel energies, so a mid downmix would
/// read up to ~3 dB low depending on channel correlation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LoudnessMeasurement {
    /// Integrated (gated) loudness in LUFS.
    pub integrated_lufs: f64,
    /// Maximum true peak across channels in dBTP.
    pub true_peak_dbtp: f64,
    /// Loudness range in LU (EBU R128 LRA).
    pub loudness_range_lu: f64,
}

impl LoudnessMeasurement {
    /// Gain in dB that brings this track's integrated loudness to
    /// `target_lufs` (negative when the track is louder than the target).
    /// The DJ-app autogain primitive.
    #[inline]
    pub fn gain_db_to(&self, target_lufs: f64) -> f64 {
        target_lufs - self.integrated_lufs
    }

    /// [`Self::gain_db_to`] as a linear amplitude factor.
    #[inline]
    pub fn gain_linear_to(&self, target_lufs: f64) -> f64 {
        10f64.powf(self.gain_db_to(target_lufs) / 20.0)
    }
}

/// A ranked tempo hypothesis (schema v7+).
///
/// The detector commits to one tempo, but the canonical failure mode of
/// any tempo tracker is the octave: half/double of the truth. Exposing
/// the metrical alternatives with their measured salience lets a DJ app
/// offer a one-tap "halve/double BPM" correction ranked by evidence
/// instead of blind ×2/÷2 buttons.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TempoCandidate {
    /// Candidate tempo in BPM.
    pub bpm: f64,
    /// Mean normalized tempogram salience along this candidate's tempo
    /// path, in [0, 1]. Comparable across candidates of the same track.
    /// The committed tempo is the entry whose `bpm` matches the grid /
    /// artifact BPM; on clearly periodic material it is also the
    /// highest-salience entry.
    pub salience: f32,
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
    /// Detected musical key. `None` when detection was inconclusive or the
    /// artifact predates schema v5.
    #[serde(default)]
    pub key: Option<KeyEstimate>,
    /// BS.1770-4 loudness measurement. Not filled by
    /// [`crate::analyze_for_dj`] (which only sees the mono analysis
    /// signal): callers measure the original interleaved audio with
    /// [`crate::measure_loudness`] and store the result here. `None` when
    /// never measured or the artifact predates schema v6.
    #[serde(default)]
    pub loudness: Option<LoudnessMeasurement>,
    /// Ranked tempo hypotheses, highest salience first: the committed
    /// tempo plus its in-range metrical alternatives (½×/2×). Empty when
    /// no tempo was detected or the artifact predates schema v7.
    #[serde(default)]
    pub tempo_candidates: Vec<TempoCandidate>,
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
        // Hash only when the artifact actually binds one (matches_identity
        // skips the comparison when the artifact's hash is 0 either way).
        let hash = if self.content_hash != 0 {
            hash_samples(samples)
        } else {
            0
        };
        self.matches_identity(sample_rate, samples.len(), hash)
    }

    /// [`Self::matches_source`] without the samples in hand: checks the
    /// same schema-version gate and the precomputed length/hash identity,
    /// skipping each binding the artifact predates. For callers that
    /// already hold the identity (e.g. the `.tsa` container's file header).
    pub fn matches_identity(
        &self,
        sample_rate: u32,
        source_len_samples: usize,
        content_hash: u64,
    ) -> bool {
        if self.version < MIN_COMPATIBLE_VERSION {
            return false;
        }
        if self.sample_rate != sample_rate {
            return false;
        }
        if self.source_len_samples != 0 && self.source_len_samples != source_len_samples {
            return false;
        }
        if self.content_hash != 0 && self.content_hash != content_hash {
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
#[deprecated(
    since = "0.11.0",
    note = "use `write_analysis_file` with the `.tsa` container (`crate::io::tsa`), which also carries waveform peaks"
)]
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
#[deprecated(
    since = "0.11.0",
    note = "use `read_analysis_file` / `read_analysis_file_validated` on the `.tsa` container (`crate::io::tsa`)"
)]
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
            key: Some(KeyEstimate {
                root: 9,
                mode: KeyMode::Minor,
                confidence: 0.4,
            }),
            loudness: Some(LoudnessMeasurement {
                integrated_lufs: -9.5,
                true_peak_dbtp: -0.2,
                loudness_range_lu: 4.0,
            }),
            tempo_candidates: vec![
                TempoCandidate {
                    bpm: 128.0,
                    salience: 0.9,
                },
                TempoCandidate {
                    bpm: 64.0,
                    salience: 0.5,
                },
            ],
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
        assert_eq!(resampled.key, artifact.key);
        assert_eq!(resampled.loudness, artifact.loudness);
        assert_eq!(resampled.tempo_candidates, artifact.tempo_candidates);
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

    #[test]
    fn test_v4_json_without_key_parses_as_none() {
        let mut artifact = test_artifact();
        artifact.version = 4;
        artifact.key = None;
        artifact.loudness = None;
        let json = serde_json::to_string(&artifact).unwrap();
        let parsed: PreAnalysisArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.key, None);
        assert_eq!(parsed.loudness, None);
        assert_eq!(parsed.version, 4);

        let round: PreAnalysisArtifact =
            serde_json::from_str(&serde_json::to_string(&test_artifact()).unwrap()).unwrap();
        assert_eq!(round.key, test_artifact().key);
        assert_eq!(round.loudness, test_artifact().loudness);
    }

    #[test]
    fn test_v5_json_without_loudness_parses_as_none() {
        // A v5 sidecar (has key, predates loudness) must stay READABLE —
        // but as of v8 it is no longer cache-valid: v4–v7 artifacts are
        // ambiguous about the rigid-grid beat fit, so `matches_source`
        // rejects them and they regenerate.
        let mut artifact = test_artifact();
        artifact.version = 5;
        artifact.loudness = None;
        artifact.tempo_candidates = Vec::new();
        let json = serde_json::to_string(&artifact).unwrap();
        let parsed: PreAnalysisArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.loudness, None);
        assert!(parsed.tempo_candidates.is_empty());
        assert_eq!(parsed.key, test_artifact().key);
        assert!(parsed.version < MIN_COMPATIBLE_VERSION);
        assert!(!parsed.matches_source(&[0.0; 4], 44100));
    }

    #[test]
    fn test_v6_json_without_candidates_parses_as_empty() {
        // A v6 sidecar (has loudness, predates tempo candidates) must
        // stay readable, and the full v7 artifact must round-trip.
        let mut artifact = test_artifact();
        artifact.version = 6;
        artifact.tempo_candidates = Vec::new();
        let json = serde_json::to_string(&artifact).unwrap();
        let parsed: PreAnalysisArtifact = serde_json::from_str(&json).unwrap();
        assert!(parsed.tempo_candidates.is_empty());
        assert_eq!(parsed.loudness, test_artifact().loudness);

        let round: PreAnalysisArtifact =
            serde_json::from_str(&serde_json::to_string(&test_artifact()).unwrap()).unwrap();
        assert_eq!(round.tempo_candidates, test_artifact().tempo_candidates);
    }

    #[test]
    fn test_key_names_and_camelot() {
        let key = |root, mode| KeyEstimate {
            root,
            mode,
            confidence: 1.0,
        };
        assert_eq!(key(0, KeyMode::Major).name(), "C major");
        assert_eq!(key(9, KeyMode::Minor).name(), "A minor");
        assert_eq!(key(6, KeyMode::Major).name(), "F# major");

        // Camelot wheel: relative keys share a number, fifths are adjacent.
        assert_eq!(key(0, KeyMode::Major).camelot(), "8B"); // C major
        assert_eq!(key(9, KeyMode::Minor).camelot(), "8A"); // A minor
        assert_eq!(key(7, KeyMode::Major).camelot(), "9B"); // G major
        assert_eq!(key(11, KeyMode::Major).camelot(), "1B"); // B major
        assert_eq!(key(8, KeyMode::Minor).camelot(), "1A"); // G# minor
        assert_eq!(key(5, KeyMode::Major).camelot(), "7B"); // F major
        assert_eq!(key(2, KeyMode::Minor).camelot(), "7A"); // D minor
    }
}

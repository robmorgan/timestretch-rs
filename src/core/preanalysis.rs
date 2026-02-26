//! Offline pre-analysis artifact for DJ beat/onset alignment.

use crate::error::StretchError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Serializable beat/onset analysis artifact produced offline and reused at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreAnalysisArtifact {
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
    /// Detected transient onset positions in samples.
    #[serde(default)]
    pub transient_onsets: Vec<usize>,
}

impl PreAnalysisArtifact {
    /// Returns true when artifact confidence passes the provided threshold.
    #[inline]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold.clamp(0.0, 1.0)
    }
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

    #[test]
    fn test_preanalysis_confidence_threshold() {
        let artifact = PreAnalysisArtifact {
            sample_rate: 44100,
            bpm: 128.0,
            downbeat_offset_samples: 100,
            confidence: 0.8,
            beat_positions: vec![0, 22050],
            transient_onsets: vec![0, 22050],
        };

        assert!(artifact.is_confident(0.5));
        assert!(!artifact.is_confident(0.9));
    }
}

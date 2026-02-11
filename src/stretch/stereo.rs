//! Mid/Side stereo processing for improved stereo coherence.
//!
//! When stretching stereo audio, processing L and R channels independently
//! causes phase drift between them, resulting in comb filtering and stereo
//! image collapse. Mid/Side encoding avoids this by processing the shared
//! content (Mid) and difference (Side) separately, preserving their
//! natural phase relationship.

use crate::core::types::StretchParams;
use crate::error::StretchError;
use crate::stretch::hybrid::HybridStretcher;

/// Stereo processing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoMode {
    /// Process L/R channels independently (legacy behavior).
    /// Faster but can cause stereo image collapse on complex material.
    Independent,
    /// Mid/Side encoding: M=(L+R)/2, S=(L-R)/2.
    /// Preserves stereo coherence by ensuring shared content (center image)
    /// is processed as a single signal. Default for stereo.
    MidSide,
}

/// Encodes stereo L/R channels into Mid/Side.
///
/// - Mid = (L + R) / 2 (center content: kick, bass, vocals)
/// - Side = (L - R) / 2 (stereo width: reverb, panning, stereo effects)
#[inline]
pub fn encode_mid_side(left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let len = left.len().min(right.len());
    let mut mid = Vec::with_capacity(len);
    let mut side = Vec::with_capacity(len);
    for i in 0..len {
        mid.push((left[i] + right[i]) * 0.5);
        side.push((left[i] - right[i]) * 0.5);
    }
    (mid, side)
}

/// Decodes Mid/Side back to stereo L/R.
///
/// - L = Mid + Side
/// - R = Mid - Side
#[inline]
pub fn decode_mid_side(mid: &[f32], side: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let len = mid.len().min(side.len());
    let mut left = Vec::with_capacity(len);
    let mut right = Vec::with_capacity(len);
    for i in 0..len {
        left.push(mid[i] + side[i]);
        right.push(mid[i] - side[i]);
    }
    (left, right)
}

/// Stretches stereo audio using Mid/Side processing.
///
/// Converts L/R to M/S, processes each through the hybrid stretcher,
/// then converts back to L/R. This preserves the stereo image because
/// the shared spectral content (Mid) is processed as a coherent signal.
pub fn stretch_mid_side(
    left: &[f32],
    right: &[f32],
    params: &StretchParams,
) -> Result<(Vec<f32>, Vec<f32>), StretchError> {
    let (mid, side) = encode_mid_side(left, right);

    // Process Mid through full hybrid pipeline
    let mid_stretcher = HybridStretcher::new(params.clone());
    let mid_stretched = mid_stretcher.process(&mid)?;

    // Process Side through full hybrid pipeline
    let side_stretcher = HybridStretcher::new(params.clone());
    let side_stretched = side_stretcher.process(&side)?;

    // Decode back to L/R
    Ok(decode_mid_side(&mid_stretched, &side_stretched))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let left: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let right: Vec<f32> = (0..100).map(|i| (i as f32 * 0.15).sin()).collect();

        let (mid, side) = encode_mid_side(&left, &right);
        let (left_out, right_out) = decode_mid_side(&mid, &side);

        for i in 0..100 {
            assert!(
                (left_out[i] - left[i]).abs() < 1e-6,
                "Left mismatch at {}: {} vs {}",
                i,
                left_out[i],
                left[i]
            );
            assert!(
                (right_out[i] - right[i]).abs() < 1e-6,
                "Right mismatch at {}: {} vs {}",
                i,
                right_out[i],
                right[i]
            );
        }
    }

    #[test]
    fn test_mono_mid_side() {
        // Identical L/R should produce zero Side
        let signal: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let (mid, side) = encode_mid_side(&signal, &signal);

        for i in 0..100 {
            assert!(
                (mid[i] - signal[i]).abs() < 1e-6,
                "Mid should equal input for mono"
            );
            assert!(side[i].abs() < 1e-6, "Side should be zero for mono");
        }
    }

    #[test]
    fn test_opposite_channels() {
        // L = -R should produce zero Mid and Side = L
        let left: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let right: Vec<f32> = left.iter().map(|s| -s).collect();

        let (mid, side) = encode_mid_side(&left, &right);

        for i in 0..100 {
            assert!(mid[i].abs() < 1e-6, "Mid should be zero for opposite channels");
            assert!(
                (side[i] - left[i]).abs() < 1e-6,
                "Side should equal left for opposite channels"
            );
        }
    }

    #[test]
    fn test_different_lengths() {
        let left = vec![1.0; 50];
        let right = vec![0.5; 100];
        let (mid, side) = encode_mid_side(&left, &right);
        assert_eq!(mid.len(), 50); // Truncates to shorter
        assert_eq!(side.len(), 50);
    }
}

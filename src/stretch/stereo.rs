//! Mid/Side stereo processing for improved stereo coherence.
//!
//! When stretching stereo audio, processing L and R channels independently
//! causes phase drift between them, resulting in comb filtering and stereo
//! image collapse. Mid/Side encoding avoids this by processing the shared
//! content (Mid) and difference (Side) separately, preserving their
//! natural phase relationship.

use crate::analysis::beat::detect_beats;
use crate::analysis::transient::{detect_transients_with_options, TransientDetectionOptions};
use crate::core::types::StretchParams;
use crate::error::StretchError;
use crate::stretch::hybrid::{merge_onsets_and_beats, HybridStretcher};

/// Maximum FFT size used for shared transient detection.
const STEREO_TRANSIENT_MAX_FFT: usize = 2048;
/// Maximum hop size used for shared transient detection.
const STEREO_TRANSIENT_MAX_HOP: usize = 512;
/// Minimum signal length before beat detection is used to augment shared map.
const STEREO_MIN_SAMPLES_FOR_BEAT_DETECTION: usize = 44100;

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
    let (shared_onsets, shared_strengths) = build_shared_onset_map(&mid, params);

    // Process Mid and Side using the same segmentation anchors.
    let mid_stretcher = HybridStretcher::new(params.clone());
    let mid_stretched =
        mid_stretcher.process_with_onsets(&mid, &shared_onsets, &shared_strengths)?;

    let side_stretcher = HybridStretcher::new(params.clone());
    let side_stretched =
        side_stretcher.process_with_onsets(&side, &shared_onsets, &shared_strengths)?;

    // Deterministic channel length agreement before decode.
    let target_len = params.output_length(mid.len());
    let mid_aligned = force_channel_length(mid_stretched, target_len);
    let side_aligned = force_channel_length(side_stretched, target_len);

    // Decode back to L/R
    Ok(decode_mid_side(&mid_aligned, &side_aligned))
}

/// Builds a shared transient/beat map from the Mid channel.
fn build_shared_onset_map(mid: &[f32], params: &StretchParams) -> (Vec<usize>, Vec<f32>) {
    let transient_map = detect_transients_with_options(
        mid,
        params.sample_rate,
        params.fft_size.min(STEREO_TRANSIENT_MAX_FFT),
        params.hop_size.min(STEREO_TRANSIENT_MAX_HOP),
        params.transient_sensitivity,
        TransientDetectionOptions::from_stretch_params(params),
    );

    let onsets = transient_map.onsets.clone();
    let strengths = if transient_map.strengths.len() == onsets.len() {
        transient_map.strengths.clone()
    } else {
        vec![1.0; onsets.len()]
    };
    if params.beat_aware && mid.len() >= STEREO_MIN_SAMPLES_FOR_BEAT_DETECTION {
        let beat_grid = detect_beats(mid, params.sample_rate);
        return merge_onsets_and_beats(&onsets, &strengths, &beat_grid.beats, mid.len());
    }

    (onsets, strengths)
}

/// Forces a channel to deterministic target length without pitch-shifting.
fn force_channel_length(mut channel: Vec<f32>, target_len: usize) -> Vec<f32> {
    if channel.len() == target_len {
        return channel;
    }
    if target_len == 0 {
        return Vec::new();
    }
    if channel.is_empty() {
        return vec![0.0; target_len];
    }
    if channel.len() > target_len {
        channel.truncate(target_len);
    } else {
        let pad = *channel.last().unwrap_or(&0.0);
        channel.resize(target_len, pad);
    }
    channel
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
            assert!(
                mid[i].abs() < 1e-6,
                "Mid should be zero for opposite channels"
            );
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

    fn mid_side_energy_ratio(left: &[f32], right: &[f32]) -> f32 {
        let n = left.len().min(right.len()).max(1);
        let mut mid_e = 0.0f64;
        let mut side_e = 0.0f64;
        for i in 0..n {
            let m = (left[i] + right[i]) * 0.5;
            let s = (left[i] - right[i]) * 0.5;
            mid_e += (m as f64) * (m as f64);
            side_e += (s as f64) * (s as f64);
        }
        let mid_rms = (mid_e / n as f64).sqrt().max(1e-12);
        let side_rms = (side_e / n as f64).sqrt();
        (side_rms / mid_rms) as f32
    }

    fn best_lag(left: &[f32], right: &[f32], max_lag: isize) -> isize {
        let n = left.len().min(right.len()) as isize;
        let mut best = 0isize;
        let mut best_score = f64::NEG_INFINITY;
        for lag in -max_lag..=max_lag {
            let mut sum = 0.0f64;
            let mut count = 0isize;
            for i in 0..n {
                let j = i + lag;
                if j < 0 || j >= n {
                    continue;
                }
                sum += left[i as usize] as f64 * right[j as usize] as f64;
                count += 1;
            }
            if count > 0 && sum > best_score {
                best_score = sum;
                best = lag;
            }
        }
        best
    }

    #[test]
    fn test_stretch_mid_side_channel_length_agreement() {
        let sample_rate = 44100u32;
        let n = sample_rate as usize * 2;
        let left: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let right: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 330.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.37)
            .with_sample_rate(sample_rate)
            .with_channels(2)
            .with_preset(crate::core::types::EdmPreset::HouseLoop);
        let (out_l, out_r) = stretch_mid_side(&left, &right, &params).unwrap();

        assert_eq!(out_l.len(), out_r.len(), "L/R output lengths must match");
        let expected = params.output_length(n);
        let err = out_l.len().abs_diff(expected);
        assert!(
            err <= 1,
            "Channel length should match target within 1 frame: got {}, expected {}",
            out_l.len(),
            expected
        );
    }

    #[test]
    fn test_stretch_mid_side_energy_coherence() {
        let sample_rate = 44100u32;
        let n = sample_rate as usize * 2;
        let left: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.7 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
            })
            .collect();
        let right: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.7 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    - 0.25 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
            })
            .collect();

        let before_ratio = mid_side_energy_ratio(&left, &right);

        let params = StretchParams::new(1.25)
            .with_sample_rate(sample_rate)
            .with_channels(2)
            .with_preset(crate::core::types::EdmPreset::DjBeatmatch);
        let (out_l, out_r) = stretch_mid_side(&left, &right, &params).unwrap();
        let after_ratio = mid_side_energy_ratio(&out_l, &out_r);

        assert!(
            (after_ratio - before_ratio).abs() < 0.35,
            "Mid/side energy ratio drift too large: before {:.4}, after {:.4}",
            before_ratio,
            after_ratio
        );
    }

    #[test]
    fn test_stretch_mid_side_phase_drift_bound() {
        let sample_rate = 44100u32;
        let n = sample_rate as usize * 2;
        let lag_in = 8usize;
        let base: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();
        let left = base.clone();
        let mut right = vec![0.0f32; n];
        for (i, sample) in right.iter_mut().enumerate().take(n) {
            let src = i.saturating_sub(lag_in);
            *sample = base[src];
        }

        let params = StretchParams::new(1.25)
            .with_sample_rate(sample_rate)
            .with_channels(2)
            .with_preset(crate::core::types::EdmPreset::HouseLoop);
        let (out_l, out_r) = stretch_mid_side(&left, &right, &params).unwrap();

        let measured_lag = best_lag(&out_l, &out_r, 64);
        let expected_lag = (lag_in as f64 * params.stretch_ratio).round() as isize;
        assert!(
            (measured_lag - expected_lag).abs() <= 16,
            "Inter-channel lag drift too large: measured {}, expected {}",
            measured_lag,
            expected_lag
        );
    }
}

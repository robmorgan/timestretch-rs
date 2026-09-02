//! Offline (batch) driver over the engine graph — ROADMAP Stage 8.
//!
//! Offline mode is the same engine with the luxuries a file affords:
//! the whole track is pre-analyzed (the artifact is always present, never a
//! fallback), the source is fed to completion with a kernel-flush pad so
//! the resampler emits every real sample, and the output length is exact
//! by construction — `round(input_frames * ratio)` frames, with the
//! pipeline-latency prefix dropped structurally rather than by post-render
//! truncation heuristics.
//!
//! Two paths cover the ratio space:
//!
//! - **DJ window** (rate deviation within the keylock's fully-corrected
//!   range): the engine graph itself, [`EngineProfile::Keylock`] at a
//!   constant rate. Streaming-vs-offline agreement is true by construction
//!   — same graph, same constants — and that includes the keylock's deck
//!   semantic: content below the 120 Hz crossover is NOT pitch-corrected
//!   (its pitch follows tempo), offline exactly as live (owner decision
//!   2026-07-16: batch copies the real-time path).
//! - **Wide ratios** (beyond the corrected range, where the live chain
//!   deliberately fades toward varispeed — a deck semantic, not a batch
//!   one): a direct batch [`PhaseVocoder`] render per channel at FFT 2048.
//!   Pitch is preserved across the whole spectrum on this path; quality
//!   (notably low-band level at heavy compression) is explicitly secondary
//!   per the binding EDM/DJ-first product boundary.

use std::sync::Arc;

use crate::analysis::preanalysis::analyze;
use crate::core::preanalysis::PreAnalysisArtifact;
use crate::core::resample::STREAM_SINC_HALF_TAPS;
use crate::core::window::WindowType;
use crate::engine::{Engine, EngineConfig, EngineProfile};
use crate::error::StretchError;
use crate::stretch::phase_locking::PhaseLockingMode;
use crate::stretch::phase_vocoder::PhaseVocoder;

/// Rate deviations at or below this render through the engine graph; the
/// bound is the keylock's fully-corrected range (the extreme-rate fade
/// starts just past it — see `CORRECTION_FADE_START_DEV`).
pub const OFFLINE_GRAPH_MAX_DEV: f64 = 0.20;

/// Batch FFT for the wide-ratio phase-vocoder path.
const WIDE_PV_FFT: usize = 2_048;

/// Renders a whole buffer at a constant stretch `ratio`
/// (output length / input length). Pitch is preserved above the keylock
/// crossover on the graph path (the low band follows tempo, matching the
/// live deck) and across the whole spectrum on the wide-ratio path.
///
/// `input` is interleaved; the output is interleaved with exactly
/// `round(input_frames * ratio)` frames. A caller-provided `pre_analysis`
/// is authoritative (even one claiming no transients — online fallbacks
/// stay suppressed); `None` pre-analyzes the input here.
pub fn stretch_offline(
    input: &[f32],
    channels: usize,
    sample_rate: u32,
    ratio: f64,
    pre_analysis: Option<Arc<PreAnalysisArtifact>>,
) -> Result<Vec<f32>, StretchError> {
    // Validate before any arithmetic: `channels == 0` would divide by
    // zero, and an interleave mismatch (input length not a whole number of
    // frames) would leave a partial frame the source ring can never accept
    // — in release builds the feed loop would spin forever on it.
    if !(1..=8).contains(&channels) {
        return Err(StretchError::InvalidFormat(format!(
            "channel count {channels} outside 1..=8"
        )));
    }
    if input.len() % channels != 0 {
        return Err(StretchError::InvalidFormat(format!(
            "input length {} is not a whole number of {}-channel frames",
            input.len(),
            channels
        )));
    }
    if !(crate::stretch::params::RATIO_MIN..=crate::stretch::params::RATIO_MAX).contains(&ratio) {
        return Err(StretchError::InvalidRatio(format!(
            "Stretch ratio must be between {} and {}, got {}",
            crate::stretch::params::RATIO_MIN,
            crate::stretch::params::RATIO_MAX,
            ratio
        )));
    }
    if input.is_empty() {
        return Ok(vec![]);
    }
    let rate = 1.0 / ratio;
    if (1.0 - rate).abs() <= OFFLINE_GRAPH_MAX_DEV {
        stretch_via_graph(
            input,
            channels,
            sample_rate,
            ratio,
            pre_analysis,
            EngineProfile::Keylock,
        )
    } else if (crate::engine::MIN_TEMPO_RATE..=crate::engine::MAX_TEMPO_RATE).contains(&rate) {
        // Wide ratios inside the engine's rate range run the SHIPPED wide
        // corrector (ROADMAP Stage 14): offline and live wide renders are
        // the same algorithm, stereo image handling included. The direct
        // batch PV below survives only for ratios beyond any deck use
        // (> 4x either way).
        stretch_via_graph(
            input,
            channels,
            sample_rate,
            ratio,
            pre_analysis,
            EngineProfile::WideKeylock,
        )
    } else {
        stretch_wide_pv(input, channels, sample_rate, ratio)
    }
}

/// Engine-graph render at constant rate with a guaranteed artifact.
fn stretch_via_graph(
    input: &[f32],
    channels: usize,
    sample_rate: u32,
    ratio: f64,
    pre_analysis: Option<Arc<PreAnalysisArtifact>>,
    profile: EngineProfile,
) -> Result<Vec<f32>, StretchError> {
    let rate = 1.0 / ratio;
    let frames = input.len() / channels;
    let expected_frames = (frames as f64 * ratio).round() as usize;

    // Whole-file pre-analysis: offline always has an artifact — the
    // caller's when provided (authoritative), computed here otherwise.
    let artifact: Arc<PreAnalysisArtifact> = match pre_analysis {
        Some(artifact) => artifact,
        None => {
            let mid: Vec<f32> = if channels == 1 {
                input.to_vec()
            } else {
                crate::downmix_to_mid(input, channels)
            };
            Arc::new(analyze(&mid, sample_rate))
        }
    };

    let handles = Engine::build(EngineConfig {
        sample_rate,
        channels,
        profile,
        initial_tempo_rate: rate,
        pre_analysis: Some(artifact),
        ..EngineConfig::default()
    })?;
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);
    source.set_track_position(0);
    let latency = processor.pipeline_latency_frames();

    // Flush pad: enough zero source to (a) push the resampler's final
    // half-kernel of real samples through and (b) keep the pipeline fed
    // while the constant latency drains — `expected + latency` frames must
    // deliver without an underrun, or the tail would depend on the
    // caller's demand pattern (underrun silence pads at the delivery
    // layer, bypassing the stage chain mid-fade).
    // The head's source-side lookahead (the wide PV's analysis window;
    // the varispeed's sinc kernel) must also be fed past the last real
    // frame, or the final windows starve into a mode-dependent terminal
    // underrun (Stage 19: reported pipeline latency is 0 for the wide
    // head — its window is lookahead, not delay).
    let flush_frames = (latency as f64 * rate).ceil() as usize
        + processor.varispeed_lookahead_frames()
        + 4 * STREAM_SINC_HALF_TAPS;
    let flush: Vec<f32> = vec![0.0; flush_frames * channels];

    let mut feed = 0usize;
    let mut flush_fed = 0usize;
    let mut finished = false;
    let mut out = vec![0.0f32; 1_024 * channels];
    let total_needed = (expected_frames + latency) * channels;
    let mut collected: Vec<f32> = Vec::with_capacity(total_needed + out.len());
    while collected.len() < total_needed {
        while feed < input.len() && source.free_frames() > 0 {
            let end = (feed + 8_192 * channels).min(input.len());
            // push() returns FRAMES accepted; the cursor tracks samples.
            feed += source.push(&input[feed..end]) * channels;
        }
        if feed >= input.len() {
            while flush_fed < flush.len() && source.free_frames() > 0 {
                flush_fed += source.push(&flush[flush_fed..]) * channels;
            }
            if flush_fed >= flush.len() && !finished {
                finished = source.finish();
            }
        }
        let underruns_before = controller.underrun_frames();
        processor.process(&mut out);
        collected.extend_from_slice(&out);
        // Terminal underrun means the source (including the flush pad) is
        // fully consumed: everything real is already in `collected`.
        if finished && controller.underrun_frames() > underruns_before {
            collected.resize(total_needed, 0.0);
            break;
        }
    }

    // Structural latency trim + exact length by construction.
    collected.drain(..latency * channels);
    collected.truncate(expected_frames * channels);
    debug_assert_eq!(collected.len(), expected_frames * channels);
    Ok(collected)
}

/// Direct batch phase-vocoder render for ratios outside the graph window.
fn stretch_wide_pv(
    input: &[f32],
    channels: usize,
    sample_rate: u32,
    ratio: f64,
) -> Result<Vec<f32>, StretchError> {
    let frames = input.len() / channels;
    let expected_frames = (frames as f64 * ratio).round() as usize;
    // 87.5% overlap, matching the live wide stage: hop = FFT/4 is the
    // documented -75% tempo level/click blowup configuration (see
    // WIDE_HOP in engine/stages/wide_keylock.rs).
    let hop = WIDE_PV_FFT / 8;

    // Inputs shorter than an analysis window can't be phase-vocoded;
    // resample to length (pitch follows — matches the old short-input
    // fallback's spirit without WSOLA).
    if frames < WIDE_PV_FFT {
        let mut out = vec![0.0f32; expected_frames * channels];
        for ch in 0..channels {
            let mono: Vec<f32> = input.iter().skip(ch).step_by(channels).copied().collect();
            let resampled = crate::core::resample::resample_linear(&mono, expected_frames.max(1));
            for (i, &sample) in resampled.iter().take(expected_frames).enumerate() {
                out[i * channels + ch] = sample;
            }
        }
        return Ok(out);
    }

    let mut outs: Vec<Vec<f32>> = Vec::with_capacity(channels);
    for ch in 0..channels {
        let mono: Vec<f32> = input.iter().skip(ch).step_by(channels).copied().collect();
        let mut pv = PhaseVocoder::with_options(
            WIDE_PV_FFT,
            hop,
            ratio,
            sample_rate,
            100.0,
            WindowType::Hann,
            PhaseLockingMode::Identity,
        );
        let mut rendered = pv.process(&mono)?;
        // The batch PV lands within a hop of the target; fit exactly.
        rendered.resize(expected_frames, 0.0);
        outs.push(rendered);
    }

    let mut interleaved = vec![0.0f32; expected_frames * channels];
    for (ch, data) in outs.iter().enumerate() {
        for (i, &s) in data.iter().enumerate() {
            interleaved[i * channels + ch] = s;
        }
    }
    Ok(interleaved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| 0.5 * (2.0 * std::f64::consts::PI * freq * i as f64 / 44_100.0).sin() as f32)
            .collect()
    }

    #[test]
    fn regression_zero_channels_errs_not_panics() {
        // Found by the Stage 12 robustness harness: `channels == 0`
        // reached `input.len() / channels` — divide-by-zero panic.
        assert!(stretch_offline(&[0.0; 16], 0, 44_100, 1.5, None).is_err());
        assert!(stretch_offline(&[0.0; 16], 9, 44_100, 1.5, None).is_err());
    }

    #[test]
    fn regression_interleave_mismatch_errs_not_hangs() {
        // Found by the Stage 12 robustness harness: an input length that
        // is not a whole number of frames left a partial frame the source
        // ring could never accept — a debug assert in `SourceProducer::
        // push`, and in release an infinite feed loop. Must be an error.
        assert!(stretch_offline(&[0.0; 3], 2, 44_100, 1.05, None).is_err());
        assert!(stretch_offline(&[0.0; 4097], 2, 44_100, 0.5, None).is_err());
    }

    #[test]
    fn regression_non_finite_and_out_of_range_ratio_errs() {
        // stretch_offline is itself public: it cannot rely on callers
        // running validate_params. NaN / 0 / out-of-range ratios must Err.
        for ratio in [f64::NAN, f64::INFINITY, 0.0, -1.0, 0.009, 100.1] {
            assert!(
                stretch_offline(&[0.0; 64], 1, 44_100, ratio, None).is_err(),
                "ratio {ratio} must be rejected"
            );
        }
    }

    #[test]
    fn graph_path_output_length_is_exact() {
        let input = sine(440.0, 44_100 * 3);
        for ratio in [0.85f64, 0.926, 1.0417, 1.08, 1.2] {
            let out = stretch_offline(&input, 1, 44_100, ratio, None).unwrap();
            let expected = (input.len() as f64 * ratio).round() as usize;
            assert_eq!(out.len(), expected, "ratio {ratio}");
        }
    }

    #[test]
    fn wide_path_output_length_is_exact() {
        let input = sine(440.0, 44_100 * 2);
        for ratio in [0.5f64, 1.5, 2.0] {
            let out = stretch_offline(&input, 1, 44_100, ratio, None).unwrap();
            let expected = (input.len() as f64 * ratio).round() as usize;
            assert_eq!(out.len(), expected, "ratio {ratio}");
        }
    }

    #[test]
    fn graph_path_preserves_pitch_and_level() {
        let input = sine(440.0, 44_100 * 4);
        let out = stretch_offline(&input, 1, 44_100, 1.06, None).unwrap();
        // Measure over the mid section, away from edges.
        let scan = &out[44_100..44_100 * 3];
        let (mut first, mut last, mut count) = (None, None, 0usize);
        for i in 1..scan.len() {
            let (a, b) = (scan[i - 1] as f64, scan[i] as f64);
            if a <= 0.0 && b > 0.0 {
                let t = (i - 1) as f64 + a / (a - b);
                if first.is_none() {
                    first = Some(t);
                }
                last = Some(t);
                count += 1;
            }
        }
        let freq = (count - 1) as f64 * 44_100.0 / (last.unwrap() - first.unwrap());
        let cents = 1_200.0 * (freq / 440.0).log2();
        assert!(cents.abs() < 6.0, "pitch off by {cents:.2} cents");
        let rms =
            (scan.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / scan.len() as f64).sqrt();
        assert!((rms - 0.3535).abs() < 0.05, "level off: rms {rms:.4}");
    }

    #[test]
    fn stereo_stays_interleaved_and_exact() {
        let mono = sine(440.0, 44_100 * 2);
        let mut input = Vec::with_capacity(mono.len() * 2);
        for &s in &mono {
            input.push(s);
            input.push(-0.5 * s);
        }
        let out = stretch_offline(&input, 2, 44_100, 1.05, None).unwrap();
        let expected = ((mono.len()) as f64 * 1.05).round() as usize * 2;
        assert_eq!(out.len(), expected);
        // The exact -0.5 channel relationship survives (graph processes
        // channels in lockstep).
        for i in (88_200..out.len() - 88_200).step_by(2) {
            assert!(
                (out[i + 1] + 0.5 * out[i]).abs() < 1e-3,
                "stereo relationship broken at frame {}",
                i / 2
            );
        }
    }
}

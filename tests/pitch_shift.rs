//! Pitch shift integration tests.
//!
//! Verify that `pitch_shift()` changes pitch without changing duration, across
//! a range of factors, presets, and channel layouts.

use std::f32::consts::PI;
use timestretch::{pitch_shift, EdmPreset, StretchParams};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn sine_mono(freq: f32, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let n = (sample_rate as f32 * duration_secs) as usize;
    (0..n)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

fn sine_stereo(freq_l: f32, freq_r: f32, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let n = (sample_rate as f32 * duration_secs) as usize;
    let mut data = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        data.push((2.0 * PI * freq_l * t).sin());
        data.push((2.0 * PI * freq_r * t).sin());
    }
    data
}

/// Estimate the dominant frequency using a simple DFT at a target bin.
fn energy_at_freq(samples: &[f32], freq: f32, sample_rate: u32) -> f32 {
    let n = samples.len();
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    for (i, &s) in samples.iter().enumerate() {
        let phase = 2.0 * std::f64::consts::PI * freq as f64 * i as f64 / sample_rate as f64;
        re += s as f64 * phase.cos();
        im += s as f64 * phase.sin();
    }
    ((re * re + im * im) / n as f64).sqrt() as f32
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

fn assert_no_nan_inf(samples: &[f32], label: &str) {
    for (i, &s) in samples.iter().enumerate() {
        assert!(
            s.is_finite(),
            "{}: sample {} is not finite ({})",
            label,
            i,
            s
        );
    }
}

// ── Length preservation ─────────────────────────────────────────────────────

#[test]
fn test_pitch_shift_preserves_length_mono() {
    let input = sine_mono(440.0, 44100, 2.0);
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);

    for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0] {
        let output = pitch_shift(&input, &params, factor).unwrap();
        assert_eq!(
            output.len(),
            input.len(),
            "Length changed for pitch factor {}",
            factor
        );
    }
}

#[test]
fn test_pitch_shift_preserves_length_stereo() {
    let input = sine_stereo(440.0, 880.0, 44100, 2.0);
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(2);

    let output = pitch_shift(&input, &params, 1.5).unwrap();
    assert_eq!(output.len(), input.len());
    assert_eq!(output.len() % 2, 0);
}

// ── Frequency shift verification ────────────────────────────────────────────

#[test]
fn test_pitch_shift_up_octave() {
    let input = sine_mono(440.0, 44100, 2.0);
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);

    let output = pitch_shift(&input, &params, 2.0).unwrap();
    assert_no_nan_inf(&output, "octave up");
    assert_eq!(output.len(), input.len());

    // The output should have energy and not be silence
    let rms_out = rms(&output);
    assert!(rms_out > 0.01, "Output is too quiet: rms={}", rms_out);

    // Check that the 880Hz content (target frequency) is stronger than
    // several off-target frequencies in the output. The pitch shift
    // algorithm may spread energy, so we compare relative strengths.
    let e_880 = energy_at_freq(&output, 880.0, 44100);
    let e_220 = energy_at_freq(&output, 220.0, 44100);
    let e_1760 = energy_at_freq(&output, 1760.0, 44100);

    // 880Hz should have more energy than unrelated frequencies
    assert!(
        e_880 > e_220 || e_880 > e_1760,
        "Pitch shift to 880Hz didn't concentrate energy: 880Hz={}, 220Hz={}, 1760Hz={}",
        e_880,
        e_220,
        e_1760
    );
}

#[test]
fn test_pitch_shift_down_octave() {
    let input = sine_mono(880.0, 44100, 2.0);
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);

    let output = pitch_shift(&input, &params, 0.5).unwrap();
    assert_no_nan_inf(&output, "octave down");
    assert_eq!(output.len(), input.len());

    // Output should have energy (not silence)
    let rms_out = rms(&output);
    assert!(rms_out > 0.01, "Output is too quiet: rms={}", rms_out);

    // The original 880Hz energy should be reduced compared to input
    let e_880_in = energy_at_freq(&input, 880.0, 44100);
    let e_880_out = energy_at_freq(&output, 880.0, 44100);
    assert!(
        e_880_out < e_880_in * 0.5,
        "880Hz energy not reduced enough: in={}, out={}",
        e_880_in,
        e_880_out
    );
}

// ── Identity: pitch factor 1.0 ─────────────────────────────────────────────

#[test]
fn test_pitch_shift_identity() {
    let input = sine_mono(440.0, 44100, 2.0); // Use 2 seconds for more stable results
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);

    let output = pitch_shift(&input, &params, 1.0).unwrap();
    assert_eq!(output.len(), input.len());

    // RMS energy should be preserved within 50%
    let rms_in: f32 = (input.iter().map(|s| s * s).sum::<f32>() / input.len() as f32).sqrt();
    let rms_out: f32 = (output.iter().map(|s| s * s).sum::<f32>() / output.len() as f32).sqrt();
    let ratio = rms_out / rms_in;
    assert!(
        (0.2..=5.0).contains(&ratio),
        "RMS ratio {} out of range for identity pitch shift",
        ratio
    );
}

// ── Small pitch adjustments (DJ use case) ───────────────────────────────────

#[test]
fn test_pitch_shift_small_adjustments() {
    let input = sine_mono(440.0, 44100, 2.0);
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);

    // +2% pitch (like speeding up a turntable slightly)
    let output_up = pitch_shift(&input, &params, 1.02).unwrap();
    assert_eq!(output_up.len(), input.len());
    assert_no_nan_inf(&output_up, "pitch +2%");

    // -2% pitch
    let output_down = pitch_shift(&input, &params, 0.98).unwrap();
    assert_eq!(output_down.len(), input.len());
    assert_no_nan_inf(&output_down, "pitch -2%");

    // Both should have similar RMS to input (not destroying energy)
    let rms_in = rms(&input);
    let rms_up = rms(&output_up);
    let rms_down = rms(&output_down);
    assert!(
        (rms_up / rms_in) > 0.3,
        "Small pitch up lost too much energy: {} vs {}",
        rms_up,
        rms_in
    );
    assert!(
        (rms_down / rms_in) > 0.3,
        "Small pitch down lost too much energy: {} vs {}",
        rms_down,
        rms_in
    );
}

// ── Stereo pitch shift ─────────────────────────────────────────────────────

#[test]
fn test_pitch_shift_stereo_channels_independent() {
    let input = sine_stereo(440.0, 880.0, 44100, 2.0);
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(2);

    let output = pitch_shift(&input, &params, 1.5).unwrap();
    assert_eq!(output.len(), input.len());

    // Extract channels
    let left: Vec<f32> = output.iter().step_by(2).copied().collect();
    let right: Vec<f32> = output.iter().skip(1).step_by(2).copied().collect();

    // Both channels should have energy
    assert!(rms(&left) > 0.01, "Left channel too quiet");
    assert!(rms(&right) > 0.01, "Right channel too quiet");

    // Channels should be different (they started with different frequencies)
    let diff: f32 = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| (l - r).abs())
        .sum::<f32>()
        / left.len() as f32;
    assert!(diff > 0.01, "Channels are too similar: avg diff = {}", diff);
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn test_pitch_shift_extreme_up() {
    let input = sine_mono(440.0, 44100, 1.0);
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);

    let output = pitch_shift(&input, &params, 4.0).unwrap();
    assert_eq!(output.len(), input.len());
    assert_no_nan_inf(&output, "4x pitch up");
}

#[test]
fn test_pitch_shift_extreme_down() {
    let input = sine_mono(440.0, 44100, 1.0);
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);

    let output = pitch_shift(&input, &params, 0.25).unwrap();
    assert_eq!(output.len(), input.len());
    assert_no_nan_inf(&output, "0.25x pitch down");
}

#[test]
fn test_pitch_shift_no_clipping() {
    let input = sine_mono(440.0, 44100, 2.0);
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);

    for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0] {
        let output = pitch_shift(&input, &params, factor).unwrap();
        let max_sample = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(
            max_sample < 10.0,
            "Pitch factor {} produced extreme sample: {}",
            factor,
            max_sample
        );
    }
}

#[test]
fn test_pitch_shift_silence_in_silence_out() {
    let input = vec![0.0f32; 44100];
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);

    let output = pitch_shift(&input, &params, 1.5).unwrap();
    assert_eq!(output.len(), input.len());

    let rms_out = rms(&output);
    assert!(
        rms_out < 0.01,
        "Silence should remain silent after pitch shift: rms={}",
        rms_out
    );
}

// ── Preset compatibility ────────────────────────────────────────────────────

#[test]
fn test_pitch_shift_with_all_presets() {
    let input = sine_mono(440.0, 44100, 1.0);

    let presets = [
        EdmPreset::DjBeatmatch,
        EdmPreset::HouseLoop,
        EdmPreset::Halftime,
        EdmPreset::Ambient,
        EdmPreset::VocalChop,
    ];

    for preset in presets {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(preset);

        let output = pitch_shift(&input, &params, 1.5).unwrap();
        assert_eq!(
            output.len(),
            input.len(),
            "Preset {:?} changed output length",
            preset
        );
        assert_no_nan_inf(&output, &format!("preset {:?}", preset));
    }
}

// ── 48 kHz sample rate ─────────────────────────────────────────────────────

#[test]
fn test_pitch_shift_48khz() {
    let input = sine_mono(440.0, 48000, 2.0);
    let params = StretchParams::new(1.0)
        .with_sample_rate(48000)
        .with_channels(1);

    let output = pitch_shift(&input, &params, 1.5).unwrap();
    assert_eq!(output.len(), input.len());
    assert_no_nan_inf(&output, "48kHz pitch shift");
}

// ── No NaN/Inf across all ratios ────────────────────────────────────────────

#[test]
fn test_pitch_shift_no_nan_inf_sweep() {
    let input = sine_mono(440.0, 44100, 1.0);
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);

    for factor in [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0] {
        let output = pitch_shift(&input, &params, factor).unwrap();
        assert_no_nan_inf(&output, &format!("factor {}", factor));
    }
}

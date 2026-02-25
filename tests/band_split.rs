//! Integration tests for sub-bass band-split processing.
//!
//! Verifies that `band_split` correctly separates sub-bass for independent
//! PV processing while the remainder goes through the hybrid algorithm.
//! Tests exercise the public `stretch()` API end-to-end.

use timestretch::{stretch, EdmPreset, StretchParams};

fn sine_wave(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|x| x * x).sum::<f32>() / signal.len() as f32).sqrt()
}

/// Creates a signal with both sub-bass and mid-range content,
/// similar to an EDM track with kick drum and synth pad.
fn edm_like_signal(sample_rate: u32, duration_secs: f64) -> Vec<f32> {
    let num_samples = (sample_rate as f64 * duration_secs) as usize;
    let beat_interval = (sample_rate as f64 * 0.5) as usize; // 120 BPM

    let mut signal = vec![0.0f32; num_samples];

    for (i, sample) in signal.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;

        // Sub-bass: 50 Hz sine (constant)
        *sample += 0.4 * (2.0 * std::f32::consts::PI * 50.0 * t).sin();

        // Mid-range synth pad: 400 Hz
        *sample += 0.3 * (2.0 * std::f32::consts::PI * 400.0 * t).sin();

        // Hi-hat: 6000 Hz
        *sample += 0.1 * (2.0 * std::f32::consts::PI * 6000.0 * t).sin();
    }

    // Add kick-like transients every beat
    for beat in 0..(num_samples / beat_interval) {
        let pos = beat * beat_interval;
        for j in 0..20.min(num_samples - pos) {
            signal[pos + j] += if j < 5 { 0.5 } else { -0.2 };
        }
    }

    // Clamp to [-1, 1]
    for s in &mut signal {
        *s = s.clamp(-1.0, 1.0);
    }

    signal
}

#[test]
fn test_band_split_enabled_by_default_with_presets() {
    // All EDM presets should enable either band_split or multi_resolution.
    // Presets with multi_resolution=true set band_split=false to avoid
    // redundant sub-bass processing paths.
    let presets = [
        EdmPreset::DjBeatmatch,
        EdmPreset::HouseLoop,
        EdmPreset::Halftime,
        EdmPreset::Ambient,
        EdmPreset::VocalChop,
    ];

    for preset in presets {
        let params = StretchParams::new(1.5).with_preset(preset);
        assert!(
            params.band_split || params.multi_resolution,
            "Preset {:?} should enable band_split or multi_resolution",
            preset
        );
    }
}

#[test]
fn test_band_split_disabled_by_default_without_preset() {
    let params = StretchParams::new(1.5);
    assert!(!params.band_split);
}

#[test]
fn test_band_split_can_be_toggled_after_preset() {
    let params = StretchParams::new(1.5)
        .with_preset(EdmPreset::HouseLoop)
        .with_band_split(false);
    assert!(!params.band_split);

    let params = StretchParams::new(1.5).with_band_split(true);
    assert!(params.band_split);
}

#[test]
fn test_band_split_stretch_edm_signal() {
    let sample_rate = 44100u32;
    let input = edm_like_signal(sample_rate, 2.0);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();

    // Output should be ~1.5x longer
    let ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (ratio - 1.5).abs() < 0.4,
        "EDM signal stretch ratio {} too far from 1.5",
        ratio
    );

    // No NaN/Inf
    assert!(
        output.iter().all(|s| s.is_finite()),
        "Output contains NaN/Inf"
    );

    // RMS should be preserved within a reasonable range
    let input_rms = rms(&input);
    let output_rms = rms(&output);
    let rms_ratio = output_rms / input_rms;
    assert!(
        (0.3..=2.0).contains(&rms_ratio),
        "RMS ratio {} out of range (input={}, output={})",
        rms_ratio,
        input_rms,
        output_rms
    );
}

#[test]
fn test_band_split_preserves_sub_bass_energy() {
    // A pure sub-bass signal should be well-preserved by band-split processing
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;
    let input = sine_wave(60.0, sample_rate, num_samples);
    let input_rms = rms(&input);

    // With band_split (via preset)
    let params_split = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);
    let output_split = stretch(&input, &params_split).unwrap();
    let output_rms_split = rms(&output_split);

    // Without band_split
    let params_no_split = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop)
        .with_band_split(false);
    let output_no_split = stretch(&input, &params_no_split).unwrap();
    let output_rms_no_split = rms(&output_no_split);

    // Both should preserve sub-bass energy reasonably
    assert!(
        output_rms_split > input_rms * 0.3,
        "Band-split sub-bass RMS {} too low (input={})",
        output_rms_split,
        input_rms
    );
    assert!(
        output_rms_no_split > input_rms * 0.3,
        "Non-split sub-bass RMS {} too low (input={})",
        output_rms_no_split,
        input_rms
    );
}

#[test]
fn test_band_split_preserves_high_freq_content() {
    // A 1000 Hz signal should pass through band-split processing intact
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;
    let input = sine_wave(1000.0, sample_rate, num_samples);
    let input_rms = rms(&input);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_band_split(true);

    let output = stretch(&input, &params).unwrap();
    let output_rms = rms(&output);

    // High-freq content should be preserved
    assert!(
        output_rms > input_rms * 0.2,
        "1 kHz signal RMS {} too low after band-split stretch (input={})",
        output_rms,
        input_rms
    );
}

#[test]
fn test_band_split_dj_beatmatch_small_ratio() {
    // DJ use case: 126 -> 128 BPM (ratio ~0.984)
    let sample_rate = 44100u32;
    let input = edm_like_signal(sample_rate, 2.0);

    let params = StretchParams::new(126.0 / 128.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let output = stretch(&input, &params).unwrap();

    // Small compression: output should be slightly shorter
    assert!(
        output.len() < input.len(),
        "126->128 BPM should produce shorter output"
    );

    // No clipping
    assert!(
        output.iter().all(|s| s.abs() <= 1.5),
        "Output exceeds ±1.5 (clipping)"
    );
}

#[test]
fn test_band_split_halftime_stretch() {
    // Halftime: 2x stretch
    let sample_rate = 44100u32;
    let input = edm_like_signal(sample_rate, 2.0);

    let params = StretchParams::new(2.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::Halftime);

    let output = stretch(&input, &params).unwrap();

    let ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (ratio - 2.0).abs() < 0.5,
        "Halftime stretch ratio {} too far from 2.0",
        ratio
    );

    assert!(
        output.iter().all(|s| s.is_finite()),
        "Output contains NaN/Inf"
    );
}

#[test]
fn test_band_split_compression() {
    // Compression: 0.75x (speed up from 120 to 160 BPM)
    let sample_rate = 44100u32;
    let input = edm_like_signal(sample_rate, 2.0);

    let params = StretchParams::new(0.75)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();

    assert!(
        output.len() < input.len(),
        "Compression should produce shorter output"
    );
    assert!(
        output.iter().all(|s| s.is_finite()),
        "Output contains NaN/Inf"
    );
}

#[test]
fn test_band_split_stereo() {
    // Stereo signal: both channels should work with band-split
    let sample_rate = 44100u32;
    let num_frames = sample_rate as usize * 2;
    let mut input = vec![0.0f32; num_frames * 2];

    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        // Left: sub-bass + mid
        input[i * 2] = 0.4 * (2.0 * std::f32::consts::PI * 50.0 * t).sin()
            + 0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        // Right: sub-bass + high
        input[i * 2 + 1] = 0.4 * (2.0 * std::f32::consts::PI * 50.0 * t).sin()
            + 0.2 * (2.0 * std::f32::consts::PI * 2000.0 * t).sin();
    }

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(2)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();

    assert!(!output.is_empty());
    assert_eq!(output.len() % 2, 0, "Stereo output must have even length");
    assert!(
        output.iter().all(|s| s.is_finite()),
        "Stereo output contains NaN/Inf"
    );
}

#[test]
fn test_band_split_48khz() {
    // Ensure band-split works at 48 kHz sample rate
    let sample_rate = 48000u32;
    let num_samples = sample_rate as usize * 2;
    let input = sine_wave(60.0, sample_rate, num_samples);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();

    assert!(!output.is_empty());
    assert!(
        output.iter().all(|s| s.is_finite()),
        "48 kHz output contains NaN/Inf"
    );
}

#[test]
fn test_band_split_ambient_extreme_stretch() {
    // Extreme stretch: 3x with Ambient preset
    let sample_rate = 44100u32;
    let input = edm_like_signal(sample_rate, 2.0);

    let params = StretchParams::new(3.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::Ambient);

    let output = stretch(&input, &params).unwrap();

    let ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (ratio - 3.0).abs() < 1.0,
        "Ambient 3x stretch ratio {} too far from 3.0",
        ratio
    );

    assert!(
        output.iter().all(|s| s.is_finite()),
        "Ambient output contains NaN/Inf"
    );
}

#[test]
fn test_band_split_vocal_chop_preset() {
    // VocalChop preset with a mid-range signal
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;
    let input = sine_wave(300.0, sample_rate, num_samples);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::VocalChop);

    let output = stretch(&input, &params).unwrap();

    assert!(!output.is_empty());
    assert!(
        output.iter().all(|s| s.is_finite()),
        "VocalChop output contains NaN/Inf"
    );

    let output_rms = rms(&output);
    let input_rms = rms(&input);
    assert!(
        output_rms > input_rms * 0.2,
        "VocalChop should preserve energy: input_rms={}, output_rms={}",
        input_rms,
        output_rms
    );
}

#[test]
fn test_band_split_with_custom_cutoff() {
    // Custom sub-bass cutoff (200 Hz instead of default 120 Hz)
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;

    // Mix of 100 Hz and 500 Hz
    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 100.0 * t).sin()
                + 0.5 * (2.0 * std::f32::consts::PI * 500.0 * t).sin()
        })
        .collect();

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_band_split(true)
        .with_sub_bass_cutoff(200.0);

    let output = stretch(&input, &params).unwrap();

    assert!(!output.is_empty());
    assert!(
        output.iter().all(|s| s.is_finite()),
        "Custom cutoff output contains NaN/Inf"
    );
}

#[test]
fn test_band_split_pitch_shift() {
    // Band-split should also work when used via pitch_shift
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;
    let input = sine_wave(440.0, sample_rate, num_samples);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = timestretch::pitch_shift(&input, &params, 1.5).unwrap();

    // Pitch shift preserves length
    assert_eq!(output.len(), input.len());
    assert!(
        output.iter().all(|s| s.is_finite()),
        "Pitch shift output contains NaN/Inf"
    );
}

use timestretch::{stretch, EdmPreset, StretchParams};

/// Helper to generate a mono sine wave.
fn sine_wave(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin()
        })
        .collect()
}

/// Helper to compute RMS of a signal.
fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|x| x * x).sum::<f32>() / signal.len() as f32).sqrt()
}

#[test]
fn test_identity_stretch_mono_440hz() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    // Length should be very close to input
    let len_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (len_ratio - 1.0).abs() < 0.15,
        "Identity stretch length ratio: {}",
        len_ratio
    );

    // RMS should be preserved
    let input_rms = rms(&input);
    let output_rms = rms(&output);
    assert!(
        (output_rms - input_rms).abs() < input_rms * 0.5,
        "RMS mismatch: input={}, output={}",
        input_rms,
        output_rms
    );
}

#[test]
fn test_identity_stretch_stereo() {
    let sample_rate = 44100;
    let num_frames = sample_rate as usize;
    let mut input = vec![0.0f32; num_frames * 2];

    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        input[i * 2] = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        input[i * 2 + 1] = (2.0 * std::f32::consts::PI * 880.0 * t).sin();
    }

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let output = stretch(&input, &params).unwrap();

    // Should maintain stereo interleaving
    assert_eq!(output.len() % 2, 0);

    let len_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (len_ratio - 1.0).abs() < 0.15,
        "Identity stereo stretch length ratio: {}",
        len_ratio
    );
}

#[test]
fn test_identity_stretch_48khz() {
    let sample_rate = 48000;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    let len_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (len_ratio - 1.0).abs() < 0.15,
        "Identity 48kHz stretch length ratio: {}",
        len_ratio
    );
}

#[test]
fn test_identity_all_presets() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let presets = [
        EdmPreset::DjBeatmatch,
        EdmPreset::HouseLoop,
        EdmPreset::Halftime,
        EdmPreset::Ambient,
        EdmPreset::VocalChop,
    ];

    for preset in &presets {
        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(*preset);

        let output = stretch(&input, &params).unwrap();
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.0).abs() < 0.2,
            "Identity with preset {:?}: length ratio {}",
            preset,
            len_ratio
        );
    }
}

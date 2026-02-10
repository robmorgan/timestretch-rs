use timestretch::{stretch, EdmPreset, StretchParams};

fn sine_wave(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

/// Generates a simple house-style pattern: kick + bass + hihat.
fn house_pattern(sample_rate: u32, beats: usize) -> Vec<f32> {
    let beat_samples = (60.0 / 128.0 * sample_rate as f64) as usize;
    let total_samples = beats * beat_samples;
    let mut signal = vec![0.0f32; total_samples];

    for beat in 0..beats {
        let offset = beat * beat_samples;

        // Kick: short 60Hz burst with decay
        for i in 0..500.min(total_samples - offset) {
            let t = i as f32 / sample_rate as f32;
            let env = (-t * 20.0).exp();
            signal[offset + i] += 0.8 * env * (2.0 * std::f32::consts::PI * 60.0 * t).sin();
        }

        // Hi-hat: noise burst on offbeats
        if beat % 2 == 1 {
            for i in 0..200.min(total_samples - offset) {
                let t = i as f32 / sample_rate as f32;
                let env = (-t * 50.0).exp();
                // Simple noise approximation using sine harmonics
                let noise = (12345.6 * t).sin() * (23456.7 * t).sin();
                signal[offset + i] += 0.3 * env * noise;
            }
        }
    }

    // Add background pad (220 Hz)
    for (i, sample) in signal.iter_mut().enumerate().take(total_samples) {
        let t = i as f32 / sample_rate as f32;
        *sample += 0.15 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
    }

    signal
}

#[test]
fn test_dj_beatmatch_preset() {
    let sample_rate = 44100;
    let input = house_pattern(sample_rate, 16);

    // 126 -> 128 BPM
    let ratio = 126.0 / 128.0;
    let params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    // Output should be shorter (we're speeding up)
    assert!(
        output.len() < input.len(),
        "Expected shorter output for speed-up"
    );
}

#[test]
fn test_house_loop_preset() {
    let sample_rate = 44100;
    let input = house_pattern(sample_rate, 16);

    let params = StretchParams::new(1.25)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    let len_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (len_ratio - 1.25).abs() < 0.5,
        "HouseLoop preset: ratio {} too far from 1.25",
        len_ratio
    );
}

#[test]
fn test_halftime_preset() {
    let sample_rate = 44100;
    let input = house_pattern(sample_rate, 8);

    let params = StretchParams::new(2.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::Halftime);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    // Output should be approximately 2x longer
    let len_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        len_ratio > 1.3,
        "Halftime: expected longer output, got ratio {}",
        len_ratio
    );
}

#[test]
fn test_ambient_preset() {
    let sample_rate = 44100;
    let input = sine_wave(220.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(3.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::Ambient);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    let len_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        len_ratio > 1.5,
        "Ambient: expected much longer output, got ratio {}",
        len_ratio
    );
}

#[test]
fn test_vocal_chop_preset() {
    let sample_rate = 44100;
    // Simulate a vocal chop: short, varied frequency content
    let num_samples = sample_rate as usize / 2; // 0.5 seconds
    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let env = (-(t - 0.1).powi(2) * 20.0).exp();
            env * (2.0 * std::f32::consts::PI * 400.0 * t).sin()
                + 0.3 * env * (2.0 * std::f32::consts::PI * 800.0 * t).sin()
        })
        .collect();

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::VocalChop);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());
}

#[test]
fn test_all_presets_produce_output() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let presets = [
        EdmPreset::DjBeatmatch,
        EdmPreset::HouseLoop,
        EdmPreset::Halftime,
        EdmPreset::Ambient,
        EdmPreset::VocalChop,
    ];

    let ratios = [0.75, 1.0, 1.5, 2.0];

    for &preset in &presets {
        for &ratio in &ratios {
            let params = StretchParams::new(ratio)
                .with_sample_rate(sample_rate)
                .with_channels(1)
                .with_preset(preset);

            let output = stretch(&input, &params).unwrap();
            assert!(
                !output.is_empty(),
                "Preset {:?} with ratio {} produced empty output",
                preset,
                ratio
            );
        }
    }
}

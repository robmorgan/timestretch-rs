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

fn generate_edm_loop(sample_rate: u32, bpm: f64, duration_secs: f64) -> Vec<f32> {
    let total_samples = (sample_rate as f64 * duration_secs) as usize;
    let samples_per_beat = (60.0 / bpm * sample_rate as f64) as usize;
    let hihat_interval = samples_per_beat / 2;

    let mut samples = vec![0.0f32; total_samples];
    for (i, sample) in samples.iter_mut().enumerate().take(total_samples) {
        let t = i as f32 / sample_rate as f32;
        *sample += (2.0 * std::f32::consts::PI * 55.0 * t).sin() * 0.2;
        *sample += (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.15;
    }

    let mut pos = 0;
    while pos < total_samples {
        let kick_dur = (0.04 * sample_rate as f64) as usize;
        for j in 0..kick_dur.min(total_samples - pos) {
            let t = j as f32 / sample_rate as f32;
            let decay = (-t * 50.0).exp();
            samples[pos + j] += (2.0 * std::f32::consts::PI * 60.0 * t).sin() * decay * 0.6;
        }
        pos += samples_per_beat;
    }

    let mut state = 42u64;
    pos = hihat_interval;
    while pos < total_samples {
        let hat_dur = (0.01 * sample_rate as f64) as usize;
        for j in 0..hat_dur.min(total_samples - pos) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let decay = (-(j as f32) / sample_rate as f32 * 200.0).exp();
            samples[pos + j] += noise * 0.1 * decay;
        }
        pos += hihat_interval;
    }

    for s in samples.iter_mut() {
        *s = s.clamp(-1.0, 1.0);
    }
    samples
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
fn test_dj_beatmatch_small_adjustments() {
    let input = generate_edm_loop(44100, 125.0, 0.5);
    let ratios = [0.98, 1.02, 1.06];

    for &ratio in &ratios {
        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::DjBeatmatch);
        let output = timestretch::stretch(&input, &params).unwrap();

        assert!(
            !output.is_empty(),
            "DJ beatmatch at ratio {ratio} should produce output"
        );
        let max = output.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(
            max > 0.01,
            "DJ beatmatch at ratio {ratio} should not be silent"
        );
    }
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
fn test_house_loop_compress() {
    let input = generate_edm_loop(44100, 125.0, 1.0);

    let params = StretchParams::new(0.8)
        .with_sample_rate(44100)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);
    let output = timestretch::stretch(&input, &params).unwrap();

    let len_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (len_ratio - 0.8).abs() < 0.35,
        "House loop 0.8x: got ratio {len_ratio}"
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
fn test_ambient_4x_stretch() {
    let input = sine_wave(440.0, 44100, 22050); // 0.5 sec

    let params = StretchParams::new(4.0)
        .with_sample_rate(44100)
        .with_channels(1)
        .with_preset(EdmPreset::Ambient);
    let output = timestretch::stretch(&input, &params).unwrap();

    assert!(!output.is_empty());
    let max = output.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.01, "4x ambient stretch should not be silent");
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

// ==================== STEREO PRESET TESTS ====================

#[test]
fn test_all_presets_stereo() {
    let sample_rate = 44100u32;
    let num_frames = 22050; // 0.5 sec
    let mut input = Vec::with_capacity(num_frames * 2);
    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        let l = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        let r = (2.0 * std::f32::consts::PI * 880.0 * t).sin();
        input.push(l);
        input.push(r);
    }

    let presets = [
        (EdmPreset::DjBeatmatch, 1.02),
        (EdmPreset::HouseLoop, 1.2),
        (EdmPreset::Ambient, 2.5),
        (EdmPreset::VocalChop, 1.3),
    ];

    for (preset, ratio) in &presets {
        let params = StretchParams::new(*ratio)
            .with_sample_rate(sample_rate)
            .with_channels(2)
            .with_preset(*preset);
        let output = timestretch::stretch(&input, &params).unwrap();

        assert_eq!(
            output.len() % 2,
            0,
            "Stereo output for {:?} must have even length",
            preset
        );
        assert!(
            !output.is_empty(),
            "Stereo {:?} at ratio {ratio} should produce output",
            preset
        );
    }
}

// ==================== 48 kHz TESTS ====================

#[test]
fn test_presets_48khz() {
    let input = sine_wave(440.0, 48000, 24000); // 0.5 sec

    let presets = [
        EdmPreset::DjBeatmatch,
        EdmPreset::HouseLoop,
        EdmPreset::Halftime,
        EdmPreset::Ambient,
        EdmPreset::VocalChop,
    ];

    for preset in &presets {
        let params = StretchParams::new(1.5)
            .with_sample_rate(48000)
            .with_channels(1)
            .with_preset(*preset);
        let output = timestretch::stretch(&input, &params).unwrap();

        assert!(
            !output.is_empty(),
            "Preset {:?} at 48kHz should produce output",
            preset
        );
    }
}

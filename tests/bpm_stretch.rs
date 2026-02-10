/// Integration tests for BPM-aware stretch API.
use std::f32::consts::PI;
use timestretch::{AudioBuffer, EdmPreset, StretchError, StretchParams};

fn generate_sine(freq: f32, sample_rate: u32, duration_secs: f64) -> Vec<f32> {
    let num_samples = (sample_rate as f64 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

/// Generate a click train at the given BPM (mono).
fn generate_click_train(bpm: f64, sample_rate: u32, duration_secs: f64) -> Vec<f32> {
    let num_samples = (sample_rate as f64 * duration_secs) as usize;
    let beat_interval = (60.0 * sample_rate as f64 / bpm) as usize;
    let mut samples = vec![0.0f32; num_samples];

    for pos in (0..num_samples).step_by(beat_interval) {
        // Short click: 10 samples of impulse
        for j in 0..10.min(num_samples - pos) {
            samples[pos + j] = if j < 5 { 0.9 } else { -0.4 };
        }
    }

    // Add a low-level tonal component so it's not pure clicks
    for (i, sample) in samples.iter_mut().enumerate() {
        *sample += 0.2 * (2.0 * PI * 200.0 * i as f32 / sample_rate as f32).sin();
    }

    samples
}

#[test]
fn test_bpm_stretch_126_to_128() {
    let sample_rate = 44100u32;
    let input = generate_sine(440.0, sample_rate, 2.0);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let output = timestretch::stretch_to_bpm(&input, 126.0, 128.0, &params).unwrap();

    // Expected ratio: 126/128 = 0.984375
    let expected_ratio = 126.0 / 128.0;
    let actual_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (actual_ratio - expected_ratio).abs() < 0.3,
        "126->128 BPM: expected ratio ~{:.4}, got {:.4}",
        expected_ratio,
        actual_ratio
    );
}

#[test]
fn test_bpm_stretch_128_to_126() {
    let sample_rate = 44100u32;
    let input = generate_sine(440.0, sample_rate, 2.0);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let output = timestretch::stretch_to_bpm(&input, 128.0, 126.0, &params).unwrap();

    // Slowing down: output should be longer
    assert!(
        output.len() > input.len(),
        "Slowing from 128 to 126 BPM should produce longer output"
    );
}

#[test]
fn test_bpm_stretch_halftime() {
    // 128 BPM -> 64 BPM = halftime (2x stretch)
    let sample_rate = 44100u32;
    let input = generate_sine(440.0, sample_rate, 2.0);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::Halftime);

    let output = timestretch::stretch_to_bpm(&input, 128.0, 64.0, &params).unwrap();

    // Ratio should be ~2.0
    let actual_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (actual_ratio - 2.0).abs() < 0.5,
        "Halftime (128->64): expected ratio ~2.0, got {:.4}",
        actual_ratio
    );
}

#[test]
fn test_bpm_stretch_doubletime() {
    // 128 BPM -> 256 BPM = doubletime (0.5x compression)
    let sample_rate = 44100u32;
    let input = generate_sine(440.0, sample_rate, 2.0);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = timestretch::stretch_to_bpm(&input, 128.0, 256.0, &params).unwrap();

    // Ratio should be ~0.5
    let actual_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (actual_ratio - 0.5).abs() < 0.3,
        "Doubletime (128->256): expected ratio ~0.5, got {:.4}",
        actual_ratio
    );
}

#[test]
fn test_bpm_stretch_preserves_rms_energy() {
    let sample_rate = 44100u32;
    let input = generate_sine(440.0, sample_rate, 2.0);
    let input_rms = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32).sqrt();

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let output = timestretch::stretch_to_bpm(&input, 126.0, 128.0, &params).unwrap();
    let output_rms = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();

    // RMS should be preserved within 50%
    assert!(
        (output_rms - input_rms).abs() < input_rms * 0.5,
        "RMS not preserved: input={:.4}, output={:.4}",
        input_rms,
        output_rms
    );
}

#[test]
fn test_bpm_stretch_no_nan_or_inf() {
    let sample_rate = 44100u32;
    let input = generate_sine(440.0, sample_rate, 2.0);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    // Test various BPM changes
    for (source, target) in &[(120.0, 128.0), (128.0, 90.0), (140.0, 70.0), (100.0, 160.0)] {
        let output = timestretch::stretch_to_bpm(&input, *source, *target, &params).unwrap();
        for (i, &sample) in output.iter().enumerate() {
            assert!(
                sample.is_finite(),
                "NaN/Inf at sample {} for {}->{}BPM",
                i,
                source,
                target
            );
        }
    }
}

#[test]
fn test_bpm_stretch_stereo() {
    let sample_rate = 44100u32;
    let num_frames = 44100 * 2;
    let mut input = vec![0.0f32; num_frames * 2];

    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        input[i * 2] = (2.0 * PI * 440.0 * t).sin();
        input[i * 2 + 1] = (2.0 * PI * 880.0 * t).sin();
    }

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(2)
        .with_preset(EdmPreset::DjBeatmatch);

    let output = timestretch::stretch_to_bpm(&input, 126.0, 128.0, &params).unwrap();
    assert!(!output.is_empty());
    assert_eq!(output.len() % 2, 0, "Stereo output must have even length");
}

#[test]
fn test_bpm_stretch_with_all_presets() {
    let sample_rate = 44100u32;
    let input = generate_sine(440.0, sample_rate, 2.0);

    for preset in &[
        EdmPreset::DjBeatmatch,
        EdmPreset::HouseLoop,
        EdmPreset::Halftime,
        EdmPreset::Ambient,
        EdmPreset::VocalChop,
    ] {
        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(*preset);

        let output = timestretch::stretch_to_bpm(&input, 126.0, 128.0, &params).unwrap();
        assert!(
            !output.is_empty(),
            "Preset {:?} produced empty output",
            preset
        );
    }
}

#[test]
fn test_bpm_stretch_buffer_api() {
    let sample_rate = 44100u32;
    let buffer = AudioBuffer::from_mono(generate_sine(440.0, sample_rate, 2.0), sample_rate);

    let params = StretchParams::new(1.0).with_preset(EdmPreset::DjBeatmatch);

    let output = timestretch::stretch_bpm_buffer(&buffer, 126.0, 128.0, &params).unwrap();
    assert_eq!(output.sample_rate, sample_rate);
    assert_eq!(output.channels, timestretch::Channels::Mono);
    assert!(!output.data.is_empty());
    // Speeding up: output shorter
    assert!(output.data.len() < buffer.data.len());
}

#[test]
fn test_bpm_stretch_auto_with_clicks() {
    // Generate a click train at 120 BPM and try auto-detection
    let sample_rate = 44100u32;
    let input = generate_click_train(120.0, sample_rate, 4.0);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let result = timestretch::stretch_to_bpm_auto(&input, 128.0, &params);

    // This may succeed or fail depending on beat detection sensitivity.
    // If it succeeds, output should be shorter (speeding up from ~120 to 128).
    // If it fails, it should fail with BpmDetectionFailed.
    match result {
        Ok(output) => {
            assert!(!output.is_empty());
            // Should produce output that's somewhat shorter or at least not hugely different
        }
        Err(StretchError::BpmDetectionFailed(_)) => {
            // Beat detection didn't find enough beats; acceptable for synthetic signals
        }
        Err(e) => panic!("Unexpected error: {}", e),
    }
}

#[test]
fn test_bpm_stretch_auto_invalid_target() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let input = vec![0.0f32; 44100];

    let result = timestretch::stretch_to_bpm_auto(&input, 0.0, &params);
    assert!(result.is_err());
    match result {
        Err(StretchError::BpmDetectionFailed(_)) => {} // expected
        other => panic!("Expected BpmDetectionFailed, got {:?}", other),
    }

    let result = timestretch::stretch_to_bpm_auto(&input, -128.0, &params);
    assert!(result.is_err());
}

#[test]
fn test_bpm_ratio_utility() {
    // Exact ratios
    assert!((timestretch::bpm_ratio(120.0, 120.0) - 1.0).abs() < 1e-10);
    assert!((timestretch::bpm_ratio(120.0, 60.0) - 2.0).abs() < 1e-10);
    assert!((timestretch::bpm_ratio(60.0, 120.0) - 0.5).abs() < 1e-10);

    // DJ-typical ratios
    let ratio = timestretch::bpm_ratio(126.0, 128.0);
    assert!((ratio - 0.984375).abs() < 1e-6);

    let ratio = timestretch::bpm_ratio(128.0, 130.0);
    assert!((ratio - 128.0 / 130.0).abs() < 1e-10);
}

#[test]
fn test_bpm_stretch_invalid_bpm_values() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let input = vec![0.0f32; 44100];

    // Zero BPMs
    assert!(timestretch::stretch_to_bpm(&input, 0.0, 128.0, &params).is_err());
    assert!(timestretch::stretch_to_bpm(&input, 128.0, 0.0, &params).is_err());

    // Negative BPMs
    assert!(timestretch::stretch_to_bpm(&input, -120.0, 128.0, &params).is_err());
    assert!(timestretch::stretch_to_bpm(&input, 120.0, -128.0, &params).is_err());
}

#[test]
fn test_bpm_stretch_48khz() {
    let sample_rate = 48000u32;
    let input = generate_sine(440.0, sample_rate, 2.0);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = timestretch::stretch_to_bpm(&input, 126.0, 128.0, &params).unwrap();
    assert!(!output.is_empty());
    assert!(output.len() < input.len());
}

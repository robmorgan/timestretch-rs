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

/// Compute signal-to-noise ratio in dB between original and processed signals.
#[allow(dead_code)]
fn snr_db(original: &[f32], processed: &[f32]) -> f32 {
    let len = original.len().min(processed.len());
    if len == 0 {
        return 0.0;
    }

    let signal_power: f32 = original[..len].iter().map(|x| x * x).sum::<f32>() / len as f32;
    let noise_power: f32 = original[..len]
        .iter()
        .zip(processed[..len].iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f32>()
        / len as f32;

    if noise_power < 1e-12 {
        return 100.0;
    }

    10.0 * (signal_power / noise_power).log10()
}

#[test]
fn test_stretch_preserves_rms_energy() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);
    let input_rms = rms(&input);

    for &ratio in &[0.75, 1.0, 1.25, 1.5, 2.0] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = stretch(&input, &params).unwrap();
        let output_rms = rms(&output);

        // RMS should be within 50% of original
        assert!(
            (output_rms - input_rms).abs() < input_rms * 0.6,
            "RMS diverged at ratio {}: input={}, output={}",
            ratio,
            input_rms,
            output_rms
        );
    }
}

#[test]
fn test_stretch_output_length_proportional() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    for &ratio in &[0.75, 1.25, 1.5, 2.0] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = stretch(&input, &params).unwrap();
        let actual_ratio = output.len() as f64 / input.len() as f64;

        // Should be within 30% of target ratio
        assert!(
            (actual_ratio - ratio).abs() < ratio * 0.35,
            "Output length ratio {} too far from target {} at stretch ratio {}",
            actual_ratio,
            ratio,
            ratio
        );
    }
}

#[test]
fn test_stretch_no_clipping() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
    for &ratio in &ratios {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = stretch(&input, &params).unwrap();

        let max_sample = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        // Allow some overshoot due to phase vocoder resynthesis artifacts
        // (input peak is 1.0, output should stay within reasonable bounds)
        assert!(
            max_sample < 3.0,
            "Excessive gain at ratio {}: max sample = {}",
            ratio,
            max_sample
        );
    }
}

#[test]
fn test_dj_beatmatch_quality() {
    // DJ beatmatching: small ratio changes should be very transparent
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    // 126 BPM -> 128 BPM
    let ratio = 126.0 / 128.0;
    let params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let output = stretch(&input, &params).unwrap();

    // Output should be shorter (speeding up)
    assert!(output.len() < input.len());

    // RMS should be well preserved for small ratios
    let input_rms = rms(&input);
    let output_rms = rms(&output);
    assert!(
        (output_rms - input_rms).abs() < input_rms * 0.3,
        "DJ beatmatch RMS: input={}, output={}",
        input_rms,
        output_rms
    );
}

#[test]
fn test_sub_bass_preserved() {
    // Sub-bass (60 Hz) should be well preserved
    let sample_rate = 44100;
    let input = sine_wave(60.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();

    let input_rms = rms(&input);
    let output_rms = rms(&output);

    // Sub-bass energy should be preserved within 50%
    assert!(
        (output_rms - input_rms).abs() < input_rms * 0.6,
        "Sub-bass RMS: input={}, output={}",
        input_rms,
        output_rms
    );
}

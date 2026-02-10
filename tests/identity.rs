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

/// Compute spectral energy at a target frequency using a DFT bin.
fn spectral_energy_at_freq(signal: &[f32], sample_rate: u32, target_freq: f32) -> f32 {
    let n = signal.len();
    if n == 0 {
        return 0.0;
    }
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut real = 0.0f64;
    let mut imag = 0.0f64;
    for (i, &s) in signal.iter().enumerate() {
        let angle = two_pi * target_freq * i as f32 / sample_rate as f32;
        real += s as f64 * angle.cos() as f64;
        imag += s as f64 * angle.sin() as f64;
    }
    ((real * real + imag * imag) / n as f64).sqrt() as f32
}

/// Compute SNR between a reference signal and test signal (in dB).
fn compute_snr_db(reference: &[f32], test: &[f32]) -> f64 {
    let len = reference.len().min(test.len());
    if len == 0 {
        return 0.0;
    }
    let signal_power: f64 = reference[..len].iter().map(|x| (*x as f64) * (*x as f64)).sum();
    let noise_power: f64 = reference[..len]
        .iter()
        .zip(test[..len].iter())
        .map(|(r, t)| {
            let diff = *r as f64 - *t as f64;
            diff * diff
        })
        .sum();
    if noise_power < 1e-20 {
        return 100.0; // Essentially identical
    }
    10.0 * (signal_power / noise_power).log10()
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

#[test]
fn test_identity_preserves_frequency_content() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    // Check that 440 Hz energy is present in output
    let input_energy = spectral_energy_at_freq(&input, sample_rate, 440.0);
    let output_energy = spectral_energy_at_freq(&output, sample_rate, 440.0);

    // Energy at target frequency should be within 50% of input
    assert!(
        output_energy > input_energy * 0.5,
        "440 Hz energy lost: input={}, output={}",
        input_energy,
        output_energy
    );

    // Energy at unrelated frequency (1234 Hz) should be low
    let noise_energy = spectral_energy_at_freq(&output, sample_rate, 1234.0);
    assert!(
        noise_energy < output_energy * 0.3,
        "Unexpected energy at 1234 Hz: signal={}, noise={}",
        output_energy,
        noise_energy
    );
}

#[test]
fn test_identity_multiple_frequencies() {
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;
    let pi = std::f32::consts::PI;

    // Mix of 200 Hz + 1000 Hz + 5000 Hz
    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.4 * (2.0 * pi * 200.0 * t).sin()
                + 0.3 * (2.0 * pi * 1000.0 * t).sin()
                + 0.2 * (2.0 * pi * 5000.0 * t).sin()
        })
        .collect();

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    // All three frequencies should be preserved
    for &freq in &[200.0, 1000.0, 5000.0] {
        let input_energy = spectral_energy_at_freq(&input, sample_rate, freq);
        let output_energy = spectral_energy_at_freq(&output, sample_rate, freq);
        assert!(
            output_energy > input_energy * 0.3,
            "Energy at {} Hz dropped too much: input={}, output={}",
            freq,
            input_energy,
            output_energy
        );
    }
}

#[test]
fn test_identity_sub_bass_coherence() {
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;

    // 60 Hz sub-bass (typical house music fundamental)
    let input = sine_wave(60.0, sample_rate, num_samples);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    // RMS should be well preserved for sub-bass
    let input_rms = rms(&input);
    let output_rms = rms(&output);
    assert!(
        (output_rms - input_rms).abs() < input_rms * 0.3,
        "Sub-bass RMS mismatch: input={}, output={}",
        input_rms,
        output_rms
    );

    // 60 Hz energy should dominate
    let energy_60 = spectral_energy_at_freq(&output, sample_rate, 60.0);
    let energy_120 = spectral_energy_at_freq(&output, sample_rate, 120.0);
    assert!(
        energy_60 > energy_120 * 2.0,
        "Sub-bass fundamental should dominate: 60Hz={}, 120Hz={}",
        energy_60,
        energy_120
    );
}

#[test]
fn test_identity_near_unity_ratios() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    // Test ratios very close to 1.0 (typical DJ adjustments)
    for &ratio in &[0.999, 0.995, 1.001, 1.005] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = stretch(&input, &params).unwrap();
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - ratio).abs() < 0.2,
            "Near-unity ratio {}: length ratio {}",
            ratio,
            len_ratio
        );
    }
}

#[test]
fn test_identity_snr() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    // Compare overlapping region
    let compare_len = input.len().min(output.len());
    // Skip the first/last few thousand samples to avoid edge effects
    let margin = 4096;
    if compare_len > margin * 2 {
        let snr = compute_snr_db(
            &input[margin..compare_len - margin],
            &output[margin..compare_len - margin],
        );
        // For identity stretch, SNR should be at least 10 dB
        assert!(
            snr > 10.0,
            "Identity SNR too low: {:.1} dB",
            snr
        );
    }
}

#[test]
fn test_identity_no_dc_offset() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    // A pure sine should have ~0 DC offset
    let dc_offset: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(
        dc_offset.abs() < 0.05,
        "Output has DC offset: {}",
        dc_offset
    );
}

#[test]
fn test_identity_click_train_timing() {
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;
    let mut input = vec![0.0f32; num_samples];

    // Place clicks at known positions
    let click_positions: Vec<usize> = (0..4)
        .map(|i| (i as f64 * 0.5 * sample_rate as f64) as usize)
        .filter(|&p| p + 20 < num_samples)
        .collect();

    for &pos in &click_positions {
        for j in 0..10 {
            input[pos + j] = if j < 3 { 0.9 } else { -0.4 };
        }
    }

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    // Find peaks in output - they should be near the original positions
    let threshold = 0.3;
    let mut output_peaks = Vec::new();
    for i in 1..output.len().saturating_sub(1) {
        if output[i] > threshold && output[i] > output[i - 1] && output[i] >= output[i + 1] {
            // Only add if far from previous peak
            if output_peaks.last().is_none_or(|&last: &usize| i - last > sample_rate as usize / 10) {
                output_peaks.push(i);
            }
        }
    }

    // We should find at least some of the clicks
    assert!(
        output_peaks.len() >= 2,
        "Expected at least 2 peaks in identity stretch of click train, found {}",
        output_peaks.len()
    );
}

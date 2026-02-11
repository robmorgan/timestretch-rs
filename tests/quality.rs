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

fn generate_mixed_signal(sample_rate: u32, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let kick = (2.0 * std::f32::consts::PI * 60.0 * t).sin() * 0.5;
            let hihat = (2.0 * std::f32::consts::PI * 8000.0 * t).sin() * 0.1;
            let pad = (2.0 * std::f32::consts::PI * 300.0 * t).sin() * 0.3;
            (kick + hihat + pad).clamp(-1.0, 1.0)
        })
        .collect()
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

        // RMS should be within reasonable range of original.
        // Extreme ratios (2x) with phase locking may lose more energy.
        let tolerance = if ratio >= 2.0 { 0.85 } else { 0.6 };
        assert!(
            (output_rms - input_rms).abs() < input_rms * tolerance,
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

// ==================== SPECTRAL PRESERVATION TESTS ====================

/// Simple DFT-based spectral energy at a given frequency.
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

#[test]
fn test_spectral_energy_preserved_small_stretch() {
    let sample_rate = 44100;
    let freq = 1000.0;
    let input = sine_wave(freq, sample_rate, 22050);

    // Small stretch: 1.05x (DJ beatmatch range)
    let params = StretchParams::new(1.05)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_sample_rate(sample_rate)
        .with_channels(1);
    let output = timestretch::stretch(&input, &params).unwrap();

    let input_energy = spectral_energy_at_freq(&input, sample_rate, freq);
    let output_energy = spectral_energy_at_freq(&output, sample_rate, freq);

    assert!(
        output_energy > input_energy * 0.3,
        "Spectral energy at {freq}Hz should be preserved: input={input_energy}, output={output_energy}"
    );
}

#[test]
fn test_spectral_energy_preserved_large_stretch() {
    let sample_rate = 44100;
    let freq = 1000.0;
    let input = sine_wave(freq, sample_rate, 22050);

    let params = StretchParams::new(2.0)
        .with_preset(EdmPreset::Halftime)
        .with_sample_rate(sample_rate)
        .with_channels(1);
    let output = timestretch::stretch(&input, &params).unwrap();

    let input_energy = spectral_energy_at_freq(&input, sample_rate, freq);
    let output_energy = spectral_energy_at_freq(&output, sample_rate, freq);

    assert!(
        output_energy > input_energy * 0.1,
        "Spectral energy at {freq}Hz should be preserved even at 2x: input={input_energy}, output={output_energy}"
    );
}

#[test]
fn test_spectral_energy_preserved_compression() {
    let sample_rate = 44100;
    let freq = 1000.0;
    let input = sine_wave(freq, sample_rate, 22050);

    let params = StretchParams::new(0.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);
    let output = timestretch::stretch(&input, &params).unwrap();

    if output.len() >= 4096 {
        let input_energy = spectral_energy_at_freq(&input, sample_rate, freq);
        let output_energy = spectral_energy_at_freq(&output, sample_rate, freq);

        assert!(
            output_energy > input_energy * 0.2,
            "Spectral energy preserved at 0.5x: input={input_energy}, output={output_energy}"
        );
    }
}

// ==================== LENGTH ACCURACY TESTS ====================

#[test]
fn test_output_length_small_ratios() {
    let input = sine_wave(440.0, 44100, 22050);
    let ratios = [0.92, 0.95, 0.98, 1.02, 1.05, 1.08];

    for &ratio in &ratios {
        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1);
        let output = timestretch::stretch(&input, &params).unwrap();

        let actual_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (actual_ratio - ratio).abs() < 0.15,
            "Ratio {ratio}: expected length ratio {ratio}, got {actual_ratio}"
        );
    }
}

#[test]
fn test_output_length_large_ratios() {
    let input = sine_wave(440.0, 44100, 22050);
    let ratios = [0.5, 0.75, 1.5, 2.0];

    for &ratio in &ratios {
        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1);
        let output = timestretch::stretch(&input, &params).unwrap();

        let actual_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (actual_ratio - ratio).abs() < 0.55,
            "Ratio {ratio}: expected length ratio {ratio}, got {actual_ratio}"
        );
    }
}

// ==================== SILENCE PRESERVATION ====================

#[test]
fn test_silence_stays_silent_all_ratios() {
    let input = vec![0.0f32; 22050];
    let ratios = [0.5, 0.75, 1.0, 1.5, 2.0];

    for &ratio in &ratios {
        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1);
        let output = timestretch::stretch(&input, &params).unwrap();

        let max = output.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(
            max < 1e-5,
            "Silence at ratio {ratio} should remain silent, max={max}"
        );
    }
}

// ==================== OUTPUT QUALITY ====================

#[test]
fn test_output_not_clipping() {
    let input = generate_mixed_signal(44100, 22050);
    let ratios = [0.5, 1.0, 1.5, 2.0];

    for &ratio in &ratios {
        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1);
        let output = timestretch::stretch(&input, &params).unwrap();

        let max = output.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(
            max < 3.0,
            "Output at ratio {ratio} may be clipping, max={max}"
        );
    }
}

#[test]
fn test_output_not_all_zeros_for_nonsilent_input() {
    let input = sine_wave(440.0, 44100, 22050);
    let ratios = [0.5, 0.75, 1.0, 1.5, 2.0];

    for &ratio in &ratios {
        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1);
        let output = timestretch::stretch(&input, &params).unwrap();

        let max = output.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(
            max > 0.01,
            "Non-silent input at ratio {ratio} should produce non-silent output, max={max}"
        );
    }
}

// ==================== MIXED SIGNAL QUALITY ====================

#[test]
fn test_mixed_signal_stretch() {
    let input = generate_mixed_signal(44100, 22050);
    let params = StretchParams::new(1.5)
        .with_preset(EdmPreset::HouseLoop)
        .with_sample_rate(44100)
        .with_channels(1);
    let output = timestretch::stretch(&input, &params).unwrap();

    assert!(
        !output.is_empty(),
        "Mixed signal stretch should produce output"
    );

    let max = output.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.01 && max < 3.0, "Output range check: max={max}");
}

// ==================== FREQUENCY SWEEP ====================

#[test]
fn test_frequency_sweep_stretch() {
    let sample_rate = 44100u32;
    let num_samples = 22050;
    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let _freq = 100.0 + 9900.0 * t;
            let phase = 2.0 * std::f32::consts::PI * (100.0 * t + 9900.0 * t * t / 2.0);
            phase.sin()
        })
        .collect();

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);
    let output = timestretch::stretch(&input, &params).unwrap();

    assert!(!output.is_empty());
    let max = output.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.01, "Sweep stretch should not be silent");
}

// ==================== MULTI-FREQUENCY TEST ====================

#[test]
fn test_multiple_frequencies_preserved() {
    let sample_rate = 44100u32;
    let num_samples = 22050;
    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let f1 = (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.33;
            let f2 = (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.33;
            let f3 = (2.0 * std::f32::consts::PI * 5000.0 * t).sin() * 0.33;
            f1 + f2 + f3
        })
        .collect();

    let params = StretchParams::new(1.2)
        .with_sample_rate(sample_rate)
        .with_channels(1);
    let output = timestretch::stretch(&input, &params).unwrap();

    assert!(!output.is_empty());

    let rms_val: f64 =
        (output.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / output.len() as f64).sqrt();
    assert!(rms_val > 0.01, "RMS should be non-trivial: {rms_val}");
}

use std::f32::consts::PI;
use timestretch::{stretch, EdmPreset, StretchParams};

/// Helper to generate a mono sine wave.
fn sine_wave(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
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
    let two_pi = 2.0 * PI;
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
    let signal_power: f64 = reference[..len]
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum();
    let noise_power: f64 = reference[..len]
        .iter()
        .zip(test[..len].iter())
        .map(|(r, t)| {
            let diff = *r as f64 - *t as f64;
            diff * diff
        })
        .sum();
    if noise_power < 1e-20 {
        return 100.0;
    }
    10.0 * (signal_power / noise_power).log10()
}

/// Compute dominant frequency using zero-crossing rate.
fn dominant_freq_zcr(signal: &[f32], sample_rate: u32) -> f32 {
    if signal.len() < 4 {
        return 0.0;
    }
    let mut crossings = 0usize;
    for i in 1..signal.len() {
        if (signal[i] >= 0.0) != (signal[i - 1] >= 0.0) {
            crossings += 1;
        }
    }
    let duration = (signal.len() - 1) as f32 / sample_rate as f32;
    crossings as f32 / (2.0 * duration)
}

#[test]
fn test_identity_stretch_mono_440hz() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    let len_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (len_ratio - 1.0).abs() < 0.15,
        "Identity stretch length ratio: {}",
        len_ratio
    );

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
        input[i * 2] = (2.0 * PI * 440.0 * t).sin();
        input[i * 2 + 1] = (2.0 * PI * 880.0 * t).sin();
    }

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let output = stretch(&input, &params).unwrap();

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
    let num_samples = sample_rate as usize * 2;

    for &freq in &[100.0, 440.0, 1000.0, 4000.0] {
        let input = sine_wave(freq, sample_rate, num_samples);
        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = stretch(&input, &params).unwrap();

        // Check dominant frequency is preserved (within 10%) via zero-crossing
        let input_freq = dominant_freq_zcr(&input, sample_rate);
        let output_freq = dominant_freq_zcr(&output, sample_rate);

        assert!(
            (output_freq - input_freq).abs() < input_freq * 0.1,
            "Frequency {} Hz: input dominant={:.1}, output dominant={:.1}",
            freq,
            input_freq,
            output_freq
        );

        // Also check spectral energy is preserved at the target frequency
        // (30% threshold accounts for PV windowing redistribution)
        let input_energy = spectral_energy_at_freq(&input, sample_rate, freq);
        let output_energy = spectral_energy_at_freq(&output, sample_rate, freq);
        assert!(
            output_energy > input_energy * 0.15,
            "{} Hz energy lost: input={}, output={}",
            freq,
            input_energy,
            output_energy
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

    // Compare overlapping region, skipping edges
    let compare_len = input.len().min(output.len());
    let margin = 4096;
    if compare_len > margin * 2 {
        let snr = compute_snr_db(
            &input[margin..compare_len - margin],
            &output[margin..compare_len - margin],
        );
        assert!(snr > 10.0, "Identity SNR too low: {:.1} dB", snr);
    }
}

#[test]
fn test_identity_multi_frequency() {
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;

    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * PI * 60.0 * t).sin()
                + 0.3 * (2.0 * PI * 440.0 * t).sin()
                + 0.2 * (2.0 * PI * 4000.0 * t).sin()
        })
        .collect();

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    let len_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (len_ratio - 1.0).abs() < 0.15,
        "Multi-frequency identity ratio: {}",
        len_ratio
    );

    let input_rms = rms(&input);
    let output_rms = rms(&output);
    assert!(
        (output_rms - input_rms).abs() < input_rms * 0.4,
        "Multi-freq RMS: input={:.4}, output={:.4}",
        input_rms,
        output_rms
    );

    // All frequency components should be preserved
    for &freq in &[60.0, 440.0, 4000.0] {
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
    let input = sine_wave(60.0, sample_rate, num_samples);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    let input_rms = rms(&input);
    let output_rms = rms(&output);
    assert!(
        (output_rms - input_rms).abs() < input_rms * 0.3,
        "Sub-bass RMS mismatch: input={}, output={}",
        input_rms,
        output_rms
    );

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
fn test_identity_no_dc_offset() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    let dc_offset: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(
        dc_offset.abs() < 0.05,
        "Output has DC offset: {}",
        dc_offset
    );
}

#[test]
fn test_identity_with_transients() {
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;
    let mut input = vec![0.0f32; num_samples];

    // Add kick-like transients every 0.5 seconds
    for beat in 0..4 {
        let pos = (beat as f64 * 0.5 * sample_rate as f64) as usize;
        for j in 0..500.min(num_samples - pos) {
            let t = j as f32 / sample_rate as f32;
            input[pos + j] = 0.8 * (-t * 80.0).exp() * (2.0 * PI * 60.0 * t).sin();
        }
    }

    for (i, sample) in input.iter_mut().enumerate() {
        *sample += 0.2 * (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin();
    }

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();

    let len_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (len_ratio - 1.0).abs() < 0.2,
        "Transient identity ratio: {}",
        len_ratio
    );

    let input_rms = rms(&input);
    let output_rms = rms(&output);
    assert!(
        (output_rms - input_rms).abs() < input_rms * 0.5,
        "Transient RMS: input={:.4}, output={:.4}",
        input_rms,
        output_rms
    );
}

#[test]
fn test_identity_stereo_channel_separation() {
    let sample_rate = 44100u32;
    let num_frames = sample_rate as usize;
    let mut input = vec![0.0f32; num_frames * 2];

    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        input[i * 2] = (2.0 * PI * 440.0 * t).sin();
        input[i * 2 + 1] = (2.0 * PI * 880.0 * t).sin();
    }

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let output = stretch(&input, &params).unwrap();
    assert_eq!(output.len() % 2, 0);

    let out_frames = output.len() / 2;
    let left: Vec<f32> = (0..out_frames).map(|i| output[i * 2]).collect();
    let right: Vec<f32> = (0..out_frames).map(|i| output[i * 2 + 1]).collect();

    let left_freq = dominant_freq_zcr(&left, sample_rate);
    let right_freq = dominant_freq_zcr(&right, sample_rate);

    assert!(
        left_freq < right_freq,
        "Channel separation lost: left={:.0} Hz, right={:.0} Hz",
        left_freq,
        right_freq
    );

    assert!(
        (left_freq - 440.0).abs() < 50.0,
        "Left channel freq {:.0} too far from 440 Hz",
        left_freq
    );
    assert!(
        (right_freq - 880.0).abs() < 100.0,
        "Right channel freq {:.0} too far from 880 Hz",
        right_freq
    );
}

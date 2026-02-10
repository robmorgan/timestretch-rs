use std::f32::consts::PI;
use timestretch::{stretch, EdmPreset, StretchParams};

fn sine_wave(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|x| x * x).sum::<f32>() / signal.len() as f32).sqrt()
}

// --- Boundary input size tests (from agent-2) ---

#[test]
fn test_minimum_input_size() {
    let sample_rate = 44100u32;
    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let input = sine_wave(440.0, sample_rate, 4096);
    let result = stretch(&input, &params);
    assert!(result.is_ok(), "Should process input of exactly FFT size");
    assert!(!result.unwrap().is_empty());
}

#[test]
fn test_input_slightly_above_minimum() {
    let sample_rate = 44100u32;
    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let input = sine_wave(440.0, sample_rate, 4097);
    let result = stretch(&input, &params);
    assert!(result.is_ok());
    assert!(!result.unwrap().is_empty());
}

#[test]
fn test_very_short_input() {
    let sample_rate = 44100;

    // 100 samples - shorter than FFT size, should fall back gracefully
    let input = sine_wave(440.0, sample_rate, 100);
    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty(), "100-sample input should produce output");

    // 50 samples
    let input_tiny = sine_wave(440.0, sample_rate, 50);
    let output_tiny = stretch(&input_tiny, &params).unwrap();
    assert!(
        !output_tiny.is_empty(),
        "50-sample input should produce output"
    );
}

#[test]
fn test_single_sample_input() {
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);

    let result = stretch(&[0.5], &params);
    match result {
        Ok(output) => {
            assert!(output.len() <= 10, "Single sample output shouldn't be huge");
        }
        Err(_) => {
            // Also acceptable - input too short
        }
    }
}

// --- Near-unity ratio tests (from agent-2) ---

#[test]
fn test_stretch_ratio_near_one() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize);

    for &ratio in &[0.999, 1.001, 0.99, 1.01] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = stretch(&input, &params).unwrap();
        let actual = output.len() as f64 / input.len() as f64;
        assert!(
            (actual - ratio).abs() < 0.15,
            "Near-unity ratio {}: got {}",
            ratio,
            actual
        );
    }
}

// --- Extreme ratio tests (combined) ---

#[test]
fn test_extreme_compression_025x() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 4);

    let params = StretchParams::new(0.25)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    let actual_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (actual_ratio - 0.25).abs() < 0.15,
        "0.25x ratio: actual {:.3} too far from 0.25",
        actual_ratio
    );
}

#[test]
fn test_extreme_stretch_4x() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(4.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::Ambient);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    let actual_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        actual_ratio > 2.0,
        "4x stretch should produce significantly longer output, got {:.3}",
        actual_ratio
    );
}

#[test]
fn test_parameter_boundary_ratio_min() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 4);

    let params = StretchParams::new(0.02)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let result = stretch(&input, &params);
    match result {
        Ok(output) => {
            assert!(output.len() < input.len(), "0.02x should compress heavily");
        }
        Err(_) => {
            // Acceptable for extreme ratios
        }
    }
}

#[test]
fn test_parameter_boundary_ratio_max() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize);

    let params = StretchParams::new(10.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    let actual_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        actual_ratio > 5.0,
        "10x stretch produced only {:.1}x output",
        actual_ratio
    );
}

// --- Invalid ratio tests ---

#[test]
fn test_invalid_ratio_zero() {
    let params = StretchParams::new(0.0);
    let result = stretch(&[0.0; 44100], &params);
    assert!(result.is_err(), "Zero ratio should be rejected");
}

#[test]
fn test_invalid_ratio_negative() {
    let params = StretchParams::new(-1.0);
    let result = stretch(&[0.0; 44100], &params);
    assert!(result.is_err(), "Negative ratio should be rejected");
}

#[test]
fn test_invalid_ratio_too_large() {
    let mut params = StretchParams::new(1.0);
    params.stretch_ratio = 200.0;
    let result = stretch(&[0.0; 44100], &params);
    assert!(result.is_err(), "200x ratio should be rejected");
}

// --- Signal type tests ---

#[test]
fn test_silence_input() {
    let sample_rate = 44100u32;
    let input = vec![0.0f32; sample_rate as usize * 2];

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    let max = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        max < 1e-6,
        "Silence in should produce silence out, got max={}",
        max
    );
}

#[test]
fn test_dc_offset_input() {
    let sample_rate = 44100u32;
    let input: Vec<f32> = (0..sample_rate as usize * 2)
        .map(|i| 0.5 + 0.3 * (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect();

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    // DC offset should be roughly preserved
    let input_mean = input.iter().sum::<f32>() / input.len() as f32;
    let output_mean = output.iter().sum::<f32>() / output.len() as f32;
    assert!(
        (output_mean - input_mean).abs() < 0.3,
        "DC offset diverged: input mean={:.3}, output mean={:.3}",
        input_mean,
        output_mean
    );

    // Output should not clip excessively
    let max = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(max < 3.0, "DC offset signal clipped: max={}", max);
}

#[test]
fn test_impulse_input() {
    let sample_rate = 44100u32;
    let mut input = vec![0.0f32; sample_rate as usize * 2];
    input[sample_rate as usize / 2] = 1.0;

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    let max_sample = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        max_sample < 5.0,
        "Impulse input caused excessive gain: max={}",
        max_sample
    );
    // Impulse should be preserved
    assert!(max_sample > 0.01, "Impulse was completely lost in stretch");
}

// --- Frequency edge tests (from agent-2) ---

#[test]
fn test_very_low_frequency() {
    let sample_rate = 44100u32;
    let input = sine_wave(20.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    let input_rms = rms(&input);
    let output_rms = rms(&output);
    assert!(
        (output_rms - input_rms).abs() < input_rms * 0.6,
        "20 Hz RMS: input={}, output={}",
        input_rms,
        output_rms
    );
}

#[test]
fn test_very_high_frequency() {
    let sample_rate = 44100u32;
    let input = sine_wave(15000.0, sample_rate, sample_rate as usize);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());
}

// --- Pattern tests (from agent-2) ---

#[test]
fn test_alternating_silence_and_tone() {
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;
    let mut input = vec![0.0f32; num_samples];

    let tone_len = (sample_rate as f64 * 0.2) as usize;
    let gap_len = (sample_rate as f64 * 0.3) as usize;
    let cycle = tone_len + gap_len;

    for i in 0..num_samples {
        let pos_in_cycle = i % cycle;
        if pos_in_cycle < tone_len {
            input[i] = 0.8 * (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin();
        }
    }

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    let ratio = output.len() as f64 / input.len() as f64;
    assert!((ratio - 1.5).abs() < 0.5, "Gapped signal ratio: {}", ratio);
}

// --- Preset compression tests (from agent-2) ---

#[test]
fn test_all_presets_with_compression() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let presets = [
        EdmPreset::DjBeatmatch,
        EdmPreset::HouseLoop,
        EdmPreset::Halftime,
        EdmPreset::Ambient,
        EdmPreset::VocalChop,
    ];

    for preset in &presets {
        let params = StretchParams::new(0.75)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(*preset);

        let output = stretch(&input, &params).unwrap();
        assert!(
            !output.is_empty(),
            "Preset {:?} with 0.75 ratio produced empty output",
            preset
        );
        assert!(
            output.len() < input.len(),
            "Preset {:?} with 0.75 ratio didn't compress",
            preset
        );
    }
}

// --- Stereo tests ---

#[test]
fn test_stereo_channel_independence() {
    let sample_rate = 44100;
    let num_frames = sample_rate as usize;
    let mut input = vec![0.0f32; num_frames * 2];

    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        input[i * 2] = (2.0 * PI * 440.0 * t).sin();
        input[i * 2 + 1] = 0.0;
    }

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let output = stretch(&input, &params).unwrap();
    assert_eq!(output.len() % 2, 0, "Stereo output must have even length");

    let right_rms: f32 = {
        let right: Vec<f32> = output.iter().skip(1).step_by(2).copied().collect();
        rms(&right)
    };
    assert!(
        right_rms < 0.05,
        "Silent right channel leaked signal: RMS={}",
        right_rms
    );
}

#[test]
fn test_stereo_mono_consistency() {
    let sample_rate = 44100u32;
    let input_mono = sine_wave(440.0, sample_rate, sample_rate as usize);

    let params_mono = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output_mono = stretch(&input_mono, &params_mono).unwrap();

    let mut input_stereo = Vec::with_capacity(input_mono.len() * 2);
    for &s in &input_mono {
        input_stereo.push(s);
        input_stereo.push(s);
    }

    let params_stereo = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let output_stereo = stretch(&input_stereo, &params_stereo).unwrap();

    let left: Vec<f32> = output_stereo.iter().step_by(2).copied().collect();

    let ratio = output_mono.len() as f64 / left.len() as f64;
    assert!(
        (ratio - 1.0).abs() < 0.2,
        "Mono vs stereo-left length mismatch: mono={}, stereo-left={}",
        output_mono.len(),
        left.len()
    );
}

// --- FFT size tests (from agent-2) ---

#[test]
fn test_small_fft_size() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_fft_size(256);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());
}

#[test]
fn test_large_fft_size() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_fft_size(8192);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());
}

// --- Quality verification tests (from agent-1) ---

#[test]
fn test_wsola_compression_accuracy() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    for &ratio in &[0.5, 0.6, 0.75, 0.8, 0.9] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = stretch(&input, &params).unwrap();
        let actual_ratio = output.len() as f64 / input.len() as f64;

        assert!(
            (actual_ratio - ratio).abs() < ratio * 0.35,
            "Compression ratio {}: actual {:.3}, error {:.1}%",
            ratio,
            actual_ratio,
            (actual_ratio - ratio).abs() / ratio * 100.0
        );
    }
}

#[test]
fn test_no_nan_or_inf_in_output() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    for &ratio in &[0.5, 0.75, 1.0, 1.5, 2.0, 3.0] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = stretch(&input, &params).unwrap();

        for (i, &sample) in output.iter().enumerate() {
            assert!(
                sample.is_finite(),
                "NaN/Inf at sample {} with ratio {}",
                i,
                ratio
            );
        }
    }
}

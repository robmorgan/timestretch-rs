use std::f32::consts::PI;
use timestretch::{stretch, EdmPreset, StretchParams};

fn sine_wave(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

#[test]
fn test_minimum_input_size() {
    // Test with input just barely large enough for processing
    let sample_rate = 44100u32;
    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    // FFT size is 4096 by default, so input must be at least that
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

    // FFT size + 1
    let input = sine_wave(440.0, sample_rate, 4097);
    let result = stretch(&input, &params);
    assert!(result.is_ok());
    assert!(!result.unwrap().is_empty());
}

#[test]
fn test_stretch_ratio_near_one() {
    // Ratios very close to 1.0 should still work correctly
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

#[test]
fn test_extreme_stretch_ratio() {
    // Large stretch ratios (4x)
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize);

    let params = StretchParams::new(4.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::Ambient);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    let ratio = output.len() as f64 / input.len() as f64;
    assert!(
        ratio > 2.0,
        "4x stretch should produce significantly longer output, got {}",
        ratio
    );
}

#[test]
fn test_extreme_compression_ratio() {
    // Small stretch ratios (0.25 = 4x speedup)
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(0.25)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());
    assert!(
        output.len() < input.len(),
        "0.25x should produce shorter output"
    );
}

#[test]
fn test_dc_offset_signal() {
    // Signal with a DC offset (not centered at zero)
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize;
    let input: Vec<f32> = (0..num_samples)
        .map(|i| 0.5 + 0.3 * (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect();

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    // Output should not clip excessively
    let max = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(max < 3.0, "DC offset signal clipped: max={}", max);
}

#[test]
fn test_silence_input() {
    // All-zero input should produce all-zero (or near-zero) output
    let sample_rate = 44100u32;
    let input = vec![0.0f32; sample_rate as usize];

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
fn test_very_low_frequency() {
    // 20 Hz sub-bass (near the edge of audibility)
    let sample_rate = 44100u32;
    let input = sine_wave(20.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    // RMS should be reasonable
    let input_rms: f32 =
        (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32).sqrt();
    let output_rms: f32 =
        (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
    assert!(
        (output_rms - input_rms).abs() < input_rms * 0.6,
        "20 Hz RMS: input={}, output={}",
        input_rms,
        output_rms
    );
}

#[test]
fn test_very_high_frequency() {
    // 15 kHz (near Nyquist for some signals)
    let sample_rate = 44100u32;
    let input = sine_wave(15000.0, sample_rate, sample_rate as usize);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());
}

#[test]
fn test_impulse_signal() {
    // Single impulse (Dirac-like) — tests transient handling
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize;
    let mut input = vec![0.0f32; num_samples];
    input[num_samples / 2] = 1.0; // Single sample impulse in the middle

    let params = StretchParams::new(2.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());

    // Output should contain some non-zero samples (the impulse should be preserved)
    let max = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(max > 0.01, "Impulse was completely lost in stretch");
}

#[test]
fn test_alternating_silence_and_tone() {
    // Signal with gaps (silence between tone bursts)
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;
    let mut input = vec![0.0f32; num_samples];

    // 200ms tone, 300ms silence, repeated
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

    // Output length should be approximately 1.5x
    let ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (ratio - 1.5).abs() < 0.5,
        "Gapped signal ratio: {}",
        ratio
    );
}

#[test]
fn test_all_presets_with_compression() {
    // Ensure all presets work with compression (not just stretching)
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

#[test]
fn test_stereo_mono_consistency() {
    // Mono processing of a single-channel signal should produce similar results
    // whether processed as mono or as a duplicate-stereo signal
    let sample_rate = 44100u32;
    let input_mono = sine_wave(440.0, sample_rate, sample_rate as usize);

    let params_mono = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output_mono = stretch(&input_mono, &params_mono).unwrap();

    // Create stereo with identical channels
    let mut input_stereo = Vec::with_capacity(input_mono.len() * 2);
    for &s in &input_mono {
        input_stereo.push(s);
        input_stereo.push(s);
    }

    let params_stereo = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let output_stereo = stretch(&input_stereo, &params_stereo).unwrap();

    // Extract left channel
    let left: Vec<f32> = output_stereo.iter().step_by(2).copied().collect();

    // Both should produce similar length output
    let mono_len = output_mono.len();
    let stereo_left_len = left.len();
    let ratio = mono_len as f64 / stereo_left_len as f64;
    assert!(
        (ratio - 1.0).abs() < 0.2,
        "Mono vs stereo-left length mismatch: mono={}, stereo-left={}",
        mono_len,
        stereo_left_len
    );
}

#[test]
fn test_small_fft_size() {
    // Test with a smaller FFT size (256)
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
    // Test with a larger FFT size (8192) for better frequency resolution
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_fft_size(8192);

    let output = stretch(&input, &params).unwrap();
    assert!(!output.is_empty());
}

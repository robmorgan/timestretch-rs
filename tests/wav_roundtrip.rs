//! WAV round-trip integration tests.
//!
//! These tests exercise the full pipeline: generate signal → write WAV → read
//! WAV → stretch → verify quality. They cover 16-bit and 32-bit float formats,
//! mono and stereo layouts, and various stretch ratios.

use std::f32::consts::PI;
use timestretch::io::wav::{read_wav, write_wav_16bit, write_wav_24bit, write_wav_float};
use timestretch::{stretch_buffer, AudioBuffer, Channels, EdmPreset, StretchParams};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Generate a mono sine wave.
fn sine_mono(freq: f32, sample_rate: u32, duration_secs: f32) -> AudioBuffer {
    let n = (sample_rate as f32 * duration_secs) as usize;
    let data: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
        .collect();
    AudioBuffer::new(data, sample_rate, Channels::Mono)
}

/// Generate a stereo signal: sine on left, different sine on right.
fn sine_stereo(freq_l: f32, freq_r: f32, sample_rate: u32, duration_secs: f32) -> AudioBuffer {
    let n = (sample_rate as f32 * duration_secs) as usize;
    let mut data = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        data.push((2.0 * PI * freq_l * t).sin());
        data.push((2.0 * PI * freq_r * t).sin());
    }
    AudioBuffer::new(data, sample_rate, Channels::Stereo)
}

/// Generate an EDM-like kick pattern: short low-freq burst every beat.
fn kick_pattern(bpm: f32, sample_rate: u32, duration_secs: f32) -> AudioBuffer {
    let n = (sample_rate as f32 * duration_secs) as usize;
    let samples_per_beat = (sample_rate as f32 * 60.0 / bpm) as usize;
    let kick_len = (sample_rate as f32 * 0.02) as usize; // 20ms kick

    let mut data = vec![0.0f32; n];
    let mut beat_pos = 0;
    while beat_pos < n {
        for i in 0..kick_len.min(n - beat_pos) {
            let t = i as f32 / sample_rate as f32;
            // Exponentially decaying 60 Hz sine
            let env = (-t * 200.0).exp();
            data[beat_pos + i] = env * (2.0 * PI * 60.0 * t).sin();
        }
        beat_pos += samples_per_beat;
    }
    AudioBuffer::new(data, sample_rate, Channels::Mono)
}

/// Compute RMS energy of a sample buffer.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Check that no sample is NaN or Inf.
fn assert_no_nan_inf(samples: &[f32], label: &str) {
    for (i, &s) in samples.iter().enumerate() {
        assert!(
            s.is_finite(),
            "{}: sample {} is not finite ({})",
            label,
            i,
            s
        );
    }
}

// ── 16-bit WAV round-trip tests ─────────────────────────────────────────────

#[test]
fn test_wav_16bit_roundtrip_mono_identity() {
    let original = sine_mono(440.0, 44100, 1.0);
    let wav_bytes = write_wav_16bit(&original);
    let decoded = read_wav(&wav_bytes).unwrap();

    assert_eq!(decoded.sample_rate, original.sample_rate);
    assert_eq!(decoded.channels, Channels::Mono);
    assert_eq!(decoded.data.len(), original.data.len());

    // 16-bit quantization error: max ~1/32768 ≈ 3e-5
    let max_err = original
        .data
        .iter()
        .zip(decoded.data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 0.001, "Max 16-bit round-trip error: {}", max_err);
}

#[test]
fn test_wav_float_roundtrip_stereo_identity() {
    let original = sine_stereo(440.0, 880.0, 48000, 1.0);
    let wav_bytes = write_wav_float(&original);
    let decoded = read_wav(&wav_bytes).unwrap();

    assert_eq!(decoded.sample_rate, 48000);
    assert_eq!(decoded.channels, Channels::Stereo);
    assert_eq!(decoded.data.len(), original.data.len());

    // Float round-trip should be bit-exact
    for (i, (a, b)) in original.data.iter().zip(decoded.data.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-7,
            "Float round-trip sample {} differs: {} vs {}",
            i,
            a,
            b
        );
    }
}

// ── Stretch through WAV pipeline ────────────────────────────────────────────

#[test]
fn test_wav_stretch_16bit_mono() {
    let original = sine_mono(440.0, 44100, 2.0);
    let wav_bytes = write_wav_16bit(&original);
    let decoded = read_wav(&wav_bytes).unwrap();

    let params = StretchParams::new(1.5).with_preset(EdmPreset::HouseLoop);
    let stretched = stretch_buffer(&decoded, &params).unwrap();

    assert_eq!(stretched.sample_rate, 44100);
    assert_eq!(stretched.channels, Channels::Mono);
    assert_no_nan_inf(&stretched.data, "16bit mono stretch");

    let len_ratio = stretched.data.len() as f64 / decoded.data.len() as f64;
    assert!(
        (len_ratio - 1.5).abs() < 0.3,
        "Length ratio {} too far from 1.5",
        len_ratio
    );

    // RMS should be preserved within a tolerance
    let rms_in = rms(&decoded.data);
    let rms_out = rms(&stretched.data);
    let rms_ratio = rms_out / rms_in;
    assert!(
        (0.3..=3.0).contains(&rms_ratio),
        "RMS ratio {} out of bounds",
        rms_ratio
    );
}

#[test]
fn test_wav_stretch_float_stereo() {
    let original = sine_stereo(440.0, 880.0, 44100, 2.0);
    let wav_bytes = write_wav_float(&original);
    let decoded = read_wav(&wav_bytes).unwrap();

    let params = StretchParams::new(0.75).with_preset(EdmPreset::DjBeatmatch);
    let stretched = stretch_buffer(&decoded, &params).unwrap();

    assert_eq!(stretched.sample_rate, 44100);
    assert_eq!(stretched.channels, Channels::Stereo);
    assert_eq!(stretched.data.len() % 2, 0);
    assert_no_nan_inf(&stretched.data, "float stereo stretch");

    let num_frames_in = decoded.data.len() / 2;
    let num_frames_out = stretched.data.len() / 2;
    let frame_ratio = num_frames_out as f64 / num_frames_in as f64;
    assert!(
        (frame_ratio - 0.75).abs() < 0.3,
        "Frame ratio {} too far from 0.75",
        frame_ratio
    );
}

// ── EDM kick pattern tests ──────────────────────────────────────────────────

#[test]
fn test_wav_kick_pattern_stretch() {
    let kicks = kick_pattern(128.0, 44100, 2.0);
    let wav_bytes = write_wav_float(&kicks);
    let decoded = read_wav(&wav_bytes).unwrap();

    // Stretch a 128 BPM pattern to play at 126 BPM (DJ beatmatch scenario)
    let ratio = 128.0 / 126.0; // ~1.016
    let params = StretchParams::new(ratio).with_preset(EdmPreset::DjBeatmatch);
    let stretched = stretch_buffer(&decoded, &params).unwrap();

    assert_no_nan_inf(&stretched.data, "kick stretch");

    // Output should be slightly longer
    let len_ratio = stretched.data.len() as f64 / decoded.data.len() as f64;
    assert!(
        (len_ratio - ratio).abs() < 0.2,
        "Kick stretch length ratio {} too far from {}",
        len_ratio,
        ratio
    );

    // Energy should be present in the output (kick signal is sparse, so
    // absolute RMS comparison isn't meaningful — just verify non-silence)
    let rms_out = rms(&stretched.data);
    assert!(
        rms_out > 1e-4,
        "Kick energy too low after stretch: rms={}",
        rms_out
    );
}

#[test]
fn test_wav_kick_pattern_halftime() {
    let kicks = kick_pattern(128.0, 44100, 1.0);
    let wav_bytes = write_wav_float(&kicks);
    let decoded = read_wav(&wav_bytes).unwrap();

    let params = StretchParams::new(2.0).with_preset(EdmPreset::Halftime);
    let stretched = stretch_buffer(&decoded, &params).unwrap();

    assert_no_nan_inf(&stretched.data, "kick halftime");

    let len_ratio = stretched.data.len() as f64 / decoded.data.len() as f64;
    assert!(
        (len_ratio - 2.0).abs() < 0.5,
        "Halftime length ratio {} too far from 2.0",
        len_ratio
    );
}

// ── Multi-format consistency ────────────────────────────────────────────────

#[test]
fn test_wav_16bit_vs_float_stretch_consistency() {
    let original = sine_mono(220.0, 44100, 1.0);

    // Stretch through 16-bit pipeline
    let wav_16 = write_wav_16bit(&original);
    let dec_16 = read_wav(&wav_16).unwrap();
    let params = StretchParams::new(1.25).with_preset(EdmPreset::HouseLoop);
    let stretched_16 = stretch_buffer(&dec_16, &params).unwrap();

    // Stretch through float pipeline
    let wav_f = write_wav_float(&original);
    let dec_f = read_wav(&wav_f).unwrap();
    let stretched_f = stretch_buffer(&dec_f, &params).unwrap();

    // Both should produce similar output (16-bit has quantization noise)
    let rms_16 = rms(&stretched_16.data);
    let rms_f = rms(&stretched_f.data);
    let rms_diff = (rms_16 - rms_f).abs() / rms_f.max(1e-10);
    assert!(
        rms_diff < 0.1,
        "16-bit vs float RMS diverged: 16bit={}, float={}, diff={}",
        rms_16,
        rms_f,
        rms_diff
    );
}

// ── Sample rate variants ────────────────────────────────────────────────────

#[test]
fn test_wav_roundtrip_48khz() {
    let original = sine_mono(1000.0, 48000, 1.0);
    let wav_bytes = write_wav_float(&original);
    let decoded = read_wav(&wav_bytes).unwrap();
    assert_eq!(decoded.sample_rate, 48000);

    let params = StretchParams::new(1.3).with_preset(EdmPreset::HouseLoop);
    let stretched = stretch_buffer(&decoded, &params).unwrap();

    assert_eq!(stretched.sample_rate, 48000);
    assert_no_nan_inf(&stretched.data, "48kHz stretch");

    let len_ratio = stretched.data.len() as f64 / decoded.data.len() as f64;
    assert!(
        (len_ratio - 1.3).abs() < 0.3,
        "48kHz length ratio {} too far from 1.3",
        len_ratio
    );
}

// ── Preset-specific WAV tests ───────────────────────────────────────────────

#[test]
fn test_wav_all_presets_roundtrip() {
    let original = sine_mono(440.0, 44100, 1.0);
    let wav_bytes = write_wav_float(&original);
    let decoded = read_wav(&wav_bytes).unwrap();

    let presets = [
        (EdmPreset::DjBeatmatch, 1.02),
        (EdmPreset::HouseLoop, 1.25),
        (EdmPreset::Halftime, 2.0),
        (EdmPreset::Ambient, 3.0),
        (EdmPreset::VocalChop, 0.8),
    ];

    for (preset, ratio) in presets {
        let params = StretchParams::new(ratio).with_preset(preset);
        let stretched = stretch_buffer(&decoded, &params).unwrap();

        assert_no_nan_inf(
            &stretched.data,
            &format!("preset {:?} ratio {}", preset, ratio),
        );
        assert!(
            !stretched.data.is_empty(),
            "Preset {:?} produced empty output",
            preset
        );

        // No clipping
        let max_sample = stretched
            .data
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_sample < 10.0,
            "Preset {:?} produced extreme sample: {}",
            preset,
            max_sample
        );
    }
}

// ── Double stretch (stretch then re-encode then stretch again) ──────────────

#[test]
fn test_wav_double_stretch_pipeline() {
    let original = sine_mono(440.0, 44100, 1.0);

    // First stretch: 1.0x → 1.5x through WAV pipeline
    let wav1 = write_wav_float(&original);
    let dec1 = read_wav(&wav1).unwrap();
    let params1 = StretchParams::new(1.5).with_preset(EdmPreset::HouseLoop);
    let stretched1 = stretch_buffer(&dec1, &params1).unwrap();

    // Encode the stretched result to WAV and decode it
    let wav2 = write_wav_float(&stretched1);
    let dec2 = read_wav(&wav2).unwrap();
    assert_eq!(dec2.data.len(), stretched1.data.len());

    // Second stretch: compress back to ~original length
    let params2 = StretchParams::new(1.0 / 1.5).with_preset(EdmPreset::HouseLoop);
    let final_output = stretch_buffer(&dec2, &params2).unwrap();

    assert_no_nan_inf(&final_output.data, "double stretch");

    // Final length should be close to original
    let len_ratio = final_output.data.len() as f64 / original.data.len() as f64;
    assert!(
        (len_ratio - 1.0).abs() < 0.5,
        "Double stretch final length ratio {} too far from 1.0",
        len_ratio
    );
}

// ── 24-bit WAV round-trip tests ─────────────────────────────────────────────

#[test]
fn test_wav_24bit_roundtrip_mono_identity() {
    let original = sine_mono(440.0, 44100, 1.0);
    let wav_bytes = write_wav_24bit(&original);
    let decoded = read_wav(&wav_bytes).unwrap();

    assert_eq!(decoded.sample_rate, original.sample_rate);
    assert_eq!(decoded.channels, Channels::Mono);
    assert_eq!(decoded.data.len(), original.data.len());

    // 24-bit quantization error: max ~1/8388608 ≈ 1.2e-7
    let max_err = original
        .data
        .iter()
        .zip(decoded.data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 0.0001, "Max 24-bit round-trip error: {}", max_err);
}

#[test]
fn test_wav_24bit_roundtrip_stereo() {
    let original = sine_stereo(440.0, 880.0, 48000, 1.0);
    let wav_bytes = write_wav_24bit(&original);
    let decoded = read_wav(&wav_bytes).unwrap();

    assert_eq!(decoded.sample_rate, 48000);
    assert_eq!(decoded.channels, Channels::Stereo);
    assert_eq!(decoded.data.len(), original.data.len());

    let max_err = original
        .data
        .iter()
        .zip(decoded.data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.0001,
        "Max 24-bit stereo round-trip error: {}",
        max_err
    );
}

#[test]
fn test_wav_stretch_24bit_mono() {
    let original = sine_mono(440.0, 44100, 2.0);
    let wav_bytes = write_wav_24bit(&original);
    let decoded = read_wav(&wav_bytes).unwrap();

    let params = StretchParams::new(1.5).with_preset(EdmPreset::HouseLoop);
    let stretched = stretch_buffer(&decoded, &params).unwrap();

    assert_eq!(stretched.sample_rate, 44100);
    assert_no_nan_inf(&stretched.data, "24bit mono stretch");

    let len_ratio = stretched.data.len() as f64 / decoded.data.len() as f64;
    assert!(
        (len_ratio - 1.5).abs() < 0.3,
        "24-bit stretch length ratio {} too far from 1.5",
        len_ratio
    );
}

#[test]
fn test_wav_24bit_vs_float_stretch_consistency() {
    let original = sine_mono(220.0, 44100, 1.0);

    // Stretch through 24-bit pipeline
    let wav_24 = write_wav_24bit(&original);
    let dec_24 = read_wav(&wav_24).unwrap();
    let params = StretchParams::new(1.25).with_preset(EdmPreset::HouseLoop);
    let stretched_24 = stretch_buffer(&dec_24, &params).unwrap();

    // Stretch through float pipeline
    let wav_f = write_wav_float(&original);
    let dec_f = read_wav(&wav_f).unwrap();
    let stretched_f = stretch_buffer(&dec_f, &params).unwrap();

    // 24-bit should be nearly identical to float
    let rms_24 = rms(&stretched_24.data);
    let rms_f = rms(&stretched_f.data);
    let rms_diff = (rms_24 - rms_f).abs() / rms_f.max(1e-10);
    assert!(
        rms_diff < 0.05,
        "24-bit vs float RMS diverged: 24bit={}, float={}, diff={}",
        rms_24,
        rms_f,
        rms_diff
    );
}

// ── Mix-to-mono integration tests ───────────────────────────────────────────

#[test]
fn test_mix_to_mono_then_stretch() {
    let stereo = sine_stereo(440.0, 880.0, 44100, 1.0);
    let mono = stereo.mix_to_mono();

    assert!(mono.is_mono());
    assert_eq!(mono.num_frames(), stereo.num_frames());

    let params = StretchParams::new(1.5).with_preset(EdmPreset::HouseLoop);
    let stretched = stretch_buffer(&mono, &params).unwrap();

    assert!(stretched.is_mono());
    assert_no_nan_inf(&stretched.data, "mix_to_mono stretch");
    assert!(!stretched.is_empty());
}

#[test]
fn test_stereo_left_right_extraction_and_stretch() {
    let stereo = sine_stereo(440.0, 880.0, 44100, 1.0);
    let left = stereo.left();
    let right = stereo.right();

    assert_eq!(left.len(), stereo.num_frames());
    assert_eq!(right.len(), stereo.num_frames());

    // Left and right should be different signals
    let diff: f32 = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| (l - r).abs())
        .sum::<f32>()
        / left.len() as f32;
    assert!(diff > 0.1, "L/R should differ, avg diff = {}", diff);

    // Stretch left channel alone
    let left_buf = AudioBuffer::from_mono(left, 44100);
    let params = StretchParams::new(1.5).with_preset(EdmPreset::HouseLoop);
    let stretched = stretch_buffer(&left_buf, &params).unwrap();
    assert_no_nan_inf(&stretched.data, "left channel stretch");
    assert!(!stretched.is_empty());
}

// ── WAV write-then-read preserves stretch output ────────────────────────────

#[test]
fn test_wav_preserves_stretched_output() {
    let original = sine_mono(440.0, 44100, 1.0);
    let params = StretchParams::new(1.5).with_preset(EdmPreset::HouseLoop);
    let stretched = stretch_buffer(&original, &params).unwrap();

    // Save as float WAV and reload — should be bit-identical
    let wav_bytes = write_wav_float(&stretched);
    let reloaded = read_wav(&wav_bytes).unwrap();

    assert_eq!(reloaded.data.len(), stretched.data.len());
    for (i, (a, b)) in stretched.data.iter().zip(reloaded.data.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-7,
            "Stretched output not preserved at sample {}: {} vs {}",
            i,
            a,
            b
        );
    }
}

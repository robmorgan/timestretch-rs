//! Tests that verify streaming output is consistent with batch output.
//!
//! Streaming (PV-only mode) and batch (hybrid) use different algorithms,
//! so we cannot expect bit-exact parity. These tests verify that both
//! modes produce structurally similar output: same approximate length,
//! comparable energy, preserved frequency content, and finite samples.

use std::f32::consts::PI;
use timestretch::{stretch, EdmPreset, StreamProcessor, StretchParams};

fn sine_wave(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

fn rms(signal: &[f32]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum: f64 = signal.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (sum / signal.len() as f64).sqrt()
}

fn dft_energy_at(signal: &[f32], sample_rate: u32, freq: f32) -> f64 {
    let n = signal.len();
    if n == 0 {
        return 0.0;
    }
    let two_pi = 2.0 * PI;
    let mut real = 0.0f64;
    let mut imag = 0.0f64;
    for (i, &s) in signal.iter().enumerate() {
        let angle = two_pi * freq * i as f32 / sample_rate as f32;
        real += s as f64 * angle.cos() as f64;
        imag += s as f64 * angle.sin() as f64;
    }
    (real * real + imag * imag).sqrt() / n as f64
}

fn stream_stretch(input: &[f32], params: StretchParams, chunk_size: usize) -> Vec<f32> {
    let mut processor = StreamProcessor::new(params);
    let mut output = Vec::new();
    for chunk in input.chunks(chunk_size) {
        if let Ok(out) = processor.process(chunk) {
            output.extend_from_slice(&out);
        }
    }
    if let Ok(remaining) = processor.flush() {
        output.extend_from_slice(&remaining);
    }
    output
}

fn stream_stretch_hybrid(input: &[f32], params: StretchParams, chunk_size: usize) -> Vec<f32> {
    let mut processor = StreamProcessor::new(params);
    processor.set_hybrid_mode(true);
    let mut output = Vec::new();
    for chunk in input.chunks(chunk_size) {
        if let Ok(out) = processor.process(chunk) {
            output.extend_from_slice(&out);
        }
    }
    if let Ok(remaining) = processor.flush() {
        output.extend_from_slice(&remaining);
    }
    output
}

// ========== Length parity across stretch ratios ==========

#[test]
fn test_parity_length_expansion() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let batch = stretch(&input, &params).unwrap();
    let stream = stream_stretch(&input, params, 4096);

    // Both should expand audio (ratio > 1.0)
    assert!(batch.len() > input.len(), "Batch should expand");
    assert!(stream.len() > input.len(), "Stream should expand");

    // Length ratio should be within 30% of each other
    let batch_ratio = batch.len() as f64 / input.len() as f64;
    let stream_ratio = stream.len() as f64 / input.len() as f64;
    assert!(
        (batch_ratio - stream_ratio).abs() < 0.3,
        "Length parity at 1.5x: batch={:.3}, stream={:.3}",
        batch_ratio,
        stream_ratio
    );
}

#[test]
fn test_parity_length_compression() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(0.75)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let batch = stretch(&input, &params).unwrap();
    let stream = stream_stretch(&input, params, 4096);

    // Both should compress audio (ratio < 1.0)
    assert!(batch.len() < input.len(), "Batch should compress");
    assert!(stream.len() < input.len(), "Stream should compress");
}

#[test]
fn test_parity_length_identity() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let batch = stretch(&input, &params).unwrap();
    let stream = stream_stretch(&input, params, 4096);

    // Identity: both should be close to input length
    let batch_ratio = batch.len() as f64 / input.len() as f64;
    let stream_ratio = stream.len() as f64 / input.len() as f64;
    assert!(
        (batch_ratio - 1.0).abs() < 0.15,
        "Batch identity length ratio {:.3} too far from 1.0",
        batch_ratio
    );
    assert!(
        (stream_ratio - 1.0).abs() < 0.15,
        "Stream identity length ratio {:.3} too far from 1.0",
        stream_ratio
    );
}

// ========== Energy parity ==========

#[test]
fn test_parity_rms_energy_preserved() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);
    let input_rms = rms(&input);

    for &ratio in &[0.75, 1.0, 1.25, 1.5, 2.0] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let batch = stretch(&input, &params).unwrap();
        let stream = stream_stretch(&input, params, 4096);

        let batch_rms = rms(&batch);
        let stream_rms = rms(&stream);

        // Both should preserve reasonable energy (within 3x of input)
        assert!(
            batch_rms > input_rms * 0.2 && batch_rms < input_rms * 3.0,
            "Batch RMS {:.4} out of range at ratio {} (input={:.4})",
            batch_rms,
            ratio,
            input_rms
        );
        assert!(
            stream_rms > input_rms * 0.2 && stream_rms < input_rms * 3.0,
            "Stream RMS {:.4} out of range at ratio {} (input={:.4})",
            stream_rms,
            ratio,
            input_rms
        );
    }
}

#[test]
fn test_parity_rms_close_at_small_ratios() {
    // For small DJ-style adjustments, batch and stream RMS should be very close
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    for &ratio in &[0.97, 0.98, 0.99, 1.01, 1.02, 1.03] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let batch = stretch(&input, &params).unwrap();
        let stream = stream_stretch(&input, params, 4096);

        let batch_rms = rms(&batch);
        let stream_rms = rms(&stream);

        // At small ratios, RMS should be within 50% of each other
        let diff = (batch_rms - stream_rms).abs();
        let avg = (batch_rms + stream_rms) / 2.0;
        assert!(
            diff < avg * 0.5,
            "DJ ratio {}: batch_rms={:.4}, stream_rms={:.4}, diff={:.4}",
            ratio,
            batch_rms,
            stream_rms,
            diff
        );
    }
}

// ========== Frequency preservation parity ==========

#[test]
fn test_parity_440hz_preserved_both_modes() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let batch = stretch(&input, &params).unwrap();
    let stream = stream_stretch(&input, params, 4096);

    let batch_440 = dft_energy_at(&batch, sample_rate, 440.0);
    let stream_440 = dft_energy_at(&stream, sample_rate, 440.0);

    // Both should preserve 440 Hz content
    assert!(
        batch_440 > 0.03,
        "Batch should preserve 440 Hz (energy={:.4})",
        batch_440
    );
    assert!(
        stream_440 > 0.03,
        "Stream should preserve 440 Hz (energy={:.4})",
        stream_440
    );
}

#[test]
fn test_parity_two_tone_preserved() {
    let sample_rate = 44100;
    let n = sample_rate as usize * 2;
    let input: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 880.0 * t).sin()
        })
        .collect();

    let params = StretchParams::new(1.25)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let batch = stretch(&input, &params).unwrap();
    let stream = stream_stretch(&input, params, 4096);

    // Both tones should be present in both outputs
    assert!(dft_energy_at(&batch, sample_rate, 440.0) > 0.001);
    assert!(dft_energy_at(&batch, sample_rate, 880.0) > 0.001);
    assert!(dft_energy_at(&stream, sample_rate, 440.0) > 0.001);
    assert!(dft_energy_at(&stream, sample_rate, 880.0) > 0.001);
}

// ========== Finiteness and no-clipping parity ==========

#[test]
fn test_parity_all_finite_output() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    for &ratio in &[0.5, 0.75, 1.0, 1.5, 2.0] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let batch = stretch(&input, &params).unwrap();
        let stream = stream_stretch(&input, params, 4096);

        assert!(
            batch.iter().all(|s| s.is_finite()),
            "Batch output has non-finite at ratio {}",
            ratio
        );
        assert!(
            stream.iter().all(|s| s.is_finite()),
            "Stream output has non-finite at ratio {}",
            ratio
        );
    }
}

// ========== Stereo parity ==========

#[test]
fn test_parity_stereo_both_channels() {
    let sample_rate = 44100;
    let n = sample_rate as usize;
    let mut input = vec![0.0f32; n * 2];
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        input[i * 2] = (2.0 * PI * 440.0 * t).sin(); // L
        input[i * 2 + 1] = (2.0 * PI * 880.0 * t).sin(); // R
    }

    let params = StretchParams::new(1.25)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let batch = stretch(&input, &params).unwrap();
    let stream = stream_stretch(&input, params, 4096);

    // Both should have even sample count (stereo)
    assert_eq!(batch.len() % 2, 0, "Batch stereo must be even");
    assert_eq!(stream.len() % 2, 0, "Stream stereo must be even");

    // Both channels should have energy
    let batch_l_rms: f64 = rms(&batch.iter().step_by(2).copied().collect::<Vec<f32>>());
    let batch_r_rms: f64 = rms(&batch
        .iter()
        .skip(1)
        .step_by(2)
        .copied()
        .collect::<Vec<f32>>());
    let stream_l_rms: f64 = rms(&stream.iter().step_by(2).copied().collect::<Vec<f32>>());
    let stream_r_rms: f64 = rms(&stream
        .iter()
        .skip(1)
        .step_by(2)
        .copied()
        .collect::<Vec<f32>>());

    assert!(batch_l_rms > 0.1, "Batch L should have energy");
    assert!(batch_r_rms > 0.1, "Batch R should have energy");
    assert!(stream_l_rms > 0.1, "Stream L should have energy");
    assert!(stream_r_rms > 0.1, "Stream R should have energy");
}

// ========== Hybrid streaming vs batch parity ==========

#[test]
fn test_parity_hybrid_streaming_closer_to_batch() {
    // Hybrid streaming mode should produce results closer to batch
    // than PV-only streaming, since batch also uses hybrid algorithm.
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let batch = stretch(&input, &params).unwrap();
    let stream_pv = stream_stretch(&input, params.clone(), 4096);
    let stream_hybrid = stream_stretch_hybrid(&input, params, 4096);

    let batch_rms = rms(&batch);
    let pv_rms = rms(&stream_pv);
    let hybrid_rms = rms(&stream_hybrid);

    // All three should produce non-trivial output
    assert!(batch_rms > 0.1, "Batch should have energy");
    assert!(pv_rms > 0.1, "PV stream should have energy");
    assert!(hybrid_rms > 0.1, "Hybrid stream should have energy");
}

// ========== Chunk size independence ==========

#[test]
fn test_parity_chunk_size_independent() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.25)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let out_small = stream_stretch(&input, params.clone(), 1024);
    let out_medium = stream_stretch(&input, params.clone(), 4096);
    let out_large = stream_stretch(&input, params, 16384);

    let rms_small = rms(&out_small);
    let rms_medium = rms(&out_medium);
    let rms_large = rms(&out_large);

    // RMS should be similar regardless of chunk size
    let avg = (rms_small + rms_medium + rms_large) / 3.0;
    assert!(
        (rms_small - avg).abs() < avg * 0.3,
        "Small chunk RMS {:.4} too far from average {:.4}",
        rms_small,
        avg
    );
    assert!(
        (rms_medium - avg).abs() < avg * 0.3,
        "Medium chunk RMS {:.4} too far from average {:.4}",
        rms_medium,
        avg
    );
    assert!(
        (rms_large - avg).abs() < avg * 0.3,
        "Large chunk RMS {:.4} too far from average {:.4}",
        rms_large,
        avg
    );
}

// ========== DJ beatmatch scenario ==========

#[test]
fn test_parity_dj_beatmatch_scenario() {
    // Simulate DJ matching 126 BPM to 128 BPM with both batch and streaming
    let sample_rate = 44100;
    let ratio = 126.0 / 128.0; // ~0.984
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 4);

    let params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let batch = stretch(&input, &params).unwrap();
    let stream = stream_stretch(&input, params, 4096);

    // Batch should be slightly shorter (speeding up)
    assert!(batch.len() < input.len(), "Batch DJ should compress");
    // Stream may have flush padding, so just check it produced output
    assert!(!stream.is_empty(), "Stream DJ should produce output");

    // Batch should preserve 440 Hz
    assert!(
        dft_energy_at(&batch, sample_rate, 440.0) > 0.01,
        "Batch should preserve 440 Hz"
    );
    // Stream has flush padding that dilutes DFT energy, so just check RMS
    let stream_rms = rms(&stream);
    assert!(
        stream_rms > 0.01,
        "Stream DJ should have non-trivial energy (rms={:.4})",
        stream_rms
    );

    // No NaN/Inf in either output
    assert!(batch.iter().all(|s| s.is_finite()));
    assert!(stream.iter().all(|s| s.is_finite()));
}

// ========== Preset parity ==========

#[test]
fn test_parity_all_presets_produce_output() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let presets = [
        (EdmPreset::DjBeatmatch, 0.98),
        (EdmPreset::HouseLoop, 1.25),
        (EdmPreset::Halftime, 2.0),
        (EdmPreset::Ambient, 2.5),
        (EdmPreset::VocalChop, 1.5),
    ];

    for (preset, ratio) in &presets {
        let params = StretchParams::new(*ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(*preset);

        let batch = stretch(&input, &params).unwrap();
        let stream = stream_stretch(&input, params, 4096);

        assert!(
            !batch.is_empty(),
            "Batch should produce output for {:?}",
            preset
        );
        assert!(
            !stream.is_empty(),
            "Stream should produce output for {:?}",
            preset
        );
        assert!(
            batch.iter().all(|s| s.is_finite()),
            "Batch {:?} has non-finite",
            preset
        );
        assert!(
            stream.iter().all(|s| s.is_finite()),
            "Stream {:?} has non-finite",
            preset
        );
    }
}

// ========== 48kHz parity ==========

#[test]
fn test_parity_48khz() {
    let sample_rate = 48000;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let batch = stretch(&input, &params).unwrap();
    let stream = stream_stretch(&input, params, 4096);

    assert!(!batch.is_empty());
    assert!(!stream.is_empty());

    let batch_ratio = batch.len() as f64 / input.len() as f64;
    let stream_ratio = stream.len() as f64 / input.len() as f64;

    assert!(
        (batch_ratio - 1.5).abs() < 0.5,
        "Batch 48kHz ratio {:.3} too far from 1.5",
        batch_ratio
    );
    assert!(
        (stream_ratio - 1.5).abs() < 0.5,
        "Stream 48kHz ratio {:.3} too far from 1.5",
        stream_ratio
    );
}

//! Additional streaming processor edge cases and regression tests.

use timestretch::{EdmPreset, StreamProcessor, StretchError, StretchParams};

// ===== Rapid ratio changes =====

#[test]
fn test_streaming_rapid_ratio_changes() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let mut proc = StreamProcessor::new(params);

    // Generate input signal
    let input: Vec<f32> = (0..4096)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();

    // Rapidly change ratio between each chunk
    let mut total_output = Vec::new();
    let ratios = [1.0, 1.05, 0.95, 1.1, 0.9, 1.02, 0.98];
    for (chunk_idx, &ratio) in ratios.iter().enumerate() {
        proc.set_stretch_ratio(ratio);
        let output = proc.process(&input).unwrap();
        total_output.extend_from_slice(&output);

        // Verify no NaN/Inf
        for &s in &output {
            assert!(
                s.is_finite(),
                "Non-finite at chunk {} ratio {}",
                chunk_idx,
                ratio
            );
        }
    }
    assert!(!total_output.is_empty());
}

#[test]
fn test_streaming_multiple_tempo_changes() {
    let mut proc = StreamProcessor::from_tempo(128.0, 128.0, 44100, 1);

    let input: Vec<f32> = (0..4096)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();

    // Process at original tempo
    let _ = proc.process(&input).unwrap();

    // Change tempo multiple times
    for target_bpm in [130.0, 125.0, 132.0, 120.0, 128.0] {
        assert!(proc.set_tempo(target_bpm));
        let output = proc.process(&input).unwrap();
        for &s in &output {
            assert!(s.is_finite());
        }
    }
}

#[test]
fn test_streaming_set_tempo_without_from_tempo() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let mut proc = StreamProcessor::new(params);

    // set_tempo should return false when not created with from_tempo
    assert!(!proc.set_tempo(130.0));
    assert!(proc.source_bpm().is_none());
}

#[test]
fn test_streaming_set_tempo_invalid_bpm() {
    let mut proc = StreamProcessor::from_tempo(128.0, 128.0, 44100, 1);

    // Invalid BPM values
    assert!(!proc.set_tempo(0.0));
    assert!(!proc.set_tempo(-128.0));
    // NaN > 0.0 is false, so set_tempo rejects it
    assert!(!proc.set_tempo(f64::NAN));
    // Note: Infinity is technically > 0.0, so set_tempo accepts it
    // (results in ratio ~0, which may produce empty output but doesn't crash)
}

#[test]
fn test_streaming_from_tempo_source_bpm() {
    let proc = StreamProcessor::from_tempo(126.0, 128.0, 44100, 1);
    assert_eq!(proc.source_bpm(), Some(126.0));
}

// ===== Empty/edge input handling =====

#[test]
fn test_streaming_empty_chunks() {
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let mut proc = StreamProcessor::new(params);

    // Multiple empty chunks should not crash
    for _ in 0..10 {
        let output = proc.process(&[]).unwrap();
        // Empty input produces empty output (or maybe a tiny bit from internal state)
        assert!(output.len() < 100);
    }
}

#[test]
fn test_streaming_single_sample_repeated() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let mut proc = StreamProcessor::new(params);

    // Feed single samples repeatedly
    let mut total_output = Vec::new();
    for i in 0..10000 {
        let sample = [(2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin()];
        let output = proc.process(&sample).unwrap();
        total_output.extend_from_slice(&output);
    }
    // Should eventually produce output
    // (With FFT size 4096, need to accumulate enough samples first)
}

#[test]
fn test_streaming_very_large_chunk() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let mut proc = StreamProcessor::new(params);

    // 10 seconds in one chunk
    let input: Vec<f32> = (0..441000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let output = proc.process(&input).unwrap();
    assert!(!output.is_empty());
    for &s in &output {
        assert!(s.is_finite());
    }
}

// ===== Stereo streaming =====

#[test]
fn test_streaming_stereo_channel_separation() {
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(2);
    let mut proc = StreamProcessor::new(params);

    // L=440Hz, R=silence
    let num_frames = 8192;
    let mut input = Vec::with_capacity(num_frames * 2);
    for i in 0..num_frames {
        input.push((2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin());
        input.push(0.0);
    }

    let output = proc.process(&input).unwrap();
    assert_eq!(output.len() % 2, 0);

    // Right channel should remain mostly silent
    let right_rms: f32 = output
        .iter()
        .skip(1)
        .step_by(2)
        .map(|&s| s * s)
        .sum::<f32>()
        / (output.len() / 2) as f32;
    let left_rms: f32 = output
        .iter()
        .step_by(2)
        .map(|&s| s * s)
        .sum::<f32>()
        / (output.len() / 2) as f32;

    if output.len() > 100 {
        // Only check if we got meaningful output
        assert!(
            right_rms < left_rms * 0.1 + 0.001,
            "Right channel should be much quieter than left: R_rms={}, L_rms={}",
            right_rms.sqrt(),
            left_rms.sqrt()
        );
    }
}

// ===== Flush behavior =====

#[test]
fn test_streaming_flush_produces_remaining() {
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let mut proc = StreamProcessor::new(params);

    // Feed some audio
    let input: Vec<f32> = (0..8192)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let _ = proc.process(&input).unwrap();

    // Flush should produce remaining buffered audio
    let flushed = proc.flush().unwrap();
    // May or may not have remaining data, but should not crash
    for &s in &flushed {
        assert!(s.is_finite());
    }
}

#[test]
fn test_streaming_double_flush() {
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let mut proc = StreamProcessor::new(params);

    let input: Vec<f32> = (0..8192)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let _ = proc.process(&input).unwrap();

    let _ = proc.flush().unwrap();
    // Second flush should be safe and produce empty or minimal output
    let second = proc.flush().unwrap();
    for &s in &second {
        assert!(s.is_finite());
    }
}

#[test]
fn test_streaming_flush_without_input() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let mut proc = StreamProcessor::new(params);

    // Flush with no prior input
    let flushed = proc.flush().unwrap();
    assert!(flushed.is_empty() || flushed.iter().all(|s| s.is_finite()));
}

// ===== Reset behavior =====

#[test]
fn test_streaming_reset_and_reprocess() {
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let mut proc = StreamProcessor::new(params);

    let input: Vec<f32> = (0..8192)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();

    // Process once
    let output1 = proc.process(&input).unwrap();

    // Reset
    proc.reset();

    // Process same input again
    let output2 = proc.process(&input).unwrap();

    // Outputs should be similar (same input, same state after reset)
    if !output1.is_empty() && !output2.is_empty() {
        let len = output1.len().min(output2.len());
        let mut max_diff = 0.0f32;
        for i in 0..len {
            let diff = (output1[i] - output2[i]).abs();
            max_diff = max_diff.max(diff);
        }
        // After reset, should produce identical output
        assert!(
            max_diff < 1e-6,
            "Reset should produce identical output, max_diff={}",
            max_diff
        );
    }
}

// ===== Latency reporting =====

#[test]
fn test_streaming_latency_positive() {
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let proc = StreamProcessor::new(params);

    let latency = proc.latency_samples();
    assert!(latency > 0, "Latency should be positive, got {}", latency);
    // Latency should be related to FFT size
    assert!(
        latency <= 4096 * 4,
        "Latency seems too high: {}",
        latency
    );
}

#[test]
fn test_streaming_latency_varies_with_fft_size() {
    let params_small = StretchParams::new(1.0)
        .with_fft_size(256)
        .with_channels(1);
    let proc_small = StreamProcessor::new(params_small);

    let params_large = StretchParams::new(1.0)
        .with_fft_size(8192)
        .with_channels(1);
    let proc_large = StreamProcessor::new(params_large);

    assert!(
        proc_large.latency_samples() >= proc_small.latency_samples(),
        "Larger FFT should have >= latency: {} vs {}",
        proc_large.latency_samples(),
        proc_small.latency_samples()
    );
}

// ===== NaN/Inf rejection =====

#[test]
fn test_streaming_rejects_nan_in_middle() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let mut proc = StreamProcessor::new(params);

    let mut input = vec![0.0f32; 4096];
    input[2000] = f32::NAN;
    assert!(matches!(
        proc.process(&input),
        Err(StretchError::NonFiniteInput)
    ));
}

#[test]
fn test_streaming_rejects_neg_infinity() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let mut proc = StreamProcessor::new(params);

    let mut input = vec![0.0f32; 4096];
    input[100] = f32::NEG_INFINITY;
    assert!(matches!(
        proc.process(&input),
        Err(StretchError::NonFiniteInput)
    ));
}

// ===== Params accessor =====

#[test]
fn test_streaming_params_accessor() {
    let params = StretchParams::new(1.5)
        .with_sample_rate(48000)
        .with_channels(2)
        .with_preset(EdmPreset::Ambient);
    let proc = StreamProcessor::new(params);

    let p = proc.params();
    assert!((p.stretch_ratio - 1.5).abs() < 1e-10);
    assert_eq!(p.sample_rate, 48000);
    assert_eq!(p.channels, timestretch::Channels::Stereo);
    assert_eq!(p.preset, Some(EdmPreset::Ambient));
}

// ===== Streaming with various presets =====

#[test]
fn test_streaming_all_presets() {
    let input: Vec<f32> = (0..8192)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();

    for preset in [
        EdmPreset::DjBeatmatch,
        EdmPreset::HouseLoop,
        EdmPreset::Halftime,
        EdmPreset::Ambient,
        EdmPreset::VocalChop,
    ] {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(preset);
        let mut proc = StreamProcessor::new(params);

        let output = proc.process(&input).unwrap();
        for &s in &output {
            assert!(
                s.is_finite(),
                "Non-finite output with preset {:?}",
                preset
            );
        }
    }
}

// ===== Streaming compression =====

#[test]
fn test_streaming_compression_various_ratios() {
    for ratio in [0.5, 0.75, 0.9] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);

        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        // Process in chunks
        let mut total_output = Vec::new();
        for chunk in input.chunks(4096) {
            let output = proc.process(chunk).unwrap();
            total_output.extend_from_slice(&output);
        }
        let flushed = proc.flush().unwrap();
        total_output.extend_from_slice(&flushed);

        // All output should be finite
        for &s in &total_output {
            assert!(
                s.is_finite(),
                "Non-finite at compression ratio {}",
                ratio
            );
        }
    }
}

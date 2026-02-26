use std::f32::consts::PI;
use timestretch::{stretch, EdmPreset, StreamProcessor, StretchParams};

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

fn p95_adjacent_diff(signal: &[f32]) -> f32 {
    if signal.len() < 2 {
        return 0.0;
    }
    let mut diffs: Vec<f32> = signal
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .filter(|d| d.is_finite())
        .collect();
    if diffs.is_empty() {
        return 0.0;
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((diffs.len() as f32) * 0.95).floor() as usize;
    diffs[idx.min(diffs.len() - 1)]
}

/// Generates an EDM-like signal with kicks, hi-hats, sub-bass, and a pad.
fn generate_edm_signal(sample_rate: u32, bpm: f64, duration_secs: f64) -> Vec<f32> {
    let total_samples = (sample_rate as f64 * duration_secs) as usize;
    let samples_per_beat = (60.0 / bpm * sample_rate as f64) as usize;
    let hihat_interval = samples_per_beat / 2;

    let mut samples = vec![0.0f32; total_samples];

    for (i, sample) in samples.iter_mut().enumerate() {
        let beat_pos = i % samples_per_beat;
        let hihat_pos = i % hihat_interval;

        // Kick: sharp attack + exponential decay on every beat
        if beat_pos < (sample_rate as usize / 20) {
            let t = beat_pos as f32 / sample_rate as f32;
            let kick_freq = 60.0 + 200.0 * (-t * 40.0).exp();
            let kick_env = (-t * 20.0).exp();
            *sample += 0.6 * kick_env * (2.0 * PI * kick_freq * t).sin();
        }

        // Hi-hat: noise burst on every eighth note
        if hihat_pos < (sample_rate as usize / 200) {
            let t = hihat_pos as f32 / sample_rate as f32;
            let hat_env = (-t * 300.0).exp();
            // Deterministic pseudo-noise
            let noise = ((i as f32 * 12_345.679).sin() * 43_758.547).fract() * 2.0 - 1.0;
            *sample += 0.15 * hat_env * noise;
        }

        // Sub-bass: continuous 50 Hz sine
        let t = i as f32 / sample_rate as f32;
        *sample += 0.25 * (2.0 * PI * 50.0 * t).sin();

        // Pad: soft 440 Hz + 660 Hz chord
        *sample += 0.1 * (2.0 * PI * 440.0 * t).sin();
        *sample += 0.08 * (2.0 * PI * 660.0 * t).sin();
    }

    samples
}

/// Run a hybrid streaming stretch and return the full output.
fn hybrid_stream_stretch(input: &[f32], params: StretchParams, chunk_size: usize) -> Vec<f32> {
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

/// Run a PV-only streaming stretch (default mode).
fn pv_stream_stretch(input: &[f32], params: StretchParams, chunk_size: usize) -> Vec<f32> {
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

// ===================== HYBRID STREAMING BASIC TESTS =====================

#[test]
fn test_hybrid_streaming_basic_mono() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = hybrid_stream_stretch(&input, params, 4096);
    assert!(!output.is_empty(), "Hybrid streaming should produce output");

    let len_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (len_ratio - 1.5).abs() < 0.4,
        "Length ratio {} too far from 1.5",
        len_ratio
    );
}

#[test]
fn test_hybrid_streaming_stereo() {
    let sample_rate = 44100u32;
    let num_frames = sample_rate as usize;
    let mut input = vec![0.0f32; num_frames * 2];
    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        input[i * 2] = (2.0 * PI * 440.0 * t).sin();
        input[i * 2 + 1] = (2.0 * PI * 880.0 * t).sin();
    }

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let output = hybrid_stream_stretch(&input, params, 4096 * 2);
    assert!(!output.is_empty(), "Hybrid stereo should produce output");
    assert_eq!(output.len() % 2, 0, "Stereo output must be even");
}

#[test]
fn test_hybrid_streaming_compression() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(0.75)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = hybrid_stream_stretch(&input, params, 4096);
    assert!(!output.is_empty());
    assert!(
        output.len() < input.len(),
        "Compression should produce shorter output"
    );
}

// ===================== HYBRID VS BATCH COMPARISON =====================

#[test]
fn test_hybrid_streaming_vs_batch_length() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    for &ratio in &[0.75, 1.0, 1.25, 1.5] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let batch_output = stretch(&input, &params).unwrap();
        // Use a large chunk so hybrid gets enough context per call
        let stream_output = hybrid_stream_stretch(&input, params, 44100);

        if batch_output.is_empty() || stream_output.is_empty() {
            continue;
        }

        let batch_ratio = batch_output.len() as f64 / input.len() as f64;
        let stream_ratio = stream_output.len() as f64 / input.len() as f64;

        // Both should achieve similar length ratios (within 40%)
        // Streaming processes chunks independently, so some divergence is expected
        assert!(
            (batch_ratio - stream_ratio).abs() < 0.4,
            "ratio={}: batch len ratio {:.3} vs stream len ratio {:.3}",
            ratio,
            batch_ratio,
            stream_ratio
        );
    }
}

#[test]
fn test_hybrid_streaming_vs_batch_rms() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let batch_output = stretch(&input, &params).unwrap();
    let stream_output = hybrid_stream_stretch(&input, params, 4096);

    let batch_rms = rms(&batch_output);
    let stream_rms = rms(&stream_output);
    let input_rms = rms(&input);

    // Both should preserve similar RMS relative to input
    assert!(
        batch_rms > input_rms * 0.3,
        "Batch RMS {} too low vs input {}",
        batch_rms,
        input_rms
    );
    assert!(
        stream_rms > input_rms * 0.3,
        "Stream RMS {} too low vs input {}",
        stream_rms,
        input_rms
    );
}

// ===================== EDM SIGNAL QUALITY TESTS =====================

#[test]
fn test_hybrid_streaming_edm_signal() {
    let sample_rate = 44100u32;
    let input = generate_edm_signal(sample_rate, 128.0, 2.0);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = hybrid_stream_stretch(&input, params, 4096);
    assert!(!output.is_empty());
    assert!(
        output.iter().all(|s| s.is_finite()),
        "EDM output must be all finite"
    );

    let len_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (len_ratio - 1.5).abs() < 0.4,
        "EDM stretch ratio {} too far from 1.5",
        len_ratio
    );
}

#[test]
fn test_hybrid_streaming_edm_vs_batch() {
    let sample_rate = 44100u32;
    let input = generate_edm_signal(sample_rate, 128.0, 2.0);

    let params = StretchParams::new(1.25)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let batch_output = stretch(&input, &params).unwrap();
    let stream_output = hybrid_stream_stretch(&input, params, 4096);

    let batch_rms = rms(&batch_output);
    let stream_rms = rms(&stream_output);

    // Both should produce non-trivial output
    assert!(batch_rms > 0.05, "Batch RMS too low: {}", batch_rms);
    assert!(stream_rms > 0.05, "Stream RMS too low: {}", stream_rms);

    // RMS should be in the same ballpark (within 50%)
    let rms_ratio = stream_rms / batch_rms;
    assert!(
        (0.5..=2.0).contains(&rms_ratio),
        "RMS ratio {:.3} (stream={:.4}, batch={:.4}) too divergent",
        rms_ratio,
        stream_rms,
        batch_rms
    );
}

#[test]
fn test_hybrid_streaming_dj_beatmatch() {
    let sample_rate = 44100u32;
    let input = generate_edm_signal(sample_rate, 126.0, 3.0);

    // 126 → 128 BPM
    let ratio = 126.0 / 128.0;
    let params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    // Use large chunks so the hybrid stretcher gets enough context
    let output = hybrid_stream_stretch(&input, params, sample_rate as usize);
    assert!(!output.is_empty());
    assert!(
        output.iter().all(|s| s.is_finite()),
        "DJ output must be all finite"
    );

    let len_ratio = output.len() as f64 / input.len() as f64;
    // DJ beatmatch: small ratio change (~0.984), output should be near input length
    assert!(
        (len_ratio - ratio).abs() < 0.3,
        "DJ stretch ratio {} too far from expected {}",
        len_ratio,
        ratio
    );
}

#[test]
fn test_hybrid_streaming_all_presets() {
    let sample_rate = 44100u32;
    let input = generate_edm_signal(sample_rate, 128.0, 2.0);

    let presets = [
        (EdmPreset::DjBeatmatch, 0.98),
        (EdmPreset::HouseLoop, 1.5),
        (EdmPreset::Halftime, 2.0),
        (EdmPreset::Ambient, 2.0),
        (EdmPreset::VocalChop, 1.5),
    ];

    for (preset, ratio) in presets {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(preset);

        let output = hybrid_stream_stretch(&input, params, 4096);
        assert!(
            !output.is_empty(),
            "Preset {:?} produced empty output",
            preset
        );
        assert!(
            output.iter().all(|s| s.is_finite()),
            "Preset {:?} produced NaN/Inf",
            preset
        );
    }
}

// ===================== HYBRID VS PV-ONLY COMPARISON =====================

#[test]
fn test_hybrid_vs_pv_both_produce_output() {
    let sample_rate = 44100u32;
    let input = generate_edm_signal(sample_rate, 128.0, 2.0);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let hybrid_output = hybrid_stream_stretch(&input, params.clone(), 4096);
    let pv_output = pv_stream_stretch(&input, params, 4096);

    assert!(!hybrid_output.is_empty(), "Hybrid should produce output");
    assert!(!pv_output.is_empty(), "PV-only should produce output");

    // Both should have reasonable length ratios
    let hybrid_ratio = hybrid_output.len() as f64 / input.len() as f64;
    let pv_ratio = pv_output.len() as f64 / input.len() as f64;

    assert!(
        (hybrid_ratio - 1.5).abs() < 0.4,
        "Hybrid ratio {} too far from 1.5",
        hybrid_ratio
    );
    assert!(
        (pv_ratio - 1.5).abs() < 0.4,
        "PV ratio {} too far from 1.5",
        pv_ratio
    );
}

#[test]
fn test_hybrid_streaming_edm_stereo() {
    let sample_rate = 44100u32;
    let mono = generate_edm_signal(sample_rate, 128.0, 2.0);
    // Create stereo by duplicating mono with slight phase offset
    let num_frames = mono.len();
    let mut stereo = vec![0.0f32; num_frames * 2];
    for i in 0..num_frames {
        stereo[i * 2] = mono[i];
        // Right channel: slight offset simulates stereo field
        let offset = if i + 10 < num_frames {
            mono[i + 10]
        } else {
            0.0
        };
        stereo[i * 2 + 1] = offset;
    }

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(2)
        .with_preset(EdmPreset::HouseLoop);

    let output = hybrid_stream_stretch(&stereo, params, 4096 * 2);
    assert!(!output.is_empty());
    assert_eq!(output.len() % 2, 0, "Stereo output must be even");
    assert!(
        output.iter().all(|s| s.is_finite()),
        "Stereo output must be all finite"
    );
}

#[test]
fn test_hybrid_streaming_flush() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let mut processor = StreamProcessor::new(params);
    processor.set_hybrid_mode(true);

    // Feed less than a full processing window
    let small_chunk = &input[..2048];
    let out1 = processor.process(small_chunk).unwrap();

    // Flush should produce the remaining output
    let out2 = processor.flush().unwrap();
    let total = out1.len() + out2.len();

    // Should have produced some output after flush
    assert!(total > 0, "Flush should produce output from buffered data");
}

#[test]
fn test_hybrid_streaming_48khz() {
    let sample_rate = 48000u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = hybrid_stream_stretch(&input, params, 4096);
    assert!(!output.is_empty());
    assert!(output.iter().all(|s| s.is_finite()));
}

#[test]
fn test_hybrid_streaming_ratio_change() {
    let sample_rate = 44100u32;
    let input = generate_edm_signal(sample_rate, 128.0, 3.0);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let mut processor = StreamProcessor::new(params);
    processor.set_hybrid_mode(true);

    let chunk_size = 4096;
    let mut output = Vec::new();

    // Process first half at ratio 1.0
    let half = input.len() / 2;
    for chunk in input[..half].chunks(chunk_size) {
        if let Ok(out) = processor.process(chunk) {
            output.extend_from_slice(&out);
        }
    }

    // Change to ratio 1.25 mid-stream
    processor.set_stretch_ratio(1.25);

    // Process second half — ratio interpolation happens inside process()
    for chunk in input[half..].chunks(chunk_size) {
        if let Ok(out) = processor.process(chunk) {
            output.extend_from_slice(&out);
        }
    }
    if let Ok(out) = processor.flush() {
        output.extend_from_slice(&out);
    }

    assert!(
        !output.is_empty(),
        "Ratio change should still produce output"
    );
    assert!(
        output.iter().all(|s| s.is_finite()),
        "Output after ratio change must be finite"
    );
}

#[test]
fn test_hybrid_streaming_reset() {
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let mut processor = StreamProcessor::new(params);
    processor.set_hybrid_mode(true);

    // Process some data
    let _ = processor.process(&input[..8192]);

    // Reset and verify hybrid mode persists
    processor.reset();
    assert!(
        processor.is_hybrid_mode(),
        "Hybrid mode should persist after reset"
    );

    // Process again after reset
    let mut output = Vec::new();
    for chunk in input.chunks(4096) {
        if let Ok(out) = processor.process(chunk) {
            output.extend_from_slice(&out);
        }
    }
    if let Ok(out) = processor.flush() {
        output.extend_from_slice(&out);
    }

    assert!(!output.is_empty(), "Should produce output after reset");
}

#[test]
fn test_hybrid_streaming_persistent_small_vs_large_chunk_length() {
    let sample_rate = 44100u32;
    let input = generate_edm_signal(sample_rate, 128.0, 4.0);

    let params = StretchParams::new(1.25)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    // Small chunks exercise persistent rolling-buffer behavior.
    let small_chunk_output = hybrid_stream_stretch(&input, params.clone(), 2048);
    // Large chunk approximates single-pass rendering.
    let large_chunk_output = hybrid_stream_stretch(&input, params, input.len());

    assert!(!small_chunk_output.is_empty());
    assert!(!large_chunk_output.is_empty());

    let len_diff = small_chunk_output.len().abs_diff(large_chunk_output.len()) as f64;
    let base = large_chunk_output.len().max(1) as f64;
    let len_diff_pct = (len_diff / base) * 100.0;
    assert!(
        len_diff_pct < 5.0,
        "Small-chunk hybrid length deviated too far from large-chunk output: {:.3}%",
        len_diff_pct
    );
}

#[test]
fn test_hybrid_streaming_chunk_boundary_artifacts_bounded() {
    let sample_rate = 44100u32;
    let input = generate_edm_signal(sample_rate, 128.0, 4.0);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let small_chunk_output = hybrid_stream_stretch(&input, params.clone(), 2048);
    let large_chunk_output = hybrid_stream_stretch(&input, params, input.len());

    let p95_small = p95_adjacent_diff(&small_chunk_output);
    let p95_large = p95_adjacent_diff(&large_chunk_output);

    // Small-chunk streaming may be noisier than single-pass, but should remain
    // within a bounded factor (no severe chunk-boundary discontinuities).
    assert!(
        p95_small <= p95_large * 3.0 + 0.02,
        "Chunk-boundary roughness too high: small={:.5}, large={:.5}",
        p95_small,
        p95_large
    );
}

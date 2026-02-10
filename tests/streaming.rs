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

/// Run a streaming stretch and return the full output.
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

// ===================== BASIC STREAMING TESTS =====================

#[test]
fn test_streaming_basic_mono() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stream_stretch(&input, params, 4096);
    assert!(!output.is_empty(), "Streaming should produce output");

    let max = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(max > 0.01, "Streaming output should not be silent");
}

#[test]
fn test_streaming_stereo() {
    let sample_rate = 44100;
    let num_frames = sample_rate as usize;
    let mut input = Vec::with_capacity(num_frames * 2);
    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        input.push((2.0 * PI * 440.0 * t).sin());
        input.push((2.0 * PI * 880.0 * t).sin());
    }

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let output = stream_stretch(&input, params, 4096);
    assert!(!output.is_empty());
    assert_eq!(output.len() % 2, 0, "Stereo output must have even length");
}

#[test]
fn test_streaming_ratio_change() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 4);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let mut processor = StreamProcessor::new(params);
    let chunk_size = 4096;
    let mut output = Vec::new();

    for (chunk_idx, chunk) in input.chunks(chunk_size).enumerate() {
        if chunk_idx == input.len() / chunk_size / 2 {
            processor.set_stretch_ratio(1.5);
        }
        if let Ok(out) = processor.process(chunk) {
            output.extend_from_slice(&out);
        }
    }
    if let Ok(remaining) = processor.flush() {
        output.extend_from_slice(&remaining);
    }

    assert!(!output.is_empty());
}

#[test]
fn test_streaming_flush() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let mut processor = StreamProcessor::new(params);
    let mut output_before_flush = Vec::new();
    for chunk in input.chunks(2048) {
        if let Ok(out) = processor.process(chunk) {
            output_before_flush.extend_from_slice(&out);
        }
    }
    let flushed = processor.flush().unwrap();

    let total = output_before_flush.len() + flushed.len();
    assert!(total > 0, "Total output should be non-empty after flush");
}

#[test]
fn test_streaming_reset() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output1 = stream_stretch(&input, params.clone(), 4096);

    // Second run after implicit reset (new processor)
    let output2 = stream_stretch(&input, params, 4096);

    let rms1 = rms(&output1);
    let rms2 = rms(&output2);
    assert!(
        (rms1 - rms2).abs() < rms1 * 0.2,
        "Reset should produce consistent output: rms1={}, rms2={}",
        rms1,
        rms2
    );
}

// ===================== STREAMING VS BATCH COMPARISON =====================

#[test]
fn test_streaming_vs_batch_length() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    for &ratio in &[0.75, 1.0, 1.25, 1.5, 2.0] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let batch_output = stretch(&input, &params).unwrap();
        let stream_output = stream_stretch(&input, params, 4096);

        let batch_ratio = batch_output.len() as f64 / input.len() as f64;
        let stream_ratio = stream_output.len() as f64 / input.len() as f64;

        assert!(
            (batch_ratio - stream_ratio).abs() < 0.5,
            "Ratio {}: batch_ratio={:.3}, stream_ratio={:.3}",
            ratio,
            batch_ratio,
            stream_ratio
        );
    }
}

#[test]
fn test_streaming_vs_batch_rms() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let batch_output = stretch(&input, &params).unwrap();
    let stream_output = stream_stretch(&input, params, 4096);

    let batch_rms = rms(&batch_output);
    let stream_rms = rms(&stream_output);

    assert!(
        (batch_rms - stream_rms).abs() < batch_rms * 0.4,
        "Streaming RMS={} should be close to batch RMS={}",
        stream_rms,
        batch_rms
    );
}

#[test]
fn test_streaming_vs_batch_frequency() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let batch_output = stretch(&input, &params).unwrap();
    let stream_output = stream_stretch(&input, params, 4096);

    let batch_energy = spectral_energy_at_freq(&batch_output, sample_rate, 440.0);
    let stream_energy = spectral_energy_at_freq(&stream_output, sample_rate, 440.0);

    assert!(
        batch_energy > 0.1 && stream_energy > 0.1,
        "440 Hz should be preserved: batch={}, stream={}",
        batch_energy,
        stream_energy
    );
}

// ===================== CHUNK SIZE TESTS =====================

#[test]
fn test_streaming_chunk_size_consistency() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let chunk_sizes = [512, 4096, 16384];
    let mut rms_values = Vec::new();

    for &chunk_size in &chunk_sizes {
        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1);
        let output = stream_stretch(&input, params, chunk_size);
        rms_values.push(rms(&output));
    }

    // All chunk sizes should give similar results
    let avg_rms = rms_values.iter().sum::<f32>() / rms_values.len() as f32;
    for (i, &r) in rms_values.iter().enumerate() {
        assert!(
            (r - avg_rms).abs() < avg_rms * 0.3,
            "Chunk size {} gave RMS {} (avg={})",
            chunk_sizes[i],
            r,
            avg_rms
        );
    }
}

#[test]
fn test_streaming_small_chunks() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stream_stretch(&input, params, 256);
    assert!(!output.is_empty(), "Small chunks should still produce output");
}

// ===================== STEREO STREAMING TESTS =====================

#[test]
fn test_streaming_stereo_vs_batch() {
    let sample_rate = 44100;
    let num_frames = sample_rate as usize;
    let mut input = Vec::with_capacity(num_frames * 2);
    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        input.push((2.0 * PI * 440.0 * t).sin());
        input.push((2.0 * PI * 880.0 * t).sin());
    }

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let batch_output = stretch(&input, &params).unwrap();
    let stream_output = stream_stretch(&input, params, 4096);

    assert_eq!(batch_output.len() % 2, 0);
    assert_eq!(stream_output.len() % 2, 0);

    let batch_rms = rms(&batch_output);
    let stream_rms = rms(&stream_output);

    assert!(
        (batch_rms - stream_rms).abs() < batch_rms * 0.4,
        "Stereo streaming RMS={} should be close to batch RMS={}",
        stream_rms,
        batch_rms
    );
}

// ===================== EDM PRESET STREAMING =====================

#[test]
fn test_streaming_with_edm_preset() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.02)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let output = stream_stretch(&input, params, 4096);
    assert!(!output.is_empty());

    let max = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(max > 0.01, "DJ beatmatch streaming should not be silent");
}

// ===================== EDGE CASE TESTS =====================

#[test]
fn test_streaming_compression() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    for &ratio in &[0.5, 0.75] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = stream_stretch(&input, params, 4096);
        assert!(
            !output.is_empty(),
            "Compression ratio {} should produce output",
            ratio
        );

        let actual_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (actual_ratio - ratio).abs() < 0.5,
            "Compression ratio {}: expected ~{}, got {}",
            ratio,
            ratio,
            actual_ratio
        );
    }
}

#[test]
fn test_streaming_empty_flush() {
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);

    let mut processor = StreamProcessor::new(params);

    // Flush without any input
    let flushed = processor.flush().unwrap();
    assert!(
        flushed.is_empty(),
        "Flush without input should give empty output"
    );

    // Double flush
    let flushed2 = processor.flush().unwrap();
    assert!(
        flushed2.is_empty(),
        "Double flush should give empty output"
    );
}

#[test]
fn test_streaming_single_sample_chunks() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, 4410); // 0.1 second

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stream_stretch(&input, params, 1);
    // May produce output or not depending on implementation, but should not crash
    let _ = output;
}

#[test]
fn test_streaming_large_fft_size() {
    let sample_rate = 44100;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_fft_size(8192);

    let output = stream_stretch(&input, params, 4096);
    assert!(!output.is_empty(), "Large FFT streaming should produce output");
}

// ===================== LATENCY REPORTING =====================

#[test]
fn test_streaming_latency_reporting() {
    let sample_rate = 44100;
    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let processor = StreamProcessor::new(params);
    let latency = processor.latency_samples();

    // Latency should be reasonable (less than 5 seconds worth)
    assert!(
        latency < sample_rate as usize * 5,
        "Latency {} samples seems too high",
        latency
    );
}

#[test]
fn test_streaming_from_tempo_dj_workflow() {
    // Simulate a DJ matching a 126 BPM track to 128 BPM
    let sample_rate = 44100u32;
    let signal = sine_wave(440.0, sample_rate, sample_rate as usize * 2);
    let chunk_size = 4096;

    let mut processor = StreamProcessor::from_tempo(126.0, 128.0, sample_rate, 1);

    // Verify initial state
    let expected_ratio = 126.0 / 128.0;
    assert!(
        (processor.current_stretch_ratio() - expected_ratio).abs() < 1e-6,
        "Initial ratio should be {}, got {}",
        expected_ratio,
        processor.current_stretch_ratio()
    );
    assert_eq!(processor.source_bpm(), Some(126.0));
    assert_eq!(processor.params().preset, Some(EdmPreset::DjBeatmatch));

    // Process audio
    let mut total_output = Vec::new();
    for chunk in signal.chunks(chunk_size) {
        total_output.extend_from_slice(&processor.process(chunk).unwrap());
    }
    total_output.extend_from_slice(&processor.flush().unwrap());

    assert!(!total_output.is_empty(), "from_tempo should produce output");

    // Output ratio should be close to 126/128 ≈ 0.984
    // Phase vocoder output length has some variance, so allow generous tolerance
    let output_ratio = total_output.len() as f64 / signal.len() as f64;
    assert!(
        (output_ratio - expected_ratio).abs() < 0.3,
        "126→128 BPM: output ratio {} too far from expected {}",
        output_ratio,
        expected_ratio
    );
}

#[test]
fn test_streaming_from_tempo_stereo() {
    let sample_rate = 44100u32;
    let num_frames = sample_rate as usize;
    let mut signal = vec![0.0f32; num_frames * 2];
    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        signal[i * 2] = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        signal[i * 2 + 1] = (2.0 * std::f32::consts::PI * 880.0 * t).sin();
    }

    let mut processor = StreamProcessor::from_tempo(120.0, 125.0, sample_rate, 2);

    let mut total_output = Vec::new();
    for chunk in signal.chunks(8192) {
        total_output.extend_from_slice(&processor.process(chunk).unwrap());
    }
    total_output.extend_from_slice(&processor.flush().unwrap());

    assert!(!total_output.is_empty());
    assert_eq!(total_output.len() % 2, 0, "Stereo output must be even");
}

#[test]
fn test_streaming_set_tempo_mid_stream() {
    // Start at 126→128 BPM, then change target to 130 BPM mid-stream
    let sample_rate = 44100u32;
    let signal = sine_wave(440.0, sample_rate, sample_rate as usize * 4);
    let chunk_size = 4096;

    let mut processor = StreamProcessor::from_tempo(126.0, 128.0, sample_rate, 1);
    let mut total_output = Vec::new();

    let chunks: Vec<&[f32]> = signal.chunks(chunk_size).collect();
    let mid = chunks.len() / 2;

    // Process first half at 128 BPM target
    for chunk in &chunks[..mid] {
        total_output.extend_from_slice(&processor.process(chunk).unwrap());
    }

    // Change target to 130 BPM
    assert!(processor.set_tempo(130.0));

    // Process second half
    for chunk in &chunks[mid..] {
        total_output.extend_from_slice(&processor.process(chunk).unwrap());
    }
    total_output.extend_from_slice(&processor.flush().unwrap());

    assert!(
        !total_output.is_empty(),
        "Should produce output across tempo change"
    );

    // Final ratio should approach 126/130
    let target_ratio = 126.0 / 130.0;
    assert!(
        (processor.current_stretch_ratio() - target_ratio).abs() < 0.05,
        "After set_tempo(130), ratio should be ~{}, got {}",
        target_ratio,
        processor.current_stretch_ratio()
    );
}

#[test]
fn test_streaming_set_tempo_without_source_returns_false() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let mut processor = StreamProcessor::new(params);

    // set_tempo requires from_tempo to have been used
    assert!(!processor.set_tempo(128.0));
}

#[test]
fn test_streaming_from_tempo_slowdown() {
    // Slow down: 130 BPM → 120 BPM (ratio > 1.0, output longer)
    let sample_rate = 44100u32;
    let signal = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let mut processor = StreamProcessor::from_tempo(130.0, 120.0, sample_rate, 1);

    let mut total_output = Vec::new();
    for chunk in signal.chunks(4096) {
        total_output.extend_from_slice(&processor.process(chunk).unwrap());
    }
    total_output.extend_from_slice(&processor.flush().unwrap());

    assert!(!total_output.is_empty());
    // 130/120 ≈ 1.083 → output should be longer
    let output_ratio = total_output.len() as f64 / signal.len() as f64;
    assert!(
        output_ratio > 1.0,
        "130→120 BPM should stretch, got ratio {}",
        output_ratio
    );
}

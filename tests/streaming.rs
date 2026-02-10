use timestretch::{EdmPreset, StretchParams, StreamProcessor};

fn sine_wave(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin()
        })
        .collect()
}

#[test]
fn test_streaming_produces_output() {
    let sample_rate = 44100;
    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let mut processor = StreamProcessor::new(params);

    let signal = sine_wave(440.0, sample_rate, sample_rate as usize * 2);
    let chunk_size = 4096;
    let mut total_output = Vec::new();

    for chunk in signal.chunks(chunk_size) {
        let output = processor.process(chunk).unwrap();
        total_output.extend_from_slice(&output);
    }

    // Flush remaining
    let remaining = processor.flush().unwrap();
    total_output.extend_from_slice(&remaining);

    assert!(
        !total_output.is_empty(),
        "Streaming should produce output"
    );
}

#[test]
fn test_streaming_stereo() {
    let sample_rate = 44100;
    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let mut processor = StreamProcessor::new(params);

    let num_frames = sample_rate as usize;
    let mut signal = vec![0.0f32; num_frames * 2];
    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        signal[i * 2] = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        signal[i * 2 + 1] = (2.0 * std::f32::consts::PI * 880.0 * t).sin();
    }

    let chunk_size = 8192; // Must be multiple of 2 for stereo
    let mut total_output = Vec::new();

    for chunk in signal.chunks(chunk_size) {
        let output = processor.process(chunk).unwrap();
        total_output.extend_from_slice(&output);
    }

    let remaining = processor.flush().unwrap();
    total_output.extend_from_slice(&remaining);

    // Output should maintain stereo interleaving
    if !total_output.is_empty() {
        assert_eq!(total_output.len() % 2, 0, "Stereo output should have even length");
    }
}

#[test]
fn test_streaming_ratio_change() {
    let sample_rate = 44100;
    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let mut processor = StreamProcessor::new(params);

    let signal = sine_wave(440.0, sample_rate, sample_rate as usize * 4);
    let chunk_size = 4096;
    for (chunk_count, chunk) in signal.chunks(chunk_size).enumerate() {
        // Change ratio midway through
        if chunk_count == signal.len() / chunk_size / 2 {
            processor.set_stretch_ratio(1.05);
        }
        let _ = processor.process(chunk);
    }

    // Ratio should have changed
    assert!(
        (processor.current_stretch_ratio() - 1.05).abs() < 0.1,
        "Ratio should have changed to ~1.05, got {}",
        processor.current_stretch_ratio()
    );
}

#[test]
fn test_streaming_reset() {
    let sample_rate = 44100;
    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let mut processor = StreamProcessor::new(params);

    // Process some data
    let signal = sine_wave(440.0, sample_rate, sample_rate as usize);
    let _ = processor.process(&signal);

    // Reset
    processor.reset();

    // After reset, should be back to initial state
    assert!(
        (processor.current_stretch_ratio() - 1.5).abs() < 1e-6,
        "After reset, ratio should be 1.5"
    );
}

#[test]
fn test_streaming_small_chunks() {
    let sample_rate = 44100;
    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let mut processor = StreamProcessor::new(params);

    let signal = sine_wave(440.0, sample_rate, sample_rate as usize * 2);
    let chunk_size = 256; // Very small chunks
    let mut total_output = Vec::new();

    for chunk in signal.chunks(chunk_size) {
        let output = processor.process(chunk).unwrap();
        total_output.extend_from_slice(&output);
    }

    let remaining = processor.flush().unwrap();
    total_output.extend_from_slice(&remaining);

    // Should still eventually produce output
    assert!(
        !total_output.is_empty(),
        "Small chunks should still produce output"
    );
}

#[test]
fn test_streaming_latency_reporting() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_fft_size(4096);

    let processor = StreamProcessor::new(params);

    let latency = processor.latency_samples();
    assert!(latency > 0, "Latency should be positive");
    assert_eq!(latency, 8192, "Latency should be 2 * FFT size");

    let latency_secs = processor.latency_secs();
    assert!(latency_secs > 0.0);
    assert!((latency_secs - 8192.0 / 44100.0).abs() < 1e-6);
}

/// Helper: process entire signal through StreamProcessor and return output.
fn stream_process_all(signal: &[f32], params: &StretchParams, chunk_size: usize) -> Vec<f32> {
    let mut processor = StreamProcessor::new(params.clone());
    let mut output = Vec::new();
    for chunk in signal.chunks(chunk_size) {
        output.extend_from_slice(&processor.process(chunk).unwrap());
    }
    output.extend_from_slice(&processor.flush().unwrap());
    output
}

/// Helper: compute RMS energy of a signal.
fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|x| x * x).sum::<f32>() / signal.len() as f32).sqrt()
}

/// Helper: compute DFT magnitude at a specific frequency.
fn dft_magnitude_at(signal: &[f32], freq: f32, sample_rate: u32) -> f32 {
    let mut real = 0.0f64;
    let mut imag = 0.0f64;
    for (i, &s) in signal.iter().enumerate() {
        let phase = 2.0 * std::f64::consts::PI * freq as f64 * i as f64 / sample_rate as f64;
        real += s as f64 * phase.cos();
        imag += s as f64 * phase.sin();
    }
    ((real * real + imag * imag).sqrt() / signal.len() as f64) as f32
}

#[test]
fn test_streaming_vs_batch_output_length() {
    let sample_rate = 44100u32;
    let signal = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    for &ratio in &[0.75, 1.0, 1.25, 1.5, 2.0] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        // Batch
        let batch_output = timestretch::stretch(&signal, &params).unwrap();

        // Streaming
        let stream_output = stream_process_all(&signal, &params, 4096);

        // Both should produce output
        assert!(!batch_output.is_empty(), "Batch empty for ratio {}", ratio);
        assert!(
            !stream_output.is_empty(),
            "Stream empty for ratio {}",
            ratio
        );

        // Output lengths should be proportional to the stretch ratio
        let batch_ratio = batch_output.len() as f64 / signal.len() as f64;
        let stream_ratio = stream_output.len() as f64 / signal.len() as f64;

        assert!(
            (batch_ratio - ratio).abs() < 0.5,
            "Batch length ratio {} too far from {} for ratio {}",
            batch_ratio,
            ratio,
            ratio
        );
        assert!(
            (stream_ratio - ratio).abs() < 0.5,
            "Stream length ratio {} too far from {} for ratio {}",
            stream_ratio,
            ratio,
            ratio
        );
    }
}

#[test]
fn test_streaming_vs_batch_rms_energy() {
    let sample_rate = 44100u32;
    let signal = sine_wave(440.0, sample_rate, sample_rate as usize * 2);
    let input_rms = rms(&signal);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let batch_output = timestretch::stretch(&signal, &params).unwrap();
    let stream_output = stream_process_all(&signal, &params, 4096);

    let batch_rms = rms(&batch_output);
    let stream_rms = rms(&stream_output);

    // Both should preserve energy roughly (within 50% of input)
    assert!(
        (batch_rms - input_rms).abs() < input_rms * 0.5,
        "Batch RMS {} too far from input RMS {}",
        batch_rms,
        input_rms
    );
    assert!(
        (stream_rms - input_rms).abs() < input_rms * 0.5,
        "Stream RMS {} too far from input RMS {}",
        stream_rms,
        input_rms
    );
}

#[test]
fn test_streaming_vs_batch_frequency_preservation() {
    let sample_rate = 44100u32;
    let freq = 440.0;
    let signal = sine_wave(freq, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let batch_output = timestretch::stretch(&signal, &params).unwrap();
    let stream_output = stream_process_all(&signal, &params, 4096);

    // Skip edges (first/last 4096 samples) to avoid windowing artifacts
    let skip = 4096;
    let batch_mid = &batch_output[skip..batch_output.len().saturating_sub(skip)];
    let stream_mid = &stream_output[skip..stream_output.len().saturating_sub(skip)];

    // Both should have strong 440Hz component
    let batch_mag = dft_magnitude_at(batch_mid, freq, sample_rate);
    let stream_mag = dft_magnitude_at(stream_mid, freq, sample_rate);

    // Check against a wrong frequency as baseline
    let batch_wrong = dft_magnitude_at(batch_mid, 880.0, sample_rate);
    let stream_wrong = dft_magnitude_at(stream_mid, 880.0, sample_rate);

    assert!(
        batch_mag > batch_wrong * 5.0,
        "Batch: 440Hz magnitude {} should dominate 880Hz {}",
        batch_mag,
        batch_wrong
    );
    assert!(
        stream_mag > stream_wrong * 5.0,
        "Stream: 440Hz magnitude {} should dominate 880Hz {}",
        stream_mag,
        stream_wrong
    );
}

#[test]
fn test_streaming_chunk_size_consistency() {
    let sample_rate = 44100u32;
    let signal = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.2)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    // Process with different chunk sizes
    let output_small = stream_process_all(&signal, &params, 512);
    let output_medium = stream_process_all(&signal, &params, 4096);
    let output_large = stream_process_all(&signal, &params, 16384);

    // All should produce output
    assert!(!output_small.is_empty(), "Small chunks: no output");
    assert!(!output_medium.is_empty(), "Medium chunks: no output");
    assert!(!output_large.is_empty(), "Large chunks: no output");

    // RMS energy should be similar across chunk sizes (within 30%)
    let rms_small = rms(&output_small);
    let rms_medium = rms(&output_medium);
    let rms_large = rms(&output_large);

    let avg_rms = (rms_small + rms_medium + rms_large) / 3.0;
    assert!(
        (rms_small - avg_rms).abs() < avg_rms * 0.3,
        "Small chunk RMS {} too far from average {}",
        rms_small,
        avg_rms
    );
    assert!(
        (rms_medium - avg_rms).abs() < avg_rms * 0.3,
        "Medium chunk RMS {} too far from average {}",
        rms_medium,
        avg_rms
    );
    assert!(
        (rms_large - avg_rms).abs() < avg_rms * 0.3,
        "Large chunk RMS {} too far from average {}",
        rms_large,
        avg_rms
    );
}

#[test]
fn test_streaming_vs_batch_stereo_equivalence() {
    let sample_rate = 44100u32;
    let num_frames = sample_rate as usize;
    let mut signal = vec![0.0f32; num_frames * 2];
    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        signal[i * 2] = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        signal[i * 2 + 1] = (2.0 * std::f32::consts::PI * 880.0 * t).sin();
    }

    let params = StretchParams::new(1.3)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let batch_output = timestretch::stretch(&signal, &params).unwrap();
    let stream_output = stream_process_all(&signal, &params, 8192);

    // Both should have even length (stereo)
    assert_eq!(batch_output.len() % 2, 0, "Batch stereo output not even");
    assert_eq!(stream_output.len() % 2, 0, "Stream stereo output not even");

    // Both should have reasonable length ratios
    let batch_ratio = batch_output.len() as f64 / signal.len() as f64;
    let stream_ratio = stream_output.len() as f64 / signal.len() as f64;
    assert!(
        (batch_ratio - 1.3).abs() < 0.5,
        "Batch stereo length ratio {} too far from 1.3",
        batch_ratio
    );
    assert!(
        (stream_ratio - 1.3).abs() < 0.5,
        "Stream stereo length ratio {} too far from 1.3",
        stream_ratio
    );

    // Check that left and right channels have different content
    let batch_frames = batch_output.len() / 2;
    if batch_frames > 8192 {
        let left: Vec<f32> = batch_output.iter().step_by(2).copied().collect();
        let right: Vec<f32> = batch_output.iter().skip(1).step_by(2).copied().collect();
        let skip = 4096;
        let left_440 = dft_magnitude_at(&left[skip..left.len() - skip], 440.0, sample_rate);
        let right_880 = dft_magnitude_at(&right[skip..right.len() - skip], 880.0, sample_rate);
        assert!(left_440 > 0.01, "Left channel should have 440Hz content");
        assert!(right_880 > 0.01, "Right channel should have 880Hz content");
    }
}

#[test]
fn test_streaming_with_preset() {
    let sample_rate = 44100u32;
    let signal = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.02) // DJ beatmatch: 126 → 128.5 BPM
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let stream_output = stream_process_all(&signal, &params, 4096);
    assert!(!stream_output.is_empty(), "DJ preset streaming should produce output");

    // For ratio ~1.0, output length should be very close to input
    let ratio = stream_output.len() as f64 / signal.len() as f64;
    assert!(
        (ratio - 1.02).abs() < 0.3,
        "DJ preset length ratio {} too far from 1.02",
        ratio
    );
}

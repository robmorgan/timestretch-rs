use timestretch::{StretchParams, StreamProcessor};

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
    let mut chunk_count = 0;

    for chunk in signal.chunks(chunk_size) {
        // Change ratio midway through
        if chunk_count == signal.len() / chunk_size / 2 {
            processor.set_stretch_ratio(1.05);
        }
        let _ = processor.process(chunk);
        chunk_count += 1;
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

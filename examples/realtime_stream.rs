//! Real-time streaming example.
//!
//! Demonstrates the StreamProcessor API for chunk-based real-time processing,
//! simulating a DJ pitch fader that changes the stretch ratio on the fly.
//!
//! Run with: cargo run --example realtime_stream

use std::f32::consts::PI;
use timestretch::{EdmPreset, StreamProcessor, StretchParams};

fn main() {
    let sample_rate = 44100u32;

    // Set up streaming processor
    let params = StretchParams::new(1.0) // Start at original tempo
        .with_preset(EdmPreset::DjBeatmatch)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let mut processor = StreamProcessor::new(params);

    println!("Real-time Streaming Demo");
    println!(
        "Latency: {} samples ({:.1}ms)",
        processor.latency_samples(),
        processor.latency_secs() * 1000.0
    );

    // Generate 4 seconds of audio
    let total_samples = sample_rate as usize * 4;
    let input: Vec<f32> = (0..total_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 660.0 * t).sin()
        })
        .collect();

    // Process in 1024-sample chunks (simulating real-time buffer)
    let chunk_size = 1024;
    let mut total_output_samples = 0;
    let mut chunk_count = 0;

    // Simulate a DJ gradually pitching up from 1.0 to 1.05 over 4 seconds
    let mut current_ratio = 1.0;
    for chunk in input.chunks(chunk_size) {
        chunk_count += 1;
        let progress = chunk_count as f64 * chunk_size as f64 / total_samples as f64;

        // Gradually increase pitch
        current_ratio = 1.0 + 0.05 * progress;
        processor.set_stretch_ratio(current_ratio);

        let output = processor.process(chunk).expect("process failed");
        total_output_samples += output.len();
    }

    // Flush remaining samples
    let remaining = processor.flush().expect("flush failed");
    total_output_samples += remaining.len();

    println!(
        "Input:  {} samples ({:.2}s)",
        total_samples,
        total_samples as f64 / sample_rate as f64
    );
    println!(
        "Output: {} samples ({:.2}s)",
        total_output_samples,
        total_output_samples as f64 / sample_rate as f64
    );
    println!("Final ratio: {:.4}", current_ratio);
    println!("Chunks processed: {}", chunk_count);
}

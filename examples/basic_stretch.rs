//! Basic time stretching example.
//!
//! Demonstrates the simplest usage of the timestretch library:
//! generating a sine wave and stretching it by 1.5x.
//!
//! Run with: cargo run --example basic_stretch

use std::f32::consts::PI;

fn main() {
    let sample_rate = 44100u32;
    let duration_secs = 1.0f32;
    let freq = 440.0f32;

    // Generate a 440 Hz sine wave (1 second)
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let input: Vec<f32> = (0..num_samples)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
        .collect();

    println!("Input: {} samples ({:.2}s)", input.len(), duration_secs);

    // Stretch by 1.5x using default parameters
    let params = timestretch::StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = timestretch::stretch(&input, &params).expect("stretch failed");

    let output_duration = output.len() as f64 / sample_rate as f64;
    println!("Output: {} samples ({:.2}s)", output.len(), output_duration);
    println!(
        "Actual ratio: {:.3}",
        output.len() as f64 / input.len() as f64
    );
}

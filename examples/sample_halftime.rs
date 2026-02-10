//! Halftime effect example.
//!
//! Demonstrates the halftime effect (2x stretch) commonly used in
//! bass music, trap, and ambient transitions.
//!
//! Run with: cargo run --example sample_halftime

use std::f32::consts::PI;
use timestretch::{EdmPreset, StretchParams};

fn main() {
    let sample_rate = 44100u32;

    // Generate a drum pattern with bass
    let duration_secs = 2.0;
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let mut input = vec![0.0f32; num_samples];

    // Kick every 0.5 seconds (120 BPM)
    let kick_interval = (0.5 * sample_rate as f32) as usize;
    for kick_idx in 0..4 {
        let pos = kick_idx * kick_interval;
        for i in 0..4000 {
            if pos + i < num_samples {
                let t = i as f32 / sample_rate as f32;
                let freq = 120.0 - 70.0 * (t / 0.1);
                let env = (-t * 25.0).exp();
                input[pos + i] += 0.8 * env * (2.0 * PI * freq * t).sin();
            }
        }
    }

    // Hi-hat offbeat
    for hat_idx in 0..8 {
        let pos = (hat_idx as f32 * 0.25 * sample_rate as f32) as usize + kick_interval / 2;
        for i in 0..500 {
            if pos + i < num_samples {
                let t = i as f32 / sample_rate as f32;
                let env = (-t * 200.0).exp();
                // Noise-like hi-hat approximation
                let noise = ((i as f32 * 7919.0).sin() * 43_758.547).fract();
                input[pos + i] += 0.3 * env * noise;
            }
        }
    }

    println!("Halftime Effect Demo");
    println!(
        "Input:  {} samples ({:.2}s at 120 BPM)",
        input.len(),
        duration_secs
    );

    // Apply halftime (120 BPM -> 60 BPM) using the Halftime preset
    let params = StretchParams::new(1.0)
        .with_preset(EdmPreset::Halftime)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = timestretch::stretch_to_bpm(&input, 120.0, 60.0, &params).expect("stretch failed");

    let output_duration = output.len() as f64 / sample_rate as f64;
    let actual_ratio = output.len() as f64 / input.len() as f64;

    println!(
        "Output: {} samples ({:.2}s at ~60 BPM)",
        output.len(),
        output_duration
    );
    println!("Actual ratio: {:.3}", actual_ratio);

    // Compute RMS to verify energy preservation
    let input_rms =
        (input.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / input.len() as f64).sqrt();
    let output_rms =
        (output.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / output.len() as f64).sqrt();

    println!("Input RMS:  {:.4}", input_rms);
    println!("Output RMS: {:.4}", output_rms);
}

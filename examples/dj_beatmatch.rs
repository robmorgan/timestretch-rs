//! DJ beatmatching example.
//!
//! Demonstrates stretching a track from 126 BPM to 128 BPM,
//! a common operation in DJ mixing software.
//!
//! Run with: cargo run --example dj_beatmatch

use std::f32::consts::PI;
use timestretch::{EdmPreset, StretchParams};

fn main() {
    let sample_rate = 44100u32;
    let original_bpm = 126.0;
    let target_bpm = 128.0;
    let stretch_ratio = target_bpm / original_bpm;

    println!("DJ Beatmatch: {} BPM -> {} BPM", original_bpm, target_bpm);
    println!("Stretch ratio: {:.4}", stretch_ratio);

    // Generate a simple house-style pattern
    let duration_secs = 4.0;
    let beat_interval = 60.0 / original_bpm;
    let num_samples = (duration_secs * sample_rate as f32) as usize;

    let mut input = vec![0.0f32; num_samples];

    // Add kick drums on quarter notes
    let num_beats = (duration_secs / beat_interval as f32) as usize;
    for beat in 0..num_beats {
        let pos = (beat as f64 * beat_interval * sample_rate as f64) as usize;
        // Simple kick: sine sweep with decay
        for i in 0..((0.05 * sample_rate as f32) as usize) {
            if pos + i < num_samples {
                let t = i as f32 / sample_rate as f32;
                let freq = 150.0 - 100.0 * (t / 0.05);
                let env = (-t * 40.0).exp();
                input[pos + i] += 0.7 * env * (2.0 * PI * freq * t).sin();
            }
        }
    }

    // Add a pad tone
    for (i, sample) in input.iter_mut().enumerate().take(num_samples) {
        let t = i as f32 / sample_rate as f32;
        *sample += 0.15 * (2.0 * PI * 200.0 * t).sin();
    }

    // Stretch using DjBeatmatch preset
    let params = StretchParams::new(stretch_ratio)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = timestretch::stretch(&input, &params).expect("stretch failed");

    let output_duration = output.len() as f64 / sample_rate as f64;
    let actual_ratio = output.len() as f64 / input.len() as f64;

    println!("Input:  {} samples ({:.2}s)", input.len(), duration_secs);
    println!("Output: {} samples ({:.2}s)", output.len(), output_duration);
    println!("Actual ratio: {:.4}", actual_ratio);
    println!("Effective BPM: {:.1}", original_bpm * actual_ratio);
}

//! Real-time streaming example.
//!
//! Demonstrates the StreamProcessor API for chunk-based real-time processing,
//! simulating a DJ pitch fader that changes the tempo on the fly.
//!
//! Run with: cargo run --example realtime_stream

use std::f32::consts::PI;
use timestretch::{EdmPreset, StreamProcessor, StretchParams};

fn main() {
    let sample_rate = 44100u32;

    println!("=== Real-time Streaming Demo ===\n");

    // --- Part 1: Using from_tempo() for DJ BPM matching ---
    println!("Part 1: DJ BPM Matching (126 → 128 BPM)");
    let source_bpm = 126.0;
    let target_bpm = 128.0;

    let mut processor = StreamProcessor::from_tempo(source_bpm, target_bpm, sample_rate, 1);
    println!("  Source BPM: {}, Target BPM: {}", source_bpm, target_bpm);
    println!("  Stretch ratio: {:.4}", processor.current_stretch_ratio());
    println!(
        "  Latency: {} samples ({:.1}ms)",
        processor.latency_samples(),
        processor.latency_secs() * 1000.0
    );

    // Generate 4 seconds of audio (kick + pad)
    let total_samples = sample_rate as usize * 4;
    let input = generate_house_pattern(sample_rate, source_bpm, 4.0);

    let chunk_size = 1024;
    let mut total_output_samples = 0;

    for chunk in input.chunks(chunk_size) {
        let output = processor.process(chunk).expect("process failed");
        total_output_samples += output.len();
    }
    let remaining = processor.flush().expect("flush failed");
    total_output_samples += remaining.len();

    println!(
        "  Input:  {} samples ({:.2}s)",
        total_samples,
        total_samples as f64 / sample_rate as f64
    );
    println!(
        "  Output: {} samples ({:.2}s)",
        total_output_samples,
        total_output_samples as f64 / sample_rate as f64
    );

    // --- Part 2: Live tempo changes with set_tempo() ---
    println!("\nPart 2: Live DJ Tempo Fader (128 → 130 BPM over 4 seconds)");

    let mut processor = StreamProcessor::from_tempo(128.0, 128.0, sample_rate, 1);
    let input = generate_house_pattern(sample_rate, 128.0, 4.0);

    let mut total_output_samples = 0;
    let total_chunks = input.len() / chunk_size;
    let mut chunk_count = 0;

    for chunk in input.chunks(chunk_size) {
        chunk_count += 1;
        let progress = chunk_count as f64 / total_chunks as f64;

        // Smoothly ramp target BPM from 128 to 130
        let current_target = 128.0 + 2.0 * progress;
        processor.set_tempo(current_target);

        let output = processor.process(chunk).expect("process failed");
        total_output_samples += output.len();
    }
    let remaining = processor.flush().expect("flush failed");
    total_output_samples += remaining.len();

    println!(
        "  Final ratio: {:.4} (target: {:.4})",
        processor.current_stretch_ratio(),
        128.0 / 130.0
    );
    println!(
        "  Output: {} samples ({:.2}s)",
        total_output_samples,
        total_output_samples as f64 / sample_rate as f64
    );

    // --- Part 3: Manual ratio control ---
    println!("\nPart 3: Manual Ratio Control (gradual pitch up)");

    let params = StretchParams::new(1.0)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let mut processor = StreamProcessor::new(params);
    let input: Vec<f32> = (0..total_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 660.0 * t).sin()
        })
        .collect();

    let mut total_output_samples = 0;
    let total_chunks = input.len() / chunk_size;
    let mut chunk_count = 0;

    for chunk in input.chunks(chunk_size) {
        chunk_count += 1;
        let progress = chunk_count as f64 / total_chunks as f64;
        let ratio = 1.0 + 0.05 * progress;
        processor.set_stretch_ratio(ratio);

        let output = processor.process(chunk).expect("process failed");
        total_output_samples += output.len();
    }
    let remaining = processor.flush().expect("flush failed");
    total_output_samples += remaining.len();

    println!("  Final ratio: {:.4}", processor.current_stretch_ratio());
    println!(
        "  Output: {} samples ({:.2}s)",
        total_output_samples,
        total_output_samples as f64 / sample_rate as f64
    );
}

/// Generates a simple house-style pattern with kicks and a pad.
fn generate_house_pattern(sample_rate: u32, bpm: f64, duration_secs: f64) -> Vec<f32> {
    let beat_interval = 60.0 / bpm;
    let num_samples = (duration_secs * sample_rate as f64) as usize;
    let mut audio = vec![0.0f32; num_samples];

    // Kick drums on quarter notes
    let num_beats = (duration_secs / beat_interval) as usize;
    for beat in 0..num_beats {
        let pos = (beat as f64 * beat_interval * sample_rate as f64) as usize;
        let kick_len = (0.05 * sample_rate as f32) as usize;
        for i in 0..kick_len {
            if pos + i < num_samples {
                let t = i as f32 / sample_rate as f32;
                let freq = 150.0 - 100.0 * (t / 0.05);
                let env = (-t * 40.0).exp();
                audio[pos + i] += 0.7 * env * (2.0 * PI * freq * t).sin();
            }
        }
    }

    // Pad tone
    for (i, sample) in audio.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;
        *sample += 0.15 * (2.0 * PI * 200.0 * t).sin();
    }

    audio
}

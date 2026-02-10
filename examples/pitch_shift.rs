//! Pitch shifting example.
//!
//! Demonstrates changing the pitch of audio without altering its duration.
//! Useful for key-matching tracks in DJ sets or creative sound design.
//!
//! Run with: cargo run --example pitch_shift

use std::f32::consts::PI;
use timestretch::StretchParams;

fn main() {
    let sample_rate = 44100u32;

    // Generate a 2-second 440 Hz (A4) sine wave
    let duration_secs = 2.0;
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let input: Vec<f32> = (0..num_samples)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect();

    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    println!("Pitch Shift Demo");
    println!(
        "Input: {} samples ({:.2}s), 440 Hz (A4)\n",
        input.len(),
        duration_secs
    );

    // Shift up by one semitone (factor ≈ 1.0595)
    let semitone_up = 2.0f64.powf(1.0 / 12.0);
    let output =
        timestretch::pitch_shift(&input, &params, semitone_up).expect("pitch shift failed");
    println!(
        "+1 semitone (factor {:.4}): {} samples -> ~{:.0} Hz",
        semitone_up,
        output.len(),
        440.0 * semitone_up
    );

    // Shift up one octave (factor = 2.0)
    let output = timestretch::pitch_shift(&input, &params, 2.0).expect("pitch shift failed");
    println!(
        "+1 octave   (factor 2.0):    {} samples -> ~880 Hz",
        output.len()
    );

    // Shift down one octave (factor = 0.5)
    let output = timestretch::pitch_shift(&input, &params, 0.5).expect("pitch shift failed");
    println!(
        "-1 octave   (factor 0.5):    {} samples -> ~220 Hz",
        output.len()
    );

    // Shift down 5 semitones (perfect fourth down)
    let fourth_down = 2.0f64.powf(-5.0 / 12.0);
    let output =
        timestretch::pitch_shift(&input, &params, fourth_down).expect("pitch shift failed");
    println!(
        "-5 semitones (factor {:.4}): {} samples -> ~{:.0} Hz",
        fourth_down,
        output.len(),
        440.0 * fourth_down
    );

    println!(
        "\nAll outputs have the same length as input ({} samples).",
        input.len()
    );
}

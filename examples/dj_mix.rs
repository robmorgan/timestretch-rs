//! DJ Mix example: full workflow with resample, stretch, reverse, and crossfade.
//!
//! Demonstrates a realistic DJ mixing scenario:
//! 1. Two tracks at different sample rates and tempos
//! 2. Resample to common rate
//! 3. Stretch to target BPM
//! 4. Create a reverse cymbal build-up
//! 5. Crossfade tracks together
//!
//! Run with: cargo run --example dj_mix

use std::f32::consts::PI;
use timestretch::{AudioBuffer, EdmPreset, StretchParams};

fn main() {
    let target_bpm = 128.0;
    let target_rate = 44100u32;

    println!("=== DJ Mix Demo ===");
    println!("Target BPM: {target_bpm}");
    println!("Target sample rate: {target_rate} Hz\n");

    // --- Track A: 126 BPM house track at 48kHz ---
    let track_a_bpm = 126.0;
    let track_a_rate = 48000u32;
    println!("Track A: {track_a_bpm} BPM @ {track_a_rate} Hz");

    let track_a = generate_house_pattern(track_a_bpm, 4.0, track_a_rate);
    println!(
        "  Generated: {} frames ({:.2}s)",
        track_a.num_frames(),
        track_a.duration_secs()
    );

    // --- Track B: 130 BPM techno track at 44.1kHz ---
    let track_b_bpm = 130.0;
    let track_b_rate = 44100u32;
    println!("Track B: {track_b_bpm} BPM @ {track_b_rate} Hz");

    let track_b = generate_techno_pattern(track_b_bpm, 4.0, track_b_rate);
    println!(
        "  Generated: {} frames ({:.2}s)",
        track_b.num_frames(),
        track_b.duration_secs()
    );

    // --- Step 1: Resample both to 44.1kHz ---
    println!("\n--- Step 1: Resample to {target_rate} Hz ---");
    let track_a = track_a.resample(target_rate);
    println!(
        "  Track A resampled: {} frames ({:.2}s)",
        track_a.num_frames(),
        track_a.duration_secs()
    );
    // Track B is already at 44.1kHz, but let's be explicit
    let track_b = track_b.resample(target_rate);
    println!(
        "  Track B resampled: {} frames ({:.2}s)",
        track_b.num_frames(),
        track_b.duration_secs()
    );

    // --- Step 2: Stretch both to 128 BPM ---
    println!("\n--- Step 2: Stretch to {target_bpm} BPM ---");

    let params_a = StretchParams::from_tempo(track_a_bpm, target_bpm)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_sample_rate(target_rate)
        .with_channels(1);
    let stretched_a =
        timestretch::stretch_buffer(&track_a, &params_a).expect("stretch track A failed");
    println!(
        "  Track A: {:.1} BPM -> {:.1} BPM = {} frames ({:.2}s)",
        track_a_bpm,
        target_bpm,
        stretched_a.num_frames(),
        stretched_a.duration_secs()
    );

    let params_b = StretchParams::from_tempo(track_b_bpm, target_bpm)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_sample_rate(target_rate)
        .with_channels(1);
    let stretched_b =
        timestretch::stretch_buffer(&track_b, &params_b).expect("stretch track B failed");
    println!(
        "  Track B: {:.1} BPM -> {:.1} BPM = {} frames ({:.2}s)",
        track_b_bpm,
        target_bpm,
        stretched_b.num_frames(),
        stretched_b.duration_secs()
    );

    // --- Step 3: Create reverse cymbal build ---
    println!("\n--- Step 3: Reverse cymbal build ---");
    let cymbal = generate_cymbal(0.5, target_rate);
    let rev_cymbal = cymbal.reverse().fade_in(cymbal.num_frames());
    println!(
        "  Reverse cymbal: {} frames ({:.2}s)",
        rev_cymbal.num_frames(),
        rev_cymbal.duration_secs()
    );

    // --- Step 4: Split track A, append reverse cymbal to tail ---
    let cymbal_frames = rev_cymbal.num_frames();
    let (track_a_body, track_a_tail) =
        stretched_a.split_at(stretched_a.num_frames().saturating_sub(cymbal_frames));
    let tail_with_cymbal = track_a_tail.mix(&rev_cymbal);
    let track_a_final = track_a_body.crossfade_into(&tail_with_cymbal, 1000);
    println!(
        "  Track A with reverse build: {} frames ({:.2}s)",
        track_a_final.num_frames(),
        track_a_final.duration_secs()
    );

    // --- Step 5: Crossfade Track A into Track B ---
    println!("\n--- Step 4: Crossfade mix ---");
    let crossfade_ms = 100.0;
    let crossfade_frames = (crossfade_ms / 1000.0 * target_rate as f32) as usize;
    let final_mix = track_a_final.crossfade_into(&stretched_b, crossfade_frames);
    println!(
        "  Crossfade: {} frames ({:.0}ms)",
        crossfade_frames, crossfade_ms
    );
    println!(
        "  Final mix: {} frames ({:.2}s)",
        final_mix.num_frames(),
        final_mix.duration_secs()
    );

    // --- Stats ---
    println!("\n=== Result ===");
    println!("  Channels: {}", final_mix.channel_count());
    println!("  Sample rate: {} Hz", final_mix.sample_rate);
    println!("  Duration: {:.2}s", final_mix.duration_secs());
    println!("  Peak: {:.4}", final_mix.peak());
    println!("  RMS: {:.4}", final_mix.rms());

    if final_mix.peak() > 1.0 {
        println!("  (Normalizing to prevent clipping...)");
        let _normalized = final_mix.normalize(1.0);
    }

    println!("\nDone! In a real app, you'd write this to a WAV file or audio output.");
}

/// Generates a simple house pattern: kick on beats, hi-hat on offbeats, pad tone.
fn generate_house_pattern(bpm: f64, duration_secs: f32, sample_rate: u32) -> AudioBuffer {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let beat_interval = 60.0 / bpm;
    let mut samples = vec![0.0f32; num_samples];

    let num_beats = (duration_secs as f64 / beat_interval) as usize;

    for beat in 0..num_beats {
        let pos = (beat as f64 * beat_interval * sample_rate as f64) as usize;

        // Kick drum (sine sweep 150→50 Hz with exponential decay)
        for i in 0..((0.08 * sample_rate as f32) as usize) {
            if pos + i < num_samples {
                let t = i as f32 / sample_rate as f32;
                let freq = 150.0 - 100.0 * (t / 0.08);
                let env = (-t * 25.0).exp();
                samples[pos + i] += 0.6 * env * (2.0 * PI * freq * t).sin();
            }
        }

        // Hi-hat on offbeats
        let hat_pos = pos + (beat_interval * sample_rate as f64 / 2.0) as usize;
        for i in 0..((0.02 * sample_rate as f32) as usize) {
            if hat_pos + i < num_samples {
                let t = i as f32 / sample_rate as f32;
                let env = (-t * 200.0).exp();
                // Noise-like hi-hat approximation
                let noise = ((i as f32 * 7919.0).sin() * 43_758.547).fract() * 2.0 - 1.0;
                samples[hat_pos + i] += 0.15 * env * noise;
            }
        }
    }

    // Pad tone
    for (i, sample) in samples.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;
        *sample += 0.1 * (2.0 * PI * 200.0 * t).sin();
    }

    AudioBuffer::from_mono(samples, sample_rate)
}

/// Generates a simple techno pattern: harder kick, no hat, bass line.
fn generate_techno_pattern(bpm: f64, duration_secs: f32, sample_rate: u32) -> AudioBuffer {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let beat_interval = 60.0 / bpm;
    let mut samples = vec![0.0f32; num_samples];

    let num_beats = (duration_secs as f64 / beat_interval) as usize;

    for beat in 0..num_beats {
        let pos = (beat as f64 * beat_interval * sample_rate as f64) as usize;

        // Harder kick
        for i in 0..((0.1 * sample_rate as f32) as usize) {
            if pos + i < num_samples {
                let t = i as f32 / sample_rate as f32;
                let freq = 180.0 - 130.0 * (t / 0.1);
                let env = (-t * 20.0).exp();
                samples[pos + i] += 0.8 * env * (2.0 * PI * freq * t).sin();
            }
        }
    }

    // Bass line (simple 16th note pattern)
    let sixteenth = beat_interval / 4.0;
    let num_sixteenths = (duration_secs as f64 / sixteenth) as usize;
    for note in 0..num_sixteenths {
        if note % 4 == 2 || note % 4 == 3 {
            // Bass on 3rd and 4th sixteenths
            let pos = (note as f64 * sixteenth * sample_rate as f64) as usize;
            for i in 0..((sixteenth * sample_rate as f64 * 0.8) as usize) {
                if pos + i < num_samples {
                    let t = i as f32 / sample_rate as f32;
                    let env = (-t * 10.0).exp();
                    samples[pos + i] += 0.3 * env * (2.0 * PI * 55.0 * t).sin();
                }
            }
        }
    }

    AudioBuffer::from_mono(samples, sample_rate)
}

/// Generates a cymbal-like sound (noise burst with high-frequency content).
fn generate_cymbal(duration_secs: f32, sample_rate: u32) -> AudioBuffer {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let env = (-t * 4.0).exp();
        let noise = ((i as f32 * 7919.0).sin() * 43_758.547).fract() * 2.0 - 1.0;
        // Mix noise with high-frequency content
        let high = (2.0 * PI * 8000.0 * t).sin() * 0.3 + (2.0 * PI * 12000.0 * t).sin() * 0.2;
        samples.push(0.4 * env * (noise * 0.5 + high));
    }

    AudioBuffer::from_mono(samples, sample_rate)
}

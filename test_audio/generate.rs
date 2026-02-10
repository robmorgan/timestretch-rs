//! Test signal generator — creates WAV files for manual listening tests.
//!
//! Run with: `cargo run --example generate_test_audio`
//!
//! Generates the following files in `test_audio/`:
//! - `sine_440hz.wav` — 2s mono 440 Hz sine wave
//! - `sine_60hz.wav` — 2s mono 60 Hz sub-bass sine
//! - `sine_stereo.wav` — 2s stereo (L=440Hz, R=880Hz)
//! - `click_train_128bpm.wav` — 4s click train at 128 BPM
//! - `kick_pattern_128bpm.wav` — 4s EDM kick at 128 BPM
//! - `white_noise.wav` — 2s white noise
//! - `sweep_20_20k.wav` — 4s logarithmic frequency sweep 20Hz–20kHz
//! - `edm_mix.wav` — 4s layered kick + bass + hihat pattern

use std::f32::consts::PI;

const SAMPLE_RATE: u32 = 44100;
const TWO_PI: f32 = 2.0 * PI;

fn main() {
    let signals: Vec<(&str, Vec<f32>, u16)> = vec![
        ("sine_440hz", sine(440.0, 2.0), 1),
        ("sine_60hz", sine(60.0, 2.0), 1),
        ("sine_stereo", sine_stereo(440.0, 880.0, 2.0), 2),
        ("click_train_128bpm", click_train(128.0, 4.0), 1),
        ("kick_pattern_128bpm", kick_pattern(128.0, 4.0), 1),
        ("white_noise", white_noise(2.0), 1),
        ("sweep_20_20k", freq_sweep(20.0, 20000.0, 4.0), 1),
        ("edm_mix", edm_mix(128.0, 4.0), 1),
    ];

    for (name, data, channels) in &signals {
        let path = format!("test_audio/{}.wav", name);
        let wav = build_wav_float(data, SAMPLE_RATE, *channels);
        std::fs::write(&path, &wav).unwrap();
        let duration = data.len() as f32 / (SAMPLE_RATE as f32 * *channels as f32);
        println!(
            "  {} ({:.1}s, {} ch, {} samples)",
            path,
            duration,
            channels,
            data.len()
        );
    }
    println!("Done — {} files generated.", signals.len());
}

// ── Signal generators ────────────────────────────────────────────────────────

fn sine(freq: f32, duration: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * duration) as usize;
    (0..n)
        .map(|i| (TWO_PI * freq * i as f32 / SAMPLE_RATE as f32).sin())
        .collect()
}

fn sine_stereo(freq_l: f32, freq_r: f32, duration: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * duration) as usize;
    let mut data = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / SAMPLE_RATE as f32;
        data.push((TWO_PI * freq_l * t).sin());
        data.push((TWO_PI * freq_r * t).sin());
    }
    data
}

fn click_train(bpm: f32, duration: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * duration) as usize;
    let samples_per_beat = (SAMPLE_RATE as f32 * 60.0 / bpm) as usize;
    let click_len = (SAMPLE_RATE as f32 * 0.001) as usize; // 1ms click

    let mut data = vec![0.0f32; n];
    let mut pos = 0;
    while pos < n {
        for i in 0..click_len.min(n - pos) {
            // Short impulse with fast decay
            let env = (-(i as f32) / click_len as f32 * 10.0).exp();
            data[pos + i] = env * 0.8;
        }
        pos += samples_per_beat;
    }
    data
}

fn kick_pattern(bpm: f32, duration: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * duration) as usize;
    let samples_per_beat = (SAMPLE_RATE as f32 * 60.0 / bpm) as usize;
    let kick_len = (SAMPLE_RATE as f32 * 0.05) as usize; // 50ms kick

    let mut data = vec![0.0f32; n];
    let mut pos = 0;
    while pos < n {
        for i in 0..kick_len.min(n - pos) {
            let t = i as f32 / SAMPLE_RATE as f32;
            // Pitch-decaying sine (classic 808 kick)
            let freq = 150.0 * (-t * 40.0).exp() + 50.0;
            let env = (-t * 30.0).exp();
            data[pos + i] = env * (TWO_PI * freq * t).sin() * 0.9;
        }
        pos += samples_per_beat;
    }
    data
}

fn white_noise(duration: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * duration) as usize;
    // Simple LCG pseudo-random generator (deterministic, no deps)
    let mut seed: u64 = 42;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1, 1]
            ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn freq_sweep(start_hz: f32, end_hz: f32, duration: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * duration) as usize;
    let log_start = start_hz.ln();
    let log_end = end_hz.ln();

    let mut phase: f32 = 0.0;
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let freq = (log_start + (log_end - log_start) * t).exp();
            phase += TWO_PI * freq / SAMPLE_RATE as f32;
            if phase > TWO_PI {
                phase -= TWO_PI;
            }
            phase.sin() * 0.8
        })
        .collect()
}

fn edm_mix(bpm: f32, duration: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * duration) as usize;
    let samples_per_beat = (SAMPLE_RATE as f32 * 60.0 / bpm) as usize;
    let samples_per_8th = samples_per_beat / 2;

    let mut data = vec![0.0f32; n];

    for (i, sample) in data.iter_mut().enumerate() {
        let t = i as f32 / SAMPLE_RATE as f32;
        let beat_pos = i % samples_per_beat;
        let eighth_pos = i % samples_per_8th;

        // Kick: every beat
        if beat_pos < (SAMPLE_RATE as f32 * 0.05) as usize {
            let kt = beat_pos as f32 / SAMPLE_RATE as f32;
            let freq = 150.0 * (-kt * 40.0).exp() + 50.0;
            let env = (-kt * 30.0).exp();
            *sample += env * (TWO_PI * freq * kt).sin() * 0.5;
        }

        // Sub-bass: continuous 60 Hz sine
        *sample += (TWO_PI * 60.0 * t).sin() * 0.2;

        // Hi-hat: every 8th note (offset from kick)
        if eighth_pos < (SAMPLE_RATE as f32 * 0.01) as usize && beat_pos >= samples_per_8th {
            let ht = eighth_pos as f32 / SAMPLE_RATE as f32;
            let env = (-ht * 500.0).exp();
            // Use noise-like high-frequency content
            let mut seed: u64 = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
            *sample += env * noise * 0.15;
        }
    }

    // Soft-clip to prevent any samples exceeding 1.0
    for s in &mut data {
        *s = s.clamp(-1.0, 1.0);
    }

    data
}

// ── WAV writer (minimal, self-contained) ─────────────────────────────────────

fn build_wav_float(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let bits_per_sample: u16 = 32;
    let data_size = (samples.len() * 4) as u32;
    let byte_rate = sample_rate * channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = channels * (bits_per_sample / 8);
    let file_size = 36 + data_size;

    let mut out = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt chunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }

    out
}

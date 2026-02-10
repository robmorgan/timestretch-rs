//! Performance benchmarks for the timestretch library.
//!
//! Run with: cargo test --release --test benchmarks -- --nocapture

use std::f32::consts::PI;
use std::time::Instant;

use timestretch::{EdmPreset, StretchParams};

/// Generates a mono sine wave test signal.
fn generate_sine(sample_rate: u32, freq: f32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

/// Generates a stereo test signal (440 Hz left, 880 Hz right).
fn generate_stereo_signal(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_frames = (sample_rate as f32 * duration_secs) as usize;
    let mut data = Vec::with_capacity(num_frames * 2);
    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        data.push((2.0 * PI * 440.0 * t).sin());
        data.push((2.0 * PI * 880.0 * t).sin());
    }
    data
}

/// Generates a complex EDM-like test signal with kicks, bass, and hats.
fn generate_edm_signal(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut signal = vec![0.0f32; num_samples];

    let bpm = 128.0;
    let beat_interval = (sample_rate as f64 * 60.0 / bpm) as usize;

    for (i, sample) in signal.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;

        // Sub-bass: 60 Hz sine
        *sample += 0.3 * (2.0 * PI * 60.0 * t).sin();

        // Mid synth: 300 Hz with vibrato
        let vibrato = 5.0 * (2.0 * PI * 4.0 * t).sin();
        *sample += 0.2 * (2.0 * PI * (300.0 + vibrato) * t).sin();

        // Hi-hat: noise bursts every half-beat
        let half_beat = beat_interval / 2;
        let pos_in_half_beat = i % half_beat;
        if pos_in_half_beat < sample_rate as usize / 200 {
            // ~5ms noise burst
            *sample += 0.1 * (((i * 7 + 13) % 1000) as f32 / 500.0 - 1.0);
        }

        // Kick: every beat
        let pos_in_beat = i % beat_interval;
        if pos_in_beat < sample_rate as usize / 50 {
            // ~20ms kick
            let kick_t = pos_in_beat as f32 / sample_rate as f32;
            let kick_freq = 150.0 * (-kick_t * 40.0).exp() + 50.0;
            *sample += 0.5 * (2.0 * PI * kick_freq * kick_t).sin() * (-kick_t * 20.0).exp();
        }
    }

    // Normalize to prevent clipping
    let peak = signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let gain = 0.9 / peak;
        for s in signal.iter_mut() {
            *s *= gain;
        }
    }

    signal
}

struct BenchResult {
    name: String,
    duration_ms: f64,
    input_samples: usize,
    output_samples: usize,
    realtime_factor: f64,
    sample_rate: u32,
}

impl BenchResult {
    fn print(&self) {
        let input_duration_secs = self.input_samples as f64 / self.sample_rate as f64;
        println!(
            "  {:<45} {:>8.1}ms  {:>7} -> {:>7} samples  ({:.1}s audio)  {:.1}x realtime",
            self.name,
            self.duration_ms,
            self.input_samples,
            self.output_samples,
            input_duration_secs,
            self.realtime_factor,
        );
    }
}

fn bench_stretch(
    name: &str,
    input: &[f32],
    params: &StretchParams,
    iterations: usize,
) -> BenchResult {
    // Warmup
    let _ = timestretch::stretch(input, params);

    let start = Instant::now();
    let mut output_len = 0;
    for _ in 0..iterations {
        let output = timestretch::stretch(input, params).unwrap();
        output_len = output.len();
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let input_duration = input.len() as f64 / params.channels.count() as f64 / params.sample_rate as f64;
    let process_secs = avg_ms / 1000.0;
    let realtime_factor = input_duration / process_secs;

    BenchResult {
        name: name.to_string(),
        duration_ms: avg_ms,
        input_samples: input.len(),
        output_samples: output_len,
        realtime_factor,
        sample_rate: params.sample_rate,
    }
}

#[test]
fn bench_phase_vocoder_mono() {
    println!("\n=== Phase Vocoder Benchmarks (Mono) ===");

    let sample_rate = 44100;
    let signal = generate_sine(sample_rate, 440.0, 5.0);
    let iterations = 3;

    // Test different stretch ratios
    for (label, ratio) in &[
        ("PV mono 1.0x (identity)", 1.0),
        ("PV mono 1.02x (DJ beatmatch)", 1.02),
        ("PV mono 1.5x (moderate stretch)", 1.5),
        ("PV mono 2.0x (halftime)", 2.0),
        ("PV mono 0.5x (double time)", 0.5),
        ("PV mono 0.75x (speed up 25%)", 0.75),
    ] {
        let params = StretchParams::new(*ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);
        let result = bench_stretch(label, &signal, &params, iterations);
        result.print();
    }
}

#[test]
fn bench_phase_vocoder_stereo() {
    println!("\n=== Phase Vocoder Benchmarks (Stereo) ===");

    let sample_rate = 44100;
    let signal = generate_stereo_signal(sample_rate, 5.0);
    let iterations = 2;

    for (label, ratio) in &[
        ("PV stereo 1.0x (identity)", 1.0),
        ("PV stereo 1.5x (moderate stretch)", 1.5),
        ("PV stereo 2.0x (halftime)", 2.0),
    ] {
        let params = StretchParams::new(*ratio)
            .with_sample_rate(sample_rate)
            .with_channels(2);
        let result = bench_stretch(label, &signal, &params, iterations);
        result.print();
    }
}

#[test]
fn bench_edm_presets() {
    println!("\n=== EDM Preset Benchmarks (5s mono EDM signal) ===");

    let sample_rate = 44100;
    let signal = generate_edm_signal(sample_rate, 5.0);
    let iterations = 2;

    for (label, preset, ratio) in &[
        ("DjBeatmatch 1.02x", EdmPreset::DjBeatmatch, 1.02),
        ("HouseLoop 1.5x", EdmPreset::HouseLoop, 1.5),
        ("Halftime 2.0x", EdmPreset::Halftime, 2.0),
        ("Ambient 3.0x", EdmPreset::Ambient, 3.0),
        ("VocalChop 1.5x", EdmPreset::VocalChop, 1.5),
    ] {
        let params = StretchParams::new(*ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(*preset);
        let result = bench_stretch(label, &signal, &params, iterations);
        result.print();
    }
}

#[test]
fn bench_fft_sizes() {
    println!("\n=== FFT Size Comparison (5s mono sine, ratio 1.5x) ===");

    let sample_rate = 44100;
    let signal = generate_sine(sample_rate, 440.0, 5.0);
    let iterations = 2;

    for fft_size in &[1024, 2048, 4096, 8192] {
        let label = format!("FFT size {} (hop {})", fft_size, fft_size / 4);
        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_fft_size(*fft_size);
        let result = bench_stretch(&label, &signal, &params, iterations);
        result.print();
    }
}

#[test]
fn bench_streaming() {
    println!("\n=== Streaming Benchmarks ===");

    let sample_rate = 44100;
    let signal = generate_sine(sample_rate, 440.0, 10.0);

    for (label, chunk_size, ratio) in &[
        ("Stream 1024-sample chunks, 1.0x", 1024usize, 1.0),
        ("Stream 1024-sample chunks, 1.5x", 1024, 1.5),
        ("Stream 4096-sample chunks, 1.0x", 4096, 1.0),
        ("Stream 4096-sample chunks, 1.5x", 4096, 1.5),
    ] {
        let params = StretchParams::new(*ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let mut processor = timestretch::StreamProcessor::new(params);

        let start = Instant::now();
        let mut total_output = 0usize;

        for chunk in signal.chunks(*chunk_size) {
            if let Ok(output) = processor.process(chunk) {
                total_output += output.len();
            }
        }

        if let Ok(remaining) = processor.flush() {
            total_output += remaining.len();
        }

        let elapsed = start.elapsed();
        let process_ms = elapsed.as_secs_f64() * 1000.0;
        let input_duration = signal.len() as f64 / sample_rate as f64;
        let realtime_factor = input_duration / (process_ms / 1000.0);

        println!(
            "  {:<45} {:>8.1}ms  {:>7} -> {:>7} samples  ({:.1}s audio)  {:.1}x realtime",
            label,
            process_ms,
            signal.len(),
            total_output,
            input_duration,
            realtime_factor,
        );
    }
}

#[test]
fn bench_signal_lengths() {
    println!("\n=== Signal Length Scaling (mono 1.5x stretch) ===");

    let sample_rate = 44100;
    let iterations = 2;

    for duration_secs in &[1.0, 2.0, 5.0, 10.0] {
        let signal = generate_sine(sample_rate, 440.0, *duration_secs);
        let label = format!("{:.0}s signal ({} samples)", duration_secs, signal.len());
        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1);
        let result = bench_stretch(&label, &signal, &params, iterations);
        result.print();
    }
}

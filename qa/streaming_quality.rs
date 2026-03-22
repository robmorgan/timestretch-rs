//! Streaming quality benchmark: measures how close streaming output matches batch output.
//!
//! Run with: cargo test --features qa-harnesses --release --test streaming_quality -- --nocapture

use std::f32::consts::PI;
use timestretch::{
    analysis::comparison,
    stretch, EdmPreset, StreamProcessor, StretchParams,
};

const SAMPLE_RATE: u32 = 44_100;
const TWO_PI: f32 = 2.0 * PI;

/// Generate an EDM-like test signal with kicks, bass, hats, and pads.
fn generate_edm_signal(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut signal = vec![0.0f32; num_samples];
    let bpm = 128.0;
    let beat_interval = (sample_rate as f64 * 60.0 / bpm) as usize;

    for (i, sample) in signal.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;

        // Sub-bass: 60 Hz sine
        *sample += 0.3 * (TWO_PI * 60.0 * t).sin();

        // Mid synth: 300 Hz with vibrato
        let vibrato = 5.0 * (TWO_PI * 4.0 * t).sin();
        *sample += 0.2 * (TWO_PI * (300.0 + vibrato) * t).sin();

        // Hi-hat: noise bursts every half-beat
        let half_beat = beat_interval / 2;
        let pos_in_half_beat = i % half_beat;
        if pos_in_half_beat < sample_rate as usize / 200 {
            *sample += 0.1 * (((i * 7 + 13) % 1000) as f32 / 500.0 - 1.0);
        }

        // Kick: every beat
        let pos_in_beat = i % beat_interval;
        if pos_in_beat < sample_rate as usize / 50 {
            let kick_t = pos_in_beat as f32 / sample_rate as f32;
            let kick_freq = 150.0 * (-kick_t * 40.0).exp() + 50.0;
            *sample += 0.5 * (TWO_PI * kick_freq * kick_t).sin() * (-kick_t * 20.0).exp();
        }
    }

    // Normalize
    let peak = signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let gain = 0.9 / peak;
        for s in signal.iter_mut() {
            *s *= gain;
        }
    }
    signal
}

/// Generate a multi-tone harmonic signal (tonal bed).
fn generate_harmonic_signal(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let env = 0.9 + 0.1 * (TWO_PI * 0.3 * t).sin();
            env * (0.4 * (TWO_PI * 110.0 * t).sin()
                + 0.25 * (TWO_PI * 220.0 * t).sin()
                + 0.15 * (TWO_PI * 440.0 * t).sin()
                + 0.1 * (TWO_PI * 880.0 * t).sin())
        })
        .collect()
}

/// Generate a percussive signal (impulse train with decays).
fn generate_percussive_signal(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut signal = vec![0.0f32; num_samples];
    let bpm = 140.0;
    let beat_interval = (sample_rate as f64 * 60.0 / bpm) as usize;

    for (i, sample) in signal.iter_mut().enumerate() {
        let pos_in_beat = i % beat_interval;
        if pos_in_beat < sample_rate as usize / 40 {
            let t = pos_in_beat as f32 / sample_rate as f32;
            let env = (-t * 80.0).exp();
            *sample += 0.8 * env * (TWO_PI * (200.0 * (-t * 30.0).exp() + 40.0) * t).sin();
        }
        // Off-beat hi-hat
        let half_beat = beat_interval / 2;
        let pos_in_half = i % half_beat;
        if pos_in_half < sample_rate as usize / 300 {
            *sample += 0.15 * (((i * 13 + 7) % 1000) as f32 / 500.0 - 1.0);
        }
    }
    signal
}

/// Run streaming stretch and return full output.
fn stream_stretch(input: &[f32], params: StretchParams, chunk_size: usize) -> Vec<f32> {
    let ratio = params.stretch_ratio;
    let mut processor = StreamProcessor::new(params);
    // Pre-allocate generous capacity for streaming output
    let estimated_output = (input.len() as f64 * ratio * 1.5) as usize + 65536;
    let mut output = Vec::with_capacity(estimated_output);
    for chunk in input.chunks(chunk_size) {
        processor.process_into(chunk, &mut output).unwrap();
    }
    processor.flush_into(&mut output).unwrap();
    output
}

/// Compute RMS of a signal.
fn rms(signal: &[f32]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum: f64 = signal.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (sum / signal.len() as f64).sqrt()
}

struct QualityResult {
    signal_name: &'static str,
    ratio: f64,
    spectral_sim: f64,
    perceptual_sim: f64,
    cross_corr: f64,
    flux_sim: f64,
    rms_ratio: f64,
    length_error_pct: f64,
}

impl QualityResult {
    fn composite_score(&self) -> f64 {
        // Weighted composite similar to QualityReport grading:
        // 35% perceptual spectral, 25% cross-correlation, 20% spectral flux,
        // 15% spectral similarity, 5% length accuracy
        let length_score = (1.0 - self.length_error_pct / 100.0).clamp(0.0, 1.0);
        0.35 * self.perceptual_sim
            + 0.25 * self.cross_corr
            + 0.20 * self.flux_sim
            + 0.15 * self.spectral_sim
            + 0.05 * length_score
    }
}

fn evaluate_quality(
    signal_name: &'static str,
    input: &[f32],
    ratio: f64,
    preset: EdmPreset,
) -> QualityResult {
    let params = StretchParams::new(ratio)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(1)
        .with_preset(preset);

    // Batch output = reference
    let batch = stretch(input, &params).unwrap();
    // Streaming output = test
    let stream = stream_stretch(input, params, 1024);

    // Trim to common length for comparison
    let min_len = batch.len().min(stream.len());
    let batch_trimmed = &batch[..min_len];
    let stream_trimmed = &stream[..min_len];

    let fft_size = 2048;
    let hop_size = 512;

    let spectral_sim = comparison::spectral_similarity(stream_trimmed, batch_trimmed, fft_size, hop_size);
    let perceptual_sim = comparison::perceptual_spectral_similarity(
        stream_trimmed, batch_trimmed, fft_size, hop_size, SAMPLE_RATE,
    );
    let xcorr = comparison::cross_correlation(
        &stream_trimmed[..min_len.min(SAMPLE_RATE as usize * 5)],
        &batch_trimmed[..min_len.min(SAMPLE_RATE as usize * 5)],
    );
    let flux_sim = comparison::spectral_flux_similarity(stream_trimmed, batch_trimmed, fft_size, hop_size);

    let batch_rms = rms(batch_trimmed);
    let stream_rms = rms(stream_trimmed);
    let rms_ratio = if batch_rms > 1e-9 { stream_rms / batch_rms } else { 1.0 };

    let expected_len = (input.len() as f64 * ratio).round() as usize;
    let length_error_pct = ((stream.len() as f64 - expected_len as f64) / expected_len as f64 * 100.0).abs();

    QualityResult {
        signal_name,
        ratio,
        spectral_sim,
        perceptual_sim,
        cross_corr: xcorr.peak_value,
        flux_sim,
        rms_ratio,
        length_error_pct,
    }
}

#[test]
fn streaming_quality_benchmark() {
    let edm = generate_edm_signal(SAMPLE_RATE, 5.0);
    let harmonic = generate_harmonic_signal(SAMPLE_RATE, 5.0);
    let percussive = generate_percussive_signal(SAMPLE_RATE, 5.0);

    let cases: Vec<(&str, &[f32], f64, EdmPreset)> = vec![
        ("edm", &edm, 1.02, EdmPreset::DjBeatmatch),
        ("edm", &edm, 1.5, EdmPreset::HouseLoop),
        ("edm", &edm, 2.0, EdmPreset::Halftime),
        ("harmonic", &harmonic, 1.02, EdmPreset::DjBeatmatch),
        ("harmonic", &harmonic, 1.5, EdmPreset::HouseLoop),
        ("percussive", &percussive, 1.02, EdmPreset::DjBeatmatch),
        ("percussive", &percussive, 1.5, EdmPreset::HouseLoop),
        ("percussive", &percussive, 2.0, EdmPreset::Halftime),
    ];

    let mut total_composite = 0.0;
    let mut count = 0;

    println!("\n=== Streaming vs Batch Quality ===");
    println!(
        "{:<16} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>10}",
        "Signal", "Ratio", "Spectral", "Perceptual", "XCorr", "Flux", "RMS%", "Len%", "Composite"
    );
    println!("{}", "-".repeat(100));

    for (name, signal, ratio, preset) in &cases {
        let result = evaluate_quality(name, signal, *ratio, *preset);
        let composite = result.composite_score();
        total_composite += composite;
        count += 1;

        println!(
            "{:<16} {:>6.2} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>7.1}% {:>7.2}% {:>10.4}",
            result.signal_name,
            result.ratio,
            result.spectral_sim,
            result.perceptual_sim,
            result.cross_corr,
            result.flux_sim,
            result.rms_ratio * 100.0,
            result.length_error_pct,
            composite,
        );
    }

    let avg_composite = total_composite / count as f64;
    println!("{}", "-".repeat(100));
    println!("Average composite score: {:.4}", avg_composite);

    // Output structured metrics for autoresearch
    // Scale to 0-1000 for more readable metric (higher is better)
    let quality_score = avg_composite * 1000.0;
    println!("\nMETRIC quality_score={:.1}", quality_score);
}

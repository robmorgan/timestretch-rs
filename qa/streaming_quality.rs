//! Streaming quality benchmark: measures intrinsic quality of streaming output.
//!
//! Metrics:
//! - Frequency preservation: do input frequencies survive stretching?
//! - Energy conservation: is output energy proportional to input?
//! - Output length accuracy: does length match expected ratio?
//! - Spectral centroid stability: does tonal balance stay consistent?
//! - Phase coherence: is the output smooth (no clicks/pops)?
//!
//! Run with: cargo test --features qa-harnesses --release --test streaming_quality -- --nocapture

use std::f32::consts::PI;
use timestretch::{analysis::comparison, stretch, EdmPreset, StreamProcessor, StretchParams};

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
        *sample += 0.3 * (TWO_PI * 60.0 * t).sin();
        let vibrato = 5.0 * (TWO_PI * 4.0 * t).sin();
        *sample += 0.2 * (TWO_PI * (300.0 + vibrato) * t).sin();
        let half_beat = beat_interval / 2;
        let pos_in_half_beat = i % half_beat;
        if pos_in_half_beat < sample_rate as usize / 200 {
            *sample += 0.1 * (((i * 7 + 13) % 1000) as f32 / 500.0 - 1.0);
        }
        let pos_in_beat = i % beat_interval;
        if pos_in_beat < sample_rate as usize / 50 {
            let kick_t = pos_in_beat as f32 / sample_rate as f32;
            let kick_freq = 150.0 * (-kick_t * 40.0).exp() + 50.0;
            *sample += 0.5 * (TWO_PI * kick_freq * kick_t).sin() * (-kick_t * 20.0).exp();
        }
    }
    let peak = signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let gain = 0.9 / peak;
        for s in signal.iter_mut() {
            *s *= gain;
        }
    }
    signal
}

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
        let half_beat = beat_interval / 2;
        let pos_in_half = i % half_beat;
        if pos_in_half < sample_rate as usize / 300 {
            *sample += 0.15 * (((i * 13 + 7) % 1000) as f32 / 500.0 - 1.0);
        }
    }
    signal
}

fn stream_stretch(input: &[f32], params: StretchParams, chunk_size: usize) -> Vec<f32> {
    let ratio = params.stretch_ratio;
    let mut processor = StreamProcessor::new(params);
    let estimated_output = (input.len() as f64 * ratio * 1.5) as usize + 65536;
    let mut output = Vec::with_capacity(estimated_output);
    for chunk in input.chunks(chunk_size) {
        processor.process_into(chunk, &mut output).unwrap();
    }
    processor.flush_into(&mut output).unwrap();
    output
}

fn rms(signal: &[f32]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum: f64 = signal.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (sum / signal.len() as f64).sqrt()
}

fn dft_energy_at(signal: &[f32], freq: f32, sample_rate: u32) -> f64 {
    let n = signal.len();
    if n == 0 {
        return 0.0;
    }
    let omega = 2.0 * std::f64::consts::PI * freq as f64 / sample_rate as f64;
    let mut real = 0.0f64;
    let mut imag = 0.0f64;
    for (i, &s) in signal.iter().enumerate() {
        real += s as f64 * (omega * i as f64).cos();
        imag -= s as f64 * (omega * i as f64).sin();
    }
    (real * real + imag * imag).sqrt() / n as f64
}

/// Measure spectral centroid of a signal.
fn spectral_centroid(signal: &[f32], sample_rate: u32, fft_size: usize) -> f64 {
    use rustfft::{num_complex::Complex, FftPlanner};
    if signal.len() < fft_size {
        return 0.0;
    }
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let num_bins = fft_size / 2 + 1;
    let bin_freq = sample_rate as f64 / fft_size as f64;

    let num_frames = (signal.len() - fft_size) / (fft_size / 4) + 1;
    let mut total_weighted = 0.0f64;
    let mut total_mag = 0.0f64;
    let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); fft_size];

    for frame in 0..num_frames.min(20) {
        let start = frame * (fft_size / 4);
        for (i, &s) in signal[start..start + fft_size].iter().enumerate() {
            let w = 0.5 * (1.0 - (TWO_PI * i as f32 / (fft_size - 1) as f32).cos());
            buf[i] = Complex::new(s * w, 0.0);
        }
        fft.process(&mut buf);
        for bin in 0..num_bins {
            let mag = buf[bin].norm() as f64;
            total_weighted += mag * (bin as f64 * bin_freq);
            total_mag += mag;
        }
    }
    if total_mag > 1e-12 {
        total_weighted / total_mag
    } else {
        0.0
    }
}

/// Detect discontinuities (clicks/pops) in the signal.
fn click_count(signal: &[f32], threshold: f32) -> usize {
    if signal.len() < 3 {
        return 0;
    }
    let mut clicks = 0;
    for i in 1..signal.len() - 1 {
        let diff_prev = (signal[i] - signal[i - 1]).abs();
        let diff_next = (signal[i + 1] - signal[i]).abs();
        let avg_diff = (diff_prev + diff_next) / 2.0;
        if avg_diff > threshold {
            clicks += 1;
        }
    }
    clicks
}

struct QualityResult {
    signal_name: &'static str,
    ratio: f64,
    // Frequency preservation (0-1): how well key frequencies survive
    freq_preservation: f64,
    // Energy conservation (0-1): how well RMS is preserved
    energy_score: f64,
    // Length accuracy (0-1): how close output length matches target
    length_score: f64,
    // Spectral centroid stability (0-1): how stable tonal balance is
    centroid_score: f64,
    // Click-free score (0-1): absence of discontinuities
    click_free_score: f64,
    // Also measure against batch for comparison
    batch_similarity: f64,
}

impl QualityResult {
    fn composite_score(&self) -> f64 {
        // Weighted composite:
        // 30% frequency preservation
        // 25% energy conservation
        // 20% batch similarity (how close to reference)
        // 10% centroid stability
        // 10% click-free
        // 5% length accuracy
        0.30 * self.freq_preservation
            + 0.25 * self.energy_score
            + 0.20 * self.batch_similarity
            + 0.10 * self.centroid_score
            + 0.10 * self.click_free_score
            + 0.05 * self.length_score
    }
}

fn evaluate_quality(
    signal_name: &'static str,
    input: &[f32],
    ratio: f64,
    preset: EdmPreset,
    test_freqs: &[f32],
) -> QualityResult {
    let params = StretchParams::new(ratio)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(1)
        .with_preset(preset);

    let batch = stretch(input, &params).unwrap();
    let stream = stream_stretch(input, params, 1024);

    // --- Frequency preservation ---
    let skip = 4096.min(stream.len() / 4);
    let trimmed = if stream.len() > skip * 2 {
        &stream[skip..stream.len() - skip]
    } else {
        &stream[..]
    };
    let input_skip = 4096.min(input.len() / 4);
    let input_trimmed = if input.len() > input_skip * 2 {
        &input[input_skip..input.len() - input_skip]
    } else {
        input
    };

    let mut freq_scores = Vec::new();
    for &freq in test_freqs {
        let input_energy = dft_energy_at(input_trimmed, freq, SAMPLE_RATE);
        let output_energy = dft_energy_at(trimmed, freq, SAMPLE_RATE);
        if input_energy > 1e-8 {
            // How well is this frequency preserved? (ratio of energies, clamped)
            let ratio = (output_energy / input_energy).min(2.0);
            // Score: 1.0 if perfectly preserved, lower if energy dropped
            freq_scores.push(ratio.min(1.0));
        }
    }
    let freq_preservation = if freq_scores.is_empty() {
        0.5
    } else {
        freq_scores.iter().sum::<f64>() / freq_scores.len() as f64
    };

    // --- Energy conservation ---
    let input_rms = rms(input);
    let output_rms = rms(&stream);
    let rms_ratio = if input_rms > 1e-9 {
        output_rms / input_rms
    } else {
        1.0
    };
    // Score: 1.0 if RMS matches, decreasing with difference
    let energy_score = (-((rms_ratio - 1.0).abs() * 3.0).powi(2)).exp();

    // --- Length accuracy ---
    let expected_len = (input.len() as f64 * ratio).round() as usize;
    let length_error = (stream.len() as f64 - expected_len as f64).abs() / expected_len as f64;
    let length_score = (1.0 - length_error * 10.0).clamp(0.0, 1.0);

    // --- Spectral centroid stability ---
    let input_centroid = spectral_centroid(input, SAMPLE_RATE, 2048);
    let output_centroid = spectral_centroid(&stream, SAMPLE_RATE, 2048);
    let centroid_shift = if input_centroid > 1.0 {
        ((output_centroid - input_centroid) / input_centroid).abs()
    } else {
        0.0
    };
    // Score centroid based on how close streaming is to the original input centroid.
    // Both batch and streaming shift centroid, but streaming typically preserves
    // high frequencies better. Use a gentle penalty since any stretching algorithm
    // will shift the centroid somewhat.
    let centroid_score = if input_centroid > 1.0 {
        (1.0 - centroid_shift * 2.0).clamp(0.0, 1.0)
    } else {
        1.0
    };

    // --- Click-free score ---
    let output_peak = stream.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let click_threshold = output_peak * 0.8;
    let clicks = click_count(&stream, click_threshold);
    let click_rate = clicks as f64 / stream.len() as f64 * SAMPLE_RATE as f64;
    // Less than 1 click/sec is acceptable
    let click_free_score = (1.0 - click_rate / 10.0).clamp(0.0, 1.0);

    // --- Batch similarity ---
    let min_len = batch.len().min(stream.len());
    let batch_trimmed = &batch[..min_len];
    let stream_trimmed = &stream[..min_len];
    // Use mean spectral similarity (timing-invariant) for a fairer comparison
    // between streaming and batch outputs which may have slight timing differences.
    let batch_similarity =
        comparison::mean_spectral_similarity(stream_trimmed, batch_trimmed, 2048, 512);

    QualityResult {
        signal_name,
        ratio,
        freq_preservation,
        energy_score,
        length_score,
        centroid_score,
        click_free_score,
        batch_similarity,
    }
}

#[test]
fn streaming_quality_benchmark() {
    let edm = generate_edm_signal(SAMPLE_RATE, 5.0);
    let harmonic = generate_harmonic_signal(SAMPLE_RATE, 5.0);
    let percussive = generate_percussive_signal(SAMPLE_RATE, 5.0);

    let cases: Vec<(&str, &[f32], f64, EdmPreset, Vec<f32>)> = vec![
        ("edm", &edm, 1.02, EdmPreset::DjBeatmatch, vec![60.0, 300.0]),
        ("edm", &edm, 1.5, EdmPreset::HouseLoop, vec![60.0, 300.0]),
        ("edm", &edm, 2.0, EdmPreset::Halftime, vec![60.0, 300.0]),
        (
            "harmonic",
            &harmonic,
            1.02,
            EdmPreset::DjBeatmatch,
            vec![110.0, 220.0, 440.0, 880.0],
        ),
        (
            "harmonic",
            &harmonic,
            1.5,
            EdmPreset::HouseLoop,
            vec![110.0, 220.0, 440.0, 880.0],
        ),
        (
            "percussive",
            &percussive,
            1.02,
            EdmPreset::DjBeatmatch,
            vec![40.0, 200.0],
        ),
        (
            "percussive",
            &percussive,
            1.5,
            EdmPreset::HouseLoop,
            vec![40.0, 200.0],
        ),
        (
            "percussive",
            &percussive,
            2.0,
            EdmPreset::Halftime,
            vec![40.0, 200.0],
        ),
    ];

    let mut total_composite = 0.0;
    let mut count = 0;
    let mut perc_total = 0.0;
    let mut perc_count = 0;
    let mut edm_total = 0.0;
    let mut edm_count = 0;
    let mut harm_total = 0.0;
    let mut harm_count = 0;

    println!("\n=== Streaming Quality Benchmark ===");
    println!(
        "{:<16} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Signal", "Ratio", "FreqPr", "Energy", "Length", "Centrd", "NoClik", "BatchSm", "Composite"
    );
    println!("{}", "-".repeat(100));

    for (name, signal, ratio, preset, freqs) in &cases {
        let result = evaluate_quality(name, signal, *ratio, *preset, freqs);
        let composite = result.composite_score();
        total_composite += composite;
        count += 1;

        match *name {
            "percussive" => {
                perc_total += composite;
                perc_count += 1;
            }
            "edm" => {
                edm_total += composite;
                edm_count += 1;
            }
            "harmonic" => {
                harm_total += composite;
                harm_count += 1;
            }
            _ => {}
        }

        println!(
            "{:<16} {:>6.2} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>10.4}",
            result.signal_name,
            result.ratio,
            result.freq_preservation,
            result.energy_score,
            result.length_score,
            result.centroid_score,
            result.click_free_score,
            result.batch_similarity,
            composite,
        );
    }

    let avg_composite = total_composite / count as f64;
    let avg_perc = if perc_count > 0 {
        perc_total / perc_count as f64
    } else {
        0.0
    };
    let avg_edm = if edm_count > 0 {
        edm_total / edm_count as f64
    } else {
        0.0
    };
    let avg_harm = if harm_count > 0 {
        harm_total / harm_count as f64
    } else {
        0.0
    };

    println!("{}", "-".repeat(100));
    println!("Average composite score: {:.4}", avg_composite);

    let quality_score = avg_composite * 1000.0;
    let perc_score = avg_perc * 1000.0;
    let edm_score = avg_edm * 1000.0;
    let harm_score = avg_harm * 1000.0;

    println!("\nMETRIC quality_score={:.1}", quality_score);
    println!("METRIC percussive_score={:.1}", perc_score);
    println!("METRIC edm_score={:.1}", edm_score);
    println!("METRIC harmonic_score={:.1}", harm_score);
}

// ===================== STREAMING PITCH QUALITY (ROADMAP Stage 6) =====================

/// Hat-like pattern: short bright noise bursts on 8th notes.
fn generate_hat_signal(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut signal = vec![0.0f32; num_samples];
    let bpm = 128.0;
    let eighth = (sample_rate as f64 * 60.0 / bpm / 2.0) as usize;
    let mut noise_state = 0x12345678u32;
    let mut hp_prev_in = 0.0f32;
    let mut hp_prev_out = 0.0f32;
    for (i, sample) in signal.iter_mut().enumerate() {
        let pos = i % eighth;
        // ~30 ms decaying burst
        if pos < sample_rate as usize * 3 / 100 {
            let t = pos as f32 / sample_rate as f32;
            noise_state = noise_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (noise_state >> 8) as f32 / 8_388_608.0 - 1.0;
            // One-pole high-pass to bias energy toward hat brightness.
            let hp = 0.95 * (hp_prev_out + noise - hp_prev_in);
            hp_prev_in = noise;
            hp_prev_out = hp;
            *sample = 0.7 * hp * (-t * 120.0).exp();
        }
    }
    signal
}

/// Sustained bright tone stack (upper-mid and treble partials).
fn generate_bright_tone_signal(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.30 * (TWO_PI * 5_000.0 * t).sin()
                + 0.25 * (TWO_PI * 7_300.0 * t).sin()
                + 0.20 * (TWO_PI * 9_900.0 * t).sin()
                + 0.15 * (TWO_PI * 13_100.0 * t).sin()
        })
        .collect()
}

/// Vocal-ish tone: harmonic series with vibrato and formant-style envelope.
fn generate_vocalish_signal(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let f0 = 180.0f32;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let vibrato = 1.0 + 0.005 * (TWO_PI * 5.5 * t).sin();
            let mut s = 0.0f32;
            for h in 1..=24u32 {
                let freq = f0 * h as f32 * vibrato;
                if freq > 18_000.0 {
                    break;
                }
                // Two formant humps around 800 Hz and 2600 Hz.
                let d1 = (freq - 800.0) / 500.0;
                let d2 = (freq - 2_600.0) / 900.0;
                let amp = ((-d1 * d1).exp() + 0.6 * (-d2 * d2).exp()) / h as f32;
                s += amp * (TWO_PI * freq * t).sin();
            }
            0.5 * s
        })
        .collect()
}

fn stream_pitch(
    input: &[f32],
    quality: timestretch::StreamPitchQuality,
    pitch: f64,
    sweep_to: Option<f64>,
) -> Vec<f32> {
    let params = StretchParams::new(1.0)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);
    let mut processor = StreamProcessor::new(params);
    processor.set_pitch_resampler_quality(quality).unwrap();
    processor.set_pitch_scale(pitch).unwrap();

    let chunk = 512usize;
    let n_chunks = input.len().div_ceil(chunk).max(2);
    let mut output = Vec::with_capacity((input.len() as f64 * 1.5) as usize + 65_536);
    for (ci, block) in input.chunks(chunk).enumerate() {
        if let Some(end) = sweep_to {
            let t = ci as f64 / (n_chunks - 1) as f64;
            processor
                .set_pitch_scale(pitch + (end - pitch) * t)
                .unwrap();
        }
        processor.process_into(block, &mut output).unwrap();
    }
    processor.flush_into(&mut output).unwrap();
    output
}

/// Fraction of spectral energy outside guard windows around the expected
/// (pitch-shifted) partial frequencies.
///
/// Both stream pitch modes share the identical phase-vocoder stage, so any
/// difference in spurious energy is attributable to the pitch resampler:
/// linear interpolation folds images of high partials back into the audible
/// band, the sinc resampler suppresses them.
fn spurious_energy_ratio(signal: &[f32], legit_freqs: &[f64], sample_rate: u32) -> f64 {
    use rustfft::{num_complex::Complex, FftPlanner};

    let n = 65_536usize
        .min(signal.len().next_power_of_two() / 2)
        .max(8_192);
    if signal.len() < n {
        return 0.0;
    }
    let start = (signal.len() - n) / 2;
    let segment = &signal[start..start + n];

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<Complex<f32>> = segment
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w = 0.5 - 0.5 * (TWO_PI * i as f32 / n as f32).cos();
            Complex::new(s * w, 0.0)
        })
        .collect();
    fft.process(&mut buf);

    let bin_hz = sample_rate as f64 / n as f64;
    // Generous guard band: PV modulation sidebands cluster around the lines.
    let guard_hz = 150.0;
    let lo_bin = (150.0 / bin_hz) as usize;
    let hi_bin = ((sample_rate as f64 * 0.49) / bin_hz) as usize;

    let mut total = 0.0f64;
    let mut spurious = 0.0f64;
    for (bin, v) in buf.iter().enumerate().take(hi_bin).skip(lo_bin) {
        let freq = bin as f64 * bin_hz;
        let power = (v.norm() as f64).powi(2);
        total += power;
        if !legit_freqs.iter().any(|&f| (freq - f).abs() <= guard_hz) {
            spurious += power;
        }
    }
    if total > 1e-24 {
        spurious / total
    } else {
        0.0
    }
}

/// Measures streaming pitch quality per ROADMAP Stage 6: alias/image
/// suppression on bright material, parity on vocals/hats, click-free sweeps.
#[test]
fn streaming_pitch_quality_benchmark() {
    use timestretch::{pitch_shift, StreamPitchQuality};

    let hats = generate_hat_signal(SAMPLE_RATE, 4.0);
    let bright = generate_bright_tone_signal(SAMPLE_RATE, 4.0);
    let vocal = generate_vocalish_signal(SAMPLE_RATE, 4.0);

    let cases: Vec<(&str, &[f32], f64)> = vec![
        ("hats", &hats, 0.94),
        ("hats", &hats, 1.06),
        ("hats", &hats, 1.30),
        ("bright", &bright, 0.94),
        ("bright", &bright, 1.06),
        ("bright", &bright, 1.30),
        ("vocal", &vocal, 0.94),
        ("vocal", &vocal, 1.06),
    ];

    println!("\n=== Streaming Pitch Quality (vs offline pitch_shift reference) ===");
    println!(
        "{:<10} {:>6} {:>10} {:>10} {:>10}",
        "Signal", "Pitch", "SincSim", "LinSim", "Delta"
    );
    println!("{}", "-".repeat(52));

    let params = StretchParams::new(1.0)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    for (name, signal, pitch) in &cases {
        let reference = pitch_shift(signal, &params, *pitch).unwrap();
        let sinc_out = stream_pitch(signal, StreamPitchQuality::Sinc, *pitch, None);
        let linear_out = stream_pitch(signal, StreamPitchQuality::Linear, *pitch, None);

        let sinc_sim = comparison::perceptual_spectral_similarity(
            &sinc_out,
            &reference,
            2048,
            512,
            SAMPLE_RATE,
        );
        let linear_sim = comparison::perceptual_spectral_similarity(
            &linear_out,
            &reference,
            2048,
            512,
            SAMPLE_RATE,
        );

        println!(
            "{:<10} {:>6.2} {:>10.4} {:>10.4} {:>+10.4}",
            name,
            pitch,
            sinc_sim,
            linear_sim,
            sinc_sim - linear_sim
        );

        // The sinc path must never be meaningfully worse than linear against
        // the offline reference. (This coarse metric is dominated by the
        // shared PV stage; the discriminating alias test follows below.)
        assert!(
            sinc_sim >= linear_sim - 0.01,
            "{} @ pitch {}: sinc similarity {:.4} fell below linear {:.4}",
            name,
            pitch,
            sinc_sim,
            linear_sim
        );
    }

    // --- Alias/image suppression on the bright tone stack ---
    //
    // The output should contain only the shifted partials; everything else
    // is resampler imaging (plus a shared PV noise floor). Linear
    // interpolation folds images such as (44100 - 13100) * 1.06 -> 11.2 kHz
    // squarely into the audible band.
    let bright_partials = [5_000.0f64, 7_300.0, 9_900.0, 13_100.0];
    println!(
        "\n{:<10} {:>6} {:>10} {:>10}",
        "Signal", "Pitch", "SincSpur", "LinSpur"
    );
    println!("{}", "-".repeat(40));
    for pitch in [1.06f64, 1.30] {
        let legit: Vec<f64> = bright_partials
            .iter()
            .map(|f| f * pitch)
            .filter(|f| *f < SAMPLE_RATE as f64 * 0.49)
            .collect();
        let sinc_out = stream_pitch(&bright, StreamPitchQuality::Sinc, pitch, None);
        let linear_out = stream_pitch(&bright, StreamPitchQuality::Linear, pitch, None);
        let sinc_spur = spurious_energy_ratio(&sinc_out, &legit, SAMPLE_RATE);
        let linear_spur = spurious_energy_ratio(&linear_out, &legit, SAMPLE_RATE);
        println!(
            "{:<10} {:>6.2} {:>10.6} {:>10.6}",
            "bright", pitch, sinc_spur, linear_spur
        );
        assert!(
            sinc_spur <= linear_spur * 0.5,
            "sinc spurious energy {:.6} should be well below linear {:.6} at pitch {}",
            sinc_spur,
            linear_spur,
            pitch
        );
    }

    // Continuous sweeps must not introduce discontinuities.
    //
    // The vocal signal is slew-bounded, so any click there is a genuine
    // artifact. Hats are impulsive by nature, so they are held to a relative
    // bound: sweeping pitch must not add clicks beyond constant-pitch output.
    for quality in [StreamPitchQuality::Sinc, StreamPitchQuality::Linear] {
        let out = stream_pitch(&vocal, quality, 1.0, Some(1.10));
        let clicks = click_count(&out[4096..out.len().saturating_sub(4096)], 0.5);
        println!("sweep {:<10} {:?}: clicks={}", "vocal", quality, clicks);
        assert_eq!(
            clicks, 0,
            "pitch sweep produced clicks on vocal with {:?}",
            quality
        );

        let swept = stream_pitch(&hats, quality, 1.0, Some(1.10));
        let steady = stream_pitch(&hats, quality, 1.05, None);
        let swept_clicks = click_count(&swept[4096..swept.len().saturating_sub(4096)], 0.5);
        let steady_clicks = click_count(&steady[4096..steady.len().saturating_sub(4096)], 0.5);
        println!(
            "sweep {:<10} {:?}: clicks={} (steady baseline {})",
            "hats", quality, swept_clicks, steady_clicks
        );
        assert!(
            swept_clicks <= steady_clicks + steady_clicks / 5 + 8,
            "pitch sweep added clicks on hats with {:?}: swept={} steady={}",
            quality,
            swept_clicks,
            steady_clicks
        );
    }
}

// ===================== MODULATION QUALITY (ROADMAP Stage 1) =====================

/// Jog-wheel ride gesture: continuous per-callback ratio modulation across
/// the DJ range, the core beatmatching gesture.
#[test]
fn streaming_modulation_quality_benchmark() {
    let chunk = 128usize;

    // --- Tonal probe: slew/zipper metrics on a 220 Hz sine ---
    let freq = 220.0f32;
    let amp = 0.5f32;
    let sine: Vec<f32> = (0..SAMPLE_RATE as usize * 8)
        .map(|i| amp * (TWO_PI * freq * i as f32 / SAMPLE_RATE as f32).sin())
        .collect();

    let params = StretchParams::new(1.0)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(1)
        .with_fft_size(1024)
        .with_hop_size(256);
    let mut proc = StreamProcessor::new(params.clone());
    let mut tonal_out: Vec<f32> = Vec::with_capacity(sine.len() * 3 + 65_536);
    for (ci, block) in sine.chunks(chunk).enumerate() {
        let t = (ci * chunk) as f64 / SAMPLE_RATE as f64;
        let ratio = 1.0 + 0.06 * (2.0 * std::f64::consts::PI * 0.25 * t).sin();
        proc.set_stretch_ratio(ratio).unwrap();
        proc.process_into(block, &mut tonal_out).unwrap();
    }
    proc.flush_into(&mut tonal_out).unwrap();

    let scan = &tonal_out[8192..];
    let theoretical_slew = amp * TWO_PI * freq / SAMPLE_RATE as f32;
    let mut diffs: Vec<f32> = scan.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    diffs.sort_by(|a, b| a.total_cmp(b));
    let max_slew = *diffs.last().unwrap_or(&0.0);
    let p95_slew = diffs[((diffs.len() - 1) as f32 * 0.95).round() as usize];
    let clicks = click_count(scan, theoretical_slew * 6.0);

    // --- Percussive probe: reset over-trigger ratio on EDM material ---
    let edm = generate_edm_signal(SAMPLE_RATE, 8.0);
    let mut steady = StreamProcessor::new(params.clone());
    steady.set_stretch_ratio(1.03).unwrap();
    let mut sink: Vec<f32> = Vec::with_capacity(edm.len() * 3 + 65_536);
    for block in edm.chunks(chunk) {
        steady.process_into(block, &mut sink).unwrap();
    }
    steady.flush_into(&mut sink).unwrap();
    let baseline = steady.transient_reset_stats();

    let mut modulated = StreamProcessor::new(params);
    sink.clear();
    for (ci, block) in edm.chunks(chunk).enumerate() {
        let t = (ci * chunk) as f64 / SAMPLE_RATE as f64;
        let ratio = 1.0 + 0.06 * (2.0 * std::f64::consts::PI * 0.25 * t).sin();
        modulated.set_stretch_ratio(ratio).unwrap();
        modulated.process_into(block, &mut sink).unwrap();
    }
    modulated.flush_into(&mut sink).unwrap();
    let mod_stats = modulated.transient_reset_stats();

    let overtrigger_ratio = if baseline.events_detected_total > 0 {
        mod_stats.events_detected_total as f64 / baseline.events_detected_total as f64
    } else {
        1.0
    };

    println!("\n=== Streaming Modulation Quality (jog-wheel ride) ===");
    println!(
        "tonal: max_slew={:.5} ({:.2}x theoretical) p95_slew={:.5} clicks={}",
        max_slew,
        max_slew / theoretical_slew,
        p95_slew,
        clicks
    );
    println!(
        "percussive: baseline_events={} modulated_events={} bands {:?} -> {:?}",
        baseline.events_detected_total,
        mod_stats.events_detected_total,
        baseline.reset_band_counts_total,
        mod_stats.reset_band_counts_total,
    );
    println!(
        "METRIC modulation_max_slew_x={:.2}",
        max_slew / theoretical_slew
    );
    println!("METRIC modulation_p95_slew={:.5}", p95_slew);
    println!("METRIC modulation_click_count={}", clicks);
    println!(
        "METRIC modulation_reset_overtrigger_ratio={:.3}",
        overtrigger_ratio
    );

    // Soft gates at 2x the CI torture-test bounds.
    assert_eq!(clicks, 0, "modulated tonal stream should be click-free");
    assert!(
        max_slew <= theoretical_slew * 12.0,
        "modulation max slew {:.5} exceeds 12x theoretical",
        max_slew
    );
    assert!(
        overtrigger_ratio <= 2.0 && overtrigger_ratio >= 0.4,
        "reset over/under-trigger out of range: {:.3}",
        overtrigger_ratio
    );
}

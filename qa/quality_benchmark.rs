use std::f32::consts::PI;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use rustfft::{num_complex::Complex, FftPlanner};
use timestretch::stretch::phase_locking::PhaseLockingMode;
use timestretch::stretch::phase_vocoder::PhaseVocoder;
use timestretch::{pitch_shift, stretch, EdmPreset, EnvelopePreset, StretchParams, WindowType};

const SAMPLE_RATE: u32 = 44_100;
const FFT_SIZE: usize = 2_048;
const HOP_SIZE: usize = FFT_SIZE / 4;

#[derive(Clone)]
struct SignalCase {
    name: &'static str,
    samples: Vec<f32>,
    expected_onsets: Vec<usize>,
}

#[derive(Clone, Copy)]
enum Algorithm {
    BaselinePv,
    OverhauledHybrid,
}

impl Algorithm {
    fn as_str(self) -> &'static str {
        match self {
            Self::BaselinePv => "baseline_phase_vocoder",
            Self::OverhauledHybrid => "overhauled_hybrid",
        }
    }
}

#[derive(Clone, Copy)]
struct Metrics {
    transient_mae_ms: f64,
    spectral_distortion: f64,
    phase_coherence_std: f64,
    unexpected_energy_ratio: f64,
}

fn benchmark_output_dir() -> PathBuf {
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target"));
    target_dir.join("quality_benchmark")
}

fn synth_tone_stack(len: usize, sample_rate: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let s = 0.55 * (2.0 * PI * 110.0 * t).sin()
                + 0.30 * (2.0 * PI * 440.0 * t).sin()
                + 0.15 * (2.0 * PI * 880.0 * t).sin();
            s.clamp(-1.0, 1.0)
        })
        .collect()
}

fn synth_impulse_train(duration_secs: f32, sample_rate: u32, interval_secs: f32) -> SignalCase {
    let len = (duration_secs * sample_rate as f32).round() as usize;
    let mut out = vec![0.0f32; len];
    let mut onsets = Vec::new();
    let step = (interval_secs * sample_rate as f32).round() as usize;
    let decay_len = (0.01 * sample_rate as f32).round() as usize;

    let mut pos = 0usize;
    while pos < len {
        onsets.push(pos);
        for i in 0..decay_len {
            let idx = pos + i;
            if idx >= len {
                break;
            }
            let env = (-6.0 * i as f32 / decay_len as f32).exp();
            out[idx] += 0.95 * env;
        }
        pos = pos.saturating_add(step);
    }

    SignalCase {
        name: "impulse_train",
        samples: out,
        expected_onsets: onsets,
    }
}

fn synth_noise_bursts(duration_secs: f32, sample_rate: u32, interval_secs: f32) -> SignalCase {
    let len = (duration_secs * sample_rate as f32).round() as usize;
    let mut out = vec![0.0f32; len];
    let mut onsets = Vec::new();
    let step = (interval_secs * sample_rate as f32).round() as usize;
    let burst_len = (0.03 * sample_rate as f32).round() as usize;

    let mut state: u32 = 0x1f2e_3d4c;
    let mut pos = 0usize;
    while pos < len {
        onsets.push(pos);
        for i in 0..burst_len {
            let idx = pos + i;
            if idx >= len {
                break;
            }
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = ((state >> 8) as f32 / (u32::MAX >> 8) as f32) * 2.0 - 1.0;
            let env = 0.5 - 0.5 * (2.0 * PI * i as f32 / burst_len as f32).cos();
            out[idx] += 0.8 * noise * env;
        }
        pos = pos.saturating_add(step);
    }

    SignalCase {
        name: "noise_bursts",
        samples: out,
        expected_onsets: onsets,
    }
}

fn synth_vowel_like(duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let len = (duration_secs * sample_rate as f32).round() as usize;
    let f0 = 130.0f32;
    let formants = [700.0f32, 1200.0, 2600.0];
    let bandwidths = [120.0f32, 170.0, 220.0];
    let mut out = Vec::with_capacity(len);

    for i in 0..len {
        let t = i as f32 / sample_rate as f32;
        let mut s = 0.0f32;
        let mut k = 1usize;
        while (k as f32 * f0) < sample_rate as f32 * 0.45 {
            let freq = k as f32 * f0;
            let mut env = 0.0f32;
            for (&f, &bw) in formants.iter().zip(bandwidths.iter()) {
                let d = (freq - f) / bw;
                env += (-0.5 * d * d).exp();
            }
            s += (2.0 * PI * freq * t).sin() * env / k as f32;
            k += 1;
        }
        out.push(s);
    }

    let peak = out.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-6);
    for x in &mut out {
        *x = *x * 0.7 / peak;
    }
    out
}

fn write_wav(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(s.clamp(-1.0, 1.0))?;
    }
    writer.finalize()?;
    Ok(())
}

/// Output from an algorithm run, including channel count for WAV writing.
struct AlgorithmOutput {
    /// Raw output samples (interleaved if stereo).
    samples: Vec<f32>,
    /// Number of channels (1 = mono, 2 = stereo interleaved).
    channels: u16,
}

impl AlgorithmOutput {
    /// Extract a mono mixdown (left channel for stereo, identity for mono).
    fn to_mono(&self) -> Vec<f32> {
        if self.channels == 2 {
            self.samples.chunks(2).map(|ch| ch[0]).collect()
        } else {
            self.samples.clone()
        }
    }
}

fn run_algorithm(
    algo: Algorithm,
    input: &[f32],
    ratio: f64,
    sample_rate: u32,
) -> Result<AlgorithmOutput, Box<dyn std::error::Error>> {
    match algo {
        Algorithm::BaselinePv => {
            let mut pv = PhaseVocoder::with_options(
                FFT_SIZE,
                HOP_SIZE,
                ratio,
                sample_rate,
                120.0,
                WindowType::Hann,
                PhaseLockingMode::Identity,
            );
            Ok(AlgorithmOutput {
                samples: pv.process(input)?,
                channels: 1,
            })
        }
        Algorithm::OverhauledHybrid => {
            // Duplicate mono input to interleaved stereo
            let stereo_input: Vec<f32> = input.iter().flat_map(|&s| [s, s]).collect();
            let params = StretchParams::new(ratio)
                .with_sample_rate(sample_rate)
                .with_channels(2)
                .with_preset(EdmPreset::DjBeatmatch)
                .with_hop_size(4096 / 8);
            Ok(AlgorithmOutput {
                samples: stretch(&stereo_input, &params)?,
                channels: 2,
            })
        }
    }
}

fn detect_output_onsets(samples: &[f32], sample_rate: u32) -> Vec<usize> {
    if samples.is_empty() {
        return Vec::new();
    }
    let peak = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let threshold = peak * 0.35;
    let min_gap = (0.04 * sample_rate as f32) as usize;

    let mut out = Vec::new();
    let mut i = 1usize;
    while i + 1 < samples.len() {
        let v = samples[i].abs();
        if v >= threshold && v >= samples[i - 1].abs() && v >= samples[i + 1].abs() {
            out.push(i);
            i = i.saturating_add(min_gap);
            continue;
        }
        i += 1;
    }
    out
}

fn transient_mae_ms(expected_input_onsets: &[usize], output_onsets: &[usize], ratio: f64) -> f64 {
    if expected_input_onsets.is_empty() || output_onsets.is_empty() {
        return f64::NAN;
    }

    let mut sum_ms = 0.0f64;
    let mut count = 0usize;
    for &src in expected_input_onsets {
        let expected = src as f64 * ratio;
        let nearest = output_onsets.iter().map(|&o| o as f64).min_by(|a, b| {
            (a - expected)
                .abs()
                .partial_cmp(&(b - expected).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(found) = nearest {
            sum_ms += ((found - expected).abs() * 1000.0) / SAMPLE_RATE as f64;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum_ms / count as f64
    }
}

fn stft_magnitude(signal: &[f32], fft_size: usize, hop_size: usize) -> Vec<Vec<f32>> {
    if signal.len() < fft_size {
        return Vec::new();
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let window: Vec<f32> = (0..fft_size)
        .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / (fft_size - 1) as f32).cos())
        .collect();

    let num_bins = fft_size / 2 + 1;
    let mut frame = vec![Complex::new(0.0f32, 0.0f32); fft_size];
    let mut out = Vec::new();

    for start in (0..=signal.len() - fft_size).step_by(hop_size) {
        for i in 0..fft_size {
            frame[i] = Complex::new(signal[start + i] * window[i], 0.0);
        }
        fft.process(&mut frame);
        let mags = frame[..num_bins]
            .iter()
            .map(|c| c.norm())
            .collect::<Vec<_>>();
        out.push(mags);
    }
    out
}

fn spectral_distortion(ref_signal: &[f32], test_signal: &[f32]) -> f64 {
    let ref_spec = stft_magnitude(ref_signal, FFT_SIZE, HOP_SIZE);
    let test_spec = stft_magnitude(test_signal, FFT_SIZE, HOP_SIZE);
    if ref_spec.is_empty() || test_spec.is_empty() {
        return f64::NAN;
    }

    let frames = ref_spec.len().min(test_spec.len());
    let bins = ref_spec[0].len().min(test_spec[0].len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for t in 0..frames {
        for b in 0..bins {
            let a = ref_spec[t][b] as f64;
            let c = test_spec[t][b] as f64;
            num += (a - c).abs();
            den += a.abs();
        }
    }
    num / den.max(1e-12)
}

fn phase_at_freq(frame: &[f32], sample_rate: u32, freq_hz: f32) -> f64 {
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    for (n, &x) in frame.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * freq_hz as f64 * n as f64 / sample_rate as f64;
        re += x as f64 * angle.cos();
        im -= x as f64 * angle.sin();
    }
    im.atan2(re)
}

fn wrap_pi(mut x: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    while x > std::f64::consts::PI {
        x -= two_pi;
    }
    while x < -std::f64::consts::PI {
        x += two_pi;
    }
    x
}

fn dft_energy_at_freq(signal: &[f32], sample_rate: u32, freq_hz: f32) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    for (n, &x) in signal.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * freq_hz as f64 * n as f64 / sample_rate as f64;
        re += x as f64 * angle.cos();
        im += x as f64 * angle.sin();
    }
    (re * re + im * im).sqrt() / signal.len() as f64
}

fn formant_band_profile(signal: &[f32], sample_rate: u32) -> [f64; 3] {
    [
        dft_energy_at_freq(signal, sample_rate, 700.0),
        dft_energy_at_freq(signal, sample_rate, 1200.0),
        dft_energy_at_freq(signal, sample_rate, 2600.0),
    ]
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa <= 1e-12 || bb <= 1e-12 {
        0.0
    } else {
        dot / (aa.sqrt() * bb.sqrt())
    }
}

fn phase_coherence_std(signal: &[f32], sample_rate: u32) -> f64 {
    let frame_size = 2048usize;
    let hop = 512usize;
    let freqs = [110.0f32, 440.0f32, 880.0f32];
    if signal.len() < frame_size {
        return f64::NAN;
    }

    let mut diffs = Vec::new();
    for start in (0..=signal.len() - frame_size).step_by(hop) {
        let frame = &signal[start..start + frame_size];
        let p0 = phase_at_freq(frame, sample_rate, freqs[0]);
        let p1 = phase_at_freq(frame, sample_rate, freqs[1]);
        let p2 = phase_at_freq(frame, sample_rate, freqs[2]);
        diffs.push(wrap_pi(p0 - p1));
        diffs.push(wrap_pi(p1 - p2));
    }

    if diffs.len() < 2 {
        return f64::NAN;
    }
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let var = diffs
        .iter()
        .map(|d| {
            let e = d - mean;
            e * e
        })
        .sum::<f64>()
        / (diffs.len() as f64 - 1.0);
    var.sqrt()
}

fn unexpected_high_band_energy_ratio(signal: &[f32], sample_rate: u32) -> f64 {
    if signal.len() < 1024 {
        return f64::NAN;
    }
    let n = signal.len().next_power_of_two().min(1 << 16);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf = vec![Complex::new(0.0f32, 0.0f32); n];
    for (dst, &src) in buf.iter_mut().zip(signal.iter()) {
        *dst = Complex::new(src, 0.0);
    }
    fft.process(&mut buf);

    let nyquist = sample_rate as f64 / 2.0;
    let mut total = 0.0f64;
    let mut high = 0.0f64;
    let half = n / 2 + 1;
    for (k, c) in buf.iter().take(half).enumerate() {
        let freq = k as f64 * sample_rate as f64 / n as f64;
        let e = c.norm_sqr() as f64;
        total += e;
        if freq > 5_000.0 && freq < nyquist {
            high += e;
        }
    }
    high / total.max(1e-12)
}

fn write_spectrogram_csv(
    path: &Path,
    signal: &[f32],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = stft_magnitude(signal, FFT_SIZE, HOP_SIZE);
    let mut w = BufWriter::new(File::create(path)?);
    if spec.is_empty() {
        return Ok(());
    }

    let bins = spec[0].len();
    write!(w, "frame")?;
    for b in 0..bins {
        let freq = b as f64 * sample_rate as f64 / FFT_SIZE as f64;
        write!(w, ",{freq:.2}")?;
    }
    writeln!(w)?;

    for (t, mags) in spec.iter().enumerate() {
        write!(w, "{t}")?;
        for &m in mags {
            let db = 20.0 * (m.max(1e-12)).log10();
            write!(w, ",{db:.6}")?;
        }
        writeln!(w)?;
    }
    Ok(())
}

fn measure_metrics(case: &SignalCase, output: &[f32], ratio: f64) -> Metrics {
    let output_onsets = detect_output_onsets(output, SAMPLE_RATE);
    let transient_mae_ms = transient_mae_ms(&case.expected_onsets, &output_onsets, ratio);

    let target_len = (case.samples.len() as f64 * ratio).round() as usize;
    let ideal_tone = synth_tone_stack(target_len.max(1), SAMPLE_RATE);
    let spectral_distortion = if case.name == "tone_stack" {
        spectral_distortion(&ideal_tone, output)
    } else {
        f64::NAN
    };

    let phase_coherence_std = if case.name == "tone_stack" {
        phase_coherence_std(output, SAMPLE_RATE)
    } else {
        f64::NAN
    };

    let unexpected_energy_ratio = unexpected_high_band_energy_ratio(output, SAMPLE_RATE);

    Metrics {
        transient_mae_ms,
        spectral_distortion,
        phase_coherence_std,
        unexpected_energy_ratio,
    }
}

#[test]
#[ignore = "long-running quality benchmark harness"]
fn quality_benchmark_harness_generates_reports() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = benchmark_output_dir();
    fs::create_dir_all(&out_dir)?;

    let ratios = [0.5f64, 0.75, 1.5, 2.0, 4.0];
    let mut cases = vec![
        SignalCase {
            name: "tone_stack",
            samples: synth_tone_stack((2.0 * SAMPLE_RATE as f32) as usize, SAMPLE_RATE),
            expected_onsets: Vec::new(),
        },
        synth_impulse_train(2.0, SAMPLE_RATE, 0.5),
        synth_noise_bursts(2.0, SAMPLE_RATE, 0.75),
    ];

    // Add deterministic onsets for tone stack (reference anchors at section boundaries).
    cases[0].expected_onsets = vec![
        (0.0 * SAMPLE_RATE as f32) as usize,
        (0.5 * SAMPLE_RATE as f32) as usize,
        (1.0 * SAMPLE_RATE as f32) as usize,
        (1.5 * SAMPLE_RATE as f32) as usize,
    ];

    let mut summary = BufWriter::new(File::create(out_dir.join("quality_report.csv"))?);
    writeln!(
        summary,
        "signal,ratio,algorithm,transient_mae_ms,spectral_distortion,phase_coherence_std,unexpected_energy_ratio"
    )?;

    for case in &cases {
        write_wav(
            &out_dir.join(format!("input_{}.wav", case.name)),
            &case.samples,
            SAMPLE_RATE,
            1,
        )?;

        for &ratio in &ratios {
            for algo in [Algorithm::BaselinePv, Algorithm::OverhauledHybrid] {
                let result = run_algorithm(algo, &case.samples, ratio, SAMPLE_RATE)?;
                let mono_output = result.to_mono();
                let metrics = measure_metrics(case, &mono_output, ratio);

                let tag = format!("{}_{}_{ratio:.2}", algo.as_str(), case.name);
                write_wav(
                    &out_dir.join(format!("{tag}.wav")),
                    &result.samples,
                    SAMPLE_RATE,
                    result.channels,
                )?;
                write_spectrogram_csv(
                    &out_dir.join(format!("spectrogram_{tag}.csv")),
                    &mono_output,
                    SAMPLE_RATE,
                )?;

                writeln!(
                    summary,
                    "{},{:.2},{},{:.6},{:.6},{:.6},{:.6}",
                    case.name,
                    ratio,
                    algo.as_str(),
                    metrics.transient_mae_ms,
                    metrics.spectral_distortion,
                    metrics.phase_coherence_std,
                    metrics.unexpected_energy_ratio
                )?;
            }
        }
    }

    summary.flush()?;
    Ok(())
}

#[test]
#[ignore = "long-running quality benchmark harness"]
fn quality_benchmark_pitch_formant_presets_generates_reports(
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = benchmark_output_dir();
    fs::create_dir_all(&out_dir)?;

    let pitch_factors = [0.75f64, 1.35, 1.5, 2.0];
    let input = synth_vowel_like(2.0, SAMPLE_RATE);
    let input_profile = formant_band_profile(&input, SAMPLE_RATE);
    write_wav(
        &out_dir.join("input_vowel_like.wav"),
        &input,
        SAMPLE_RATE,
        1,
    )?;

    let mut summary = BufWriter::new(File::create(out_dir.join("pitch_formant_report.csv"))?);
    writeln!(
        summary,
        "pitch_factor,preset,formant_profile_similarity,spectral_distortion,unexpected_energy_ratio"
    )?;
    let mut delta = BufWriter::new(File::create(out_dir.join("pitch_formant_delta.csv"))?);
    writeln!(
        delta,
        "pitch_factor,delta_similarity_vocal_minus_off,delta_spectral_vocal_minus_off,delta_unexpected_energy_vocal_minus_off"
    )?;

    for &pitch_factor in &pitch_factors {
        let params_off = StretchParams::new(1.0)
            .with_sample_rate(SAMPLE_RATE)
            .with_channels(1)
            .with_envelope_preset(EnvelopePreset::Off);
        let params_vocal = StretchParams::new(1.0)
            .with_sample_rate(SAMPLE_RATE)
            .with_channels(1)
            .with_envelope_preset(EnvelopePreset::Vocal)
            .with_envelope_strength(1.4)
            .with_adaptive_envelope_order(true);

        let out_off = pitch_shift(&input, &params_off, pitch_factor)?;
        let out_vocal = pitch_shift(&input, &params_vocal, pitch_factor)?;

        let tag_off = format!("pitch_formant_off_{pitch_factor:.2}");
        let tag_vocal = format!("pitch_formant_vocal_{pitch_factor:.2}");
        write_wav(
            &out_dir.join(format!("{tag_off}.wav")),
            &out_off,
            SAMPLE_RATE,
            1,
        )?;
        write_wav(
            &out_dir.join(format!("{tag_vocal}.wav")),
            &out_vocal,
            SAMPLE_RATE,
            1,
        )?;
        write_spectrogram_csv(
            &out_dir.join(format!("spectrogram_{tag_off}.csv")),
            &out_off,
            SAMPLE_RATE,
        )?;
        write_spectrogram_csv(
            &out_dir.join(format!("spectrogram_{tag_vocal}.csv")),
            &out_vocal,
            SAMPLE_RATE,
        )?;

        let profile_off = formant_band_profile(&out_off, SAMPLE_RATE);
        let profile_vocal = formant_band_profile(&out_vocal, SAMPLE_RATE);
        let sim_off = cosine_similarity(&input_profile, &profile_off);
        let sim_vocal = cosine_similarity(&input_profile, &profile_vocal);
        let spectral_off = spectral_distortion(&input, &out_off);
        let spectral_vocal = spectral_distortion(&input, &out_vocal);
        let high_off = unexpected_high_band_energy_ratio(&out_off, SAMPLE_RATE);
        let high_vocal = unexpected_high_band_energy_ratio(&out_vocal, SAMPLE_RATE);

        writeln!(
            summary,
            "{:.2},off,{:.6},{:.6},{:.6}",
            pitch_factor, sim_off, spectral_off, high_off
        )?;
        writeln!(
            summary,
            "{:.2},vocal,{:.6},{:.6},{:.6}",
            pitch_factor, sim_vocal, spectral_vocal, high_vocal
        )?;

        let similarity_delta = sim_vocal - sim_off;
        assert!(
            sim_off.is_finite()
                && sim_vocal.is_finite()
                && spectral_off.is_finite()
                && spectral_vocal.is_finite()
                && high_off.is_finite()
                && high_vocal.is_finite(),
            "non-finite metric in pitch/formant benchmark at factor {pitch_factor}"
        );
        writeln!(
            delta,
            "{:.2},{:.6},{:.6},{:.6}",
            pitch_factor,
            similarity_delta,
            spectral_vocal - spectral_off,
            high_vocal - high_off
        )?;
    }

    summary.flush()?;
    delta.flush()?;

    Ok(())
}

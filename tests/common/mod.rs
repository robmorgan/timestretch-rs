use std::f32::consts::PI;

use timestretch::{StreamProcessor, StretchError, StretchParams};

pub fn gen_sine<F>(freq_hz: f32, sr: u32, n: usize, amp_fn: F) -> Vec<f32>
where
    F: Fn(usize) -> f32,
{
    (0..n)
        .map(|i| {
            let phase = 2.0 * PI * freq_hz * i as f32 / sr as f32;
            amp_fn(i) * phase.sin()
        })
        .collect()
}

pub fn gen_impulse_train(period: usize, n: usize, amp: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    if period == 0 {
        return out;
    }
    for i in (0..n).step_by(period) {
        out[i] = amp;
    }
    out
}

pub fn gen_two_tone(
    freq_a: f32,
    amp_a: f32,
    freq_b: f32,
    amp_b: f32,
    sr: u32,
    n: usize,
) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            amp_a * (2.0 * PI * freq_a * t).sin() + amp_b * (2.0 * PI * freq_b * t).sin()
        })
        .collect()
}

pub fn gen_click_pad(sr: u32, n: usize, click_positions: &[usize]) -> Vec<f32> {
    let mut out = gen_sine(220.0, sr, n, |_| 0.16);
    for &p in click_positions {
        if p < n {
            out[p] += 1.0;
        }
        if p + 1 < n {
            out[p + 1] -= 0.7;
        }
    }
    out
}

pub fn windowed_rms(signal: &[f32], start: usize, len: usize) -> f64 {
    if signal.is_empty() || len == 0 {
        return 0.0;
    }
    let start = start.min(signal.len());
    let end = (start + len).min(signal.len());
    if end <= start {
        return 0.0;
    }
    let sum_sq: f64 = signal[start..end]
        .iter()
        .map(|&s| {
            let v = s as f64;
            v * v
        })
        .sum();
    (sum_sq / (end - start) as f64).sqrt()
}

pub fn count_positive_zero_crossings(signal: &[f32], start: usize, end: usize) -> usize {
    if signal.len() < 2 {
        return 0;
    }
    let start = start.min(signal.len() - 1);
    let end = end.min(signal.len());
    if end <= start + 1 {
        return 0;
    }
    let mut count = 0usize;
    for i in start..(end - 1) {
        if signal[i] <= 0.0 && signal[i + 1] > 0.0 {
            count += 1;
        }
    }
    count
}

pub fn estimate_freq_zero_crossings(signal: &[f32], sr: u32, start: usize, end: usize) -> f64 {
    if end <= start + 1 {
        return 0.0;
    }
    let crossings = count_positive_zero_crossings(signal, start, end) as f64;
    let duration_secs = (end - start) as f64 / sr as f64;
    if duration_secs <= 0.0 {
        0.0
    } else {
        crossings / duration_secs
    }
}

pub fn best_lag_crosscorr(a: &[f32], b: &[f32], max_lag: usize) -> isize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let mut best_lag = 0isize;
    let mut best_score = f64::NEG_INFINITY;

    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let mut dot = 0.0f64;
        let mut a2 = 0.0f64;
        let mut b2 = 0.0f64;
        let mut n = 0usize;

        for (i, &av) in a.iter().enumerate() {
            let j = i as isize + lag;
            if j < 0 || j >= b.len() as isize {
                continue;
            }
            let bv = b[j as usize];
            let av64 = av as f64;
            let bv64 = bv as f64;
            dot += av64 * bv64;
            a2 += av64 * av64;
            b2 += bv64 * bv64;
            n += 1;
        }

        if n < 16 || a2 <= 0.0 || b2 <= 0.0 {
            continue;
        }
        let score = dot / (a2.sqrt() * b2.sqrt());
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    best_lag
}

pub fn rmse_with_lag(reference: &[f32], test: &[f32], lag: isize, start: usize, end: usize) -> f64 {
    if reference.is_empty() || test.is_empty() {
        return f64::INFINITY;
    }
    let end = end.min(reference.len());
    if end <= start {
        return f64::INFINITY;
    }

    let mut sum_sq = 0.0f64;
    let mut n = 0usize;
    for (i, &rv) in reference.iter().enumerate().take(end).skip(start) {
        let j = i as isize + lag;
        if j < 0 || j >= test.len() as isize {
            continue;
        }
        let diff = rv as f64 - test[j as usize] as f64;
        sum_sq += diff * diff;
        n += 1;
    }

    if n == 0 {
        f64::INFINITY
    } else {
        (sum_sq / n as f64).sqrt()
    }
}

pub fn energy_at_freq(signal: &[f32], sr: u32, freq_hz: f32) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    for (i, &s) in signal.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * freq_hz as f64 * i as f64 / sr as f64;
        let sv = s as f64;
        re += sv * angle.cos();
        im -= sv * angle.sin();
    }
    (re * re + im * im).sqrt() / signal.len() as f64
}

pub fn detect_peaks(signal: &[f32], threshold: f32, min_distance: usize) -> Vec<usize> {
    if signal.len() < 3 {
        return Vec::new();
    }
    let mut peaks = Vec::new();
    let mut last = usize::MAX / 2;
    for i in 1..(signal.len() - 1) {
        let v = signal[i].abs();
        if v < threshold {
            continue;
        }
        if v >= signal[i - 1].abs()
            && v >= signal[i + 1].abs()
            && (i >= last.saturating_add(min_distance))
        {
            peaks.push(i);
            last = i;
        }
    }
    peaks
}

pub fn run_streaming_mono(
    input: &[f32],
    params: StretchParams,
    chunk_size: usize,
    hybrid: bool,
) -> Result<Vec<f32>, StretchError> {
    let mut processor = StreamProcessor::new(params);
    if hybrid {
        processor.set_hybrid_mode(true);
    }

    let mut output = Vec::new();
    for chunk in input.chunks(chunk_size.max(1)) {
        let rendered = processor.process(chunk)?;
        output.extend_from_slice(&rendered);
    }
    let tail = processor.flush()?;
    output.extend_from_slice(&tail);
    Ok(output)
}

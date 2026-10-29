use super::*;
use timestretch::{PreparedSession, ProcessStatus, TempoRate};
const SAMPLE_RATE: u32 = 44100;
fn sine(freq: f64, len: usize, amp: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            amp * (2.0 * std::f64::consts::PI * freq * i as f64 / SAMPLE_RATE as f64).sin() as f32
        })
        .collect()
}

fn multitone(len: usize) -> Vec<f32> {
    const FREQS: [f64; 10] = [
        220.0, 440.0, 1_000.0, 2_300.0, 4_700.0, 6_100.0, 8_900.0, 11_700.0, 12_700.0, 15_600.0,
    ];
    (0..len)
        .map(|i| {
            let t = i as f64 / SAMPLE_RATE as f64;
            FREQS
                .iter()
                .enumerate()
                .map(|(k, &f)| {
                    0.06 * (2.0 * std::f64::consts::PI * (f * t + k as f64 * 0.37)).sin()
                })
                .sum::<f64>() as f32
        })
        .collect()
}

fn click_train(len: usize) -> Vec<f32> {
    let mut input = sine(220.0, len, 0.2);
    for start in (SAMPLE_RATE as usize / 4..len).step_by(SAMPLE_RATE as usize / 2) {
        for s in input.iter_mut().skip(start).take(24) {
            *s += 1.5;
        }
    }
    input
}

// --- metrics ----------------------------------------------------------------

fn zero_crossing_freq(window: &[f32]) -> Option<f64> {
    let (mut first, mut last, mut count) = (None, None, 0usize);
    for i in 1..window.len() {
        let (a, b) = (window[i - 1] as f64, window[i] as f64);
        if a <= 0.0 && b > 0.0 {
            let t = (i - 1) as f64 + a / (a - b);
            if first.is_none() {
                first = Some(t);
            }
            last = Some(t);
            count += 1;
        }
    }
    match (first, last) {
        (Some(f), Some(l)) if count >= 3 && l > f => {
            Some((count - 1) as f64 * SAMPLE_RATE as f64 / (l - f))
        }
        _ => None,
    }
}

/// p95 absolute cents deviation from `reference_hz` over the render.
fn cents_p95(output: &[f32], reference_hz: f64) -> f64 {
    let mut deviations = Vec::new();
    let mut pos = (SAMPLE_RATE / 2) as usize;
    while pos + 4_410 <= output.len() {
        if let Some(freq) = zero_crossing_freq(&output[pos..pos + 4_410]) {
            deviations.push((1_200.0 * (freq / reference_hz).log2()).abs());
        }
        pos += 1_102;
    }
    if deviations.is_empty() {
        return f64::NAN;
    }
    deviations.sort_by(|a, b| a.total_cmp(b));
    deviations[((deviations.len() - 1) as f64 * 0.95).round() as usize]
}

/// Median onset attack sharpness (max adjacent rise near expected
/// onsets). `latency` is the rendered chain's reported pipeline delay
/// (see `AbRender::latency_frames`) — never hardcode a profile's figure.
fn sharpness(output: &[f32], rate: f64, latency: usize) -> f64 {
    let mut scores = Vec::new();
    let mut k = 1usize;
    loop {
        let source_pos = SAMPLE_RATE as usize / 4 + k * SAMPLE_RATE as usize / 2;
        let expected = (source_pos as f64 / rate) as usize + latency;
        if expected + 2_048 >= output.len() {
            break;
        }
        let lo = expected.saturating_sub(1_024);
        let hi = expected + 1_024;
        let peak = output[lo..hi]
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(0.0f32, f32::max);
        scores.push(peak);
        k += 1;
    }
    if scores.is_empty() {
        return f64::NAN;
    }
    scores.sort_by(|a, b| a.total_cmp(b));
    scores[scores.len() / 2] as f64
}

/// Incoherent top-octave (12–16 kHz) power delta vs the source, in dB.
fn hf_retention_db(output: &[f32], source: &[f32]) -> f64 {
    fn band_power(signal: &[f32]) -> f64 {
        const WIN: usize = 2_205;
        const FREQS: [f64; 2] = [12_700.0, 15_600.0];
        let mut total = 0.0;
        for &f in &FREQS {
            let w = 2.0 * std::f64::consts::PI * f / SAMPLE_RATE as f64;
            let coeff = 2.0 * w.cos();
            let (mut sum, mut n) = (0.0f64, 0usize);
            let mut pos = SAMPLE_RATE as usize;
            while pos + WIN <= signal.len() {
                let (mut s1, mut s2) = (0.0f64, 0.0f64);
                for &x in &signal[pos..pos + WIN] {
                    let s0 = x as f64 + coeff * s1 - s2;
                    s2 = s1;
                    s1 = s0;
                }
                sum += (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (WIN as f64 / 2.0).powi(2);
                n += 1;
                pos += WIN;
            }
            total += sum / n.max(1) as f64;
        }
        total
    }
    10.0 * (band_power(output) / band_power(source)).log10()
}

/// Peak-to-trough envelope swing over 1024-sample RMS windows, in dB.
fn envelope_swing_db(output: &[f32]) -> f64 {
    const WIN: usize = 1_024;
    let (mut min_rms, mut max_rms) = (f64::MAX, 0.0f64);
    let mut pos = SAMPLE_RATE as usize;
    while pos + WIN <= output.len() {
        let rms = (output[pos..pos + WIN]
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum::<f64>()
            / WIN as f64)
            .sqrt();
        min_rms = min_rms.min(rms);
        max_rms = max_rms.max(rms);
        pos += WIN / 2;
    }
    20.0 * (max_rms / min_rms.max(1e-9)).log10()
}

/// Max adjacent diff over the theoretical tone slew (click indicator).
fn click_ratio(output: &[f32], tone_hz: f64, amp: f64, max_rate: f64) -> f64 {
    let bound = amp * 2.0 * std::f64::consts::PI * tone_hz * max_rate / SAMPLE_RATE as f64;
    let worst = output[(SAMPLE_RATE as usize)..]
        .windows(2)
        .map(|w| (w[1] - w[0]).abs() as f64)
        .fold(0.0, f64::max);
    worst / bound
}

/// RMS level delta vs source, in dB.
fn level_db(output: &[f32], source: &[f32]) -> f64 {
    fn rms(s: &[f32]) -> f64 {
        (s.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / s.len().max(1) as f64).sqrt()
    }
    let skip = SAMPLE_RATE as usize;
    20.0 * (rms(&output[skip.min(output.len())..]) / rms(&source[skip.min(source.len())..])).log10()
}

fn render_schedule(
    backend: Backend,
    input: &[f32],
    rate_at: &dyn Fn(f64) -> f64,
) -> Result<Vec<f32>> {
    let mut session = PreparedSession::new(SessionConfig {
        backend,
        channels: 1,
        source_frames: Some(input.len() as u64),
        max_commands: 256,
        parameters: Parameters::keylock(TempoRate::new(rate_at(0.0))?),
        ..Default::default()
    })?
    .activate();
    let mut next_command = 0;
    let mut output = Vec::with_capacity(input.len() * 4);
    let mut block = [0.0; 256];
    while !session.finished() {
        while next_command < session.commit_horizon() + 32768 {
            session.schedule(
                next_command / 256 + 1,
                next_command,
                Parameters::keylock(TempoRate::new(rate_at(
                    next_command as f64 / SAMPLE_RATE as f64,
                ))?),
            )?;
            next_command += 256;
        }
        match session.advance(4).status {
            ProcessStatus::NeedSource => {
                let r = session.source_request().ok_or("source request missing")?;
                let start = r.start_frame as usize;
                session.supply(r, &input[start..start + r.frames])?;
            }
            ProcessStatus::Fault => return Err(session.fault().unwrap().into()),
            _ => {}
        }
        while session.next_applied().is_some() {}
        let stamp = session.read_output(&mut block)?;
        output.extend_from_slice(&block[..stamp.frames]);
        session.acknowledge(stamp.consumed(stamp.frames)?)?;
    }
    Ok(output)
}
/// Same fixture definitions, evaluators and numerical bounds as the archived
/// product dashboard. Failed retained gates remain failures; no normalizing or
/// cross-correlation alignment is applied to make the new algorithm pass.
pub(super) fn run() -> Result<Value> {
    let tone = sine(440.0, SAMPLE_RATE as usize * 10, 0.7);
    let clicks = click_train(SAMPLE_RATE as usize * 10);
    let tones = multitone(SAMPLE_RATE as usize * 10);
    let envelope = sine(880.0, SAMPLE_RATE as usize * 16, 0.5);
    let middle = sine(660.0, SAMPLE_RATE as usize * 10, 0.5);
    let tau = std::f64::consts::TAU;
    let mut rows = Vec::new();
    for backend in [Backend::Portable, Backend::Accelerate] {
        if backend.resolve().is_err() {
            continue;
        }
        let mut add = |metric: &str, value: f64, bound: f64, higher: bool| {
            rows.push(json!({"backend":format!("{backend:?}"),"metric":metric,
            "value":value,"bound":bound,"higher_is_better":higher,"pass":value.is_finite()&&if higher{value>=bound}else{value<=bound},
            "threshold_source":"retained product-level quality bounds (unchanged)"}))
        };
        for (name, amplitude, bound) in [
            ("cents_p95_ride8", 0.08, 1.5),
            ("cents_p95_ride4", 0.04, 0.8),
        ] {
            add(
                name,
                cents_p95(
                    &render_schedule(backend, &tone, &|t| 1.0 + amplitude * (tau * t / 2.0).sin())?,
                    440.0,
                ),
                bound,
                false,
            );
        }
        for (name, rate, bound) in [
            ("cents_p95_steady106", 1.06, 1.0),
            ("range_cents_p95_steady150", 1.5, 0.5),
            ("range_cents_p95_steady070", 0.7, 1.0),
            ("range_cents_p95_steady200", 2.0, 0.8),
        ] {
            add(
                name,
                cents_p95(&render_schedule(backend, &tone, &|_| rate)?, 440.0),
                bound,
                false,
            );
        }
        add(
            "range_cents_p95_ride_full",
            cents_p95(
                &render_schedule(backend, &tone, &|t| 1.25 + 0.75 * (tau * 0.25 * t).sin())?,
                440.0,
            ),
            55.0,
            false,
        );
        for (name, rate, bound) in [
            ("sharpness_rate104", 1.04, 0.90),
            ("sharpness_rate096", 0.96, 0.90),
            ("range_sharpness_rate130", 1.30, 0.05),
            ("range_sharpness_rate070", 0.70, 0.15),
        ] {
            add(
                name,
                sharpness(&render_schedule(backend, &clicks, &|_| rate)?, rate, 0),
                bound,
                true,
            );
        }
        add(
            "hf_retention_db_108",
            hf_retention_db(&render_schedule(backend, &tones, &|_| 1.08)?, &tones),
            -1.0,
            true,
        );
        add(
            "env_swing_db_cross11",
            envelope_swing_db(&render_schedule(backend, &envelope, &|t| {
                1.0 + 0.11 * (tau * 0.25 * t).sin()
            })?),
            1.0,
            false,
        );
        add(
            "click_ratio_ride6",
            click_ratio(
                &render_schedule(backend, &middle, &|t| 1.0 + 0.06 * (tau * 0.25 * t).sin())?,
                660.0,
                0.5,
                1.0,
            ),
            1.3,
            false,
        );
        if let Ok(audio) = read_wav_file("test_audio/edm_mix.wav") {
            add(
                "level_db_steady106",
                level_db(
                    &render_schedule(backend, &audio.data, &|_| 1.06)?,
                    &audio.data,
                )
                .abs(),
                0.5,
                false,
            );
        }
    }
    Ok(json!(rows))
}

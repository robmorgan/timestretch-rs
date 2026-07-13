//! Full A/B dashboard: old engine vs new, every gated metric family, one
//! machine-readable report (ROADMAP Stage 7 machine side).
//!
//! Run with:
//! `cargo test --features qa-harnesses --release --test engine_ab_matrix -- --nocapture`
//!
//! Writes `ab_matrix.csv` to `TIMESTRETCH_QUALITY_DASHBOARD_DIR` (or
//! `target/ab_matrix/`) with one row per (metric, fixture, arm) plus a
//! parity verdict per metric (new vs old-Live on the same fixture; the
//! binding definition is new ≥ old on the same fixture and metric).
//! This harness REPORTS; the Stage 7 sign-off decides which rows become
//! hard gates and at what thresholds.

// Each harness compiles the shared adapter separately, so arms another
// harness uses read as dead code here.
#[allow(dead_code)]
#[path = "ab/mod.rs"]
mod ab;

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

use ab::{render_with_rate_schedule, Arm};
use timestretch::StreamProfile;

const SAMPLE_RATE: u32 = 44_100;
const CALLBACK_FRAMES: usize = 256;

// --- fixtures ---------------------------------------------------------------

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

/// Real corpus file, when present (test_audio is generated in CI).
fn corpus_wav(name: &str) -> Option<Vec<f32>> {
    let path = format!("test_audio/{name}");
    let mut reader = hound::WavReader::open(&path).ok()?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE {
        return None;
    }
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 * scale)
                .collect()
        }
    };
    Some(
        samples
            .chunks(channels)
            .map(|f| f.iter().sum::<f32>() / channels as f32)
            .collect(),
    )
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

/// Median onset attack sharpness (max adjacent rise near expected onsets).
fn sharpness(output: &[f32], rate: f64) -> f64 {
    let mut scores = Vec::new();
    let mut k = 1usize;
    loop {
        let source_pos = SAMPLE_RATE as usize / 4 + k * SAMPLE_RATE as usize / 2;
        let expected = (source_pos as f64 / rate) as usize + 560;
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

// --- matrix -----------------------------------------------------------------

struct Row {
    metric: &'static str,
    fixture: &'static str,
    arm: &'static str,
    value: f64,
    /// Whether larger is better (for the parity verdict).
    higher_better: bool,
}

fn arms() -> [(Arm, &'static str); 3] {
    [
        (Arm::OldVarispeed(StreamProfile::Live), "old-live"),
        (Arm::OldVarispeed(StreamProfile::Club), "old-club"),
        (Arm::NewKeylock, "new-keylock"),
    ]
}

fn render(arm: Arm, input: &[f32], rate_at: &dyn Fn(f64) -> f64) -> Vec<f32> {
    render_with_rate_schedule(arm, input, 1, SAMPLE_RATE, CALLBACK_FRAMES, rate_at).output
}

#[test]
fn ab_matrix_report() {
    let mut rows: Vec<Row> = Vec::new();

    // Pitch stability family (440 Hz tone).
    let tone = sine(440.0, SAMPLE_RATE as usize * 10, 0.7);
    let wide = |t: f64| 1.0 + 0.08 * (2.0 * std::f64::consts::PI * t / 2.0).sin();
    let dj = |t: f64| 1.0 + 0.04 * (2.0 * std::f64::consts::PI * t / 2.0).sin();
    for (arm, name) in arms() {
        rows.push(Row {
            metric: "cents_p95_ride8",
            fixture: "tone440",
            arm: name,
            value: cents_p95(&render(arm, &tone, &wide), 440.0),
            higher_better: false,
        });
        rows.push(Row {
            metric: "cents_p95_ride4",
            fixture: "tone440",
            arm: name,
            value: cents_p95(&render(arm, &tone, &dj), 440.0),
            higher_better: false,
        });
        rows.push(Row {
            metric: "cents_p95_steady106",
            fixture: "tone440",
            arm: name,
            value: cents_p95(&render(arm, &tone, &|_| 1.06), 440.0),
            higher_better: false,
        });
    }

    // Transient family (click train).
    let clicks = click_train(SAMPLE_RATE as usize * 10);
    for (arm, name) in arms() {
        for rate in [1.04f64, 0.96] {
            rows.push(Row {
                metric: if rate > 1.0 {
                    "sharpness_rate104"
                } else {
                    "sharpness_rate096"
                },
                fixture: "clicks",
                arm: name,
                value: sharpness(&render(arm, &clicks, &|_| rate), rate),
                higher_better: true,
            });
        }
    }

    // Top-octave retention (multitone @ 1.08).
    let tones = multitone(SAMPLE_RATE as usize * 10);
    for (arm, name) in arms() {
        rows.push(Row {
            metric: "hf_retention_db_108",
            fixture: "multitone",
            arm: name,
            value: hf_retention_db(&render(arm, &tones, &|_| 1.08), &tones),
            higher_better: true,
        });
    }

    // Envelope stability under a wide unity-crossing ride (880 Hz).
    let tone_hi = sine(880.0, SAMPLE_RATE as usize * 16, 0.5);
    let cross = |t: f64| 1.0 + 0.11 * (2.0 * std::f64::consts::PI * 0.25 * t).sin();
    for (arm, name) in arms() {
        rows.push(Row {
            metric: "env_swing_db_cross11",
            fixture: "tone880",
            arm: name,
            value: envelope_swing_db(&render(arm, &tone_hi, &cross)),
            higher_better: false,
        });
    }

    // Click-freeness under the DJ ride (keylocked pitch => rate factor 1).
    let tone_mid = sine(660.0, SAMPLE_RATE as usize * 10, 0.5);
    for (arm, name) in arms() {
        rows.push(Row {
            metric: "click_ratio_ride6",
            fixture: "tone660",
            arm: name,
            value: click_ratio(
                &render(arm, &tone_mid, &|t: f64| {
                    1.0 + 0.06 * (2.0 * std::f64::consts::PI * 0.25 * t).sin()
                }),
                660.0,
                0.5,
                1.0,
            ),
            higher_better: false,
        });
    }

    // Level integrity on corpus material (when present).
    if let Some(mix) = corpus_wav("edm_mix.wav") {
        let mix = &mix[..mix.len().min(SAMPLE_RATE as usize * 12)];
        for (arm, name) in arms() {
            rows.push(Row {
                metric: "level_db_steady106",
                fixture: "edm_mix",
                arm: name,
                value: level_db(&render(arm, mix, &|_| 1.06), mix),
                higher_better: false, // judged as |value|, see verdicts
            });
        }
    } else {
        println!("ab-matrix: test_audio/edm_mix.wav absent, skipping corpus rows");
    }

    // --- report ---------------------------------------------------------
    let dir = std::env::var("TIMESTRETCH_QUALITY_DASHBOARD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/ab_matrix"));
    fs::create_dir_all(&dir).expect("create report dir");
    let csv_path = dir.join("ab_matrix.csv");
    let mut csv = fs::File::create(&csv_path).expect("create csv");
    writeln!(csv, "metric,fixture,arm,value").unwrap();
    for row in &rows {
        writeln!(
            csv,
            "{},{},{},{}",
            row.metric, row.fixture, row.arm, row.value
        )
        .unwrap();
        println!(
            "ab-matrix: {:<22} {:<10} {:<12} {:>10.4}",
            row.metric, row.fixture, row.arm, row.value
        );
    }

    // Parity verdicts: new vs old-live per metric (same fixture).
    println!("--- parity (new-keylock vs old-live; binding: new >= old) ---");
    let mut verdicts = String::from("metric,fixture,old_live,new_keylock,new_wins_or_ties\n");
    let metrics: Vec<(&str, &str, bool)> = rows
        .iter()
        .filter(|r| r.arm == "old-live")
        .map(|r| (r.metric, r.fixture, r.higher_better))
        .collect();
    for (metric, fixture, higher_better) in metrics {
        let get = |arm: &str| {
            rows.iter()
                .find(|r| r.metric == metric && r.fixture == fixture && r.arm == arm)
                .map(|r| r.value)
                .unwrap_or(f64::NAN)
        };
        let old = get("old-live");
        let new = get("new-keylock");
        // Level rows judge distance from 0 dB; others by direction.
        let wins = if metric.starts_with("level_db") {
            new.abs() <= old.abs() + 0.25
        } else if higher_better {
            new >= old * 0.95
        } else {
            new <= old * 1.05 + 0.25
        };
        println!("parity: {metric:<22} {fixture:<10} old={old:>9.3} new={new:>9.3} ok={wins}");
        verdicts.push_str(&format!("{metric},{fixture},{old},{new},{wins}\n"));
        assert!(
            old.is_finite() && new.is_finite(),
            "matrix row not finite: {metric}/{fixture}"
        );
    }
    fs::write(dir.join("ab_matrix_parity.csv"), verdicts).expect("write parity csv");
    println!("ab-matrix: wrote {}", csv_path.display());
}

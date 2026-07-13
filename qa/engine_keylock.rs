//! Keylock chain quality gates for the new pull engine (ROADMAP new
//! Stage 2), ported from `qa/varispeed_keylock.rs` via the A/B adapter.
//!
//! Run with:
//! `cargo test --features qa-harnesses --release --test engine_keylock -- --nocapture`
//!
//! Gates:
//! - Cents-wobble under the ±8%/2 s torture ride: absolute gates at the old
//!   Live baseline (p95 ≤ 15 / max ≤ 22 cents) plus new ≤ old on the same
//!   fixture.
//! - Crossover-seam re-summation: no dip or beating at the band seam.
//!
//! The un-keylocked low-band falsification experiment's listening pairs are
//! rendered by the `#[ignore]` test at the bottom (see ROADMAP Stage 2).

// Each harness compiles the shared adapter separately, so arms another
// harness uses read as dead code here.
#[allow(dead_code)]
#[path = "ab/mod.rs"]
mod ab;

use ab::{render_with_rate_schedule, Arm};
use timestretch::StreamProfile;

const SAMPLE_RATE: u32 = 44_100;
const CALLBACK_FRAMES: usize = 256;
const REFERENCE_HZ: f64 = 440.0;
/// Analysis window (~100 ms) and hop (~25 ms) for the frequency track.
const ESTIMATE_WINDOW: usize = 4_410;
const ESTIMATE_HOP: usize = 1_102;
/// Output settle time excluded from the metric (pipeline fill + filters).
const WARMUP_SECS: f64 = 0.5;
/// Absolute gates inherited from the old engine's Live-profile baseline
/// (`qa/varispeed_keylock.rs`: p95 ≤ 15 / max ≤ 22 cents on this ride).
const P95_CENTS_LIMIT: f64 = 15.0;
const MAX_CENTS_LIMIT: f64 = 22.0;

fn sine(freq: f64, len: usize, amp: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            amp * (2.0 * std::f64::consts::PI * freq * i as f64 / SAMPLE_RATE as f64).sin() as f32
        })
        .collect()
}

/// Instantaneous frequency over a window from linearly-interpolated
/// positive-going zero crossings (sub-cent accuracy on a clean sine).
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

/// (p95, max) absolute deviation from the reference pitch, in cents.
fn cents_deviation_track(output: &[f32]) -> (f64, f64) {
    let start = (WARMUP_SECS * SAMPLE_RATE as f64) as usize;
    let mut deviations: Vec<f64> = Vec::new();
    let mut pos = start;
    while pos + ESTIMATE_WINDOW <= output.len() {
        if let Some(freq) = zero_crossing_freq(&output[pos..pos + ESTIMATE_WINDOW]) {
            deviations.push((1_200.0 * (freq / REFERENCE_HZ).log2()).abs());
        }
        pos += ESTIMATE_HOP;
    }
    assert!(
        deviations.len() > 50,
        "not enough frequency estimates: {}",
        deviations.len()
    );
    deviations.sort_by(|a, b| a.total_cmp(b));
    let p95 = deviations[((deviations.len() - 1) as f64 * 0.95).round() as usize];
    let max = *deviations.last().unwrap();
    (p95, max)
}

/// The qa torture ride: ±8% tempo over a 2 s period.
fn torture_ride(t: f64) -> f64 {
    1.0 + 0.08 * (2.0 * std::f64::consts::PI * t / 2.0).sin()
}

#[test]
fn keylock_cents_wobble_under_torture_ride() {
    // 440 Hz sits in the corrected high band; keylock must hold it at
    // 440 Hz (cents-level) while the tempo rides ±8%.
    let input = sine(REFERENCE_HZ, SAMPLE_RATE as usize * 10, 0.7);

    let new = render_with_rate_schedule(
        Arm::NewKeylock,
        &input,
        1,
        SAMPLE_RATE,
        CALLBACK_FRAMES,
        &torture_ride,
    );
    let old = render_with_rate_schedule(
        Arm::OldVarispeed(StreamProfile::Live),
        &input,
        1,
        SAMPLE_RATE,
        CALLBACK_FRAMES,
        &torture_ride,
    );

    let (new_p95, new_max) = cents_deviation_track(&new.output);
    let (old_p95, old_max) = cents_deviation_track(&old.output);
    println!(
        "keylock cents wobble: new p95={new_p95:.2} max={new_max:.2} | old Live p95={old_p95:.2} max={old_max:.2}"
    );

    // Absolute gates at the old Live baseline.
    assert!(
        new_p95 <= P95_CENTS_LIMIT,
        "keylock p95 wobble {new_p95:.2} cents > {P95_CENTS_LIMIT}"
    );
    assert!(
        new_max <= MAX_CENTS_LIMIT,
        "keylock max wobble {new_max:.2} cents > {MAX_CENTS_LIMIT}"
    );
    // Parity: new ≥ old on the same fixture and metric (lower is better);
    // small tolerance keeps measurement noise from flaking the gate.
    assert!(
        new_p95 <= old_p95 * 1.1 + 0.5,
        "new keylock p95 {new_p95:.2} worse than old {old_p95:.2}"
    );
    assert_eq!(new.underrun_frames, 0);
}

#[test]
fn keylock_cents_wobble_sola_band_ride() {
    // A ±4% ride stays inside the SOLA corrector's band (no threshold
    // crossings): time-domain correction plus correlation-matched splices
    // must hold pitch tighter than the PV does on the wide ride.
    let input = sine(REFERENCE_HZ, SAMPLE_RATE as usize * 10, 0.7);
    let render = render_with_rate_schedule(
        Arm::NewKeylock,
        &input,
        1,
        SAMPLE_RATE,
        CALLBACK_FRAMES,
        &|t| 1.0 + 0.04 * (2.0 * std::f64::consts::PI * t / 2.0).sin(),
    );
    let (p95, max) = cents_deviation_track(&render.output);
    println!("keylock SOLA-band ±4% ride: p95={p95:.2} max={max:.2} cents");
    assert!(
        p95 <= 6.0,
        "SOLA-band ride p95 {p95:.2} cents exceeds the time-domain budget"
    );
    assert!(max <= 10.0, "SOLA-band ride max {max:.2} cents");
}

#[test]
fn keylock_steady_ratio_is_subcent() {
    // At a steady ratio the delay-matched correction is exact; residual
    // wobble only appears under rides. The old path measured < 1 cent here.
    let input = sine(REFERENCE_HZ, SAMPLE_RATE as usize * 6, 0.7);
    let render = render_with_rate_schedule(
        Arm::NewKeylock,
        &input,
        1,
        SAMPLE_RATE,
        CALLBACK_FRAMES,
        &|_| 1.06,
    );
    let (p95, max) = cents_deviation_track(&render.output);
    println!("keylock steady 1.06: p95={p95:.3} max={max:.3} cents");
    assert!(
        p95 <= 1.0,
        "steady-ratio p95 {p95:.3} cents should be ~exact"
    );
    assert!(
        max <= 2.0,
        "steady-ratio max {max:.3} cents should be ~exact"
    );
}

/// (mean level dB vs source, max envelope deviation fraction) of a seam
/// tone rendered at `rate`.
fn seam_metrics(rate: f64) -> (f64, f64) {
    let seam_hz = 150.0;
    let input = sine(seam_hz, SAMPLE_RATE as usize * 8, 0.6);
    let render = render_with_rate_schedule(
        Arm::NewKeylock,
        &input,
        1,
        SAMPLE_RATE,
        CALLBACK_FRAMES,
        &|_| rate,
    );
    let out = &render.output;
    let start = (WARMUP_SECS * SAMPLE_RATE as f64) as usize;

    // Envelope over ~50 ms windows.
    let win = 2_205;
    let mut env: Vec<f64> = Vec::new();
    let mut pos = start;
    while pos + win <= out.len() {
        let rms = (out[pos..pos + win]
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum::<f64>()
            / win as f64)
            .sqrt();
        env.push(rms);
        pos += win / 2;
    }
    let mean = env.iter().sum::<f64>() / env.len() as f64;
    let input_rms = 0.6 / std::f64::consts::SQRT_2;
    let level_db = 20.0 * (mean / input_rms).log10();
    let max_dev = env
        .iter()
        .map(|&e| (e - mean).abs() / mean)
        .fold(0.0f64, f64::max);
    (level_db, max_dev)
}

#[test]
fn crossover_seam_resums_cleanly_at_unity() {
    // Re-summation CORRECTNESS gate: at unity rate the two bands carry the
    // same pitch, so any level dip or envelope modulation at the seam means
    // the low-band delay does not match the corrector (comb filtering) —
    // a wiring bug, not architecture.
    let (level_db, max_dev) = seam_metrics(1.0);
    println!(
        "seam @ unity: level {level_db:+.2} dB, max envelope deviation {:.1}%",
        max_dev * 100.0
    );
    assert!(
        level_db.abs() <= 1.0,
        "unity seam level {level_db:+.2} dB — delay mismatch / comb filtering"
    );
    assert!(
        max_dev <= 0.1,
        "unity seam envelope deviation {:.1}% — delay mismatch",
        max_dev * 100.0
    );
}

#[test]
fn crossover_seam_detuning_at_dj_rates_stays_in_known_envelope() {
    // At non-unity rates a tone exactly at the seam inherently exists as
    // two detuned copies (low band follows tempo, high band is corrected):
    // some level loss and beating there is a property of the un-keylocked
    // low band, bounded by the LR8 overlap width. This records the measured
    // envelope so regressions (e.g. a widened seam) are caught; whether the
    // effect is audible on real material is the Stage 2 falsification
    // listening question, not this gate.
    let (level_db, max_dev) = seam_metrics(1.06);
    println!(
        "seam @ 1.06: level {level_db:+.2} dB, max envelope deviation {:.1}% \
         (measured 2026-07 with the corrected LR8: -3.0 dB / 39% — the power \
         sum of two detuned copies; worst case: pure tone exactly at the seam)",
        max_dev * 100.0
    );
    assert!(
        level_db >= -4.5,
        "seam detuning level {level_db:+.2} dB regressed past the known envelope"
    );
    assert!(
        max_dev <= 0.55,
        "seam beating {:.1}% regressed past the known envelope",
        max_dev * 100.0
    );
}

/// Falsification experiment (ROADMAP Stage 2): renders listening pairs for
/// the un-keylocked low band on bass-heavy material at ±8% and ±20%.
///
/// For each rate: `new_keylock` (low band follows tempo) vs `old_keylock`
/// (old engine corrects the full band, sub-bass included). WAVs land in
/// `target/keylock_falsification/`. Run explicitly with:
/// `cargo test --features qa-harnesses --release --test engine_keylock -- --ignored --nocapture`
#[test]
#[ignore = "renders listening material; run explicitly"]
fn falsification_render_low_band_listening_pairs() {
    // Bass-heavy synthetic fixture: 55 Hz sub with kick-like 60 Hz thumps
    // and a light hat pattern so the bands' relationship is audible.
    let len = SAMPLE_RATE as usize * 12;
    let mut input = vec![0.0f32; len];
    for (i, s) in input.iter_mut().enumerate() {
        let t = i as f64 / SAMPLE_RATE as f64;
        *s += 0.45 * (2.0 * std::f64::consts::PI * 55.0 * t).sin() as f32;
        *s += 0.15 * (2.0 * std::f64::consts::PI * 220.0 * t).sin() as f32;
        *s += 0.1 * (2.0 * std::f64::consts::PI * 880.0 * t).sin() as f32;
    }
    // 120 BPM kicks: decaying 60 Hz bursts.
    let beat = SAMPLE_RATE as usize / 2;
    for start in (0..len).step_by(beat) {
        for k in 0..2_000.min(len - start) {
            let t = k as f64 / SAMPLE_RATE as f64;
            let env = (-t * 18.0).exp();
            input[start + k] += (0.8 * env * (2.0 * std::f64::consts::PI * 60.0 * t).sin()) as f32;
        }
    }

    let out_dir = std::path::Path::new("target/keylock_falsification");
    std::fs::create_dir_all(out_dir).expect("create output dir");

    write_wav(&out_dir.join("source.wav"), &input);
    for rate in [1.08, 0.92, 1.2, 0.8] {
        let new = render_with_rate_schedule(
            Arm::NewKeylock,
            &input,
            1,
            SAMPLE_RATE,
            CALLBACK_FRAMES,
            &|_| rate,
        );
        let old = render_with_rate_schedule(
            Arm::OldVarispeed(StreamProfile::Live),
            &input,
            1,
            SAMPLE_RATE,
            CALLBACK_FRAMES,
            &|_| rate,
        );
        let tag = format!("{:+.0}pct", (rate - 1.0) * 100.0);
        write_wav(&out_dir.join(format!("new_keylock_{tag}.wav")), &new.output);
        write_wav(&out_dir.join(format!("old_keylock_{tag}.wav")), &old.output);
        println!(
            "rendered {tag}: new {} frames, old {} frames",
            new.output.len(),
            old.output.len()
        );
    }
    println!("listening pairs in {}", out_dir.display());
}

fn write_wav(path: &std::path::Path, samples: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
    for &s in samples {
        writer
            .write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .expect("write sample");
    }
    writer.finalize().expect("finalize wav");
}

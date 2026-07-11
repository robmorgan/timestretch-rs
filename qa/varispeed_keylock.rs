//! Varispeed-first keylock pitch-stability gate (Stage 15).
//!
//! The varispeed input stage changes pitch instantly at the input and the
//! phase vocoder corrects it a buffering gate later, so any mismatch between
//! the correction and the audio actually being consumed reads as pitch
//! wobble during tempo rides. This harness streams a pure tone through a
//! fast DJ ratio ride and gates the instantaneous-frequency deviation in
//! cents.
//!
//! Run with:
//! `cargo test --features qa-harnesses --release --test varispeed_keylock -- --nocapture`

use timestretch::{ControlPath, StreamProcessor, StreamProfile, StretchParams};

const SAMPLE_RATE: u32 = 44_100;
const CHUNK: usize = 1024;
const REFERENCE_HZ: f64 = 440.0;
/// Analysis window (~100 ms) and hop (~25 ms) for the frequency track.
const ESTIMATE_WINDOW: usize = 4410;
const ESTIMATE_HOP: usize = 1102;
/// Output settle time excluded from the metric (gate fill + convergence).
const WARMUP_SECS: f64 = 0.5;
/// Absolute gates for the Live profile — the DJ control profile Stage 15
/// exists for — set from the measured floor with headroom (2026-07: Live
/// p95 ≈ 12, max ≈ 16 on this torture ride, versus ~100/137 on the vocoder
/// path). The ±8%/2 s pure-tone ride is far beyond real beatmatching
/// gestures — at a ±2% nudge the deviation scales down ~4x, well under the
/// ~5-10 cent pure-tone JND. The residual scales with the analysis window
/// (it is the PV's overlap-add under a moving ratio — ROADMAP Stage 16
/// targets it), so Club/Quality carry only the relative gate below; the
/// delay-matched correction itself is exact at steady ratio (< 1 cent).
const LIVE_P95_CENTS_LIMIT: f64 = 15.0;
const LIVE_MAX_CENTS_LIMIT: f64 = 22.0;
/// Relative gate (all profiles): the varispeed path must beat the vocoder
/// path's wobble by a clear margin — instant tempo control must not cost
/// pitch stability. (Measured 2026-07: Live 0.12x, Club 0.23x, Quality
/// 0.52x — the Quality baseline is itself smoother, so the margin there is
/// structurally narrower.)
const P95_BASELINE_FRACTION: f64 = 0.65;

fn sine(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            0.7 * (2.0 * std::f64::consts::PI * REFERENCE_HZ * i as f64 / SAMPLE_RATE as f64).sin()
                as f32
        })
        .collect()
}

/// Instantaneous frequency over a window from linearly-interpolated
/// positive-going zero crossings (sub-cent accuracy on a clean sine).
fn zero_crossing_freq(window: &[f32]) -> Option<f64> {
    let mut first: Option<f64> = None;
    let mut last: Option<f64> = None;
    let mut count = 0usize;
    for i in 1..window.len() {
        let (a, b) = (window[i - 1], window[i]);
        if a <= 0.0 && b > 0.0 {
            let t = (i - 1) as f64 + a as f64 / (a as f64 - b as f64);
            if first.is_none() {
                first = Some(t);
            }
            last = Some(t);
            count += 1;
        }
    }
    let (first, last) = (first?, last?);
    if count < 8 || last <= first {
        return None;
    }
    Some((count - 1) as f64 * SAMPLE_RATE as f64 / (last - first))
}

/// Streams the tone under the qa ratio ride (1.0 ± 0.08, one cycle per 2 s)
/// on the given control path and returns the output.
fn stream_ride(path: ControlPath, profile: StreamProfile) -> Vec<f32> {
    let input = sine(SAMPLE_RATE as usize * 6);
    let params = StretchParams::new(1.04)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(1)
        .with_stream_profile(profile);
    let mut processor = StreamProcessor::new(params);
    processor.set_control_path(path).expect("control path");

    let mut output = Vec::with_capacity(input.len() * 2);
    for (ci, chunk) in input.chunks(CHUNK).enumerate() {
        let t = (ci * CHUNK) as f64 / SAMPLE_RATE as f64;
        let ratio = 1.0 + 0.08 * (2.0 * std::f64::consts::PI * t / 2.0).sin();
        processor.set_stretch_ratio(ratio).expect("valid ratio");
        let rendered = processor.process(chunk).expect("stream chunk");
        output.extend_from_slice(&rendered);
    }
    output
}

/// Cents-deviation track of `signal` against the reference tone.
fn cents_track(signal: &[f32]) -> Vec<f64> {
    let start = (SAMPLE_RATE as f64 * WARMUP_SECS) as usize;
    let mut cents = Vec::new();
    let mut pos = start;
    while pos + ESTIMATE_WINDOW <= signal.len() {
        if let Some(freq) = zero_crossing_freq(&signal[pos..pos + ESTIMATE_WINDOW]) {
            cents.push(1200.0 * (freq / REFERENCE_HZ).log2());
        }
        pos += ESTIMATE_HOP;
    }
    cents
}

fn p95(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((values.len() as f64) * 0.95).ceil() as usize;
    values[idx.min(values.len() - 1).saturating_sub(0)]
}

#[test]
fn varispeed_keylock_pitch_wobble_under_ride() {
    for &profile in StreamProfile::ALL {
        // Vocoder-path baseline row: printed for comparison, not gated —
        // the PV preserves pitch by construction there, so its wobble is
        // pure vocoder artifact.
        let baseline = stream_ride(ControlPath::VocoderTempo, profile);
        let mut base_cents: Vec<f64> = cents_track(&baseline).iter().map(|c| c.abs()).collect();
        assert!(
            base_cents.len() >= 100,
            "baseline frequency track too sparse"
        );
        let base_max = base_cents.iter().cloned().fold(0.0, f64::max);
        let base_p95 = p95(&mut base_cents);

        let varispeed = stream_ride(ControlPath::VarispeedFirst, profile);
        let mut vs_cents: Vec<f64> = cents_track(&varispeed).iter().map(|c| c.abs()).collect();
        assert!(
            vs_cents.len() >= 100,
            "varispeed frequency track too sparse"
        );
        let vs_max = vs_cents.iter().cloned().fold(0.0, f64::max);
        let vs_p95 = p95(&mut vs_cents);

        println!(
            "METRIC keylock_cents_{} vocoder p95={:.2} max={:.2} | varispeed p95={:.2} max={:.2}",
            profile.label().to_lowercase(),
            base_p95,
            base_max,
            vs_p95,
            vs_max
        );

        if profile == StreamProfile::Live {
            assert!(
                vs_p95 <= LIVE_P95_CENTS_LIMIT,
                "Live varispeed keylock p95 wobble {:.2} cents exceeds {:.1}",
                vs_p95,
                LIVE_P95_CENTS_LIMIT
            );
            assert!(
                vs_max <= LIVE_MAX_CENTS_LIMIT,
                "Live varispeed keylock max wobble {:.2} cents exceeds {:.1}",
                vs_max,
                LIVE_MAX_CENTS_LIMIT
            );
        }
        assert!(
            vs_p95 <= base_p95 * P95_BASELINE_FRACTION,
            "{profile} varispeed p95 wobble {:.2} not clearly below vocoder baseline {:.2}",
            vs_p95,
            base_p95
        );
    }
}

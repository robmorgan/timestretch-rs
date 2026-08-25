//! The keylock chain's bass correction must not depend on the build
//! sample rate (regression, 2026-08-25, found via Halo at a 96 kHz
//! device): with the corrector's frame constants unscaled, a 96 kHz
//! build could not represent sub-74 Hz periods — a house kick/sub
//! fundamental was unsearchable, and as bassline content moved in and
//! out of the searchable range the low band flapped between corrected
//! (in key) and pitch-follow (−233 cents at −12.6%), an audible low-end
//! lurch several times per bar.

use timestretch::engine::{Engine, EngineConfig, EngineProfile};

/// Sustained −12.6% tempo — the reported DJ setting, past the ±8%
/// blind-validated envelope and well past full engagement.
const RATE: f64 = 0.874;

/// Goertzel power at `freq` over `scan`.
fn power_at(scan: &[f32], freq: f64, sample_rate: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * freq / sample_rate;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0);
    for &x in scan {
        let s0 = f64::from(x) + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let n = scan.len() as f64;
    (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (n * n / 4.0)
}

/// Streams the flap scenario — steady 55 Hz sub under an 85 Hz bassline
/// gated at 2 Hz — and returns (corrected windows, detuned windows):
/// 250 ms windows classified by whether the sub's energy sits at 55 Hz
/// (corrected) or 55·RATE Hz (pitch-follow).
fn sub_state_windows(sample_rate: u32) -> (usize, usize) {
    let handles = Engine::build(EngineConfig {
        sample_rate,
        channels: 1,
        profile: EngineProfile::Keylock,
        initial_tempo_rate: RATE,
        ..EngineConfig::default()
    })
    .unwrap();
    let (mut processor, mut source, _controller) =
        (handles.processor, handles.source, handles.controller);
    let sr = sample_rate as usize;
    let mut n_gen = 0usize;
    let (mut p_sub, mut p_bass) = (0.0f64, 0.0f64);
    let mut next_source = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                p_sub += 2.0 * std::f64::consts::PI * 55.0 / f64::from(sample_rate);
                p_bass += 2.0 * std::f64::consts::PI * 85.0 / f64::from(sample_rate);
                let gate = (n_gen % (sr / 2)) < sr / 4;
                n_gen += 1;
                (0.35 * p_sub.sin() + if gate { 0.45 * p_bass.sin() } else { 0.0 }) as f32
            })
            .collect()
    };
    let total_frames = 12 * sr;
    let mut out = Vec::with_capacity(total_frames);
    let mut chunk = vec![0.0f32; 1024];
    while out.len() < total_frames {
        while source.occupied_frames() < 8192 {
            let feed = next_source(2048);
            assert_eq!(source.push(&feed), feed.len(), "source ring saturated");
        }
        processor.process(&mut chunk);
        out.extend_from_slice(&chunk);
    }
    // Skip 4 s of engagement settling, then classify.
    let scan = &out[4 * sr..];
    let win = sr / 4;
    let (mut corrected, mut detuned) = (0usize, 0usize);
    for w in scan.chunks(win) {
        if w.len() < win {
            break;
        }
        if power_at(w, 55.0, f64::from(sample_rate))
            > power_at(w, 55.0 * RATE, f64::from(sample_rate))
        {
            corrected += 1;
        } else {
            detuned += 1;
        }
    }
    (corrected, detuned)
}

#[test]
fn sub_band_stays_corrected_at_96k() {
    for sample_rate in [44_100u32, 96_000] {
        let (corrected, detuned) = sub_state_windows(sample_rate);
        assert!(
            detuned == 0 && corrected > 0,
            "{sample_rate} Hz build: sub flapped out of correction \
             ({corrected} corrected / {detuned} detuned windows)"
        );
    }
}

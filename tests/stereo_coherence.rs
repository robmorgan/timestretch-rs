//! Product stereo thresholds retained from the archived coherence tests.
const SR: u32 = 44_100;

/// Renders `input` (interleaved stereo) through the wide profile at a
/// constant rate, returning interleaved output past latency + settle.
fn render_stereo(input: &[f32], rate: f64) -> Vec<f32> {
    let audio = timestretch::render(
        input,
        timestretch::SessionConfig {
            backend: timestretch::Backend::Portable,
            parameters: timestretch::Parameters::keylock(
                timestretch::TempoRate::new(rate).unwrap(),
            ),
            ..Default::default()
        },
    )
    .unwrap();
    // Retained evaluator's fixed 4096-frame settling exclusion, no alignment correction.
    audio.samples[8192.min(audio.samples.len())..].to_vec()
}

fn stereo_fixture(mono_gain_r: f32, frames: usize) -> Vec<f32> {
    (0..frames)
        .flat_map(|i| {
            let t = i as f64 / SR as f64;
            let s = (0.4 * (2.0 * std::f64::consts::PI * 330.0 * t).sin()
                + 0.2 * (2.0 * std::f64::consts::PI * 2_700.0 * t).sin())
                as f32;
            [s, s * mono_gain_r]
        })
        .collect()
}

/// Retain the product's 60 dB center/side rejection bound. The archived
/// bit-identity test specifically relied on the deleted M/S zero-side path;
/// timestretch's shared-reference L/R arithmetic has floating-point rounding differences.
#[test]
fn identical_channels_preserve_center_image() {
    for rate in [0.6, 1.5] {
        let input = stereo_fixture(1.0, SR as usize * 6);
        let out = render_stereo(&input, rate);
        let mut mid = 0.0f64;
        let mut side = 0.0f64;
        let mut peak = 0.0f32;
        for frame in out.chunks_exact(2) {
            mid += (f64::from(frame[0]) + f64::from(frame[1])).powi(2);
            side += (f64::from(frame[0]) - f64::from(frame[1])).powi(2);
            peak = peak.max((frame[0] - frame[1]).abs());
        }
        let rejection = 10.0 * (mid / side.max(1e-30)).log10();
        eprintln!(
            "identical input: tempo={rate} channel peak error={peak}, rejection={rejection} dB"
        );
        assert!(rejection > 60.0);
    }
}

/// Goertzel amplitude of `signal` at `freq`.
fn goertzel(signal: &[f32], freq: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * freq / SR as f64;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &x in signal {
        let s0 = x as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    ((s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0)).sqrt() / (signal.len() as f64 / 2.0)
}

/// The discriminating gate for per-channel phase divergence: a CENTER
/// component under DIFFERENT side content per channel. With independent
/// L/R vocoders, L's and R's peak landscapes differ, the shared center's
/// phase evolves differently in each, and center energy leaks into the
/// side channel (image wander/width modulation). In M/S the center lives
/// in M under one vocoder and cannot leak by construction. Honest
/// measurement on this fixture: the per-channel leak was modest — 64.6 dB
/// rejection before, 70.9 dB after — so this gate is a regression
/// tripwire, and the audible verdict on width belongs to the owner
/// listen (Stage 11 heard ours "a bit crowded" vs R3).
#[test]
fn center_component_does_not_leak_into_side() {
    let frames = SR as usize * 6;
    // Independent noise beds per channel keep the peak landscape around
    // the center bin different and MOVING in L vs R — steady side tones
    // cannot discriminate (a strong center is always its own locked
    // peak and never diverges).
    let mut seed_l = 0x9e3779b97f4a7c15u64;
    let mut seed_r = 0x2545f4914f6cdd1du64;
    let noise = move |seed: &mut u64| -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    let input: Vec<f32> = (0..frames)
        .flat_map(|i| {
            let t = i as f64 / SR as f64;
            let center = 0.25 * (2.0 * std::f64::consts::PI * 500.0 * t).sin();
            let l = center + 0.2 * noise(&mut seed_l);
            let r = center + 0.2 * noise(&mut seed_r);
            [l as f32, r as f32]
        })
        .collect();
    let out = render_stereo(&input, 1.5);
    assert!(out.len() > SR as usize);
    let side: Vec<f32> = out
        .chunks_exact(2)
        .map(|fr| 0.5 * (fr[0] - fr[1]))
        .collect();
    let mid: Vec<f32> = out
        .chunks_exact(2)
        .map(|fr| 0.5 * (fr[0] + fr[1]))
        .collect();
    let leak = goertzel(&side, 500.0);
    let center_level = goertzel(&mid, 500.0);
    let rejection_db = 20.0 * (center_level / leak.max(1e-12)).log10();
    println!("center-to-side rejection: {rejection_db:.1} dB");
    assert!(
        rejection_db > 60.0,
        "center leaks into the side channel: {rejection_db:.1} dB rejection \
         (measured 70.9 dB with M/S, 64.6 dB per-channel)"
    );
}

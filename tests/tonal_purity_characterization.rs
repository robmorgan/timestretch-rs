//! Tonal-purity characterization at DJ ratios (ROADMAP Stage 16).
//!
//! These pin the MEASURED granulation floor of the shipped keylock path
//! on music-like tonal probes — characterization, not quality gates: the
//! floors sit just below current behavior so a regression is caught, but
//! the numbers themselves are the evidence base for the Stage 16
//! listening verdict (a sine-purity figure overstates audibility on
//! dense mixes, which is exactly why the verdict is by ear).

use timestretch::{StretchParams, stretch};

const SR: u32 = 44_100;

fn goertzel(seg: &[f32], freq: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * freq / SR as f64;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &x in seg {
        let s0 = x as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    ((s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0)).sqrt() / (seg.len() as f64 / 2.0)
}

fn render(input: &[f32], ratio: f64) -> Vec<f32> {
    let params = StretchParams::new(ratio)
        .with_sample_rate(SR)
        .with_channels(1);
    stretch(input, &params).expect("stretch")
}

/// Harmonic stack (a sustained string/organ-like probe): fundamental at
/// 220 Hz with 1/k harmonics through 4.4 kHz. Measures how much energy a
/// high harmonic keeps versus the sidebands SOLA splicing spreads around
/// it. Measured 2026-08-07 (pre-Stage-18): harmonic-15 purity 52.0 dB at
/// +8% tempo but 22.1 dB at −8% — the asymmetric granulation floor the
/// Stage 16 blind verdict confirmed as audible ("roboty" on sustained
/// slowdowns). Re-measured 2026-08-13 with the Stage 18 cadence stretch:
/// 59.5 dB (+8%) / 62.8 dB (−8%) — the asymmetry is gone and the floor
/// sits ~40 dB higher on the weak direction; the re-pinned floor locks
/// that in.
#[test]
fn harmonic_stack_purity_at_dj_ratios() {
    let n = SR as usize * 6;
    let input: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / SR as f64;
            (1..=20)
                .map(|k| (2.0 * std::f64::consts::PI * 220.0 * k as f64 * t).sin() / k as f64)
                .sum::<f64>() as f32
                * 0.2
        })
        .collect();
    for ratio in [1.0 / 1.08, 1.0 / 0.92] {
        let out = render(&input, ratio);
        let mid = &out[out.len() / 2 - (1 << 16)..out.len() / 2 + (1 << 16)];
        // Harmonic 15 (3300 Hz) vs the energy at half-harmonic offsets
        // around it (splice sidebands land off the harmonic grid).
        let h15 = goertzel(mid, 3_300.0);
        let side = goertzel(mid, 3_190.0).max(goertzel(mid, 3_410.0));
        let purity_db = 20.0 * (h15 / side.max(1e-12)).log10();
        println!("harmonic stack ratio {ratio:.3}: h15 vs sidebands {purity_db:.1} dB");
        assert!(
            purity_db > 45.0,
            "harmonic-15 purity regressed: {purity_db:.1} dB \
             (characterized 59.5/+8%, 62.8/-8% with the Stage 18 cadence stretch)"
        );
    }
}

/// Two nearby tones (2000 and 2030 Hz): the 30 Hz beat pattern must
/// survive time-stretching — splice-rate modulation that disrupts it
/// reads as roughness. Characterized 2026-08-07: carriers-vs-spur floor
/// 47.3 dB (+8%) / 41.7 dB (−8%). Re-measured 2026-08-13 with the
/// Stage 18 cadence stretch: 62.1 dB (+8%) / 41.3 dB (−8%) — the −8%
/// direction is within noise of the old figure, +8% improved.
#[test]
fn nearby_tone_pair_keeps_beating_structure() {
    let n = SR as usize * 6;
    let input: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / SR as f64;
            (0.3 * (2.0 * std::f64::consts::PI * 2_000.0 * t).sin()
                + 0.3 * (2.0 * std::f64::consts::PI * 2_030.0 * t).sin()) as f32
        })
        .collect();
    for ratio in [1.0 / 1.08, 1.0 / 0.92] {
        let out = render(&input, ratio);
        let mid = &out[out.len() / 2 - (1 << 16)..out.len() / 2 + (1 << 16)];
        let carriers = goertzel(mid, 2_000.0) + goertzel(mid, 2_030.0);
        // Spur floor away from the pair and its beat neighborhood.
        let spur = goertzel(mid, 1_850.0).max(goertzel(mid, 2_180.0));
        let clarity_db = 20.0 * (carriers / spur.max(1e-12)).log10();
        println!("tone pair ratio {ratio:.3}: carriers vs spurs {clarity_db:.1} dB");
        assert!(
            clarity_db > 32.0,
            "tone-pair clarity regressed: {clarity_db:.1} dB \
             (characterized 62.1/+8%, 41.3/-8% with the Stage 18 cadence stretch)"
        );
    }
}

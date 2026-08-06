//! Stereo image coherence through the wide keylock profile (ROADMAP
//! Stage 14). The corrected path runs in mid/side: centered content lives
//! entirely in M under one phase vocoder, so the image cannot wander
//! between independently-evolving per-channel phase states. Before the
//! change, identical L/R inputs measurably diverged through the two
//! independent vocoders.

use timestretch::engine::{Engine, EngineConfig, EngineProfile};

const SR: u32 = 44_100;

/// Renders `input` (interleaved stereo) through the wide profile at a
/// constant rate, returning interleaved output past latency + settle.
fn render_wide_stereo(input: &[f32], rate: f64) -> Vec<f32> {
    let handles = Engine::build(EngineConfig {
        sample_rate: SR,
        channels: 2,
        profile: EngineProfile::WideKeylock,
        initial_tempo_rate: rate,
        ..EngineConfig::default()
    })
    .unwrap();
    let (_controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);
    let latency = processor.pipeline_latency_frames();
    let expected = (input.len() as f64 / 2.0 / rate) as usize;
    let mut fed = 0usize;
    let mut out = vec![0.0f32; 512 * 2];
    let mut collected = Vec::with_capacity(expected * 2);
    while collected.len() < expected * 2 {
        while fed < input.len() && source.free_frames() > 1024 {
            fed += source.push(&input[fed..input.len().min(fed + 8_192)]) * 2;
        }
        if fed >= input.len() {
            break; // enough source consumed for the assertion span
        }
        processor.process(&mut out);
        collected.extend_from_slice(&out);
    }
    // Skip the latency prime and the stage's warm-up settle.
    let skip = (latency + 4_096) * 2;
    collected.split_off(skip.min(collected.len()))
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

/// Identical channels must stay bit-identical: S encodes to exactly zero,
/// zero stays zero through the vocoder and resampler, and decode adds
/// M ± 0.
#[test]
fn identical_channels_stay_identical_through_wide_keylock() {
    for rate in [0.6f64, 1.5] {
        let input = stereo_fixture(1.0, SR as usize * 6);
        let out = render_wide_stereo(&input, rate);
        assert!(out.len() > SR as usize, "not enough output at rate {rate}");
        for (i, fr) in out.chunks_exact(2).enumerate() {
            assert!(
                fr[0] == fr[1],
                "center image diverged at frame {i} (rate {rate}): L={} R={}",
                fr[0],
                fr[1]
            );
        }
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
    let out = render_wide_stereo(&input, 1.5);
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

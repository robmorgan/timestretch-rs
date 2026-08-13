//! Stage 19 kill experiment (temporary, not for commit): direct-ratio
//! chunked PV under DYNAMIC rate. Constant-rate quality is proven
//! (Stage 14 attribution); this measures the risk — rate transitions.
//!
//! Scenarios, each on a pure tone (objective click/pitch metrics) and
//! per-scenario worst adjacent-diff reported against the tone's slew
//! bound (the soak/torture click criterion):
//!   1. instant ratio steps 1.4286 <-> 2.0 every 2 s (worst case)
//!   2. the live stage's log-slewed steps (0.05 ln per 32-frame block)
//!   3. continuous ride: ratio oscillating +/-10% around 1.6 at 0.3 Hz
//!
//! Note the design's key property: the PV does TEMPO, so pitch is
//! constant through every transition — measured to confirm.

use timestretch::core::window::WindowType;
use timestretch::stretch::{PhaseLockingMode, PhaseVocoder};

const SR: u32 = 44_100;
const FFT: usize = 2048;
const HOP: usize = 256;
const TONE_HZ: f64 = 440.0;
const AMP: f32 = 0.4;
const SLEW_LN_PER_BLOCK: f64 = 0.05;

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

fn peak_freq(seg: &[f32], hint: f64) -> f64 {
    let mut best = (hint, 0.0f64);
    let mut f = hint - 12.0;
    while f <= hint + 12.0 {
        let m = goertzel(seg, f);
        if m > best.1 {
            best = (f, m);
        }
        f += 0.25;
    }
    best.0
}

/// Drives the chunked direct-ratio PV with a per-block ratio schedule.
fn run(schedule: impl Fn(usize) -> f64, secs: usize) -> Vec<f32> {
    let n = SR as usize * secs;
    let input: Vec<f32> = (0..n)
        .map(|i| AMP * (2.0 * std::f64::consts::PI * TONE_HZ * i as f64 / SR as f64).sin() as f32)
        .collect();
    let mut pv = PhaseVocoder::with_options(
        FFT,
        HOP,
        schedule(0),
        SR,
        100.0,
        WindowType::Hann,
        PhaseLockingMode::Identity,
    );
    let mut window: Vec<f32> = Vec::with_capacity(FFT + 64);
    let mut out: Vec<f32> = Vec::new();
    let mut chunk: Vec<f32> = Vec::with_capacity(4 * FFT);
    for (block_idx, block) in input.chunks(32).enumerate() {
        pv.set_stretch_ratio(schedule(block_idx));
        window.extend_from_slice(block);
        while window.len() >= FFT {
            pv.process_streaming_into(&window[..FFT], &mut chunk).expect("pv");
            out.extend_from_slice(&chunk);
            window.copy_within(HOP.., 0);
            window.truncate(window.len() - HOP);
        }
    }
    out
}

fn report(name: &str, out: &[f32], max_ratio: f64) {
    // Click bound: the tone's max adjacent-sample step at the highest
    // OUTPUT-side frequency content. Pitch is preserved (440 Hz), so the
    // bound is the source tone's slew x3 (the soak criterion).
    let _ = max_ratio;
    let bound = AMP * (2.0 * std::f64::consts::PI * TONE_HZ / SR as f64) as f32 * 3.0;
    let mut worst = (0usize, 0.0f32);
    for (i, w) in out[8_192..out.len() - 4_096].windows(2).enumerate() {
        let d = (w[1] - w[0]).abs();
        if d > worst.1 {
            worst = (i + 8_192, d);
        }
    }
    // Pitch in three windows: early, mid (straddling transitions), late.
    let n = out.len();
    let w = 1 << 16;
    let pitches: Vec<f64> = [n / 6, n / 2, 5 * n / 6]
        .iter()
        .map(|&c| peak_freq(&out[c - w / 2..c + w / 2], TONE_HZ))
        .collect();
    let purity_mid = {
        let seg = &out[n / 2 - w / 2..n / 2 + w / 2];
        let c = goertzel(seg, TONE_HZ);
        let side = goertzel(seg, TONE_HZ - 55.0).max(goertzel(seg, TONE_HZ + 55.0));
        20.0 * (c / side.max(1e-12)).log10()
    };
    println!(
        "{name}: max_diff {:.5} (bound {bound:.5}, {}) at {} | pitch {:.1}/{:.1}/{:.1} Hz | mid purity {:.1} dB",
        worst.1,
        if worst.1 <= bound { "OK" } else { "CLICK" },
        worst.0,
        pitches[0],
        pitches[1],
        pitches[2],
        purity_mid
    );
}

fn main() {
    let blocks_per_sec = SR as usize / 32;

    // 1. Instant steps between 1.4286 and 2.0 every 2 s.
    let out = run(
        |b| if (b / (2 * blocks_per_sec)) % 2 == 0 { 1.0 / 0.70 } else { 2.0 },
        12,
    );
    report("instant steps ", &out, 2.0);

    // 2. Live-style log-slewed steps between the same targets.
    let out = run(
        |b| {
            let target: f64 = if (b / (2 * blocks_per_sec)) % 2 == 0 { 1.0 / 0.70 } else { 2.0 };
            // emulate per-block slew from a running value: recompute by
            // walking the schedule (deterministic, cheap for the probe)
            let mut cur: f64 = 1.0 / 0.70;
            for k in 1..=b {
                let t: f64 = if (k / (2 * blocks_per_sec)) % 2 == 0 { 1.0 / 0.70 } else { 2.0 };
                let step = (t.ln() - cur.ln()).clamp(-SLEW_LN_PER_BLOCK, SLEW_LN_PER_BLOCK);
                cur = (cur.ln() + step).exp();
            }
            let _ = target;
            cur
        },
        12,
    );
    report("slewed steps  ", &out, 2.0);

    // 3. Continuous ride: +/-10% around 1.6 at 0.3 Hz.
    let out = run(
        |b| {
            let t = b as f64 * 32.0 / SR as f64;
            1.6 * (1.0 + 0.10 * (2.0 * std::f64::consts::PI * 0.3 * t).sin())
        },
        12,
    );
    report("continuous ride", &out, 1.8);
}

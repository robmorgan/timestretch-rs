//! Stage 18 build-out sweep (temporary): find the green boundary for each
//! trigger scale across the keylock range. Per (rate, scale): sine pitch
//! accuracy (the ratio-1.25 failure mode — cursor stall flattens pitch)
//! and harmonic-15 purity. Scale comes from the prototype env knob, so
//! this binary is invoked once per scale by the driver.

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

/// Dominant frequency near `hint` by parabolic-refined Goertzel scan.
fn peak_freq(seg: &[f32], hint: f64) -> f64 {
    let mut best = (hint, 0.0f64);
    let mut f = hint - 8.0;
    while f <= hint + 8.0 {
        let m = goertzel(seg, f);
        if m > best.1 {
            best = (f, m);
        }
        f += 0.25;
    }
    best.0
}

fn main() {
    let scale =
        std::env::var("TIMESTRETCH_PROTO_SOLA_TRIGGER_SCALE").unwrap_or_else(|_| "1".into());
    let n = SR as usize * 6;
    // Sine for pitch accuracy; harmonic stack for purity.
    let sine: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * 220.0 * i as f64 / SR as f64).sin() as f32 * 0.4)
        .collect();
    let stack: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / SR as f64;
            (1..=20)
                .map(|k| (2.0 * std::f64::consts::PI * 220.0 * k as f64 * t).sin() / k as f64)
                .sum::<f64>() as f32
                * 0.2
        })
        .collect();
    // Slowdown side only (T > 1 is the binding constraint): rates 0.95 .. 0.80.
    for rate in [0.95f64, 0.92, 0.90, 0.87, 0.85, 0.83, 0.80] {
        let ratio = 1.0 / rate;
        let params = StretchParams::new(ratio)
            .with_sample_rate(SR)
            .with_channels(1);
        let out = stretch(&sine, &params).expect("stretch");
        let mid = &out[out.len() / 2 - (1 << 16)..out.len() / 2 + (1 << 16)];
        let pitch = peak_freq(mid, 220.0);
        let cents = 1200.0 * (pitch / 220.0).log2();
        let outh = stretch(&stack, &params).expect("stretch");
        let midh = &outh[outh.len() / 2 - (1 << 16)..outh.len() / 2 + (1 << 16)];
        let h15 = goertzel(midh, 3_300.0);
        let side = goertzel(midh, 3_190.0).max(goertzel(midh, 3_410.0));
        let purity = 20.0 * (h15 / side.max(1e-12)).log10();
        println!("scale={scale} rate={rate:.2} pitch_cents={cents:+.1} purity={purity:.1} dB");
    }
}

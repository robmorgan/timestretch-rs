//! Direct phase-vocoder quality gates.
//!
//! Everything in `tests/identity.rs` goes through `stretch()`, whose exact
//! ratio-1.0 bypass (`lib.rs`) short-circuits before any DSP runs — so none
//! of it exercises the vocoder. These tests drive `PhaseVocoder` itself:
//! a null test at ratio 1.0 and a long-render phase-precision gate. Both
//! were added with the Stage 13 phase-hygiene fixes (wrapped accumulator,
//! wrapped-difference coherence blends, real DC/Nyquist) and fail without
//! them.

use timestretch::core::window::WindowType;
use timestretch::stretch::{PhaseLockingMode, PhaseVocoder};

const SAMPLE_RATE: u32 = 44_100;
const FFT_SIZE: usize = 2048;
const HOP: usize = 256;
/// Sub-bass cutoff matching the production wide-keylock configuration.
const SUB_BASS_HZ: f32 = 100.0;

fn make_pv(ratio: f64) -> PhaseVocoder {
    PhaseVocoder::with_options(
        FFT_SIZE,
        HOP,
        ratio,
        SAMPLE_RATE,
        SUB_BASS_HZ,
        WindowType::Hann,
        PhaseLockingMode::Identity,
    )
}

fn sine(num_samples: usize, freq: f64) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / SAMPLE_RATE as f64).sin() as f32)
        .collect()
}

/// Deterministic broadband test mix: two tones plus low-level noise.
fn tone_mix(num_samples: usize) -> Vec<f32> {
    let low = sine(num_samples, 220.0);
    let high = sine(num_samples, 3_300.0);
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    (0..num_samples)
        .map(|i| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32;
            0.5 * low[i] + 0.3 * high[i] + 0.05 * noise
        })
        .collect()
}

/// Signal-to-error ratio (dB) of `out` against `reference` at the best
/// integer lag in `[-max_lag, max_lag]`, plus the residual peak and the
/// winning lag. Both slices are compared over `reference`'s full length at
/// each lag (out-of-range samples excluded).
fn best_aligned_ser(reference: &[f32], out: &[f32], max_lag: i64) -> (f64, f64, i64) {
    let mut best = (f64::NEG_INFINITY, f64::INFINITY, 0i64);
    for lag in -max_lag..=max_lag {
        let mut signal = 0.0f64;
        let mut error = 0.0f64;
        let mut peak = 0.0f64;
        let mut count = 0usize;
        for (i, &r) in reference.iter().enumerate() {
            let j = i as i64 + lag;
            if j < 0 || j as usize >= out.len() {
                continue;
            }
            let e = (r as f64 - out[j as usize] as f64).abs();
            signal += (r as f64) * (r as f64);
            error += e * e;
            peak = peak.max(e);
            count += 1;
        }
        if count < reference.len() / 2 {
            continue;
        }
        let ser = 10.0 * (signal / error.max(1e-30)).log10();
        if ser > best.0 {
            best = (ser, peak, lag);
        }
    }
    (best.0, best.1, best.2)
}

/// Tone power at `freq` over `segment` via Goertzel, as a fraction of total
/// power, in dB: `10*log10(P_tone / (P_total - P_tone))`.
fn tone_purity_db(segment: &[f32], freq: f64) -> f64 {
    let n = segment.len();
    // Hann window against spectral leakage from non-integer cycle counts.
    let windowed: Vec<f64> = segment
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
            s as f64 * w
        })
        .collect();
    let total: f64 = windowed.iter().map(|&s| s * s).sum();
    // Sum Goertzel power over +-2 bins around the tone to absorb the
    // window's main lobe.
    let bin_hz = SAMPLE_RATE as f64 / n as f64;
    let mut tone = 0.0f64;
    for k in -2i64..=2 {
        let f = freq + k as f64 * bin_hz;
        let w = 2.0 * std::f64::consts::PI * f / SAMPLE_RATE as f64;
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &x in &windowed {
            let s0 = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        tone += (s1 * s1 + s2 * s2 - coeff * s1 * s2) * 2.0 / n as f64;
    }
    let rest = (total - tone).max(1e-30);
    10.0 * (tone / rest).log10()
}

/// At ratio 1.0 the vocoder must be a near-transparent identity: analysis
/// phases are tracked exactly (mod 2*PI) by the accumulator, identity
/// locking rotates by ~0, and the coherence blends interpolate along a
/// wrapped difference of ~0. Measured 139 dB with the Stage 13 fixes;
/// 59 dB before them (unwrapped-blend corruption, partially masked at
/// short durations by the locking overwrite).
#[test]
fn pv_null_at_ratio_one() {
    let input = tone_mix(5 * SAMPLE_RATE as usize);
    let mut pv = make_pv(1.0);
    let output = pv.process(&input).expect("pv process");

    assert_eq!(
        output.len(),
        input.len(),
        "ratio-1.0 output length must equal input length"
    );

    // Compare the steady-state middle, clear of the batch path's edge
    // padding and gain ramps.
    let margin = 4 * FFT_SIZE;
    let reference = &input[margin..input.len() - margin];
    let window = &output[margin - FFT_SIZE..input.len() - margin + FFT_SIZE];
    let (ser, peak_residual, lag) = best_aligned_ser(reference, window, 2 * FFT_SIZE as i64);

    assert!(
        ser > 100.0,
        "ratio-1.0 null: SER {ser:.1} dB (peak residual {peak_residual:.2e}, lag {lag}) — \
         expected > 100 dB (measured 139 dB fixed, 59 dB with unwrapped blends)"
    );
}

/// Tone purity at a non-unity ratio, early and late in a long render.
/// Gates two failure modes at once: unwrapped-phase blend corruption
/// (drops purity to ~25 dB at any duration) and accumulator-precision
/// decay from the per-frame f32 downcast of an unwrapped accumulator
/// (purity erodes as the render grows). Measured 57 dB flat with the
/// Stage 13 fixes.
#[test]
fn pv_long_render_purity_stable() {
    let seconds = 30;
    let freq = 5_000.0;
    let input = sine(seconds * SAMPLE_RATE as usize, freq);
    let mut pv = make_pv(1.5);
    let output = pv.process(&input).expect("pv process");

    let sr = SAMPLE_RATE as usize;
    let early = tone_purity_db(&output[5 * sr..6 * sr], freq);
    let late_start = output.len() - 10 * sr;
    let late = tone_purity_db(&output[late_start..late_start + sr], freq);

    assert!(
        early > 45.0 && late > 45.0,
        "tone purity {early:.1} dB early / {late:.1} dB late — expected > 45 dB (measured \
         ~57 dB fixed, ~25 dB with unwrapped blends)"
    );
    assert!(
        late > early - 3.0,
        "tone purity decayed over the render: {early:.1} dB at 5 s vs {late:.1} dB near the \
         end — phase accumulator precision is degrading"
    );
}

/// DC must survive as DC: a constant-offset signal keeps its offset sign
/// and magnitude instead of being scaled by the cosine of an accumulated
/// synthetic phase.
#[test]
fn pv_preserves_dc_component() {
    let n = 3 * SAMPLE_RATE as usize;
    let input: Vec<f32> = sine(n, 440.0).iter().map(|&s| 0.3 * s + 0.25).collect();
    let mut pv = make_pv(1.25);
    let output = pv.process(&input).expect("pv process");

    let margin = 4 * FFT_SIZE;
    let mid = &output[margin..output.len() - margin];
    let mean = mid.iter().map(|&s| s as f64).sum::<f64>() / mid.len() as f64;
    assert!(
        (mean - 0.25).abs() < 0.02,
        "DC offset not preserved: input +0.25, output mean {mean:+.3}"
    );
}

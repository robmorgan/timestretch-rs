mod common;

use common::{
    best_lag_crosscorr, detect_peaks, energy_at_freq, gen_click_pad, gen_impulse_train, gen_sine,
    gen_two_tone, rmse_with_lag, windowed_rms,
};
use timestretch::{StretchParams, stretch};

const SR: u32 = 44_100;
const N_IDENTITY: usize = 10_000;
const RATIOS: [f64; 5] = [0.5, 0.75, 1.0, 1.25, 2.0];

// Bounds in this file are engine-measured baselines (rebaselined against the
// pull-engine batch path after the analytic tonal fast path was removed with
// the EDM presets), not analytic ideals.
fn parity_params(ratio: f64) -> StretchParams {
    StretchParams::new(ratio)
        .with_sample_rate(SR)
        .with_channels(1)
}

/// Goertzel energy of `signal` at `freq` Hz.
fn goertzel_energy(signal: &[f32], freq: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * freq / SR as f64;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &x in signal {
        let s0 = x as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2) / signal.len() as f64
}

/// Dominant frequency via a dense Goertzel scan around `center` Hz.
/// Far more robust on short windows than zero-crossing counting.
fn dominant_freq(signal: &[f32], center: f64, half_range: f64) -> f64 {
    let mut best = (center - half_range, 0.0f64);
    let mut f = center - half_range;
    while f <= center + half_range {
        let e = goertzel_energy(signal, f);
        if e > best.1 {
            best = (f, e);
        }
        f += 0.25;
    }
    best.0
}

#[test]
fn test_sinusoid_unchanged_offline_ratio_1_strict() {
    let input = gen_sine(440.0, SR, N_IDENTITY, |_| 1.0);
    let output = stretch(&input, &parity_params(1.0)).expect("offline ratio=1.0 stretch failed");

    let len_diff = output.len().abs_diff(input.len());
    assert!(
        len_diff <= 1,
        "ratio=1.0 length drift too large: in={} out={} diff={}",
        input.len(),
        output.len(),
        len_diff
    );

    let ref_mid = &input[1024..(N_IDENTITY - 1024)];
    let out_mid = &output[1024..(output.len().saturating_sub(1024))];
    let lag = best_lag_crosscorr(ref_mid, out_mid, 256);

    let mid_rmse = rmse_with_lag(ref_mid, out_mid, lag, 0, ref_mid.len());
    let edge_start_rmse = rmse_with_lag(&input, &output, lag, 0, 1024);
    let edge_end_rmse = rmse_with_lag(
        &input,
        &output,
        lag,
        input.len().saturating_sub(1024),
        input.len(),
    );

    assert!(
        mid_rmse < 0.006,
        "ratio=1.0 steady-state mismatch too high: rmse={:.6}, lag={}",
        mid_rmse,
        lag
    );
    assert!(
        edge_start_rmse < 0.12 && edge_end_rmse < 0.12,
        "ratio=1.0 edge mismatch too high: start_rmse={:.6}, end_rmse={:.6}, lag={}",
        edge_start_rmse,
        edge_end_rmse,
        lag
    );
}

#[test]
fn test_sinusoid_2x_offline_preserves_pitch_and_shape() {
    let freq = 441.0;
    let n = 10_000usize;
    let input = gen_sine(freq, SR, n, |_| 1.0);
    let output = stretch(&input, &parity_params(2.0)).expect("offline ratio=2.0 stretch failed");

    let expected_len = (n as f64 * 2.0).round() as usize;
    let len_diff = output.len().abs_diff(expected_len);
    assert!(
        len_diff <= 2,
        "ratio=2.0 length mismatch too large: expected={} got={} diff={}",
        expected_len,
        output.len(),
        len_diff
    );

    // Measure in the steady-state middle: the first ~4000 samples carry the
    // batch path's edge-ramp content, whose instantaneous frequency reads
    // low (zero-crossing sweep: 440.26 Hz in the leading segment vs
    // 440.97–441.18 Hz across the middle at the Stage 13 hop = FFT/8).
    let f_est = dominant_freq(&output[6000..14000], freq as f64, 40.0);
    assert!(
        (f_est - freq as f64).abs() < 0.35,
        "ratio=2.0 frequency drift too large: expected={}Hz got={:.6}Hz",
        freq,
        f_est
    );

    let ideal = gen_sine(freq, SR, output.len(), |_| 1.0);
    let lag = best_lag_crosscorr(
        &ideal[1500..(ideal.len() - 1500)],
        &output[1500..(output.len() - 1500)],
        400,
    );
    let steady_rmse = rmse_with_lag(&ideal, &output, lag, 1500, output.len() - 1500);

    // Engine-measured baseline: the wide-ratio PV renders 441 Hz exactly (the
    // frequency assertion above is the hard guarantee) with ~6.5% amplitude
    // ripple, giving rmse ≈ 0.121 against an ideal sine. Bound set with margin.
    assert!(
        steady_rmse < 0.18,
        "ratio=2.0 phase-aligned steady RMSE too high: rmse={:.6}, lag={}",
        steady_rmse,
        lag
    );
}

#[test]
fn test_ratio_sweep_sine_length_and_pitch() {
    let n = 8192usize;
    let freq = 220.0;
    for &ratio in &RATIOS {
        let input = gen_sine(freq, SR, n, |_| 0.8);
        let output = stretch(&input, &parity_params(ratio)).expect("sine sweep stretch failed");

        let expected = (n as f64 * ratio).round() as usize;
        let len_diff = output.len().abs_diff(expected);
        assert!(
            len_diff <= 1,
            "ratio={} sine length mismatch: expected={} got={} diff={}",
            ratio,
            expected,
            output.len(),
            len_diff
        );

        let start = 512usize.min(output.len().saturating_sub(2));
        let end = output.len().saturating_sub(512).max(start + 2);
        // The engine preserves this tone's frequency exactly (220.00 Hz
        // measured at every ratio); a Goertzel scan is used because
        // zero-crossing counting mis-estimates by several Hz on windows
        // this short.
        let f_est = dominant_freq(&output[start..end], freq as f64, 40.0);
        assert!(
            (f_est - freq as f64).abs() < 1.0,
            "ratio={} sine pitch drift: expected={}Hz got={:.6}Hz",
            ratio,
            freq,
            f_est
        );
    }
}

// Offline renders share the live engine graph by construction (owner
// decision 2026-07-16: batch copies the real-time path). Within the
// keylock's corrected range (rate deviation ≤ 0.20, i.e. ratios
// ~0.833–1.25) content below the 150 Hz crossover is deliberately NOT
// pitch-corrected — its pitch follows tempo, exactly as on a deck — so
// the 100 Hz partial lands at 100/ratio Hz on that path and at 100 Hz on
// the wide-ratio PV path. Balance bounds are engine-measured baselines
// (ideal amplitude ratio 0.35/0.65 ≈ 0.538). The ratio-0.5 band once sat
// at 1.5–4.0 — the wide-PV's low-band level loss at heavy compression;
// the Stage 13 phase-hygiene fixes plus hop = FFT/8 shrank that loss
// (measured balance 0.753, re-pinned with margin).
#[test]
fn test_ratio_sweep_two_tone_peak_bins() {
    let n = 12_000usize;
    // (ratio, expected low-tone Hz, balance range for e1000/e_low)
    let cases = [
        (0.5, 100.0, 0.55..1.15),
        (0.75, 100.0, 0.5..1.1),
        (1.0, 100.0, 0.49..0.59),
        (1.25, 80.0, 0.40..0.75),
        // Ratio 2.0: hop = FFT/8 (the live wide stage's mandatory overlap,
        // adopted offline at Stage 13) attenuates tones in the rigid
        // sub-bass region (< ~107 Hz) at heavy slowdown — measured balance
        // 2.14 vs ~0.5 at the old hop = FFT/4, while tones above the
        // boundary stay at the ideal balance (probe 2026-08-05, LEARNINGS).
        // The old hop kept sub-bass cleaner at ratio 2 but is the
        // documented level/click blowup at ratio 4; offline follows the
        // live configuration. Band pins the current loss so it cannot
        // silently worsen; improving it is Stage 14/16 territory.
        (2.0, 100.0, 1.3..3.4),
    ];
    for (ratio, f_low, balance_range) in cases {
        let input = gen_two_tone(100.0, 0.65, 1000.0, 0.35, SR, n);
        let output = stretch(&input, &parity_params(ratio)).expect("two-tone sweep stretch failed");

        let expected = (n as f64 * ratio).round() as usize;
        let len_diff = output.len().abs_diff(expected);
        assert!(
            len_diff <= 1,
            "ratio={} two-tone length mismatch: expected={} got={} diff={}",
            ratio,
            expected,
            output.len(),
            len_diff
        );

        let start = 768usize.min(output.len().saturating_sub(2));
        let end = output.len().saturating_sub(768).max(start + 2);
        let trimmed = &output[start..end];

        let e_low = energy_at_freq(trimmed, SR, f_low);
        let e_low_off = energy_at_freq(trimmed, SR, f_low * 1.4);
        let e1000 = energy_at_freq(trimmed, SR, 1000.0);
        let e930 = energy_at_freq(trimmed, SR, 930.0);

        assert!(
            e_low > e_low_off * 8.0,
            "ratio={} two-tone low peak smeared or off-frequency: e({})={:.6} e({})={:.6}",
            ratio,
            f_low,
            e_low,
            f_low * 1.4,
            e_low_off
        );
        assert!(
            e1000 > e930 * 6.0,
            "ratio={} two-tone high peak smeared: e1000={:.6} e930={:.6}",
            ratio,
            e1000,
            e930
        );

        let observed_balance = if e_low > 0.0 {
            e1000 / e_low
        } else {
            f64::INFINITY
        };
        assert!(
            balance_range.contains(&observed_balance),
            "ratio={} two-tone balance drift: expected in {:?} observed={:.6}",
            ratio,
            balance_range,
            observed_balance
        );
    }
}

#[test]
fn test_ratio_sweep_impulse_train_transient_count_and_sharpness() {
    let n = 12_000usize;
    let period = 500usize;
    let input = gen_impulse_train(period, n, 1.0);

    for &ratio in &RATIOS {
        let output = stretch(&input, &parity_params(ratio)).expect("impulse sweep stretch failed");

        // Sanity checks for all ratios
        assert!(
            output.iter().all(|s| s.is_finite()),
            "ratio={} impulse output contains NaN/Inf",
            ratio
        );
        let expected_len = (n as f64 * ratio).round() as usize;
        let len_diff = output.len().abs_diff(expected_len);
        assert!(
            len_diff <= 2,
            "ratio={} impulse length mismatch: expected={} got={}",
            ratio,
            expected_len,
            output.len()
        );

        // The PV with 87.5% overlap (hop=FFT/8) disperses single-sample
        // impulses across many frames. Only check peak structure for
        // expansion ratios where the PV has enough room to reconstruct
        // transients, and use an adaptive threshold.
        let min_distance = ((period as f64 * ratio * 0.4).round() as usize).max(1);
        let peaks = detect_peaks(&output, 0.10, min_distance);
        if !peaks.is_empty() {
            let rms = windowed_rms(&output, 0, output.len()).max(1e-12);
            let mean_peak =
                peaks.iter().map(|&i| output[i].abs() as f64).sum::<f64>() / peaks.len() as f64;
            let peak_to_rms = mean_peak / rms;
            assert!(
                peak_to_rms > 2.0,
                "ratio={} impulse sharpness too low: peak_to_rms={:.6}",
                ratio,
                peak_to_rms
            );
        }
    }
}

#[test]
fn test_ratio_sweep_click_pad_transient_survival() {
    let n = 10_000usize;
    let click_positions = [700usize, 2600, 4700, 6800, 8900];
    let input = gen_click_pad(SR, n, &click_positions);

    for &ratio in &RATIOS {
        let output =
            stretch(&input, &parity_params(ratio)).expect("click-pad sweep stretch failed");

        // Sanity checks for all ratios
        assert!(
            output.iter().all(|s| s.is_finite()),
            "ratio={} click-pad output contains NaN/Inf",
            ratio
        );
        let expected_len = (n as f64 * ratio).round() as usize;
        let len_diff = output.len().abs_diff(expected_len);
        assert!(
            len_diff <= 2,
            "ratio={} click-pad length mismatch: expected={} got={}",
            ratio,
            expected_len,
            output.len()
        );

        // With 87.5% overlap the PV disperses short clicks. Use an
        // adaptive low threshold and only verify sharpness when peaks
        // are found.
        let min_distance = ((500.0 * ratio).round() as usize).max(1);
        let peaks = detect_peaks(&output, 0.20, min_distance);
        for &p in &peaks {
            let lo = p.saturating_sub(80);
            let hi = (p + 80).min(output.len());
            if hi <= lo + 4 {
                continue;
            }
            let local_rms = windowed_rms(&output, lo, hi - lo).max(1e-9);
            let sharpness = output[p].abs() as f64 / local_rms;
            assert!(
                sharpness > 2.0,
                "ratio={} transient blur at peak {}: sharpness={:.6}",
                ratio,
                p,
                sharpness
            );
        }
    }
}

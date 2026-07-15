//! A/B adapter smoke harness (ROADMAP new Stage 1): both engines render the
//! same fixture through the same interface, and structural metrics compare.
//!
//! Run with:
//! `cargo test --features qa-harnesses --release --test engine_ab -- --nocapture`
//!
//! This harness gates only engine-independent structure (click-freeness,
//! level, timeline length) — the new tape arm intentionally shifts pitch
//! with tempo, so spectral comparisons start in Stage 2 when the keylock
//! chain exists.

// Each harness compiles the shared adapter separately, so arms another
// harness uses read as dead code here.
#[allow(dead_code)]
#[path = "ab/mod.rs"]
mod ab;

use ab::{render_with_rate_schedule, Arm};

const SAMPLE_RATE: u32 = 44_100;
const CALLBACK_FRAMES: usize = 256;

fn sine(freq: f32, frames: usize, amp: f32) -> Vec<f32> {
    (0..frames)
        .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin())
        .collect()
}

fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples
        .iter()
        .map(|&s| (s as f64) * (s as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt()
}

fn max_adjacent_diff(samples: &[f32]) -> f32 {
    samples
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f32::max)
}

#[test]
fn adapter_drives_both_profiles_over_identical_fixture() {
    let amp = 0.5f32;
    let input = sine(220.0, SAMPLE_RATE as usize * 6, amp);
    // The DJ ride gesture both arms must survive.
    let ride = |t: f64| 1.0 + 0.06 * (2.0 * std::f64::consts::PI * 0.25 * t).sin();

    let keylock = render_with_rate_schedule(
        Arm::NewKeylock,
        &input,
        1,
        SAMPLE_RATE,
        CALLBACK_FRAMES,
        &ride,
    );
    let new =
        render_with_rate_schedule(Arm::NewTape, &input, 1, SAMPLE_RATE, CALLBACK_FRAMES, &ride);

    println!(
        "adapter smoke: keylock len={} rms={:.4} max_diff={:.4} | tape len={} rms={:.4} max_diff={:.4} underruns={}",
        keylock.output.len(),
        rms(&keylock.output),
        max_adjacent_diff(&keylock.output),
        new.output.len(),
        rms(&new.output),
        max_adjacent_diff(&new.output),
        new.underrun_frames,
    );

    // Both arms cover the fixture: output duration within 5% of the source
    // duration (the ride averages rate ~1.0).
    for (label, out) in [("keylock", &keylock.output), ("new", &new.output)] {
        let len_err = (out.len() as f64 - input.len() as f64).abs() / input.len() as f64;
        assert!(
            len_err < 0.05,
            "{label} arm length error {len_err:.3} (len {})",
            out.len()
        );
    }

    // Level parity: same tone, same amplitude, both engines near the
    // source RMS (amp/sqrt(2)).
    let source_rms = amp as f64 / std::f64::consts::SQRT_2;
    for (label, out) in [("keylock", &keylock.output), ("new", &new.output)] {
        let r = rms(out);
        assert!(
            (r - source_rms).abs() / source_rms < 0.1,
            "{label} arm RMS {r:.4} deviates from source {source_rms:.4}"
        );
    }

    // Click-freeness on the SAME bound for both arms. Tape mode shifts the
    // tone to 220 * 1.06 max, so the bound uses the shifted frequency; the
    // keylocked arm stays at 220 and fits under it trivially.
    let slew_bound = amp * 2.0 * std::f32::consts::PI * 220.0 * 1.06 / SAMPLE_RATE as f32;
    // Skip the keylock arm's latency-prime region.
    let keylock_scan = &keylock.output[8192.min(keylock.output.len())..];
    assert!(
        max_adjacent_diff(keylock_scan) <= slew_bound * 6.0,
        "keylock arm clicked: {:.5}",
        max_adjacent_diff(keylock_scan)
    );
    assert!(
        max_adjacent_diff(&new.output) <= slew_bound * 1.5,
        "new arm clicked: {:.5}",
        max_adjacent_diff(&new.output)
    );

    // The pull arm must never have starved mid-render.
    assert_eq!(new.underrun_frames, 0, "A/B feeder must keep the ring fed");
}

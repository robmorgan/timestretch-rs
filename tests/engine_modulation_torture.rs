//! Tape-mode jog-wheel torture (ROADMAP new Stage 1): continuous
//! per-callback tempo modulation through the pull engine must be click-free.
//!
//! Port of the old `modulation_torture.rs` gesture generators onto the pull
//! API. In tape mode pitch follows tempo, so the output tone's frequency —
//! and therefore its slew bound — scales with the maximum gesture rate.

use std::f32::consts::PI;

use timestretch::engine::{Engine, EngineConfig, EngineProfile};

const SAMPLE_RATE: u32 = 44_100;
const CALLBACK_FRAMES: usize = 128;

fn sine(freq: f32, num_samples: usize, amp: f32) -> Vec<f32> {
    (0..num_samples)
        .map(|i| amp * (2.0 * PI * freq * i as f32 / SAMPLE_RATE as f32).sin())
        .collect()
}

/// Peak sample-to-sample step of a pure sine at `freq * max_rate` (tape
/// mode shifts pitch with tempo).
fn sine_max_slew(freq: f32, amp: f32, max_rate: f32) -> f32 {
    amp * 2.0 * PI * freq * max_rate / SAMPLE_RATE as f32
}

fn p95_adjacent_diff(samples: &[f32]) -> f32 {
    let mut diffs: Vec<f32> = samples
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .filter(|d| d.is_finite())
        .collect();
    if diffs.is_empty() {
        return 0.0;
    }
    diffs.sort_by(|a, b| a.total_cmp(b));
    diffs[((diffs.len() - 1) as f32 * 0.95).round() as usize]
}

fn max_adjacent_diff(samples: &[f32]) -> (usize, f32) {
    let mut worst = (0usize, 0.0f32);
    for (i, w) in samples.windows(2).enumerate() {
        let d = (w[1] - w[0]).abs();
        if d > worst.1 {
            worst = (i, d);
        }
    }
    worst
}

#[derive(Debug, Clone, Copy)]
enum Gesture {
    /// Platter nudge: 1.0 -> 1.04 over 200 ms, hold, back, rest; 1 Hz cycle.
    Nudge,
    /// Continuous pitch-fader ride: 1.0 + 0.06 * sin(2*pi*0.25*t).
    Ride,
    /// Sync snaps: instant jumps 1.08 / 0.92 / 1.0 within each second.
    Snap,
    /// Threshold ride: 1.0 + 0.11 * sin(2*pi*0.25*t) — sweeps the rate
    /// deviation through the SOLA/PV selection boundary (engage 8.5% /
    /// release 9.5%) twice per 4 s cycle, forcing repeated corrector
    /// handoffs mid-gesture.
    ThresholdRide,
}

impl Gesture {
    fn target_rate_at(self, t_secs: f64) -> f64 {
        match self {
            Gesture::Nudge => {
                let phase = t_secs.fract();
                if phase < 0.2 {
                    1.0 + 0.04 * (phase / 0.2)
                } else if phase < 0.3 {
                    1.04
                } else if phase < 0.5 {
                    1.04 - 0.04 * ((phase - 0.3) / 0.2)
                } else {
                    1.0
                }
            }
            Gesture::Ride => 1.0 + 0.06 * (2.0 * std::f64::consts::PI * 0.25 * t_secs).sin(),
            Gesture::ThresholdRide => {
                1.0 + 0.11 * (2.0 * std::f64::consts::PI * 0.25 * t_secs).sin()
            }
            Gesture::Snap => {
                let phase = t_secs.fract();
                if phase < 0.4 {
                    1.08
                } else if phase < 0.8 {
                    0.92
                } else {
                    1.0
                }
            }
        }
    }

    fn max_rate(self) -> f32 {
        match self {
            Gesture::Nudge => 1.04,
            Gesture::Ride => 1.06,
            Gesture::Snap => 1.08,
            Gesture::ThresholdRide => 1.11,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Gesture::Nudge => "nudge",
            Gesture::Ride => "ride",
            Gesture::Snap => "snap",
            Gesture::ThresholdRide => "threshold-ride",
        }
    }
}

/// Streams `input` through the engine, driving the tempo rate from the
/// gesture on every callback. Returns the collected output.
fn stream_profile_with_gesture(
    profile: EngineProfile,
    input: &[f32],
    channels: usize,
    gesture: Gesture,
) -> Vec<f32> {
    let handles = Engine::build(EngineConfig {
        sample_rate: SAMPLE_RATE,
        channels,
        profile,
        ..EngineConfig::default()
    })
    .unwrap();
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);

    // Render 70% of the source duration so the ride (rate can dip below
    // 1.0) never starves and every callback is fully covered by real audio.
    let total_frames = input.len() / channels;
    let callbacks = (total_frames as f64 * 0.7 / CALLBACK_FRAMES as f64) as usize;

    let mut feed_cursor = 0usize;
    let mut out = vec![0.0f32; CALLBACK_FRAMES * channels];
    let mut collected: Vec<f32> = Vec::with_capacity(callbacks * CALLBACK_FRAMES * channels);

    for cb in 0..callbacks {
        let t = (cb * CALLBACK_FRAMES) as f64 / SAMPLE_RATE as f64;
        controller.set_tempo_rate(gesture.target_rate_at(t));

        while feed_cursor < input.len()
            && source.occupied_frames() < source.demand_hint(CALLBACK_FRAMES, 1.1)
        {
            let end = (feed_cursor + 4096 * channels).min(input.len());
            let accepted = source.push(&input[feed_cursor..end]);
            feed_cursor += accepted * channels;
            if accepted == 0 {
                break;
            }
        }

        processor.process(&mut out);
        collected.extend_from_slice(&out);
    }

    assert_eq!(
        controller.underrun_frames(),
        0,
        "torture[{}] must not underrun",
        gesture.label()
    );
    collected
}

/// Tonal torture: gesture-modulated 220 Hz sine must stay click-free over
/// the full collected output (no warmup exclusion — the pull engine has no
/// warm-up region).
fn assert_tonal_torture(gesture: Gesture) {
    let freq = 220.0f32;
    let amp = 0.5f32;
    let input = sine(freq, SAMPLE_RATE as usize * 8, amp);
    let output = stream_profile_with_gesture(EngineProfile::Tape, &input, 1, gesture);

    let theoretical = sine_max_slew(freq, amp, gesture.max_rate());
    // The varispeed path is structurally click-free: no PV ratio-step seam,
    // no frame overlap. Bounds are far tighter than the old engine's 6x/1.5x.
    let hard_bound = theoretical * 1.5;
    let p95_bound = theoretical * 1.1;

    let (max_idx, max_diff) = max_adjacent_diff(&output);
    let p95 = p95_adjacent_diff(&output);
    println!(
        "engine-torture[{}]: len={} max_diff={:.5}@{} (cb_off={}) p95={:.5} bounds={:.5}/{:.5}",
        gesture.label(),
        output.len(),
        max_diff,
        max_idx,
        max_idx % CALLBACK_FRAMES,
        p95,
        hard_bound,
        p95_bound,
    );

    assert!(
        max_diff <= hard_bound,
        "torture[{}]: click at sample {} (callback offset {}): |delta|={:.5} > {:.5}",
        gesture.label(),
        max_idx,
        max_idx % CALLBACK_FRAMES,
        max_diff,
        hard_bound
    );
    assert!(
        p95 <= p95_bound,
        "torture[{}]: p95 adjacent diff {:.5} > {:.5}",
        gesture.label(),
        p95,
        p95_bound
    );
}

#[test]
fn engine_torture_nudge_tonal_no_clicks() {
    assert_tonal_torture(Gesture::Nudge);
}

#[test]
fn engine_torture_ride_tonal_no_clicks() {
    assert_tonal_torture(Gesture::Ride);
}

#[test]
fn engine_torture_snap_tonal_no_clicks() {
    assert_tonal_torture(Gesture::Snap);
}

/// Keylock torture: the corrected high band must stay click-free under the
/// same gestures (ROADMAP new Stage 2 exit criterion: clicks = 0 with
/// keylock engaged). The tone stays at its source pitch, so the slew bound
/// carries no rate factor; the margin covers the PV's overlap-add under a
/// moving transposition.
fn assert_keylock_torture(gesture: Gesture) {
    let freq = 220.0f32; // above the 150 Hz crossover: fully corrected
    let amp = 0.5f32;
    let input = sine(freq, SAMPLE_RATE as usize * 8, amp);
    let output = stream_profile_with_gesture(EngineProfile::Keylock, &input, 1, gesture);

    // Skip the pipeline-delay prefix of silence.
    let scan = &output[16_384..];
    let theoretical = sine_max_slew(freq, amp, 1.0);
    let hard_bound = theoretical * 3.0;
    let p95_bound = theoretical * 1.3;

    let (max_idx, max_diff) = max_adjacent_diff(scan);
    let p95 = p95_adjacent_diff(scan);
    println!(
        "keylock-torture[{}]: len={} max_diff={:.5}@{} p95={:.5} bounds={:.5}/{:.5}",
        gesture.label(),
        output.len(),
        max_diff,
        max_idx,
        p95,
        hard_bound,
        p95_bound,
    );
    assert!(
        max_diff <= hard_bound,
        "keylock torture[{}]: click at {}: {:.5} > {:.5}",
        gesture.label(),
        max_idx,
        max_diff,
        hard_bound
    );
    assert!(
        p95 <= p95_bound,
        "keylock torture[{}]: p95 {:.5} > {:.5}",
        gesture.label(),
        p95,
        p95_bound
    );
}

#[test]
fn engine_keylock_torture_nudge_no_clicks() {
    assert_keylock_torture(Gesture::Nudge);
}

#[test]
fn engine_keylock_torture_ride_no_clicks() {
    assert_keylock_torture(Gesture::Ride);
}

#[test]
fn engine_keylock_torture_snap_no_clicks() {
    assert_keylock_torture(Gesture::Snap);
}

#[test]
fn engine_keylock_torture_threshold_crossing_no_clicks() {
    // ROADMAP Stage 3 exit criterion: repeated rides through the SOLA/PV
    // boundary — clicks = 0, no audible mode-switch signature. 16 s covers
    // four cycles = eight boundary crossings with the ~250 ms dwell.
    // 880 Hz keeps the tone clear of the 150 Hz seam so the envelope gate
    // measures the HANDOFF, not band-edge detuning beat (that inherent
    // seam-adjacent ripple is gated separately in qa/engine_keylock.rs).
    let freq = 880.0f32;
    let amp = 0.5f32;
    let input = sine(freq, SAMPLE_RATE as usize * 16, amp);
    let output =
        stream_profile_with_gesture(EngineProfile::Keylock, &input, 1, Gesture::ThresholdRide);

    let scan = &output[16_384..];
    let theoretical = sine_max_slew(freq, amp, 1.0);
    let (max_idx, max_diff) = max_adjacent_diff(scan);
    let p95 = p95_adjacent_diff(scan);
    println!(
        "threshold-crossing torture: len={} max_diff={:.5}@{} p95={:.5}",
        output.len(),
        max_diff,
        max_idx,
        p95
    );
    assert!(
        max_diff <= theoretical * 3.0,
        "handoff click at {max_idx}: {max_diff:.5} > {:.5}",
        theoretical * 3.0
    );
    assert!(
        p95 <= theoretical * 1.3,
        "handoff p95 {p95:.5} > {:.5}",
        theoretical * 1.3
    );

    // Mode-switch signature gate: the tone's envelope must stay flat
    // through every handoff (a corrector swap that dips or doubles the
    // level reads as an audible switch even without a click).
    let win = 1_024;
    let mut min_rms = f64::MAX;
    let mut max_rms = 0.0f64;
    let mut pos = 16_384;
    while pos + win <= output.len() {
        let rms = (output[pos..pos + win]
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum::<f64>()
            / win as f64)
            .sqrt();
        min_rms = min_rms.min(rms);
        max_rms = max_rms.max(rms);
        pos += win / 2;
    }
    let swing_db = 20.0 * (max_rms / min_rms).log10();
    println!("threshold-crossing envelope swing: {swing_db:.2} dB");
    // Re-derived at Stage 9 (PV corrector deleted; SOLA carries the whole
    // corrected range): measured 0.03 dB on this ride — the old engine
    // measured 12.7 dB at just ±7.5%, and the PV-era chain 11.6 dB. Gate
    // with a wide regression envelope over the measured value.
    assert!(
        swing_db <= 1.0,
        "envelope swings {swing_db:.2} dB across the threshold ride \
         (known envelope 11.6 dB from the pinned PV sag)"
    );
}

#[test]
fn engine_keylock_torture_low_band_no_clicks() {
    // A 60 Hz tone rides the UN-corrected low band: tape behavior inside
    // the keylock chain. Pitch follows tempo, so the bound carries the
    // gesture's max rate; the band split and delay must add no clicks.
    let freq = 60.0f32;
    let amp = 0.6f32;
    let input = sine(freq, SAMPLE_RATE as usize * 8, amp);
    let output = stream_profile_with_gesture(EngineProfile::Keylock, &input, 1, Gesture::Ride);

    let scan = &output[16_384..];
    let theoretical = sine_max_slew(freq, amp, Gesture::Ride.max_rate());
    let (max_idx, max_diff) = max_adjacent_diff(scan);
    println!("keylock low-band torture: max_diff={max_diff:.5}@{max_idx}");
    assert!(
        max_diff <= theoretical * 1.5,
        "low-band click at {max_idx}: {max_diff:.5} > {:.5}",
        theoretical * 1.5
    );
}

#[test]
fn engine_torture_snap_stereo_stays_click_free_and_aligned() {
    let freq = 220.0f32;
    let amp = 0.5f32;
    let mono = sine(freq, SAMPLE_RATE as usize * 6, amp);
    let input: Vec<f32> = mono.iter().flat_map(|&s| [s, s * 0.9]).collect();

    let output = stream_profile_with_gesture(EngineProfile::Tape, &input, 2, Gesture::Snap);

    let left: Vec<f32> = output.iter().step_by(2).copied().collect();
    let right: Vec<f32> = output.iter().skip(1).step_by(2).copied().collect();
    let bound = sine_max_slew(freq, amp, Gesture::Snap.max_rate()) * 1.5;
    let (li, lmax) = max_adjacent_diff(&left);
    let (ri, rmax) = max_adjacent_diff(&right);
    assert!(lmax <= bound, "left click at {li}: {lmax:.5} > {bound:.5}");
    assert!(
        rmax <= bound * 0.9,
        "right click at {ri}: {rmax:.5} > {:.5}",
        bound * 0.9
    );

    // Channels must stay in exact varispeed lockstep: R = 0.9 * L.
    for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
        assert!(
            (r - 0.9 * l).abs() < 1e-4,
            "stereo drift at frame {i}: L={l} R={r}"
        );
    }
}

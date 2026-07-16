//! Live keylock toggle: `EngineController::set_keylock` must switch the
//! keylock chain between corrected and delay-matched varispeed output
//! mid-stream — no gap, no click, no latency change — so a deck can flip
//! Tape ↔ Keylock during live playback.

use timestretch::engine::{Engine, EngineConfig, EngineProfile};

const SAMPLE_RATE: u32 = 44_100;
const RATE: f64 = 1.06;
const SOURCE_HZ: f64 = 440.0;

/// Zero-crossing frequency estimate over a slice.
fn measure_freq(scan: &[f32]) -> f64 {
    let (mut first, mut last, mut count) = (None, None, 0usize);
    for i in 1..scan.len() {
        let (a, b) = (scan[i - 1] as f64, scan[i] as f64);
        if a <= 0.0 && b > 0.0 {
            let t = (i - 1) as f64 + a / (a - b);
            if first.is_none() {
                first = Some(t);
            }
            last = Some(t);
            count += 1;
        }
    }
    (count - 1) as f64 * SAMPLE_RATE as f64 / (last.unwrap() - first.unwrap())
}

#[test]
fn keylock_toggle_streams_through_without_gap_or_click() {
    let handles = Engine::build(EngineConfig {
        sample_rate: SAMPLE_RATE,
        channels: 1,
        profile: EngineProfile::Keylock,
        initial_tempo_rate: RATE,
        ..EngineConfig::default()
    })
    .unwrap();
    let (mut processor, mut source, controller) =
        (handles.processor, handles.source, handles.controller);
    let latency = processor.pipeline_latency_frames();
    assert!(
        latency > 0,
        "keylock chain reports a constant pipeline delay"
    );

    // Source sine at 440 Hz: varispeed plays it at 440 * rate; keylock
    // corrects the (high-band) pitch back to 440.
    let sr = SAMPLE_RATE as usize;
    let mut src_phase = 0.0f64;
    let mut next_source = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                src_phase += 2.0 * std::f64::consts::PI * SOURCE_HZ / SAMPLE_RATE as f64;
                0.6 * src_phase.sin() as f32
            })
            .collect()
    };

    // Stream 6 s of output; keylock off during [2 s, 4 s).
    let total_frames = 6 * sr;
    let mut out = Vec::with_capacity(total_frames);
    let mut chunk = vec![0.0f32; 1024];
    let mut keylock_on = true;
    while out.len() < total_frames {
        let want_on = !(2 * sr..4 * sr).contains(&out.len());
        if want_on != keylock_on {
            controller.set_keylock(want_on);
            keylock_on = want_on;
        }
        // Keep the ring comfortably ahead of the fastest consumption.
        while source.occupied_frames() < 8192 {
            let feed = next_source(2048);
            let accepted = source.push(&feed);
            assert_eq!(accepted, feed.len(), "source ring must not saturate here");
        }
        processor.process(&mut chunk);
        out.extend_from_slice(&chunk);
    }
    assert_eq!(
        controller.underrun_frames(),
        0,
        "no gap: stream never ran dry"
    );
    assert_eq!(
        processor.pipeline_latency_frames(),
        latency,
        "latency is constant across the toggle"
    );

    // No click: past cold start, no sample-to-sample step beyond the
    // signal's own slew (a hard, unramped switch fails this).
    let max_step = out
        .windows(2)
        .skip(sr / 2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max);
    let signal_slew =
        0.6 * (2.0 * std::f64::consts::PI * SOURCE_HZ * RATE / SAMPLE_RATE as f64) as f32;
    assert!(
        max_step < signal_slew * 1.5,
        "toggle clicked: max step {max_step:.4} vs signal slew {signal_slew:.4}"
    );

    // Each phase converges on its mode's pitch (measured well past fades).
    let cents = |f: f64, target: f64| 1200.0 * (f / target).log2();
    let corrected = measure_freq(&out[sr..2 * sr]);
    let bypassed = measure_freq(&out[3 * sr..4 * sr]);
    let recorrected = measure_freq(&out[5 * sr..]);
    assert!(
        cents(corrected, SOURCE_HZ).abs() < 12.0,
        "keylock phase off: {corrected:.2} Hz"
    );
    assert!(
        cents(bypassed, SOURCE_HZ * RATE).abs() < 12.0,
        "bypass phase not at varispeed pitch: {bypassed:.2} Hz"
    );
    assert!(
        cents(recorrected, SOURCE_HZ).abs() < 12.0,
        "re-enabled phase off: {recorrected:.2} Hz"
    );
}

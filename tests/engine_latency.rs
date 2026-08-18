//! New-engine port of the streaming-latency gates (ROADMAP new Stage 1):
//!
//! - First-sample-out equals the reported pipeline latency exactly (tape
//!   mode reports 0: output frame 0 IS source frame 0).
//! - Control-to-audio: a tempo retarget is audible within the resampler
//!   lookahead plus one caller block.

use timestretch::engine::{Engine, EngineConfig, EngineProfile};

const SAMPLE_RATE: u32 = 44_100;

fn mono_config() -> EngineConfig {
    EngineConfig {
        sample_rate: SAMPLE_RATE,
        channels: 1,
        profile: EngineProfile::Tape,
        ..EngineConfig::default()
    }
}

#[test]
fn first_sample_out_equals_reported_latency_at_unity() {
    let handles = Engine::build(mono_config()).unwrap();
    let (mut processor, mut source) = (handles.processor, handles.source);

    // Distinctive first samples so misalignment cannot pass.
    let input: Vec<f32> = (0..4096)
        .map(|i| ((i * 37 + 11) % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    source.push(&input);

    let latency = processor.pipeline_latency_frames();
    assert_eq!(latency, 0, "tape chain must report zero pipeline delay");

    let mut out = vec![0.0f32; 1024];
    processor.process(&mut out);

    // Exactly at the reported latency: output[latency] is source[0], and at
    // unity the sinc kernel is a delta, so the match is sample-exact.
    for (i, (&got, &want)) in out[latency..].iter().zip(input.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "output[{}] = {got}, expected source[{i}] = {want}",
            latency + i
        );
    }
}

#[test]
fn impulse_lands_at_mapped_position_at_non_unity_rate() {
    let rate = 1.05f64;
    let handles = Engine::build(EngineConfig {
        initial_tempo_rate: rate,
        ..mono_config()
    })
    .unwrap();
    let (mut processor, mut source) = (handles.processor, handles.source);

    let impulse_at = 1000usize;
    let mut input = vec![0.0f32; 8192];
    input[impulse_at] = 1.0;
    source.push(&input);

    let mut out = vec![0.0f32; 4096];
    processor.process(&mut out);

    let (peak_idx, _) = out
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .unwrap();
    // Output frame j reads source position j * rate: the impulse must land
    // at impulse_at / rate, within one sample of interpolation spread.
    let expected = (impulse_at as f64 / rate).round() as isize;
    assert!(
        (peak_idx as isize - expected).abs() <= 1,
        "impulse at output {peak_idx}, expected {expected}"
    );
}

#[test]
fn control_to_audio_within_lookahead_plus_one_block() {
    const CALLBACK_FRAMES: usize = 128;
    let handles = Engine::build(mono_config()).unwrap();
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);

    // A linear ramp turns the tempo rate into the output's local slope:
    // slope 1.0 at unity, slope = rate after the retarget.
    let ramp: Vec<f32> = (0..262_144).map(|i| i as f32).collect();
    source.push(&ramp[..32_768]);
    let mut feed_cursor = 32_768usize;

    let mut out = vec![0.0f32; CALLBACK_FRAMES];
    let mut collected: Vec<f32> = Vec::new();
    let retarget_at_callback = 40usize;
    let new_rate = 1.30f64;

    for cb in 0..80 {
        if cb == retarget_at_callback {
            controller.set_tempo_rate(new_rate);
        }
        if source.occupied_frames() < source.demand_hint(CALLBACK_FRAMES, 2.0) {
            let end = (feed_cursor + 8192).min(ramp.len());
            source.push(&ramp[feed_cursor..end]);
            feed_cursor = end;
        }
        processor.process(&mut out);
        collected.extend_from_slice(&out);
    }
    assert_eq!(controller.underrun_frames(), 0);

    // Find the first output frame whose local slope departs from 1.0.
    let request_frame = retarget_at_callback * CALLBACK_FRAMES;
    let mut first_change = None;
    for j in (request_frame.saturating_sub(256))..collected.len() - 1 {
        let slope = collected[j + 1] - collected[j];
        if (slope - 1.0).abs() > 0.02 {
            first_change = Some(j);
            break;
        }
    }
    let first_change = first_change.expect("retarget must become audible");

    assert!(
        first_change >= request_frame,
        "retarget audible at {first_change}, BEFORE it was requested at {request_frame}"
    );
    // Exit criterion: control-to-audio <= resampler lookahead + one caller
    // block. The engine's own bound must also honestly cover the measurement.
    let gate = 16 + CALLBACK_FRAMES;
    let measured = first_change - request_frame;
    assert!(
        measured <= gate,
        "control-to-audio {measured} frames exceeds lookahead + one block = {gate}"
    );
    assert!(
        measured <= processor.control_to_audio_bound_frames(),
        "measured {measured} exceeds the engine's reported bound {}",
        processor.control_to_audio_bound_frames()
    );

    // And the ramp must complete: slope settles at the new rate.
    let tail = &collected[first_change + 256..first_change + 512];
    for w in tail.windows(2) {
        let slope = (w[1] - w[0]) as f64;
        assert!(
            (slope - new_rate).abs() < 0.02,
            "slope {slope} never settled at {new_rate}"
        );
    }
}

#[test]
fn keylock_control_to_audio_unchanged_from_tape() {
    // ROADMAP Stage 2 exit criterion: control-to-audio unchanged from
    // Stage 1. The keylock chain adds a CONSTANT pipeline delay
    // (host-compensated); a tempo retarget must become audible exactly one
    // pipeline delay after the tape-mode deadline, i.e. the EXTRA control
    // latency beyond the constant delay stays within lookahead + one block.
    const CALLBACK_FRAMES: usize = 128;
    let handles = Engine::build(EngineConfig {
        sample_rate: SAMPLE_RATE,
        channels: 1,
        profile: EngineProfile::Keylock,
        ..EngineConfig::default()
    })
    .unwrap();
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);
    let pipeline = processor.pipeline_latency_frames();

    // Low-band ramp trick: below the crossover the chain is split + delay
    // only, so a linear ramp's local slope still reads the tempo rate
    // directly (the LR8 low-pass passes the ramp's low-frequency content).
    let ramp: Vec<f32> = (0..262_144).map(|i| i as f32 * 1e-4).collect();
    source.push(&ramp[..32_768]);
    let mut feed_cursor = 32_768usize;

    let mut out = vec![0.0f32; CALLBACK_FRAMES];
    let mut collected: Vec<f32> = Vec::new();
    let retarget_at_callback = 60usize;
    let new_rate = 1.30f64;

    for cb in 0..160 {
        if cb == retarget_at_callback {
            controller.set_tempo_rate(new_rate);
        }
        if source.occupied_frames() < source.demand_hint(CALLBACK_FRAMES, 2.0) {
            let end = (feed_cursor + 8192).min(ramp.len());
            source.push(&ramp[feed_cursor..end]);
            feed_cursor = end;
        }
        processor.process(&mut out);
        collected.extend_from_slice(&out);
    }
    assert_eq!(controller.underrun_frames(), 0);

    let request_frame = retarget_at_callback * CALLBACK_FRAMES;
    let mut first_change = None;
    for j in request_frame..collected.len() - 8 {
        // Slope over 8 frames to ride over filter ripple.
        let slope = (collected[j + 8] - collected[j]) as f64 / 8.0 / 1e-4;
        if (slope - 1.0).abs() > 0.05 {
            first_change = Some(j);
            break;
        }
    }
    let first_change = first_change.expect("retarget must become audible");
    let measured_extra = first_change as i64 - request_frame as i64 - pipeline as i64;
    // The ramp probe reads the LOW band, which carries the LR8 low-pass's
    // dispersive group delay at DC on top of the reported constant latency:
    // sum of 1/(Q_i * w0) over the four sections = (2/0.5412 + 2/1.3066)
    // / (2*pi*150 Hz) = 5.5 ms = ~245 frames at 44.1 kHz. Measured extra
    // 2026-07: 236 frames — the control path itself adds only the Stage 1
    // budget (lookahead + one caller block).
    const LR8_DC_GROUP_DELAY_FRAMES: i64 = 250;
    let gate = (16 + CALLBACK_FRAMES) as i64 + LR8_DC_GROUP_DELAY_FRAMES;
    assert!(
        measured_extra <= gate,
        "keylock control-to-audio: retarget audible {measured_extra} frames past the \
         constant pipeline delay (gate {gate} incl. crossover group delay)"
    );
}

#[test]
fn wide_keylock_control_to_audio_unchanged_from_tape() {
    // Same contract as the keylock row for the wide chain (ROADMAP Stage
    // 11): the profile adds a CONSTANT (larger) pipeline delay; a tempo
    // retarget must become audible within lookahead + one block past it.
    // The low-band ramp trick is unavailable — the wide corrector
    // keylocks the full spectrum — so the probe runs with the keylock
    // toggle OFF: the bypass arm is a pure matched delay, and the ramp
    // slope reads the tempo rate directly with no crossover group-delay
    // term.
    const CALLBACK_FRAMES: usize = 128;
    let handles = Engine::build(EngineConfig {
        sample_rate: SAMPLE_RATE,
        channels: 1,
        profile: EngineProfile::WideKeylock,
        ..EngineConfig::default()
    })
    .unwrap();
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);
    controller.set_keylock(false);
    let pipeline = processor.pipeline_latency_frames();

    let ramp: Vec<f32> = (0..262_144).map(|i| i as f32 * 1e-4).collect();
    source.push(&ramp[..32_768]);
    let mut feed_cursor = 32_768usize;

    let mut out = vec![0.0f32; CALLBACK_FRAMES];
    let mut collected: Vec<f32> = Vec::new();
    let retarget_at_callback = 60usize;
    let new_rate = 1.30f64;

    for cb in 0..220 {
        if cb == retarget_at_callback {
            controller.set_tempo_rate(new_rate);
        }
        if source.occupied_frames() < source.demand_hint(CALLBACK_FRAMES, 2.0) {
            let end = (feed_cursor + 8192).min(ramp.len());
            source.push(&ramp[feed_cursor..end]);
            feed_cursor = end;
        }
        processor.process(&mut out);
        collected.extend_from_slice(&out);
    }
    assert_eq!(controller.underrun_frames(), 0);

    let request_frame = retarget_at_callback * CALLBACK_FRAMES;
    let mut first_change = None;
    for j in request_frame..collected.len() - 8 {
        let slope = (collected[j + 8] - collected[j]) as f64 / 8.0 / 1e-4;
        if (slope - 1.0).abs() > 0.05 {
            first_change = Some(j);
            break;
        }
    }
    let first_change = first_change.expect("retarget must become audible");
    let measured_extra = first_change as i64 - request_frame as i64 - pipeline as i64;
    let gate = (16 + CALLBACK_FRAMES + 64) as i64;
    assert!(
        measured_extra <= gate,
        "wide keylock control-to-audio: retarget audible {measured_extra} frames past \
         the constant pipeline delay (gate {gate})"
    );
}

#[test]
fn timestamped_retarget_lands_on_exact_output_sample() {
    // ROADMAP Stage 5 exit criterion: a timestamped tempo step lands on
    // the requested output sample exactly. The ramp fixture makes the
    // output's local slope read the rate directly.
    const CALLBACK_FRAMES: usize = 128;
    let handles = Engine::build(mono_config()).unwrap();
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);

    let ramp: Vec<f32> = (0..262_144).map(|i| i as f32 * 1e-4).collect();
    source.push(&ramp[..32_768]);
    let mut feed_cursor = 32_768usize;

    // Schedule the step at a frame that is NOT block- or callback-aligned.
    let at_frame = 40 * CALLBACK_FRAMES as u64 + 37;
    controller.set_tempo_rate_at(1.5, at_frame);

    let mut out = vec![0.0f32; CALLBACK_FRAMES];
    let mut collected: Vec<f32> = Vec::new();
    for _ in 0..80 {
        if source.occupied_frames() < source.demand_hint(CALLBACK_FRAMES, 2.0) {
            let end = (feed_cursor + 8192).min(ramp.len());
            source.push(&ramp[feed_cursor..end]);
            feed_cursor = end;
        }
        processor.process(&mut out);
        collected.extend_from_slice(&out);
    }
    assert_eq!(controller.underrun_frames(), 0);

    // Slope must be exactly 1.0 up to (and including the step INTO)
    // `at_frame - 1`, and start deviating at `at_frame`.
    for j in 2_048..at_frame as usize - 1 {
        let slope = (collected[j + 1] - collected[j]) as f64 / 1e-4;
        assert!(
            (slope - 1.0).abs() < 0.01,
            "slope deviated before the timestamp at {j}: {slope}"
        );
    }
    let mut first_change = None;
    for j in at_frame as usize - 1..collected.len() - 1 {
        let slope = (collected[j + 1] - collected[j]) as f64 / 1e-4;
        if (slope - 1.0).abs() > 0.02 {
            first_change = Some(j);
            break;
        }
    }
    let first_change = first_change.expect("retarget must apply");
    assert!(
        first_change as u64 >= at_frame - 1 && first_change as u64 <= at_frame + 2,
        "retarget landed at {first_change}, requested {at_frame}"
    );
}

#[test]
fn retarget_before_first_process_applies_from_frame_zero() {
    let handles = Engine::build(mono_config()).unwrap();
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);

    controller.set_tempo_rate(2.0);
    let ramp: Vec<f32> = (0..16_384).map(|i| i as f32).collect();
    source.push(&ramp);

    let mut out = vec![0.0f32; 512];
    processor.process(&mut out);
    // Slope 2.0 right after the kernel's leading edge (the first ~20 output
    // frames convolve the zero-initialized history at the dilated pitch-up
    // kernel width): no glide, no stale-rate prefix.
    for w in out[40..500].windows(2) {
        let slope = w[1] - w[0];
        assert!(
            (slope - 2.0).abs() < 0.02,
            "pre-start retarget did not apply from frame zero (slope {slope})"
        );
    }
}

/// Issue #45: bulk-scheduling more than `MAX_PENDING_RETARGETS`
/// timestamped retargets degraded the overflow to ASAP latest-wins
/// SILENTLY — `dropped_events()` stayed 0 while a 92-event tempo curve
/// froze at the 8th event's rate with seconds of position drift. The
/// degradation is now counted.
#[test]
fn retarget_overflow_is_counted_not_silent() {
    use timestretch::engine::MAX_PENDING_RETARGETS;
    let handles = Engine::build(EngineConfig {
        sample_rate: SAMPLE_RATE,
        channels: 1,
        ..EngineConfig::default()
    })
    .unwrap();
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);

    // Within the cap: nothing degrades.
    for k in 0..MAX_PENDING_RETARGETS {
        controller.set_tempo_rate_at(1.0 + 0.001 * k as f64, 10_000 + 1_000 * k as u64);
    }
    source.push(&vec![0.0f32; 8_192]);
    let mut out = vec![0.0f32; 512];
    processor.process(&mut out);
    assert_eq!(controller.retargets_degraded(), 0);
    assert_eq!(controller.dropped_events(), 0);

    // Four past the cap: each degradation is counted, none silently.
    for k in 0..4u64 {
        controller.set_tempo_rate_at(1.05, 200_000 + 1_000 * k);
    }
    processor.process(&mut out);
    assert_eq!(
        controller.retargets_degraded(),
        4,
        "every overflow degradation must be counted"
    );
    assert_eq!(controller.dropped_events(), 0);
}

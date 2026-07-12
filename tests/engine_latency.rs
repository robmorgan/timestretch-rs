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

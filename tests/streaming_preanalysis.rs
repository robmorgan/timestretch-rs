//! Streaming pre-analysis integration: source-position tracking and
//! artifact-driven transient handling.

use std::f32::consts::PI;
use timestretch::{analyze_for_dj, ControlPath, EdmPreset, StreamProcessor, StretchParams};

const SR: u32 = 44_100;
const CHUNK: usize = 1024;

fn stream_params(ratio: f64) -> StretchParams {
    StretchParams::new(ratio)
        .with_sample_rate(SR)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch)
}

fn click_train(sample_rate: u32, bpm: f64, seconds: f64) -> Vec<f32> {
    let len = (sample_rate as f64 * seconds) as usize;
    let beat_interval = (60.0 * sample_rate as f64 / bpm) as usize;
    let mut out = vec![0.0f32; len];
    for i in (0..len).step_by(beat_interval.max(1)) {
        for j in 0..10.min(len - i) {
            out[i + j] = if j < 5 { 1.0 } else { -0.4 };
        }
    }
    for (i, s) in out.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;
        *s += 0.15 * (2.0 * PI * 110.0 * t).sin();
    }
    out
}

#[test]
fn test_source_position_tracks_passthrough_then_dsp() {
    let mut processor = StreamProcessor::new(stream_params(1.0));
    assert_eq!(processor.source_position(), 0);

    // Unity ratio: the passthrough fast path bypasses the input ring but
    // must still advance the source position.
    let input = click_train(SR, 128.0, 2.0);
    let mut output = Vec::with_capacity(input.len() * 4);
    let mut pushed = 0usize;
    for chunk in input.chunks(CHUNK).take(8) {
        processor
            .process_into(chunk, &mut output)
            .expect("unity process");
        pushed += chunk.len();
        assert_eq!(
            processor.source_position(),
            pushed,
            "passthrough position must equal frames pushed"
        );
    }

    // Engage the DSP path; the position now advances as frames are consumed,
    // always monotonic and never ahead of what was pushed.
    processor
        .set_stretch_ratio(1.1)
        .expect("ratio change should be accepted");
    let mut last_position = processor.source_position();
    for chunk in input.chunks(CHUNK).skip(8) {
        processor.process_into(chunk, &mut output).expect("process");
        pushed += chunk.len();
        let position = processor.source_position();
        assert!(position >= last_position, "position must be monotonic");
        assert!(
            position <= pushed,
            "position {} must not exceed pushed frames {}",
            position,
            pushed
        );
        last_position = position;
    }
    assert!(
        last_position > 8 * CHUNK,
        "DSP path must advance the position past the passthrough prefix"
    );
}

#[test]
fn test_set_source_position_rejected_mid_stream() {
    // After DSP-path input.
    let mut processor = StreamProcessor::new(stream_params(1.1));
    let input = click_train(SR, 128.0, 1.0);
    let mut output = Vec::with_capacity(input.len() * 4);
    processor
        .process_into(&input[..CHUNK], &mut output)
        .expect("process");
    assert!(processor.set_source_position(0).is_err());

    // After unity passthrough input.
    let mut processor = StreamProcessor::new(stream_params(1.0));
    output.clear();
    processor
        .process_into(&input[..CHUNK], &mut output)
        .expect("unity process");
    assert!(processor.set_source_position(0).is_err());
}

#[test]
fn test_reset_returns_position_to_zero() {
    let mut processor = StreamProcessor::new(stream_params(1.1));
    processor
        .set_source_position(123_456)
        .expect("fresh processor accepts a source position");
    assert_eq!(processor.source_position(), 123_456);

    let input = click_train(SR, 128.0, 1.0);
    let mut output = Vec::with_capacity(input.len() * 4);
    for chunk in input.chunks(CHUNK) {
        processor.process_into(chunk, &mut output).expect("process");
    }
    assert!(processor.source_position() > 123_456);

    processor.reset();
    assert_eq!(processor.source_position(), 0);
    processor
        .set_source_position(777)
        .expect("freshly-reset processor accepts a source position");
    assert_eq!(processor.source_position(), 777);
}

#[test]
fn test_artifact_attaches_via_params_and_setter() {
    let input = click_train(SR, 128.0, 4.0);
    let artifact = analyze_for_dj(&input, SR);

    // Via params at construction.
    let params = stream_params(1.1)
        .with_pre_analysis(artifact.clone())
        .with_beat_snap_confidence_threshold(0.1);
    let mut processor = StreamProcessor::new(params);
    let mut output = Vec::with_capacity(input.len() * 4);
    for chunk in input.chunks(CHUNK) {
        processor.process_into(chunk, &mut output).expect("process");
    }
    assert!(!output.is_empty());

    // Via the runtime setter (build/rebuild-time API).
    let mut processor =
        StreamProcessor::new(stream_params(1.1).with_beat_snap_confidence_threshold(0.1));
    processor.set_pre_analysis(Some(artifact));
    output.clear();
    for chunk in input.chunks(CHUNK) {
        processor.process_into(chunk, &mut output).expect("process");
    }
    assert!(!output.is_empty());
}

/// Simple local peak detector: samples above `threshold` that are local
/// maxima, deduplicated by `min_distance`.
fn detect_peaks(signal: &[f32], threshold: f32, min_distance: usize) -> Vec<usize> {
    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..signal.len().saturating_sub(1) {
        let s = signal[i].abs();
        if s >= threshold && s >= signal[i - 1].abs() && s >= signal[i + 1].abs() {
            if let Some(&last) = peaks.last() {
                if i - last < min_distance {
                    if s > signal[last].abs() {
                        *peaks.last_mut().unwrap() = i;
                    }
                    continue;
                }
            }
            peaks.push(i);
        }
    }
    peaks
}

#[test]
fn test_artifact_scheduled_resets_replace_online_scheduler() {
    let input = click_train(SR, 128.0, 4.0);
    let artifact = analyze_for_dj(&input, SR);
    assert!(artifact.transient_onsets.len() >= 6);

    let params = stream_params(1.1)
        .with_pre_analysis(artifact.clone())
        .with_beat_snap_confidence_threshold(0.1);
    let mut processor = StreamProcessor::new(params);
    let mut output = Vec::with_capacity(input.len() * 4);
    for chunk in input.chunks(CHUNK) {
        processor.process_into(chunk, &mut output).expect("process");
    }

    let stats = processor.transient_reset_stats();
    assert_eq!(
        stats.events_detected_total, 0,
        "online scheduler must stay idle while an artifact drives resets"
    );
    // Render passes tile the consumed timeline contiguously from frame 0,
    // so exactly the onsets inside the consumed span are scheduled.
    let consumed = stats.input_frames_consumed_total;
    let expected = artifact
        .transient_onsets
        .iter()
        .filter(|&&onset| onset < consumed)
        .count() as u64;
    assert!(expected > 0, "test input too short to consume any onsets");
    assert_eq!(stats.artifact_events_scheduled_total, expected);
    assert!(
        stats.reset_band_counts_total[2] > 0 && stats.reset_band_counts_total[3] > 0,
        "artifact events must reset the upper bands"
    );
}

#[test]
fn test_seek_offsets_artifact_scheduling() {
    let input = click_train(SR, 128.0, 4.0);
    let artifact = analyze_for_dj(&input, SR);
    assert!(artifact.transient_onsets.len() >= 4);
    // Start between onsets 1 and 2, as a seek-rebuild flow would.
    let seek = (artifact.transient_onsets[1] + artifact.transient_onsets[2]) / 2;

    let params = stream_params(1.1)
        .with_pre_analysis(artifact.clone())
        .with_beat_snap_confidence_threshold(0.1);
    let mut processor = StreamProcessor::new(params);
    processor.set_source_position(seek).expect("seek position");

    let mut output = Vec::with_capacity(input.len() * 4);
    for chunk in input[seek..].chunks(CHUNK) {
        processor.process_into(chunk, &mut output).expect("process");
    }

    let stats = processor.transient_reset_stats();
    assert_eq!(stats.events_detected_total, 0);
    let consumed_end = seek + stats.input_frames_consumed_total;
    let expected = artifact
        .transient_onsets
        .iter()
        .filter(|&&onset| onset >= seek && onset < consumed_end)
        .count() as u64;
    assert!(expected >= 2, "seek test needs onsets past the seek point");
    assert_eq!(
        stats.artifact_events_scheduled_total, expected,
        "onsets before the seek point must not fire; those after must"
    );
}

#[test]
fn test_artifact_low_band_resets_suppressed_during_ratio_glide() {
    let input = click_train(SR, 128.0, 4.0);
    let artifact = analyze_for_dj(&input, SR);

    let params = stream_params(1.08)
        .with_pre_analysis(artifact)
        .with_beat_snap_confidence_threshold(0.1);
    let mut processor = StreamProcessor::new(params);
    let mut output = Vec::with_capacity(input.len() * 4);
    let mut toggle = false;
    for chunk in input.chunks(CHUNK) {
        // Keep a ratio slew in flight for the entire stream.
        toggle = !toggle;
        processor
            .set_stretch_ratio(if toggle { 1.12 } else { 1.08 })
            .expect("ratio change");
        processor.process_into(chunk, &mut output).expect("process");
    }

    let stats = processor.transient_reset_stats();
    assert!(stats.artifact_events_scheduled_total > 0);
    assert_eq!(
        stats.reset_band_counts_total[0], 0,
        "sub-bass resets must stay suppressed during modulation"
    );
    assert_eq!(
        stats.reset_band_counts_total[1], 0,
        "low-band resets must stay suppressed during modulation"
    );
    assert!(
        stats.reset_band_counts_total[2] > 0,
        "upper bands still re-lock during modulation"
    );
}

#[test]
fn test_varispeed_artifact_onsets_fire_once_at_source_positions_under_ride() {
    let input = click_train(SR, 128.0, 4.0);
    let artifact = analyze_for_dj(&input, SR);
    assert!(artifact.transient_onsets.len() >= 6);

    let params = stream_params(1.06)
        .with_pre_analysis(artifact.clone())
        .with_beat_snap_confidence_threshold(0.1);
    let mut processor = StreamProcessor::new(params);
    processor
        .set_control_path(ControlPath::VarispeedFirst)
        .expect("varispeed path");

    // Ride the tempo 1.02..1.10 with sample-accurate retargets every chunk;
    // onset positions are source frames, so the resampled->source mapping
    // must keep them firing exactly once each despite the moving ratio.
    let mut output = Vec::with_capacity(input.len() * 6);
    for (i, chunk) in input.chunks(CHUNK).enumerate() {
        let t = i as f64 * CHUNK as f64 / SR as f64;
        processor
            .set_stretch_ratio(1.06 + 0.04 * (2.0 * std::f64::consts::PI * t / 2.0).sin())
            .expect("ride ratio");
        processor.process_into(chunk, &mut output).expect("process");
    }

    let stats = processor.transient_reset_stats();
    assert_eq!(
        stats.events_detected_total, 0,
        "online scheduler must stay idle while an artifact drives resets"
    );
    // The consumed span in SOURCE frames is the source position; exactly the
    // onsets inside it must have been scheduled — no doubles, no misses.
    let consumed_source = processor.source_position();
    let expected = artifact
        .transient_onsets
        .iter()
        .filter(|&&onset| onset < consumed_source)
        .count() as u64;
    assert!(expected >= 4, "ride test too short to consume onsets");
    assert_eq!(
        stats.artifact_events_scheduled_total, expected,
        "artifact onsets must fire exactly once at mapped source positions"
    );
    assert!(
        stats.reset_band_counts_total[2] > 0 && stats.reset_band_counts_total[3] > 0,
        "artifact events must reset the upper bands"
    );
}

#[test]
fn test_artifact_stream_transient_preservation_parity() {
    let bpm = 128.0;
    let input = click_train(SR, bpm, 4.0);
    let artifact = analyze_for_dj(&input, SR);
    let ratio = 1.1;

    let run = |params: StretchParams| {
        let mut processor = StreamProcessor::new(params);
        let mut output = Vec::with_capacity(input.len() * 4);
        for chunk in input.chunks(CHUNK) {
            processor.process_into(chunk, &mut output).expect("process");
        }
        let mut tail = Vec::with_capacity(input.len() * 2);
        processor.flush_into(&mut tail).expect("flush");
        output.extend_from_slice(&tail);
        output
    };

    let out_online = run(stream_params(ratio));
    let out_artifact = run(stream_params(ratio)
        .with_pre_analysis(artifact)
        .with_beat_snap_confidence_threshold(0.1));

    let beat_interval = 60.0 * SR as f64 / bpm;
    let min_distance = (beat_interval * ratio * 0.5) as usize;
    let peaks_online = detect_peaks(&out_online, 0.3, min_distance);
    let peaks_artifact = detect_peaks(&out_artifact, 0.3, min_distance);
    assert!(
        peaks_artifact.len() + 1 >= peaks_online.len(),
        "artifact-driven stream preserved fewer transients than online: {} vs {}",
        peaks_artifact.len(),
        peaks_online.len()
    );
    assert!(
        peaks_artifact.len() >= 4,
        "artifact-driven stream lost transients: {} peaks",
        peaks_artifact.len()
    );
}

use std::f32::consts::PI;
use timestretch::{EdmPreset, StreamProcessor, StreamingEngine, StretchParams};

fn stereo_input(sample_rate: u32, frames: usize) -> Vec<f32> {
    let mut input = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        let t = i as f32 / sample_rate as f32;
        input.push((2.0 * PI * 110.0 * t).sin() * 0.35 + (2.0 * PI * 220.0 * t).sin() * 0.12);
        input.push((2.0 * PI * 330.0 * t).sin() * 0.30 + (2.0 * PI * 550.0 * t).sin() * 0.10);
    }
    input
}

fn mono_input(sample_rate: u32, frames: usize) -> Vec<f32> {
    let mut input = Vec::with_capacity(frames);
    for i in 0..frames {
        let t = i as f32 / sample_rate as f32;
        input.push(
            (2.0 * PI * 147.0 * t).sin() * 0.42
                + (2.0 * PI * 293.0 * t).sin() * 0.17
                + (2.0 * PI * 587.0 * t).sin() * 0.08,
        );
    }
    input
}

#[test]
fn fixed_buffer_budget_helpers_cover_public_callback_flow() {
    let params = StretchParams::new(1.03)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_sample_rate(44_100)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256);
    let mut vec_proc = StreamProcessor::new(params.clone());
    let mut fixed_proc = StreamProcessor::new(params);
    vec_proc.set_streaming_engine(StreamingEngine::Deterministic);
    fixed_proc.set_streaming_engine(StreamingEngine::Deterministic);
    vec_proc.set_pitch_scale(1.05).unwrap();
    fixed_proc.set_pitch_scale(1.05).unwrap();

    let input = stereo_input(44_100, 256 * 16);
    let mut expected = Vec::with_capacity(input.len() * 2);
    let mut actual = Vec::with_capacity(input.len() * 2);

    for chunk in input.chunks(256 * 2) {
        vec_proc.process_into(chunk, &mut expected).unwrap();

        let budget = fixed_proc
            .max_process_interleaved_output_samples(chunk.len())
            .unwrap();
        assert_eq!(budget % 2, 0, "budget must stay frame-aligned");

        let mut callback_output = vec![0.0f32; budget];
        let written = fixed_proc
            .process_interleaved_into(chunk, &mut callback_output)
            .unwrap();
        assert!(written <= budget);
        assert_eq!(written % 2, 0, "written samples must stay frame-aligned");
        actual.extend_from_slice(&callback_output[..written]);
    }

    vec_proc.flush_into(&mut expected).unwrap();

    let mut flush_calls = 0usize;
    loop {
        let budget = fixed_proc.max_flush_interleaved_output_samples().unwrap();
        if budget == 0 {
            break;
        }
        flush_calls += 1;

        let mut flush_output = vec![0.0f32; budget];
        let written = fixed_proc
            .flush_interleaved_into(&mut flush_output)
            .unwrap();
        assert!(written > 0);
        assert!(written <= budget);
        assert_eq!(written % 2, 0, "flush writes must stay frame-aligned");
        actual.extend_from_slice(&flush_output[..written]);
    }

    assert!(flush_calls > 0, "expected an end-of-stream tail to drain");
    assert_eq!(
        fixed_proc.max_flush_interleaved_output_samples().unwrap(),
        0
    );

    let mut empty = [0.0f32; 0];
    assert_eq!(fixed_proc.flush_interleaved_into(&mut empty).unwrap(), 0);

    assert_eq!(expected.len(), actual.len());
    for (idx, (&lhs, &rhs)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-6,
            "Mismatch at sample {idx}: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn fixed_buffer_callbacks_match_vec_flush_on_irregular_stream_lengths() {
    let params = StretchParams::new(0.91)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_sample_rate(44_100)
        .with_channels(1)
        .with_fft_size(1024)
        .with_hop_size(256);
    let mut vec_proc = StreamProcessor::new(params.clone());
    let mut fixed_proc = StreamProcessor::new(params);
    vec_proc.set_streaming_engine(StreamingEngine::Deterministic);
    fixed_proc.set_streaming_engine(StreamingEngine::Deterministic);
    vec_proc.set_pitch_scale(1.07).unwrap();
    fixed_proc.set_pitch_scale(1.07).unwrap();

    let input = mono_input(44_100, 256 * 9 + 73);
    let chunk_sizes = [97usize, 131, 83, 191, 109];
    let mut expected = Vec::with_capacity(input.len() * 2);
    let mut actual = Vec::with_capacity(input.len() * 2);

    let mut offset = 0usize;
    let mut chunk_idx = 0usize;
    while offset < input.len() {
        let chunk_len = chunk_sizes[chunk_idx % chunk_sizes.len()];
        let end = (offset + chunk_len).min(input.len());
        let chunk = &input[offset..end];

        vec_proc.process_into(chunk, &mut expected).unwrap();

        let budget = fixed_proc
            .max_process_interleaved_output_samples(chunk.len())
            .unwrap();
        let mut callback_output = vec![0.0f32; budget];
        let written = fixed_proc
            .process_interleaved_into(chunk, &mut callback_output)
            .unwrap();
        actual.extend_from_slice(&callback_output[..written]);

        offset = end;
        chunk_idx += 1;
    }

    vec_proc.flush_into(&mut expected).unwrap();

    let mut flush_output = [0.0f32; 11];
    loop {
        let written = fixed_proc
            .flush_interleaved_into(&mut flush_output)
            .unwrap();
        if written == 0 {
            break;
        }
        actual.extend_from_slice(&flush_output[..written]);
    }

    assert_eq!(expected.len(), actual.len());
    for (idx, (&lhs, &rhs)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-6,
            "Mismatch at sample {idx}: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn fixed_buffer_queued_output_helper_drains_midstream_pending_exactly() {
    let params = StretchParams::new(1.03)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_sample_rate(44_100)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256);
    let mut vec_proc = StreamProcessor::new(params.clone());
    let mut fixed_proc = StreamProcessor::new(params);
    vec_proc.set_streaming_engine(StreamingEngine::Deterministic);
    fixed_proc.set_streaming_engine(StreamingEngine::Deterministic);
    vec_proc.set_pitch_scale(1.05).unwrap();
    fixed_proc.set_pitch_scale(1.05).unwrap();

    let input = stereo_input(44_100, 256 * 12);
    let mut expected = Vec::with_capacity(input.len() * 2);
    let mut actual = Vec::with_capacity(input.len() * 2);

    let mut callback_output = [0.0f32; 6];
    for chunk in input.chunks(256 * 2) {
        vec_proc.process_into(chunk, &mut expected).unwrap();

        let written = fixed_proc
            .process_interleaved_into(chunk, &mut callback_output)
            .unwrap();
        actual.extend_from_slice(&callback_output[..written]);

        if fixed_proc.queued_interleaved_output_samples().unwrap() >= 24 {
            break;
        }
    }

    let queued = fixed_proc.queued_interleaved_output_samples().unwrap();
    assert!(queued >= 24, "expected pending output to exercise helper");
    assert_eq!(queued % 2, 0, "queued samples must stay frame-aligned");

    let mut drain_output = vec![0.0f32; queued];
    let drained = fixed_proc
        .process_interleaved_into(&[], &mut drain_output)
        .unwrap();
    assert_eq!(drained, queued);
    actual.extend_from_slice(&drain_output[..drained]);

    assert_eq!(fixed_proc.queued_interleaved_output_samples().unwrap(), 0);
    assert_eq!(expected.len(), actual.len());
    for (idx, (&lhs, &rhs)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-6,
            "Mismatch at sample {idx}: {lhs} vs {rhs}"
        );
    }
}

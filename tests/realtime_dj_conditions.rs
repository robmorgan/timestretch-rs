use std::f32::consts::PI;

use timestretch::{QualityMode, StreamProcessor, StretchParams};

fn stereo_test_signal(sample_rate: u32, seconds: usize) -> Vec<f32> {
    let frames = sample_rate as usize * seconds;
    let mut out = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        let t = i as f32 / sample_rate as f32;
        let left = (2.0 * PI * 110.0 * t).sin() * 0.6 + (2.0 * PI * 440.0 * t).sin() * 0.25;
        let right = (2.0 * PI * 55.0 * t).sin() * 0.5 + (2.0 * PI * 880.0 * t).sin() * 0.2;
        out.push(left);
        out.push(right);
    }
    out
}

fn run_stream_case(
    input: &[f32],
    callback_frames: usize,
    mut processor: StreamProcessor,
) -> Vec<f32> {
    let mut output = Vec::with_capacity((input.len() as f64 * 1.25) as usize + 32768);
    let (_, _, in_cap0, pending_cap0) = processor.capacities();

    for (idx, chunk) in input.chunks(callback_frames * 2).enumerate() {
        processor
            .process_into(chunk, &mut output)
            .expect("process_into failed");

        if idx > 4 {
            let (_, _, in_cap, pending_cap) = processor.capacities();
            assert_eq!(in_cap, in_cap0, "input ring capacity grew");
            assert_eq!(pending_cap, pending_cap0, "pending ring capacity grew");
        }
    }

    processor
        .flush_into(&mut output)
        .expect("flush_into failed");
    output
}

#[test]
fn realtime_callback_128_frames_10s() {
    let sr = 44_100;
    let ratio = 1.02;
    let input = stereo_test_signal(sr, 10);

    let params = StretchParams::new(ratio)
        .with_sample_rate(sr)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256)
        .with_quality_mode(QualityMode::Balanced);
    let processor = StreamProcessor::new(params);

    let output = run_stream_case(&input, 128, processor);
    assert!(!output.is_empty());

    let length_ratio = output.len() as f64 / input.len() as f64;
    assert!(
        (length_ratio - ratio).abs() < 0.35,
        "length ratio {} diverged from expected {}",
        length_ratio,
        ratio
    );
}

#[test]
fn realtime_callback_64_frames_10s() {
    let sr = 44_100;
    let ratio = 1.02;
    let input = stereo_test_signal(sr, 10);

    let params = StretchParams::new(ratio)
        .with_sample_rate(sr)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256)
        .with_quality_mode(QualityMode::LowLatency);
    let processor = StreamProcessor::new(params);

    let output = run_stream_case(&input, 64, processor);
    assert!(!output.is_empty());
}

#[test]
fn realtime_callback_1024_frames_no_overflow() {
    let sr = 48_000;
    let input = stereo_test_signal(sr, 10);

    let params = StretchParams::new(1.01)
        .with_sample_rate(sr)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256)
        .with_quality_mode(QualityMode::MaxQuality);
    let processor = StreamProcessor::new(params);

    let output = run_stream_case(&input, 1024, processor);
    assert!(!output.is_empty());
}

#[test]
fn realtime_midstream_ratio_changes_every_500ms() {
    let sr = 44_100;
    let callback_frames = 128;
    let input = stereo_test_signal(sr, 10);

    let params = StretchParams::new(1.0)
        .with_sample_rate(sr)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256)
        .with_quality_mode(QualityMode::Balanced);
    let mut processor = StreamProcessor::new(params);

    let mut output = Vec::with_capacity((input.len() as f64 * 1.35) as usize + 32768);
    let change_interval_frames = sr as usize / 2;
    let mut next_change = change_interval_frames;
    let ratios = [0.98, 1.03, 0.95, 1.06, 1.0];
    let mut ratio_idx = 0usize;

    let mut processed_frames = 0usize;
    let mut max_callback_write = 0usize;

    for chunk in input.chunks(callback_frames * 2) {
        if processed_frames >= next_change {
            processor.set_stretch_ratio(ratios[ratio_idx % ratios.len()]);
            ratio_idx += 1;
            next_change += change_interval_frames;
        }

        let before = output.len();
        processor
            .process_into(chunk, &mut output)
            .expect("process_into failed after ratio change");
        let written = output.len() - before;
        max_callback_write = max_callback_write.max(written);

        processed_frames += chunk.len() / 2;
    }

    processor
        .flush_into(&mut output)
        .expect("flush_into failed after ratio changes");

    assert!(!output.is_empty());
    assert!(
        max_callback_write <= callback_frames * 2 * 64,
        "unexpected callback write spike: {} samples",
        max_callback_write
    );
}

/// Integration tests for DJ workflow features: resample, crossfade, reverse, and
/// combined stretch+crossfade pipelines.
use timestretch::{AudioBuffer, EdmPreset, StretchParams};

const SAMPLE_RATE: u32 = 44100;

fn make_sine(freq: f32, duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let n = (duration_secs * sample_rate as f32) as usize;
    (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

fn make_stereo_sine(freq_l: f32, freq_r: f32, duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let n = (duration_secs * sample_rate as f32) as usize;
    let mut data = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        data.push((2.0 * std::f32::consts::PI * freq_l * t).sin());
        data.push((2.0 * std::f32::consts::PI * freq_r * t).sin());
    }
    data
}

// --- Resample integration tests ---

#[test]
fn test_resample_then_stretch() {
    // A 48kHz buffer resampled to 44.1kHz, then stretched
    let samples = make_sine(440.0, 1.0, 48000);
    let buf = AudioBuffer::from_mono(samples, 48000);
    let resampled = buf.resample(SAMPLE_RATE);

    let params = StretchParams::new(1.5)
        .with_channels(1)
        .with_sample_rate(SAMPLE_RATE);
    let output = timestretch::stretch_buffer(&resampled, &params).unwrap();

    // Output should be ~1.5x longer
    let ratio = output.num_frames() as f64 / resampled.num_frames() as f64;
    assert!(
        (ratio - 1.5).abs() < 0.15,
        "Stretch ratio should be ~1.5, got {}",
        ratio
    );
    assert!(output.data.iter().all(|s| s.is_finite()));
}

#[test]
fn test_resample_stereo_preserves_channels() {
    let data = make_stereo_sine(440.0, 880.0, 0.5, 48000);
    let buf = AudioBuffer::from_stereo(data, 48000);
    let resampled = buf.resample(SAMPLE_RATE);

    assert!(resampled.is_stereo());
    assert_eq!(resampled.channel_count(), 2);

    // Duration should be preserved (~0.5 seconds)
    assert!((resampled.duration_secs() - 0.5).abs() < 0.02);
}

#[test]
fn test_resample_roundtrip_preserves_frequency() {
    // 44100 → 48000 → 44100 should approximately preserve a 440Hz sine
    let samples = make_sine(440.0, 0.5, SAMPLE_RATE);
    let buf = AudioBuffer::from_mono(samples, SAMPLE_RATE);
    let up = buf.resample(48000);
    let back = up.resample(SAMPLE_RATE);

    // Should have same number of frames
    assert_eq!(back.num_frames(), buf.num_frames());

    // Middle section should be close to original (skip edges for boundary effects)
    let skip = 100;
    let end = buf.num_frames() - skip;
    let mut max_err: f32 = 0.0;
    for i in skip..end {
        let err = (buf.data[i] - back.data[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }
    // Cubic interpolation round-trip should be reasonably accurate
    assert!(
        max_err < 0.1,
        "Resample round-trip max error too high: {}",
        max_err
    );
}

// --- Crossfade integration tests ---

#[test]
fn test_crossfade_two_stretched_tracks() {
    // Simulate DJ mixing: stretch two tracks to same BPM, then crossfade
    let track_a = make_sine(440.0, 1.0, SAMPLE_RATE);
    let track_b = make_sine(660.0, 1.0, SAMPLE_RATE);
    let buf_a = AudioBuffer::from_mono(track_a, SAMPLE_RATE);
    let buf_b = AudioBuffer::from_mono(track_b, SAMPLE_RATE);

    let params_a = StretchParams::new(1.02)
        .with_channels(1)
        .with_sample_rate(SAMPLE_RATE)
        .with_preset(EdmPreset::DjBeatmatch);
    let params_b = StretchParams::new(0.98)
        .with_channels(1)
        .with_sample_rate(SAMPLE_RATE)
        .with_preset(EdmPreset::DjBeatmatch);

    let stretched_a = timestretch::stretch_buffer(&buf_a, &params_a).unwrap();
    let stretched_b = timestretch::stretch_buffer(&buf_b, &params_b).unwrap();

    // Crossfade with 2000 frames (~45ms) overlap
    let mixed = stretched_a.crossfade_into(&stretched_b, 2000);

    assert!(!mixed.is_empty());
    let expected_len = stretched_a.num_frames() + stretched_b.num_frames() - 2000;
    assert_eq!(mixed.num_frames(), expected_len);
    assert!(mixed.data.iter().all(|s| s.is_finite()));
    // No clipping
    assert!(
        mixed.data.iter().all(|s| s.abs() <= 2.0),
        "Crossfade output should not clip excessively"
    );
}

#[test]
fn test_crossfade_stereo_dj_workflow() {
    let data_a = make_stereo_sine(440.0, 880.0, 0.5, SAMPLE_RATE);
    let data_b = make_stereo_sine(330.0, 660.0, 0.5, SAMPLE_RATE);
    let buf_a = AudioBuffer::from_stereo(data_a, SAMPLE_RATE);
    let buf_b = AudioBuffer::from_stereo(data_b, SAMPLE_RATE);

    let mixed = buf_a.crossfade_into(&buf_b, 1000);

    let expected_frames = (SAMPLE_RATE as usize / 2) * 2 - 1000;
    assert_eq!(mixed.num_frames(), expected_frames);
    assert!(mixed.is_stereo());
    assert!(mixed.data.iter().all(|s| s.is_finite()));
}

// --- Reverse integration tests ---

#[test]
fn test_reverse_preserves_length() {
    let buf = AudioBuffer::from_mono(make_sine(440.0, 0.5, SAMPLE_RATE), SAMPLE_RATE);
    let rev = buf.reverse();
    assert_eq!(rev.num_frames(), buf.num_frames());
    assert_eq!(rev.sample_rate, buf.sample_rate);
}

#[test]
fn test_reverse_stereo_preserves_channel_pairing() {
    let data = make_stereo_sine(440.0, 880.0, 0.1, SAMPLE_RATE);
    let buf = AudioBuffer::from_stereo(data, SAMPLE_RATE);
    let rev = buf.reverse();

    // First frame of reversed = last frame of original
    let last_frame = buf.num_frames() - 1;
    assert_eq!(rev.data[0], buf.data[last_frame * 2]); // L
    assert_eq!(rev.data[1], buf.data[last_frame * 2 + 1]); // R
}

#[test]
fn test_reverse_then_stretch() {
    // Reverse a signal, stretch it, result should be finite and non-empty
    let buf = AudioBuffer::from_mono(make_sine(440.0, 0.5, SAMPLE_RATE), SAMPLE_RATE);
    let rev = buf.reverse();

    let params = StretchParams::new(2.0)
        .with_channels(1)
        .with_sample_rate(SAMPLE_RATE)
        .with_preset(EdmPreset::Halftime);
    let output = timestretch::stretch_buffer(&rev, &params).unwrap();

    assert!(!output.is_empty());
    assert!(output.data.iter().all(|s| s.is_finite()));
}

// --- Combined workflow tests ---

#[test]
fn test_full_dj_workflow_resample_stretch_crossfade() {
    // Complete DJ workflow:
    // 1. Track A at 48kHz, Track B at 44.1kHz
    // 2. Resample both to 44.1kHz
    // 3. Stretch both to match 128 BPM
    // 4. Crossfade

    let track_a = AudioBuffer::from_mono(make_sine(440.0, 1.0, 48000), 48000);
    let track_b = AudioBuffer::from_mono(make_sine(330.0, 1.0, SAMPLE_RATE), SAMPLE_RATE);

    // Step 1: Resample track A to 44.1kHz
    let track_a_resampled = track_a.resample(SAMPLE_RATE);
    assert_eq!(track_a_resampled.sample_rate, SAMPLE_RATE);

    // Step 2: Stretch both (126 BPM → 128 BPM for A, 130 BPM → 128 BPM for B)
    let params = StretchParams::new(126.0 / 128.0)
        .with_channels(1)
        .with_sample_rate(SAMPLE_RATE)
        .with_preset(EdmPreset::DjBeatmatch);
    let stretched_a = timestretch::stretch_buffer(&track_a_resampled, &params).unwrap();

    let params_b = StretchParams::new(130.0 / 128.0)
        .with_channels(1)
        .with_sample_rate(SAMPLE_RATE)
        .with_preset(EdmPreset::DjBeatmatch);
    let stretched_b = timestretch::stretch_buffer(&track_b, &params_b).unwrap();

    // Step 3: Crossfade
    let mixed = stretched_a.crossfade_into(&stretched_b, 4410); // 100ms crossfade
    assert!(!mixed.is_empty());
    assert!(mixed.data.iter().all(|s| s.is_finite()));
}

#[test]
fn test_reverse_cymbal_build() {
    // Creative DJ effect: reverse a sound, fade in, then crossfade into original
    let buf = AudioBuffer::from_mono(make_sine(1000.0, 0.5, SAMPLE_RATE), SAMPLE_RATE);

    let reversed = buf.reverse().fade_in(buf.num_frames());
    let mixed = reversed.crossfade_into(&buf, 2000);

    assert!(!mixed.is_empty());
    assert!(mixed.data.iter().all(|s| s.is_finite()));
    // The result should be smooth (no extreme jumps at crossfade)
    let crossfade_start = reversed.num_frames() - 2000;
    for i in crossfade_start..(crossfade_start + 2000).min(mixed.num_frames() - 1) {
        let diff = (mixed.data[i + 1] - mixed.data[i]).abs();
        assert!(diff < 1.0, "Unexpected discontinuity at sample {}", i);
    }
}

#[test]
fn test_slice_stretch_concatenate() {
    // Chop a sample into segments, stretch differently, recombine
    let buf = AudioBuffer::from_mono(make_sine(440.0, 1.0, SAMPLE_RATE), SAMPLE_RATE);

    let first_half = buf.slice(0, SAMPLE_RATE as usize / 2);
    let second_half = buf.slice(SAMPLE_RATE as usize / 2, SAMPLE_RATE as usize / 2);

    let params_slow = StretchParams::new(1.5)
        .with_channels(1)
        .with_sample_rate(SAMPLE_RATE);
    let params_fast = StretchParams::new(0.75)
        .with_channels(1)
        .with_sample_rate(SAMPLE_RATE);

    let stretched_first = timestretch::stretch_buffer(&first_half, &params_slow).unwrap();
    let stretched_second = timestretch::stretch_buffer(&second_half, &params_fast).unwrap();

    let combined = stretched_first.crossfade_into(&stretched_second, 1000);
    assert!(!combined.is_empty());
    assert!(combined.data.iter().all(|s| s.is_finite()));
}

#[test]
fn test_channel_count_after_operations() {
    let mono = AudioBuffer::from_mono(vec![0.0; 100], SAMPLE_RATE);
    let stereo = AudioBuffer::from_stereo(vec![0.0; 200], SAMPLE_RATE);

    assert_eq!(mono.channel_count(), 1);
    assert_eq!(stereo.channel_count(), 2);

    // After resample
    assert_eq!(mono.resample(48000).channel_count(), 1);
    assert_eq!(stereo.resample(48000).channel_count(), 2);

    // After reverse
    assert_eq!(mono.reverse().channel_count(), 1);
    assert_eq!(stereo.reverse().channel_count(), 2);
}

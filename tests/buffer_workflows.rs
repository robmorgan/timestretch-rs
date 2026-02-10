//! Integration tests for AudioBuffer utility methods in real stretch workflows.
//!
//! These tests combine slice, concatenate, normalize, fade, trim_silence,
//! peak, rms, and frames() with the time-stretching pipeline to validate
//! that the utilities work correctly in practice.

use timestretch::{AudioBuffer, Channels, EdmPreset, StretchParams};

/// Generate a mono sine wave at the given frequency.
fn sine_mono(freq: f32, sample_rate: u32, num_frames: usize) -> AudioBuffer {
    let data: Vec<f32> = (0..num_frames)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
        .collect();
    AudioBuffer::from_mono(data, sample_rate)
}

/// Generate a stereo sine wave (same frequency both channels).
fn sine_stereo(freq: f32, sample_rate: u32, num_frames: usize) -> AudioBuffer {
    let mut data = Vec::with_capacity(num_frames * 2);
    for i in 0..num_frames {
        let s = (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin();
        data.push(s);
        data.push(s);
    }
    AudioBuffer::from_stereo(data, sample_rate)
}

// ==================== SLICE + STRETCH ====================

#[test]
fn test_slice_then_stretch() {
    // Take a longer buffer, slice a portion, stretch it
    let full = sine_mono(440.0, 44100, 44100); // 1 second
    let portion = full.slice(10000, 20000); // middle ~0.45s
    assert_eq!(portion.num_frames(), 20000);

    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let output = timestretch::stretch_buffer(&portion, &params).unwrap();

    // Output should be roughly 1.5x the input length
    let ratio = output.num_frames() as f64 / portion.num_frames() as f64;
    assert!(
        (ratio - 1.5).abs() < 0.15,
        "Expected ~1.5x, got {:.3}x",
        ratio
    );
    assert!(output.peak() > 0.01, "Output should not be silent");
}

#[test]
fn test_stretch_then_slice() {
    // Stretch a full buffer, then slice the output
    let input = sine_mono(220.0, 44100, 22050); // 0.5s
    let params = StretchParams::new(2.0)
        .with_sample_rate(44100)
        .with_channels(1)
        .with_preset(EdmPreset::Halftime);
    let stretched = timestretch::stretch_buffer(&input, &params).unwrap();

    // Slice first and second half
    let half_len = stretched.num_frames() / 2;
    let first_half = stretched.slice(0, half_len);
    let second_half = stretched.slice(half_len, half_len);

    assert!(first_half.peak() > 0.01);
    assert!(second_half.peak() > 0.01);
    assert_eq!(
        first_half.num_frames() + second_half.num_frames(),
        half_len * 2
    );
}

// ==================== CONCATENATE + STRETCH ====================

#[test]
fn test_concatenate_then_stretch() {
    // Join two different-frequency segments, then stretch
    let low = sine_mono(110.0, 44100, 11025); // 0.25s at 110 Hz
    let high = sine_mono(880.0, 44100, 11025); // 0.25s at 880 Hz
    let combined = AudioBuffer::concatenate(&[&low, &high]);
    assert_eq!(combined.num_frames(), 22050);

    let params = StretchParams::new(1.2)
        .with_sample_rate(44100)
        .with_channels(1);
    let output = timestretch::stretch_buffer(&combined, &params).unwrap();

    assert!(!output.is_empty());
    assert!(output.peak() > 0.01);
}

#[test]
fn test_stretch_then_concatenate() {
    // Stretch two segments independently, then join
    let a = sine_mono(440.0, 44100, 22050);
    let b = sine_mono(220.0, 44100, 22050);

    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);

    let stretched_a = timestretch::stretch_buffer(&a, &params).unwrap();
    let stretched_b = timestretch::stretch_buffer(&b, &params).unwrap();

    let combined = AudioBuffer::concatenate(&[&stretched_a, &stretched_b]);
    assert_eq!(
        combined.num_frames(),
        stretched_a.num_frames() + stretched_b.num_frames()
    );
    assert!(combined.peak() > 0.01);
}

// ==================== NORMALIZE + STRETCH ====================

#[test]
fn test_normalize_before_stretch() {
    // Normalize a quiet signal, then stretch
    let quiet = AudioBuffer::from_mono(
        (0..22050)
            .map(|i| 0.1 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect(),
        44100,
    );
    assert!(quiet.peak() < 0.15);

    let normalized = quiet.normalize(1.0);
    assert!((normalized.peak() - 1.0).abs() < 1e-4);

    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let output = timestretch::stretch_buffer(&normalized, &params).unwrap();

    assert!(!output.is_empty());
    // Output should have significant energy since input was normalized
    assert!(
        output.rms() > 0.1,
        "Normalized input should produce loud output"
    );
}

#[test]
fn test_stretch_then_normalize() {
    let input = sine_mono(440.0, 44100, 22050);
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let stretched = timestretch::stretch_buffer(&input, &params).unwrap();

    let normalized = stretched.normalize(0.5);
    assert!((normalized.peak() - 0.5).abs() < 1e-4);
}

// ==================== FADE + STRETCH ====================

#[test]
fn test_fade_in_out_then_stretch() {
    let input = sine_mono(440.0, 44100, 44100); // 1 second
    let faded = input.fade_in(4410).fade_out(4410); // 100ms fades

    // First samples should be near zero, middle should have content
    assert!(faded.data[0].abs() < 0.01);
    // Middle of faded region should have significant energy
    let mid_rms = AudioBuffer::from_mono(faded.data[20000..24000].to_vec(), 44100).rms();
    assert!(
        mid_rms > 0.3,
        "Middle region RMS should be significant: {}",
        mid_rms
    );

    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let output = timestretch::stretch_buffer(&faded, &params).unwrap();

    assert!(!output.is_empty());
}

#[test]
fn test_stretch_then_fade() {
    let input = sine_mono(440.0, 44100, 22050);
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let stretched = timestretch::stretch_buffer(&input, &params).unwrap();

    let fade_frames = stretched.num_frames() / 10;
    let faded = stretched.fade_in(fade_frames).fade_out(fade_frames);

    // First sample should be zero (fade in starts at 0)
    assert!(faded.data[0].abs() < 1e-6);
    // Last sample should be near zero (fade out ends at 0)
    assert!(faded.data[faded.data.len() - 1].abs() < 0.1);
}

// ==================== TRIM SILENCE + STRETCH ====================

#[test]
fn test_trim_silence_after_stretch() {
    // Stretch a signal with leading/trailing silence
    let mut data = vec![0.0f32; 4410]; // 100ms silence
    data.extend(
        (0..22050).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin()),
    );
    data.extend(vec![0.0f32; 4410]); // 100ms silence
    let input = AudioBuffer::from_mono(data, 44100);

    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let stretched = timestretch::stretch_buffer(&input, &params).unwrap();
    let trimmed = stretched.trim_silence(0.01);

    // Trimmed should be shorter (silence removed)
    assert!(
        trimmed.num_frames() < stretched.num_frames(),
        "Trimming should reduce frame count"
    );
    // But still have content
    assert!(trimmed.peak() > 0.1);
}

// ==================== PEAK / RMS + STRETCH ====================

#[test]
fn test_rms_preserved_after_identity_stretch() {
    let input = sine_mono(440.0, 44100, 22050);
    let input_rms = input.rms();

    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let output = timestretch::stretch_buffer(&input, &params).unwrap();
    let output_rms = output.rms();

    // RMS should be roughly preserved at identity ratio
    let ratio = output_rms / input_rms;
    assert!(
        (ratio - 1.0).abs() < 0.3,
        "RMS should be roughly preserved: ratio={}",
        ratio
    );
}

#[test]
fn test_peak_gain_roundtrip() {
    let input = sine_mono(440.0, 44100, 22050);
    let original_peak = input.peak();

    // Apply -6dB, then normalize back
    let quieter = input.apply_gain(-6.0);
    assert!(quieter.peak() < original_peak);

    let restored = quieter.normalize(original_peak);
    assert!((restored.peak() - original_peak).abs() < 1e-4);
}

// ==================== FRAMES ITERATOR + STRETCH ====================

#[test]
fn test_frames_iterator_with_stereo_stretch() {
    let input = sine_stereo(440.0, 44100, 22050);
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(2);
    let output = timestretch::stretch_buffer(&input, &params).unwrap();

    // Iterate frames and verify each is stereo
    let mut frame_count = 0;
    for frame in &output {
        assert_eq!(frame.len(), 2, "Each stereo frame should have 2 samples");
        assert!(frame[0].is_finite());
        assert!(frame[1].is_finite());
        frame_count += 1;
    }
    assert_eq!(frame_count, output.num_frames());
}

#[test]
fn test_frames_iterator_peak_matches() {
    let input = sine_mono(440.0, 44100, 22050);
    let params = StretchParams::new(1.2)
        .with_sample_rate(44100)
        .with_channels(1);
    let output = timestretch::stretch_buffer(&input, &params).unwrap();

    // Compute peak manually via frames iterator
    let manual_peak = output.frames().map(|f| f[0].abs()).fold(0.0f32, f32::max);
    assert!((manual_peak - output.peak()).abs() < 1e-6);
}

// ==================== AS_REF + PARTIAL_EQ ====================

#[test]
fn test_as_ref_interop() {
    let input = sine_mono(440.0, 44100, 22050);
    let slice: &[f32] = input.as_ref();

    // Verify we can pass the slice to stretch directly
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let output = timestretch::stretch(slice, &params).unwrap();
    assert!(!output.is_empty());
}

#[test]
fn test_partial_eq_after_clone_and_modify() {
    let a = sine_mono(440.0, 44100, 22050);
    let b = a.clone();
    assert_eq!(a, b);

    // After applying gain, they should differ
    let c = a.apply_gain(6.0);
    assert_ne!(a, c);
}

// ==================== COMPLEX WORKFLOW ====================

#[test]
fn test_dj_crossfade_workflow() {
    // Simulate a DJ crossfade: stretch two tracks, fade them, merge
    let track_a = sine_mono(440.0, 44100, 44100); // 1s at 440 Hz
    let track_b = sine_mono(330.0, 44100, 44100); // 1s at 330 Hz

    let params = StretchParams::new(126.0 / 128.0) // slight tempo adjust
        .with_sample_rate(44100)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let a_stretched = timestretch::stretch_buffer(&track_a, &params).unwrap();
    let b_stretched = timestretch::stretch_buffer(&track_b, &params).unwrap();

    // Fade out track A, fade in track B
    let crossfade_frames = a_stretched.num_frames().min(b_stretched.num_frames()) / 4;
    let a_faded = a_stretched.fade_out(crossfade_frames);
    let b_faded = b_stretched.fade_in(crossfade_frames);

    assert!(a_faded.peak() > 0.01);
    assert!(b_faded.peak() > 0.01);
}

#[test]
fn test_sample_chop_workflow() {
    // Simulate chopping a sample: slice, stretch, normalize, fade
    let full_sample = sine_mono(220.0, 44100, 88200); // 2 seconds

    // Chop out 4 equal slices
    let frames_per_chop = full_sample.num_frames() / 4;
    let chops: Vec<AudioBuffer> = (0..4)
        .map(|i| full_sample.slice(i * frames_per_chop, frames_per_chop))
        .collect();

    // Stretch each chop to 2x (halftime effect)
    let params = StretchParams::new(2.0)
        .with_sample_rate(44100)
        .with_channels(1)
        .with_preset(EdmPreset::Halftime);

    let stretched_chops: Vec<AudioBuffer> = chops
        .iter()
        .map(|c| timestretch::stretch_buffer(c, &params).unwrap())
        .collect();

    // Normalize each to 0.8 peak
    let normalized: Vec<AudioBuffer> = stretched_chops.iter().map(|c| c.normalize(0.8)).collect();

    // Add fades
    let faded: Vec<AudioBuffer> = normalized
        .iter()
        .map(|c| {
            let fade = c.num_frames() / 20; // 5% fade
            c.fade_in(fade).fade_out(fade)
        })
        .collect();

    // Concatenate back together
    let refs: Vec<&AudioBuffer> = faded.iter().collect();
    let result = AudioBuffer::concatenate(&refs);

    assert!(result.num_frames() > full_sample.num_frames()); // Stretched 2x
    assert!((result.peak() - 0.8).abs() < 0.05); // Normalized to 0.8 (fades reduce slightly)
    assert!(result.rms() > 0.1); // Has content
}

#[test]
fn test_channels_from_count_in_params() {
    // Use Channels::from_count in a real workflow
    let channels = Channels::from_count(2).unwrap();
    assert_eq!(channels, Channels::Stereo);

    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(channels.count() as u32);
    assert_eq!(params.channels, Channels::Stereo);
}

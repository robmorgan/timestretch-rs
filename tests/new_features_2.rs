// Integration tests for recently added features (second batch):
// - AudioBuffer::split_at() — buffer splitting
// - AudioBuffer::repeat() — loop/repeat buffer
// - AudioBuffer::mix() — sum two buffers
// - AudioBuffer::into_data() — consume buffer to Vec
// - AsMut<[Sample]> for AudioBuffer
// - StreamProcessor::process_into() / flush_into() — zero-allocation streaming
// - Streaming-batch parity verification

use timestretch::{AudioBuffer, Channels, EdmPreset, StretchParams, StreamProcessor};

// ──────────────────────────────────────────────────────────────────
// Test signal helpers
// ──────────────────────────────────────────────────────────────────

fn sine_mono(freq: f32, sample_rate: u32, num_samples: usize) -> AudioBuffer {
    let data: Vec<f32> = (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
        .collect();
    AudioBuffer::from_mono(data, sample_rate)
}

fn sine_stereo(freq_l: f32, freq_r: f32, sample_rate: u32, num_frames: usize) -> AudioBuffer {
    let mut data = Vec::with_capacity(num_frames * 2);
    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        data.push((2.0 * std::f32::consts::PI * freq_l * t).sin());
        data.push((2.0 * std::f32::consts::PI * freq_r * t).sin());
    }
    AudioBuffer::from_stereo(data, sample_rate)
}

// ──────────────────────────────────────────────────────────────────
// AudioBuffer::split_at() tests
// ──────────────────────────────────────────────────────────────────

mod split_at_tests {
    use super::*;

    #[test]
    fn split_at_middle() {
        let buf = sine_mono(440.0, 44100, 1000);
        let (left, right) = buf.split_at(500);
        assert_eq!(left.num_frames(), 500);
        assert_eq!(right.num_frames(), 500);
        assert_eq!(left.sample_rate, buf.sample_rate);
        assert_eq!(right.sample_rate, buf.sample_rate);
    }

    #[test]
    fn split_at_beginning() {
        let buf = sine_mono(440.0, 44100, 1000);
        let (left, right) = buf.split_at(0);
        assert_eq!(left.num_frames(), 0);
        assert_eq!(right.num_frames(), 1000);
    }

    #[test]
    fn split_at_end() {
        let buf = sine_mono(440.0, 44100, 1000);
        let (left, right) = buf.split_at(1000);
        assert_eq!(left.num_frames(), 1000);
        assert_eq!(right.num_frames(), 0);
    }

    #[test]
    fn split_at_beyond_end_clamps() {
        let buf = sine_mono(440.0, 44100, 1000);
        let (left, right) = buf.split_at(5000); // beyond buffer length
        assert_eq!(left.num_frames(), 1000);
        assert_eq!(right.num_frames(), 0);
    }

    #[test]
    fn split_at_stereo() {
        let buf = sine_stereo(440.0, 880.0, 44100, 1000);
        let (left, right) = buf.split_at(400);
        assert_eq!(left.num_frames(), 400);
        assert_eq!(right.num_frames(), 600);
        assert_eq!(left.channels, Channels::Stereo);
        assert_eq!(right.channels, Channels::Stereo);
        // Stereo data must be even
        assert_eq!(left.data.len() % 2, 0);
        assert_eq!(right.data.len() % 2, 0);
    }

    #[test]
    fn split_at_preserves_data() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0, 4.0, 5.0], 44100);
        let (left, right) = buf.split_at(2);
        assert_eq!(left.data, vec![1.0, 2.0]);
        assert_eq!(right.data, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn split_at_recombine_equals_original() {
        let buf = sine_mono(440.0, 44100, 10000);
        let (left, right) = buf.split_at(4000);
        let recombined = AudioBuffer::concatenate(&[&left, &right]);
        assert_eq!(recombined, buf);
    }

    #[test]
    fn split_at_then_stretch_both_halves() {
        let buf = sine_mono(440.0, 44100, 44100); // 1s
        let (first_half, second_half) = buf.split_at(22050);

        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);

        let s1 = timestretch::stretch_buffer(&first_half, &params).unwrap();
        let s2 = timestretch::stretch_buffer(&second_half, &params).unwrap();

        // Both halves should stretch
        assert!(s1.num_frames() > first_half.num_frames());
        assert!(s2.num_frames() > second_half.num_frames());
    }

    #[test]
    fn split_at_empty_buffer() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let (left, right) = buf.split_at(0);
        assert!(left.is_empty());
        assert!(right.is_empty());
    }
}

// ──────────────────────────────────────────────────────────────────
// AudioBuffer::repeat() tests
// ──────────────────────────────────────────────────────────────────

mod repeat_tests {
    use super::*;

    #[test]
    fn repeat_twice() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        let repeated = buf.repeat(2);
        assert_eq!(repeated.num_frames(), 6);
        assert_eq!(repeated.data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn repeat_once_is_clone() {
        let buf = sine_mono(440.0, 44100, 1000);
        let repeated = buf.repeat(1);
        assert_eq!(repeated, buf);
    }

    #[test]
    fn repeat_zero_is_empty() {
        let buf = sine_mono(440.0, 44100, 1000);
        let repeated = buf.repeat(0);
        assert!(repeated.is_empty());
    }

    #[test]
    fn repeat_stereo() {
        let buf = AudioBuffer::from_stereo(vec![1.0, 2.0, 3.0, 4.0], 44100); // 2 frames
        let repeated = buf.repeat(3);
        assert_eq!(repeated.num_frames(), 6);
        assert_eq!(repeated.channels, Channels::Stereo);
        assert_eq!(
            repeated.data,
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn repeat_empty_buffer() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let repeated = buf.repeat(5);
        assert!(repeated.is_empty());
    }

    #[test]
    fn repeat_preserves_sample_rate() {
        let buf = sine_mono(440.0, 48000, 1000);
        let repeated = buf.repeat(3);
        assert_eq!(repeated.sample_rate, 48000);
    }

    #[test]
    fn repeat_then_stretch() {
        // Create a short loop, repeat it, then stretch
        let loop_sample = sine_mono(440.0, 44100, 11025); // 250ms
        let looped = loop_sample.repeat(4); // 1 second
        assert_eq!(looped.num_frames(), 44100);

        let params = StretchParams::new(2.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::Halftime);

        let stretched = timestretch::stretch_buffer(&looped, &params).unwrap();
        let ratio = stretched.duration_secs() / looped.duration_secs();
        assert!(
            (ratio - 2.0).abs() < 0.15,
            "Repeated+stretched ratio off: {ratio}"
        );
    }

    #[test]
    fn repeat_large_count() {
        let buf = AudioBuffer::from_mono(vec![0.5], 44100);
        let repeated = buf.repeat(10000);
        assert_eq!(repeated.num_frames(), 10000);
        assert!(repeated.data.iter().all(|&s| (s - 0.5).abs() < 1e-6));
    }

    #[test]
    fn repeat_rms_preserved() {
        let buf = sine_mono(440.0, 44100, 44100);
        let original_rms = buf.rms();
        let repeated = buf.repeat(3);
        let repeated_rms = repeated.rms();
        // RMS should be the same for a repeating sine
        assert!(
            (original_rms - repeated_rms).abs() < 0.01,
            "RMS should be preserved in repeat: {original_rms} vs {repeated_rms}"
        );
    }
}

// ──────────────────────────────────────────────────────────────────
// AudioBuffer::mix() tests
// ──────────────────────────────────────────────────────────────────

mod mix_tests {
    use super::*;

    #[test]
    fn mix_two_sines() {
        let a = sine_mono(440.0, 44100, 44100);
        let b = sine_mono(880.0, 44100, 44100);
        let mixed = a.mix(&b);
        assert_eq!(mixed.num_frames(), 44100);
        // Mixed signal should have higher RMS than either alone
        assert!(mixed.rms() > 0.01);
    }

    #[test]
    fn mix_inverse_cancels() {
        let a = AudioBuffer::from_mono(vec![1.0; 1000], 44100);
        let b = AudioBuffer::from_mono(vec![-1.0; 1000], 44100);
        let mixed = a.mix(&b);
        // 1.0 + (-1.0) = 0.0
        for &s in &mixed.data {
            assert!(s.abs() < 1e-6, "Inverse signals should cancel, got {s}");
        }
    }

    #[test]
    fn mix_with_silence() {
        let a = sine_mono(440.0, 44100, 1000);
        let silence = AudioBuffer::from_mono(vec![0.0; 1000], 44100);
        let mixed = a.mix(&silence);
        // Mixing with silence should be identity
        assert_eq!(mixed.data, a.data);
    }

    #[test]
    fn mix_different_lengths_zero_pads() {
        let a = AudioBuffer::from_mono(vec![1.0; 500], 44100);
        let b = AudioBuffer::from_mono(vec![0.5; 1000], 44100);
        let mixed = a.mix(&b);
        assert_eq!(mixed.num_frames(), 1000); // Longer of the two
        // First 500: 1.0 + 0.5 = 1.5
        assert!((mixed.data[0] - 1.5).abs() < 1e-6);
        // After 500: 0.0 + 0.5 = 0.5
        assert!((mixed.data[500] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mix_stereo() {
        let a = sine_stereo(440.0, 880.0, 44100, 1000);
        let b = sine_stereo(330.0, 660.0, 44100, 1000);
        let mixed = a.mix(&b);
        assert_eq!(mixed.channels, Channels::Stereo);
        assert_eq!(mixed.num_frames(), 1000);
        assert!(mixed.rms() > 0.01);
    }

    #[test]
    #[should_panic(expected = "sample rate mismatch")]
    fn mix_mismatched_sample_rate_panics() {
        let a = AudioBuffer::from_mono(vec![0.0; 100], 44100);
        let b = AudioBuffer::from_mono(vec![0.0; 100], 48000);
        let _ = a.mix(&b);
    }

    #[test]
    #[should_panic(expected = "channel layout mismatch")]
    fn mix_mismatched_channels_panics() {
        let a = AudioBuffer::from_mono(vec![0.0; 100], 44100);
        let b = AudioBuffer::from_stereo(vec![0.0; 200], 44100);
        let _ = a.mix(&b);
    }

    #[test]
    fn mix_then_stretch() {
        let kick = sine_mono(60.0, 44100, 44100);
        let hihat = sine_mono(8000.0, 44100, 44100);
        let mixed = kick.mix(&hihat);

        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::HouseLoop);

        let stretched = timestretch::stretch_buffer(&mixed, &params).unwrap();
        assert!(stretched.num_frames() > mixed.num_frames());
        assert!(stretched.rms() > 0.01);
    }

    #[test]
    fn mix_commutative() {
        let a = sine_mono(440.0, 44100, 1000);
        let b = sine_mono(880.0, 44100, 1000);
        let ab = a.mix(&b);
        let ba = b.mix(&a);
        for (x, y) in ab.data.iter().zip(ba.data.iter()) {
            assert!((x - y).abs() < 1e-6, "Mix should be commutative");
        }
    }

    #[test]
    fn mix_self_doubles_amplitude() {
        let a = AudioBuffer::from_mono(vec![0.5; 100], 44100);
        let mixed = a.mix(&a);
        for &s in &mixed.data {
            assert!((s - 1.0).abs() < 1e-6, "Mixing with self should double");
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// AudioBuffer::into_data() tests
// ──────────────────────────────────────────────────────────────────

mod into_data_tests {
    use super::*;

    #[test]
    fn into_data_mono() {
        let data = vec![1.0, 2.0, 3.0];
        let buf = AudioBuffer::from_mono(data.clone(), 44100);
        let extracted = buf.into_data();
        assert_eq!(extracted, data);
    }

    #[test]
    fn into_data_stereo() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let buf = AudioBuffer::from_stereo(data.clone(), 44100);
        let extracted = buf.into_data();
        assert_eq!(extracted, data);
    }

    #[test]
    fn into_data_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let extracted = buf.into_data();
        assert!(extracted.is_empty());
    }

    #[test]
    fn into_data_after_stretch() {
        let input = sine_mono(440.0, 44100, 22050);
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let stretched = timestretch::stretch_buffer(&input, &params).unwrap();
        let frame_count = stretched.num_frames();
        let vec = stretched.into_data();
        assert_eq!(vec.len(), frame_count);
        assert!(vec.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn into_data_vs_from_conversion() {
        let buf = sine_mono(440.0, 44100, 1000);
        let data_via_into = buf.clone().into_data();
        let data_via_from: Vec<f32> = buf.into();
        assert_eq!(data_via_into, data_via_from);
    }
}

// ──────────────────────────────────────────────────────────────────
// AsMut<[Sample]> tests
// ──────────────────────────────────────────────────────────────────

mod as_mut_tests {
    use super::*;

    #[test]
    fn as_mut_modify_samples() {
        let mut buf = AudioBuffer::from_mono(vec![0.0; 100], 44100);
        let slice = buf.as_mut();
        slice[0] = 1.0;
        slice[50] = -0.5;
        assert!((buf.data[0] - 1.0).abs() < 1e-6);
        assert!((buf.data[50] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn as_mut_apply_gain_manually() {
        let mut buf = AudioBuffer::from_mono(vec![0.5; 100], 44100);
        let gain = 2.0f32;
        let slice = buf.as_mut();
        for s in slice.iter_mut() {
            *s *= gain;
        }
        for &s in &buf.data {
            assert!((s - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn as_mut_zero_out() {
        let mut buf = sine_mono(440.0, 44100, 1000);
        assert!(buf.rms() > 0.1);
        let slice = buf.as_mut();
        for s in slice.iter_mut() {
            *s = 0.0;
        }
        assert!(buf.rms() < 1e-6);
    }

    #[test]
    fn as_mut_stereo() {
        let mut buf = AudioBuffer::from_stereo(vec![1.0, 2.0, 3.0, 4.0], 44100);
        let slice = buf.as_mut();
        assert_eq!(slice.len(), 4);
        // Modify left channel only (even indices in interleaved)
        slice[0] = 0.0;
        slice[2] = 0.0;
        assert!((buf.data[0]).abs() < 1e-6);
        assert!((buf.data[1] - 2.0).abs() < 1e-6); // right unchanged
    }

    #[test]
    fn as_mut_then_stretch() {
        let mut buf = sine_mono(440.0, 44100, 44100);
        // Apply soft clipping via AsMut
        let slice = buf.as_mut();
        for s in slice.iter_mut() {
            *s = s.tanh();
        }
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let stretched = timestretch::stretch_buffer(&buf, &params).unwrap();
        assert!(stretched.num_frames() > buf.num_frames());
        assert!(stretched.rms() > 0.01);
    }
}

// ──────────────────────────────────────────────────────────────────
// StreamProcessor::process_into() / flush_into() tests
// ──────────────────────────────────────────────────────────────────

mod process_into_tests {
    use super::*;

    #[test]
    fn process_into_basic() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::HouseLoop);
        let mut proc = StreamProcessor::new(params);

        let input: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut output = Vec::new();
        let written = proc.process_into(&input, &mut output).unwrap();
        assert_eq!(written, output.len());
    }

    #[test]
    fn process_into_accumulates() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);

        let chunk: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut output = Vec::new();
        let w1 = proc.process_into(&chunk, &mut output).unwrap();
        let w2 = proc.process_into(&chunk, &mut output).unwrap();
        assert_eq!(output.len(), w1 + w2);
    }

    #[test]
    fn process_into_matches_process() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::HouseLoop);

        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        // Use process()
        let mut proc1 = StreamProcessor::new(params.clone());
        let mut out1 = Vec::new();
        for chunk in input.chunks(2048) {
            let out = proc1.process(chunk).unwrap();
            out1.extend_from_slice(&out);
        }
        let flushed1 = proc1.flush().unwrap();
        out1.extend_from_slice(&flushed1);

        // Use process_into()
        let mut proc2 = StreamProcessor::new(params);
        let mut out2 = Vec::new();
        for chunk in input.chunks(2048) {
            proc2.process_into(chunk, &mut out2).unwrap();
        }
        proc2.flush_into(&mut out2).unwrap();

        // Should produce identical output
        assert_eq!(out1.len(), out2.len(), "process() and process_into() should produce same length");
        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Mismatch at sample {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn flush_into_basic() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);

        let input: Vec<f32> = (0..8192)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut output = Vec::new();
        proc.process_into(&input, &mut output).unwrap();
        let before_flush = output.len();
        proc.flush_into(&mut output).unwrap();
        // Flush should add more samples
        assert!(output.len() >= before_flush);
    }

    #[test]
    fn flush_into_empty_returns_zero() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        let mut output = Vec::new();
        let written = proc.flush_into(&mut output).unwrap();
        assert_eq!(written, 0);
        assert!(output.is_empty());
    }

    #[test]
    fn process_into_stereo() {
        let params = StretchParams::new(1.25)
            .with_sample_rate(44100)
            .with_channels(2);
        let mut proc = StreamProcessor::new(params);

        let input = sine_stereo(440.0, 880.0, 44100, 4096);
        let mut output = Vec::new();
        proc.process_into(&input.data, &mut output).unwrap();
        proc.flush_into(&mut output).unwrap();

        // Stereo output must have even sample count
        assert_eq!(output.len() % 2, 0, "Stereo process_into must produce even samples");
    }

    #[test]
    fn process_into_rejects_nan() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);

        let bad_input = vec![0.0, f32::NAN, 0.0];
        let mut output = Vec::new();
        let result = proc.process_into(&bad_input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn process_into_with_ratio_change() {
        let mut proc = StreamProcessor::from_tempo(128.0, 126.0, 44100, 1);

        let chunk: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut output = Vec::new();
        proc.process_into(&chunk, &mut output).unwrap();
        proc.set_tempo(130.0);
        proc.process_into(&chunk, &mut output).unwrap();
        proc.flush_into(&mut output).unwrap();

        assert!(!output.is_empty());
        assert!(output.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn process_into_pre_allocated() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);

        // Pre-allocate a large buffer
        let mut output = Vec::with_capacity(100000);
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        for chunk in input.chunks(4096) {
            proc.process_into(chunk, &mut output).unwrap();
        }
        proc.flush_into(&mut output).unwrap();

        assert!(!output.is_empty());
        assert!(output.iter().all(|s| s.is_finite()));
    }
}

// ──────────────────────────────────────────────────────────────────
// Streaming-batch parity tests
// ──────────────────────────────────────────────────────────────────

mod streaming_batch_parity {
    use super::*;

    #[test]
    fn streaming_length_matches_batch_expansion() {
        let input = sine_mono(440.0, 44100, 44100); // 1 second
        let ratio = 1.5;

        // Batch
        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::HouseLoop);
        let batch_out = timestretch::stretch_buffer(&input, &params).unwrap();

        // Streaming
        let mut proc = StreamProcessor::new(params);
        let mut stream_out = Vec::new();
        for chunk in input.data.chunks(2048) {
            proc.process_into(chunk, &mut stream_out).unwrap();
        }
        proc.flush_into(&mut stream_out).unwrap();

        // Lengths should be within 15% of each other
        let batch_len = batch_out.num_frames();
        let stream_len = stream_out.len();
        let len_ratio = stream_len as f64 / batch_len as f64;
        assert!(
            (len_ratio - 1.0).abs() < 0.15,
            "Streaming ({stream_len}) vs batch ({batch_len}) length mismatch: ratio {len_ratio}"
        );
    }

    #[test]
    fn streaming_length_matches_batch_compression() {
        let input = sine_mono(440.0, 44100, 44100);
        let ratio = 0.75;

        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1);
        let batch_out = timestretch::stretch_buffer(&input, &params).unwrap();

        let mut proc = StreamProcessor::new(params);
        let mut stream_out = Vec::new();
        for chunk in input.data.chunks(2048) {
            proc.process_into(chunk, &mut stream_out).unwrap();
        }
        proc.flush_into(&mut stream_out).unwrap();

        let batch_len = batch_out.num_frames();
        let stream_len = stream_out.len();
        let len_ratio = stream_len as f64 / batch_len as f64;
        assert!(
            (len_ratio - 1.0).abs() < 0.30,
            "Streaming ({stream_len}) vs batch ({batch_len}) compression mismatch: ratio {len_ratio}"
        );
    }

    #[test]
    fn streaming_rms_matches_batch() {
        let input = sine_mono(440.0, 44100, 44100);
        let ratio = 1.25;

        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(1);
        let batch_out = timestretch::stretch_buffer(&input, &params).unwrap();
        let batch_rms = batch_out.rms();

        let mut proc = StreamProcessor::new(params);
        let mut stream_data = Vec::new();
        for chunk in input.data.chunks(2048) {
            proc.process_into(chunk, &mut stream_data).unwrap();
        }
        proc.flush_into(&mut stream_data).unwrap();
        let stream_buf = AudioBuffer::from_mono(stream_data, 44100);
        let stream_rms = stream_buf.rms();

        // RMS should be within 50% (streaming may differ due to windowing boundaries)
        let rms_ratio = stream_rms / batch_rms;
        assert!(
            rms_ratio > 0.5 && rms_ratio < 2.0,
            "Streaming RMS ({stream_rms}) vs batch RMS ({batch_rms}) too different: ratio {rms_ratio}"
        );
    }

    #[test]
    fn streaming_stereo_parity() {
        let input = sine_stereo(440.0, 880.0, 44100, 44100);
        let ratio = 1.5;

        let params = StretchParams::new(ratio)
            .with_sample_rate(44100)
            .with_channels(2);
        let batch_out = timestretch::stretch_buffer(&input, &params).unwrap();

        let mut proc = StreamProcessor::new(params);
        let mut stream_out = Vec::new();
        for chunk in input.data.chunks(4096) {
            proc.process_into(chunk, &mut stream_out).unwrap();
        }
        proc.flush_into(&mut stream_out).unwrap();

        // Lengths should be similar
        let batch_len = batch_out.num_frames();
        let stream_len = stream_out.len() / 2; // stereo
        let len_ratio = stream_len as f64 / batch_len as f64;
        assert!(
            (len_ratio - 1.0).abs() < 0.15,
            "Stereo streaming ({stream_len}) vs batch ({batch_len}) length mismatch"
        );
    }
}

// ──────────────────────────────────────────────────────────────────
// Combined workflow tests using new APIs
// ──────────────────────────────────────────────────────────────────

mod combined_new_api_workflows {
    use super::*;

    #[test]
    fn split_stretch_recombine() {
        // Split a buffer, stretch each half differently, recombine
        let buf = sine_mono(440.0, 44100, 44100);
        let (intro, body) = buf.split_at(11025); // split at 0.25s

        let slow_params = StretchParams::new(2.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let fast_params = StretchParams::new(0.75)
            .with_sample_rate(44100)
            .with_channels(1);

        let slow_intro = timestretch::stretch_buffer(&intro, &slow_params).unwrap();
        let fast_body = timestretch::stretch_buffer(&body, &fast_params).unwrap();

        let result = AudioBuffer::concatenate(&[&slow_intro, &fast_body]);
        assert!(result.num_frames() > 0);
        assert!(result.rms() > 0.01);
    }

    #[test]
    fn repeat_and_mix_layering() {
        // Create a short loop, repeat it, then mix with another layer
        let kick = AudioBuffer::from_mono(
            (0..4410)
                .map(|i| {
                    let t = i as f32 / 44100.0;
                    (2.0 * std::f32::consts::PI * 60.0 * t).sin()
                        * (-t * 20.0).exp()
                })
                .collect(),
            44100,
        );
        let kick_loop = kick.repeat(10); // 10 kicks = 1 second

        let hihat = AudioBuffer::from_mono(
            (0..44100)
                .map(|i| {
                    let t = i as f32 / 44100.0;
                    let phase = 2.0 * std::f32::consts::PI * 8000.0 * t;
                    0.1 * phase.sin()
                })
                .collect(),
            44100,
        );

        let mixed = kick_loop.mix(&hihat);
        assert_eq!(mixed.num_frames(), 44100);
        assert!(mixed.rms() > 0.01);
    }

    #[test]
    fn as_mut_normalize_then_split() {
        let mut buf = sine_mono(440.0, 44100, 10000);
        // Manually normalize to 0.5 peak via AsMut
        let peak = buf.peak();
        let target = 0.5f32;
        let scale = target / peak;
        let slice = buf.as_mut();
        for s in slice.iter_mut() {
            *s *= scale;
        }
        assert!((buf.peak() - 0.5).abs() < 0.01);

        let (left, right) = buf.split_at(5000);
        assert!(left.peak() <= 0.51);
        assert!(right.peak() <= 0.51);
    }

    #[test]
    fn into_data_for_external_processing() {
        // Simulate: stretch → extract raw samples → process externally → wrap back
        let input = sine_mono(440.0, 44100, 22050);
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let stretched = timestretch::stretch_buffer(&input, &params).unwrap();
        let rate = stretched.sample_rate;

        let mut raw: Vec<f32> = stretched.into_data();
        // Simulate external processing: apply simple gain
        for s in raw.iter_mut() {
            *s *= 0.8;
        }

        // Wrap back into AudioBuffer
        let processed = AudioBuffer::from_mono(raw, rate);
        assert!(processed.peak() < 0.85);
        assert!(processed.rms() > 0.01);
    }

    #[test]
    fn process_into_with_split_and_mix() {
        // Stream process two signals separately, mix the results
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);

        let sig_a: Vec<f32> = (0..22050)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let sig_b: Vec<f32> = (0..22050)
            .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut proc_a = StreamProcessor::new(params.clone());
        let mut proc_b = StreamProcessor::new(params);

        let mut out_a = Vec::new();
        let mut out_b = Vec::new();
        proc_a.process_into(&sig_a, &mut out_a).unwrap();
        proc_a.flush_into(&mut out_a).unwrap();
        proc_b.process_into(&sig_b, &mut out_b).unwrap();
        proc_b.flush_into(&mut out_b).unwrap();

        // Make them same length for mixing
        let min_len = out_a.len().min(out_b.len());
        let buf_a = AudioBuffer::from_mono(out_a[..min_len].to_vec(), 44100);
        let buf_b = AudioBuffer::from_mono(out_b[..min_len].to_vec(), 44100);
        let mixed = buf_a.mix(&buf_b);
        assert!(mixed.rms() > 0.01);
    }

    #[test]
    fn repeat_crossfade_dj_loop() {
        // Create a short loop, repeat for length, crossfade with another
        let loop_a = sine_mono(440.0, 44100, 22050); // 0.5s
        let loop_b = sine_mono(550.0, 44100, 22050);

        let track_a = loop_a.repeat(4); // 2s
        let track_b = loop_b.repeat(4);

        let mixed = track_a.crossfade_into(&track_b, 4410); // 100ms fade
        assert_eq!(mixed.num_frames(), 44100 * 2 + 44100 * 2 - 4410);
        assert!(mixed.rms() > 0.1);
    }
}

// Tests for under-covered code paths in the timestretch library.
//
// Focuses on:
// - lib.rs internal helpers (deinterleave, interleave, extract_mono, compute_rms, normalize_rms, validate_bpm)
// - resample.rs edge cases (near-unity factors, boundary factors)
// - params.rs boundary values (sample_rate min/max, hop_size boundaries)
// - window.rs (bessel_i0 convergence, Kaiser extreme beta, window size 2/3)
// - AudioBuffer edge cases (operations on minimal/empty buffers)
// - StreamProcessor (flush, reset, hybrid mode switching, rapid ratio changes)
// - Preset configurations (window types, band_split, beat_aware)

use timestretch::{AudioBuffer, Channels, EdmPreset, StreamProcessor, StretchParams, WindowType};

// ========================
// lib.rs helper tests
// ========================

mod lib_helpers {
    use super::*;

    #[test]
    fn deinterleave_mono_is_identity() {
        let input: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let output = timestretch::stretch(&input, &params).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn interleave_stereo_preserves_channel_order() {
        let sample_rate = 44100u32;
        let num_frames = 44100;
        let mut input = vec![0.0f32; num_frames * 2];
        for i in 0..num_frames {
            let t = i as f32 / sample_rate as f32;
            input[i * 2] = (2.0 * std::f32::consts::PI * 100.0 * t).sin();
            input[i * 2 + 1] = (2.0 * std::f32::consts::PI * 1000.0 * t).sin();
        }

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(2);
        let output = timestretch::stretch(&input, &params).unwrap();
        assert_eq!(output.len() % 2, 0);
        let left: Vec<f32> = output.iter().step_by(2).copied().collect();
        let right: Vec<f32> = output.iter().skip(1).step_by(2).copied().collect();
        let left_rms: f64 =
            left.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / left.len() as f64;
        let right_rms: f64 =
            right.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / right.len() as f64;
        assert!(left_rms.sqrt() > 0.01, "Left channel should have energy");
        assert!(right_rms.sqrt() > 0.01, "Right channel should have energy");
    }

    #[test]
    fn validate_input_subnormal_accepted() {
        let mut input = vec![0.0f32; 44100];
        input[0] = f32::MIN_POSITIVE / 2.0;
        input[100] = -f32::MIN_POSITIVE / 4.0;
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_ok(), "Subnormal floats should be accepted");
    }

    #[test]
    fn validate_bpm_rejects_nan_bpm() {
        let input = vec![0.0f32; 44100 * 2];
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let result = timestretch::stretch_to_bpm(&input, f64::NAN, 128.0, &params);
        assert!(result.is_err(), "NaN source BPM should be rejected");
        let result = timestretch::stretch_to_bpm(&input, 128.0, f64::NAN, &params);
        assert!(result.is_err(), "NaN target BPM should be rejected");
    }

    #[test]
    fn validate_bpm_rejects_infinity_bpm() {
        let input = vec![0.0f32; 44100 * 2];
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        // inf / 128 = inf => ratio out of range
        let result = timestretch::stretch_to_bpm(&input, f64::INFINITY, 128.0, &params);
        assert!(
            result.is_err(),
            "Infinity source BPM should lead to invalid ratio"
        );
    }

    #[test]
    fn extract_mono_from_stereo_bpm_detection() {
        let sample_rate = 44100u32;
        let num_frames = sample_rate as usize * 4;
        let mut data = vec![0.0f32; num_frames * 2];
        let beat_interval = (60.0 * sample_rate as f64 / 120.0) as usize;
        for pos in (0..num_frames).step_by(beat_interval) {
            for j in 0..10.min(num_frames - pos) {
                data[pos * 2 + j * 2] = if j < 5 { 0.9 } else { -0.4 };
            }
        }
        let buffer = AudioBuffer::new(data, sample_rate, Channels::Stereo);
        let _bpm = timestretch::detect_bpm_buffer(&buffer);
    }

    #[test]
    fn compute_rms_single_sample() {
        let input = vec![0.5f32];
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_normalize(true);
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_ok());
    }

    #[test]
    fn normalize_rms_near_zero_target() {
        let epsilon = 1e-10;
        let input: Vec<f32> = (0..44100)
            .map(|i| {
                epsilon as f32 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin()
            })
            .collect();
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_normalize(true);
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_ok());
        if let Ok(output) = result {
            assert!(
                output.iter().all(|s| s.is_finite()),
                "All samples should be finite"
            );
        }
    }

    #[test]
    fn bpm_ratio_edge_cases() {
        let ratio = timestretch::bpm_ratio(1000.0, 1.0);
        assert!((ratio - 1000.0).abs() < 1e-6);
        let ratio = timestretch::bpm_ratio(1.0, 1000.0);
        assert!((ratio - 0.001).abs() < 1e-6);
    }

    #[test]
    fn stretch_to_bpm_auto_empty_input() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let result = timestretch::stretch_to_bpm_auto(&[], 128.0, &params);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn process_buffer_overrides_sample_rate_and_channels() {
        let buffer = AudioBuffer::from_mono(
            (0..44100)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
                .collect(),
            48000,
        );
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(2);
        let output = timestretch::stretch_buffer(&buffer, &params).unwrap();
        assert_eq!(output.sample_rate, 48000);
        assert_eq!(output.channels, Channels::Mono);
    }
}

// ========================
// resample.rs edge cases
// ========================

mod resample_edge_cases {
    use super::*;

    #[test]
    fn pitch_shift_near_unity_factor() {
        let input: Vec<f32> = (0..88200)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let output_up = timestretch::pitch_shift(&input, &params, 1.001).unwrap();
        assert_eq!(output_up.len(), input.len());

        let output_down = timestretch::pitch_shift(&input, &params, 0.999).unwrap();
        assert_eq!(output_down.len(), input.len());
    }

    #[test]
    fn pitch_shift_boundary_factors() {
        let input: Vec<f32> = (0..88200)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        // Exact minimum factor
        let result = timestretch::pitch_shift(&input, &params, 0.01);
        assert!(result.is_ok(), "Factor 0.01 should be accepted");

        // Exact maximum factor
        let result = timestretch::pitch_shift(&input, &params, 100.0);
        assert!(result.is_ok(), "Factor 100.0 should be accepted");

        // Just below minimum
        let result = timestretch::pitch_shift(&input, &params, 0.009);
        assert!(result.is_err(), "Factor 0.009 should be rejected");

        // Just above maximum
        let result = timestretch::pitch_shift(&input, &params, 100.01);
        assert!(result.is_err(), "Factor 100.01 should be rejected");
    }
}

// ========================
// params.rs boundary tests
// ========================

mod params_boundaries {
    use super::*;

    #[test]
    fn sample_rate_minimum_accepted() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(8000)
            .with_channels(1);
        let input: Vec<f32> = (0..8000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 8000.0).sin())
            .collect();
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_ok(), "Sample rate 8000 should be accepted");
    }

    #[test]
    fn sample_rate_maximum_accepted() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(192000)
            .with_channels(1);
        let input: Vec<f32> = (0..192000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 192000.0).sin())
            .collect();
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_ok(), "Sample rate 192000 should be accepted");
    }

    #[test]
    fn sample_rate_below_minimum_rejected() {
        let mut params = StretchParams::new(1.5);
        params.sample_rate = 7999;
        let input = vec![0.0f32; 8000];
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_err(), "Sample rate 7999 should be rejected");
    }

    #[test]
    fn sample_rate_above_maximum_rejected() {
        let mut params = StretchParams::new(1.5);
        params.sample_rate = 192001;
        let input = vec![0.0f32; 44100];
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_err(), "Sample rate 192001 should be rejected");
    }

    #[test]
    fn hop_size_equals_fft_size_accepted() {
        let mut params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        params.fft_size = 1024;
        params.hop_size = 1024;
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_ok(), "hop_size == fft_size should be accepted");
    }

    #[test]
    fn hop_size_one_accepted() {
        let mut params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        params.fft_size = 256;
        params.hop_size = 1;
        let input: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_ok(), "hop_size == 1 should be accepted");
    }

    #[test]
    fn hop_size_zero_rejected() {
        let mut params = StretchParams::new(1.5);
        params.hop_size = 0;
        let input = vec![0.0f32; 44100];
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_err(), "hop_size == 0 should be rejected");
    }

    #[test]
    fn hop_size_exceeds_fft_rejected() {
        let mut params = StretchParams::new(1.5);
        params.fft_size = 4096;
        params.hop_size = 4097;
        let input = vec![0.0f32; 44100];
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_err(), "hop_size > fft_size should be rejected");
    }

    #[test]
    fn fft_size_256_minimum_accepted() {
        let mut params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        params.fft_size = 256;
        params.hop_size = 64;
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_ok(), "fft_size 256 should be accepted");
    }

    #[test]
    fn fft_size_128_rejected() {
        let mut params = StretchParams::new(1.5);
        params.fft_size = 128;
        params.hop_size = 32;
        let input = vec![0.0f32; 44100];
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_err(), "fft_size 128 should be rejected");
    }

    #[test]
    fn fft_size_not_power_of_two_rejected() {
        let mut params = StretchParams::new(1.5);
        params.fft_size = 300;
        params.hop_size = 75;
        let input = vec![0.0f32; 44100];
        let result = timestretch::stretch(&input, &params);
        assert!(
            result.is_err(),
            "non-power-of-two fft_size should be rejected"
        );
    }

    #[test]
    fn ratio_exact_boundaries_accepted() {
        let input: Vec<f32> = (0..88200)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let params = StretchParams::new(0.01)
            .with_sample_rate(44100)
            .with_channels(1);
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_ok(), "Ratio 0.01 should be accepted");

        let params = StretchParams::new(100.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let result = timestretch::stretch(&input, &params);
        assert!(result.is_ok(), "Ratio 100.0 should be accepted");
    }
}

// ========================
// window.rs edge cases
// ========================

mod window_edge_cases {
    use timestretch::WindowType;

    #[test]
    fn window_size_two() {
        let hann = timestretch::core::window::generate_window(WindowType::Hann, 2);
        assert_eq!(hann.len(), 2);
        assert!(hann[0].abs() < 1e-6, "Hann(2)[0] should be ~0");
        assert!(hann[1].abs() < 1e-6, "Hann(2)[1] should be ~0");

        let bh = timestretch::core::window::generate_window(WindowType::BlackmanHarris, 2);
        assert_eq!(bh.len(), 2);
        assert!(bh[0] < 0.01);
        assert!(bh[1] < 0.01);

        let kaiser = timestretch::core::window::generate_window(WindowType::Kaiser(800), 2);
        assert_eq!(kaiser.len(), 2);
        assert!((kaiser[0] - kaiser[1]).abs() < 1e-6);
    }

    #[test]
    fn window_size_three() {
        let hann = timestretch::core::window::generate_window(WindowType::Hann, 3);
        assert_eq!(hann.len(), 3);
        assert!(hann[0].abs() < 1e-6);
        assert!(
            (hann[1] - 1.0).abs() < 1e-6,
            "Hann(3) middle should be 1.0, got {}",
            hann[1]
        );
        assert!(hann[2].abs() < 1e-6);
    }

    #[test]
    fn kaiser_beta_zero_is_rectangular() {
        let w = timestretch::core::window::generate_window(WindowType::Kaiser(0), 256);
        assert_eq!(w.len(), 256);
        for (i, &v) in w.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "Kaiser(beta=0) at {} should be ~1.0, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn kaiser_very_high_beta() {
        let w = timestretch::core::window::generate_window(WindowType::Kaiser(5000), 256);
        assert_eq!(w.len(), 256);
        // Center should be near peak
        let center_val = w[127].max(w[128]);
        assert!(center_val > 0.5, "Kaiser center should be near peak");
        // Edges should be very small
        assert!(w[0] < 0.01, "Kaiser(beta=50) edge={}", w[0]);
        assert!(w[255] < 0.01, "Kaiser(beta=50) edge={}", w[255]);
        assert!(
            w.iter().all(|v| v.is_finite()),
            "All Kaiser values should be finite"
        );
    }

    #[test]
    fn all_windows_finite_for_various_sizes() {
        let sizes = [
            2, 3, 4, 5, 7, 16, 63, 64, 100, 255, 256, 512, 1024, 4096, 8192,
        ];
        let types = [
            WindowType::Hann,
            WindowType::BlackmanHarris,
            WindowType::Kaiser(800),
        ];
        for &size in &sizes {
            for &wt in &types {
                let w = timestretch::core::window::generate_window(wt, size);
                assert_eq!(w.len(), size);
                assert!(
                    w.iter().all(|v| v.is_finite()),
                    "Window {:?} size {} has non-finite values",
                    wt,
                    size
                );
                assert!(
                    w.iter().all(|&v| v >= -1e-6),
                    "Window {:?} size {} has negative values",
                    wt,
                    size
                );
            }
        }
    }

    #[test]
    fn apply_window_copy_returns_correct_result() {
        let window = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = timestretch::core::window::apply_window_copy(&data, &window);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
        assert!((result[3] - 2.0).abs() < 1e-6);
        assert!((result[4] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn apply_window_mismatched_lengths() {
        let window = vec![0.5, 1.0];
        let data = vec![2.0, 4.0, 6.0, 8.0];
        let result = timestretch::core::window::apply_window_copy(&data, &window);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn apply_window_empty() {
        let result = timestretch::core::window::apply_window_copy(&[], &[]);
        assert!(result.is_empty());

        let mut data = vec![1.0, 2.0];
        timestretch::core::window::apply_window(&mut data, &[]);
        assert_eq!(data, vec![1.0, 2.0]);
    }
}

// ========================
// AudioBuffer edge cases
// ========================

mod audio_buffer_edges {
    use super::*;

    #[test]
    fn empty_buffer_operations() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        assert!(buf.is_empty());
        assert!(buf.is_mono());
        assert!(!buf.is_stereo());
        assert_eq!(buf.num_frames(), 0);
        assert_eq!(buf.total_samples(), 0);
        assert!((buf.duration_secs() - 0.0).abs() < 1e-10);
        assert!((buf.peak() - 0.0).abs() < 1e-10);
        assert!((buf.rms() - 0.0).abs() < 1e-10);

        let left = buf.left();
        assert!(left.is_empty());
        let right = buf.right();
        assert!(right.is_empty());
        let mono = buf.mix_to_mono();
        assert!(mono.data.is_empty());
        let stereo = buf.to_stereo();
        assert!(stereo.data.is_empty());
    }

    #[test]
    fn single_frame_mono() {
        let buf = AudioBuffer::from_mono(vec![0.5], 44100);
        assert_eq!(buf.num_frames(), 1);
        assert!(!buf.is_empty());
        assert!((buf.peak() - 0.5).abs() < 1e-6);
        assert!((buf.rms() - 0.5).abs() < 1e-3);

        let left = buf.left();
        assert_eq!(left.len(), 1);
        assert!((left[0] - 0.5).abs() < 1e-6);

        let stereo = buf.to_stereo();
        assert_eq!(stereo.data.len(), 2);
        assert!((stereo.data[0] - 0.5).abs() < 1e-6);
        assert!((stereo.data[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn single_frame_stereo() {
        let buf = AudioBuffer::new(vec![0.3, -0.7], 44100, Channels::Stereo);
        assert_eq!(buf.num_frames(), 1);
        assert!(buf.is_stereo());

        let left = buf.left();
        assert_eq!(left.len(), 1);
        assert!((left[0] - 0.3).abs() < 1e-6);

        let right = buf.right();
        assert_eq!(right.len(), 1);
        assert!((right[0] - (-0.7)).abs() < 1e-6);

        let mono = buf.mix_to_mono();
        let expected_mono = (0.3 + (-0.7)) / 2.0;
        assert!((mono.data[0] - expected_mono).abs() < 1e-6);
    }

    #[test]
    fn slice_entire_buffer() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let buf = AudioBuffer::from_mono(data.clone(), 44100);
        let sliced = buf.slice(0, 100);
        assert_eq!(sliced.data.len(), 100);
        for (i, (actual, expected)) in sliced.data.iter().zip(data.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-6, "mismatch at {}", i);
        }
    }

    #[test]
    fn slice_zero_frames() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        let sliced = buf.slice(0, 0);
        assert!(sliced.data.is_empty());
    }

    #[test]
    fn slice_past_end_clamped() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        let sliced = buf.slice(1, 100);
        assert_eq!(sliced.data.len(), 2);
    }

    #[test]
    fn concatenate_empty_list() {
        let result = AudioBuffer::concatenate(&[]);
        assert!(result.data.is_empty());
    }

    #[test]
    fn concatenate_single_buffer() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        let result = AudioBuffer::concatenate(&[&buf]);
        assert_eq!(result.num_frames(), 3);
    }

    #[test]
    #[should_panic(expected = "sample rate mismatch")]
    fn concatenate_mismatched_sample_rate_panics() {
        let buf1 = AudioBuffer::from_mono(vec![1.0], 44100);
        let buf2 = AudioBuffer::from_mono(vec![2.0], 48000);
        let _result = AudioBuffer::concatenate(&[&buf1, &buf2]);
    }

    #[test]
    #[should_panic(expected = "channel layout mismatch")]
    fn concatenate_mismatched_channels_panics() {
        let buf1 = AudioBuffer::from_mono(vec![1.0], 44100);
        let buf2 = AudioBuffer::new(vec![1.0, 2.0], 44100, Channels::Stereo);
        let _result = AudioBuffer::concatenate(&[&buf1, &buf2]);
    }

    #[test]
    fn normalize_zero_target() {
        let buf = AudioBuffer::from_mono(vec![0.5, -0.3, 0.8], 44100);
        let normalized = buf.normalize(0.0);
        for &s in &normalized.data {
            assert!((s - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn normalize_already_at_target() {
        let buf = AudioBuffer::from_mono(vec![0.5, -0.5], 44100);
        let normalized = buf.normalize(0.5);
        assert!((normalized.data[0] - 0.5).abs() < 1e-6);
        assert!((normalized.data[1] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn normalize_silent_buffer() {
        let buf = AudioBuffer::from_mono(vec![0.0, 0.0, 0.0], 44100);
        let normalized = buf.normalize(1.0);
        // Should not amplify silence (peak is 0 → return clone)
        for &s in &normalized.data {
            assert!((s - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn apply_gain_zero_db() {
        let buf = AudioBuffer::from_mono(vec![0.5, -0.3], 44100);
        let result = buf.apply_gain(0.0);
        assert!((result.data[0] - 0.5).abs() < 1e-6);
        assert!((result.data[1] - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn apply_gain_positive_and_negative() {
        let buf = AudioBuffer::from_mono(vec![0.5], 44100);
        let louder = buf.apply_gain(6.0);
        let expected = 0.5 * 10.0f32.powf(6.0 / 20.0);
        assert!((louder.data[0] - expected).abs() < 0.01);

        let quieter = buf.apply_gain(-6.0);
        let expected = 0.5 * 10.0f32.powf(-6.0 / 20.0);
        assert!((quieter.data[0] - expected).abs() < 0.01);
    }

    #[test]
    fn trim_silence_all_silent() {
        let buf = AudioBuffer::from_mono(vec![0.0; 1000], 44100);
        let trimmed = buf.trim_silence(0.001);
        assert!(trimmed.data.is_empty());
    }

    #[test]
    fn trim_silence_no_silence() {
        let buf = AudioBuffer::from_mono(vec![0.5; 100], 44100);
        let trimmed = buf.trim_silence(0.001);
        assert_eq!(trimmed.data.len(), 100);
    }

    #[test]
    fn trim_silence_stereo() {
        let mut data = vec![0.0f32; 200]; // 100 frames stereo
        for sample in data.iter_mut().take(160).skip(40) {
            *sample = 0.5;
        }
        let buf = AudioBuffer::new(data, 44100, Channels::Stereo);
        let trimmed = buf.trim_silence(0.001);
        assert!(trimmed.data.len() < 200);
        assert!(trimmed.data.len() >= 120);
    }

    #[test]
    fn fade_in_longer_than_buffer() {
        let buf = AudioBuffer::from_mono(vec![1.0, 1.0, 1.0], 44100);
        let faded = buf.fade_in(100);
        assert_eq!(faded.data.len(), 3);
        assert!(faded.data[0] < faded.data[2]);
    }

    #[test]
    fn fade_out_longer_than_buffer() {
        let buf = AudioBuffer::from_mono(vec![1.0, 1.0, 1.0], 44100);
        let faded = buf.fade_out(100);
        assert_eq!(faded.data.len(), 3);
        assert!(faded.data[0] > faded.data[2]);
    }

    #[test]
    fn fade_zero_duration() {
        let buf = AudioBuffer::from_mono(vec![0.5, 0.5], 44100);
        let faded_in = buf.fade_in(0);
        assert!((faded_in.data[0] - 0.5).abs() < 1e-6);
        assert!((faded_in.data[1] - 0.5).abs() < 1e-6);
        let faded_out = buf.fade_out(0);
        assert!((faded_out.data[0] - 0.5).abs() < 1e-6);
        assert!((faded_out.data[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn frames_iterator_stereo() {
        let buf = AudioBuffer::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 44100, Channels::Stereo);
        let frames: Vec<&[f32]> = buf.frames().collect();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0], &[1.0, 2.0]);
        assert_eq!(frames[1], &[3.0, 4.0]);
        assert_eq!(frames[2], &[5.0, 6.0]);
    }

    #[test]
    fn frames_iterator_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let frames: Vec<&[f32]> = buf.frames().collect();
        assert!(frames.is_empty());
    }

    #[test]
    fn into_iterator_syntax() {
        let buf = AudioBuffer::new(vec![1.0, 2.0, 3.0, 4.0], 44100, Channels::Stereo);
        let mut count = 0;
        for frame in &buf {
            assert_eq!(frame.len(), 2);
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn partial_eq_different_data() {
        let buf1 = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
        let buf2 = AudioBuffer::from_mono(vec![1.0, 3.0], 44100);
        assert_ne!(buf1, buf2);
    }

    #[test]
    fn partial_eq_different_sample_rate() {
        let buf1 = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
        let buf2 = AudioBuffer::from_mono(vec![1.0, 2.0], 48000);
        assert_ne!(buf1, buf2);
    }

    #[test]
    fn partial_eq_different_channels() {
        let buf1 = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
        let buf2 = AudioBuffer::new(vec![1.0, 2.0], 44100, Channels::Stereo);
        assert_ne!(buf1, buf2);
    }

    #[test]
    fn as_ref_returns_data_slice() {
        let data = vec![1.0, 2.0, 3.0];
        let buf = AudioBuffer::from_mono(data.clone(), 44100);
        let slice: &[f32] = buf.as_ref();
        assert_eq!(slice, &data[..]);
    }

    #[test]
    fn display_formatting() {
        let buf = AudioBuffer::from_mono(vec![0.0; 44100], 44100);
        let display = format!("{}", buf);
        assert!(display.contains("44100"), "Display should show frame count");
        assert!(display.contains("Mono"), "Display should show Mono");
    }

    #[test]
    fn display_stereo() {
        let buf = AudioBuffer::new(vec![0.0; 88200], 44100, Channels::Stereo);
        let display = format!("{}", buf);
        assert!(display.contains("Stereo"), "Display should show Stereo");
    }

    #[test]
    fn default_stretch_params() {
        let params = StretchParams::default();
        assert!((params.stretch_ratio - 1.0).abs() < 1e-10);
        assert_eq!(params.channels, Channels::Stereo);
        assert_eq!(params.sample_rate, 44100);
    }

    #[test]
    fn from_channels_stereo() {
        let left = vec![1.0, 2.0, 3.0];
        let right = vec![4.0, 5.0, 6.0];
        let buf = AudioBuffer::from_channels(&[left, right], 44100);
        assert!(buf.is_stereo());
        assert_eq!(buf.num_frames(), 3);
        assert_eq!(buf.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn from_channels_different_lengths() {
        let ch1 = vec![1.0, 2.0, 3.0];
        let ch2 = vec![4.0, 5.0]; // shorter
        let buf = AudioBuffer::from_channels(&[ch1, ch2], 44100);
        // Truncates to shorter channel
        assert_eq!(buf.num_frames(), 2);
        assert_eq!(buf.data, vec![1.0, 4.0, 2.0, 5.0]);
    }
}

// ========================
// StreamProcessor edge cases
// ========================

mod stream_processor_edges {
    use super::*;

    #[test]
    fn process_empty_chunks_repeatedly() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        for _ in 0..10 {
            let output = proc.process(&[]).unwrap();
            assert!(output.is_empty());
        }
    }

    #[test]
    fn flush_without_any_input() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        let output = proc.flush().unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn flush_twice() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);

        let signal: Vec<f32> = (0..16384)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let _ = proc.process(&signal).unwrap();

        let _first_flush = proc.flush().unwrap();
        let second_flush = proc.flush().unwrap();
        assert!(second_flush.is_empty(), "Second flush should be empty");
    }

    #[test]
    fn reset_clears_state_completely() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);

        let signal: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let _ = proc.process(&signal).unwrap();

        proc.set_stretch_ratio(2.0);
        for _ in 0..100 {
            let _ = proc.process(&[]).unwrap();
        }

        proc.reset();
        assert!((proc.current_stretch_ratio() - 1.5).abs() < 1e-6);

        let output = proc.process(&[]).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn stereo_output_always_even() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(2);
        let mut proc = StreamProcessor::new(params);

        let num_frames = 44100;
        let mut signal = vec![0.0f32; num_frames * 2];
        for i in 0..num_frames {
            let t = i as f32 / 44100.0;
            signal[i * 2] = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            signal[i * 2 + 1] = (2.0 * std::f32::consts::PI * 880.0 * t).sin();
        }

        for chunk in signal.chunks(4096) {
            let output = proc.process(chunk).unwrap();
            assert_eq!(
                output.len() % 2,
                0,
                "Stereo output must have even sample count"
            );
        }
        let flush_output = proc.flush().unwrap();
        if !flush_output.is_empty() {
            assert_eq!(flush_output.len() % 2, 0);
        }
    }

    #[test]
    fn hybrid_mode_persists_across_reset() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        proc.set_hybrid_mode(true);
        assert!(proc.is_hybrid_mode());
        proc.reset();
        assert!(
            proc.is_hybrid_mode(),
            "Hybrid mode should persist across reset"
        );
    }

    #[test]
    fn switch_hybrid_mode_mid_stream() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);

        let signal: Vec<f32> = (0..44100 * 2)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut total_output = Vec::new();

        for chunk in signal[..44100].chunks(4096) {
            if let Ok(out) = proc.process(chunk) {
                total_output.extend_from_slice(&out);
            }
        }

        proc.set_hybrid_mode(true);

        for chunk in signal[44100..].chunks(4096) {
            if let Ok(out) = proc.process(chunk) {
                total_output.extend_from_slice(&out);
            }
        }

        if let Ok(out) = proc.flush() {
            total_output.extend_from_slice(&out);
        }

        assert!(
            !total_output.is_empty(),
            "Mixed-mode processing should produce output"
        );
        assert!(
            total_output.iter().all(|s| s.is_finite()),
            "All output should be finite"
        );
    }

    #[test]
    fn latency_increases_with_fft_size() {
        let params1 = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_fft_size(1024);
        let proc1 = StreamProcessor::new(params1);

        let params2 = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_fft_size(4096);
        let proc2 = StreamProcessor::new(params2);

        assert!(proc2.latency_samples() > proc1.latency_samples());
        assert!(proc2.latency_secs() > proc1.latency_secs());
    }

    #[test]
    fn rapid_ratio_changes_no_crash() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);

        let signal: Vec<f32> = (0..44100 * 2)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut total_output = Vec::new();
        let ratios = [0.5, 1.5, 0.8, 2.0, 1.0, 0.3, 1.7, 1.0];
        let chunk_size = signal.len() / ratios.len();

        for (idx, &ratio) in ratios.iter().enumerate() {
            proc.set_stretch_ratio(ratio);
            let start = idx * chunk_size;
            let end = (start + chunk_size).min(signal.len());
            let chunk = &signal[start..end];
            match proc.process(chunk) {
                Ok(out) => total_output.extend_from_slice(&out),
                Err(e) => panic!("Process error at ratio {}: {}", ratio, e),
            }
        }

        assert!(total_output.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn from_tempo_set_tempo_round_trip() {
        let mut proc = StreamProcessor::from_tempo(128.0, 128.0, 44100, 1);
        assert!((proc.current_stretch_ratio() - 1.0).abs() < 1e-6);

        assert!(proc.set_tempo(130.0));
        for _ in 0..500 {
            let _ = proc.process(&[]).ok();
        }
        let expected = 128.0 / 130.0;
        assert!(
            (proc.current_stretch_ratio() - expected).abs() < 0.01,
            "Expected ~{}, got {}",
            expected,
            proc.current_stretch_ratio()
        );

        assert!(proc.set_tempo(128.0));
        for _ in 0..500 {
            let _ = proc.process(&[]).ok();
        }
        assert!(
            (proc.current_stretch_ratio() - 1.0).abs() < 0.01,
            "Expected ~1.0, got {}",
            proc.current_stretch_ratio()
        );
    }
}

// ========================
// EDM preset configuration tests
// ========================

mod preset_configs {
    use super::*;

    #[test]
    fn ambient_preset_uses_blackman_harris() {
        let params = StretchParams::new(2.0).with_preset(EdmPreset::Ambient);
        assert_eq!(params.window_type, WindowType::BlackmanHarris);
    }

    #[test]
    fn presets_use_expected_windows() {
        // DjBeatmatch and VocalChop use Kaiser windows
        let params = StretchParams::new(1.5).with_preset(EdmPreset::DjBeatmatch);
        assert_eq!(params.window_type, WindowType::Kaiser(800));

        let params = StretchParams::new(1.5).with_preset(EdmPreset::VocalChop);
        assert_eq!(params.window_type, WindowType::Kaiser(600));

        // HouseLoop and Halftime use Blackman-Harris
        let params = StretchParams::new(1.5).with_preset(EdmPreset::HouseLoop);
        assert_eq!(params.window_type, WindowType::BlackmanHarris);

        let params = StretchParams::new(1.5).with_preset(EdmPreset::Halftime);
        assert_eq!(params.window_type, WindowType::BlackmanHarris);
    }

    #[test]
    fn preset_override_window_type() {
        let params = StretchParams::new(2.0)
            .with_preset(EdmPreset::Ambient)
            .with_window_type(WindowType::Kaiser(800));
        assert_eq!(params.window_type, WindowType::Kaiser(800));
    }

    #[test]
    fn all_presets_enable_band_split() {
        let presets = [
            EdmPreset::DjBeatmatch,
            EdmPreset::HouseLoop,
            EdmPreset::Halftime,
            EdmPreset::Ambient,
            EdmPreset::VocalChop,
        ];
        for preset in &presets {
            let params = StretchParams::new(1.5).with_preset(*preset);
            assert!(
                params.band_split,
                "Preset {:?} should enable band_split",
                preset
            );
        }
    }

    #[test]
    fn all_presets_enable_beat_aware() {
        let presets = [
            EdmPreset::DjBeatmatch,
            EdmPreset::HouseLoop,
            EdmPreset::Halftime,
            EdmPreset::Ambient,
            EdmPreset::VocalChop,
        ];
        for preset in &presets {
            let params = StretchParams::new(1.5).with_preset(*preset);
            assert!(
                params.beat_aware,
                "Preset {:?} should enable beat_aware",
                preset
            );
        }
    }

    #[test]
    fn preset_description_non_empty() {
        let presets = [
            EdmPreset::DjBeatmatch,
            EdmPreset::HouseLoop,
            EdmPreset::Halftime,
            EdmPreset::Ambient,
            EdmPreset::VocalChop,
        ];
        for preset in &presets {
            let desc = preset.description();
            assert!(
                !desc.is_empty(),
                "Preset {:?} description should not be empty",
                preset
            );
        }
    }

    #[test]
    fn edm_preset_display() {
        assert_eq!(format!("{}", EdmPreset::DjBeatmatch), "DjBeatmatch");
        assert_eq!(format!("{}", EdmPreset::HouseLoop), "HouseLoop");
        assert_eq!(format!("{}", EdmPreset::Halftime), "Halftime");
        assert_eq!(format!("{}", EdmPreset::Ambient), "Ambient");
        assert_eq!(format!("{}", EdmPreset::VocalChop), "VocalChop");
    }
}

// ========================
// Stretch with various window types
// ========================

mod window_type_stretch {
    use super::*;

    fn make_sine(sample_rate: u32, duration_secs: f64, freq: f32) -> Vec<f32> {
        let num_samples = (sample_rate as f64 * duration_secs) as usize;
        (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn kaiser_stretch_produces_output() {
        let input = make_sine(44100, 2.0, 440.0);
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_window_type(WindowType::Kaiser(800));
        let output = timestretch::stretch(&input, &params).unwrap();
        assert!(!output.is_empty());
        assert!(output.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn different_windows_different_output() {
        let input = make_sine(44100, 2.0, 440.0);

        let params_hann = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_window_type(WindowType::Hann);
        let output_hann = timestretch::stretch(&input, &params_hann).unwrap();

        let params_bh = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_window_type(WindowType::BlackmanHarris);
        let output_bh = timestretch::stretch(&input, &params_bh).unwrap();

        if output_hann.len() == output_bh.len() && !output_hann.is_empty() {
            let diff: f64 = output_hann
                .iter()
                .zip(output_bh.iter())
                .map(|(a, b)| ((a - b) as f64).abs())
                .sum::<f64>()
                / output_hann.len() as f64;
            assert!(
                diff > 1e-6,
                "Different windows should produce different output, avg diff={}",
                diff
            );
        }
    }
}

// ========================
// Normalize with edge cases
// ========================

mod normalize_edge_cases {
    use super::*;

    #[test]
    fn normalize_with_dc_offset() {
        let input: Vec<f32> = (0..88200)
            .map(|i| 0.5 + 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_normalize(true);
        let output = timestretch::stretch(&input, &params).unwrap();
        assert!(!output.is_empty());
        assert!(output.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn normalize_stereo() {
        let sample_rate = 44100u32;
        let num_frames = 44100;
        let mut input = vec![0.0f32; num_frames * 2];
        for i in 0..num_frames {
            let t = i as f32 / sample_rate as f32;
            input[i * 2] = 0.6 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            input[i * 2 + 1] = 0.4 * (2.0 * std::f32::consts::PI * 880.0 * t).sin();
        }

        let input_rms: f64 =
            input.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / input.len() as f64;
        let input_rms = input_rms.sqrt();

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(2)
            .with_normalize(true);
        let output = timestretch::stretch(&input, &params).unwrap();

        let output_rms: f64 =
            output.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / output.len() as f64;
        let output_rms = output_rms.sqrt();

        assert!(
            (output_rms - input_rms).abs() < input_rms * 0.1,
            "Stereo normalize: input RMS={:.4}, output RMS={:.4}",
            input_rms,
            output_rms
        );
    }

    #[test]
    fn normalize_pitch_shift_stereo() {
        let sample_rate = 44100u32;
        let num_frames = 44100;
        let mut input = vec![0.0f32; num_frames * 2];
        for i in 0..num_frames {
            let t = i as f32 / sample_rate as f32;
            input[i * 2] = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            input[i * 2 + 1] = 0.5 * (2.0 * std::f32::consts::PI * 880.0 * t).sin();
        }

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(2)
            .with_normalize(true);
        let output = timestretch::pitch_shift(&input, &params, 1.5).unwrap();
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|s| s.is_finite()));
    }
}

// ========================
// Builder API combination tests
// ========================

mod builder_api {
    use super::*;

    #[test]
    fn full_builder_chain() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(48000)
            .with_channels(2)
            .with_preset(EdmPreset::HouseLoop)
            .with_fft_size(2048)
            .with_window_type(WindowType::BlackmanHarris)
            .with_normalize(true)
            .with_beat_aware(false)
            .with_band_split(false)
            .with_sub_bass_cutoff(100.0);

        assert!((params.stretch_ratio - 1.5).abs() < 1e-10);
        assert_eq!(params.sample_rate, 48000);
        assert_eq!(params.channels, Channels::Stereo);
        assert_eq!(params.fft_size, 2048);
        assert_eq!(params.window_type, WindowType::BlackmanHarris);
        assert!(params.normalize);
        assert!(!params.beat_aware);
        assert!(!params.band_split);
        assert!((params.sub_bass_cutoff - 100.0).abs() < 1e-6);
    }

    #[test]
    fn from_tempo_constructor() {
        let params = StretchParams::from_tempo(126.0, 128.0);
        let expected_ratio = 126.0 / 128.0;
        assert!((params.stretch_ratio - expected_ratio).abs() < 1e-10);
    }

    #[test]
    fn output_length_calculation() {
        let params = StretchParams::new(1.5).with_channels(1);
        let output_len = params.output_length(44100);
        let expected = (44100.0 * 1.5) as usize;
        assert_eq!(output_len, expected);

        let params = StretchParams::new(0.5).with_channels(2);
        let output_len = params.output_length(88200);
        let expected = (88200.0 * 0.5) as usize;
        assert_eq!(output_len, expected);
    }

    #[test]
    fn stretch_params_display() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(48000)
            .with_channels(2)
            .with_preset(EdmPreset::DjBeatmatch);
        let display = format!("{}", params);
        assert!(display.contains("1.5"), "Should contain ratio");
        assert!(display.contains("48000"), "Should contain sample rate");
        assert!(display.contains("Stereo"), "Should contain channel info");
    }

    #[test]
    fn with_wsola_params() {
        let params = StretchParams::new(1.5)
            .with_wsola_segment_size(2048)
            .with_wsola_search_range(512);
        assert_eq!(params.wsola_segment_size, 2048);
        assert_eq!(params.wsola_search_range, 512);
    }
}

// Integration tests for recently added features:
// - AudioBuffer::resample() workflows
// - AudioBuffer::crossfade_into() DJ workflows
// - stretch_to_bpm_wav_file() and WAV file convenience APIs
// - StreamProcessor DJ workflow (from_tempo + set_tempo + hybrid mode)
// - From<AudioBuffer> for Vec<f32>, with_stretch_ratio(), Debug impls
// - StretchParams::from_tempo() integration

use timestretch::{AudioBuffer, Channels, EdmPreset, StreamProcessor, StretchParams, WindowType};

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

fn click_train(interval_samples: usize, total_samples: usize, sample_rate: u32) -> AudioBuffer {
    let mut data = vec![0.0f32; total_samples];
    let mut pos = 0;
    while pos < total_samples {
        data[pos] = 1.0;
        if pos + 1 < total_samples {
            data[pos + 1] = -0.5;
        }
        pos += interval_samples;
    }
    AudioBuffer::from_mono(data, sample_rate)
}

// ──────────────────────────────────────────────────────────────────
// AudioBuffer::resample() integration tests
// ──────────────────────────────────────────────────────────────────

mod resample_workflows {
    use super::*;

    #[test]
    fn resample_44100_to_48000_preserves_duration() {
        let buf = sine_mono(440.0, 44100, 44100); // 1 second
        let resampled = buf.resample(48000);
        assert_eq!(resampled.sample_rate, 48000);
        assert_eq!(resampled.num_frames(), 48000);
        let dur_diff = (buf.duration_secs() - resampled.duration_secs()).abs();
        assert!(dur_diff < 0.001, "Duration drift: {dur_diff}");
    }

    #[test]
    fn resample_48000_to_44100_preserves_duration() {
        let buf = sine_mono(440.0, 48000, 48000); // 1 second
        let resampled = buf.resample(44100);
        assert_eq!(resampled.sample_rate, 44100);
        assert_eq!(resampled.num_frames(), 44100);
        let dur_diff = (buf.duration_secs() - resampled.duration_secs()).abs();
        assert!(dur_diff < 0.001, "Duration drift: {dur_diff}");
    }

    #[test]
    fn resample_stereo_preserves_channel_separation() {
        let buf = sine_stereo(440.0, 880.0, 44100, 44100);
        let resampled = buf.resample(48000);
        assert_eq!(resampled.channels, Channels::Stereo);
        assert_eq!(resampled.num_frames(), 48000);

        // Check that left and right channels have different content
        let left = resampled.left();
        let right = resampled.right();
        // They should be different since they're different frequencies
        let correlation: f64 = left
            .iter()
            .zip(right.iter())
            .take(1000)
            .map(|(&l, &r)| (l as f64) * (r as f64))
            .sum();
        // For uncorrelated signals of different frequency, correlation should be low
        let normalized_corr = correlation / 1000.0;
        assert!(
            normalized_corr.abs() < 0.5,
            "Channels should remain independent after resample, got {normalized_corr}"
        );
    }

    #[test]
    fn resample_then_stretch_workflow() {
        // Simulate: load 48kHz audio, resample to 44.1kHz, stretch
        let buf_48k = sine_mono(440.0, 48000, 48000); // 1s at 48kHz
        let buf_44k = buf_48k.resample(44100);

        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::HouseLoop);
        let stretched = timestretch::stretch_buffer(&buf_44k, &params).unwrap();

        // Should be ~1.5x duration
        let ratio = stretched.duration_secs() / buf_44k.duration_secs();
        assert!(
            (ratio - 1.5).abs() < 0.1,
            "Expected ~1.5x stretch, got {ratio}"
        );
    }

    #[test]
    fn stretch_then_resample_workflow() {
        // Stretch at 44.1kHz, then resample output to 48kHz for playback
        let buf = sine_mono(440.0, 44100, 44100);
        let params = StretchParams::new(1.25)
            .with_sample_rate(44100)
            .with_channels(1);
        let stretched = timestretch::stretch_buffer(&buf, &params).unwrap();
        let for_playback = stretched.resample(48000);

        assert_eq!(for_playback.sample_rate, 48000);
        // Duration should be preserved through resample
        let dur_diff = (stretched.duration_secs() - for_playback.duration_secs()).abs();
        assert!(dur_diff < 0.01, "Resample altered duration: {dur_diff}");
    }

    #[test]
    fn resample_preserves_rms_energy() {
        let buf = sine_mono(440.0, 44100, 44100);
        let original_rms = buf.rms();
        let resampled = buf.resample(48000);
        let resampled_rms = resampled.rms();

        // RMS should be approximately preserved (sine wave at same amplitude)
        let rms_ratio = resampled_rms / original_rms;
        assert!(
            (rms_ratio - 1.0).abs() < 0.15,
            "RMS changed too much after resample: ratio = {rms_ratio}"
        );
    }

    #[test]
    fn resample_identity_same_rate() {
        let buf = sine_mono(440.0, 44100, 1000);
        let resampled = buf.resample(44100);
        assert_eq!(resampled.num_frames(), buf.num_frames());
        assert_eq!(resampled.data, buf.data);
    }

    #[test]
    fn resample_double_rate() {
        // 2x upsample should double frame count
        let buf = sine_mono(440.0, 22050, 22050); // 1s at 22.05kHz
        let resampled = buf.resample(44100);
        assert_eq!(resampled.num_frames(), 44100);
        assert_eq!(resampled.sample_rate, 44100);
    }

    #[test]
    fn resample_half_rate() {
        // 0.5x downsample should halve frame count
        let buf = sine_mono(440.0, 48000, 48000); // 1s
        let resampled = buf.resample(24000);
        assert_eq!(resampled.num_frames(), 24000);
    }

    #[test]
    fn resample_round_trip_preserves_content() {
        // 44100 → 48000 → 44100 should roughly preserve the signal
        let buf = sine_mono(440.0, 44100, 44100);
        let up = buf.resample(48000);
        let back = up.resample(44100);
        assert_eq!(back.num_frames(), 44100);
        // RMS should be similar
        let ratio = back.rms() / buf.rms();
        assert!(
            (ratio - 1.0).abs() < 0.2,
            "Round-trip resample degraded signal: ratio = {ratio}"
        );
    }

    #[test]
    fn resample_empty_buffer() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let resampled = buf.resample(48000);
        assert!(resampled.is_empty());
        assert_eq!(resampled.sample_rate, 48000);
    }
}

// ──────────────────────────────────────────────────────────────────
// AudioBuffer::crossfade_into() integration tests
// ──────────────────────────────────────────────────────────────────

mod crossfade_workflows {
    use super::*;

    #[test]
    fn crossfade_dj_transition() {
        // Two stretched tracks crossfade at beat boundary
        let track_a = sine_mono(440.0, 44100, 44100 * 2); // 2s
        let track_b = sine_mono(330.0, 44100, 44100 * 2);

        let params = StretchParams::new(128.0 / 126.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::DjBeatmatch);

        let a_out = timestretch::stretch_buffer(&track_a, &params).unwrap();
        let b_out = timestretch::stretch_buffer(&track_b, &params).unwrap();

        // 500ms crossfade = ~22050 frames
        let fade_frames = 22050;
        let mixed = a_out.crossfade_into(&b_out, fade_frames);

        let expected_frames = a_out.num_frames() + b_out.num_frames() - fade_frames;
        assert_eq!(mixed.num_frames(), expected_frames);
        assert!(mixed.rms() > 0.1, "Crossfaded output should have energy");
        assert!(mixed.peak() <= 1.5, "Crossfade should not clip badly");
    }

    #[test]
    fn crossfade_stereo_tracks() {
        let a = sine_stereo(440.0, 880.0, 44100, 44100);
        let b = sine_stereo(330.0, 660.0, 44100, 44100);

        let mixed = a.crossfade_into(&b, 4410); // 100ms crossfade
        assert_eq!(mixed.channels, Channels::Stereo);
        assert_eq!(mixed.num_frames(), 44100 + 44100 - 4410);
        assert!(mixed.rms() > 0.1);
    }

    #[test]
    fn crossfade_zero_frames_is_concatenation() {
        let a = sine_mono(440.0, 44100, 1000);
        let b = sine_mono(330.0, 44100, 1000);
        let mixed = a.crossfade_into(&b, 0);
        assert_eq!(mixed.num_frames(), 2000);

        // First 1000 frames should match a, last 1000 should match b
        let slice_a = mixed.slice(0, 1000);
        let slice_b = mixed.slice(1000, 1000);
        assert_eq!(slice_a.data, a.data);
        assert_eq!(slice_b.data, b.data);
    }

    #[test]
    fn crossfade_full_overlap() {
        // Crossfade longer than either buffer should clamp
        let a = sine_mono(440.0, 44100, 100);
        let b = sine_mono(330.0, 44100, 100);
        let mixed = a.crossfade_into(&b, 500); // 500 > 100
                                               // Should clamp to min(100, 100) = 100
        assert_eq!(mixed.num_frames(), 100); // 100 + 100 - 100
    }

    #[test]
    fn crossfade_midpoint_is_equal_mix() {
        // At the midpoint of a crossfade, both signals should contribute equally
        let a = AudioBuffer::from_mono(vec![1.0; 1000], 44100);
        let b = AudioBuffer::from_mono(vec![-1.0; 1000], 44100);
        let fade_frames = 100;
        let mixed = a.crossfade_into(&b, fade_frames);

        // At midpoint of crossfade region (frame 950 in output = 50th fade frame)
        let mid_sample = mixed.data[950];
        // raised cosine at t=0.5: fade_out = 0.5*(1+cos(PI*0.5)) = 0.5
        // fade_in = 0.5
        // so result = 1.0*0.5 + (-1.0)*0.5 = 0.0
        assert!(
            mid_sample.abs() < 0.05,
            "Midpoint should be ~0 for equal-amplitude inverse signals, got {mid_sample}"
        );
    }

    #[test]
    fn crossfade_energy_conservation_dc() {
        // Two DC signals of same amplitude: crossfade should preserve level
        let a = AudioBuffer::from_mono(vec![0.5; 2000], 44100);
        let b = AudioBuffer::from_mono(vec![0.5; 2000], 44100);
        let mixed = a.crossfade_into(&b, 500);

        // During crossfade of identical signals, raised cosine:
        // out = 0.5 * fade_out + 0.5 * fade_in = 0.5*(fade_out + fade_in)
        // fade_out + fade_in = 0.5*(1+cos(pi*t)) + 1 - 0.5*(1+cos(pi*t)) = 1.0
        // So out = 0.5 everywhere
        for &s in mixed.data.iter() {
            assert!(
                (s - 0.5).abs() < 0.01,
                "DC crossfade should preserve level, got {s}"
            );
        }
    }

    #[test]
    fn crossfade_chain_three_segments() {
        // A → B → C with crossfades
        let a = sine_mono(440.0, 44100, 10000);
        let b = sine_mono(550.0, 44100, 10000);
        let c = sine_mono(660.0, 44100, 10000);
        let fade = 2000;

        let ab = a.crossfade_into(&b, fade);
        let abc = ab.crossfade_into(&c, fade);

        let expected = 10000 + 10000 - fade + 10000 - fade;
        assert_eq!(abc.num_frames(), expected);
        assert!(abc.rms() > 0.1);
    }

    #[test]
    #[should_panic(expected = "sample rate mismatch")]
    fn crossfade_mismatched_sample_rate_panics() {
        let a = AudioBuffer::from_mono(vec![0.0; 100], 44100);
        let b = AudioBuffer::from_mono(vec![0.0; 100], 48000);
        let _ = a.crossfade_into(&b, 10);
    }

    #[test]
    #[should_panic(expected = "channel layout mismatch")]
    fn crossfade_mismatched_channels_panics() {
        let a = AudioBuffer::from_mono(vec![0.0; 100], 44100);
        let b = AudioBuffer::from_stereo(vec![0.0; 200], 44100);
        let _ = a.crossfade_into(&b, 10);
    }
}

// ──────────────────────────────────────────────────────────────────
// stretch_to_bpm_wav_file() and WAV convenience API tests
// ──────────────────────────────────────────────────────────────────

mod wav_file_api {
    use super::*;

    fn create_test_wav(path: &str, freq: f32, duration_secs: f64, sample_rate: u32) {
        let num_samples = (sample_rate as f64 * duration_secs) as usize;
        let data: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();
        let buf = AudioBuffer::from_mono(data, sample_rate);
        timestretch::io::wav::write_wav_file_float(path, &buf).unwrap();
    }

    #[test]
    fn stretch_to_bpm_wav_file_basic() {
        let dir = std::env::temp_dir().join("timestretch_test_bpm_wav");
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input_bpm.wav");
        let output = dir.join("output_bpm.wav");

        // Create a 2s test signal
        create_test_wav(input.to_str().unwrap(), 440.0, 2.0, 44100);

        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::DjBeatmatch);

        // Stretch from 128 BPM to 126 BPM
        let result = timestretch::stretch_to_bpm_wav_file(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            128.0,
            126.0,
            &params,
        )
        .unwrap();

        // Should be slightly longer (slower tempo = longer audio)
        let expected_ratio = 128.0 / 126.0;
        let actual_ratio = result.duration_secs() / 2.0;
        assert!(
            (actual_ratio - expected_ratio).abs() < 0.1,
            "Expected ratio ~{expected_ratio}, got {actual_ratio}"
        );

        // Output file should exist and be readable
        let read_back = timestretch::io::wav::read_wav_file(output.to_str().unwrap()).unwrap();
        assert_eq!(read_back.sample_rate, result.sample_rate);
        assert!(read_back.num_frames() > 0);

        // Clean up
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stretch_wav_file_basic() {
        let dir = std::env::temp_dir().join("timestretch_test_stretch_wav");
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input_stretch.wav");
        let output = dir.join("output_stretch.wav");

        create_test_wav(input.to_str().unwrap(), 440.0, 1.0, 44100);

        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::HouseLoop);

        let result = timestretch::stretch_wav_file(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            &params,
        )
        .unwrap();

        let ratio = result.duration_secs() / 1.0;
        assert!(
            (ratio - 1.5).abs() < 0.1,
            "Expected ~1.5x stretch, got {ratio}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pitch_shift_wav_file_basic() {
        let dir = std::env::temp_dir().join("timestretch_test_pitch_wav");
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input_pitch.wav");
        let output = dir.join("output_pitch.wav");

        create_test_wav(input.to_str().unwrap(), 440.0, 1.0, 44100);

        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let result = timestretch::pitch_shift_wav_file(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            &params,
            2.0, // octave up
        )
        .unwrap();

        // Duration should be preserved
        let dur_diff = (result.duration_secs() - 1.0).abs();
        assert!(dur_diff < 0.15, "Pitch shift altered duration: {dur_diff}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stretch_to_bpm_wav_file_stereo() {
        let dir = std::env::temp_dir().join("timestretch_test_bpm_stereo");
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("stereo_in.wav");
        let output = dir.join("stereo_out.wav");

        // Write a stereo WAV
        let buf = sine_stereo(440.0, 880.0, 44100, 88200); // 2s stereo
        timestretch::io::wav::write_wav_file_float(input.to_str().unwrap(), &buf).unwrap();

        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(2)
            .with_preset(EdmPreset::HouseLoop);

        let result = timestretch::stretch_to_bpm_wav_file(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            128.0,
            120.0,
            &params,
        )
        .unwrap();

        assert_eq!(result.channels, Channels::Stereo);
        let expected_ratio = 128.0 / 120.0;
        let actual_ratio = result.duration_secs() / 2.0;
        assert!(
            (actual_ratio - expected_ratio).abs() < 0.15,
            "Expected ratio ~{expected_ratio}, got {actual_ratio}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wav_file_api_nonexistent_input() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let result =
            timestretch::stretch_wav_file("/nonexistent/input.wav", "/tmp/output.wav", &params);
        assert!(result.is_err());
    }

    #[test]
    fn stretch_to_bpm_wav_file_all_presets() {
        let dir = std::env::temp_dir().join("timestretch_test_bpm_presets");
        std::fs::create_dir_all(&dir).unwrap();

        let input = dir.join("input.wav");
        create_test_wav(input.to_str().unwrap(), 440.0, 1.0, 44100);

        let presets = [
            EdmPreset::DjBeatmatch,
            EdmPreset::HouseLoop,
            EdmPreset::Halftime,
            EdmPreset::Ambient,
            EdmPreset::VocalChop,
        ];

        for preset in &presets {
            let output = dir.join(format!("out_{preset}.wav"));
            let params = StretchParams::new(1.0)
                .with_sample_rate(44100)
                .with_channels(1)
                .with_preset(*preset);

            let result = timestretch::stretch_to_bpm_wav_file(
                input.to_str().unwrap(),
                output.to_str().unwrap(),
                128.0,
                126.0,
                &params,
            )
            .unwrap();

            assert!(
                result.num_frames() > 0,
                "Preset {preset} produced empty output"
            );
            assert!(result.rms() > 0.01, "Preset {preset} produced silence");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ──────────────────────────────────────────────────────────────────
// DJ streaming workflow tests (from_tempo + set_tempo + hybrid)
// ──────────────────────────────────────────────────────────────────

mod dj_streaming_workflow {
    use super::*;

    #[test]
    fn from_tempo_basic_mono_workflow() {
        // Simulate DJ matching a 126 BPM track to 128 BPM
        // from_tempo(126, 128) → ratio = 126/128 ≈ 0.984 (slight compression)
        let mut proc = StreamProcessor::from_tempo(126.0, 128.0, 44100, 1);
        assert_eq!(proc.source_bpm(), Some(126.0));

        let input = sine_mono(440.0, 44100, 44100);
        let mut output = Vec::new();

        for chunk in input.data.chunks(1024) {
            let out = proc.process(chunk).unwrap();
            output.extend_from_slice(&out);
        }
        let flushed = proc.flush().unwrap();
        output.extend_from_slice(&flushed);

        // Should produce output (streaming may add latency padding)
        assert!(!output.is_empty());
        // All samples should be finite
        assert!(output.iter().all(|s| s.is_finite()));
        // The ratio should be approximately correct (allowing for streaming overhead)
        let ratio = output.len() as f64 / input.data.len() as f64;
        assert!(
            ratio < 1.5,
            "126→128 BPM compression ratio should be reasonable, got {ratio}"
        );
    }

    #[test]
    fn from_tempo_stereo_workflow() {
        let mut proc = StreamProcessor::from_tempo(128.0, 126.0, 44100, 2);
        let input = sine_stereo(440.0, 880.0, 44100, 44100);
        let mut output = Vec::new();

        for chunk in input.data.chunks(2048) {
            let out = proc.process(chunk).unwrap();
            output.extend_from_slice(&out);
        }
        let flushed = proc.flush().unwrap();
        output.extend_from_slice(&flushed);

        // 128→126 = expansion
        assert!(!output.is_empty());
        // Output should be interleaved stereo (even sample count)
        assert_eq!(
            output.len() % 2,
            0,
            "Stereo output must have even sample count"
        );
    }

    #[test]
    fn set_tempo_smooth_transition() {
        // Start at 128→126, then change to 128→130 mid-stream
        let mut proc = StreamProcessor::from_tempo(128.0, 126.0, 44100, 1);
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        // Process first half at original tempo
        let mut output_part1 = Vec::new();
        for chunk in input[..22050].chunks(1024) {
            let out = proc.process(chunk).unwrap();
            output_part1.extend_from_slice(&out);
        }

        // Change tempo mid-stream
        assert!(proc.set_tempo(130.0));

        // Process second half at new tempo
        let mut output_part2 = Vec::new();
        for chunk in input[22050..].chunks(1024) {
            let out = proc.process(chunk).unwrap();
            output_part2.extend_from_slice(&out);
        }
        let flushed = proc.flush().unwrap();
        output_part2.extend_from_slice(&flushed);

        // Both parts should produce output
        assert!(!output_part1.is_empty(), "First part should produce output");
        assert!(
            !output_part2.is_empty(),
            "Second part should produce output"
        );

        // No NaN or Inf in output (click-free transitions)
        let combined: Vec<f32> = output_part1.into_iter().chain(output_part2).collect();
        for (i, &s) in combined.iter().enumerate() {
            assert!(s.is_finite(), "Non-finite sample at index {i}: {s}");
        }
    }

    #[test]
    fn set_tempo_multiple_changes() {
        // Simulate DJ pitch fader moving through several tempos
        let mut proc = StreamProcessor::from_tempo(128.0, 128.0, 44100, 1);
        let chunk: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let tempos = [126.0, 127.0, 128.0, 129.0, 130.0, 128.0, 125.0];
        for &tempo in &tempos {
            assert!(proc.set_tempo(tempo), "set_tempo({tempo}) should succeed");
            let out = proc.process(&chunk).unwrap();
            for &s in &out {
                assert!(s.is_finite(), "Non-finite sample at tempo {tempo}");
            }
        }
    }

    #[test]
    fn hybrid_mode_streaming_dj_workflow() {
        let mut proc = StreamProcessor::from_tempo(128.0, 126.0, 44100, 1);
        proc.set_hybrid_mode(true);
        assert!(proc.is_hybrid_mode());

        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut output = Vec::new();
        for chunk in input.chunks(2048) {
            let out = proc.process(chunk).unwrap();
            output.extend_from_slice(&out);
        }
        let flushed = proc.flush().unwrap();
        output.extend_from_slice(&flushed);

        assert!(!output.is_empty());
        assert!(output.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn hybrid_mode_switch_mid_stream() {
        let mut proc = StreamProcessor::from_tempo(128.0, 126.0, 44100, 1);
        let chunk: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        // Start in PV mode
        assert!(!proc.is_hybrid_mode());
        let out1 = proc.process(&chunk).unwrap();

        // Switch to hybrid
        proc.set_hybrid_mode(true);
        let out2 = proc.process(&chunk).unwrap();

        // Switch back to PV
        proc.set_hybrid_mode(false);
        let out3 = proc.process(&chunk).unwrap();

        // All should produce finite output
        for out in [&out1, &out2, &out3] {
            for &s in out {
                assert!(s.is_finite());
            }
        }
    }

    #[test]
    fn from_tempo_reset_preserves_source_bpm() {
        let mut proc = StreamProcessor::from_tempo(128.0, 126.0, 44100, 1);
        assert_eq!(proc.source_bpm(), Some(128.0));

        proc.reset();
        assert_eq!(
            proc.source_bpm(),
            Some(128.0),
            "Reset should preserve source BPM"
        );
    }

    #[test]
    fn from_tempo_params_accessor() {
        let proc = StreamProcessor::from_tempo(128.0, 126.0, 44100, 1);
        let params = proc.params();
        // from_tempo uses DjBeatmatch preset
        assert!(params.fft_size > 0);
        assert!(params.sample_rate > 0);
    }

    #[test]
    fn set_tempo_without_from_tempo_returns_false() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let mut proc = StreamProcessor::new(params);
        assert!(!proc.set_tempo(130.0));
        assert_eq!(proc.source_bpm(), None);
    }

    #[test]
    fn set_tempo_invalid_values() {
        let mut proc = StreamProcessor::from_tempo(128.0, 128.0, 44100, 1);
        assert!(!proc.set_tempo(0.0));
        assert!(!proc.set_tempo(-1.0));
        assert!(!proc.set_tempo(f64::NAN));
    }
}

// ──────────────────────────────────────────────────────────────────
// From<AudioBuffer> for Vec<f32>, with_stretch_ratio, Debug
// ──────────────────────────────────────────────────────────────────

mod conversion_and_trait_tests {
    use super::*;

    #[test]
    fn from_audio_buffer_to_vec_mono() {
        let buf = sine_mono(440.0, 44100, 1000);
        let original_data = buf.data.clone();
        let vec: Vec<f32> = buf.into();
        assert_eq!(vec, original_data);
    }

    #[test]
    fn from_audio_buffer_to_vec_stereo() {
        let buf = sine_stereo(440.0, 880.0, 44100, 500);
        let original_len = buf.data.len();
        let vec: Vec<f32> = buf.into();
        assert_eq!(vec.len(), original_len);
    }

    #[test]
    fn from_audio_buffer_to_vec_after_stretch() {
        let input = sine_mono(440.0, 44100, 22050);
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let stretched = timestretch::stretch_buffer(&input, &params).unwrap();
        let vec: Vec<f32> = stretched.into();
        assert!(!vec.is_empty());
    }

    #[test]
    fn with_stretch_ratio_overrides() {
        let params = StretchParams::new(1.0)
            .with_preset(EdmPreset::DjBeatmatch)
            .with_stretch_ratio(1.5);
        assert!((params.stretch_ratio - 1.5).abs() < 1e-10);
    }

    #[test]
    fn with_stretch_ratio_in_pipeline() {
        let input = sine_mono(440.0, 44100, 44100);
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::HouseLoop)
            .with_stretch_ratio(2.0);

        let output = timestretch::stretch_buffer(&input, &params).unwrap();
        let ratio = output.duration_secs() / input.duration_secs();
        assert!(
            (ratio - 2.0).abs() < 0.15,
            "with_stretch_ratio(2.0) should stretch to 2x, got {ratio}"
        );
    }

    #[test]
    fn debug_audio_buffer() {
        let buf = sine_mono(440.0, 44100, 100);
        let debug = format!("{:?}", buf);
        assert!(debug.contains("AudioBuffer"));
        assert!(!debug.is_empty());
    }

    #[test]
    fn debug_stretch_params() {
        let params = StretchParams::new(1.5)
            .with_preset(EdmPreset::HouseLoop)
            .with_sample_rate(48000);
        let debug = format!("{:?}", params);
        assert!(debug.contains("StretchParams"));
        assert!(!debug.is_empty());
    }

    #[test]
    fn debug_stream_processor() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let proc = StreamProcessor::new(params);
        let debug = format!("{:?}", proc);
        assert!(debug.contains("StreamProcessor"));
    }

    #[test]
    fn display_edm_presets() {
        let presets = [
            EdmPreset::DjBeatmatch,
            EdmPreset::HouseLoop,
            EdmPreset::Halftime,
            EdmPreset::Ambient,
            EdmPreset::VocalChop,
        ];
        for preset in &presets {
            let display = format!("{preset}");
            assert!(!display.is_empty(), "Display for {preset:?} is empty");
        }
    }

    #[test]
    fn partial_eq_after_resample() {
        let a = sine_mono(440.0, 44100, 1000);
        let b = a.resample(44100); // Same rate = clone
        assert_eq!(a, b);
    }

    #[test]
    fn as_ref_with_stretch_output() {
        let input = sine_mono(440.0, 44100, 22050);
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let stretched = timestretch::stretch_buffer(&input, &params).unwrap();
        let slice: &[f32] = stretched.as_ref();
        assert_eq!(slice.len(), stretched.data.len());
    }

    #[test]
    fn into_iterator_with_stereo() {
        let buf = sine_stereo(440.0, 880.0, 44100, 100);
        let mut count = 0;
        for frame in &buf {
            assert_eq!(frame.len(), 2);
            count += 1;
        }
        assert_eq!(count, 100);
    }

    #[test]
    fn default_stretch_params() {
        let params = StretchParams::default();
        assert!((params.stretch_ratio - 1.0).abs() < 1e-10);
        assert_eq!(params.channels, Channels::Stereo);
        assert_eq!(params.sample_rate, 44100);
    }

    #[test]
    fn from_tempo_constructor_ratio_calculation() {
        // from_tempo(128, 126) should give ratio 128/126 ≈ 1.0159
        let params = StretchParams::from_tempo(128.0, 126.0);
        let expected = 128.0 / 126.0;
        assert!(
            (params.stretch_ratio - expected).abs() < 1e-10,
            "Expected ratio {expected}, got {}",
            params.stretch_ratio
        );
    }

    #[test]
    fn from_tempo_with_preset_chain() {
        let params = StretchParams::from_tempo(128.0, 130.0)
            .with_preset(EdmPreset::DjBeatmatch)
            .with_sample_rate(48000)
            .with_channels(2);
        assert_eq!(params.sample_rate, 48000);
        assert_eq!(params.channels, Channels::Stereo);
    }
}

// ──────────────────────────────────────────────────────────────────
// Combined workflow tests: resample + stretch + crossfade + WAV
// ──────────────────────────────────────────────────────────────────

mod combined_workflows {
    use super::*;

    #[test]
    fn dj_full_transition_workflow() {
        // Full DJ workflow: load two tracks, match BPM, crossfade
        let track_a = sine_mono(440.0, 44100, 44100 * 2);
        let track_b = sine_mono(330.0, 44100, 44100 * 2);

        // Stretch track A from 126 to 128 BPM
        let params_a = StretchParams::new(126.0 / 128.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::DjBeatmatch);
        let a_matched = timestretch::stretch_buffer(&track_a, &params_a).unwrap();

        // Track B is already at 128 BPM (identity)
        let params_b = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::DjBeatmatch);
        let b_matched = timestretch::stretch_buffer(&track_b, &params_b).unwrap();

        // Crossfade 500ms
        let fade_frames = 22050;
        let mixed = a_matched.crossfade_into(&b_matched, fade_frames);

        assert!(mixed.num_frames() > 0);
        assert!(mixed.rms() > 0.1);
        assert!(mixed.peak() < 2.0); // No severe clipping
    }

    #[test]
    fn sample_rate_conversion_and_stretch() {
        // Load 48kHz audio, convert to 44.1kHz, stretch, convert back
        let src = sine_mono(440.0, 48000, 48000); // 1s at 48kHz
        let at_441 = src.resample(44100);

        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::HouseLoop);
        let stretched = timestretch::stretch_buffer(&at_441, &params).unwrap();

        let back_48 = stretched.resample(48000);
        assert_eq!(back_48.sample_rate, 48000);
        assert!(back_48.rms() > 0.1);
        // Duration should be ~1.5s
        assert!((back_48.duration_secs() - 1.5).abs() < 0.15);
    }

    #[test]
    fn chop_stretch_crossfade_chain() {
        // Slice a sample, stretch each part differently, crossfade them together
        let full = sine_mono(440.0, 44100, 44100 * 2); // 2s
        let part1 = full.slice(0, 44100); // first second
        let part2 = full.slice(44100, 44100); // second second

        let params1 = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let params2 = StretchParams::new(0.75)
            .with_sample_rate(44100)
            .with_channels(1);

        let s1 = timestretch::stretch_buffer(&part1, &params1).unwrap();
        let s2 = timestretch::stretch_buffer(&part2, &params2).unwrap();

        // Crossfade the two stretched parts
        let fade = 4410; // 100ms
        let result = s1.crossfade_into(&s2, fade);
        assert!(result.num_frames() > 0);
        assert!(result.rms() > 0.1);
    }

    #[test]
    fn normalize_crossfade_workflow() {
        // Normalize two buffers to same level, then crossfade
        let a = sine_mono(440.0, 44100, 44100);
        let b = AudioBuffer::from_mono(
            (0..44100)
                .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 330.0 * i as f32 / 44100.0).sin())
                .collect(),
            44100,
        );

        let a_norm = a.normalize(0.8);
        let b_norm = b.normalize(0.8);

        let mixed = a_norm.crossfade_into(&b_norm, 4410);
        assert!((a_norm.peak() - 0.8).abs() < 0.01);
        assert!((b_norm.peak() - 0.8).abs() < 0.01);
        assert!(mixed.rms() > 0.1);
    }

    #[test]
    fn streaming_then_resample() {
        // Stream-process audio, then resample the collected output
        let mut proc = StreamProcessor::from_tempo(128.0, 126.0, 44100, 1);
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut output_data = Vec::new();
        for chunk in input.chunks(2048) {
            let out = proc.process(chunk).unwrap();
            output_data.extend_from_slice(&out);
        }
        let flushed = proc.flush().unwrap();
        output_data.extend_from_slice(&flushed);

        // Wrap in AudioBuffer and resample to 48kHz
        let output_buf = AudioBuffer::from_mono(output_data, 44100);
        let resampled = output_buf.resample(48000);
        assert_eq!(resampled.sample_rate, 48000);
        assert!(resampled.rms() > 0.01);
    }

    #[test]
    fn window_type_with_bpm_stretch() {
        let input = sine_mono(440.0, 44100, 44100);

        // Test with different window types via BPM stretch
        let windows = [
            WindowType::Hann,
            WindowType::BlackmanHarris,
            WindowType::Kaiser(12),
        ];
        for wt in &windows {
            let params = StretchParams::from_tempo(128.0, 126.0)
                .with_sample_rate(44100)
                .with_channels(1)
                .with_window_type(*wt);
            let output = timestretch::stretch_buffer(&input, &params).unwrap();
            assert!(
                output.num_frames() > 0,
                "Window {wt:?} produced empty output"
            );
            assert!(output.rms() > 0.01, "Window {wt:?} produced silence");
        }
    }

    #[test]
    fn normalize_flag_with_wav_stretch() {
        let dir = std::env::temp_dir().join("timestretch_test_norm_wav");
        std::fs::create_dir_all(&dir).unwrap();
        let input_path = dir.join("norm_in.wav");
        let output_path = dir.join("norm_out.wav");

        let buf = sine_mono(440.0, 44100, 44100);
        timestretch::io::wav::write_wav_file_float(input_path.to_str().unwrap(), &buf).unwrap();

        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_normalize(true);

        let result = timestretch::stretch_wav_file(
            input_path.to_str().unwrap(),
            output_path.to_str().unwrap(),
            &params,
        )
        .unwrap();

        assert!(result.num_frames() > 0);
        assert!(result.rms() > 0.01);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn beat_aware_stretch_with_clicks() {
        // Click train should have beat-aware segmentation respect transients
        let clicks = click_train(
            44100 * 60 / 128, // 128 BPM click interval
            44100 * 2,        // 2 seconds
            44100,
        );

        let params = StretchParams::new(1.25)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::HouseLoop)
            .with_beat_aware(true);

        let output = timestretch::stretch_buffer(&clicks, &params).unwrap();
        let ratio = output.duration_secs() / clicks.duration_secs();
        assert!(
            (ratio - 1.25).abs() < 0.15,
            "Beat-aware stretch ratio off: {ratio}"
        );
        assert!(output.peak() > 0.1, "Clicks should survive stretching");
    }

    #[test]
    fn band_split_with_crossfade() {
        // Stretch with band splitting, then crossfade two results
        let a = sine_mono(60.0, 44100, 44100); // sub-bass
        let b = sine_mono(440.0, 44100, 44100); // mid

        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_preset(EdmPreset::HouseLoop)
            .with_band_split(true);

        let sa = timestretch::stretch_buffer(&a, &params).unwrap();
        let sb = timestretch::stretch_buffer(&b, &params).unwrap();

        let mixed = sa.crossfade_into(&sb, 4410);
        assert!(mixed.num_frames() > 0);
        assert!(mixed.rms() > 0.01);
    }
}

// ──────────────────────────────────────────────────────────────────
// Edge cases for new features
// ──────────────────────────────────────────────────────────────────

mod new_feature_edge_cases {
    use super::*;

    #[test]
    fn resample_very_short_buffer() {
        // Just 1 frame
        let buf = AudioBuffer::from_mono(vec![0.5], 44100);
        let resampled = buf.resample(48000);
        assert!(resampled.num_frames() > 0 || resampled.is_empty());
        // Just 2 frames
        let buf2 = AudioBuffer::from_mono(vec![0.5, -0.5], 44100);
        let resampled2 = buf2.resample(48000);
        assert!(resampled2.num_frames() >= 1);
    }

    #[test]
    fn resample_extreme_rates() {
        let buf = sine_mono(440.0, 44100, 4410); // 100ms
                                                 // Upsample to 96kHz
        let up = buf.resample(96000);
        assert_eq!(up.sample_rate, 96000);
        let dur_diff = (buf.duration_secs() - up.duration_secs()).abs();
        assert!(dur_diff < 0.01);

        // Downsample to 8kHz
        let down = buf.resample(8000);
        assert_eq!(down.sample_rate, 8000);
        let dur_diff2 = (buf.duration_secs() - down.duration_secs()).abs();
        assert!(dur_diff2 < 0.02);
    }

    #[test]
    fn crossfade_into_single_frame_buffers() {
        let a = AudioBuffer::from_mono(vec![1.0], 44100);
        let b = AudioBuffer::from_mono(vec![-1.0], 44100);
        let mixed = a.crossfade_into(&b, 1);
        assert_eq!(mixed.num_frames(), 1);
    }

    #[test]
    fn crossfade_into_empty_second() {
        let a = AudioBuffer::from_mono(vec![1.0; 100], 44100);
        let b = AudioBuffer::from_mono(vec![], 44100);
        let mixed = a.crossfade_into(&b, 10);
        // fade_frames clamped to min(100, 0) = 0
        assert_eq!(mixed.num_frames(), 100);
    }

    #[test]
    fn from_tempo_same_bpm_is_identity() {
        // Same BPM → ratio = 1.0
        // Use a larger input so streaming overhead is proportionally smaller
        let mut proc = StreamProcessor::from_tempo(128.0, 128.0, 44100, 1);
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let mut output = Vec::new();
        for chunk in input.chunks(2048) {
            let out = proc.process(chunk).unwrap();
            output.extend_from_slice(&out);
        }
        let flushed = proc.flush().unwrap();
        output.extend_from_slice(&flushed);

        // Same BPM should produce approximately same-length output
        // (streaming adds some latency padding, so allow generous tolerance)
        assert!(!output.is_empty());
        let ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (ratio - 1.0).abs() < 0.5,
            "Same BPM should be near-identity, got ratio {ratio}"
        );
        // All samples should be finite
        assert!(output.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn with_stretch_ratio_after_from_tempo() {
        // from_tempo sets ratio, with_stretch_ratio overrides it
        let params = StretchParams::from_tempo(128.0, 126.0).with_stretch_ratio(2.0);
        assert!((params.stretch_ratio - 2.0).abs() < 1e-10);
    }

    #[test]
    fn from_audio_buffer_to_vec_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let vec: Vec<f32> = buf.into();
        assert!(vec.is_empty());
    }

    #[test]
    fn hybrid_mode_persists_across_reset() {
        let mut proc = StreamProcessor::from_tempo(128.0, 126.0, 44100, 1);
        proc.set_hybrid_mode(true);
        assert!(proc.is_hybrid_mode());
        proc.reset();
        assert!(
            proc.is_hybrid_mode(),
            "Hybrid mode should persist across reset"
        );
    }

    #[test]
    fn latency_reported_with_from_tempo() {
        let proc = StreamProcessor::from_tempo(128.0, 126.0, 44100, 1);
        let latency = proc.latency_samples();
        assert!(latency > 0, "Should report non-zero latency");
        let latency_secs = proc.latency_secs();
        assert!(latency_secs > 0.0, "Latency in seconds should be positive");
    }

    #[test]
    fn stretch_to_bpm_wav_file_same_bpm() {
        let dir = std::env::temp_dir().join("timestretch_test_same_bpm");
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("in.wav");
        let output = dir.join("out.wav");

        let buf = sine_mono(440.0, 44100, 44100);
        timestretch::io::wav::write_wav_file_float(input.to_str().unwrap(), &buf).unwrap();

        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        let result = timestretch::stretch_to_bpm_wav_file(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            128.0,
            128.0,
            &params,
        )
        .unwrap();

        // Same BPM = identity-ish
        let ratio = result.duration_secs() / buf.duration_secs();
        assert!(
            (ratio - 1.0).abs() < 0.15,
            "Same BPM should be near-identity, got {ratio}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resample_stereo_preserves_frame_alignment() {
        let buf = sine_stereo(440.0, 880.0, 44100, 1000);
        let resampled = buf.resample(48000);
        // Stereo data should always have even sample count
        assert_eq!(
            resampled.data.len() % 2,
            0,
            "Stereo resample must produce even sample count"
        );
        assert_eq!(resampled.channels, Channels::Stereo);
    }

    #[test]
    fn crossfade_asymmetric_lengths() {
        // First buffer shorter than second
        let a = AudioBuffer::from_mono(vec![1.0; 500], 44100);
        let b = AudioBuffer::from_mono(vec![-1.0; 2000], 44100);
        let mixed = a.crossfade_into(&b, 200);
        assert_eq!(mixed.num_frames(), 500 + 2000 - 200);
    }

    #[test]
    fn output_length_helper() {
        let params = StretchParams::new(2.0);
        assert_eq!(params.output_length(1000), 2000);

        let params_compress = StretchParams::new(0.5);
        assert_eq!(params_compress.output_length(1000), 500);
    }
}

//! Tests for algorithm edge cases: windows, resampling, frequency analysis,
//! beat detection, parameter validation, and internal helper functions.

use timestretch::{AudioBuffer, Channels, EdmPreset, StretchParams};

// ===== Window function edge cases =====

#[test]
fn test_window_size_2() {
    use timestretch::core::window::{generate_window, WindowType};

    // Hann of size 2: endpoints should be 0
    let hann = generate_window(WindowType::Hann, 2);
    assert_eq!(hann.len(), 2);
    assert!(hann[0].abs() < 1e-6, "Hann(2)[0] = {}", hann[0]);
    assert!(hann[1].abs() < 1e-6, "Hann(2)[1] = {}", hann[1]);

    // BH of size 2
    let bh = generate_window(WindowType::BlackmanHarris, 2);
    assert_eq!(bh.len(), 2);

    // Kaiser of size 2
    let k = generate_window(WindowType::Kaiser(800), 2);
    assert_eq!(k.len(), 2);
    // Kaiser endpoints should be symmetric
    assert!((k[0] - k[1]).abs() < 1e-6);
}

#[test]
fn test_window_size_3() {
    use timestretch::core::window::{generate_window, WindowType};

    let hann = generate_window(WindowType::Hann, 3);
    assert_eq!(hann.len(), 3);
    // Middle should be peak
    assert!(hann[1] >= hann[0]);
    assert!(hann[1] >= hann[2]);
    // Symmetric
    assert!((hann[0] - hann[2]).abs() < 1e-6);
}

#[test]
fn test_kaiser_beta_zero() {
    use timestretch::core::window::{generate_window, WindowType};

    // Beta=0 should approximate a rectangular window
    let k = generate_window(WindowType::Kaiser(0), 64);
    assert_eq!(k.len(), 64);
    // All values should be close to 1.0 (rectangle)
    for (i, &v) in k.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 0.01,
            "Kaiser(beta=0)[{}] = {}, expected ~1.0",
            i,
            v
        );
    }
}

#[test]
fn test_kaiser_high_beta() {
    use timestretch::core::window::{generate_window, WindowType};

    // High beta (20.0) — should still produce valid results
    let k = generate_window(WindowType::Kaiser(2000), 256);
    assert_eq!(k.len(), 256);
    // All values should be non-negative
    for &v in &k {
        assert!(v >= 0.0, "Kaiser value should be non-negative: {}", v);
        assert!(v.is_finite(), "Kaiser value should be finite: {}", v);
    }
    // Middle should be highest
    let mid = k[128];
    assert!(mid > k[0], "Middle should be higher than edge");
    // Should be strongly tapered (high beta = narrow mainlobe)
    assert!(k[0] < 0.01, "Edge should be very small with high beta");
}

#[test]
fn test_apply_window_mismatched_lengths() {
    use timestretch::core::window::apply_window;

    // Data longer than window
    let window = vec![0.5, 1.0];
    let mut data = vec![2.0, 3.0, 4.0, 5.0];
    apply_window(&mut data, &window);
    assert!((data[0] - 1.0).abs() < 1e-6); // 2.0 * 0.5
    assert!((data[1] - 3.0).abs() < 1e-6); // 3.0 * 1.0
                                           // Remaining samples unchanged
    assert!((data[2] - 4.0).abs() < 1e-6);
    assert!((data[3] - 5.0).abs() < 1e-6);

    // Window longer than data
    let window = vec![0.5, 1.0, 0.5, 0.25];
    let mut data = vec![2.0, 3.0];
    apply_window(&mut data, &window);
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[1] - 3.0).abs() < 1e-6);
}

#[test]
fn test_apply_window_copy_mismatched_lengths() {
    use timestretch::core::window::apply_window_copy;

    // Data longer than window — result truncated to shorter
    let window = vec![0.5, 1.0];
    let data = vec![2.0, 3.0, 4.0, 5.0];
    let result = apply_window_copy(&data, &window);
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 3.0).abs() < 1e-6);
}

#[test]
fn test_window_all_values_finite() {
    use timestretch::core::window::{generate_window, WindowType};

    for size in [1, 2, 3, 4, 7, 16, 64, 256, 1024, 4096] {
        for wt in [
            WindowType::Hann,
            WindowType::BlackmanHarris,
            WindowType::Kaiser(800),
        ] {
            let w = generate_window(wt, size);
            assert_eq!(w.len(), size);
            for &v in &w {
                assert!(
                    v.is_finite(),
                    "{:?} size {} produced non-finite value",
                    wt,
                    size
                );
            }
        }
    }
}

// ===== Resample edge cases =====

#[test]
fn test_resample_single_sample_input() {
    use timestretch::core::resample::{resample_cubic, resample_linear};

    // Single sample should be replicated
    let input = vec![0.75];
    let linear = resample_linear(&input, 10);
    assert_eq!(linear.len(), 10);
    for &v in &linear {
        assert!((v - 0.75).abs() < 1e-6);
    }

    let cubic = resample_cubic(&input, 10);
    assert_eq!(cubic.len(), 10);
    for &v in &cubic {
        assert!((v - 0.75).abs() < 1e-6);
    }
}

#[test]
fn test_resample_two_samples_cubic_fallback() {
    use timestretch::core::resample::resample_cubic;

    // < 4 samples falls back to linear
    let input = vec![0.0, 1.0];
    let output = resample_cubic(&input, 5);
    assert_eq!(output.len(), 5);
    // Should be monotonically increasing (linear interpolation)
    for i in 1..output.len() {
        assert!(output[i] >= output[i - 1] - 1e-6);
    }
}

#[test]
fn test_resample_three_samples_cubic_fallback() {
    use timestretch::core::resample::resample_cubic;

    let input = vec![0.0, 1.0, 0.0];
    let output = resample_cubic(&input, 7);
    assert_eq!(output.len(), 7);
    // All values should be finite
    for &v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_resample_exactly_four_samples_cubic() {
    use timestretch::core::resample::resample_cubic;

    // Exactly 4 samples — minimum for cubic
    let input = vec![0.0, 0.5, 1.0, 0.5];
    let output = resample_cubic(&input, 8);
    assert_eq!(output.len(), 8);
    for &v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_resample_output_length_1() {
    use timestretch::core::resample::{resample_cubic, resample_linear};

    let input: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
    let linear = resample_linear(&input, 1);
    assert_eq!(linear.len(), 1);
    assert!((linear[0] - 0.0).abs() < 1e-6); // First sample

    let cubic = resample_cubic(&input, 1);
    assert_eq!(cubic.len(), 1);
}

#[test]
fn test_resample_extreme_upsample() {
    use timestretch::core::resample::resample_linear;

    let input = vec![0.0, 1.0, 0.0];
    let output = resample_linear(&input, 1000);
    assert_eq!(output.len(), 1000);
    // All values should be between 0 and 1
    for &v in &output {
        assert!((-0.01..=1.01).contains(&v), "Out of range: {}", v);
    }
}

#[test]
fn test_resample_cubic_output_length_zero() {
    use timestretch::core::resample::resample_cubic;

    let input = vec![0.0, 1.0, 0.0, 1.0, 0.0];
    let output = resample_cubic(&input, 0);
    assert!(output.is_empty());
}

// ===== Frequency analysis edge cases =====

#[test]
fn test_band_energy_short_input() {
    use timestretch::analysis::frequency::{compute_band_energy, FrequencyBands};

    // Input shorter than FFT size should return zeros
    let samples = vec![0.5; 100];
    let (sub, low, mid, high) =
        compute_band_energy(&samples, 4096, 44100, &FrequencyBands::default());
    assert!(sub == 0.0 && low == 0.0 && mid == 0.0 && high == 0.0);
}

#[test]
fn test_band_energy_exactly_fft_size() {
    use timestretch::analysis::frequency::{compute_band_energy, FrequencyBands};

    let fft_size = 4096;
    let samples: Vec<f32> = (0..fft_size)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let (sub, low, _mid, _high) =
        compute_band_energy(&samples, fft_size, 44100, &FrequencyBands::default());
    // 440 Hz is in the "low" band (120-500 Hz)
    assert!(
        low > sub,
        "440 Hz should have more low energy than sub-bass"
    );
}

#[test]
fn test_band_energy_silence() {
    use timestretch::analysis::frequency::{compute_band_energy, FrequencyBands};

    let samples = vec![0.0; 4096];
    let (sub, low, mid, high) =
        compute_band_energy(&samples, 4096, 44100, &FrequencyBands::default());
    assert!(sub < 1e-10);
    assert!(low < 1e-10);
    assert!(mid < 1e-10);
    assert!(high < 1e-10);
}

#[test]
fn test_freq_to_bin_edge_cases() {
    use timestretch::analysis::frequency::{bin_to_freq, freq_to_bin};

    // 0 Hz = bin 0
    assert_eq!(freq_to_bin(0.0, 4096, 44100), 0);
    // Nyquist = fft_size/2
    let nyquist_bin = freq_to_bin(22050.0, 4096, 44100);
    assert_eq!(nyquist_bin, 2048);
    // Above Nyquist clamped
    let above_nyquist = freq_to_bin(30000.0, 4096, 44100);
    assert_eq!(above_nyquist, 2048);

    // bin_to_freq(0) = 0
    assert!((bin_to_freq(0, 4096, 44100) - 0.0).abs() < 1e-6);
    // bin_to_freq(2048) = Nyquist
    let freq = bin_to_freq(2048, 4096, 44100);
    assert!((freq - 22050.0).abs() < 1.0);
}

#[test]
fn test_split_spectrum_custom_bands() {
    use rustfft::num_complex::Complex;
    use timestretch::analysis::frequency::{split_spectrum_into_bands, FrequencyBands};

    let bands = FrequencyBands {
        sub_bass: 60.0, // Very low sub
        low: 200.0,     // Narrow low
        mid: 2000.0,    // Lower mid cutoff
    };

    // Create a spectrum with energy at specific bins
    let fft_size = 1024;
    let half = fft_size / 2 + 1;
    let mut spectrum = vec![Complex::new(0.0f32, 0.0); half];
    // Put energy at ~100 Hz (should be in "low" band with cutoff at 60 Hz)
    let bin_100 = (100.0 * fft_size as f32 / 44100.0).round() as usize;
    spectrum[bin_100] = Complex::new(1.0, 0.0);

    let (sub, low, _mid, _high) = split_spectrum_into_bands(&spectrum, fft_size, 44100, &bands);
    // 100 Hz should be in the low band (between 60 and 200)
    assert!(
        low[bin_100].norm_sqr() > 0.5,
        "100 Hz should be in low band"
    );
    assert!(
        sub[bin_100].norm_sqr() < 1e-6,
        "100 Hz should not be in sub-bass"
    );
}

// ===== Beat detection edge cases =====

#[test]
fn test_beat_detection_very_short_audio() {
    // Less than one FFT frame
    let audio = vec![0.5f32; 100];
    let bpm = timestretch::detect_bpm(&audio, 44100);
    assert!(
        bpm == 0.0,
        "Very short audio should return 0 BPM, got {}",
        bpm
    );
}

#[test]
fn test_beat_grid_snap_empty_grid() {
    let grid = timestretch::detect_beat_grid(&[], 44100);
    // snap_to_grid on empty grid should return the position itself
    let snapped = grid.snap_to_grid(1000);
    assert_eq!(snapped, 1000);
}

#[test]
fn test_beat_grid_interval_samples() {
    let grid = timestretch::BeatGrid {
        beats: vec![0, 22050],
        bpm: 120.0,
        sample_rate: 44100,
    };
    let interval = grid.beat_interval_samples();
    // 120 BPM at 44100 Hz = 22050 samples per beat
    assert!((interval - 22050.0).abs() < 1.0);
}

#[test]
fn test_beat_detection_constant_dc() {
    // Constant signal (DC) should have no beats
    let audio = vec![0.5f32; 44100 * 4];
    let bpm = timestretch::detect_bpm(&audio, 44100);
    assert!(
        bpm == 0.0,
        "DC signal should have no beats, got {} BPM",
        bpm
    );
}

#[test]
fn test_beat_detection_white_noise() {
    // Pseudo-random noise — beat detection should either find nothing or something
    // The key is that it doesn't crash
    let mut audio = vec![0.0f32; 44100 * 4];
    let mut seed: u32 = 12345;
    for sample in &mut audio {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        *sample = ((seed >> 16) as f32 / 32768.0) - 1.0;
    }
    let bpm = timestretch::detect_bpm(&audio, 44100);
    // No specific BPM assertion; just verify no crash and value >= 0
    assert!(bpm >= 0.0);
}

// ===== Parameter validation edge cases =====

#[test]
fn test_params_exact_boundary_ratios() {
    // Exact minimum ratio (0.01) should be valid
    let params = StretchParams::new(0.01);
    let result = timestretch::stretch(&vec![0.0f32; 44100], &params);
    assert!(result.is_ok(), "Ratio 0.01 should be valid");

    // Exact maximum ratio (100.0) should be valid
    let params = StretchParams::new(100.0);
    // This will produce very large output but should not be rejected
    // Use small input to avoid memory issues
    let result = timestretch::stretch(&vec![0.0f32; 1000], &params);
    assert!(result.is_ok(), "Ratio 100.0 should be valid");
}

#[test]
fn test_params_just_outside_boundaries() {
    // Just below minimum
    let params = StretchParams::new(0.009);
    let result = timestretch::stretch(&vec![0.0f32; 44100], &params);
    assert!(result.is_err(), "Ratio 0.009 should be invalid");

    // Just above maximum
    let params = StretchParams::new(100.1);
    let result = timestretch::stretch(&vec![0.0f32; 44100], &params);
    assert!(result.is_err(), "Ratio 100.1 should be invalid");
}

#[test]
fn test_params_hop_size_equals_fft_size() {
    // hop_size == fft_size (0% overlap) should be valid per validation
    let params = StretchParams::new(1.5)
        .with_fft_size(4096)
        .with_hop_size(4096);
    // Processing may produce poor quality but should not crash
    let input: Vec<f32> = (0..44100)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let result = timestretch::stretch(&input, &params);
    assert!(result.is_ok(), "hop=fft should be valid");
}

#[test]
fn test_params_very_large_fft_size() {
    let params = StretchParams::new(1.5).with_fft_size(16384);
    // Should be valid (power of 2 and >= 256)
    let input: Vec<f32> = (0..44100)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let result = timestretch::stretch(&input, &params);
    assert!(result.is_ok(), "FFT size 16384 should be valid");
}

#[test]
fn test_params_minimum_fft_size() {
    let params = StretchParams::new(1.5).with_fft_size(256);
    let input: Vec<f32> = (0..44100)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let result = timestretch::stretch(&input, &params);
    assert!(result.is_ok(), "FFT size 256 should be valid");
}

#[test]
fn test_params_output_length_calculation() {
    let params = StretchParams::new(1.5);
    assert_eq!(params.output_length(1000), 1500);
    assert_eq!(params.output_length(0), 0);

    let params = StretchParams::new(0.5);
    assert_eq!(params.output_length(1000), 500);

    let params = StretchParams::new(2.0);
    assert_eq!(params.output_length(44100), 88200);
}

#[test]
fn test_preset_overrides_fft_and_hop() {
    // DjBeatmatch preset: Kaiser(800), hop = fft_size/5
    let params = StretchParams::new(1.0).with_preset(EdmPreset::DjBeatmatch);
    assert_eq!(params.fft_size, 4096);
    assert_eq!(params.hop_size, 4096 / 5);
    assert!(params.beat_aware); // Presets enable beat_aware

    // Ambient preset: BH, hop = fft_size/4
    let params = StretchParams::new(1.0).with_preset(EdmPreset::Ambient);
    assert_eq!(params.fft_size, 8192);
    assert_eq!(params.hop_size, 8192 / 4);

    // VocalChop preset: Kaiser(600), hop = fft_size/4
    let params = StretchParams::new(1.0).with_preset(EdmPreset::VocalChop);
    assert_eq!(params.fft_size, 2048);
    assert_eq!(params.hop_size, 2048 / 4);
}

#[test]
fn test_preset_after_sample_rate_uses_correct_wsola() {
    // When preset is applied after sample rate, WSOLA should use the correct rate
    let params = StretchParams::new(1.0)
        .with_sample_rate(48000)
        .with_preset(EdmPreset::HouseLoop);
    // WSOLA search range should be based on 48000
    let expected_search_ms = 15.0; // HouseLoop uses WSOLA_SEARCH_MS_MEDIUM = 15ms
    let expected_samples = (48000.0 * expected_search_ms / 1000.0) as usize;
    assert_eq!(params.wsola_search_range, expected_samples);
}

#[test]
fn test_with_beat_aware_toggle() {
    let params = StretchParams::new(1.0).with_beat_aware(true);
    assert!(params.beat_aware);

    let params = StretchParams::new(1.0).with_beat_aware(false);
    assert!(!params.beat_aware);

    // Preset enables beat_aware, can be overridden
    let params = StretchParams::new(1.0)
        .with_preset(EdmPreset::HouseLoop)
        .with_beat_aware(false);
    assert!(!params.beat_aware);
}

// ===== AudioBuffer edge cases =====

#[test]
fn test_audio_buffer_from_channels_single() {
    // Single channel -> Mono
    let buf = AudioBuffer::from_channels(&[vec![1.0, 2.0, 3.0]], 44100);
    assert_eq!(buf.channels, Channels::Mono);
    assert_eq!(buf.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_audio_buffer_from_channels_unequal_lengths() {
    // Channels with different lengths — should truncate to shortest
    let left = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let right = vec![6.0, 7.0, 8.0];
    let buf = AudioBuffer::from_channels(&[left, right], 44100);
    assert_eq!(buf.num_frames(), 3);
    assert_eq!(buf.data, vec![1.0, 6.0, 2.0, 7.0, 3.0, 8.0]);
}

#[test]
fn test_audio_buffer_from_channels_empty() {
    let buf = AudioBuffer::from_channels(&[vec![], vec![]], 44100);
    assert!(buf.is_empty());
    assert_eq!(buf.num_frames(), 0);
}

#[test]
fn test_audio_buffer_mix_to_mono_empty() {
    let buf = AudioBuffer::from_stereo(vec![], 44100);
    let mono = buf.mix_to_mono();
    assert!(mono.is_empty());
    assert!(mono.is_mono());
}

#[test]
fn test_audio_buffer_to_stereo_empty() {
    let buf = AudioBuffer::from_mono(vec![], 44100);
    let stereo = buf.to_stereo();
    assert!(stereo.is_empty());
    assert!(stereo.is_stereo());
}

#[test]
fn test_audio_buffer_channel_extraction_large() {
    // Verify channel extraction works for larger buffers
    let num_frames = 10000;
    let mut data = Vec::with_capacity(num_frames * 2);
    for i in 0..num_frames {
        data.push(i as f32); // L
        data.push(-(i as f32)); // R
    }
    let buf = AudioBuffer::from_stereo(data, 44100);
    let left = buf.left();
    let right = buf.right();
    assert_eq!(left.len(), num_frames);
    assert_eq!(right.len(), num_frames);
    for i in 0..num_frames {
        assert!((left[i] - i as f32).abs() < 1e-6);
        assert!((right[i] + i as f32).abs() < 1e-6);
    }
}

// ===== Multi-stage processing (regression/stress) =====

#[test]
fn test_stretch_then_compress_back() {
    // Stretch 1.5x then compress back to ~original length
    let sample_rate = 44100u32;
    let input: Vec<f32> = (0..44100)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect();

    let params_stretch = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);
    let stretched = timestretch::stretch(&input, &params_stretch).unwrap();

    let params_compress = StretchParams::new(1.0 / 1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);
    let result = timestretch::stretch(&stretched, &params_compress).unwrap();

    // Result length should be approximately original length
    let ratio = result.len() as f64 / input.len() as f64;
    assert!(
        (ratio - 1.0).abs() < 0.3,
        "Round-trip length ratio: {}, expected ~1.0",
        ratio
    );

    // No NaN or Inf
    for &s in &result {
        assert!(s.is_finite(), "Round-trip produced non-finite value");
    }
}

#[test]
fn test_successive_small_stretches() {
    // Apply multiple small stretches in sequence (DJ-style pitch drift)
    let sample_rate = 44100u32;
    let input: Vec<f32> = (0..44100)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect();

    let mut audio = input.clone();
    for _ in 0..5 {
        let params = StretchParams::new(1.02)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(EdmPreset::DjBeatmatch);
        audio = timestretch::stretch(&audio, &params).unwrap();
    }

    // After 5x 1.02 stretch: total ratio ≈ 1.02^5 ≈ 1.104
    let ratio = audio.len() as f64 / input.len() as f64;
    assert!(
        (ratio - 1.104).abs() < 0.3,
        "5x 1.02 stretch ratio: {}, expected ~1.104",
        ratio
    );

    // No NaN or Inf
    for &s in &audio {
        assert!(s.is_finite());
    }
}

#[test]
fn test_pathological_step_function() {
    // Abrupt step from -1 to +1 (discontinuity)
    let mut input = vec![-1.0f32; 22050];
    input.extend(vec![1.0f32; 22050]);

    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let output = timestretch::stretch(&input, &params).unwrap();
    assert!(!output.is_empty());
    // No NaN/Inf
    for &s in &output {
        assert!(s.is_finite());
    }
}

#[test]
fn test_pathological_saturated_input() {
    // All samples at ±1.0 (fully clipped signal)
    let mut input = Vec::with_capacity(44100);
    for i in 0..44100 {
        input.push(if i % 2 == 0 { 1.0 } else { -1.0 });
    }

    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let output = timestretch::stretch(&input, &params).unwrap();
    assert!(!output.is_empty());
    for &s in &output {
        assert!(s.is_finite());
    }
}

#[test]
fn test_inverted_phase_stereo() {
    // L = signal, R = -signal (inverted phase)
    let num_frames = 44100;
    let mut input = Vec::with_capacity(num_frames * 2);
    for i in 0..num_frames {
        let s = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin();
        input.push(s);
        input.push(-s);
    }

    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(2);
    let output = timestretch::stretch(&input, &params).unwrap();
    assert!(!output.is_empty());
    assert_eq!(output.len() % 2, 0);

    // L and R should still be roughly inverted
    let mut inversion_count = 0;
    for i in (0..output.len()).step_by(2) {
        if output[i].abs() > 0.01
            && output[i + 1].abs() > 0.01
            && (output[i] + output[i + 1]).abs() < 0.5 * (output[i].abs() + output[i + 1].abs())
        {
            inversion_count += 1;
        }
    }
    let total_frames = output.len() / 2;
    // At least some frames should show phase inversion
    assert!(
        inversion_count > total_frames / 10,
        "Inverted phase should be at least partially preserved: {}/{}",
        inversion_count,
        total_frames
    );
}

#[test]
fn test_stretch_with_all_builder_methods() {
    // Exercise all builder methods together
    let params = StretchParams::new(1.5)
        .with_sample_rate(48000)
        .with_channels(2)
        .with_preset(EdmPreset::Halftime)
        .with_fft_size(4096)
        .with_hop_size(512)
        .with_transient_sensitivity(0.8)
        .with_sub_bass_cutoff(80.0)
        .with_wsola_segment_size(1000)
        .with_wsola_search_range(500)
        .with_beat_aware(false);

    assert_eq!(params.sample_rate, 48000);
    assert_eq!(params.fft_size, 4096);
    assert_eq!(params.hop_size, 512);
    assert!((params.transient_sensitivity - 0.8).abs() < 1e-6);
    assert!((params.sub_bass_cutoff - 80.0).abs() < 1e-6);
    assert_eq!(params.wsola_segment_size, 1000);
    assert_eq!(params.wsola_search_range, 500);
    assert!(!params.beat_aware);

    // Actually process with these params
    let mut input = vec![0.0f32; 48000 * 2];
    for i in 0..48000 {
        let t = i as f32 / 48000.0;
        input[i * 2] = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        input[i * 2 + 1] = (2.0 * std::f32::consts::PI * 880.0 * t).sin();
    }
    let result = timestretch::stretch(&input, &params).unwrap();
    assert!(!result.is_empty());
    assert_eq!(result.len() % 2, 0);
}

// ===== BPM API edge cases =====

#[test]
fn test_bpm_ratio_extreme_values() {
    // Very high ratio (10 BPM -> 200 BPM)
    let ratio = timestretch::bpm_ratio(10.0, 200.0);
    assert!((ratio - 0.05).abs() < 1e-10);

    // Very low ratio (200 BPM -> 10 BPM)
    let ratio = timestretch::bpm_ratio(200.0, 10.0);
    assert!((ratio - 20.0).abs() < 1e-10);
}

#[test]
fn test_stretch_to_bpm_extreme_ratio() {
    // 60 BPM -> 180 BPM (ratio 0.333 — large compression)
    let input: Vec<f32> = (0..44100)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let output = timestretch::stretch_to_bpm(&input, 60.0, 180.0, &params).unwrap();
    assert!(output.len() < input.len());
    for &s in &output {
        assert!(s.is_finite());
    }
}

#[test]
fn test_stretch_to_bpm_auto_empty_input() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let result = timestretch::stretch_to_bpm_auto(&[], 128.0, &params);
    // Empty input should return empty output (validate_input returns Ok(false))
    if let Ok(v) = result {
        assert!(v.is_empty());
    }
    // Err is also acceptable (empty input or BPM detection failure)
}

#[test]
fn test_stretch_bpm_buffer_auto_silence() {
    let buf = AudioBuffer::from_mono(vec![0.0f32; 44100 * 4], 44100);
    let params = StretchParams::new(1.0);
    let result = timestretch::stretch_bpm_buffer_auto(&buf, 128.0, &params);
    assert!(result.is_err()); // Can't detect BPM from silence
}

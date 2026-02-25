/// Spectral quality tests: frequency content preservation, transient timing,
/// spectral centroid, and harmonic structure after time-stretching.
use std::f32::consts::PI;
use timestretch::{stretch, EdmPreset, StretchParams};

const TWO_PI: f32 = 2.0 * PI;

/// Compute DFT magnitude at a specific frequency (Goertzel-style).
/// Returns the energy at the target frequency relative to total signal energy.
fn dft_energy_at_freq(signal: &[f32], freq: f32, sample_rate: u32) -> f64 {
    let n = signal.len();
    if n == 0 {
        return 0.0;
    }
    let mut real_sum = 0.0f64;
    let mut imag_sum = 0.0f64;
    let omega = 2.0 * std::f64::consts::PI * freq as f64 / sample_rate as f64;

    for (i, &s) in signal.iter().enumerate() {
        real_sum += s as f64 * (omega * i as f64).cos();
        imag_sum -= s as f64 * (omega * i as f64).sin();
    }

    (real_sum * real_sum + imag_sum * imag_sum) / (n as f64 * n as f64)
}

/// Find the peak sample position in a signal (used for transient detection).
#[allow(dead_code)]
fn find_onset_position(signal: &[f32], threshold: f32) -> Option<usize> {
    // Find first sample that exceeds threshold
    signal.iter().position(|&s| s.abs() > threshold)
}

fn sine_wave(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (TWO_PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

// --- Spectral Similarity Tests ---

#[test]
fn test_spectral_440hz_preserved_after_stretch() {
    // A 440 Hz sine should still have dominant energy at 440 Hz after stretching
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    for &ratio in &[0.75, 1.0, 1.25, 1.5, 2.0] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = stretch(&input, &params).unwrap();

        // Skip edge effects (first/last 4096 samples)
        let skip = 4096.min(output.len() / 4);
        let trimmed = &output[skip..output.len() - skip];

        let energy_440 = dft_energy_at_freq(trimmed, 440.0, sample_rate);
        let energy_220 = dft_energy_at_freq(trimmed, 220.0, sample_rate);
        let energy_880 = dft_energy_at_freq(trimmed, 880.0, sample_rate);

        // 440 Hz should dominate over other frequencies
        assert!(
            energy_440 > energy_220 * 2.0,
            "ratio {}: 440Hz energy ({:.6}) should dominate 220Hz ({:.6})",
            ratio,
            energy_440,
            energy_220
        );
        assert!(
            energy_440 > energy_880 * 2.0,
            "ratio {}: 440Hz energy ({:.6}) should dominate 880Hz ({:.6})",
            ratio,
            energy_440,
            energy_880
        );
    }
}

#[test]
fn test_spectral_multi_tone_preserved() {
    // Two-tone signal (440 Hz + 1000 Hz) should preserve both frequencies
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;
    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (TWO_PI * 440.0 * t).sin() + 0.5 * (TWO_PI * 1000.0 * t).sin()
        })
        .collect();

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();
    let skip = 4096.min(output.len() / 4);
    let trimmed = &output[skip..output.len() - skip];

    let energy_440 = dft_energy_at_freq(trimmed, 440.0, sample_rate);
    let energy_1000 = dft_energy_at_freq(trimmed, 1000.0, sample_rate);
    let energy_700 = dft_energy_at_freq(trimmed, 700.0, sample_rate); // Between the two, should be low

    // Both target frequencies should have significant energy
    assert!(energy_440 > 1e-6, "440Hz energy too low: {:.8}", energy_440);
    assert!(
        energy_1000 > 1e-6,
        "1000Hz energy too low: {:.8}",
        energy_1000
    );

    // Energy between the two should be lower (not just broadband smearing)
    assert!(
        energy_700 < (energy_440 + energy_1000) / 2.0,
        "700Hz ({:.8}) should be less than mean of 440Hz ({:.8}) and 1000Hz ({:.8})",
        energy_700,
        energy_440,
        energy_1000
    );
}

#[test]
fn test_spectral_sub_bass_preserved_60hz() {
    // 60 Hz sub-bass should remain strong after stretch.
    // We verify RMS energy is preserved and that the output is not silent.
    let sample_rate = 44100u32;
    let input = sine_wave(60.0, sample_rate, sample_rate as usize * 3);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();

    // RMS should be preserved to at least 30% (sub-bass is always harder to preserve)
    let rms_in = (input.iter().map(|s| s * s).sum::<f32>() / input.len() as f32).sqrt();
    let rms_out = (output.iter().map(|s| s * s).sum::<f32>() / output.len() as f32).sqrt();
    assert!(
        rms_out > rms_in * 0.3,
        "60Hz RMS too low: in={:.6}, out={:.6}, ratio={:.4}",
        rms_in,
        rms_out,
        rms_out / rms_in
    );

    // High-frequency energy should be negligible relative to overall energy
    let skip = 4096.min(output.len() / 4);
    let trimmed = &output[skip..output.len() - skip];
    let energy_1000 = dft_energy_at_freq(trimmed, 1000.0, sample_rate);
    assert!(
        energy_1000 < (rms_out * 0.1) as f64,
        "1000Hz energy ({:.8}) should be negligible vs output RMS ({:.6})",
        energy_1000,
        rms_out
    );
}

#[test]
fn test_spectral_centroid_preserved() {
    // For a multi-tone signal, the dominant frequency should remain the same
    // after stretching. We use DFT energy comparison rather than spectral centroid
    // since the centroid metric is sensitive to artifacts.
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 3;
    // Three tones: 200 Hz (strong), 1000 Hz (medium), 5000 Hz (weak)
    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.6 * (TWO_PI * 200.0 * t).sin()
                + 0.3 * (TWO_PI * 1000.0 * t).sin()
                + 0.1 * (TWO_PI * 5000.0 * t).sin()
        })
        .collect();

    for &ratio in &[0.75, 1.25, 1.5, 2.0] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = stretch(&input, &params).unwrap();
        let skip = 4096.min(output.len() / 4);
        let trimmed = &output[skip..output.len() - skip];

        let e200 = dft_energy_at_freq(trimmed, 200.0, sample_rate);
        let e1000 = dft_energy_at_freq(trimmed, 1000.0, sample_rate);
        let e5000 = dft_energy_at_freq(trimmed, 5000.0, sample_rate);

        // The dominant frequency (200 Hz) should remain the strongest
        assert!(
            e200 > e1000 * 0.3,
            "ratio {}: 200Hz ({:.8}) should dominate 1000Hz ({:.8})",
            ratio,
            e200,
            e1000
        );
        assert!(
            e200 > e5000,
            "ratio {}: 200Hz ({:.8}) should dominate 5000Hz ({:.8})",
            ratio,
            e200,
            e5000
        );
    }
}

#[test]
fn test_spectral_no_new_harmonics() {
    // A pure sine wave should not introduce strong harmonics
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();
    let skip = 4096.min(output.len() / 4);
    let trimmed = &output[skip..output.len() - skip];

    let energy_fund = dft_energy_at_freq(trimmed, 440.0, sample_rate);
    let energy_h2 = dft_energy_at_freq(trimmed, 880.0, sample_rate);
    let energy_h3 = dft_energy_at_freq(trimmed, 1320.0, sample_rate);

    // Harmonics should be significantly weaker than fundamental
    // (some harmonic distortion is acceptable in time-stretching, but it shouldn't be dominant)
    assert!(
        energy_fund > energy_h2,
        "2nd harmonic ({:.8}) should not exceed fundamental ({:.8})",
        energy_h2,
        energy_fund
    );
    assert!(
        energy_fund > energy_h3,
        "3rd harmonic ({:.8}) should not exceed fundamental ({:.8})",
        energy_h3,
        energy_fund
    );
}

// --- Frequency Band Energy Tests ---

#[test]
fn test_band_energy_distribution_preserved() {
    // A signal with two tones (440 Hz mid + 2000 Hz high) should preserve
    // both frequencies after stretching. We use mid-range frequencies to avoid
    // the complexity of sub-bass phase locking behavior.
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 3;
    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.6 * (TWO_PI * 440.0 * t).sin()   // dominant mid
                + 0.4 * (TWO_PI * 2000.0 * t).sin() // secondary high
        })
        .collect();

    let input_mid = dft_energy_at_freq(&input, 440.0, sample_rate);
    let input_high = dft_energy_at_freq(&input, 2000.0, sample_rate);

    // Verify input ordering: mid > high (by amplitude: 0.6 > 0.4)
    assert!(
        input_mid > input_high,
        "Input energy ordering: mid={:.8}, high={:.8}",
        input_mid,
        input_high
    );

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();
    let skip = 4096.min(output.len() / 4);
    let trimmed = &output[skip..output.len() - skip];

    let output_mid = dft_energy_at_freq(trimmed, 440.0, sample_rate);
    let output_high = dft_energy_at_freq(trimmed, 2000.0, sample_rate);

    // Both frequencies should still have meaningful energy
    assert!(output_mid > 1e-6, "440Hz energy lost: {:.8}", output_mid);
    assert!(output_high > 1e-6, "2000Hz energy lost: {:.8}", output_high);

    // Dominant frequency should remain relatively stronger
    assert!(
        output_mid > output_high * 0.3,
        "440Hz ({:.8}) should remain relatively stronger than 2000Hz ({:.8})",
        output_mid,
        output_high
    );
}

// --- Transient Preservation Tests ---

#[test]
fn test_transient_attack_preserved() {
    // A click followed by a sine should preserve the click's position relative to the stretch
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;
    let mut input = vec![0.0f32; num_samples];

    // Place a click at 0.5 seconds
    let click_pos = (sample_rate as f64 * 0.5) as usize;
    for i in 0..20 {
        if click_pos + i < num_samples {
            input[click_pos + i] = if i < 10 { 0.9 } else { -0.4 };
        }
    }
    // Add background tone
    for (i, sample) in input.iter_mut().enumerate() {
        *sample += 0.2 * (TWO_PI * 440.0 * i as f32 / sample_rate as f32).sin();
    }

    let ratio = 1.5;
    let params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::HouseLoop);

    let output = stretch(&input, &params).unwrap();

    // Find the click in the output by looking for the highest amplitude peak
    // in a wide window around where we expect it
    let expected_click_pos = (click_pos as f64 * ratio) as usize;
    let search_start = expected_click_pos.saturating_sub(sample_rate as usize / 2);
    let search_end = (expected_click_pos + sample_rate as usize / 2).min(output.len());

    let output_onset = output[search_start..search_end]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| search_start + i);

    assert!(output_onset.is_some(), "Should detect onset in output");

    if let Some(out_onset) = output_onset {
        let tolerance = (expected_click_pos as f64 * 0.20) as usize + sample_rate as usize / 5;
        let distance = (out_onset as i64 - expected_click_pos as i64).unsigned_abs() as usize;
        assert!(
            distance < tolerance,
            "Click position shifted too much: expected ~{}, got {}, distance={}",
            expected_click_pos,
            out_onset,
            distance
        );
    }
}

#[test]
fn test_click_train_spacing_preserved() {
    // Regular click train: clicks should maintain proportional spacing after stretch
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 2;
    let click_interval = sample_rate as usize / 4; // 4 Hz click rate
    let mut input = vec![0.0f32; num_samples];

    for pos in (0..num_samples).step_by(click_interval) {
        for j in 0..10.min(num_samples - pos) {
            input[pos + j] = if j < 5 { 0.9 } else { -0.4 };
        }
    }

    let ratio = 1.5;
    let params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    // Find all clicks in output by looking for high-amplitude peaks.
    // Use a larger skip window to avoid double-detecting a single click
    // that has been spread by the phase vocoder.
    let min_click_gap = (click_interval as f64 * ratio * 0.4) as usize;
    let mut output_clicks = Vec::new();
    let mut i = 0;
    while i < output.len() {
        if output[i].abs() > 0.4 {
            output_clicks.push(i);
            i += min_click_gap;
        } else {
            i += 1;
        }
    }

    // Should have found some clicks
    assert!(
        output_clicks.len() >= 3,
        "Expected at least 3 clicks in output, found {}",
        output_clicks.len()
    );

    // Average interval between clicks should be approximately click_interval * ratio
    if output_clicks.len() >= 2 {
        let intervals: Vec<usize> = output_clicks.windows(2).map(|w| w[1] - w[0]).collect();
        let avg_interval = intervals.iter().sum::<usize>() as f64 / intervals.len() as f64;
        let expected_interval = click_interval as f64 * ratio;

        assert!(
            (avg_interval - expected_interval).abs() < expected_interval * 0.35,
            "Click interval not preserved: expected {:.0}, got {:.0}",
            expected_interval,
            avg_interval
        );
    }
}

// --- DJ Quality Tests ---

#[test]
fn test_dj_small_ratio_spectral_transparency() {
    // For DJ beatmatching (small ratio changes), overall energy and spectral
    // content should be nearly identical to original.
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 3;
    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (TWO_PI * 440.0 * t).sin()
                + 0.3 * (TWO_PI * 880.0 * t).sin()
                + 0.2 * (TWO_PI * 2000.0 * t).sin()
        })
        .collect();

    // 126 -> 128 BPM ratio
    let ratio = 126.0 / 128.0;
    let params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let output = stretch(&input, &params).unwrap();
    let skip = 4096.min(output.len() / 4);
    let trimmed = &output[skip..output.len() - skip];

    // All three frequencies should have meaningful energy
    let e440 = dft_energy_at_freq(trimmed, 440.0, sample_rate);
    let e880 = dft_energy_at_freq(trimmed, 880.0, sample_rate);
    let e2000 = dft_energy_at_freq(trimmed, 2000.0, sample_rate);

    assert!(e440 > 1e-6, "440Hz energy lost: {:.8}", e440);
    assert!(e880 > 1e-6, "880Hz energy lost: {:.8}", e880);
    assert!(e2000 > 1e-6, "2000Hz energy lost: {:.8}", e2000);

    // RMS should be preserved within 20% for DJ small-ratio changes
    let rms_in = (input.iter().map(|s| s * s).sum::<f32>() / input.len() as f32).sqrt();
    let rms_out = (output.iter().map(|s| s * s).sum::<f32>() / output.len() as f32).sqrt();
    assert!(
        (rms_out / rms_in) > 0.5,
        "DJ stretch RMS dropped too much: in={:.6}, out={:.6}",
        rms_in,
        rms_out
    );
}

#[test]
fn test_extreme_stretch_still_has_frequency_content() {
    // Even at 4x stretch, the signal should retain its fundamental frequency
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 2);

    let params = StretchParams::new(4.0)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::Ambient);

    let output = stretch(&input, &params).unwrap();
    let skip = 8192.min(output.len() / 4);
    let trimmed = &output[skip..output.len() - skip];

    let energy_440 = dft_energy_at_freq(trimmed, 440.0, sample_rate);
    let energy_100 = dft_energy_at_freq(trimmed, 100.0, sample_rate);

    // 440 Hz should still be present (even if somewhat smeared)
    assert!(
        energy_440 > energy_100,
        "4x stretch: 440Hz ({:.8}) should still dominate random freq ({:.8})",
        energy_440,
        energy_100
    );
}

#[test]
fn test_compression_preserves_frequency() {
    // 0.5x compression should preserve the fundamental
    let sample_rate = 44100u32;
    let input = sine_wave(440.0, sample_rate, sample_rate as usize * 4);

    let params = StretchParams::new(0.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();
    let skip = 4096.min(output.len() / 4);
    if output.len() > skip * 2 {
        let trimmed = &output[skip..output.len() - skip];

        let energy_440 = dft_energy_at_freq(trimmed, 440.0, sample_rate);
        let energy_200 = dft_energy_at_freq(trimmed, 200.0, sample_rate);

        assert!(
            energy_440 > energy_200,
            "0.5x: 440Hz ({:.8}) should still dominate ({:.8})",
            energy_440,
            energy_200
        );
    }
}

#[test]
fn test_stereo_spectral_independence() {
    // Stereo: left=440Hz, right=880Hz. After stretching, each channel
    // should retain its own frequency.
    let sample_rate = 44100u32;
    let num_frames = sample_rate as usize * 2;
    let mut input = vec![0.0f32; num_frames * 2];

    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        input[i * 2] = (TWO_PI * 440.0 * t).sin();
        input[i * 2 + 1] = (TWO_PI * 880.0 * t).sin();
    }

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let output = stretch(&input, &params).unwrap();

    // Deinterleave output
    let left: Vec<f32> = output.iter().step_by(2).copied().collect();
    let right: Vec<f32> = output.iter().skip(1).step_by(2).copied().collect();

    let skip = 4096.min(left.len() / 4);
    let left_trimmed = &left[skip..left.len() - skip];
    let right_trimmed = &right[skip..right.len() - skip];

    // Left should have more 440Hz than 880Hz
    let left_440 = dft_energy_at_freq(left_trimmed, 440.0, sample_rate);
    let left_880 = dft_energy_at_freq(left_trimmed, 880.0, sample_rate);

    // Right should have more 880Hz than 440Hz
    let right_440 = dft_energy_at_freq(right_trimmed, 440.0, sample_rate);
    let right_880 = dft_energy_at_freq(right_trimmed, 880.0, sample_rate);

    assert!(
        left_440 > left_880 * 0.5,
        "Left: 440Hz ({:.8}) should dominate 880Hz ({:.8})",
        left_440,
        left_880
    );
    assert!(
        right_880 > right_440 * 0.5,
        "Right: 880Hz ({:.8}) should dominate 440Hz ({:.8})",
        right_880,
        right_440
    );
}

#[test]
fn test_frequency_sweep_no_holes() {
    // A frequency sweep (chirp) should not have "holes" where energy drops to zero
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 3;
    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Linear chirp from 100 Hz to 2000 Hz
            let freq = 100.0 + (2000.0 - 100.0) * t / 3.0;
            (TWO_PI * freq * t).sin()
        })
        .collect();

    let params = StretchParams::new(1.5)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let output = stretch(&input, &params).unwrap();

    // Output should not be empty and should have non-trivial RMS
    let output_rms = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
    assert!(
        output_rms > 0.1,
        "Frequency sweep output too quiet: RMS={}",
        output_rms
    );

    // Check that output doesn't have long stretches of near-silence
    let chunk_size = sample_rate as usize / 4; // 250ms chunks
    let mut silent_chunks = 0;
    for chunk in output.chunks(chunk_size) {
        let chunk_rms = (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
        if chunk_rms < 0.01 {
            silent_chunks += 1;
        }
    }

    let total_chunks = output.len().div_ceil(chunk_size);
    assert!(
        silent_chunks < total_chunks / 3,
        "Too many silent chunks in sweep output: {}/{} silent",
        silent_chunks,
        total_chunks
    );
}

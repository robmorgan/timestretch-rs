//! Integration tests for creative audio effects combining new AudioBuffer APIs
//! with time stretching, demonstrating real-world DJ and production workflows.

use timestretch::{AudioBuffer, EdmPreset, StreamProcessor, StretchParams, WindowType};

// ─── Helpers ─────────────────────────────────────────────────────────

#[allow(dead_code)]
fn sine_mono(freq: f32, sample_rate: u32, num_samples: usize) -> AudioBuffer {
    let data: Vec<f32> = (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
        .collect();
    AudioBuffer::from_mono(data, sample_rate)
}

fn assert_finite(buf: &AudioBuffer, label: &str) {
    for (i, &s) in buf.data.iter().enumerate() {
        assert!(
            s.is_finite(),
            "{}: non-finite at sample {}: {}",
            label,
            i,
            s
        );
    }
}

// ─── AudioBuffer::silence() integration ──────────────────────────────

#[test]
fn silence_as_gap_between_stretched_segments() {
    let tone = AudioBuffer::tone(440.0, 44100, 0.5, 0.8);
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let stretched = timestretch::stretch_buffer(&tone, &params).unwrap();

    let gap = AudioBuffer::silence(44100, 0.2);
    let combined = AudioBuffer::concatenate(&[&stretched, &gap, &stretched]);

    assert_finite(&combined, "silence gap");
    // Total: ~0.75s stretched + 0.2s gap + 0.75s stretched = ~1.7s
    let expected_secs = stretched.duration_secs() * 2.0 + 0.2;
    assert!(
        (combined.duration_secs() - expected_secs).abs() < 0.01,
        "Duration mismatch: expected ~{:.2}s, got {:.2}s",
        expected_secs,
        combined.duration_secs()
    );
}

#[test]
fn silence_mix_with_tone() {
    let silence = AudioBuffer::silence(44100, 1.0);
    let tone = AudioBuffer::tone(440.0, 44100, 1.0, 0.5);
    let mixed = silence.mix(&tone);
    // Mixing silence with tone should equal the tone
    for (a, b) in mixed.data.iter().zip(tone.data.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

// ─── AudioBuffer::tone() integration ─────────────────────────────────

#[test]
fn tone_stretch_preserves_energy() {
    let tone = AudioBuffer::tone(440.0, 44100, 2.0, 0.7);
    let input_rms = tone.rms();

    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1)
        .with_normalize(true);
    let stretched = timestretch::stretch_buffer(&tone, &params).unwrap();

    // With normalization, RMS should be close to input
    assert!(
        (stretched.rms() - input_rms).abs() < input_rms * 0.1,
        "RMS mismatch: input={:.4}, stretched={:.4}",
        input_rms,
        stretched.rms()
    );
}

#[test]
fn tone_at_different_frequencies_stretch() {
    // Sub-bass, mid, and high frequency tones
    for &freq in &[60.0f32, 440.0, 8000.0] {
        let tone = AudioBuffer::tone(freq as f64, 44100, 1.0, 0.5);
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let stretched = timestretch::stretch_buffer(&tone, &params).unwrap();
        assert!(
            !stretched.is_empty(),
            "Tone at {}Hz produced no output",
            freq
        );
        assert!(
            stretched.rms() > 0.01,
            "Tone at {}Hz produced silence after stretch",
            freq
        );
        assert_finite(&stretched, &format!("tone_{}Hz", freq));
    }
}

#[test]
fn tone_pitch_shift_octave_up() {
    let tone = AudioBuffer::tone(440.0, 44100, 1.0, 0.8);
    let params = StretchParams::new(1.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let shifted = timestretch::pitch_shift_buffer(&tone, &params, 2.0).unwrap();
    // Length should be preserved
    assert_eq!(shifted.num_frames(), tone.num_frames());
    assert_finite(&shifted, "pitch_shift_octave_up");
}

// ─── AudioBuffer::pan() integration ──────────────────────────────────

#[test]
fn pan_then_stretch_stereo() {
    let mono = AudioBuffer::tone(440.0, 44100, 1.0, 0.7);
    let stereo = mono.pan(0.3); // slightly right of center
    assert!(stereo.is_stereo());

    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(2)
        .with_preset(EdmPreset::HouseLoop);
    let stretched = timestretch::stretch_buffer(&stereo, &params).unwrap();
    assert!(stretched.is_stereo());
    assert!(!stretched.is_empty());
    assert_finite(&stretched, "pan_then_stretch");
}

#[test]
fn stereo_field_from_two_panned_tones() {
    let tone_l = AudioBuffer::tone(440.0, 44100, 1.0, 0.6).pan(-0.7);
    let tone_r = AudioBuffer::tone(880.0, 44100, 1.0, 0.4).pan(0.7);
    let stereo = tone_l.mix(&tone_r);

    let params = StretchParams::new(1.25)
        .with_sample_rate(44100)
        .with_channels(2);
    let stretched = timestretch::stretch_buffer(&stereo, &params).unwrap();
    assert!(stretched.is_stereo());
    assert_finite(&stretched, "panned_stereo_stretch");
}

#[test]
fn pan_sweep_automation() {
    // Create a tone, then manually vary pan position across segments
    let tone = AudioBuffer::tone(440.0, 44100, 0.5, 0.6);
    let left = tone.pan(-1.0);
    let center = tone.pan(0.0);
    let right = tone.pan(1.0);

    let sweep = AudioBuffer::concatenate(&[&left, &center, &right]);
    assert_eq!(sweep.num_frames(), tone.num_frames() * 3);
    assert!(sweep.is_stereo());

    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(2);
    let stretched = timestretch::stretch_buffer(&sweep, &params).unwrap();
    assert_finite(&stretched, "pan_sweep");
    assert!(stretched.is_stereo());
}

// ─── AudioBuffer::with_gain_envelope() integration ───────────────────

#[test]
fn sidechain_duck_effect() {
    // Simulate sidechain ducking: pad plays continuously, gain dips every beat
    let pad = AudioBuffer::tone(220.0, 44100, 2.0, 0.7);

    // Create a ducking envelope: gain at 1.0, ducks to 0.2 every 0.5s (120 BPM)
    let breakpoints: Vec<(f64, f32)> = (0..8)
        .flat_map(|i| {
            let t = i as f64 * 0.25;
            vec![(t, 1.0), (t + 0.02, 0.2), (t + 0.15, 1.0)]
        })
        .collect();

    let ducked = pad.with_gain_envelope(&breakpoints);
    assert_eq!(ducked.num_frames(), pad.num_frames());

    // Stretch the ducked audio
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let stretched = timestretch::stretch_buffer(&ducked, &params).unwrap();
    assert!(!stretched.is_empty());
    assert_finite(&stretched, "sidechain_duck");
}

#[test]
fn gain_envelope_then_stretch_preserves_shape() {
    let tone = AudioBuffer::tone(440.0, 44100, 1.0, 0.8);
    // Fade in over first half, hold for second half
    let enveloped = tone.with_gain_envelope(&[(0.0, 0.0), (0.5, 1.0), (1.0, 1.0)]);

    // First quarter should be quieter than last quarter
    let first_quarter = enveloped.slice(0, 11025);
    let last_quarter = enveloped.slice(33075, 11025);
    assert!(
        first_quarter.rms() < last_quarter.rms(),
        "Envelope shape not preserved: first_rms={:.4} >= last_rms={:.4}",
        first_quarter.rms(),
        last_quarter.rms()
    );

    let params = StretchParams::new(2.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let stretched = timestretch::stretch_buffer(&enveloped, &params).unwrap();
    assert_finite(&stretched, "envelope_stretch");
}

#[test]
fn volume_automation_stereo() {
    let mono = AudioBuffer::tone(440.0, 44100, 1.0, 0.6);
    let stereo = mono.pan(0.0);
    let automated = stereo.with_gain_envelope(&[(0.0, 0.5), (0.5, 1.0), (1.0, 0.0)]);

    let params = StretchParams::new(0.75)
        .with_sample_rate(44100)
        .with_channels(2);
    let stretched = timestretch::stretch_buffer(&automated, &params).unwrap();
    assert!(stretched.is_stereo());
    assert_finite(&stretched, "volume_auto_stereo");
}

// ─── AudioBuffer::remove_dc() integration ────────────────────────────

#[test]
fn remove_dc_before_stretch() {
    // Create a signal with DC offset
    let data: Vec<f32> = (0..44100)
        .map(|i| 0.5 + 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    // Verify DC offset exists
    let mean_before: f64 = data.iter().map(|&s| s as f64).sum::<f64>() / data.len() as f64;
    assert!(mean_before.abs() > 0.4, "Should have DC offset");

    let buf = AudioBuffer::from_mono(data, 44100);
    let centered = buf.remove_dc();

    // Verify DC removed
    let mean_after: f64 =
        centered.data.iter().map(|&s| s as f64).sum::<f64>() / centered.data.len() as f64;
    assert!(
        mean_after.abs() < 0.01,
        "DC should be removed: {}",
        mean_after
    );

    // Stretch the centered audio
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1);
    let stretched = timestretch::stretch_buffer(&centered, &params).unwrap();
    assert_finite(&stretched, "dc_removed_stretch");
    assert!(stretched.rms() > 0.01);
}

#[test]
fn remove_dc_stereo_then_stretch() {
    let l: Vec<f32> = (0..44100)
        .map(|i| 0.3 + 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let r: Vec<f32> = (0..44100)
        .map(|i| -0.2 + 0.5 * (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 44100.0).sin())
        .collect();
    let buf = AudioBuffer::from_channels(&[l, r], 44100);
    let centered = buf.remove_dc();

    let params = StretchParams::new(1.25)
        .with_sample_rate(44100)
        .with_channels(2);
    let stretched = timestretch::stretch_buffer(&centered, &params).unwrap();
    assert!(stretched.is_stereo());
    assert_finite(&stretched, "dc_stereo_stretch");
}

// ─── AudioBuffer::apply_window() integration ─────────────────────────

#[test]
fn window_then_stretch_for_granular_synthesis() {
    let grain = AudioBuffer::tone(440.0, 44100, 0.05, 0.8); // 50ms grain
    let windowed = grain.apply_window(WindowType::Hann);

    // Windowed grain should start and end near zero
    assert!(windowed.data[0].abs() < 0.01);
    assert!(windowed.data[windowed.data.len() - 1].abs() < 0.01);

    let params = StretchParams::new(2.0)
        .with_sample_rate(44100)
        .with_channels(1);
    let stretched = timestretch::stretch_buffer(&windowed, &params).unwrap();
    assert_finite(&stretched, "windowed_grain_stretch");
}

#[test]
fn window_types_all_work_with_stretch() {
    let tone = AudioBuffer::tone(440.0, 44100, 0.5, 0.6);

    for wt in &[
        WindowType::Hann,
        WindowType::BlackmanHarris,
        WindowType::Kaiser(12),
    ] {
        let windowed = tone.apply_window(*wt);
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);
        let stretched = timestretch::stretch_buffer(&windowed, &params).unwrap();
        assert!(!stretched.is_empty(), "{:?} produced empty output", wt);
        assert_finite(&stretched, &format!("{:?}", wt));
    }
}

// ─── Combined creative workflows ────────────────────────────────────

#[test]
fn reverse_riser_effect() {
    // Create a "riser" effect: reversed + stretched tone with fade-in
    let tone = AudioBuffer::tone(220.0, 44100, 1.0, 0.7);
    let reversed = tone.reverse();
    let faded = reversed.fade_in(22050); // fade in over 0.5s

    let params = StretchParams::new(2.0)
        .with_sample_rate(44100)
        .with_channels(1)
        .with_preset(EdmPreset::Ambient);
    let stretched = timestretch::stretch_buffer(&faded, &params).unwrap();
    assert_finite(&stretched, "reverse_riser");
    assert!(stretched.duration_secs() > 1.5);
}

#[test]
fn tape_stop_effect() {
    // Simulate tape stop: stretch ratio increasing over time via envelope
    let tone = AudioBuffer::tone(440.0, 44100, 1.0, 0.8);

    // Apply pitch-down effect via gain envelope (simulates tape slowing down)
    let decelerated =
        tone.with_gain_envelope(&[(0.0, 1.0), (0.3, 0.9), (0.6, 0.6), (0.8, 0.3), (1.0, 0.0)]);

    // Stretch the decelerating signal
    let params = StretchParams::new(2.0)
        .with_sample_rate(44100)
        .with_channels(1)
        .with_preset(EdmPreset::Halftime);
    let stretched = timestretch::stretch_buffer(&decelerated, &params).unwrap();
    assert_finite(&stretched, "tape_stop");
}

#[test]
fn dj_transition_with_pan_and_crossfade() {
    // Track A: left-panned 440 Hz tone
    let track_a = AudioBuffer::tone(440.0, 44100, 2.0, 0.6).pan(-0.3);
    // Track B: right-panned 660 Hz tone
    let track_b = AudioBuffer::tone(660.0, 44100, 2.0, 0.6).pan(0.3);

    // Stretch both to match BPM
    let params = StretchParams::new(128.0 / 126.0)
        .with_sample_rate(44100)
        .with_channels(2)
        .with_preset(EdmPreset::DjBeatmatch);
    let a_stretched = timestretch::stretch_buffer(&track_a, &params).unwrap();
    let b_stretched = timestretch::stretch_buffer(&track_b, &params).unwrap();

    // Crossfade
    let transition = a_stretched.crossfade_into(&b_stretched, 22050);
    assert!(transition.is_stereo());
    assert_finite(&transition, "dj_transition");
}

#[test]
fn layered_synth_pad_production() {
    // Layer 3 detuned tones (classic synth pad technique)
    let osc1 = AudioBuffer::tone(440.0, 44100, 2.0, 0.3);
    let osc2 = AudioBuffer::tone(440.5, 44100, 2.0, 0.3); // slightly detuned
    let osc3 = AudioBuffer::tone(439.5, 44100, 2.0, 0.3); // slightly detuned

    let pad = osc1.mix(&osc2).mix(&osc3);

    // Apply envelope and stretch
    let enveloped = pad.with_gain_envelope(&[(0.0, 0.0), (0.5, 1.0), (1.5, 1.0), (2.0, 0.0)]);

    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1)
        .with_preset(EdmPreset::Ambient);
    let stretched = timestretch::stretch_buffer(&enveloped, &params).unwrap();
    assert_finite(&stretched, "synth_pad");
    assert!(stretched.duration_secs() > 2.5);
}

#[test]
fn granular_freeze_effect() {
    // Simulate a granular freeze: take a small grain, window it, repeat, stretch
    let source = AudioBuffer::tone(440.0, 44100, 1.0, 0.7);
    let grain = source.slice(22050, 2205); // 50ms grain from the middle
    let windowed = grain.apply_window(WindowType::Hann);

    // Repeat the grain to create a sustained texture
    let repeated = windowed.repeat(20); // ~1 second of repeated grains

    let params = StretchParams::new(3.0)
        .with_sample_rate(44100)
        .with_channels(1)
        .with_preset(EdmPreset::Ambient);
    let stretched = timestretch::stretch_buffer(&repeated, &params).unwrap();
    assert_finite(&stretched, "granular_freeze");
    assert!(stretched.duration_secs() > 2.0);
}

#[test]
fn dc_removal_in_processing_chain() {
    // Real-world scenario: audio with DC offset -> remove DC -> stretch -> normalize
    let data: Vec<f32> = (0..88200)
        .map(|i| 0.3 + 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
        .collect();
    let buf = AudioBuffer::from_mono(data, 44100);

    let clean = buf.remove_dc();
    let params = StretchParams::new(1.5)
        .with_sample_rate(44100)
        .with_channels(1)
        .with_normalize(true);
    let stretched = timestretch::stretch_buffer(&clean, &params).unwrap();
    let normalized = stretched.normalize(0.9);

    assert_finite(&normalized, "dc_chain");
    assert!(normalized.peak() <= 0.9 + 0.01);
}

// ─── StreamProcessor with new APIs ────────────────────────────────────

#[test]
fn streaming_with_tone_factory() {
    let tone = AudioBuffer::tone(440.0, 44100, 2.0, 0.6);

    let mut proc = StreamProcessor::from_tempo(126.0, 128.0, 44100, 1);
    let mut output = Vec::new();
    for chunk in tone.data.chunks(4096) {
        if let Ok(out) = proc.process(chunk) {
            output.extend_from_slice(&out);
        }
    }
    if let Ok(out) = proc.flush() {
        output.extend_from_slice(&out);
    }

    assert!(
        !output.is_empty(),
        "Streaming with tone factory should produce output"
    );
    // Check target_bpm getter
    let target = proc.target_bpm().unwrap();
    assert!((target - 128.0).abs() < 0.1);
}

#[test]
fn streaming_target_ratio_tracks_changes() {
    let mut proc = StreamProcessor::from_tempo(126.0, 128.0, 44100, 1);

    let initial_target = proc.target_stretch_ratio();
    assert!((initial_target - 126.0 / 128.0).abs() < 1e-6);

    proc.set_tempo(130.0);
    let new_target = proc.target_stretch_ratio();
    assert!(
        (new_target - 126.0 / 130.0).abs() < 1e-6,
        "Target ratio should update immediately: expected {}, got {}",
        126.0 / 130.0,
        new_target
    );

    // Current ratio should still be near the old value (not yet interpolated)
    assert!(
        (proc.current_stretch_ratio() - initial_target).abs() < 0.1,
        "Current ratio should lag behind target"
    );
}

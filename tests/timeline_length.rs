use std::f32::consts::PI;
use timestretch::{stretch, EdmPreset, StretchParams};

fn generate_long_form_signal(sample_rate: u32, duration_secs: f64, bpm: f64) -> Vec<f32> {
    let total_samples = (sample_rate as f64 * duration_secs) as usize;
    let beat_interval = (60.0 * sample_rate as f64 / bpm) as usize;
    let mut out = vec![0.0f32; total_samples];

    for (i, sample) in out.iter_mut().enumerate().take(total_samples) {
        let t = i as f32 / sample_rate as f32;
        // Tonal bed
        *sample += 0.20 * (2.0 * PI * 55.0 * t).sin();
        *sample += 0.15 * (2.0 * PI * 220.0 * t).sin();
        *sample += 0.10 * (2.0 * PI * 440.0 * t).sin();

        // Beat transient every quarter note
        let beat_pos = i % beat_interval;
        if beat_pos < (sample_rate as usize / 100) {
            // ~10ms pulse with exponential decay
            let x = beat_pos as f32 / sample_rate as f32;
            let env = (-x * 120.0).exp();
            *sample += 0.7 * env;
        }
    }

    out
}

fn length_error_pct(actual: usize, expected: usize) -> f64 {
    if expected == 0 {
        return 0.0;
    }
    (actual.abs_diff(expected) as f64 / expected as f64) * 100.0
}

#[test]
fn test_batch_duration_error_under_point_one_percent_long_form() {
    let sample_rate = 44100u32;
    let input = generate_long_form_signal(sample_rate, 12.0, 124.0);

    let cases = [
        (126.0 / 128.0, EdmPreset::DjBeatmatch),
        (1.5, EdmPreset::HouseLoop),
    ];

    for (ratio, preset) in cases {
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(preset)
            .with_bpm(124.0);

        let output = stretch(&input, &params).expect("stretch should succeed");
        let expected = params.output_length(input.len());
        let err_pct = length_error_pct(output.len(), expected);

        assert!(
            err_pct <= 0.1,
            "ratio {:.4} preset {:?}: duration error {:.6}% > 0.1% (actual={}, expected={})",
            ratio,
            preset,
            err_pct,
            output.len(),
            expected
        );
    }
}

#[test]
fn test_batch_no_multi_second_tempo_drift() {
    let sample_rate = 44100u32;
    let source_bpm = 126.0;
    let target_bpm = 128.0;
    let ratio = source_bpm / target_bpm;

    let input = generate_long_form_signal(sample_rate, 20.0, source_bpm);
    let params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_bpm(source_bpm);

    let output = stretch(&input, &params).expect("stretch should succeed");

    let expected_frames = params.output_length(input.len());
    let drift_frames = output.len().abs_diff(expected_frames);
    let drift_secs = drift_frames as f64 / sample_rate as f64;

    assert!(
        drift_secs < 1.0,
        "tempo drift too large: {:.4}s (actual={}, expected={})",
        drift_secs,
        output.len(),
        expected_frames
    );
}

use std::f32::consts::PI;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use timestretch::{AudioBuffer, Channels, StretchParams};

fn sine_mono(freq: f32, sample_rate: u32, num_samples: usize) -> AudioBuffer {
    let data: Vec<f32> = (0..num_samples)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
        .collect();
    AudioBuffer::from_mono(data, sample_rate)
}

fn sine_stereo(freq_l: f32, freq_r: f32, sample_rate: u32, num_frames: usize) -> AudioBuffer {
    let mut data = Vec::with_capacity(num_frames * 2);
    for i in 0..num_frames {
        let t = i as f32 / sample_rate as f32;
        data.push((2.0 * PI * freq_l * t).sin());
        data.push((2.0 * PI * freq_r * t).sin());
    }
    AudioBuffer::from_stereo(data, sample_rate)
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "timestretch_{label}_{}_{}",
        std::process::id(),
        stamp
    ))
}

#[test]
fn stereo_bpm_wav_pipeline_preserves_channel_layout() {
    let dir = unique_temp_dir("stereo_bpm_pipeline");
    std::fs::create_dir_all(&dir).unwrap();

    let input_path = dir.join("input.wav");
    let output_path = dir.join("output.wav");
    let input = sine_stereo(440.0, 880.0, 44_100, 88_200);
    timestretch::io::wav::write_wav_file_float(input_path.to_str().unwrap(), &input).unwrap();

    let params = StretchParams::new(1.0)
        .with_sample_rate(44_100)
        .with_channels(2);

    let result = timestretch::stretch_to_bpm_wav_file(
        input_path.to_str().unwrap(),
        output_path.to_str().unwrap(),
        128.0,
        120.0,
        &params,
    )
    .unwrap();

    let expected_ratio = 128.0 / 120.0;
    let actual_ratio = result.duration_secs() / input.duration_secs();
    let reloaded = timestretch::io::wav::read_wav_file(output_path.to_str().unwrap()).unwrap();

    assert_eq!(result.channels, Channels::Stereo);
    assert_eq!(result.sample_rate, 44_100);
    assert_eq!(reloaded.channels, Channels::Stereo);
    assert!(output_path.exists());
    assert!(
        (actual_ratio - expected_ratio).abs() < 0.15,
        "expected stereo BPM workflow ratio ~{expected_ratio}, got {actual_ratio}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn resample_stretch_resample_chain_preserves_rate_and_duration() {
    let source = sine_mono(440.0, 48_000, 48_000);
    let at_44k = source.resample(44_100);
    let params = StretchParams::new(1.5)
        .with_sample_rate(44_100)
        .with_channels(1);

    let stretched = timestretch::stretch_buffer(&at_44k, &params).unwrap();
    let playback = stretched.resample(48_000);

    assert_eq!(playback.sample_rate, 48_000);
    assert!(playback.rms() > 0.05);
    assert!(
        (playback.duration_secs() - 1.5).abs() < 0.15,
        "resample/stretch chain duration drifted to {}s",
        playback.duration_secs()
    );
}

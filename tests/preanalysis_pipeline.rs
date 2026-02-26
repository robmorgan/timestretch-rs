use std::f32::consts::PI;
use std::path::PathBuf;
use timestretch::{
    analyze_for_dj, read_preanalysis_json, stretch, write_preanalysis_json, EdmPreset,
    PreAnalysisArtifact, StretchParams,
};

fn click_train(sample_rate: u32, bpm: f64, seconds: f64) -> Vec<f32> {
    let len = (sample_rate as f64 * seconds) as usize;
    let beat_interval = (60.0 * sample_rate as f64 / bpm) as usize;
    let mut out = vec![0.0f32; len];
    for i in (0..len).step_by(beat_interval.max(1)) {
        for j in 0..10.min(len - i) {
            out[i + j] = if j < 5 { 1.0 } else { -0.4 };
        }
    }
    for (i, s) in out.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;
        *s += 0.15 * (2.0 * PI * 110.0 * t).sin();
    }
    out
}

#[test]
fn test_preanalysis_roundtrip_json() {
    let artifact = PreAnalysisArtifact {
        sample_rate: 44100,
        bpm: 128.0,
        downbeat_offset_samples: 100,
        confidence: 0.9,
        beat_positions: vec![0, 22050, 44100],
        transient_onsets: vec![0, 22050, 44100],
    };

    let path = PathBuf::from("target/test_preanalysis_roundtrip.json");
    write_preanalysis_json(&path, &artifact).expect("write should succeed");
    let read_back = read_preanalysis_json(&path).expect("read should succeed");

    assert_eq!(read_back.sample_rate, artifact.sample_rate);
    assert_eq!(read_back.bpm, artifact.bpm);
    assert_eq!(
        read_back.downbeat_offset_samples,
        artifact.downbeat_offset_samples
    );
    assert_eq!(read_back.beat_positions, artifact.beat_positions);
}

#[test]
fn test_runtime_uses_confident_preanalysis_when_bpm_missing() {
    let sample_rate = 44100u32;
    let input = click_train(sample_rate, 128.0, 4.0);
    let artifact = analyze_for_dj(&input, sample_rate);

    let ratio = 1.1;
    let params_with_artifact = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_pre_analysis(artifact.clone())
        .with_beat_snap_confidence_threshold(0.1);

    let params_with_bpm = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_bpm(artifact.bpm);

    let out_artifact =
        stretch(&input, &params_with_artifact).expect("artifact stretch should work");
    let out_bpm = stretch(&input, &params_with_bpm).expect("bpm stretch should work");

    assert!(!out_artifact.is_empty());
    assert!(!out_bpm.is_empty());

    let len_ratio = out_artifact.len() as f64 / out_bpm.len().max(1) as f64;
    assert!(
        (len_ratio - 1.0).abs() < 0.05,
        "Artifact-driven runtime should be close to explicit BPM runtime (ratio={})",
        len_ratio
    );
}

#[test]
fn test_runtime_fallback_when_preanalysis_unavailable() {
    let sample_rate = 44100u32;
    let input = click_train(sample_rate, 124.0, 3.0);

    let unavailable_artifact = PreAnalysisArtifact {
        sample_rate: 48000, // mismatched on purpose
        bpm: 140.0,
        downbeat_offset_samples: 0,
        confidence: 0.95,
        beat_positions: vec![0, 1000, 2000],
        transient_onsets: vec![0, 1000, 2000],
    };

    let ratio = 1.02;
    let params_with_bad_artifact = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_pre_analysis(unavailable_artifact)
        .with_beat_snap_confidence_threshold(0.5);

    let params_fallback = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let out_bad = stretch(&input, &params_with_bad_artifact).expect("fallback stretch should work");
    let out_base = stretch(&input, &params_fallback).expect("base stretch should work");

    assert!(!out_bad.is_empty());
    assert!(!out_base.is_empty());
    let len_diff = out_bad.len().abs_diff(out_base.len()) as f64;
    let len_diff_pct = len_diff / out_base.len().max(1) as f64;
    assert!(
        len_diff_pct < 0.05,
        "Fallback path with unavailable artifact should stay close to live path (diff={:.3}%)",
        len_diff_pct * 100.0
    );
}

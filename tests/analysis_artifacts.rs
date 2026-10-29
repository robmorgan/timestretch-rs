#![cfg(all(feature = "analysis", feature = "io"))]
use std::f32::consts::PI;
use std::path::PathBuf;
use timestretch::analyze;
// Deprecated JSON pair, still round-trip-tested until removal.
#[allow(deprecated)]
use timestretch::{read_preanalysis_json, write_preanalysis_json};

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

/// The deprecated JSON path must keep round-tripping until it's removed.
#[test]
#[allow(deprecated)]
fn test_preanalysis_roundtrip_json() {
    let sample_rate = 44100u32;
    let input = click_train(sample_rate, 128.0, 4.0);
    let artifact = analyze(&input, sample_rate);

    let path = PathBuf::from("target/test_preanalysis_roundtrip.json");
    write_preanalysis_json(&path, &artifact).expect("write should succeed");
    let read_back = read_preanalysis_json(&path).expect("read should succeed");

    assert_eq!(read_back.version, artifact.version);
    assert_eq!(read_back.sample_rate, artifact.sample_rate);
    assert_eq!(read_back.bpm, artifact.bpm);
    assert_eq!(
        read_back.downbeat_offset_samples,
        artifact.downbeat_offset_samples
    );
    assert_eq!(read_back.beat_positions, artifact.beat_positions);
    assert_eq!(read_back.transient_onsets, artifact.transient_onsets);
    assert_eq!(read_back.transient_strengths, artifact.transient_strengths);
    assert_eq!(read_back.onset_band_flux, artifact.onset_band_flux);
    assert_eq!(read_back.analysis_hop_size, artifact.analysis_hop_size);
    assert_eq!(read_back.source_len_samples, artifact.source_len_samples);
    assert_eq!(read_back.content_hash, artifact.content_hash);
    assert!(read_back.matches_source(&input, sample_rate));
}

#[test]
fn test_analysis_file_roundtrip_tsa() {
    let sample_rate = 44100u32;
    let input = click_train(sample_rate, 128.0, 4.0);
    let artifact = analyze(&input, sample_rate);

    let mut analysis = timestretch::AnalysisFile::for_source(&input, sample_rate);
    analysis.artifact = Some(artifact.clone());
    analysis.peaks = Some(timestretch::BandPeaks::compute(&input, 1, sample_rate));
    let path = PathBuf::from("target/test_preanalysis_roundtrip.tsa");
    timestretch::write_analysis_file(&path, &analysis).expect("write should succeed");

    let back = timestretch::read_analysis_file_validated(
        &path,
        sample_rate,
        analysis.source_len_samples,
        analysis.content_hash,
    )
    .expect("validated read should succeed against the real identity");
    let read_back = back.artifact.expect("artifact chunk present");
    assert_eq!(read_back.bpm, artifact.bpm);
    assert_eq!(read_back.beat_positions, artifact.beat_positions);
    assert!(read_back.matches_source(&input, sample_rate));
    assert!(back.peaks.is_some(), "peaks chunk present");
}

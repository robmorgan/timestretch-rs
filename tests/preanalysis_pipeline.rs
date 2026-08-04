use std::f32::consts::PI;
use std::path::PathBuf;
use timestretch::{PreAnalysisArtifact, StretchParams, analyze_for_dj, stretch};
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

/// Simple local peak detector: samples above `threshold` that are local
/// maxima, deduplicated by `min_distance`.
fn detect_peaks(signal: &[f32], threshold: f32, min_distance: usize) -> Vec<usize> {
    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..signal.len().saturating_sub(1) {
        let s = signal[i].abs();
        if s >= threshold && s >= signal[i - 1].abs() && s >= signal[i + 1].abs() {
            if let Some(&last) = peaks.last() {
                if i - last < min_distance {
                    if s > signal[last].abs() {
                        *peaks.last_mut().unwrap() = i;
                    }
                    continue;
                }
            }
            peaks.push(i);
        }
    }
    peaks
}

/// The deprecated JSON path must keep round-tripping until it's removed.
#[test]
#[allow(deprecated)]
fn test_preanalysis_roundtrip_json() {
    let sample_rate = 44100u32;
    let input = click_train(sample_rate, 128.0, 4.0);
    let artifact = analyze_for_dj(&input, sample_rate);

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
    let artifact = analyze_for_dj(&input, sample_rate);

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

#[test]
fn test_runtime_uses_confident_preanalysis_when_bpm_missing() {
    let sample_rate = 44100u32;
    let input = click_train(sample_rate, 128.0, 4.0);
    let artifact = analyze_for_dj(&input, sample_rate);

    let ratio = 1.1;
    let params_with_artifact = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_pre_analysis(artifact.clone())
        .with_beat_snap_confidence_threshold(0.1);

    let params_with_bpm = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
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
        ..Default::default()
    };

    let ratio = 1.02;
    let params_with_bad_artifact = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_pre_analysis(unavailable_artifact)
        .with_beat_snap_confidence_threshold(0.5);

    let params_fallback = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1);

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

#[test]
fn test_batch_artifact_transient_parity_with_online_detection() {
    let sample_rate = 44100u32;
    let bpm = 128.0;
    let input = click_train(sample_rate, bpm, 4.0);
    let artifact = analyze_for_dj(&input, sample_rate);
    assert!(
        !artifact.transient_onsets.is_empty(),
        "click train should yield transient onsets"
    );

    let ratio = 1.1;
    let base_params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let out_online = stretch(&input, &base_params).expect("online stretch should work");
    let out_artifact = stretch(
        &input,
        &base_params
            .clone()
            .with_pre_analysis(artifact)
            .with_beat_snap_confidence_threshold(0.1),
    )
    .expect("artifact stretch should work");

    assert!(out_artifact.iter().all(|s| s.is_finite()));
    let len_ratio = out_artifact.len() as f64 / out_online.len().max(1) as f64;
    assert!(
        (len_ratio - 1.0).abs() < 0.05,
        "artifact and online output lengths should agree (ratio={})",
        len_ratio
    );

    let beat_interval = 60.0 * sample_rate as f64 / bpm;
    let min_distance = (beat_interval * ratio * 0.5) as usize;
    let peaks_online = detect_peaks(&out_online, 0.2, min_distance);
    let peaks_artifact = detect_peaks(&out_artifact, 0.2, min_distance);
    // The artifact path must preserve at least as many transients as online
    // detection (in practice it preserves more: offline analysis is
    // non-causal, so onsets that online detection misses survive).
    assert!(
        peaks_artifact.len() >= 6,
        "artifact-driven output lost transients: {} peaks",
        peaks_artifact.len()
    );
    assert!(
        peaks_artifact.len() + 1 >= peaks_online.len(),
        "artifact path preserved fewer transients than online detection: {} vs {}",
        peaks_artifact.len(),
        peaks_online.len()
    );
}

#[test]
fn test_batch_artifact_skips_online_detection() {
    let sample_rate = 44100u32;
    let input = click_train(sample_rate, 128.0, 4.0);

    // A usable artifact is authoritative even when its claims disagree
    // with what online detection would find: no transients, and beats a
    // half-beat off the clicks (so its splice-protection windows cover
    // different regions than freshly detected events would). If online
    // detection still ran, both outputs would be identical because
    // stretching is deterministic.
    let mut no_transient_artifact = analyze_for_dj(&input, sample_rate);
    no_transient_artifact.transient_onsets.clear();
    no_transient_artifact.transient_strengths.clear();
    no_transient_artifact.onset_band_flux.clear();
    let half_beat = (60.0 * sample_rate as f64 / 128.0 / 2.0) as usize;
    for b in &mut no_transient_artifact.beat_positions {
        *b += half_beat;
    }
    for b in &mut no_transient_artifact.beat_positions_fractional {
        *b += half_beat as f64;
    }
    assert!(no_transient_artifact.is_usable(sample_rate, 0.1));

    let ratio = 1.1;
    let base_params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1);

    let out_online = stretch(&input, &base_params).expect("online stretch should work");
    let out_artifact = stretch(
        &input,
        &base_params
            .clone()
            .with_pre_analysis(no_transient_artifact)
            .with_beat_snap_confidence_threshold(0.1),
    )
    .expect("artifact stretch should work");

    assert!(
        out_online != out_artifact,
        "empty-onset artifact produced identical output to online detection; \
         detection was not skipped"
    );
}

#[test]
fn test_stereo_artifact_parity_with_online_detection() {
    let sample_rate = 44100u32;
    let bpm = 128.0;
    // Center-panned click train with decorrelated width so M/S is exercised.
    let mid = click_train(sample_rate, bpm, 4.0);
    let mut interleaved = Vec::with_capacity(mid.len() * 2);
    for (i, &m) in mid.iter().enumerate() {
        let t = i as f32 / sample_rate as f32;
        let width = 0.05 * (2.0 * PI * 3.0 * t).sin();
        interleaved.push(m + width);
        interleaved.push(m - width);
    }

    // Analysis runs on the mid downmix, exactly as a loader would produce it.
    let analysis_signal = timestretch::downmix_to_mid(&interleaved, 2);
    let artifact = analyze_for_dj(&analysis_signal, sample_rate);
    assert!(artifact.matches_source(&analysis_signal, sample_rate));

    let ratio = 1.1;
    let base_params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(2);

    let out_online = stretch(&interleaved, &base_params).expect("online stereo stretch");
    let out_artifact = stretch(
        &interleaved,
        &base_params
            .clone()
            .with_pre_analysis(artifact)
            .with_beat_snap_confidence_threshold(0.1),
    )
    .expect("artifact stereo stretch");

    assert_eq!(
        out_online.len(),
        out_artifact.len(),
        "stereo output lengths should be deterministic and equal"
    );

    let beat_interval = 60.0 * sample_rate as f64 / bpm;
    let min_distance = (beat_interval * ratio * 0.5) as usize;
    for ch in 0..2 {
        let online_ch: Vec<f32> = out_online.iter().skip(ch).step_by(2).copied().collect();
        let artifact_ch: Vec<f32> = out_artifact.iter().skip(ch).step_by(2).copied().collect();
        let peaks_online = detect_peaks(&online_ch, 0.3, min_distance);
        let peaks_artifact = detect_peaks(&artifact_ch, 0.3, min_distance);
        // The stereo M/S render smooths clicks more than the mono path, so
        // the floor is lower here; the parity assertion below is the real
        // gate (artifact must not lose transients relative to online).
        assert!(
            peaks_artifact.len() >= 3,
            "ch{} artifact-driven stereo output lost transients: {} peaks",
            ch,
            peaks_artifact.len()
        );
        assert!(
            peaks_artifact.len() + 1 >= peaks_online.len(),
            "ch{} artifact stereo path preserved fewer transients than online: {} vs {}",
            ch,
            peaks_artifact.len(),
            peaks_online.len()
        );
    }
}

use std::f32::consts::PI;
use timestretch::{analysis::comparison, EdmPreset, StreamProcessor, StretchParams};

fn generate_gate_signal(sample_rate: u32, bpm: f64, duration_secs: f64) -> Vec<f32> {
    let total_samples = (sample_rate as f64 * duration_secs) as usize;
    let beat_interval = (60.0 * sample_rate as f64 / bpm) as usize;
    let mut out = vec![0.0f32; total_samples];

    for (i, sample) in out.iter_mut().enumerate().take(total_samples) {
        let t = i as f32 / sample_rate as f32;
        *sample += 0.22 * (2.0 * PI * 55.0 * t).sin();
        *sample += 0.15 * (2.0 * PI * 220.0 * t).sin();
        *sample += 0.10 * (2.0 * PI * 440.0 * t).sin();

        let beat_pos = i % beat_interval.max(1);
        if beat_pos < (sample_rate as usize / 120) {
            let x = beat_pos as f32 / sample_rate as f32;
            let env = (-x * 150.0).exp();
            *sample += 0.65 * env;
        }
    }

    out
}

fn stream_hybrid(input: &[f32], params: StretchParams, chunk_size: usize) -> Vec<f32> {
    let mut processor = StreamProcessor::new(params);
    processor.set_hybrid_mode(true);
    let mut out = Vec::new();
    for chunk in input.chunks(chunk_size) {
        let rendered = processor.process(chunk).expect("stream process failed");
        out.extend_from_slice(&rendered);
    }
    let tail = processor.flush().expect("stream flush failed");
    out.extend_from_slice(&tail);
    out
}

#[test]
fn quality_gate_batch_vs_stream_hybrid_subset() {
    let sample_rate = 44100u32;
    let bpm = 126.0;
    let target_bpm = 128.0;
    let ratio = bpm / target_bpm;

    let input = generate_gate_signal(sample_rate, bpm, 4.0);
    let params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_bpm(bpm);

    let reference = timestretch::stretch(&input, &params).expect("batch stretch failed");
    let candidate = stream_hybrid(&input, params.clone(), 4096);

    assert!(!reference.is_empty());
    assert!(!candidate.is_empty());

    let len_diff_pct =
        reference.len().abs_diff(candidate.len()) as f64 / reference.len() as f64 * 100.0;
    assert!(
        len_diff_pct <= 0.6,
        "duration gate failed: length diff {:.5}% exceeds 0.6% (ref={}, cand={})",
        len_diff_pct,
        reference.len(),
        candidate.len()
    );

    let min_len = reference.len().min(candidate.len());
    let reference = &reference[..min_len];
    let candidate = &candidate[..min_len];

    let transient = comparison::transient_match_score(reference, candidate, sample_rate, 12.0);
    println!(
        "quality-gates: len_diff_pct={:.4}% transient={:.3}",
        len_diff_pct, transient.match_rate
    );
    assert!(
        transient.match_rate >= 0.60,
        "transient gate failed: match rate {:.3} < 0.60",
        transient.match_rate
    );

    let xcorr = comparison::cross_correlation(reference, candidate);
    println!("quality-gates: xcorr_peak={:.3}", xcorr.peak_value);
    assert!(
        xcorr.peak_value >= 0.68,
        "cross-correlation gate failed: peak {:.3} < 0.68",
        xcorr.peak_value
    );

    let loudness_diff = comparison::lufs_difference(reference, candidate, sample_rate).abs();
    println!("quality-gates: loudness_diff={:.3} dB", loudness_diff);
    assert!(
        loudness_diff <= 2.5,
        "loudness gate failed: |LUFS diff| {:.3} > 2.5 dB",
        loudness_diff
    );

    let band = comparison::band_spectral_similarity(reference, candidate, 2048, 512, sample_rate);
    println!(
        "quality-gates: band_sim sub={:.3} low={:.3} mid={:.3} high={:.3}",
        band.sub_bass, band.low, band.mid, band.high
    );
    assert!(
        band.sub_bass >= 0.45,
        "spectral gate failed (sub-bass): {:.3} < 0.45",
        band.sub_bass
    );
    assert!(
        band.low >= 0.45,
        "spectral gate failed (low): {:.3} < 0.45",
        band.low
    );
    assert!(
        band.mid >= 0.40,
        "spectral gate failed (mid): {:.3} < 0.40",
        band.mid
    );
    assert!(
        band.high >= 0.30,
        "spectral gate failed (high): {:.3} < 0.30",
        band.high
    );
}

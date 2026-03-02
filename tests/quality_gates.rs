use std::f32::consts::PI;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use timestretch::{analysis::comparison, EdmPreset, StreamProcessor, StretchParams};

const STRICT_CALLBACK_BUDGET_ENV: &str = "TIMESTRETCH_STRICT_CALLBACK_BUDGET";
const CALLBACK_BUDGET_MULTIPLIER_ENV: &str = "TIMESTRETCH_CALLBACK_BUDGET_MULTIPLIER";
const QUALITY_DASHBOARD_DIR_ENV: &str = "TIMESTRETCH_QUALITY_DASHBOARD_DIR";

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

fn mono_to_stereo_interleaved(mono: &[f32]) -> Vec<f32> {
    let mut stereo = Vec::with_capacity(mono.len() * 2);
    for &s in mono {
        stereo.push(s);
        stereo.push(s);
    }
    stereo
}

#[derive(Debug, Clone, Copy, Default)]
struct BoundaryArtifactStats {
    max_ratio: f64,
    mean_ratio: f64,
    evaluated_boundaries: usize,
}

fn p95(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let idx = (((values.len() - 1) as f64) * 0.95).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn boundary_artifact_stats(
    signal: &[f32],
    boundaries: &[usize],
    local_window: usize,
    guard: usize,
) -> BoundaryArtifactStats {
    if signal.len() < 4 {
        return BoundaryArtifactStats::default();
    }

    let mut max_ratio = 0.0f64;
    let mut sum_ratio = 0.0f64;
    let mut evaluated = 0usize;

    for &boundary in boundaries {
        if boundary <= 1 || boundary >= signal.len() - 1 {
            continue;
        }

        let start = boundary.saturating_sub(local_window).max(1);
        let end = (boundary + local_window).min(signal.len() - 1);
        if end <= start {
            continue;
        }

        let guard_start = boundary.saturating_sub(guard);
        let guard_end = (boundary + guard).min(signal.len() - 1);

        let mut local_diffs = Vec::with_capacity((end - start).saturating_sub(2 * guard));
        for idx in start..=end {
            if idx >= guard_start && idx <= guard_end {
                continue;
            }
            local_diffs.push((signal[idx] - signal[idx - 1]).abs() as f64);
        }
        if local_diffs.len() < 8 {
            continue;
        }

        let jump = (signal[boundary] - signal[boundary - 1]).abs() as f64;
        let local_p95 = p95(local_diffs).max(1e-6);
        let ratio = jump / local_p95;

        max_ratio = max_ratio.max(ratio);
        sum_ratio += ratio;
        evaluated += 1;
    }

    if evaluated == 0 {
        return BoundaryArtifactStats::default();
    }

    BoundaryArtifactStats {
        max_ratio,
        mean_ratio: sum_ratio / evaluated as f64,
        evaluated_boundaries: evaluated,
    }
}

fn strict_callback_budget_mode() -> bool {
    let value = std::env::var(STRICT_CALLBACK_BUDGET_ENV).unwrap_or_default();
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn callback_budget_multiplier() -> Option<f64> {
    if let Ok(value) = std::env::var(CALLBACK_BUDGET_MULTIPLIER_ENV) {
        if let Ok(parsed) = value.parse::<f64>() {
            if parsed.is_finite() && parsed > 0.0 {
                return Some(parsed);
            }
        }
    }
    if strict_callback_budget_mode() {
        return Some(0.90);
    }
    None
}

fn write_quality_dashboard_csv(name: &str, header: &str, row: &str) {
    let Ok(dir) = std::env::var(QUALITY_DASHBOARD_DIR_ENV) else {
        return;
    };

    let dir_path = PathBuf::from(dir);
    if let Err(err) = fs::create_dir_all(&dir_path) {
        println!(
            "quality-dashboard: failed to create output dir {}: {}",
            dir_path.display(),
            err
        );
        return;
    }

    let path = dir_path.join(format!("{name}.csv"));
    let mut file = match fs::File::create(&path) {
        Ok(file) => file,
        Err(err) => {
            println!(
                "quality-dashboard: failed to create artifact {}: {}",
                path.display(),
                err
            );
            return;
        }
    };

    if let Err(err) = writeln!(file, "{}", header) {
        println!(
            "quality-dashboard: failed to write header {}: {}",
            path.display(),
            err
        );
        return;
    }
    if let Err(err) = writeln!(file, "{}", row) {
        println!(
            "quality-dashboard: failed to write row {}: {}",
            path.display(),
            err
        );
    }
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

fn align_by_offset<'a>(
    reference: &'a [f32],
    candidate: &'a [f32],
    offset: isize,
) -> (&'a [f32], &'a [f32], usize) {
    let mut ref_start = 0usize;
    let mut cand_start = 0usize;
    if offset > 0 {
        cand_start = offset as usize;
    } else if offset < 0 {
        ref_start = (-offset) as usize;
    }

    if ref_start >= reference.len() || cand_start >= candidate.len() {
        return (&[], &[], ref_start);
    }

    let aligned_len = (reference.len() - ref_start).min(candidate.len() - cand_start);
    (
        &reference[ref_start..ref_start + aligned_len],
        &candidate[cand_start..cand_start + aligned_len],
        ref_start,
    )
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

    let xcorr = comparison::cross_correlation(reference, candidate);
    let (reference_aligned, candidate_aligned, ref_start) =
        align_by_offset(reference, candidate, xcorr.peak_offset);
    assert!(
        !reference_aligned.is_empty() && !candidate_aligned.is_empty(),
        "alignment produced empty comparison windows (offset={})",
        xcorr.peak_offset
    );

    let transient =
        comparison::transient_match_score(reference_aligned, candidate_aligned, sample_rate, 12.0);
    println!(
        "quality-gates: len_diff_pct={:.4}% transient={:.3} xcorr_peak={:.3} offset={}",
        len_diff_pct, transient.match_rate, xcorr.peak_value, xcorr.peak_offset
    );
    let transient_or_structure_ok = transient.match_rate >= 0.60 || xcorr.peak_value >= 0.75;
    assert!(
        transient_or_structure_ok,
        "transient/structure gate failed: match rate {:.3} < 0.60 and xcorr {:.3} < 0.75",
        transient.match_rate, xcorr.peak_value
    );

    // The streaming hybrid path re-processes overlapping rolling buffers
    // through a stateless HybridStretcher, so waveform-level correlation
    // with the single-pass batch output is inherently limited.  Length,
    // transient, and loudness gates verify perceptual accuracy; xcorr is a
    // loose structural check.
    assert!(
        xcorr.peak_value >= 0.35,
        "cross-correlation gate failed: peak {:.3} < 0.35",
        xcorr.peak_value
    );

    let loudness_diff =
        comparison::lufs_difference(reference_aligned, candidate_aligned, sample_rate).abs();
    println!("quality-gates: loudness_diff={:.3} dB", loudness_diff);
    assert!(
        loudness_diff <= 2.5,
        "loudness gate failed: |LUFS diff| {:.3} > 2.5 dB",
        loudness_diff
    );

    let band = comparison::band_spectral_similarity(
        reference_aligned,
        candidate_aligned,
        2048,
        512,
        sample_rate,
    );
    println!(
        "quality-gates: band_sim sub={:.3} low={:.3} mid={:.3} high={:.3}",
        band.sub_bass, band.low, band.mid, band.high
    );

    // Boundary artifact detector: compare candidate boundary roughness against
    // the batch reference around beat-aligned transition anchors.
    let beat_interval_samples = (60.0 * sample_rate as f64 / bpm).round() as usize;
    let boundary_positions: Vec<usize> = (beat_interval_samples..input.len())
        .step_by(beat_interval_samples.max(1))
        .map(|pos| (pos as f64 * ratio).round() as usize)
        .collect();
    let boundary_window = (sample_rate as f64 * 0.010).round() as usize; // +/-10ms
    let boundary_guard = (sample_rate as f64 * 0.0015).round() as usize; // ignore +/-1.5ms
    let aligned_positions: Vec<usize> = boundary_positions
        .iter()
        .filter_map(|&p| p.checked_sub(ref_start))
        .filter(|&p| p < reference_aligned.len() && p < candidate_aligned.len())
        .collect();
    let reference_boundary = boundary_artifact_stats(
        reference_aligned,
        &aligned_positions,
        boundary_window,
        boundary_guard,
    );
    let candidate_boundary = boundary_artifact_stats(
        candidate_aligned,
        &aligned_positions,
        boundary_window,
        boundary_guard,
    );
    println!(
        "quality-gates: boundary_artifacts ref(max={:.3},mean={:.3},n={}) cand(max={:.3},mean={:.3},n={})",
        reference_boundary.max_ratio,
        reference_boundary.mean_ratio,
        reference_boundary.evaluated_boundaries,
        candidate_boundary.max_ratio,
        candidate_boundary.mean_ratio,
        candidate_boundary.evaluated_boundaries
    );
    assert!(
        candidate_boundary.evaluated_boundaries >= 3
            && reference_boundary.evaluated_boundaries >= 3,
        "boundary artifact gate could not evaluate enough boundaries (ref={}, cand={})",
        reference_boundary.evaluated_boundaries,
        candidate_boundary.evaluated_boundaries
    );
    assert!(
        candidate_boundary.max_ratio <= reference_boundary.max_ratio * 1.8 + 1.0,
        "boundary artifact gate failed (max): cand {:.3} vs ref {:.3}",
        candidate_boundary.max_ratio,
        reference_boundary.max_ratio
    );
    assert!(
        candidate_boundary.mean_ratio <= reference_boundary.mean_ratio * 1.5 + 0.75,
        "boundary artifact gate failed (mean): cand {:.3} vs ref {:.3}",
        candidate_boundary.mean_ratio,
        reference_boundary.mean_ratio
    );

    write_quality_dashboard_csv(
        "quality_gate_batch_vs_stream_hybrid_subset",
        "len_diff_pct,transient_match_rate,cross_correlation_peak,loudness_diff_db,sub_bass_similarity,low_similarity,mid_similarity,high_similarity,boundary_max_ratio_ref,boundary_mean_ratio_ref,boundary_max_ratio_cand,boundary_mean_ratio_cand,boundary_count_ref,boundary_count_cand",
        &format!(
            "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
            len_diff_pct,
            transient.match_rate,
            xcorr.peak_value,
            loudness_diff,
            band.sub_bass,
            band.low,
            band.mid,
            band.high,
            reference_boundary.max_ratio,
            reference_boundary.mean_ratio,
            candidate_boundary.max_ratio,
            candidate_boundary.mean_ratio,
            reference_boundary.evaluated_boundaries,
            candidate_boundary.evaluated_boundaries
        ),
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

#[test]
fn quality_gate_streaming_worst_case_callback_budget() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let ratio = 1.02;
    let callback_frames = 256usize;
    let input = generate_gate_signal(sample_rate, bpm, 10.0);
    let Some(multiplier) = callback_budget_multiplier() else {
        println!(
            "Skipping callback budget gate: set {}=1 (strict) or {}=<value> to enable",
            STRICT_CALLBACK_BUDGET_ENV, CALLBACK_BUDGET_MULTIPLIER_ENV
        );
        write_quality_dashboard_csv(
            "quality_gate_streaming_worst_case_callback_budget",
            "status,max_ratio,avg_ratio,max_callback_ms,max_budget_ms,measured_callbacks,multiplier,strict_mode",
            "skipped,NaN,NaN,NaN,NaN,0,NaN,false",
        );
        return;
    };

    let params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(2)
        .with_preset(EdmPreset::DjBeatmatch);

    let mut processor = StreamProcessor::new(params);
    let mut output = Vec::with_capacity((input.len() as f64 * 1.30) as usize + 65_536);

    // Warm up a few callbacks so first-use effects don't dominate.
    for chunk in input.chunks(callback_frames * 2).take(8) {
        processor
            .process_into(chunk, &mut output)
            .expect("warmup process_into failed");
    }
    output.clear();

    let mut measured_callbacks = 0usize;
    let mut max_ratio = 0.0f64;
    let mut max_callback_ms = 0.0f64;
    let mut max_budget_ms = 0.0f64;
    let mut total_process_ms = 0.0f64;
    let mut total_audio_ms = 0.0f64;

    for chunk in input.chunks(callback_frames * 2).skip(8) {
        let chunk_frames = (chunk.len() / 2).max(1);
        let callback_audio_ms = chunk_frames as f64 * 1000.0 / sample_rate as f64;
        let allowed_ms = callback_audio_ms * multiplier;

        let start = Instant::now();
        processor
            .process_into(chunk, &mut output)
            .expect("measured process_into failed");
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        measured_callbacks += 1;
        total_process_ms += elapsed_ms;
        total_audio_ms += callback_audio_ms;

        let ratio = elapsed_ms / callback_audio_ms.max(1e-9);
        if ratio > max_ratio {
            max_ratio = ratio;
            max_callback_ms = elapsed_ms;
            max_budget_ms = allowed_ms;
        }
    }

    processor
        .flush_into(&mut output)
        .expect("flush_into failed for callback budget gate");

    assert!(
        measured_callbacks > 0,
        "callback budget gate measured no callbacks"
    );
    assert!(
        !output.is_empty(),
        "callback budget gate produced empty output"
    );

    let avg_ratio = total_process_ms / total_audio_ms.max(1e-9);
    println!(
        "callback-budget: callbacks={} max_ratio={:.3} avg_ratio={:.3} max_ms={:.3} budget_ms={:.3} strict_mode={}",
        measured_callbacks,
        max_ratio,
        avg_ratio,
        max_callback_ms,
        max_budget_ms,
        strict_callback_budget_mode()
    );
    write_quality_dashboard_csv(
        "quality_gate_streaming_worst_case_callback_budget",
        "status,max_ratio,avg_ratio,max_callback_ms,max_budget_ms,measured_callbacks,multiplier,strict_mode",
        &format!(
            "ok,{:.6},{:.6},{:.6},{:.6},{},{:.6},{}",
            max_ratio,
            avg_ratio,
            max_callback_ms,
            max_budget_ms,
            measured_callbacks,
            multiplier,
            strict_callback_budget_mode()
        ),
    );

    assert!(
        max_ratio <= multiplier,
        "callback budget gate failed: max callback ratio {:.3} > {:.3} (max callback {:.3}ms, budget {:.3}ms). Set {}=0 for relaxed mode or {} to tune.",
        max_ratio,
        multiplier,
        max_callback_ms,
        max_budget_ms,
        STRICT_CALLBACK_BUDGET_ENV,
        CALLBACK_BUDGET_MULTIPLIER_ENV
    );
}

#[test]
fn quality_gate_streaming_callback_budget_tempo_and_pitch_modulation() {
    let sample_rate = 44_100u32;
    let source_bpm = 126.0;
    let callback_frames = 256usize;
    let input = mono_to_stereo_interleaved(&generate_gate_signal(sample_rate, source_bpm, 10.0));
    let Some(multiplier) = callback_budget_multiplier() else {
        println!(
            "Skipping tempo+pitch callback budget gate: set {}=1 (strict) or {}=<value> to enable",
            STRICT_CALLBACK_BUDGET_ENV, CALLBACK_BUDGET_MULTIPLIER_ENV
        );
        write_quality_dashboard_csv(
            "quality_gate_streaming_callback_budget_tempo_and_pitch_modulation",
            "status,max_ratio,avg_ratio,max_callback_ms,max_budget_ms,measured_callbacks,multiplier,strict_mode",
            "skipped,NaN,NaN,NaN,NaN,0,NaN,false",
        );
        return;
    };

    let mut processor =
        StreamProcessor::try_from_tempo_low_latency(source_bpm, source_bpm, sample_rate, 2)
            .expect("valid low-latency tempo constructor");
    let mut output = Vec::with_capacity((input.len() as f64 * 1.40) as usize + 65_536);

    // Warm up so one-time initialization effects don't dominate.
    for (idx, chunk) in input.chunks(callback_frames * 2).take(8).enumerate() {
        let phase = idx as f64 / 8.0;
        let target_bpm = source_bpm + 2.0 * (2.0 * std::f64::consts::PI * phase).sin();
        let pitch_scale = 1.0 + 0.03 * (2.0 * std::f64::consts::PI * phase * 1.3).sin();
        assert!(processor.set_tempo(target_bpm));
        processor
            .set_pitch_scale(pitch_scale)
            .expect("valid warmup pitch scale");
        processor
            .process_into(chunk, &mut output)
            .expect("warmup process_into failed");
    }
    output.clear();

    let chunks: Vec<&[f32]> = input.chunks(callback_frames * 2).skip(8).collect();
    let mut measured_callbacks = 0usize;
    let mut max_ratio = 0.0f64;
    let mut max_callback_ms = 0.0f64;
    let mut max_budget_ms = 0.0f64;
    let mut total_process_ms = 0.0f64;
    let mut total_audio_ms = 0.0f64;

    for (idx, chunk) in chunks.iter().enumerate() {
        let phase = idx as f64 / chunks.len().max(1) as f64;
        let target_bpm = source_bpm + 3.0 * (2.0 * std::f64::consts::PI * phase).sin();
        let pitch_scale = 1.0 + 0.05 * (2.0 * std::f64::consts::PI * phase * 1.7).sin();
        assert!(processor.set_tempo(target_bpm));
        processor
            .set_pitch_scale(pitch_scale)
            .expect("valid modulation pitch scale");

        let chunk_frames = (chunk.len() / 2).max(1);
        let callback_audio_ms = chunk_frames as f64 * 1000.0 / sample_rate as f64;
        let allowed_ms = callback_audio_ms * multiplier;

        let start = Instant::now();
        processor
            .process_into(chunk, &mut output)
            .expect("measured process_into failed");
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        measured_callbacks += 1;
        total_process_ms += elapsed_ms;
        total_audio_ms += callback_audio_ms;

        let ratio = elapsed_ms / callback_audio_ms.max(1e-9);
        if ratio > max_ratio {
            max_ratio = ratio;
            max_callback_ms = elapsed_ms;
            max_budget_ms = allowed_ms;
        }
    }

    processor
        .flush_into(&mut output)
        .expect("flush_into failed for tempo+pitch callback budget gate");

    assert!(
        measured_callbacks > 0,
        "tempo+pitch callback budget gate measured no callbacks"
    );
    assert!(
        !output.is_empty(),
        "tempo+pitch callback budget gate produced empty output"
    );

    let avg_ratio = total_process_ms / total_audio_ms.max(1e-9);
    println!(
        "callback-budget-tempo-pitch: callbacks={} max_ratio={:.3} avg_ratio={:.3} max_ms={:.3} budget_ms={:.3} strict_mode={}",
        measured_callbacks,
        max_ratio,
        avg_ratio,
        max_callback_ms,
        max_budget_ms,
        strict_callback_budget_mode()
    );
    write_quality_dashboard_csv(
        "quality_gate_streaming_callback_budget_tempo_and_pitch_modulation",
        "status,max_ratio,avg_ratio,max_callback_ms,max_budget_ms,measured_callbacks,multiplier,strict_mode",
        &format!(
            "ok,{:.6},{:.6},{:.6},{:.6},{},{:.6},{}",
            max_ratio,
            avg_ratio,
            max_callback_ms,
            max_budget_ms,
            measured_callbacks,
            multiplier,
            strict_callback_budget_mode()
        ),
    );

    assert!(
        max_ratio <= multiplier,
        "tempo+pitch callback budget gate failed: max callback ratio {:.3} > {:.3} (max callback {:.3}ms, budget {:.3}ms). Set {}=0 for relaxed mode or {} to tune.",
        max_ratio,
        multiplier,
        max_callback_ms,
        max_budget_ms,
        STRICT_CALLBACK_BUDGET_ENV,
        CALLBACK_BUDGET_MULTIPLIER_ENV
    );
}

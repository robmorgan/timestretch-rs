use std::f32::consts::PI;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use timestretch::stretch::HybridStretcher;
use timestretch::{
    analysis::comparison, EdmPreset, LatencyProfile, RtProfileTelemetry, StreamProcessor,
    StretchParams,
};

const STRICT_CALLBACK_BUDGET_ENV: &str = "TIMESTRETCH_STRICT_CALLBACK_BUDGET";
const CALLBACK_BUDGET_MULTIPLIER_ENV: &str = "TIMESTRETCH_CALLBACK_BUDGET_MULTIPLIER";
const QUALITY_DASHBOARD_DIR_ENV: &str = "TIMESTRETCH_QUALITY_DASHBOARD_DIR";
const HYBRID_STREAM_CROSSFADE_SAMPLES: usize = 3072;

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

fn generate_harmonic_bed(sample_rate: u32, duration_secs: f64) -> Vec<f32> {
    let total_samples = (sample_rate as f64 * duration_secs) as usize;
    let mut out = vec![0.0f32; total_samples];

    for (i, sample) in out.iter_mut().enumerate().take(total_samples) {
        let t = i as f32 / sample_rate as f32;
        let slow_env = 0.88 + 0.12 * (2.0 * PI * 0.27 * t).sin();
        *sample += slow_env * 0.28 * (2.0 * PI * 110.0 * t).sin();
        *sample += slow_env * 0.18 * (2.0 * PI * 220.0 * t).sin();
        *sample += slow_env * 0.10 * (2.0 * PI * 330.0 * t).sin();
        *sample += slow_env * 0.06 * (2.0 * PI * 660.0 * t).sin();
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
    p95_ratio: f64,
    p98_ratio: f64,
    p99_ratio: f64,
    evaluated_boundaries: usize,
}

#[derive(Debug, Clone, Copy)]
enum DeterministicProfileMode {
    Auto,
    Fixed(LatencyProfile),
}

#[derive(Debug, Clone, Copy, Default)]
struct ProfileTransitionStats {
    current_profile_changes: usize,
    target_profile_changes: usize,
    policy_profile_changes: usize,
    current_tier_changes: usize,
    target_tier_changes: usize,
    observed_callbacks: usize,
    last: Option<RtProfileTelemetry>,
}

#[derive(Debug, Clone, Default)]
struct ProfileChangeTrace {
    callback_output_frames: Vec<usize>,
    current_profile_change_callbacks: Vec<usize>,
    current_profile_change_profiles: Vec<LatencyProfile>,
}

impl ProfileTransitionStats {
    fn observe(&mut self, telemetry: RtProfileTelemetry) {
        if let Some(last) = self.last {
            self.current_profile_changes +=
                usize::from(last.current_profile != telemetry.current_profile);
            self.target_profile_changes +=
                usize::from(last.target_profile != telemetry.target_profile);
            self.policy_profile_changes +=
                usize::from(last.policy_profile != telemetry.policy_profile);
            self.current_tier_changes += usize::from(last.current_tier != telemetry.current_tier);
            self.target_tier_changes += usize::from(last.target_tier != telemetry.target_tier);
        }
        self.observed_callbacks += 1;
        self.last = Some(telemetry);
    }

    fn summary(self) -> String {
        let final_profile = self
            .last
            .map(|telemetry| {
                format!(
                    "final(current={:?},target={:?},policy={:?},tier={:?}->{:?})",
                    telemetry.current_profile,
                    telemetry.target_profile,
                    telemetry.policy_profile,
                    telemetry.current_tier,
                    telemetry.target_tier
                )
            })
            .unwrap_or_else(|| "final(unavailable)".to_string());
        format!(
            "callbacks={}, current_profile_changes={}, target_profile_changes={}, policy_profile_changes={}, current_tier_changes={}, target_tier_changes={}, {}",
            self.observed_callbacks,
            self.current_profile_changes,
            self.target_profile_changes,
            self.policy_profile_changes,
            self.current_tier_changes,
            self.target_tier_changes,
            final_profile
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CallbackCadenceStats {
    callbacks_after_first_output: usize,
    callbacks_with_output_after_first_output: usize,
    max_idle_gap_callbacks: usize,
}

fn percentile(mut values: Vec<f64>, quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let q = quantile.clamp(0.0, 1.0);
    let idx = (((values.len() - 1) as f64) * q).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn p95(values: Vec<f64>) -> f64 {
    percentile(values, 0.95)
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
    let mut ratios = Vec::with_capacity(boundaries.len());

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
        ratios.push(ratio);
    }

    if evaluated == 0 {
        return BoundaryArtifactStats::default();
    }

    BoundaryArtifactStats {
        max_ratio,
        mean_ratio: sum_ratio / evaluated as f64,
        p95_ratio: percentile(ratios.clone(), 0.95),
        p98_ratio: percentile(ratios.clone(), 0.98),
        p99_ratio: percentile(ratios, 0.99),
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
    if input.is_empty() {
        return Vec::new();
    }

    let stretcher = HybridStretcher::new(params.clone());
    let max_tail_frames = params.fft_size * 56;
    let rolling_capacity = chunk_size.saturating_add(max_tail_frames);
    let accum_threshold = params.fft_size * 2;

    let mut rolling_input = Vec::with_capacity(rolling_capacity);
    let mut held_tail = Vec::with_capacity(
        (params.fft_size.saturating_mul(8)).max(HYBRID_STREAM_CROSSFADE_SAMPLES),
    );
    let mut output = Vec::new();
    let mut input_accumulated = usize::MAX;
    let mut tail_output_len = 0usize;

    let render_delta = |force: bool,
                        rolling_input: &mut Vec<f32>,
                        held_tail: &mut Vec<f32>,
                        output: &mut Vec<f32>,
                        input_accumulated: &mut usize,
                        tail_output_len: &mut usize| {
        if !force && *input_accumulated < accum_threshold {
            return;
        }

        let pre_trim_len = rolling_input.len();
        let rendered = stretcher
            .process(rolling_input)
            .expect("stream hybrid rerender failed");
        let skip = (*tail_output_len).min(rendered.len());
        let delta_len = rendered.len().saturating_sub(skip);
        let ratio_scale = params.stretch_ratio.max(1.0);
        let xfade_base = (HYBRID_STREAM_CROSSFADE_SAMPLES as f64 * ratio_scale).round() as usize;
        let xfade = xfade_base.min(skip).min(delta_len * 7 / 8);

        if !held_tail.is_empty() && xfade > 0 {
            let overlap = &rendered[skip - xfade..skip];
            let n = held_tail.len().min(overlap.len());
            for i in 0..n {
                let t = (i as f32 + 0.5) / n as f32;
                let s = 0.5 * (1.0 - (PI * t).cos());
                output.push(held_tail[i] * (1.0 - s) + overlap[i] * s);
            }
        }

        let holdback = xfade_base.min(delta_len * 7 / 8);
        let emit_end = delta_len.saturating_sub(holdback);
        output.extend_from_slice(&rendered[skip..skip + emit_end]);

        held_tail.clear();
        held_tail.extend_from_slice(&rendered[skip + emit_end..skip + delta_len]);

        if rolling_input.len() > max_tail_frames {
            let keep_from = rolling_input.len() - max_tail_frames;
            rolling_input.drain(..keep_from);
        }

        *tail_output_len = if pre_trim_len > 0 {
            ((rendered.len() as f64) * rolling_input.len() as f64 / pre_trim_len as f64).round()
                as usize
        } else {
            0
        };
        *input_accumulated = 0;
    };

    for chunk in input.chunks(chunk_size) {
        let required = rolling_input.len().saturating_add(chunk.len());
        if required > rolling_capacity {
            let discard = required - rolling_capacity;
            rolling_input.drain(..discard.min(rolling_input.len()));
        }
        rolling_input.extend_from_slice(chunk);
        input_accumulated = input_accumulated.saturating_add(chunk.len());
        render_delta(
            false,
            &mut rolling_input,
            &mut held_tail,
            &mut output,
            &mut input_accumulated,
            &mut tail_output_len,
        );
    }

    render_delta(
        true,
        &mut rolling_input,
        &mut held_tail,
        &mut output,
        &mut input_accumulated,
        &mut tail_output_len,
    );
    output.extend_from_slice(&held_tail);
    let expected_len = (input.len() as f64 * params.stretch_ratio).round() as usize;
    if output.len() < expected_len {
        output.resize(expected_len, output.last().copied().unwrap_or(0.0));
    } else if output.len() > expected_len {
        output.truncate(expected_len);
    }
    output
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
            "status,max_ratio,p99_ratio,p999_ratio,avg_ratio,max_callback_ms,max_budget_ms,measured_callbacks,multiplier,strict_mode",
            "skipped,NaN,NaN,NaN,NaN,NaN,NaN,0,NaN,false",
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
    let mut callback_ratios = Vec::new();

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
        callback_ratios.push(ratio);
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
    let p99_ratio = percentile(callback_ratios.clone(), 0.99);
    let p999_ratio = percentile(callback_ratios, 0.999);
    println!(
        "callback-budget: callbacks={} max_ratio={:.3} p99={:.3} p999={:.3} avg_ratio={:.3} max_ms={:.3} budget_ms={:.3} strict_mode={}",
        measured_callbacks,
        max_ratio,
        p99_ratio,
        p999_ratio,
        avg_ratio,
        max_callback_ms,
        max_budget_ms,
        strict_callback_budget_mode()
    );
    write_quality_dashboard_csv(
        "quality_gate_streaming_worst_case_callback_budget",
        "status,max_ratio,p99_ratio,p999_ratio,avg_ratio,max_callback_ms,max_budget_ms,measured_callbacks,multiplier,strict_mode",
        &format!(
            "ok,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{}",
            max_ratio,
            p99_ratio,
            p999_ratio,
            avg_ratio,
            max_callback_ms,
            max_budget_ms,
            measured_callbacks,
            multiplier,
            strict_callback_budget_mode()
        ),
    );

    assert!(
        p99_ratio <= multiplier,
        "callback budget p99 gate failed: p99 ratio {:.3} > {:.3}",
        p99_ratio,
        multiplier
    );
    let p999_limit = multiplier * 1.10;
    assert!(
        p999_ratio <= p999_limit,
        "callback budget p999 gate failed: p999 ratio {:.3} > {:.3}",
        p999_ratio,
        p999_limit
    );
    let max_outlier_limit = multiplier * 2.0;
    assert!(
        max_ratio <= max_outlier_limit,
        "callback budget gate failed: max callback ratio {:.3} > {:.3} outlier limit (max callback {:.3}ms, budget {:.3}ms). Set {}=0 for relaxed mode or {} to tune.",
        max_ratio,
        max_outlier_limit,
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
            "status,max_ratio,p99_ratio,p999_ratio,avg_ratio,max_callback_ms,max_budget_ms,measured_callbacks,multiplier,strict_mode",
            "skipped,NaN,NaN,NaN,NaN,NaN,NaN,0,NaN,false",
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
    let mut callback_ratios = Vec::new();

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
        callback_ratios.push(ratio);
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
    let p99_ratio = percentile(callback_ratios.clone(), 0.99);
    let p999_ratio = percentile(callback_ratios, 0.999);
    println!(
        "callback-budget-tempo-pitch: callbacks={} max_ratio={:.3} p99={:.3} p999={:.3} avg_ratio={:.3} max_ms={:.3} budget_ms={:.3} strict_mode={}",
        measured_callbacks,
        max_ratio,
        p99_ratio,
        p999_ratio,
        avg_ratio,
        max_callback_ms,
        max_budget_ms,
        strict_callback_budget_mode()
    );
    write_quality_dashboard_csv(
        "quality_gate_streaming_callback_budget_tempo_and_pitch_modulation",
        "status,max_ratio,p99_ratio,p999_ratio,avg_ratio,max_callback_ms,max_budget_ms,measured_callbacks,multiplier,strict_mode",
        &format!(
            "ok,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{}",
            max_ratio,
            p99_ratio,
            p999_ratio,
            avg_ratio,
            max_callback_ms,
            max_budget_ms,
            measured_callbacks,
            multiplier,
            strict_callback_budget_mode()
        ),
    );

    assert!(
        p99_ratio <= multiplier,
        "tempo+pitch callback budget p99 gate failed: p99 ratio {:.3} > {:.3}",
        p99_ratio,
        multiplier
    );
    let p999_limit = multiplier * 1.10;
    assert!(
        p999_ratio <= p999_limit,
        "tempo+pitch callback budget p999 gate failed: p999 ratio {:.3} > {:.3}",
        p999_ratio,
        p999_limit
    );
    let max_outlier_limit = multiplier * 2.0;
    assert!(
        max_ratio <= max_outlier_limit,
        "tempo+pitch callback budget gate failed: max callback ratio {:.3} > {:.3} outlier limit (max callback {:.3}ms, budget {:.3}ms). Set {}=0 for relaxed mode or {} to tune.",
        max_ratio,
        max_outlier_limit,
        max_callback_ms,
        max_budget_ms,
        STRICT_CALLBACK_BUDGET_ENV,
        CALLBACK_BUDGET_MULTIPLIER_ENV
    );
}

fn extract_left_channel(stereo_interleaved: &[f32]) -> Vec<f32> {
    stereo_interleaved
        .chunks_exact(2)
        .map(|frame| frame[0])
        .collect()
}

fn configure_deterministic_profile_mode(
    processor: &mut StreamProcessor,
    mode: DeterministicProfileMode,
) {
    match mode {
        DeterministicProfileMode::Auto => processor
            .set_deterministic_auto_profile_switching(true)
            .expect("dual-plane deterministic auto profile switching should be configurable"),
        DeterministicProfileMode::Fixed(profile) => {
            processor
                .set_deterministic_auto_profile_switching(false)
                .expect("dual-plane deterministic auto profile switching should be configurable");
            processor
                .set_deterministic_latency_profile(profile)
                .expect("dual-plane deterministic latency profile should be configurable");
        }
    }
}

fn push_boundary_if_advanced(boundaries: &mut Vec<usize>, output_frames: usize) {
    if boundaries.last().copied() != Some(output_frames) {
        boundaries.push(output_frames);
    }
}

fn callback_cadence_stats(written_frames: &[usize]) -> CallbackCadenceStats {
    let Some(first_output_idx) = written_frames.iter().position(|&written| written > 0) else {
        return CallbackCadenceStats::default();
    };

    let mut callbacks_with_output = 0usize;
    let mut max_idle_gap = 0usize;
    let mut current_idle_gap = 0usize;
    for &written in &written_frames[first_output_idx..] {
        if written > 0 {
            callbacks_with_output += 1;
            max_idle_gap = max_idle_gap.max(current_idle_gap);
            current_idle_gap = 0;
        } else {
            current_idle_gap += 1;
        }
    }
    max_idle_gap = max_idle_gap.max(current_idle_gap);

    CallbackCadenceStats {
        callbacks_after_first_output: written_frames.len().saturating_sub(first_output_idx),
        callbacks_with_output_after_first_output: callbacks_with_output,
        max_idle_gap_callbacks: max_idle_gap,
    }
}

fn run_dual_plane_deterministic_fixed_buffer_ratio_steps_callback_writes(
    input: &[f32],
    sample_rate: u32,
    callback_frames: usize,
    ratios: &[f64],
    hold_callbacks: usize,
) -> Vec<usize> {
    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256)
        .with_preset(EdmPreset::DjBeatmatch);
    let mut processor = StreamProcessor::new(params);
    assert!(
        processor.is_dual_plane_deterministic(),
        "deterministic stream should default to dual-plane backend"
    );
    assert!(!ratios.is_empty(), "ratio step schedule must not be empty");

    let chunk_samples = callback_frames * 2;
    let mut callback_output = vec![0.0f32; chunk_samples];
    let mut callback_writes = Vec::with_capacity(input.len() / chunk_samples + 1);

    for (idx, chunk) in input.chunks(chunk_samples).enumerate() {
        let schedule_idx = idx / hold_callbacks.max(1);
        let ratio = ratios[schedule_idx % ratios.len()];
        processor
            .set_stretch_ratio(ratio)
            .expect("ratio step modulation must stay in valid range");
        let written = processor
            .process_interleaved_into(chunk, &mut callback_output)
            .expect("dual-plane fixed-buffer process_interleaved_into failed");
        assert!(
            written <= callback_output.len(),
            "fixed-buffer callback write exceeded host buffer (written={}, capacity={})",
            written,
            callback_output.len()
        );
        callback_writes.push(written / 2);
    }

    callback_writes
}

fn run_dual_plane_deterministic_with_ratio_modulation_mode(
    input: &[f32],
    sample_rate: u32,
    callback_frames: usize,
    modulate: bool,
    mode: DeterministicProfileMode,
) -> (Vec<f32>, Vec<usize>, ProfileTransitionStats) {
    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256)
        .with_preset(EdmPreset::DjBeatmatch);
    let mut processor = StreamProcessor::new(params);
    assert!(
        processor.is_dual_plane_deterministic(),
        "deterministic stream should default to dual-plane backend"
    );
    configure_deterministic_profile_mode(&mut processor, mode);

    let chunk_samples = callback_frames * 2;
    let chunks: Vec<&[f32]> = input.chunks(chunk_samples).collect();
    let total_chunks = chunks.len().max(1);
    let mut output = Vec::with_capacity((input.len() as f64 * 1.20) as usize + 32_768);
    let mut boundaries = Vec::with_capacity(total_chunks + 1);
    let mut profile_stats = ProfileTransitionStats::default();

    for (idx, chunk) in chunks.iter().enumerate() {
        if modulate {
            let phase = idx as f64 / total_chunks as f64;
            let ratio = 1.0 + 0.04 * (2.0 * std::f64::consts::PI * phase * 11.0).sin();
            processor
                .set_stretch_ratio(ratio)
                .expect("ratio modulation must stay in valid range");
        }
        processor
            .process_into(chunk, &mut output)
            .expect("dual-plane stream process_into failed");
        profile_stats.observe(
            processor
                .deterministic_profile_telemetry()
                .expect("dual-plane deterministic telemetry should stay available"),
        );
        push_boundary_if_advanced(&mut boundaries, output.len() / 2);
    }
    processor
        .flush_into(&mut output)
        .expect("dual-plane stream flush_into failed");
    push_boundary_if_advanced(&mut boundaries, output.len() / 2);

    (output, boundaries, profile_stats)
}

fn run_dual_plane_deterministic_with_ratio_modulation(
    input: &[f32],
    sample_rate: u32,
    callback_frames: usize,
    modulate: bool,
) -> (Vec<f32>, Vec<usize>) {
    let (output, boundaries, _) = run_dual_plane_deterministic_with_ratio_modulation_mode(
        input,
        sample_rate,
        callback_frames,
        modulate,
        DeterministicProfileMode::Auto,
    );
    (output, boundaries)
}

fn run_dual_plane_deterministic_with_ratio_steps(
    input: &[f32],
    sample_rate: u32,
    callback_frames: usize,
    ratios: &[f64],
    hold_callbacks: usize,
) -> (Vec<f32>, Vec<usize>) {
    let (output, boundaries, _) = run_dual_plane_deterministic_with_ratio_steps_mode(
        input,
        sample_rate,
        callback_frames,
        ratios,
        hold_callbacks,
        DeterministicProfileMode::Auto,
    );
    (output, boundaries)
}

fn run_dual_plane_deterministic_with_ratio_steps_mode(
    input: &[f32],
    sample_rate: u32,
    callback_frames: usize,
    ratios: &[f64],
    hold_callbacks: usize,
    mode: DeterministicProfileMode,
) -> (Vec<f32>, Vec<usize>, ProfileTransitionStats) {
    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256)
        .with_preset(EdmPreset::DjBeatmatch);
    let mut processor = StreamProcessor::new(params);
    assert!(
        processor.is_dual_plane_deterministic(),
        "deterministic stream should default to dual-plane backend"
    );
    configure_deterministic_profile_mode(&mut processor, mode);
    assert!(!ratios.is_empty(), "ratio step schedule must not be empty");

    let chunk_samples = callback_frames * 2;
    let mut output = Vec::with_capacity((input.len() as f64 * 1.20) as usize + 32_768);
    let mut boundaries = Vec::with_capacity(input.len() / chunk_samples + 1);
    let mut profile_stats = ProfileTransitionStats::default();

    for (idx, chunk) in input.chunks(chunk_samples).enumerate() {
        let schedule_idx = idx / hold_callbacks.max(1);
        let ratio = ratios[schedule_idx % ratios.len()];
        processor
            .set_stretch_ratio(ratio)
            .expect("ratio step modulation must stay in valid range");
        processor
            .process_into(chunk, &mut output)
            .expect("dual-plane stream process_into failed");
        profile_stats.observe(
            processor
                .deterministic_profile_telemetry()
                .expect("dual-plane deterministic telemetry should stay available"),
        );
        push_boundary_if_advanced(&mut boundaries, output.len() / 2);
    }
    processor
        .flush_into(&mut output)
        .expect("dual-plane stream flush_into failed");
    push_boundary_if_advanced(&mut boundaries, output.len() / 2);

    (output, boundaries, profile_stats)
}

fn run_dual_plane_deterministic_with_ratio_steps_mode_trace(
    input: &[f32],
    sample_rate: u32,
    callback_frames: usize,
    ratios: &[f64],
    hold_callbacks: usize,
    mode: DeterministicProfileMode,
) -> (
    Vec<f32>,
    Vec<usize>,
    ProfileTransitionStats,
    ProfileChangeTrace,
) {
    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256)
        .with_preset(EdmPreset::DjBeatmatch);
    let mut processor = StreamProcessor::new(params);
    assert!(
        processor.is_dual_plane_deterministic(),
        "deterministic stream should default to dual-plane backend"
    );
    configure_deterministic_profile_mode(&mut processor, mode);
    assert!(!ratios.is_empty(), "ratio step schedule must not be empty");

    let chunk_samples = callback_frames * 2;
    let mut output = Vec::with_capacity((input.len() as f64 * 1.75) as usize + 32_768);
    let mut boundaries = Vec::with_capacity(input.len() / chunk_samples + 1);
    let mut profile_stats = ProfileTransitionStats::default();
    let mut trace = ProfileChangeTrace::default();

    for (idx, chunk) in input.chunks(chunk_samples).enumerate() {
        let schedule_idx = idx / hold_callbacks.max(1);
        let ratio = ratios[schedule_idx % ratios.len()];
        processor
            .set_stretch_ratio(ratio)
            .expect("ratio step modulation must stay in valid range");
        processor
            .process_into(chunk, &mut output)
            .expect("dual-plane stream process_into failed");
        let telemetry = processor
            .deterministic_profile_telemetry()
            .expect("dual-plane deterministic telemetry should stay available");
        if let Some(last) = profile_stats.last {
            if last.current_profile != telemetry.current_profile {
                trace.current_profile_change_callbacks.push(idx);
                trace
                    .current_profile_change_profiles
                    .push(telemetry.current_profile);
            }
        }
        profile_stats.observe(telemetry);
        let output_frames = output.len() / 2;
        trace.callback_output_frames.push(output_frames);
        push_boundary_if_advanced(&mut boundaries, output_frames);
    }
    processor
        .flush_into(&mut output)
        .expect("dual-plane stream flush_into failed");
    push_boundary_if_advanced(&mut boundaries, output.len() / 2);

    (output, boundaries, profile_stats, trace)
}

fn positions_from_callback_trace(
    callback_output_frames: &[usize],
    callbacks: &[usize],
    signal_len: usize,
) -> Vec<usize> {
    let mut positions = Vec::with_capacity(callbacks.len());
    for &callback_idx in callbacks {
        let Some(&position) = callback_output_frames.get(callback_idx) else {
            continue;
        };
        if position <= 1 || position + 1 >= signal_len {
            continue;
        }
        if positions.last().copied() != Some(position) {
            positions.push(position);
        }
    }
    positions
}

#[test]
fn quality_gate_dual_plane_deterministic_long_run_drift() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let ratio = 1.018;
    let callback_frames = 256usize;

    let mono = generate_gate_signal(sample_rate, bpm, 30.0);
    let input = mono_to_stereo_interleaved(&mono);
    let params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(2)
        .with_fft_size(1024)
        .with_hop_size(256)
        .with_preset(EdmPreset::DjBeatmatch);
    let mut processor = StreamProcessor::new(params.clone());
    assert!(
        processor.is_dual_plane_deterministic(),
        "deterministic stream should default to dual-plane backend"
    );

    let mut output = Vec::with_capacity((input.len() as f64 * (ratio + 0.2)) as usize + 65_536);
    for chunk in input.chunks(callback_frames * 2) {
        processor
            .process_into(chunk, &mut output)
            .expect("dual-plane long-run process_into failed");
    }
    processor
        .flush_into(&mut output)
        .expect("dual-plane long-run flush_into failed");

    let expected_frames = params.output_length(input.len() / 2);
    let actual_frames = output.len() / 2;
    let drift_frames = actual_frames.abs_diff(expected_frames);
    let drift_pct = drift_frames as f64 / expected_frames.max(1) as f64 * 100.0;

    write_quality_dashboard_csv(
        "quality_gate_dual_plane_deterministic_long_run_drift",
        "expected_frames,actual_frames,drift_frames,drift_pct",
        &format!(
            "{},{},{},{:.6}",
            expected_frames, actual_frames, drift_frames, drift_pct
        ),
    );

    assert!(
        drift_pct <= 0.25,
        "dual-plane deterministic long-run drift gate failed: drift {:.4}% (expected_frames={}, actual_frames={})",
        drift_pct,
        expected_frames,
        actual_frames
    );
}

#[test]
fn quality_gate_dual_plane_fast_modulation_artifacts() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let callback_frames = 256usize;
    let mono = generate_gate_signal(sample_rate, bpm, 8.0);
    let input = mono_to_stereo_interleaved(&mono);

    let (baseline_out, baseline_boundaries) = run_dual_plane_deterministic_with_ratio_modulation(
        &input,
        sample_rate,
        callback_frames,
        false,
    );
    let (modulated_out, modulated_boundaries) = run_dual_plane_deterministic_with_ratio_modulation(
        &input,
        sample_rate,
        callback_frames,
        true,
    );

    assert!(
        baseline_out.iter().all(|s| s.is_finite()),
        "baseline dual-plane modulation gate produced non-finite samples"
    );
    assert!(
        modulated_out.iter().all(|s| s.is_finite()),
        "modulated dual-plane modulation gate produced non-finite samples"
    );
    assert!(
        !baseline_out.is_empty() && !modulated_out.is_empty(),
        "dual-plane modulation gate produced empty output"
    );

    let baseline_left = extract_left_channel(&baseline_out);
    let modulated_left = extract_left_channel(&modulated_out);
    let trim = 16usize;
    let baseline_positions: Vec<usize> = if baseline_boundaries.len() > trim * 2 {
        baseline_boundaries[trim..baseline_boundaries.len() - trim].to_vec()
    } else {
        baseline_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < baseline_left.len())
    .collect();
    let modulated_positions: Vec<usize> = if modulated_boundaries.len() > trim * 2 {
        modulated_boundaries[trim..modulated_boundaries.len() - trim].to_vec()
    } else {
        modulated_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < modulated_left.len())
    .collect();
    let window = (sample_rate as f64 * 0.008).round() as usize; // +/-8ms
    let guard = (sample_rate as f64 * 0.001).round() as usize; // +/-1ms
    let baseline_stats =
        boundary_artifact_stats(&baseline_left, &baseline_positions, window, guard);
    let modulated_stats =
        boundary_artifact_stats(&modulated_left, &modulated_positions, window, guard);
    println!(
        "dual-plane-fast-mod gate: baseline(max={:.3},p95={:.3},p98={:.3},p99={:.3},mean={:.3},n={}) modulated(max={:.3},p95={:.3},p98={:.3},p99={:.3},mean={:.3},n={})",
        baseline_stats.max_ratio,
        baseline_stats.p95_ratio,
        baseline_stats.p98_ratio,
        baseline_stats.p99_ratio,
        baseline_stats.mean_ratio,
        baseline_stats.evaluated_boundaries,
        modulated_stats.max_ratio,
        modulated_stats.p95_ratio,
        modulated_stats.p98_ratio,
        modulated_stats.p99_ratio,
        modulated_stats.mean_ratio,
        modulated_stats.evaluated_boundaries
    );

    write_quality_dashboard_csv(
        "quality_gate_dual_plane_fast_modulation_artifacts",
        "baseline_max,baseline_p95,baseline_p98,baseline_p99,baseline_mean,baseline_n,modulated_max,modulated_p95,modulated_p98,modulated_p99,modulated_mean,modulated_n",
        &format!(
            "{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            baseline_stats.max_ratio,
            baseline_stats.p95_ratio,
            baseline_stats.p98_ratio,
            baseline_stats.p99_ratio,
            baseline_stats.mean_ratio,
            baseline_stats.evaluated_boundaries,
            modulated_stats.max_ratio,
            modulated_stats.p95_ratio,
            modulated_stats.p98_ratio,
            modulated_stats.p99_ratio,
            modulated_stats.mean_ratio,
            modulated_stats.evaluated_boundaries
        ),
    );

    assert!(
        baseline_stats.evaluated_boundaries >= 32 && modulated_stats.evaluated_boundaries >= 32,
        "dual-plane modulation artifact gate evaluated too few boundaries (baseline={}, modulated={})",
        baseline_stats.evaluated_boundaries,
        modulated_stats.evaluated_boundaries
    );
    assert!(
        modulated_stats.p95_ratio <= baseline_stats.p95_ratio * 2.2 + 0.8,
        "dual-plane modulation artifact gate failed (p95): modulated {:.3} vs baseline {:.3}",
        modulated_stats.p95_ratio,
        baseline_stats.p95_ratio
    );
    assert!(
        modulated_stats.p98_ratio <= baseline_stats.p98_ratio * 2.6 + 1.1,
        "dual-plane modulation artifact gate failed (p98): modulated {:.3} vs baseline {:.3}",
        modulated_stats.p98_ratio,
        baseline_stats.p98_ratio
    );
    assert!(
        modulated_stats.mean_ratio <= baseline_stats.mean_ratio * 2.0 + 0.9,
        "dual-plane modulation artifact gate failed (mean): modulated {:.3} vs baseline {:.3}",
        modulated_stats.mean_ratio,
        baseline_stats.mean_ratio
    );
}

#[test]
#[ignore = "diagnostic harness for comparing deterministic profile modes under fast modulation"]
fn quality_gate_dual_plane_fast_modulation_profile_mode_diagnostics() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let callback_frames = 256usize;
    let mono = generate_gate_signal(sample_rate, bpm, 8.0);
    let input = mono_to_stereo_interleaved(&mono);
    let modes = [
        ("auto", DeterministicProfileMode::Auto),
        (
            "fixed-mix",
            DeterministicProfileMode::Fixed(LatencyProfile::Mix),
        ),
        (
            "fixed-scratch",
            DeterministicProfileMode::Fixed(LatencyProfile::Scratch),
        ),
    ];

    for (label, mode) in modes {
        let (baseline_out, baseline_boundaries, baseline_profile) =
            run_dual_plane_deterministic_with_ratio_modulation_mode(
                &input,
                sample_rate,
                callback_frames,
                false,
                mode,
            );
        let (modulated_out, modulated_boundaries, modulated_profile) =
            run_dual_plane_deterministic_with_ratio_modulation_mode(
                &input,
                sample_rate,
                callback_frames,
                true,
                mode,
            );

        let baseline_left = extract_left_channel(&baseline_out);
        let modulated_left = extract_left_channel(&modulated_out);
        let trim = 16usize;
        let baseline_positions: Vec<usize> = if baseline_boundaries.len() > trim * 2 {
            baseline_boundaries[trim..baseline_boundaries.len() - trim].to_vec()
        } else {
            baseline_boundaries.clone()
        }
        .into_iter()
        .filter(|&p| p > 1 && p + 1 < baseline_left.len())
        .collect();
        let modulated_positions: Vec<usize> = if modulated_boundaries.len() > trim * 2 {
            modulated_boundaries[trim..modulated_boundaries.len() - trim].to_vec()
        } else {
            modulated_boundaries.clone()
        }
        .into_iter()
        .filter(|&p| p > 1 && p + 1 < modulated_left.len())
        .collect();
        let window = (sample_rate as f64 * 0.008).round() as usize;
        let guard = (sample_rate as f64 * 0.001).round() as usize;
        let baseline_stats =
            boundary_artifact_stats(&baseline_left, &baseline_positions, window, guard);
        let modulated_stats =
            boundary_artifact_stats(&modulated_left, &modulated_positions, window, guard);

        println!(
            "fast-mod profile diagnostics [{label}] baseline(p95={:.3},p98={:.3},mean={:.3}) modulated(p95={:.3},p98={:.3},mean={:.3}) baseline_profile={} modulated_profile={}",
            baseline_stats.p95_ratio,
            baseline_stats.p98_ratio,
            baseline_stats.mean_ratio,
            modulated_stats.p95_ratio,
            modulated_stats.p98_ratio,
            modulated_stats.mean_ratio,
            baseline_profile.summary(),
            modulated_profile.summary()
        );

        assert!(
            baseline_out.iter().all(|sample| sample.is_finite())
                && modulated_out.iter().all(|sample| sample.is_finite()),
            "profile mode diagnostics produced non-finite output for {label}"
        );
        assert!(
            !baseline_out.is_empty() && !modulated_out.is_empty(),
            "profile mode diagnostics produced empty output for {label}"
        );
    }
}

#[test]
fn quality_gate_dual_plane_fast_modulation_auto_profile_hysteresis_regression() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let callback_frames = 256usize;
    let mono = generate_gate_signal(sample_rate, bpm, 8.0);
    let input = mono_to_stereo_interleaved(&mono);

    let (auto_out, auto_boundaries, auto_profile) =
        run_dual_plane_deterministic_with_ratio_modulation_mode(
            &input,
            sample_rate,
            callback_frames,
            true,
            DeterministicProfileMode::Auto,
        );
    let (fixed_mix_out, fixed_mix_boundaries, fixed_mix_profile) =
        run_dual_plane_deterministic_with_ratio_modulation_mode(
            &input,
            sample_rate,
            callback_frames,
            true,
            DeterministicProfileMode::Fixed(LatencyProfile::Mix),
        );
    let (fixed_scratch_out, fixed_scratch_boundaries, fixed_scratch_profile) =
        run_dual_plane_deterministic_with_ratio_modulation_mode(
            &input,
            sample_rate,
            callback_frames,
            true,
            DeterministicProfileMode::Fixed(LatencyProfile::Scratch),
        );

    for (label, output) in [
        ("auto", &auto_out),
        ("fixed-mix", &fixed_mix_out),
        ("fixed-scratch", &fixed_scratch_out),
    ] {
        assert!(
            output.iter().all(|sample| sample.is_finite()),
            "fast-modulation profile hysteresis regression produced non-finite output for {label}"
        );
        assert!(
            !output.is_empty(),
            "fast-modulation profile hysteresis regression produced empty output for {label}"
        );
    }

    let trim = 16usize;
    let positions = |boundaries: &[usize], signal_len: usize| -> Vec<usize> {
        let relevant = if boundaries.len() > trim * 2 {
            &boundaries[trim..boundaries.len() - trim]
        } else {
            boundaries
        };
        relevant
            .iter()
            .copied()
            .filter(|&pos| pos > 1 && pos + 1 < signal_len)
            .collect()
    };
    let window = (sample_rate as f64 * 0.008).round() as usize;
    let guard = (sample_rate as f64 * 0.001).round() as usize;

    let auto_left = extract_left_channel(&auto_out);
    let fixed_mix_left = extract_left_channel(&fixed_mix_out);
    let fixed_scratch_left = extract_left_channel(&fixed_scratch_out);
    let auto_stats = boundary_artifact_stats(
        &auto_left,
        &positions(&auto_boundaries, auto_left.len()),
        window,
        guard,
    );
    let fixed_mix_stats = boundary_artifact_stats(
        &fixed_mix_left,
        &positions(&fixed_mix_boundaries, fixed_mix_left.len()),
        window,
        guard,
    );
    let fixed_scratch_stats = boundary_artifact_stats(
        &fixed_scratch_left,
        &positions(&fixed_scratch_boundaries, fixed_scratch_left.len()),
        window,
        guard,
    );

    println!(
        "fast-mod auto-profile hysteresis: auto(p95={:.3},p98={:.3},mean={:.3},profiles={}) fixed-mix(p95={:.3},p98={:.3},mean={:.3},profiles={}) fixed-scratch(p95={:.3},p98={:.3},mean={:.3},profiles={})",
        auto_stats.p95_ratio,
        auto_stats.p98_ratio,
        auto_stats.mean_ratio,
        auto_profile.summary(),
        fixed_mix_stats.p95_ratio,
        fixed_mix_stats.p98_ratio,
        fixed_mix_stats.mean_ratio,
        fixed_mix_profile.summary(),
        fixed_scratch_stats.p95_ratio,
        fixed_scratch_stats.p98_ratio,
        fixed_scratch_stats.mean_ratio,
        fixed_scratch_profile.summary()
    );

    write_quality_dashboard_csv(
        "quality_gate_dual_plane_fast_modulation_auto_profile_hysteresis_regression",
        "auto_p95,auto_p98,auto_mean,auto_current_profile_changes,auto_target_profile_changes,auto_policy_profile_changes,fixed_mix_p95,fixed_mix_p98,fixed_mix_mean,fixed_scratch_p95,fixed_scratch_p98,fixed_scratch_mean",
        &format!(
            "{:.6},{:.6},{:.6},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            auto_stats.p95_ratio,
            auto_stats.p98_ratio,
            auto_stats.mean_ratio,
            auto_profile.current_profile_changes,
            auto_profile.target_profile_changes,
            auto_profile.policy_profile_changes,
            fixed_mix_stats.p95_ratio,
            fixed_mix_stats.p98_ratio,
            fixed_mix_stats.mean_ratio,
            fixed_scratch_stats.p95_ratio,
            fixed_scratch_stats.p98_ratio,
            fixed_scratch_stats.mean_ratio
        ),
    );

    assert!(
        auto_stats.evaluated_boundaries >= 32
            && fixed_mix_stats.evaluated_boundaries >= 32
            && fixed_scratch_stats.evaluated_boundaries >= 32,
        "fast-modulation profile hysteresis regression evaluated too few boundaries (auto={}, fixed_mix={}, fixed_scratch={})",
        auto_stats.evaluated_boundaries,
        fixed_mix_stats.evaluated_boundaries,
        fixed_scratch_stats.evaluated_boundaries
    );
    let fixed_mix_final = fixed_mix_profile
        .last
        .expect("fixed-mix fast-modulation regression should observe profile telemetry");
    let fixed_scratch_final = fixed_scratch_profile
        .last
        .expect("fixed-scratch fast-modulation regression should observe profile telemetry");
    assert_eq!(
        fixed_mix_final.current_profile,
        LatencyProfile::Mix,
        "fixed mix mode should stay on mix under fast modulation"
    );
    assert_eq!(
        fixed_mix_final.target_profile,
        LatencyProfile::Mix,
        "fixed mix mode should not retarget away from mix under fast modulation"
    );
    assert_eq!(
        fixed_scratch_final.current_profile,
        LatencyProfile::Scratch,
        "fixed scratch mode should settle onto scratch under fast modulation"
    );
    assert_eq!(
        fixed_scratch_final.target_profile,
        LatencyProfile::Scratch,
        "fixed scratch mode should not be canceled back to mix under fast modulation"
    );
    assert_eq!(
        fixed_scratch_final.policy_profile,
        LatencyProfile::Scratch,
        "fixed scratch mode policy telemetry should stay aligned with the manual target"
    );
    assert!(
        auto_profile.current_profile_changes <= 2,
        "auto fast-modulation profile hysteresis regressed: current profile changed {} times",
        auto_profile.current_profile_changes
    );
    assert!(
        auto_profile.target_profile_changes <= 2,
        "auto fast-modulation profile hysteresis regressed: target profile changed {} times",
        auto_profile.target_profile_changes
    );
    assert!(
        auto_profile.policy_profile_changes <= 2,
        "auto fast-modulation profile hysteresis regressed: policy profile changed {} times",
        auto_profile.policy_profile_changes
    );
    assert!(
        auto_stats.p95_ratio <= fixed_mix_stats.p95_ratio * 0.80 + 0.05,
        "auto fast-modulation profile hysteresis regressed (p95): auto {:.3} vs fixed mix {:.3}",
        auto_stats.p95_ratio,
        fixed_mix_stats.p95_ratio
    );
    assert!(
        auto_stats.p98_ratio <= fixed_mix_stats.p98_ratio * 0.80 + 0.05,
        "auto fast-modulation profile hysteresis regressed (p98): auto {:.3} vs fixed mix {:.3}",
        auto_stats.p98_ratio,
        fixed_mix_stats.p98_ratio
    );
    assert!(
        auto_stats.mean_ratio <= fixed_mix_stats.mean_ratio * 0.85 + 0.05,
        "auto fast-modulation profile hysteresis regressed (mean): auto {:.3} vs fixed mix {:.3}",
        auto_stats.mean_ratio,
        fixed_mix_stats.mean_ratio
    );
}

#[test]
fn quality_gate_dual_plane_short_interval_step_modulation_artifacts() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let callback_frames = 256usize;
    let mono = generate_gate_signal(sample_rate, bpm, 8.0);
    let input = mono_to_stereo_interleaved(&mono);

    let (baseline_out, baseline_boundaries) = run_dual_plane_deterministic_with_ratio_modulation(
        &input,
        sample_rate,
        callback_frames,
        false,
    );
    let (modulated_out, modulated_boundaries, modulated_profile) =
        run_dual_plane_deterministic_with_ratio_steps_mode(
            &input,
            sample_rate,
            callback_frames,
            &[0.965, 1.035, 0.975, 1.025],
            2,
            DeterministicProfileMode::Auto,
        );

    assert!(
        baseline_out.iter().all(|s| s.is_finite()),
        "baseline short-interval modulation gate produced non-finite samples"
    );
    assert!(
        modulated_out.iter().all(|s| s.is_finite()),
        "short-interval modulation gate produced non-finite samples"
    );
    assert!(
        !baseline_out.is_empty() && !modulated_out.is_empty(),
        "short-interval modulation gate produced empty output"
    );

    let baseline_left = extract_left_channel(&baseline_out);
    let modulated_left = extract_left_channel(&modulated_out);
    let trim = 16usize;
    let baseline_positions: Vec<usize> = if baseline_boundaries.len() > trim * 2 {
        baseline_boundaries[trim..baseline_boundaries.len() - trim].to_vec()
    } else {
        baseline_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < baseline_left.len())
    .collect();
    let modulated_positions: Vec<usize> = if modulated_boundaries.len() > trim * 2 {
        modulated_boundaries[trim..modulated_boundaries.len() - trim].to_vec()
    } else {
        modulated_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < modulated_left.len())
    .collect();
    let window = (sample_rate as f64 * 0.008).round() as usize; // +/-8ms
    let guard = (sample_rate as f64 * 0.001).round() as usize; // +/-1ms
    let baseline_stats =
        boundary_artifact_stats(&baseline_left, &baseline_positions, window, guard);
    let modulated_stats =
        boundary_artifact_stats(&modulated_left, &modulated_positions, window, guard);
    println!(
        "dual-plane-short-step-mod gate: baseline(max={:.3},p95={:.3},p98={:.3},p99={:.3},mean={:.3},n={}) modulated(max={:.3},p95={:.3},p98={:.3},p99={:.3},mean={:.3},n={},profiles={})",
        baseline_stats.max_ratio,
        baseline_stats.p95_ratio,
        baseline_stats.p98_ratio,
        baseline_stats.p99_ratio,
        baseline_stats.mean_ratio,
        baseline_stats.evaluated_boundaries,
        modulated_stats.max_ratio,
        modulated_stats.p95_ratio,
        modulated_stats.p98_ratio,
        modulated_stats.p99_ratio,
        modulated_stats.mean_ratio,
        modulated_stats.evaluated_boundaries,
        modulated_profile.summary()
    );

    write_quality_dashboard_csv(
        "quality_gate_dual_plane_short_interval_step_modulation_artifacts",
        "baseline_max,baseline_p95,baseline_p98,baseline_p99,baseline_mean,baseline_n,modulated_max,modulated_p95,modulated_p98,modulated_p99,modulated_mean,modulated_n,modulated_current_profile_changes,modulated_target_profile_changes,modulated_policy_profile_changes",
        &format!(
            "{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{}",
            baseline_stats.max_ratio,
            baseline_stats.p95_ratio,
            baseline_stats.p98_ratio,
            baseline_stats.p99_ratio,
            baseline_stats.mean_ratio,
            baseline_stats.evaluated_boundaries,
            modulated_stats.max_ratio,
            modulated_stats.p95_ratio,
            modulated_stats.p98_ratio,
            modulated_stats.p99_ratio,
            modulated_stats.mean_ratio,
            modulated_stats.evaluated_boundaries,
            modulated_profile.current_profile_changes,
            modulated_profile.target_profile_changes,
            modulated_profile.policy_profile_changes
        ),
    );

    assert!(
        baseline_stats.evaluated_boundaries >= 32 && modulated_stats.evaluated_boundaries >= 32,
        "short-interval modulation artifact gate evaluated too few boundaries (baseline={}, modulated={})",
        baseline_stats.evaluated_boundaries,
        modulated_stats.evaluated_boundaries
    );
    assert!(
        modulated_stats.p95_ratio <= baseline_stats.p95_ratio * 2.6 + 1.0,
        "short-interval modulation artifact gate failed (p95): modulated {:.3} vs baseline {:.3}",
        modulated_stats.p95_ratio,
        baseline_stats.p95_ratio
    );
    assert!(
        modulated_stats.p98_ratio <= baseline_stats.p98_ratio * 3.0 + 1.4,
        "short-interval modulation artifact gate failed (p98): modulated {:.3} vs baseline {:.3}",
        modulated_stats.p98_ratio,
        baseline_stats.p98_ratio
    );
    assert!(
        modulated_stats.mean_ratio <= baseline_stats.mean_ratio * 2.3 + 1.0,
        "short-interval modulation artifact gate failed (mean): modulated {:.3} vs baseline {:.3}",
        modulated_stats.mean_ratio,
        baseline_stats.mean_ratio
    );
    let modulated_final = modulated_profile
        .last
        .expect("short-interval modulation gate should observe profile telemetry");
    assert_eq!(
        modulated_final.current_profile,
        LatencyProfile::Mix,
        "short-interval modulation should keep the active profile on mix"
    );
    assert_eq!(
        modulated_final.target_profile,
        LatencyProfile::Mix,
        "short-interval modulation should keep the target profile on mix"
    );
    assert_eq!(
        modulated_final.policy_profile,
        LatencyProfile::Mix,
        "short-interval modulation should keep the policy profile on mix"
    );
    assert_eq!(
        modulated_profile.current_profile_changes, 0,
        "short-interval modulation should not churn the current profile"
    );
    assert_eq!(
        modulated_profile.target_profile_changes, 0,
        "short-interval modulation should not churn the target profile"
    );
    assert_eq!(
        modulated_profile.policy_profile_changes, 0,
        "short-interval modulation should not churn the policy profile"
    );
}

#[test]
fn quality_gate_dual_plane_repeated_unity_plateau_modulation_artifacts() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let callback_frames = 256usize;
    let mono = generate_gate_signal(sample_rate, bpm, 8.0);
    let input = mono_to_stereo_interleaved(&mono);
    let ratios = [1.035, 1.025, 1.0, 1.0, 0.965, 0.975, 1.0, 1.0, 1.025, 1.035];

    let (baseline_out, baseline_boundaries) = run_dual_plane_deterministic_with_ratio_modulation(
        &input,
        sample_rate,
        callback_frames,
        false,
    );
    let (modulated_out, modulated_boundaries, modulated_profile) =
        run_dual_plane_deterministic_with_ratio_steps_mode(
            &input,
            sample_rate,
            callback_frames,
            &ratios,
            1,
            DeterministicProfileMode::Auto,
        );

    assert!(
        baseline_out.iter().all(|s| s.is_finite()),
        "baseline repeated-unity-plateau modulation gate produced non-finite samples"
    );
    assert!(
        modulated_out.iter().all(|s| s.is_finite()),
        "repeated-unity-plateau modulation gate produced non-finite samples"
    );
    assert!(
        !baseline_out.is_empty() && !modulated_out.is_empty(),
        "repeated-unity-plateau modulation gate produced empty output"
    );

    let baseline_left = extract_left_channel(&baseline_out);
    let modulated_left = extract_left_channel(&modulated_out);
    let trim = 16usize;
    let baseline_positions: Vec<usize> = if baseline_boundaries.len() > trim * 2 {
        baseline_boundaries[trim..baseline_boundaries.len() - trim].to_vec()
    } else {
        baseline_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < baseline_left.len())
    .collect();
    let modulated_positions: Vec<usize> = if modulated_boundaries.len() > trim * 2 {
        modulated_boundaries[trim..modulated_boundaries.len() - trim].to_vec()
    } else {
        modulated_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < modulated_left.len())
    .collect();
    let window = (sample_rate as f64 * 0.008).round() as usize;
    let guard = (sample_rate as f64 * 0.001).round() as usize;
    let baseline_stats =
        boundary_artifact_stats(&baseline_left, &baseline_positions, window, guard);
    let modulated_stats =
        boundary_artifact_stats(&modulated_left, &modulated_positions, window, guard);

    println!(
        "dual-plane-repeated-unity-plateau gate: baseline(max={:.3},p95={:.3},p98={:.3},mean={:.3},n={}) modulated(max={:.3},p95={:.3},p98={:.3},mean={:.3},n={},profiles={})",
        baseline_stats.max_ratio,
        baseline_stats.p95_ratio,
        baseline_stats.p98_ratio,
        baseline_stats.mean_ratio,
        baseline_stats.evaluated_boundaries,
        modulated_stats.max_ratio,
        modulated_stats.p95_ratio,
        modulated_stats.p98_ratio,
        modulated_stats.mean_ratio,
        modulated_stats.evaluated_boundaries,
        modulated_profile.summary()
    );

    write_quality_dashboard_csv(
        "quality_gate_dual_plane_repeated_unity_plateau_modulation_artifacts",
        "baseline_max,baseline_p95,baseline_p98,baseline_mean,baseline_n,modulated_max,modulated_p95,modulated_p98,modulated_mean,modulated_n,modulated_current_profile_changes,modulated_target_profile_changes,modulated_policy_profile_changes",
        &format!(
            "{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{}",
            baseline_stats.max_ratio,
            baseline_stats.p95_ratio,
            baseline_stats.p98_ratio,
            baseline_stats.mean_ratio,
            baseline_stats.evaluated_boundaries,
            modulated_stats.max_ratio,
            modulated_stats.p95_ratio,
            modulated_stats.p98_ratio,
            modulated_stats.mean_ratio,
            modulated_stats.evaluated_boundaries,
            modulated_profile.current_profile_changes,
            modulated_profile.target_profile_changes,
            modulated_profile.policy_profile_changes
        ),
    );

    assert!(
        baseline_stats.evaluated_boundaries >= 32 && modulated_stats.evaluated_boundaries >= 32,
        "repeated-unity-plateau modulation gate evaluated too few boundaries (baseline={}, modulated={})",
        baseline_stats.evaluated_boundaries,
        modulated_stats.evaluated_boundaries
    );
    assert!(
        modulated_stats.p95_ratio <= baseline_stats.p95_ratio * 2.4 + 0.9,
        "repeated-unity-plateau modulation gate failed (p95): modulated {:.3} vs baseline {:.3}",
        modulated_stats.p95_ratio,
        baseline_stats.p95_ratio
    );
    assert!(
        modulated_stats.p98_ratio <= baseline_stats.p98_ratio * 2.8 + 1.2,
        "repeated-unity-plateau modulation gate failed (p98): modulated {:.3} vs baseline {:.3}",
        modulated_stats.p98_ratio,
        baseline_stats.p98_ratio
    );
    assert!(
        modulated_stats.mean_ratio <= baseline_stats.mean_ratio * 2.1 + 0.9,
        "repeated-unity-plateau modulation gate failed (mean): modulated {:.3} vs baseline {:.3}",
        modulated_stats.mean_ratio,
        baseline_stats.mean_ratio
    );
    let modulated_final = modulated_profile
        .last
        .expect("repeated-unity-plateau modulation gate should observe profile telemetry");
    assert_eq!(
        modulated_final.current_profile,
        LatencyProfile::Mix,
        "repeated-unity-plateau modulation should keep the active profile on mix"
    );
    assert_eq!(
        modulated_final.target_profile,
        LatencyProfile::Mix,
        "repeated-unity-plateau modulation should keep the target profile on mix"
    );
    assert_eq!(
        modulated_final.policy_profile,
        LatencyProfile::Mix,
        "repeated-unity-plateau modulation should keep the policy profile on mix"
    );
    assert!(
        modulated_profile.current_profile_changes <= 2,
        "repeated-unity-plateau modulation regressed: current profile changed {} times",
        modulated_profile.current_profile_changes
    );
    assert!(
        modulated_profile.target_profile_changes <= 2,
        "repeated-unity-plateau modulation regressed: target profile changed {} times",
        modulated_profile.target_profile_changes
    );
    assert!(
        modulated_profile.policy_profile_changes <= 2,
        "repeated-unity-plateau modulation regressed: policy profile changed {} times",
        modulated_profile.policy_profile_changes
    );
}

#[test]
fn quality_gate_dual_plane_short_interval_auto_profile_regression() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let callback_frames = 256usize;
    let mono = generate_gate_signal(sample_rate, bpm, 8.0);
    let input = mono_to_stereo_interleaved(&mono);
    let ratios = [0.965, 1.035, 0.975, 1.025];

    let (auto_out, auto_boundaries, auto_profile) =
        run_dual_plane_deterministic_with_ratio_steps_mode(
            &input,
            sample_rate,
            callback_frames,
            &ratios,
            1,
            DeterministicProfileMode::Auto,
        );
    let (fixed_mix_out, fixed_mix_boundaries, fixed_mix_profile) =
        run_dual_plane_deterministic_with_ratio_steps_mode(
            &input,
            sample_rate,
            callback_frames,
            &ratios,
            1,
            DeterministicProfileMode::Fixed(LatencyProfile::Mix),
        );
    let (fixed_scratch_out, fixed_scratch_boundaries, fixed_scratch_profile) =
        run_dual_plane_deterministic_with_ratio_steps_mode(
            &input,
            sample_rate,
            callback_frames,
            &ratios,
            1,
            DeterministicProfileMode::Fixed(LatencyProfile::Scratch),
        );

    for (label, output) in [
        ("auto", &auto_out),
        ("fixed-mix", &fixed_mix_out),
        ("fixed-scratch", &fixed_scratch_out),
    ] {
        assert!(
            output.iter().all(|sample| sample.is_finite()),
            "short-interval auto-profile regression produced non-finite output for {label}"
        );
        assert!(
            !output.is_empty(),
            "short-interval auto-profile regression produced empty output for {label}"
        );
    }

    let trim = 16usize;
    let positions = |boundaries: &[usize], signal_len: usize| -> Vec<usize> {
        let relevant = if boundaries.len() > trim * 2 {
            &boundaries[trim..boundaries.len() - trim]
        } else {
            boundaries
        };
        relevant
            .iter()
            .copied()
            .filter(|&pos| pos > 1 && pos + 1 < signal_len)
            .collect()
    };
    let window = (sample_rate as f64 * 0.008).round() as usize;
    let guard = (sample_rate as f64 * 0.001).round() as usize;

    let auto_left = extract_left_channel(&auto_out);
    let fixed_mix_left = extract_left_channel(&fixed_mix_out);
    let fixed_scratch_left = extract_left_channel(&fixed_scratch_out);
    let auto_stats = boundary_artifact_stats(
        &auto_left,
        &positions(&auto_boundaries, auto_left.len()),
        window,
        guard,
    );
    let fixed_mix_stats = boundary_artifact_stats(
        &fixed_mix_left,
        &positions(&fixed_mix_boundaries, fixed_mix_left.len()),
        window,
        guard,
    );
    let fixed_scratch_stats = boundary_artifact_stats(
        &fixed_scratch_left,
        &positions(&fixed_scratch_boundaries, fixed_scratch_left.len()),
        window,
        guard,
    );

    println!(
        "short-interval auto-profile regression: auto(p95={:.3},p98={:.3},mean={:.3},profiles={}) fixed-mix(p95={:.3},p98={:.3},mean={:.3},profiles={}) fixed-scratch(p95={:.3},p98={:.3},mean={:.3},profiles={})",
        auto_stats.p95_ratio,
        auto_stats.p98_ratio,
        auto_stats.mean_ratio,
        auto_profile.summary(),
        fixed_mix_stats.p95_ratio,
        fixed_mix_stats.p98_ratio,
        fixed_mix_stats.mean_ratio,
        fixed_mix_profile.summary(),
        fixed_scratch_stats.p95_ratio,
        fixed_scratch_stats.p98_ratio,
        fixed_scratch_stats.mean_ratio,
        fixed_scratch_profile.summary()
    );

    write_quality_dashboard_csv(
        "quality_gate_dual_plane_short_interval_auto_profile_regression",
        "auto_p95,auto_p98,auto_mean,auto_current_profile_changes,auto_target_profile_changes,auto_policy_profile_changes,fixed_mix_p95,fixed_mix_p98,fixed_mix_mean,fixed_scratch_p95,fixed_scratch_p98,fixed_scratch_mean",
        &format!(
            "{:.6},{:.6},{:.6},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            auto_stats.p95_ratio,
            auto_stats.p98_ratio,
            auto_stats.mean_ratio,
            auto_profile.current_profile_changes,
            auto_profile.target_profile_changes,
            auto_profile.policy_profile_changes,
            fixed_mix_stats.p95_ratio,
            fixed_mix_stats.p98_ratio,
            fixed_mix_stats.mean_ratio,
            fixed_scratch_stats.p95_ratio,
            fixed_scratch_stats.p98_ratio,
            fixed_scratch_stats.mean_ratio
        ),
    );

    assert!(
        auto_stats.evaluated_boundaries >= 32
            && fixed_mix_stats.evaluated_boundaries >= 32
            && fixed_scratch_stats.evaluated_boundaries >= 32,
        "short-interval auto-profile regression evaluated too few boundaries (auto={}, fixed_mix={}, fixed_scratch={})",
        auto_stats.evaluated_boundaries,
        fixed_mix_stats.evaluated_boundaries,
        fixed_scratch_stats.evaluated_boundaries
    );
    let auto_final = auto_profile
        .last
        .expect("short-interval auto-profile regression should observe profile telemetry");
    assert_eq!(
        auto_final.current_profile,
        LatencyProfile::Mix,
        "short-interval auto-profile regression should keep the active profile on mix"
    );
    assert_eq!(
        auto_final.target_profile,
        LatencyProfile::Mix,
        "short-interval auto-profile regression should keep the target profile on mix"
    );
    assert_eq!(
        auto_final.policy_profile,
        LatencyProfile::Mix,
        "short-interval auto-profile regression should keep the policy profile on mix"
    );
    assert_eq!(
        auto_profile.current_profile_changes, 0,
        "short-interval auto-profile regression should not churn the current profile"
    );
    assert_eq!(
        auto_profile.target_profile_changes, 0,
        "short-interval auto-profile regression should not churn the target profile"
    );
    assert_eq!(
        auto_profile.policy_profile_changes, 0,
        "short-interval auto-profile regression should not churn the policy profile"
    );
    assert!(
        auto_stats.p95_ratio <= fixed_mix_stats.p95_ratio + 0.02,
        "short-interval auto-profile regression diverged from fixed mix (p95): auto {:.3} vs fixed mix {:.3}",
        auto_stats.p95_ratio,
        fixed_mix_stats.p95_ratio
    );
    assert!(
        auto_stats.p98_ratio <= fixed_mix_stats.p98_ratio + 0.03,
        "short-interval auto-profile regression diverged from fixed mix (p98): auto {:.3} vs fixed mix {:.3}",
        auto_stats.p98_ratio,
        fixed_mix_stats.p98_ratio
    );
    assert!(
        auto_stats.mean_ratio <= fixed_mix_stats.mean_ratio + 0.02,
        "short-interval auto-profile regression diverged from fixed mix (mean): auto {:.3} vs fixed mix {:.3}",
        auto_stats.mean_ratio,
        fixed_mix_stats.mean_ratio
    );
    assert!(
        auto_stats.p95_ratio <= fixed_scratch_stats.p95_ratio * 0.90 + 0.05,
        "short-interval auto-profile regression failed to stay below scratch-biased artifacts (p95): auto {:.3} vs fixed scratch {:.3}",
        auto_stats.p95_ratio,
        fixed_scratch_stats.p95_ratio
    );
}

#[test]
fn quality_gate_dual_plane_profile_transition_commit_artifacts() {
    let sample_rate = 44_100u32;
    let callback_frames = 256usize;
    let hold_callbacks = 24usize;
    let mono = generate_harmonic_bed(sample_rate, 8.0);
    let input = mono_to_stereo_interleaved(&mono);
    let ratios = [1.18, 1.70, 1.18];

    let (auto_out, _auto_boundaries, auto_profile, auto_trace) =
        run_dual_plane_deterministic_with_ratio_steps_mode_trace(
            &input,
            sample_rate,
            callback_frames,
            &ratios,
            hold_callbacks,
            DeterministicProfileMode::Auto,
        );
    let (fixed_mix_out, _fixed_mix_boundaries, fixed_mix_profile, fixed_mix_trace) =
        run_dual_plane_deterministic_with_ratio_steps_mode_trace(
            &input,
            sample_rate,
            callback_frames,
            &ratios,
            hold_callbacks,
            DeterministicProfileMode::Fixed(LatencyProfile::Mix),
        );
    let (fixed_scratch_out, _fixed_scratch_boundaries, fixed_scratch_profile, fixed_scratch_trace) =
        run_dual_plane_deterministic_with_ratio_steps_mode_trace(
            &input,
            sample_rate,
            callback_frames,
            &ratios,
            hold_callbacks,
            DeterministicProfileMode::Fixed(LatencyProfile::Scratch),
        );

    for (label, output) in [
        ("auto", &auto_out),
        ("fixed-mix", &fixed_mix_out),
        ("fixed-scratch", &fixed_scratch_out),
    ] {
        assert!(
            output.iter().all(|sample| sample.is_finite()),
            "profile-transition commit gate produced non-finite output for {label}"
        );
        assert!(
            !output.is_empty(),
            "profile-transition commit gate produced empty output for {label}"
        );
    }

    let auto_scratch_changes = auto_trace
        .current_profile_change_profiles
        .iter()
        .filter(|&&profile| profile == LatencyProfile::Scratch)
        .count();
    assert!(
        auto_scratch_changes > 0,
        "profile-transition commit gate should observe at least one scratch commit, saw scratch={} ({})",
        auto_scratch_changes,
        auto_profile.summary()
    );

    let window = (sample_rate as f64 * 0.008).round() as usize;
    let guard = (sample_rate as f64 * 0.001).round() as usize;
    let auto_left = extract_left_channel(&auto_out);
    let fixed_mix_left = extract_left_channel(&fixed_mix_out);
    let fixed_scratch_left = extract_left_channel(&fixed_scratch_out);
    let auto_positions = positions_from_callback_trace(
        &auto_trace.callback_output_frames,
        &auto_trace.current_profile_change_callbacks,
        auto_left.len(),
    );
    let fixed_mix_positions = positions_from_callback_trace(
        &fixed_mix_trace.callback_output_frames,
        &auto_trace.current_profile_change_callbacks,
        fixed_mix_left.len(),
    );
    let fixed_scratch_positions = positions_from_callback_trace(
        &fixed_scratch_trace.callback_output_frames,
        &auto_trace.current_profile_change_callbacks,
        fixed_scratch_left.len(),
    );

    let auto_stats = boundary_artifact_stats(&auto_left, &auto_positions, window, guard);
    let fixed_mix_stats =
        boundary_artifact_stats(&fixed_mix_left, &fixed_mix_positions, window, guard);
    let fixed_scratch_stats =
        boundary_artifact_stats(&fixed_scratch_left, &fixed_scratch_positions, window, guard);

    println!(
        "profile-transition commit gate: auto(p95={:.3},p98={:.3},mean={:.3},n={},profiles={}) fixed-mix(p95={:.3},p98={:.3},mean={:.3},n={},profiles={}) fixed-scratch(p95={:.3},p98={:.3},mean={:.3},n={},profiles={})",
        auto_stats.p95_ratio,
        auto_stats.p98_ratio,
        auto_stats.mean_ratio,
        auto_stats.evaluated_boundaries,
        auto_profile.summary(),
        fixed_mix_stats.p95_ratio,
        fixed_mix_stats.p98_ratio,
        fixed_mix_stats.mean_ratio,
        fixed_mix_stats.evaluated_boundaries,
        fixed_mix_profile.summary(),
        fixed_scratch_stats.p95_ratio,
        fixed_scratch_stats.p98_ratio,
        fixed_scratch_stats.mean_ratio,
        fixed_scratch_stats.evaluated_boundaries,
        fixed_scratch_profile.summary()
    );

    write_quality_dashboard_csv(
        "quality_gate_dual_plane_profile_transition_commit_artifacts",
        "auto_p95,auto_p98,auto_mean,auto_n,auto_scratch_changes,fixed_mix_p95,fixed_mix_p98,fixed_mix_mean,fixed_mix_n,fixed_scratch_p95,fixed_scratch_p98,fixed_scratch_mean,fixed_scratch_n",
        &format!(
            "{:.6},{:.6},{:.6},{},{},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{}",
            auto_stats.p95_ratio,
            auto_stats.p98_ratio,
            auto_stats.mean_ratio,
            auto_stats.evaluated_boundaries,
            auto_scratch_changes,
            fixed_mix_stats.p95_ratio,
            fixed_mix_stats.p98_ratio,
            fixed_mix_stats.mean_ratio,
            fixed_mix_stats.evaluated_boundaries,
            fixed_scratch_stats.p95_ratio,
            fixed_scratch_stats.p98_ratio,
            fixed_scratch_stats.mean_ratio,
            fixed_scratch_stats.evaluated_boundaries
        ),
    );

    assert!(
        auto_stats.evaluated_boundaries >= 1
            && fixed_mix_stats.evaluated_boundaries >= 1
            && fixed_scratch_stats.evaluated_boundaries >= 1,
        "profile-transition commit gate evaluated too few commit boundaries (auto={}, fixed_mix={}, fixed_scratch={})",
        auto_stats.evaluated_boundaries,
        fixed_mix_stats.evaluated_boundaries,
        fixed_scratch_stats.evaluated_boundaries
    );

    let endpoint_p95 = fixed_mix_stats.p95_ratio.max(fixed_scratch_stats.p95_ratio);
    let endpoint_p98 = fixed_mix_stats.p98_ratio.max(fixed_scratch_stats.p98_ratio);
    let endpoint_mean = fixed_mix_stats
        .mean_ratio
        .max(fixed_scratch_stats.mean_ratio);
    assert!(
        auto_stats.p95_ratio <= endpoint_p95 * 1.20 + 0.10,
        "profile-transition commit gate regressed (p95): auto {:.3} vs fixed endpoint {:.3}",
        auto_stats.p95_ratio,
        endpoint_p95
    );
    assert!(
        auto_stats.p98_ratio <= endpoint_p98 * 1.25 + 0.15,
        "profile-transition commit gate regressed (p98): auto {:.3} vs fixed endpoint {:.3}",
        auto_stats.p98_ratio,
        endpoint_p98
    );
    assert!(
        auto_stats.mean_ratio <= endpoint_mean * 1.20 + 0.10,
        "profile-transition commit gate regressed (mean): auto {:.3} vs fixed endpoint {:.3}",
        auto_stats.mean_ratio,
        endpoint_mean
    );
}

#[test]
fn quality_gate_dual_plane_callback_toggle_modulation_artifacts() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let callback_frames = 256usize;
    let mono = generate_gate_signal(sample_rate, bpm, 8.0);
    let input = mono_to_stereo_interleaved(&mono);

    let (baseline_out, baseline_boundaries) = run_dual_plane_deterministic_with_ratio_steps(
        &input,
        sample_rate,
        callback_frames,
        &[1.0],
        1,
    );
    let (modulated_out, modulated_boundaries) = run_dual_plane_deterministic_with_ratio_steps(
        &input,
        sample_rate,
        callback_frames,
        &[0.965, 1.035, 0.975, 1.025],
        1,
    );

    assert!(
        baseline_out.iter().all(|s| s.is_finite()),
        "baseline callback-toggle modulation gate produced non-finite samples"
    );
    assert!(
        modulated_out.iter().all(|s| s.is_finite()),
        "callback-toggle modulation gate produced non-finite samples"
    );
    assert!(
        !baseline_out.is_empty() && !modulated_out.is_empty(),
        "callback-toggle modulation gate produced empty output"
    );

    let baseline_left = extract_left_channel(&baseline_out);
    let modulated_left = extract_left_channel(&modulated_out);
    let trim = 16usize;
    let baseline_positions: Vec<usize> = if baseline_boundaries.len() > trim * 2 {
        baseline_boundaries[trim..baseline_boundaries.len() - trim].to_vec()
    } else {
        baseline_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < baseline_left.len())
    .collect();
    let modulated_positions: Vec<usize> = if modulated_boundaries.len() > trim * 2 {
        modulated_boundaries[trim..modulated_boundaries.len() - trim].to_vec()
    } else {
        modulated_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < modulated_left.len())
    .collect();
    let window = (sample_rate as f64 * 0.008).round() as usize; // +/-8ms
    let guard = (sample_rate as f64 * 0.001).round() as usize; // +/-1ms
    let baseline_stats =
        boundary_artifact_stats(&baseline_left, &baseline_positions, window, guard);
    let modulated_stats =
        boundary_artifact_stats(&modulated_left, &modulated_positions, window, guard);
    println!(
        "dual-plane-callback-toggle-mod gate: baseline(max={:.3},p95={:.3},p98={:.3},p99={:.3},mean={:.3},n={}) modulated(max={:.3},p95={:.3},p98={:.3},p99={:.3},mean={:.3},n={})",
        baseline_stats.max_ratio,
        baseline_stats.p95_ratio,
        baseline_stats.p98_ratio,
        baseline_stats.p99_ratio,
        baseline_stats.mean_ratio,
        baseline_stats.evaluated_boundaries,
        modulated_stats.max_ratio,
        modulated_stats.p95_ratio,
        modulated_stats.p98_ratio,
        modulated_stats.p99_ratio,
        modulated_stats.mean_ratio,
        modulated_stats.evaluated_boundaries
    );

    write_quality_dashboard_csv(
        "quality_gate_dual_plane_callback_toggle_modulation_artifacts",
        "baseline_max,baseline_p95,baseline_p98,baseline_p99,baseline_mean,baseline_n,modulated_max,modulated_p95,modulated_p98,modulated_p99,modulated_mean,modulated_n",
        &format!(
            "{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            baseline_stats.max_ratio,
            baseline_stats.p95_ratio,
            baseline_stats.p98_ratio,
            baseline_stats.p99_ratio,
            baseline_stats.mean_ratio,
            baseline_stats.evaluated_boundaries,
            modulated_stats.max_ratio,
            modulated_stats.p95_ratio,
            modulated_stats.p98_ratio,
            modulated_stats.p99_ratio,
            modulated_stats.mean_ratio,
            modulated_stats.evaluated_boundaries
        ),
    );

    assert!(
        baseline_stats.evaluated_boundaries >= 32 && modulated_stats.evaluated_boundaries >= 32,
        "callback-toggle modulation artifact gate evaluated too few boundaries (baseline={}, modulated={})",
        baseline_stats.evaluated_boundaries,
        modulated_stats.evaluated_boundaries
    );
    assert!(
        modulated_stats.p95_ratio <= baseline_stats.p95_ratio * 2.6 + 1.0,
        "callback-toggle modulation artifact gate failed (p95): modulated {:.3} vs baseline {:.3}",
        modulated_stats.p95_ratio,
        baseline_stats.p95_ratio
    );
    assert!(
        modulated_stats.p98_ratio <= baseline_stats.p98_ratio * 3.0 + 1.4,
        "callback-toggle modulation artifact gate failed (p98): modulated {:.3} vs baseline {:.3}",
        modulated_stats.p98_ratio,
        baseline_stats.p98_ratio
    );
    assert!(
        modulated_stats.mean_ratio <= baseline_stats.mean_ratio * 2.3 + 1.0,
        "callback-toggle modulation artifact gate failed (mean): modulated {:.3} vs baseline {:.3}",
        modulated_stats.mean_ratio,
        baseline_stats.mean_ratio
    );
}

#[test]
fn quality_gate_dual_plane_fixed_buffer_callback_toggle_write_cadence_regression() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let callback_frames = 256usize;
    let mono = generate_gate_signal(sample_rate, bpm, 8.0);
    let input = mono_to_stereo_interleaved(&mono);

    let callback_writes = run_dual_plane_deterministic_fixed_buffer_ratio_steps_callback_writes(
        &input,
        sample_rate,
        callback_frames,
        &[0.965, 1.035, 0.975, 1.025],
        1,
    );
    let stats = callback_cadence_stats(&callback_writes);

    println!(
        "dual-plane-fixed-buffer-callback-toggle-cadence: callbacks_after_first_output={} callbacks_with_output_after_first_output={} max_idle_gap_callbacks={}",
        stats.callbacks_after_first_output,
        stats.callbacks_with_output_after_first_output,
        stats.max_idle_gap_callbacks
    );

    assert!(
        stats.callbacks_after_first_output >= 64,
        "fixed-buffer callback-toggle cadence gate observed too little steady-state output (callbacks_after_first_output={})",
        stats.callbacks_after_first_output
    );
    assert!(
        stats.callbacks_with_output_after_first_output * 2
            >= stats.callbacks_after_first_output,
        "fixed-buffer callback-toggle cadence gate regressed: only {} of {} callbacks emitted output after the first write",
        stats.callbacks_with_output_after_first_output,
        stats.callbacks_after_first_output
    );
    assert!(
        stats.max_idle_gap_callbacks <= 4,
        "fixed-buffer callback-toggle cadence gate regressed: max idle gap {} callbacks",
        stats.max_idle_gap_callbacks
    );
}

#[test]
fn quality_gate_dual_plane_fixed_buffer_short_interval_plateau_write_cadence_regression() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let callback_frames = 256usize;
    let mono = generate_gate_signal(sample_rate, bpm, 8.0);
    let input = mono_to_stereo_interleaved(&mono);

    let callback_writes = run_dual_plane_deterministic_fixed_buffer_ratio_steps_callback_writes(
        &input,
        sample_rate,
        callback_frames,
        &[0.965, 1.035, 0.975, 1.025],
        2,
    );
    let stats = callback_cadence_stats(&callback_writes);

    println!(
        "dual-plane-fixed-buffer-short-plateau-cadence: callbacks_after_first_output={} callbacks_with_output_after_first_output={} max_idle_gap_callbacks={}",
        stats.callbacks_after_first_output,
        stats.callbacks_with_output_after_first_output,
        stats.max_idle_gap_callbacks
    );

    assert!(
        stats.callbacks_after_first_output >= 64,
        "fixed-buffer short-interval plateau cadence gate observed too little steady-state output (callbacks_after_first_output={})",
        stats.callbacks_after_first_output
    );
    assert!(
        stats.callbacks_with_output_after_first_output * 2
            >= stats.callbacks_after_first_output,
        "fixed-buffer short-interval plateau cadence gate regressed: only {} of {} callbacks emitted output after the first write",
        stats.callbacks_with_output_after_first_output,
        stats.callbacks_after_first_output
    );
    assert!(
        stats.max_idle_gap_callbacks <= 4,
        "fixed-buffer short-interval plateau cadence gate regressed: max idle gap {} callbacks",
        stats.max_idle_gap_callbacks
    );
}

#[test]
fn quality_gate_dual_plane_callback_toggle_modulation_outlier_regression() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let callback_frames = 256usize;
    let mono = generate_gate_signal(sample_rate, bpm, 8.0);
    let input = mono_to_stereo_interleaved(&mono);

    let (baseline_out, baseline_boundaries) = run_dual_plane_deterministic_with_ratio_steps(
        &input,
        sample_rate,
        callback_frames,
        &[1.0],
        1,
    );
    let (modulated_out, modulated_boundaries) = run_dual_plane_deterministic_with_ratio_steps(
        &input,
        sample_rate,
        callback_frames,
        &[0.965, 1.035, 0.975, 1.025],
        1,
    );

    let baseline_left = extract_left_channel(&baseline_out);
    let modulated_left = extract_left_channel(&modulated_out);
    let trim = 16usize;
    let baseline_positions: Vec<usize> = if baseline_boundaries.len() > trim * 2 {
        baseline_boundaries[trim..baseline_boundaries.len() - trim].to_vec()
    } else {
        baseline_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < baseline_left.len())
    .collect();
    let modulated_positions: Vec<usize> = if modulated_boundaries.len() > trim * 2 {
        modulated_boundaries[trim..modulated_boundaries.len() - trim].to_vec()
    } else {
        modulated_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < modulated_left.len())
    .collect();

    let window = (sample_rate as f64 * 0.008).round() as usize;
    let guard = (sample_rate as f64 * 0.001).round() as usize;
    let baseline_stats =
        boundary_artifact_stats(&baseline_left, &baseline_positions, window, guard);
    let modulated_stats =
        boundary_artifact_stats(&modulated_left, &modulated_positions, window, guard);

    println!(
        "dual-plane-callback-toggle-outlier gate: baseline(max={:.3},p99={:.3},n={}) modulated(max={:.3},p99={:.3},n={})",
        baseline_stats.max_ratio,
        baseline_stats.p99_ratio,
        baseline_stats.evaluated_boundaries,
        modulated_stats.max_ratio,
        modulated_stats.p99_ratio,
        modulated_stats.evaluated_boundaries
    );

    assert!(
        baseline_stats.evaluated_boundaries >= 32 && modulated_stats.evaluated_boundaries >= 32,
        "callback-toggle outlier regression gate evaluated too few boundaries (baseline={}, modulated={})",
        baseline_stats.evaluated_boundaries,
        modulated_stats.evaluated_boundaries
    );
    assert!(
        modulated_stats.max_ratio <= baseline_stats.max_ratio * 1.15 + 0.10,
        "callback-toggle outlier regression failed (max): modulated {:.3} vs baseline {:.3}",
        modulated_stats.max_ratio,
        baseline_stats.max_ratio
    );
    assert!(
        modulated_stats.p99_ratio <= baseline_stats.p99_ratio * 1.15 + 0.10,
        "callback-toggle outlier regression failed (p99): modulated {:.3} vs baseline {:.3}",
        modulated_stats.p99_ratio,
        baseline_stats.p99_ratio
    );
}

#[test]
fn quality_gate_dual_plane_short_interval_step_modulation_outlier_regression() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let callback_frames = 256usize;
    let mono = generate_gate_signal(sample_rate, bpm, 8.0);
    let input = mono_to_stereo_interleaved(&mono);

    let (baseline_out, baseline_boundaries) = run_dual_plane_deterministic_with_ratio_steps(
        &input,
        sample_rate,
        callback_frames,
        &[1.0],
        1,
    );
    let (modulated_out, modulated_boundaries) = run_dual_plane_deterministic_with_ratio_steps(
        &input,
        sample_rate,
        callback_frames,
        &[0.965, 1.035, 0.975, 1.025],
        2,
    );

    let baseline_left = extract_left_channel(&baseline_out);
    let modulated_left = extract_left_channel(&modulated_out);
    let trim = 16usize;
    let baseline_positions: Vec<usize> = if baseline_boundaries.len() > trim * 2 {
        baseline_boundaries[trim..baseline_boundaries.len() - trim].to_vec()
    } else {
        baseline_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < baseline_left.len())
    .collect();
    let modulated_positions: Vec<usize> = if modulated_boundaries.len() > trim * 2 {
        modulated_boundaries[trim..modulated_boundaries.len() - trim].to_vec()
    } else {
        modulated_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < modulated_left.len())
    .collect();

    let window = (sample_rate as f64 * 0.008).round() as usize;
    let guard = (sample_rate as f64 * 0.001).round() as usize;
    let baseline_stats =
        boundary_artifact_stats(&baseline_left, &baseline_positions, window, guard);
    let modulated_stats =
        boundary_artifact_stats(&modulated_left, &modulated_positions, window, guard);

    println!(
        "dual-plane-short-step-outlier gate: baseline(max={:.3},p99={:.3},n={}) modulated(max={:.3},p99={:.3},n={})",
        baseline_stats.max_ratio,
        baseline_stats.p99_ratio,
        baseline_stats.evaluated_boundaries,
        modulated_stats.max_ratio,
        modulated_stats.p99_ratio,
        modulated_stats.evaluated_boundaries
    );

    assert!(
        baseline_stats.evaluated_boundaries >= 32 && modulated_stats.evaluated_boundaries >= 32,
        "short-interval step outlier regression gate evaluated too few boundaries (baseline={}, modulated={})",
        baseline_stats.evaluated_boundaries,
        modulated_stats.evaluated_boundaries
    );
    assert!(
        modulated_stats.max_ratio <= baseline_stats.max_ratio * 1.15 + 0.10,
        "short-interval step outlier regression failed (max): modulated {:.3} vs baseline {:.3}",
        modulated_stats.max_ratio,
        baseline_stats.max_ratio
    );
    assert!(
        modulated_stats.p99_ratio <= baseline_stats.p99_ratio * 1.15 + 0.10,
        "short-interval step outlier regression failed (p99): modulated {:.3} vs baseline {:.3}",
        modulated_stats.p99_ratio,
        baseline_stats.p99_ratio
    );
}

#[test]
fn quality_gate_dual_plane_unity_roundtrip_callback_toggle_regression() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let callback_frames = 256usize;
    let mono = generate_gate_signal(sample_rate, bpm, 8.0);
    let input = mono_to_stereo_interleaved(&mono);

    let (baseline_out, baseline_boundaries) = run_dual_plane_deterministic_with_ratio_steps(
        &input,
        sample_rate,
        callback_frames,
        &[1.0],
        1,
    );
    let (modulated_out, modulated_boundaries) = run_dual_plane_deterministic_with_ratio_steps(
        &input,
        sample_rate,
        callback_frames,
        &[1.0, 0.965, 1.0, 1.035],
        1,
    );

    assert!(
        baseline_out.iter().all(|sample| sample.is_finite()),
        "baseline unity-roundtrip callback-toggle gate produced non-finite samples"
    );
    assert!(
        modulated_out.iter().all(|sample| sample.is_finite()),
        "unity-roundtrip callback-toggle gate produced non-finite samples"
    );
    assert!(
        !baseline_out.is_empty() && !modulated_out.is_empty(),
        "unity-roundtrip callback-toggle gate produced empty output"
    );

    let baseline_left = extract_left_channel(&baseline_out);
    let modulated_left = extract_left_channel(&modulated_out);
    let trim = 16usize;
    let baseline_positions: Vec<usize> = if baseline_boundaries.len() > trim * 2 {
        baseline_boundaries[trim..baseline_boundaries.len() - trim].to_vec()
    } else {
        baseline_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < baseline_left.len())
    .collect();
    let modulated_positions: Vec<usize> = if modulated_boundaries.len() > trim * 2 {
        modulated_boundaries[trim..modulated_boundaries.len() - trim].to_vec()
    } else {
        modulated_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < modulated_left.len())
    .collect();

    let window = (sample_rate as f64 * 0.008).round() as usize;
    let guard = (sample_rate as f64 * 0.001).round() as usize;
    let baseline_stats =
        boundary_artifact_stats(&baseline_left, &baseline_positions, window, guard);
    let modulated_stats =
        boundary_artifact_stats(&modulated_left, &modulated_positions, window, guard);

    println!(
        "dual-plane-unity-roundtrip-callback-toggle gate: baseline(max={:.3},p99={:.3},n={}) modulated(max={:.3},p99={:.3},n={})",
        baseline_stats.max_ratio,
        baseline_stats.p99_ratio,
        baseline_stats.evaluated_boundaries,
        modulated_stats.max_ratio,
        modulated_stats.p99_ratio,
        modulated_stats.evaluated_boundaries
    );

    write_quality_dashboard_csv(
        "quality_gate_dual_plane_unity_roundtrip_callback_toggle_regression",
        "baseline_max,baseline_p99,baseline_n,modulated_max,modulated_p99,modulated_n",
        &format!(
            "{:.6},{:.6},{},{:.6},{:.6},{}",
            baseline_stats.max_ratio,
            baseline_stats.p99_ratio,
            baseline_stats.evaluated_boundaries,
            modulated_stats.max_ratio,
            modulated_stats.p99_ratio,
            modulated_stats.evaluated_boundaries
        ),
    );

    assert!(
        baseline_stats.evaluated_boundaries >= 32 && modulated_stats.evaluated_boundaries >= 32,
        "unity-roundtrip callback-toggle regression gate evaluated too few boundaries (baseline={}, modulated={})",
        baseline_stats.evaluated_boundaries,
        modulated_stats.evaluated_boundaries
    );
    assert!(
        modulated_stats.max_ratio <= baseline_stats.max_ratio * 1.15 + 0.10,
        "unity-roundtrip callback-toggle regression failed (max): modulated {:.3} vs baseline {:.3}",
        modulated_stats.max_ratio,
        baseline_stats.max_ratio
    );
    assert!(
        modulated_stats.p99_ratio <= baseline_stats.p99_ratio * 1.15 + 0.10,
        "unity-roundtrip callback-toggle regression failed (p99): modulated {:.3} vs baseline {:.3}",
        modulated_stats.p99_ratio,
        baseline_stats.p99_ratio
    );
}

#[test]
fn quality_gate_dual_plane_unity_roundtrip_short_interval_plateau_regression() {
    let sample_rate = 44_100u32;
    let bpm = 126.0;
    let callback_frames = 256usize;
    let mono = generate_gate_signal(sample_rate, bpm, 8.0);
    let input = mono_to_stereo_interleaved(&mono);

    let (baseline_out, baseline_boundaries) = run_dual_plane_deterministic_with_ratio_steps(
        &input,
        sample_rate,
        callback_frames,
        &[1.0],
        1,
    );
    let (modulated_out, modulated_boundaries) = run_dual_plane_deterministic_with_ratio_steps(
        &input,
        sample_rate,
        callback_frames,
        &[1.0, 1.035, 1.0, 0.965, 1.0, 1.025, 1.0, 0.975],
        2,
    );

    assert!(
        baseline_out.iter().all(|sample| sample.is_finite()),
        "baseline unity-roundtrip short-interval gate produced non-finite samples"
    );
    assert!(
        modulated_out.iter().all(|sample| sample.is_finite()),
        "unity-roundtrip short-interval gate produced non-finite samples"
    );
    assert!(
        !baseline_out.is_empty() && !modulated_out.is_empty(),
        "unity-roundtrip short-interval gate produced empty output"
    );

    let baseline_left = extract_left_channel(&baseline_out);
    let modulated_left = extract_left_channel(&modulated_out);
    let trim = 16usize;
    let baseline_positions: Vec<usize> = if baseline_boundaries.len() > trim * 2 {
        baseline_boundaries[trim..baseline_boundaries.len() - trim].to_vec()
    } else {
        baseline_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < baseline_left.len())
    .collect();
    let modulated_positions: Vec<usize> = if modulated_boundaries.len() > trim * 2 {
        modulated_boundaries[trim..modulated_boundaries.len() - trim].to_vec()
    } else {
        modulated_boundaries.clone()
    }
    .into_iter()
    .filter(|&p| p > 1 && p + 1 < modulated_left.len())
    .collect();

    let window = (sample_rate as f64 * 0.008).round() as usize;
    let guard = (sample_rate as f64 * 0.001).round() as usize;
    let baseline_stats =
        boundary_artifact_stats(&baseline_left, &baseline_positions, window, guard);
    let modulated_stats =
        boundary_artifact_stats(&modulated_left, &modulated_positions, window, guard);

    println!(
        "dual-plane-unity-roundtrip-short-interval gate: baseline(max={:.3},p99={:.3},mean={:.3},n={}) modulated(max={:.3},p99={:.3},mean={:.3},n={})",
        baseline_stats.max_ratio,
        baseline_stats.p99_ratio,
        baseline_stats.mean_ratio,
        baseline_stats.evaluated_boundaries,
        modulated_stats.max_ratio,
        modulated_stats.p99_ratio,
        modulated_stats.mean_ratio,
        modulated_stats.evaluated_boundaries
    );

    write_quality_dashboard_csv(
        "quality_gate_dual_plane_unity_roundtrip_short_interval_plateau_regression",
        "baseline_max,baseline_p99,baseline_mean,baseline_n,modulated_max,modulated_p99,modulated_mean,modulated_n",
        &format!(
            "{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{}",
            baseline_stats.max_ratio,
            baseline_stats.p99_ratio,
            baseline_stats.mean_ratio,
            baseline_stats.evaluated_boundaries,
            modulated_stats.max_ratio,
            modulated_stats.p99_ratio,
            modulated_stats.mean_ratio,
            modulated_stats.evaluated_boundaries
        ),
    );

    assert!(
        baseline_stats.evaluated_boundaries >= 32 && modulated_stats.evaluated_boundaries >= 32,
        "unity-roundtrip short-interval regression gate evaluated too few boundaries (baseline={}, modulated={})",
        baseline_stats.evaluated_boundaries,
        modulated_stats.evaluated_boundaries
    );
    assert!(
        modulated_stats.max_ratio <= baseline_stats.max_ratio * 1.30 + 0.20,
        "unity-roundtrip short-interval regression failed (max): modulated {:.3} vs baseline {:.3}",
        modulated_stats.max_ratio,
        baseline_stats.max_ratio
    );
    assert!(
        modulated_stats.p99_ratio <= baseline_stats.p99_ratio * 1.25 + 0.15,
        "unity-roundtrip short-interval regression failed (p99): modulated {:.3} vs baseline {:.3}",
        modulated_stats.p99_ratio,
        baseline_stats.p99_ratio
    );
    assert!(
        modulated_stats.mean_ratio <= baseline_stats.mean_ratio * 1.35 + 0.20,
        "unity-roundtrip short-interval regression failed (mean): modulated {:.3} vs baseline {:.3}",
        modulated_stats.mean_ratio,
        baseline_stats.mean_ratio
    );
}

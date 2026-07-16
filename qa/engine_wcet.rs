//! Worst-case callback budget gates for the engine (ROADMAP new
//! Stage 6), extending the callback-budget pattern from
//! `tests/quality_gates.rs`: per-callback wall time over the callback's
//! audio duration, gated on p99.9 (a lone OS scheduling hiccup cannot fail
//! the gate; a systematic worst case does), absolute max reported.
//!
//! Enable timing assertions with `TIMESTRETCH_STRICT_CALLBACK_BUDGET=1`
//! (bound 0.5) or `TIMESTRETCH_CALLBACK_BUDGET_MULTIPLIER=<m>` (bound
//! 0.5·m for slower CI hardware); without either the harness measures and
//! prints only. Run with:
//! `TIMESTRETCH_STRICT_CALLBACK_BUDGET=1 cargo test --features qa-harnesses --release --test engine_wcet -- --nocapture`

use std::time::Instant;

use timestretch::PreAnalysisArtifact;
use timestretch::engine::{Engine, EngineConfig, EngineProfile};

const SAMPLE_RATE: u32 = 44_100;
/// Hard per-callback bound (processing time / callback duration) at the
/// strict multiplier 1.0, gated at p99.9 (ROADMAP Stage 6: proposed 0.5 at
/// 64-frame callbacks on the CI reference machine).
const P999_RATIO_BOUND: f64 = 0.5;

fn budget_multiplier() -> Option<f64> {
    if let Ok(value) = std::env::var("TIMESTRETCH_CALLBACK_BUDGET_MULTIPLIER") {
        if let Ok(parsed) = value.parse::<f64>() {
            if parsed.is_finite() && parsed > 0.0 {
                return Some(parsed);
            }
        }
    }
    let strict = std::env::var("TIMESTRETCH_STRICT_CALLBACK_BUDGET").unwrap_or_default();
    if matches!(
        strict.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    ) {
        return Some(1.0);
    }
    None
}

/// Nearest-rank percentile of `ratios` (see `tests/quality_gates.rs`).
fn percentile(ratios: &mut [f64], pct: f64) -> f64 {
    if ratios.is_empty() {
        return 0.0;
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (pct * (ratios.len() as f64 - 1.0)).round() as usize;
    ratios[rank.min(ratios.len() - 1)]
}

/// EDM-ish fixture: kicks over a tonal bed, plus its onset artifact.
fn fixture(seconds: usize) -> (Vec<f32>, std::sync::Arc<PreAnalysisArtifact>) {
    let frames = SAMPLE_RATE as usize * seconds;
    let mut audio = vec![0.0f32; frames * 2];
    for f in 0..frames {
        let t = f as f64 / SAMPLE_RATE as f64;
        let bed = 0.2 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
            + 0.15 * (2.0 * std::f64::consts::PI * 880.0 * t).sin()
            + 0.1 * (2.0 * std::f64::consts::PI * 55.0 * t).sin();
        audio[f * 2] = bed as f32;
        audio[f * 2 + 1] = 0.9 * bed as f32;
    }
    let beat = SAMPLE_RATE as usize / 2;
    let mut onsets = Vec::new();
    for start in (beat / 2..frames).step_by(beat) {
        onsets.push(start);
        for k in 0..1_500.min(frames - start) {
            let t = k as f64 / SAMPLE_RATE as f64;
            let env = (-t * 20.0).exp();
            let kick = (0.7 * env * (2.0 * std::f64::consts::PI * 60.0 * t).sin()) as f32;
            audio[(start + k) * 2] += kick;
            audio[(start + k) * 2 + 1] += kick;
        }
    }
    let artifact = std::sync::Arc::new(PreAnalysisArtifact {
        version: timestretch::PREANALYSIS_VERSION,
        sample_rate: SAMPLE_RATE,
        bpm: 120.0,
        confidence: 0.95,
        transient_strengths: vec![0.8; onsets.len()],
        beat_positions: onsets.clone(),
        transient_onsets: onsets,
        ..Default::default()
    });
    (audio, artifact)
}

struct WcetReport {
    p99: f64,
    p999: f64,
    max: f64,
    mean: f64,
    callbacks: usize,
}

/// Runs one configuration: full deck workload — tempo ride crossing the
/// corrector threshold, a mid-run warm-start seek (priming callbacks are
/// measured too), artifact attached.
fn measure(profile: EngineProfile, callback_frames: usize, seconds: usize) -> WcetReport {
    let (audio, artifact) = fixture(seconds + 2);
    let handles = Engine::build(EngineConfig {
        sample_rate: SAMPLE_RATE,
        channels: 2,
        profile,
        max_block_frames: callback_frames.clamp(64, 8192),
        pre_analysis: Some(artifact),
        ..EngineConfig::default()
    })
    .unwrap();
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);
    source.set_track_position(0);

    let mut cursor = 0usize;
    let mut out = vec![0.0f32; callback_frames * 2];
    let total_callbacks = SAMPLE_RATE as usize * seconds / callback_frames;
    let warmup = 64usize;
    let seek_at = total_callbacks / 2;
    let preroll = processor.warm_start_preroll_frames();
    let mut ratios: Vec<f64> = Vec::with_capacity(total_callbacks);
    let callback_secs = callback_frames as f64 / SAMPLE_RATE as f64;

    for cb in 0..total_callbacks {
        let t = (cb * callback_frames) as f64 / SAMPLE_RATE as f64;
        // Ride crossing the SOLA/PV threshold (worst selection churn).
        controller.set_tempo_rate(1.0 + 0.1 * (2.0 * std::f64::consts::PI * 0.25 * t).sin());
        if cb == seek_at {
            // Mid-run warm-start seek: priming work is part of the WCET.
            processor.reset();
            let target = SAMPLE_RATE as usize / 2;
            let feed_from = target - preroll.min(target);
            cursor = feed_from * 2;
            source.set_track_position(feed_from as u64);
            controller.warm_start(preroll as u32);
        }
        while source.occupied_frames() < source.demand_hint(callback_frames, 1.2) {
            let end = (cursor + 8_192).min(audio.len());
            let accepted = source.push(&audio[cursor..end]);
            cursor += accepted * 2;
            if accepted == 0 || cursor >= audio.len() {
                cursor = 0; // loop the fixture
                source.set_track_position(0);
            }
        }

        let start = Instant::now();
        processor.process(&mut out);
        let elapsed = start.elapsed().as_secs_f64();
        if cb >= warmup {
            ratios.push(elapsed / callback_secs);
        }
    }

    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let max = ratios.iter().copied().fold(0.0, f64::max);
    let callbacks = ratios.len();
    let p99 = percentile(&mut ratios, 0.99);
    let p999 = percentile(&mut ratios, 0.999);
    WcetReport {
        p99,
        p999,
        max,
        mean,
        callbacks,
    }
}

#[test]
fn engine_worst_case_callback_budget() {
    let multiplier = budget_multiplier();
    if multiplier.is_none() {
        println!(
            "engine-wcet: measuring only (set TIMESTRETCH_STRICT_CALLBACK_BUDGET=1 \
             or TIMESTRETCH_CALLBACK_BUDGET_MULTIPLIER=<m> to gate)"
        );
    }

    let mut failures = Vec::new();
    for (profile, label) in [
        (EngineProfile::Keylock, "keylock"),
        (EngineProfile::Tape, "tape"),
    ] {
        for callback_frames in [64usize, 256, 1024] {
            // Longer runs at small callbacks so p99.9 has support.
            let seconds = if callback_frames == 64 { 24 } else { 16 };
            let report = measure(profile, callback_frames, seconds);
            println!(
                "engine-wcet[{label} @{callback_frames}]: callbacks={} mean={:.3} \
                 p99={:.3} p99.9={:.3} max={:.3}",
                report.callbacks, report.mean, report.p99, report.p999, report.max
            );
            if let Some(m) = multiplier {
                let bound = P999_RATIO_BOUND * m;
                if report.p999 > bound {
                    failures.push(format!(
                        "{label} @{callback_frames}: p99.9 {:.3} > {bound:.3}",
                        report.p999
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "WCET gate failures:\n{}",
        failures.join("\n")
    );
}

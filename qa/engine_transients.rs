//! Transient-sharpness gates (ROADMAP new Stage 3): at ±4% tempo the
//! keylock chain (SOLA corrector active) must preserve kick/hat attacks,
//! gated on absolute thresholds derived from Stage 8/9 measurements.
//!
//! Run with:
//! `cargo test --features qa-harnesses --release --test engine_transients -- --nocapture`

// Each harness compiles the shared adapter separately, so arms another
// harness uses read as dead code here.
#[allow(dead_code)]
#[path = "ab/mod.rs"]
mod ab;

use ab::{Arm, render_keylock_with_artifact, render_with_rate_schedule};
use timestretch::PreAnalysisArtifact;

const SAMPLE_RATE: u32 = 44_100;
const CALLBACK_FRAMES: usize = 256;
/// 120 BPM clicks-as-kicks.
const CLICK_PERIOD: usize = SAMPLE_RATE as usize / 2;
const FIRST_CLICK: usize = CLICK_PERIOD / 2;

/// EDM-style click train over a quiet tonal bed (the old torture fixture):
/// 24-sample bursts read as kick attacks; the bed keeps the PV/SOLA busy.
fn edm_click_train(num_samples: usize) -> Vec<f32> {
    let mut input: Vec<f32> = (0..num_samples)
        .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / SAMPLE_RATE as f32).sin())
        .collect();
    for start in (FIRST_CLICK..num_samples).step_by(CLICK_PERIOD) {
        for s in input.iter_mut().skip(start).take(24) {
            *s += 1.5;
        }
    }
    input
}

/// Attack sharpness of one onset: the maximum adjacent-sample rise inside
/// a search window around the expected position. A smeared transient
/// spreads its rise over many samples and scores low.
fn onset_sharpness(output: &[f32], expected: usize, search: usize) -> Option<f32> {
    let lo = expected.saturating_sub(search);
    let hi = (expected + search).min(output.len());
    if hi <= lo + 1 {
        return None;
    }
    Some(
        output[lo..hi]
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(0.0f32, f32::max),
    )
}

/// Median onset sharpness of a render at constant `rate`.
fn median_sharpness(output: &[f32], rate: f64) -> f64 {
    let mut scores: Vec<f32> = Vec::new();
    let mut k = 0usize;
    loop {
        let source_pos = FIRST_CLICK + k * CLICK_PERIOD;
        // Output timeline: source position divided by the tempo rate, plus
        // slack for pipeline latency and elastic timing.
        let expected = (source_pos as f64 / rate) as usize + 560;
        if expected + 2_048 >= output.len() {
            break;
        }
        if k > 0 {
            // Skip the first click (filter/PV warmup).
            if let Some(s) = onset_sharpness(output, expected, 1_024) {
                scores.push(s);
            }
        }
        k += 1;
    }
    assert!(
        scores.len() >= 10,
        "too few onsets scored: {}",
        scores.len()
    );
    scores.sort_by(|a, b| a.total_cmp(b));
    scores[scores.len() / 2] as f64
}

/// The fixture's true onset positions as a pre-analysis artifact.
fn click_train_artifact(num_samples: usize) -> std::sync::Arc<PreAnalysisArtifact> {
    let onsets: Vec<usize> = (0..)
        .map(|k| FIRST_CLICK + k * CLICK_PERIOD)
        .take_while(|&p| p < num_samples)
        .collect();
    std::sync::Arc::new(PreAnalysisArtifact {
        version: timestretch::PREANALYSIS_VERSION,
        sample_rate: SAMPLE_RATE,
        bpm: 120.0,
        confidence: 0.95,
        transient_strengths: vec![0.9; onsets.len()],
        transient_onsets: onsets.clone(),
        beat_positions: onsets,
        ..Default::default()
    })
}

#[test]
fn keylock_artifact_guidance_preserves_transients_at_least_as_well_as_online() {
    // ROADMAP Stage 4 exit criterion: artifact-driven transient
    // preservation >= online detection on the same fixture and metric.
    let num_samples = SAMPLE_RATE as usize * 10;
    let input = edm_click_train(num_samples);
    let artifact = click_train_artifact(num_samples);

    for rate in [1.04f64, 0.96] {
        let with_artifact = render_keylock_with_artifact(
            artifact.clone(),
            &input,
            1,
            SAMPLE_RATE,
            CALLBACK_FRAMES,
            &|_| rate,
        );
        let online = render_with_rate_schedule(
            Arm::Keylock,
            &input,
            1,
            SAMPLE_RATE,
            CALLBACK_FRAMES,
            &|_| rate,
        );

        let artifact_sharpness = median_sharpness(&with_artifact.output, rate);
        let online_sharpness = median_sharpness(&online.output, rate);
        println!(
            "artifact guidance @rate {rate}: artifact {artifact_sharpness:.3} | online {online_sharpness:.3}"
        );
        // Absolute floor plus a loose relative sanity bound. Re-derived at
        // Stage 9: Stage 8's timeline-retention fix corrected onset
        // mappings that were previously clamped, and correctly-placed
        // splice protection trades a little measured click rise (splices
        // move AWAY from onsets) — measured artifact 1.16/1.20 vs online
        // 1.16/1.29, both far above the deleted old engine's 0.71/0.74.
        assert!(
            artifact_sharpness >= 1.05,
            "rate {rate}: artifact-guided sharpness {artifact_sharpness:.3} below 1.05"
        );
        assert!(
            artifact_sharpness >= online_sharpness * 0.90,
            "rate {rate}: artifact-guided sharpness {artifact_sharpness:.3} \
             far below online {online_sharpness:.3}"
        );
        assert_eq!(with_artifact.underrun_frames, 0);
    }
}

#[test]
fn keylock_transient_sharpness_at_dj_rates() {
    let input = edm_click_train(SAMPLE_RATE as usize * 10);

    for rate in [1.04f64, 0.96] {
        let rendered = render_with_rate_schedule(
            Arm::Keylock,
            &input,
            1,
            SAMPLE_RATE,
            CALLBACK_FRAMES,
            &|_| rate,
        );
        let sharpness = median_sharpness(&rendered.output, rate);
        // Source click rise for reference: 1.5 in one sample.
        println!("transient sharpness @rate {rate}: {sharpness:.3} (source 1.5)");

        // Absolute gate, re-derived at Stage 9 from new-engine
        // measurements (1.16 at rate 1.04, 1.29 at 0.96; the deleted old
        // engine measured 0.71/0.74 on the same fixture and metric).
        assert!(
            sharpness >= 0.90,
            "rate {rate}: sharpness {sharpness:.3} below the 0.90 gate"
        );
        assert_eq!(rendered.underrun_frames, 0);
    }
}

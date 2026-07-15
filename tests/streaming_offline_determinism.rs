//! Same-graph streaming-vs-offline determinism (ROADMAP Stage 8).
//!
//! Replaces the old `streaming_batch_parity.rs`: streaming and batch used
//! different algorithms there, so "parity" meant loose structural
//! similarity. Both now run the same engine graph, so agreement is exact
//! by construction: an offline render and a pull-engine render at the same
//! constant rate with the same artifact must produce identical samples.

use std::sync::Arc;

use timestretch::engine::offline::stretch_offline;
use timestretch::engine::{Engine, EngineConfig, EngineProfile};

const SR: u32 = 44_100;

fn fixture(len: usize) -> Vec<f32> {
    // Tonal bed + kick-ish transients: exercises band split, SOLA splices
    // and the artifact path.
    (0..len)
        .map(|i| {
            let t = i as f64 / SR as f64;
            let tone = 0.4 * (2.0 * std::f64::consts::PI * 330.0 * t).sin()
                + 0.15 * (2.0 * std::f64::consts::PI * 3_700.0 * t).sin();
            let beat_pos = t * 2.0; // 120 BPM
            let phase = beat_pos.fract();
            let kick = if phase < 0.05 {
                0.5 * (1.0 - phase / 0.05) * (2.0 * std::f64::consts::PI * 55.0 * t).sin()
            } else {
                0.0
            };
            (tone + kick) as f32
        })
        .collect()
}

/// Drives the pull engine the way a host callback would (irregular
/// callback sizes) at a constant rate, with the same artifact offline
/// analysis would compute.
fn render_streaming(input: &[f32], rate: f64) -> Vec<f32> {
    let artifact = Arc::new(timestretch::analyze_for_dj(input, SR));
    let handles = Engine::build(EngineConfig {
        sample_rate: SR,
        channels: 1,
        profile: EngineProfile::Keylock,
        initial_tempo_rate: rate,
        pre_analysis: Some(artifact),
        ..EngineConfig::default()
    })
    .unwrap();
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);
    source.set_track_position(0);
    let latency = processor.pipeline_latency_frames();
    let ratio = 1.0 / rate;
    let expected = (input.len() as f64 * ratio).round() as usize;

    // Same terminal treatment as the offline driver: latency + kernel pad.
    let flush = vec![0.0f32; (latency as f64 * rate).ceil() as usize + 64];
    let (mut feed, mut flush_fed, mut finished) = (0usize, 0usize, false);
    let mut collected = Vec::with_capacity(expected + latency + 4_096);
    // Irregular callback sizes — determinism must not depend on chunking.
    let sizes = [64usize, 256, 96, 512, 128, 1_024];
    let mut k = 0usize;
    while collected.len() < expected + latency {
        while feed < input.len() && source.free_frames() > 0 {
            let end = (feed + 4_096).min(input.len());
            feed += source.push(&input[feed..end]);
        }
        if feed >= input.len() {
            while flush_fed < flush.len() && source.free_frames() > 0 {
                flush_fed += source.push(&flush[flush_fed..]);
            }
            if flush_fed >= flush.len() && !finished {
                finished = source.finish();
            }
        }
        let mut out = vec![0.0f32; sizes[k % sizes.len()]];
        k += 1;
        let underruns_before = controller.underrun_frames();
        processor.process(&mut out);
        collected.extend_from_slice(&out);
        if finished && controller.underrun_frames() > underruns_before {
            collected.resize(expected + latency, 0.0);
            break;
        }
    }
    collected.drain(..latency);
    collected.truncate(expected);
    collected
}

#[test]
fn streaming_and_offline_are_sample_identical() {
    let input = fixture(SR as usize * 4);
    for rate in [0.94f64, 1.0, 1.06] {
        let ratio = 1.0 / rate;
        let offline = stretch_offline(&input, 1, SR, ratio, None).unwrap();
        let streaming = render_streaming(&input, rate);
        assert_eq!(
            offline.len(),
            streaming.len(),
            "length differs at rate {rate}"
        );
        for (i, (a, b)) in offline.iter().zip(streaming.iter()).enumerate() {
            assert!(
                a == b,
                "sample {i} differs at rate {rate}: offline {a} vs streaming {b}"
            );
        }
    }
}

#[test]
fn offline_render_is_deterministic_across_runs() {
    let input = fixture(SR as usize * 3);
    let a = stretch_offline(&input, 1, SR, 1.0638, None).unwrap();
    let b = stretch_offline(&input, 1, SR, 1.0638, None).unwrap();
    assert_eq!(a, b, "two offline renders of the same input differ");
}

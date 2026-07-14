//! Fixture-driving adapter for the engine QA harnesses (originally the
//! old-vs-new A/B adapter from ROADMAP Stage 1; the old push-engine arm
//! was deleted with the old engine at Stage 9, and every gate is anchored
//! on absolute thresholds re-derived from new-engine measurements).
//!
//! Drives the pull engine over deterministic fixtures — source samples,
//! callback size, and a per-callback tempo-rate schedule.

use std::sync::Arc;

use timestretch::engine::{Engine, EngineConfig, EngineProfile};
use timestretch::PreAnalysisArtifact;

/// Which profile renders the fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// Pull engine, tape profile (pitch follows tempo).
    NewTape,
    /// Pull engine, keylock profile (two-band: low band follows tempo,
    /// high band corrected).
    NewKeylock,
}

/// Renders the new keylock arm with a pre-analysis artifact attached (feed
/// anchored at track frame 0) — the artifact-first control path.
pub fn render_new_keylock_with_artifact(
    artifact: Arc<PreAnalysisArtifact>,
    input: &[f32],
    channels: usize,
    sample_rate: u32,
    callback_frames: usize,
    rate_at: &dyn Fn(f64) -> f64,
) -> AbRender {
    render_new_inner(
        EngineProfile::Keylock,
        Some(artifact),
        input,
        channels,
        sample_rate,
        callback_frames,
        rate_at,
    )
}

/// Output and bookkeeping from one A/B render.
#[derive(Debug)]
pub struct AbRender {
    /// Interleaved output samples.
    pub output: Vec<f32>,
    /// Mid-stream source underrun frames. Excludes the final callback's
    /// expected end-of-stream shortfall, which the adapter uses to detect
    /// termination and trims from the output.
    pub underrun_frames: u64,
}

/// Streams `input` through the chosen arm at `callback_frames` per callback,
/// setting the tempo rate from `rate_at(t_secs)` before every callback.
///
/// The render stops when the source is exhausted (the new arm's collected
/// output is trimmed to fully-covered callbacks so trailing underrun
/// silence never pollutes metrics).
pub fn render_with_rate_schedule(
    arm: Arm,
    input: &[f32],
    channels: usize,
    sample_rate: u32,
    callback_frames: usize,
    rate_at: &dyn Fn(f64) -> f64,
) -> AbRender {
    match arm {
        Arm::NewTape => render_new(
            EngineProfile::Tape,
            input,
            channels,
            sample_rate,
            callback_frames,
            rate_at,
        ),
        Arm::NewKeylock => render_new(
            EngineProfile::Keylock,
            input,
            channels,
            sample_rate,
            callback_frames,
            rate_at,
        ),
    }
}

fn render_new(
    profile: EngineProfile,
    input: &[f32],
    channels: usize,
    sample_rate: u32,
    callback_frames: usize,
    rate_at: &dyn Fn(f64) -> f64,
) -> AbRender {
    render_new_inner(
        profile,
        None,
        input,
        channels,
        sample_rate,
        callback_frames,
        rate_at,
    )
}

fn render_new_inner(
    profile: EngineProfile,
    artifact: Option<Arc<PreAnalysisArtifact>>,
    input: &[f32],
    channels: usize,
    sample_rate: u32,
    callback_frames: usize,
    rate_at: &dyn Fn(f64) -> f64,
) -> AbRender {
    let handles = Engine::build(EngineConfig {
        sample_rate,
        channels,
        profile,
        initial_tempo_rate: rate_at(0.0),
        max_block_frames: callback_frames.clamp(64, 8192),
        source_capacity_frames: (callback_frames * 16).max(32_768),
        pre_analysis: artifact,
    })
    .expect("engine builds");
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);
    source.set_track_position(0);

    let mut feed_cursor = 0usize;
    let mut finished = false;
    let mut out = vec![0.0f32; callback_frames * channels];
    let mut output: Vec<f32> = Vec::with_capacity(input.len() * 4 + 65_536);

    let mut cb = 0usize;
    loop {
        let t = (cb * callback_frames) as f64 / sample_rate as f64;
        controller.set_tempo_rate(rate_at(t));

        while feed_cursor < input.len()
            && source.occupied_frames() < source.demand_hint(callback_frames, 4.0)
        {
            let end = (feed_cursor + 8192 * channels).min(input.len());
            let accepted = source.push(&input[feed_cursor..end]);
            feed_cursor += accepted * channels;
            if accepted == 0 {
                break;
            }
        }
        if feed_cursor >= input.len() && !finished {
            finished = source.finish();
        }

        let underruns_before = controller.underrun_frames();
        processor.process(&mut out);
        if controller.underrun_frames() > underruns_before {
            // Source exhausted mid-callback: keep only the media portion.
            // This terminal shortfall is expected, not starvation.
            let shortfall = controller.underrun_frames() - underruns_before;
            let missing = shortfall as usize * channels;
            output.extend_from_slice(&out[..out.len() - missing.min(out.len())]);
            return AbRender {
                output,
                underrun_frames: underruns_before,
            };
        }
        output.extend_from_slice(&out);
        cb += 1;

        // Safety valve: 4x realtime worth of callbacks at the slowest rate.
        if cb > 4 * input.len() / (callback_frames.max(1) * channels) + 1024 {
            panic!("A/B render did not terminate");
        }
    }
}

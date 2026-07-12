//! Engine-agnostic A/B adapter (ROADMAP new Stage 1).
//!
//! Drives either the frozen push-based [`StreamProcessor`] or the new
//! pull-based engine over **identical fixtures** — same source samples,
//! same callback size, same per-callback tempo-rate schedule — so every
//! later stage can measure new-vs-old directly on the same metric. Parity
//! is always "new ≥ old on the same fixture and metric", never similarity
//! to the old engine's output.
//!
//! Tempo language: the schedule speaks **rate** (playback speed multiplier,
//! the new engine's native control). The old arm receives the reciprocal as
//! its stretch ratio.

use timestretch::engine::{Engine, EngineConfig, EngineProfile};
use timestretch::{ControlPath, StreamProcessor, StreamProfile, StretchParams};

/// Which engine renders the fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// Frozen push engine on the varispeed-first control path.
    OldVarispeed(StreamProfile),
    /// New pull engine, tape profile (pitch follows tempo).
    NewTape,
}

/// Output and bookkeeping from one A/B render.
#[derive(Debug)]
pub struct AbRender {
    /// Interleaved output samples.
    pub output: Vec<f32>,
    /// Mid-stream source underrun frames (new arm only; the push arm cannot
    /// underrun). Excludes the final callback's expected end-of-stream
    /// shortfall, which the adapter uses to detect termination and trims
    /// from the output.
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
        Arm::OldVarispeed(profile) => render_old(
            profile,
            input,
            channels,
            sample_rate,
            callback_frames,
            rate_at,
        ),
        Arm::NewTape => render_new(input, channels, sample_rate, callback_frames, rate_at),
    }
}

fn render_old(
    profile: StreamProfile,
    input: &[f32],
    channels: usize,
    sample_rate: u32,
    callback_frames: usize,
    rate_at: &dyn Fn(f64) -> f64,
) -> AbRender {
    let params = StretchParams::new(1.0)
        .with_sample_rate(sample_rate)
        .with_channels(channels as u32)
        .with_stream_profile(profile);
    let mut processor = StreamProcessor::new(params);
    processor
        .set_control_path(ControlPath::VarispeedFirst)
        .expect("varispeed control path");

    let chunk = callback_frames * channels;
    let mut output: Vec<f32> = Vec::with_capacity(input.len() * 4 + 65_536);
    for (ci, block) in input.chunks(chunk).enumerate() {
        let t = (ci * callback_frames) as f64 / sample_rate as f64;
        let rate = rate_at(t).clamp(0.25, 4.0);
        processor
            .set_stretch_ratio(1.0 / rate)
            .expect("ratio in varispeed range");
        processor.process_into(block, &mut output).expect("process");
    }
    processor.flush_into(&mut output).expect("flush");
    AbRender {
        output,
        underrun_frames: 0,
    }
}

fn render_new(
    input: &[f32],
    channels: usize,
    sample_rate: u32,
    callback_frames: usize,
    rate_at: &dyn Fn(f64) -> f64,
) -> AbRender {
    let handles = Engine::build(EngineConfig {
        sample_rate,
        channels,
        profile: EngineProfile::Tape,
        initial_tempo_rate: rate_at(0.0),
        max_block_frames: callback_frames.clamp(64, 8192),
        source_capacity_frames: (callback_frames * 16).max(32_768),
    })
    .expect("engine builds");
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);

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

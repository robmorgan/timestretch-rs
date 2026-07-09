//! Warm-start seek/cue/loop torture (ROADMAP Stage 11).
//!
//! Two distinct jump operations:
//! - **Loop wrap / gapless jump**: the input feed splices from loop end to
//!   loop start and the processor keeps rendering continuously
//!   (`notify_source_jump` re-anchors the timeline only). Output is as
//!   seamless as the source splice.
//! - **Cue jump / scrub seek**: the output pipeline is cut and
//!   `warm_start_seek` re-primes from preceding audio, so playback resumes
//!   converged — no cold buffering gap, no phase-vocoder warm-up transient —
//!   with a short declick fade-in.

use std::f32::consts::PI;
use timestretch::{StreamProcessor, StreamProfile, StretchParams};

const SR: u32 = 44_100;
const CHUNK: usize = 256;

fn params(ratio: f64, profile: StreamProfile) -> StretchParams {
    StretchParams::new(ratio)
        .with_sample_rate(SR)
        .with_channels(1)
        .with_stream_profile(profile)
}

fn sine(freq: f32, len: usize, amp: f32) -> Vec<f32> {
    (0..len)
        .map(|i| amp * (2.0 * PI * freq * i as f32 / SR as f32).sin())
        .collect()
}

/// Peak sample-to-sample step of a pure sine (`amp * 2*pi*f / sr`).
fn sine_max_slew(freq: f32, amp: f32) -> f32 {
    amp * 2.0 * PI * freq / SR as f32
}

/// Click bound: the phase vocoder's overlap-add ripple on a pure stretched
/// sine reaches ~5x the ideal slew (measured on a continuous baseline
/// stream), so 6x is the ripple ceiling. A cold restart splices a
/// near-full-scale discontinuity (~40x), well clear of this.
const CLICK_SLEW_MULTIPLE: f32 = 6.0;

fn max_adjacent_diff(samples: &[f32]) -> (usize, f32) {
    let mut worst = (0usize, 0.0f32);
    for (i, w) in samples.windows(2).enumerate() {
        let d = (w[1] - w[0]).abs();
        if d > worst.1 {
            worst = (i, d);
        }
    }
    worst
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

fn stream_range(
    processor: &mut StreamProcessor,
    input: &[f32],
    range: std::ops::Range<usize>,
    output: &mut Vec<f32>,
) {
    for chunk in input[range].chunks(CHUNK) {
        processor.process_into(chunk, output).expect("process");
    }
}

/// Loop wraps are input-side splices: the processor keeps rendering, the
/// caller just feeds from the loop start again. With a loop that is an
/// exact number of sine periods the source is mathematically continuous
/// across the wrap, so the output must be click-free at every seam.
#[test]
fn loop_wrap_seams_are_click_free() {
    let freq = 210.0f32; // 210 Hz at 44.1 kHz = exact 210-sample period
    let amp = 0.5f32;
    let period = SR as usize / 210;
    let input = sine(freq, SR as usize * 4, amp);

    for &(ratio, profile) in &[
        (1.05f64, StreamProfile::Live),
        (0.94, StreamProfile::Live),
        (1.05, StreamProfile::Quality),
    ] {
        let mut processor = StreamProcessor::new(params(ratio, profile));

        // Period-aligned ~1 s loop body.
        let loop_start = 10 * period;
        let loop_len = (SR as usize / period) * period;
        let loop_end = loop_start + loop_len;

        let mut output: Vec<f32> = Vec::with_capacity(input.len() * 8);
        stream_range(&mut processor, &input, 0..loop_end, &mut output);
        for _ in 0..4 {
            // Input-side splice: no state reset, only timeline re-anchor.
            processor.notify_source_jump(loop_start);
            stream_range(&mut processor, &input, loop_start..loop_end, &mut output);
        }

        let bound = sine_max_slew(freq, amp) * CLICK_SLEW_MULTIPLE;
        let skip = 8192;
        let (idx, worst) = max_adjacent_diff(&output[skip..]);
        assert!(
            worst <= bound,
            "ratio {ratio} {profile}: loop-wrap click at output {} (|delta|={:.4} > {:.4})",
            skip + idx,
            worst,
            bound
        );
    }
}

/// Cue jumps: after `warm_start_seek`, output resumes immediately (no cold
/// buffering gap), reaches steady level right away (no warm-up transient),
/// and each post-jump segment is internally click-free.
#[test]
fn cue_drumming_resumes_converged() {
    let freq = 210.0f32;
    let amp = 0.5f32;
    let input = sine(freq, SR as usize * 6, amp);

    let mut processor = StreamProcessor::new(params(1.06, StreamProfile::Live));
    let preroll_frames = processor.warm_start_preroll_frames();
    let cue = SR as usize * 2;

    // Establish steady state and a reference level.
    let mut warmup_out: Vec<f32> = Vec::with_capacity(input.len() * 4);
    stream_range(&mut processor, &input, 0..SR as usize, &mut warmup_out);
    let steady_rms = rms(&warmup_out[warmup_out.len().saturating_sub(4096)..]);
    assert!(steady_rms > 0.1, "steady reference not established");

    let bound = sine_max_slew(freq, amp) * CLICK_SLEW_MULTIPLE;
    let burst = SR as usize / 10; // ~100 ms of playback per cue hit

    for hit in 0..20 {
        processor
            .warm_start_seek(cue, &input[cue - preroll_frames..cue])
            .expect("cue jump");

        // The app cuts the old output (ring flush); each post-jump segment
        // is evaluated on its own.
        let mut segment: Vec<f32> = Vec::with_capacity(burst * 4);
        stream_range(&mut processor, &input, cue..cue + burst, &mut segment);

        // No gap: a cold start would emit nothing for the first
        // `latency_samples()` (1536+) input frames; warm start streams
        // within the normal pipeline cadence.
        assert!(
            segment.len() > burst / 2,
            "hit {hit}: warm start gapped — only {} output samples for {} input frames",
            segment.len(),
            burst
        );

        // No warm-up transient: right after the declick fade the level is
        // already at steady state.
        let post_fade = &segment[512..2048.min(segment.len())];
        let level = rms(post_fade);
        assert!(
            level > steady_rms * 0.7,
            "hit {hit}: post-seek level {:.3} below steady {:.3} — warm-up transient",
            level,
            steady_rms
        );

        // Internally click-free past the fade.
        let (idx, worst) = max_adjacent_diff(&segment[512..]);
        assert!(
            worst <= bound,
            "hit {hit}: click inside post-seek segment at {} (|delta|={:.4} > {:.4})",
            512 + idx,
            worst,
            bound
        );
    }
}

/// Control state — targets AND in-flight glides — survives a warm start.
#[test]
fn warm_start_preserves_control_state() {
    let input = sine(210.0, SR as usize * 4, 0.5);
    let mut processor = StreamProcessor::new(params(1.0, StreamProfile::Live));
    let preroll_frames = processor.warm_start_preroll_frames();

    let mut output = Vec::with_capacity(input.len() * 4);
    stream_range(&mut processor, &input, 0..SR as usize, &mut output);

    // Put a glide in flight: target far from current.
    processor.set_stretch_ratio(1.08).expect("ratio");
    processor.set_pitch_scale(1.05).expect("pitch");
    processor
        .process_into(&input[SR as usize..SR as usize + CHUNK], &mut output)
        .expect("process");
    let current_before = processor.current_stretch_ratio();
    let target_before = processor.target_stretch_ratio();
    let pitch_before = processor.pitch_scale();
    assert!(
        (current_before - target_before).abs() > 0.001,
        "test premise: glide still in flight"
    );

    let target = SR as usize * 2;
    processor
        .warm_start_seek(target, &input[target - preroll_frames..target])
        .expect("warm start");

    assert_eq!(processor.current_stretch_ratio(), current_before);
    assert_eq!(processor.target_stretch_ratio(), target_before);
    assert_eq!(processor.pitch_scale(), pitch_before);

    // Position: the primed preroll's unconsumed residue is still in the
    // input ring; position + in-flight frames lands exactly on the target.
    let in_flight = processor.capacities().0; // mono: samples == frames
    assert_eq!(processor.source_position() + in_flight, target);
}

/// A warm start to a position near the track head (not enough preceding
/// audio for a full preroll) degrades gracefully.
#[test]
fn warm_start_short_preroll_degrades_gracefully() {
    let input = sine(210.0, SR as usize * 2, 0.5);
    let mut processor = StreamProcessor::new(params(1.05, StreamProfile::Live));

    // Seek to frame 300 with only 300 frames of preroll available.
    processor
        .warm_start_seek(300, &input[..300])
        .expect("short preroll");
    let mut output = Vec::with_capacity(input.len() * 4);
    stream_range(&mut processor, &input, 300..SR as usize, &mut output);
    assert!(!output.is_empty(), "short-preroll seek must still stream");

    // Empty preroll behaves like a positioned cold start.
    processor
        .warm_start_seek(SR as usize, &[])
        .expect("empty preroll");
    assert_eq!(processor.source_position(), SR as usize);
}

/// Unity fast path: with no stretch/pitch in play, warm start needs no
/// priming and the passthrough resumes instantly (with only the declick
/// fade applied).
#[test]
fn warm_start_unity_resumes_passthrough() {
    let input = sine(210.0, SR as usize * 3, 0.5);
    let mut processor = StreamProcessor::new(params(1.0, StreamProfile::Live));
    let preroll_frames = processor.warm_start_preroll_frames();

    let mut output = Vec::with_capacity(input.len() * 2);
    stream_range(&mut processor, &input, 0..SR as usize, &mut output);

    let target = SR as usize * 2;
    processor
        .warm_start_seek(target, &input[target - preroll_frames..target])
        .expect("unity warm start");
    assert_eq!(processor.source_position(), target);

    let before = output.len();
    stream_range(&mut processor, &input, target..target + 2048, &mut output);
    // Passthrough: output appears immediately; past the declick fade it is
    // bit-exact.
    assert_eq!(output.len() - before, 2048);
    assert_eq!(&output[before + 512..], &input[target + 512..target + 2048]);
}

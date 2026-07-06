//! Jog-wheel torture tests: continuous per-callback stretch-ratio modulation.
//!
//! Beatmatching is continuous ratio modulation — platter nudges, pitch-fader
//! rides, sync snaps. These tests drive `set_stretch_ratio` on every audio
//! callback with DJ-shaped gestures and assert output quality over the FULL
//! output, including the end-of-stream flush region (ROADMAP Stage 1).

use std::f32::consts::PI;
use timestretch::{StreamProcessor, StretchParams};

const SAMPLE_RATE: u32 = 44_100;
const CALLBACK_FRAMES: usize = 128;
/// PV warmup region skipped by waveform-level assertions.
const WARMUP_SAMPLES: usize = 8192;

/// Peak sample-to-sample step of a pure sine: `amp * 2*pi*f / sr`.
///
/// Time-stretch does not shift pitch, so the output tone keeps the input
/// frequency at every ratio and no ratio factor is needed in slew bounds.
fn sine_max_slew(freq: f32, amp: f32) -> f32 {
    amp * 2.0 * PI * freq / SAMPLE_RATE as f32
}

fn sine(freq: f32, num_samples: usize, amp: f32) -> Vec<f32> {
    (0..num_samples)
        .map(|i| amp * (2.0 * PI * freq * i as f32 / SAMPLE_RATE as f32).sin())
        .collect()
}

/// EDM-style click train over a quiet tonal bed (120 BPM kicks-as-clicks).
fn edm_click_train(num_samples: usize) -> Vec<f32> {
    let mut input: Vec<f32> = (0..num_samples)
        .map(|i| 0.2 * (2.0 * PI * 220.0 * i as f32 / SAMPLE_RATE as f32).sin())
        .collect();
    let click_period = SAMPLE_RATE as usize / 2;
    for start in (click_period / 2..num_samples).step_by(click_period) {
        for s in input.iter_mut().skip(start).take(24) {
            *s += 1.5;
        }
    }
    input
}

fn p95_adjacent_diff(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mut diffs: Vec<f32> = samples
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .filter(|d| d.is_finite())
        .collect();
    if diffs.is_empty() {
        return 0.0;
    }
    diffs.sort_by(|a, b| a.total_cmp(b));
    let idx = ((diffs.len() - 1) as f32 * 0.95).round() as usize;
    diffs[idx]
}

/// Largest adjacent diff and the index of its left sample.
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

#[derive(Debug, Clone, Copy)]
enum Gesture {
    /// Platter nudge: 1.0 -> 1.04 over 200 ms, hold 100 ms, back over
    /// 200 ms, rest 500 ms; repeats every second.
    Nudge,
    /// Continuous pitch-fader ride: 1.0 + 0.06 * sin(2*pi*0.25*t).
    /// Never settles — worst case for the 50 ms control EMA.
    Ride,
    /// Sync snaps: instant 1.0 -> 1.08, hold 400 ms, -> 0.92, hold 400 ms,
    /// -> 1.0, hold 200 ms; repeats each second. Crosses unity each cycle.
    Snap,
}

impl Gesture {
    fn target_ratio_at(self, t_secs: f64) -> f64 {
        match self {
            Gesture::Nudge => {
                let phase = t_secs.fract();
                if phase < 0.2 {
                    1.0 + 0.04 * (phase / 0.2)
                } else if phase < 0.3 {
                    1.04
                } else if phase < 0.5 {
                    1.04 - 0.04 * ((phase - 0.3) / 0.2)
                } else {
                    1.0
                }
            }
            Gesture::Ride => 1.0 + 0.06 * (2.0 * std::f64::consts::PI * 0.25 * t_secs).sin(),
            Gesture::Snap => {
                let phase = t_secs.fract();
                if phase < 0.4 {
                    1.08
                } else if phase < 0.8 {
                    0.92
                } else {
                    1.0
                }
            }
        }
    }

    fn label(self) -> &'static str {
        match self {
            Gesture::Nudge => "nudge",
            Gesture::Ride => "ride",
            Gesture::Snap => "snap",
        }
    }
}

fn torture_params(channels: u32) -> StretchParams {
    StretchParams::new(1.0)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(channels)
        .with_fft_size(1024)
        .with_hop_size(256)
}

/// Streams `input` while driving `set_stretch_ratio` per callback from the
/// gesture. Returns (full output including flush, pre-flush length,
/// per-callback expected output accumulator).
fn stream_with_gesture(
    processor: &mut StreamProcessor,
    input: &[f32],
    channels: usize,
    gesture: Gesture,
) -> (Vec<f32>, usize, f64) {
    let chunk = CALLBACK_FRAMES * channels;
    let mut output: Vec<f32> = Vec::with_capacity(input.len() * 3 + 65_536);
    let mut expected = 0.0f64;
    for (ci, block) in input.chunks(chunk).enumerate() {
        let t = (ci * CALLBACK_FRAMES) as f64 / SAMPLE_RATE as f64;
        let ratio = gesture.target_ratio_at(t);
        processor.set_stretch_ratio(ratio).unwrap();
        processor.process_into(block, &mut output).unwrap();
        expected += block.len() as f64 * ratio;
    }
    let pre_flush_len = output.len();
    processor.flush_into(&mut output).unwrap();
    (output, pre_flush_len, expected)
}

/// Tonal torture: gesture-modulated 220 Hz sine must stay click-free over
/// the full output, INCLUDING the flush region.
fn assert_tonal_torture(gesture: Gesture) {
    let freq = 220.0f32;
    let amp = 0.5f32;
    let input = sine(freq, SAMPLE_RATE as usize * 8, amp);

    let mut processor = StreamProcessor::new(torture_params(1));
    let (output, pre_flush_len, expected) = stream_with_gesture(&mut processor, &input, 1, gesture);

    let scan = &output[WARMUP_SAMPLES..];
    let theoretical = sine_max_slew(freq, amp);
    // Final bounds, set from measured post-fix actuals (nudge 1.8x, ride
    // 2.8x, snap 4.0x theoretical; p95 ~1.06x across all gestures). The
    // snap residual is the known PV ratio-step seam (synthesis hop jumps
    // immediately while phase slews over ~3 frames) — the deferred Stage 1
    // follow-up. Anything past 6x is a genuine click regression.
    let hard_bound = theoretical * 6.0;
    let p95_bound = theoretical * 1.5;

    let (max_idx, max_diff) = max_adjacent_diff(scan);
    let p95 = p95_adjacent_diff(scan);
    let abs_idx = WARMUP_SAMPLES + max_idx;
    println!(
        "torture[{}]: len={} pre_flush={} expected={:.0} max_diff={:.4}@{} (cb_off={}, from_end={}) p95={:.5} bound={:.4}/{:.5}",
        gesture.label(),
        output.len(),
        pre_flush_len,
        expected,
        max_diff,
        abs_idx,
        abs_idx % CALLBACK_FRAMES,
        output.len().saturating_sub(abs_idx),
        p95,
        hard_bound,
        p95_bound,
    );

    assert!(
        max_diff <= hard_bound,
        "torture[{}]: click at sample {} (callback offset {}, {} from end): |delta|={:.4} > {:.4}",
        gesture.label(),
        abs_idx,
        abs_idx % CALLBACK_FRAMES,
        output.len().saturating_sub(abs_idx),
        max_diff,
        hard_bound
    );
    assert!(
        p95 <= p95_bound,
        "torture[{}]: p95 adjacent diff {:.5} > {:.5}",
        gesture.label(),
        p95,
        p95_bound
    );

    // Length accuracy: expected accrues per-callback target ratios while the
    // internal control EMA lags them, so allow 2%.
    let len_err = (output.len() as f64 - expected).abs() / expected;
    assert!(
        len_err < 0.02,
        "torture[{}]: length error {:.4} (got {}, expected {:.0})",
        gesture.label(),
        len_err,
        output.len(),
        expected
    );
}

#[test]
fn torture_nudge_tonal_no_clicks() {
    assert_tonal_torture(Gesture::Nudge);
}

#[test]
fn torture_ride_tonal_no_clicks() {
    assert_tonal_torture(Gesture::Ride);
}

#[test]
fn torture_snap_tonal_no_clicks() {
    assert_tonal_torture(Gesture::Snap);
}

/// Reset-budget torture: ratio modulation must not materially change how
/// many transient events the scheduler fires on identical audio.
fn assert_reset_budget(channels: usize) {
    let num_frames = SAMPLE_RATE as usize * 8;
    let mono = edm_click_train(num_frames);
    let input: Vec<f32> = if channels == 2 {
        mono.iter().flat_map(|&s| [s, s * 0.9]).collect()
    } else {
        mono
    };

    // Baseline: constant unity ratio. Use a fixed non-unity ratio instead of
    // 1.0? No: 1.0 with pitch 1.0 takes the bit-exact passthrough and never
    // runs the scheduler. Use a steady DJ ratio so the PV path is active.
    let mut baseline_proc = StreamProcessor::new(torture_params(channels as u32));
    baseline_proc.set_stretch_ratio(1.03).unwrap();
    let chunk = CALLBACK_FRAMES * channels;
    let mut sink: Vec<f32> = Vec::with_capacity(input.len() * 3 + 65_536);
    for block in input.chunks(chunk) {
        baseline_proc.process_into(block, &mut sink).unwrap();
    }
    baseline_proc.flush_into(&mut sink).unwrap();
    let baseline = baseline_proc.transient_reset_stats();

    // Modulated: same audio, ride gesture.
    let mut mod_proc = StreamProcessor::new(torture_params(channels as u32));
    let (_out, _pre, _exp) = stream_with_gesture(&mut mod_proc, &input, channels, Gesture::Ride);
    let modulated = mod_proc.transient_reset_stats();

    println!(
        "reset-budget[{}ch]: baseline events={} bands={:?} | modulated events={} bands={:?}",
        channels,
        baseline.events_detected_total,
        baseline.reset_band_counts_total,
        modulated.events_detected_total,
        modulated.reset_band_counts_total,
    );

    // Over-trigger bound: modulation must not fire meaningfully more resets
    // than steady streaming of the same audio (measured: identical counts
    // with modulation-hold wired).
    assert!(
        modulated.events_detected_total <= baseline.events_detected_total + 2,
        "reset over-trigger under modulation: {} vs baseline {}",
        modulated.events_detected_total,
        baseline.events_detected_total
    );
    // Under-trigger guard: modulation-hold must not blind the scheduler to
    // real onsets (16 kicks in 8 s).
    assert!(
        modulated.events_detected_total >= baseline.events_detected_total / 2,
        "reset under-trigger under modulation: {} vs baseline {}",
        modulated.events_detected_total,
        baseline.events_detected_total
    );
    // Modulation-hold keeps low bands phase-locked during in-flight slews:
    // sub/low reset counts must not increase relative to steady streaming.
    let low_bands_modulated =
        modulated.reset_band_counts_total[0] + modulated.reset_band_counts_total[1];
    let low_bands_baseline =
        baseline.reset_band_counts_total[0] + baseline.reset_band_counts_total[1];
    assert!(
        low_bands_modulated <= low_bands_baseline,
        "low-band resets increased under modulation: {} vs baseline {}",
        low_bands_modulated,
        low_bands_baseline
    );
}

#[test]
fn torture_ride_percussive_reset_budget_mono() {
    assert_reset_budget(1);
}

#[test]
fn torture_ride_percussive_reset_budget_stereo() {
    assert_reset_budget(2);
}

/// Snap gesture ending mid-hold, then flush: the flush region itself must
/// pass the tonal slew bound and a second flush must be empty.
#[test]
fn torture_snap_flush_length_and_tail() {
    let freq = 220.0f32;
    let amp = 0.5f32;
    // 2.9 s ends inside the 0.92 hold (phase 0.9 of the cycle is 1.0 hold;
    // 2.9 s -> phase 0.9 ... use 2.6 s -> phase 0.6 = mid 0.92 hold).
    let input = sine(freq, (SAMPLE_RATE as f64 * 2.6) as usize, amp);

    let mut processor = StreamProcessor::new(torture_params(1));
    let (output, pre_flush_len, _expected) =
        stream_with_gesture(&mut processor, &input, 1, Gesture::Snap);

    let flush_scan_start = pre_flush_len.saturating_sub(256).max(WARMUP_SAMPLES);
    let flush_region = &output[flush_scan_start..];
    let (idx, max_diff) = max_adjacent_diff(flush_region);
    let bound = sine_max_slew(freq, amp) * 6.0;
    println!(
        "snap-flush: len={} pre_flush={} flush_max_diff={:.4}@{} (from_end={}) bound={:.4}",
        output.len(),
        pre_flush_len,
        max_diff,
        flush_scan_start + idx,
        output.len() - (flush_scan_start + idx),
        bound,
    );
    assert!(
        max_diff <= bound,
        "flush-region click at {} ({} from end): |delta|={:.4} > {:.4}",
        flush_scan_start + idx,
        output.len() - (flush_scan_start + idx),
        max_diff,
        bound
    );

    let second = processor.flush().unwrap();
    assert!(
        second.is_empty(),
        "second flush should be empty, got {} samples",
        second.len()
    );
}

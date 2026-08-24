//! Stage 12 soak harness: randomized (seeded, deterministic) deck-gesture
//! marathon over the pull engine.
//!
//! Composes the torture-test gesture generators
//! (`tests/engine_modulation_torture.rs`) into a random sequence of tempo
//! rides/nudges/snaps, mid-play seeks (the reset + `warm_start` protocol
//! from `tests/engine_realtime_allocations.rs`), live keylock toggles, and
//! per-segment artifact swaps (profile/artifact changes are seek-priced
//! rebuilds by design — the engine has no live artifact morph).
//!
//! Gates, per segment:
//! - zero source underruns (the feeder keeps the ring topped up);
//! - every output sample finite, including seek/priming regions;
//! - zero clicks by the torture tests' max-adjacent-diff bound
//!   (`slew(max_gesture_rate) * 3`), excluding a settle window at segment
//!   start and after each seek (priming silence + declick fade-in);
//! - bounded drift, in two parts (the engine's stash is bounded — drift
//!   must not accumulate with time). Mid-gesture, between settle
//!   checkpoints, source frames consumed track the integral of the
//!   commanded tempo rate within [`drift_bound_frames`] — a constant,
//!   but a per-profile one, since the elastic reservoir a head may hold
//!   is a property of its topology. Then every segment ends with a
//!   quiescent unity-rate tail, where the same accounting must land
//!   within the far tighter [`SETTLED_DRIFT_BOUND_FRAMES`]: buffering
//!   swings settle out, a leak does not.
//!
//! Zero-allocation on the audio thread is machine-verified separately in
//! `tests/engine_realtime_allocations.rs` and is not re-gated here.
//!
//! The CI-bounded variant streams ~60 s of audio time. The hours-equivalent
//! recipe is `soak_long_hours_equivalent` (`#[ignore]`): one hour of audio
//! time across 60 segments with ~8x the seek/toggle count —
//!
//! ```text
//! cargo test --release --features qa-harnesses --test soak -- --ignored --nocapture
//! ```
//!
//! Both variants are fully deterministic from the fixed seeds below; to
//! widen the campaign, set `TIMESTRETCH_FUZZ_SEED` (each value is an
//! independent, reproducible hour — the scheduled CI campaign passes its
//! run id and logs it).

use timestretch::engine::{Engine, EngineConfig, EngineProfile};
use timestretch::{PREANALYSIS_VERSION, PreAnalysisArtifact};

const SAMPLE_RATE: u32 = 44_100;
const CALLBACK_FRAMES: usize = 128;
/// 210 Hz at 44.1 kHz: exactly 210 frames per period, so the infinite
/// test tone is a pure function of the absolute track frame.
const TONE_PERIOD_FRAMES: u64 = 210;
const TONE_FREQ: f64 = SAMPLE_RATE as f64 / TONE_PERIOD_FRAMES as f64;
const TONE_AMP: f32 = 0.5;
/// Output frames excluded from the click gate at segment start (pipeline
/// delay + corrector settle, the torture tests' prefix).
const START_SETTLE_FRAMES: usize = 16_384;
/// Output frames excluded after a seek: warm-start priming renders
/// silence, then a declick fade-in; give it the same settle again.
const SEEK_SETTLE_FRAMES: usize = 32_768;

const CI_SEED: u64 = 0xD15C_0517_50AC_0001;
const LONG_SEED: u64 = 0xD15C_0517_50AC_1000;

/// Campaign widening: XORs an optional `TIMESTRETCH_FUZZ_SEED` (decimal
/// u64, set by the scheduled CI campaign to its run id) into the fixed
/// per-test seed. Unset, every run is byte-deterministic from the
/// constants; set, each campaign is an independent — but reproducible,
/// the workflow logs the value — random exploration.
fn campaign_seed(base: u64) -> u64 {
    match std::env::var("TIMESTRETCH_FUZZ_SEED") {
        Ok(v) => base ^ v.trim().parse::<u64>().unwrap_or(0),
        Err(_) => base,
    }
}

/// xorshift64* — deterministic, dependency-free (same generator as
/// qa/robustness.rs).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.unit_f64() * (hi - lo)
    }
}

/// The infinite source: a pure tone as a function of absolute track frame.
#[inline]
fn tone_at(track_frame: u64) -> f32 {
    let phase = (track_frame % TONE_PERIOD_FRAMES) as f64 / TONE_PERIOD_FRAMES as f64;
    TONE_AMP * (std::f64::consts::TAU * phase).sin() as f32
}

/// Peak adjacent-sample step of the tone at `max_rate` (tape/varispeed
/// worst case; corrected output stays at the source pitch and sits well
/// inside the same bound).
fn tone_max_slew(max_rate: f64) -> f32 {
    TONE_AMP * (std::f64::consts::TAU * TONE_FREQ * max_rate / SAMPLE_RATE as f64) as f32
}

/// Randomized deck gestures, parameterized versions of the torture
/// generators (Nudge / Ride / Snap / Hold).
#[derive(Debug, Clone, Copy)]
enum Gesture {
    Hold {
        rate: f64,
    },
    /// Sinusoidal fader ride around `center`.
    Ride {
        center: f64,
        depth: f64,
        hz: f64,
    },
    /// Instant sync snaps between two rates.
    Snap {
        a: f64,
        b: f64,
        period_s: f64,
    },
    /// Platter nudge: ramp up over 200 ms, hold, ramp back, rest (1 Hz).
    Nudge {
        peak: f64,
    },
}

impl Gesture {
    fn random(rng: &mut Rng, lo: f64, hi: f64) -> Self {
        let span = hi - lo;
        match rng.below(4) {
            0 => Gesture::Hold {
                rate: rng.range_f64(lo, hi),
            },
            1 => {
                let center = (lo + hi) * 0.5;
                Gesture::Ride {
                    center,
                    depth: rng.range_f64(0.2, 0.5) * span * 0.5,
                    hz: rng.range_f64(0.1, 0.5),
                }
            }
            2 => Gesture::Snap {
                a: rng.range_f64(lo, hi),
                b: rng.range_f64(lo, hi),
                period_s: rng.range_f64(0.3, 0.8),
            },
            _ => Gesture::Nudge {
                peak: rng.range_f64((lo + hi) * 0.5, hi),
            },
        }
    }

    fn rate_at(self, t_secs: f64) -> f64 {
        match self {
            Gesture::Hold { rate } => rate,
            Gesture::Ride { center, depth, hz } => {
                center + depth * (std::f64::consts::TAU * hz * t_secs).sin()
            }
            Gesture::Snap { a, b, period_s } => {
                if (t_secs / period_s) as u64 % 2 == 0 {
                    a
                } else {
                    b
                }
            }
            Gesture::Nudge { peak } => {
                let phase = t_secs.fract();
                if phase < 0.2 {
                    1.0 + (peak - 1.0) * (phase / 0.2)
                } else if phase < 0.3 {
                    peak
                } else if phase < 0.5 {
                    peak - (peak - 1.0) * ((phase - 0.3) / 0.2)
                } else {
                    1.0
                }
            }
        }
    }
}

/// Synthetic artifacts for the swap axis: the engine treats a provided
/// artifact as authoritative, so segments alternate between none, a dense
/// 126 BPM grid, and a sparse onset-only variant.
fn artifact_variant(variant: usize, rng: &mut Rng) -> Option<PreAnalysisArtifact> {
    // 126 BPM at 44.1 kHz: exactly 21_000 frames per beat (100 periods of
    // the test tone), covering 20 minutes of track.
    let beat = 21_000u64;
    let beats: Vec<u64> = (0..3_600).map(|i| i * beat).collect();
    match variant % 3 {
        0 => None,
        1 => Some(PreAnalysisArtifact {
            version: PREANALYSIS_VERSION,
            sample_rate: SAMPLE_RATE,
            bpm: 126.0,
            confidence: 0.9,
            beat_positions: beats.iter().map(|&b| b as usize).collect(),
            beat_positions_fractional: beats.iter().map(|&b| b as f64).collect(),
            downbeat_beat_indices: (0..beats.len() / 4).map(|i| i * 4).collect(),
            transient_onsets: beats.iter().map(|&b| b as usize).collect(),
            transient_strengths: beats
                .iter()
                .map(|_| 0.6 + 0.4 * rng.unit_f64() as f32)
                .collect(),
            onset_band_flux: beats.iter().map(|_| [1.0, 0.5, 0.2, 0.1]).collect(),
            analysis_hop_size: 512,
            ..Default::default()
        }),
        _ => Some(PreAnalysisArtifact {
            version: PREANALYSIS_VERSION,
            sample_rate: SAMPLE_RATE,
            bpm: 126.0,
            confidence: 0.55,
            beat_positions: beats.iter().step_by(4).map(|&b| b as usize).collect(),
            beat_positions_fractional: beats.iter().step_by(4).map(|&b| b as f64).collect(),
            downbeat_beat_indices: vec![0],
            transient_onsets: beats.iter().step_by(4).map(|&b| b as usize).collect(),
            transient_strengths: beats.iter().step_by(4).map(|_| 1.0).collect(),
            onset_band_flux: Vec::new(), // deliberately not parallel: robustness axis
            analysis_hop_size: 512,
            ..Default::default()
        }),
    }
}

struct SegmentReport {
    profile: EngineProfile,
    seeks: usize,
    toggles: usize,
    max_diff: f32,
    bound: f32,
    frames: usize,
    max_drift: f64,
    settled_drift: f64,
}

/// Bounded-drift gate, part 1 — the PEAK bound, sampled mid-gesture at
/// every seek and at the end of the ride. Between checkpoints (set once
/// a settle window expires, cleared at every seek), the source frames
/// consumed must track the integral of the commanded tempo rate. The
/// engine holds a bounded stash — stage FIFOs, resampler kernel
/// margins, SOLA's elastic cursor, the profile latency — so the allowed
/// error is a CONSTANT, not per-time. It is NOT the same constant for
/// every profile: the stash a head is entitled to hold is a property of
/// its topology, so the bound is derived per profile below.
///
/// Sampled mid-gesture this is a coarse gate — a head is free to swing
/// its whole reservoir under a rate step and swing it back. The tight
/// leak detector is part 2, [`SETTLED_DRIFT_BOUND_FRAMES`].
fn drift_bound_frames(profile: EngineProfile) -> f64 {
    match profile {
        // The direct-ratio wide head (`engine/stages/wide_pv_head.rs`)
        // runs two arms and is entitled to a large elastic reservoir of
        // rendered-but-unemitted OUTPUT: `ARM_SURPLUS_MAX` (8_192) of
        // inaudible-arm surplus plus the per-arm pending cap
        // (8 * MAX_OUT_PER_HOP = 4_096). A rate step revalues that
        // backlog in SOURCE frames by up to the profile's rate ceiling
        // (2.0), and the swing shows up here in full.
        //
        // Learned 2026-08-24: the flat 2_048 below was calibrated on
        // 2026-08-13 against the wide path's PREVIOUS topology
        // (varispeed prepass -> PV -> resampler) hours before
        // `WidePvHead` was wired in as the wide head, and was never
        // re-derived. The re-seeded campaign eventually drew a seed
        // whose seek landed on a -2_926 frame excursion (run
        // 32688905856). The excursion returned to ~0 on its own — no
        // frames were lost.
        EngineProfile::WideKeylock => (8_192.0 + 8.0 * 512.0) * 2.0,
        // Narrow profiles: measured worst case (2026-08-13) 28 frames
        // over the CI soak's ~5 s runs, 197 frames (4.5 ms) over the
        // hour-equivalent soak's 60 s runs. ~10x that, and a steady
        // leak of one frame per callback would still cross it within
        // ~6 s of audio time.
        _ => 2_048.0,
    }
}

/// Bounded-drift gate, part 2 — the SETTLED bound, and the real leak
/// detector. After the ride, every segment holds unity rate for
/// [`TAIL_DRAIN_FRAMES`] so the elastic reservoirs reach steady state,
/// then re-anchors and measures over [`TAIL_MEASURE_FRAMES`]. With
/// nothing swinging, a head that merely BUFFERS lands within a few
/// dozen frames; a head that LEAKS cannot. This gate is roughly two
/// orders of magnitude tighter than the peak bound on the wide profile,
/// which is what makes an hour of gesturing worth soaking.
///
/// Measured 2026-08-24 over four independent campaign hours (240
/// segments): the narrow profiles land on exactly 0 every time, the wide
/// head within ±128 frames. The bound is 4x the wide worst case.
const SETTLED_DRIFT_BOUND_FRAMES: f64 = 512.0;

/// Unity-rate tail after the ride: drain to steady state, then measure.
///
/// The measurement window is deliberately long (5 s). A head's settled
/// error is a QUANTIZATION offset — hop and feed-chunk granularity, a
/// constant — while a leak accumulates with the window, so the window
/// length is the gate's sensitivity dial, not the bound.
const TAIL_DRAIN_FRAMES: usize = 32_768;
const TAIL_MEASURE_FRAMES: usize = 5 * 44_100;

/// Accounting anchor for the drift gate. Consumed frames between the
/// anchor and now = frames pushed into the ring minus the ring-fill
/// growth (the feeder keeps the ring topped, so both are observable).
struct DriftAnchor {
    pushed: u64,
    free: usize,
    expected: f64,
}

fn measured_drift(anchor: &DriftAnchor, pushed_total: u64, free_now: usize) -> f64 {
    let pushed_delta = (pushed_total - anchor.pushed) as f64;
    let consumed = pushed_delta + free_now as f64 - anchor.free as f64;
    consumed - anchor.expected
}

/// Streams one engine lifetime (one profile + artifact combination) for
/// `secs` of audio time, driving random gestures every callback with
/// random seeks and keylock toggles, and gates the collected output.
fn run_segment(
    rng: &mut Rng,
    profile: EngineProfile,
    artifact: Option<PreAnalysisArtifact>,
    secs: usize,
    seeks_target: usize,
) -> SegmentReport {
    // Wide profile exercises the full deck range; the narrow profiles stay
    // inside the keylock's fully-corrected window like the torture rides.
    let (rate_lo, rate_hi) = match profile {
        EngineProfile::WideKeylock => (0.5, 2.0),
        _ => (0.9, 1.12),
    };

    let peak_bound = drift_bound_frames(profile);

    let handles = Engine::build(EngineConfig {
        sample_rate: SAMPLE_RATE,
        channels: 1,
        profile,
        pre_analysis: artifact.map(std::sync::Arc::new),
        ..EngineConfig::default()
    })
    .expect("soak engine builds");
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);

    let callbacks = secs * SAMPLE_RATE as usize / CALLBACK_FRAMES;
    let mut track_frame: u64 = 0;
    source.set_track_position(track_frame);

    // Seek schedule: `seeks_target` random callbacks, past the initial
    // settle and clear of the segment end.
    let mut seek_cbs: Vec<usize> = (0..seeks_target)
        .map(|_| 400 + rng.below(callbacks.saturating_sub(800).max(1)))
        .collect();
    seek_cbs.sort_unstable();
    seek_cbs.dedup();

    let mut gesture = Gesture::random(rng, rate_lo, rate_hi);
    let mut gesture_until = 0usize;
    let mut scratch = vec![0.0f32; 1_024];
    let mut out = vec![0.0f32; CALLBACK_FRAMES];
    let mut collected: Vec<f32> = Vec::with_capacity(callbacks * CALLBACK_FRAMES);
    // Half-open output-frame ranges excluded from the click gate: the
    // segment-start settle plus one window per seek.
    let mut excluded: Vec<(usize, usize)> = vec![(0, START_SETTLE_FRAMES)];
    let mut seeks = 0usize;
    let mut toggles = 0usize;
    // Drift accounting: anchored after each settle window (start and
    // post-seek) so warm-start priming consumption stays out of the
    // integral; measured and asserted at every seek and at segment end.
    let mut pushed_total: u64 = 0;
    let mut settle_until_cb = START_SETTLE_FRAMES / CALLBACK_FRAMES + 1;
    let mut drift_anchor: Option<DriftAnchor> = None;
    let mut max_drift = 0.0f64;

    for cb in 0..callbacks {
        // New gesture every 1–3 s.
        if cb >= gesture_until {
            gesture = Gesture::random(rng, rate_lo, rate_hi);
            gesture_until = cb
                + (SAMPLE_RATE as usize / CALLBACK_FRAMES)
                + rng.below(2 * SAMPLE_RATE as usize / CALLBACK_FRAMES);
        }

        // Seek: reset + re-anchor + warm-start, feeding the preroll of
        // audio preceding the period-aligned target.
        if seek_cbs.first() == Some(&cb) {
            seek_cbs.remove(0);
            seeks += 1;
            if let Some(anchor) = drift_anchor.take() {
                let drift = measured_drift(&anchor, pushed_total, source.free_frames());
                max_drift = max_drift.max(drift.abs());
                assert!(
                    drift.abs() <= peak_bound,
                    "soak[{profile:?}]: source consumption drifted {drift:.0} frames from \
                     the tempo integral before seek {seeks} (bound {peak_bound})"
                );
            }
            settle_until_cb = cb + SEEK_SETTLE_FRAMES / CALLBACK_FRAMES + 1;
            let preroll = processor.warm_start_preroll_frames() as u64;
            processor.reset();
            let target =
                (rng.below(10_000) as u64 + 1) * TONE_PERIOD_FRAMES + TONE_PERIOD_FRAMES * 20_000;
            let start = target.saturating_sub(preroll);
            source.set_track_position(start);
            controller.warm_start(preroll as u32);
            track_frame = start;
            excluded.push((collected.len(), collected.len() + SEEK_SETTLE_FRAMES));
        }

        // Keylock toggle: ~every 2 s on average (ignored by profiles
        // without a keylock stage — that indifference is itself under
        // test).
        if rng.below(2 * SAMPLE_RATE as usize / CALLBACK_FRAMES) == 0 {
            toggles += 1;
            controller.set_keylock(rng.below(2) == 0);
        }

        let t = (cb * CALLBACK_FRAMES) as f64 / SAMPLE_RATE as f64;
        let rate = gesture.rate_at(t);
        controller.set_tempo_rate(rate);

        // Keep the ring topped up (covers playback plus priming budget).
        while source.free_frames() >= scratch.len() {
            for (i, s) in scratch.iter_mut().enumerate() {
                *s = tone_at(track_frame + i as u64);
            }
            let accepted = source.push(&scratch);
            track_frame += accepted as u64;
            pushed_total += accepted as u64;
            if accepted == 0 {
                break;
            }
        }

        processor.process(&mut out);
        collected.extend_from_slice(&out);

        match &mut drift_anchor {
            Some(anchor) => anchor.expected += rate * CALLBACK_FRAMES as f64,
            None if cb >= settle_until_cb => {
                drift_anchor = Some(DriftAnchor {
                    pushed: pushed_total,
                    free: source.free_frames(),
                    expected: 0.0,
                });
            }
            None => {}
        }
    }

    if let Some(anchor) = drift_anchor.take() {
        let drift = measured_drift(&anchor, pushed_total, source.free_frames());
        max_drift = max_drift.max(drift.abs());
        assert!(
            drift.abs() <= peak_bound,
            "soak[{profile:?}]: source consumption drifted {drift:.0} frames from \
             the tempo integral over the final run (bound {peak_bound})"
        );
    }

    // Quiescent tail: hold unity rate so every elastic reservoir reaches
    // steady state, then re-anchor and measure. Buffering swings settle
    // out here; a leak does not. The tail is excluded from the click
    // gate: the ride's own Snap gestures already put instant full-range
    // rate steps under that scrutiny, so the tail's step to unity earns
    // no exemption.
    controller.set_tempo_rate(1.0);
    let tail_cbs = (TAIL_DRAIN_FRAMES + TAIL_MEASURE_FRAMES) / CALLBACK_FRAMES;
    let drain_cbs = TAIL_DRAIN_FRAMES / CALLBACK_FRAMES;
    let mut tail_anchor: Option<DriftAnchor> = None;
    for cb in 0..tail_cbs {
        while source.free_frames() >= scratch.len() {
            for (i, s) in scratch.iter_mut().enumerate() {
                *s = tone_at(track_frame + i as u64);
            }
            let accepted = source.push(&scratch);
            track_frame += accepted as u64;
            pushed_total += accepted as u64;
            if accepted == 0 {
                break;
            }
        }

        processor.process(&mut out);
        collected.extend_from_slice(&out);

        match &mut tail_anchor {
            Some(anchor) => anchor.expected += CALLBACK_FRAMES as f64,
            None if cb >= drain_cbs => {
                tail_anchor = Some(DriftAnchor {
                    pushed: pushed_total,
                    free: source.free_frames(),
                    expected: 0.0,
                });
            }
            None => {}
        }
    }

    let anchor = tail_anchor.expect("quiescent tail anchors");
    let settled_drift = measured_drift(&anchor, pushed_total, source.free_frames());
    assert!(
        settled_drift.abs() <= SETTLED_DRIFT_BOUND_FRAMES,
        "soak[{profile:?}]: source consumption leaked {settled_drift:.0} frames over the \
         quiescent unity-rate tail (bound {SETTLED_DRIFT_BOUND_FRAMES}) — the engine's \
         stash is bounded, so a settled run must not drift"
    );

    assert_eq!(
        controller.underrun_frames(),
        0,
        "soak[{profile:?}]: fed engine must not underrun"
    );
    assert_eq!(
        controller.dropped_events(),
        0,
        "soak[{profile:?}]: mailbox overflow"
    );
    assert_eq!(
        controller.retargets_degraded(),
        0,
        "soak[{profile:?}]: timestamped retargets silently degraded (issue #45 class)"
    );

    // Finiteness everywhere, including priming/declick regions.
    for (i, &s) in collected.iter().enumerate() {
        assert!(
            s.is_finite(),
            "soak[{profile:?}]: non-finite output at frame {i}"
        );
    }

    // Click gate outside the settle windows: a pair (i, i+1) is checked
    // only when it lies past every exclusion range covering it. Ranges
    // are pushed in ascending start order, so a single forward sweep
    // suffices.
    let bound = tone_max_slew(rate_hi) * 3.0;
    let mut max_diff = 0.0f32;
    let mut max_at = 0usize;
    let mut range_idx = 0usize;
    for (i, pair) in collected.windows(2).enumerate() {
        while range_idx < excluded.len() && i >= excluded[range_idx].1 {
            range_idx += 1;
        }
        if range_idx < excluded.len() && i + 1 >= excluded[range_idx].0 && i < excluded[range_idx].1
        {
            continue;
        }
        let d = (pair[1] - pair[0]).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }

    assert!(
        max_diff <= bound,
        "soak[{profile:?}]: click at output frame {max_at} — max adjacent diff \
         {max_diff:.5} > bound {bound:.5}"
    );

    SegmentReport {
        profile,
        seeks,
        toggles,
        max_diff,
        bound,
        frames: collected.len(),
        max_drift,
        settled_drift,
    }
}

/// Drives `total_secs` of audio time split into `segment_secs` segments,
/// cycling profiles and artifact variants (the artifact-swap axis).
fn soak(seed: u64, total_secs: usize, segment_secs: usize, seeks_per_segment: usize) {
    let mut rng = Rng::new(seed);
    let profiles = [
        EngineProfile::Keylock,
        EngineProfile::Tape,
        EngineProfile::WideKeylock,
    ];
    let segments = total_secs.div_ceil(segment_secs);
    for seg in 0..segments {
        let profile = profiles[seg % profiles.len()];
        let artifact = artifact_variant(seg, &mut rng);
        let report = run_segment(&mut rng, profile, artifact, segment_secs, seeks_per_segment);
        println!(
            "soak segment {seg}: {:?} frames={} seeks={} toggles={} max_diff={:.5} \
             bound={:.5} max_drift={:.0} settled_drift={:.0}",
            report.profile,
            report.frames,
            report.seeks,
            report.toggles,
            report.max_diff,
            report.bound,
            report.max_drift,
            report.settled_drift
        );
    }
}

/// CI-bounded soak: ~60 s of audio time (6 x 10 s segments), 2 seeks per
/// segment, deterministic from `CI_SEED`.
#[test]
fn soak_ci_bounded_randomized_gestures() {
    soak(campaign_seed(CI_SEED), 60, 10, 2);
}

/// Hours-equivalent soak (see module docs for the invocation): one hour of
/// audio time, 60 x 60 s segments, 8 seeks per segment. Deterministic from
/// `LONG_SEED`; change the seed for an independent campaign hour.
#[test]
#[ignore = "hours-equivalent soak; run explicitly in release (see module docs)"]
fn soak_long_hours_equivalent() {
    soak(campaign_seed(LONG_SEED), 3_600, 60, 8);
}

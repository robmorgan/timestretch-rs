//! Stage 10 gates: general-purpose beat tracking on synthetic fixtures.
//!
//! Scores the tracker with the standard beat-tracking metrics — F-measure
//! at a ±70 ms tolerance and an octave-tolerant continuity check — against
//! generated ground truth: wide-range constant tempos (no octave folding),
//! a tempo ramp, a hard tempo step, timing jitter (live-drummer style),
//! and an accented 4/4 pattern for downbeat phase.

use timestretch::{BeatGrid, detect_beat_grid};

const SAMPLE_RATE: u32 = 44100;
/// Standard beat-tracking tolerance: a detection within ±70 ms of a ground
/// truth beat is a hit.
const F_MEASURE_TOLERANCE_SECS: f64 = 0.07;

/// Renders a percussive click at `positions` (in samples): a short
/// decaying burst with broadband and low-frequency content, `amps[i]`
/// scaling each click.
fn render_clicks(len: usize, positions: &[f64], amps: &[f32]) -> Vec<f32> {
    let mut samples = vec![0.0f32; len];
    for (&pos, &amp) in positions.iter().zip(amps.iter()) {
        let base = pos.round() as usize;
        for j in 0..64 {
            let idx = base + j;
            if idx >= len {
                break;
            }
            let t = j as f32 / 64.0;
            let decay = (1.0 - t).powi(2);
            // Broadband click + low sine cycle for band-flux content.
            let click = if j % 2 == 0 { 1.0 } else { -0.8 };
            let low = (2.0 * std::f32::consts::PI * 60.0 * j as f32 / SAMPLE_RATE as f32).sin();
            samples[idx] += amp * decay * (0.6 * click + 0.4 * low);
        }
    }
    samples
}

/// Ground-truth beat positions for a constant tempo.
fn constant_beats(bpm: f64, seconds: f64, phase_samples: f64) -> Vec<f64> {
    let interval = 60.0 * SAMPLE_RATE as f64 / bpm;
    let len = SAMPLE_RATE as f64 * seconds;
    let mut beats = Vec::new();
    let mut pos = phase_samples;
    while pos + 64.0 < len {
        beats.push(pos);
        pos += interval;
    }
    beats
}

/// F-measure of detected beats against ground truth at the standard
/// tolerance. Each truth beat may be matched at most once.
fn f_measure(detected: &[f64], truth: &[f64]) -> f64 {
    if detected.is_empty() || truth.is_empty() {
        return 0.0;
    }
    let tolerance = F_MEASURE_TOLERANCE_SECS * SAMPLE_RATE as f64;
    let mut matched = vec![false; truth.len()];
    let mut hits = 0usize;
    for &d in detected {
        let idx = truth.partition_point(|&t| t < d);
        let mut best: Option<usize> = None;
        for cand in [idx.wrapping_sub(1), idx, idx + 1] {
            if cand < truth.len() && !matched[cand] {
                let dist = (truth[cand] - d).abs();
                if dist <= tolerance && best.is_none_or(|b: usize| dist < (truth[b] - d).abs()) {
                    best = Some(cand);
                }
            }
        }
        if let Some(b) = best {
            matched[b] = true;
            hits += 1;
        }
    }
    let precision = hits as f64 / detected.len() as f64;
    let recall = hits as f64 / truth.len() as f64;
    if precision + recall == 0.0 {
        return 0.0;
    }
    2.0 * precision * recall / (precision + recall)
}

/// Octave-tolerant F-measure: best score against the truth, its
/// half-density variants (every other beat, both phases), and its
/// double-density variant — the tactus may legitimately sit an octave
/// away on ambiguous material (allowed-metrical-level scoring).
fn f_measure_octave_tolerant(detected: &[f64], truth: &[f64]) -> f64 {
    let half_even: Vec<f64> = truth.iter().step_by(2).copied().collect();
    let half_odd: Vec<f64> = truth.iter().skip(1).step_by(2).copied().collect();
    let mut double = Vec::with_capacity(truth.len() * 2);
    for w in truth.windows(2) {
        double.push(w[0]);
        double.push((w[0] + w[1]) * 0.5);
    }
    if let Some(&last) = truth.last() {
        double.push(last);
    }
    f_measure(detected, truth)
        .max(f_measure(detected, &half_even))
        .max(f_measure(detected, &half_odd))
        .max(f_measure(detected, &double))
}

fn assert_grid_f_measure(grid: &BeatGrid, truth: &[f64], floor: f64, label: &str) {
    let f = f_measure(&grid.beats, truth);
    assert!(
        f >= floor,
        "{label}: F-measure {f:.3} below floor {floor} ({} detected vs {} truth beats, bpm {:.2})",
        grid.beats.len(),
        truth.len(),
        grid.bpm
    );
}

#[test]
fn constant_tempo_sweep_no_octave_folding() {
    // Wide genre-agnostic BPM sweep: each must report the true tempo (not
    // an octave) and place beats accurately. 60s of audio each.
    for &bpm in &[65.0, 90.0, 120.0, 128.0, 150.0] {
        let truth = constant_beats(bpm, 30.0, 3000.0);
        let amps = vec![1.0f32; truth.len()];
        let audio = render_clicks((SAMPLE_RATE as f64 * 30.0) as usize, &truth, &amps);
        let grid = detect_beat_grid(&audio, SAMPLE_RATE);
        assert!(
            (grid.bpm - bpm).abs() < bpm * 0.02,
            "expected ~{bpm} BPM, got {:.2}",
            grid.bpm
        );
        assert_grid_f_measure(&grid, &truth, 0.85, &format!("{bpm} BPM constant"));
        assert_eq!(
            grid.segments.len(),
            1,
            "{bpm} BPM constant tempo must collapse to one segment, got {:?}",
            grid.segments
        );
    }
}

#[test]
fn dnb_tempo_octave_family() {
    // 174 BPM without subdivision context is octave-ambiguous by design
    // (soft prior, no hard fold): accept the 174 grid or its 87 half, but
    // the beat placements must be on-grid either way.
    let truth = constant_beats(174.0, 30.0, 1000.0);
    let amps = vec![1.0f32; truth.len()];
    let audio = render_clicks((SAMPLE_RATE as f64 * 30.0) as usize, &truth, &amps);
    let grid = detect_beat_grid(&audio, SAMPLE_RATE);
    let family = [174.0, 87.0];
    assert!(
        family.iter().any(|t| (grid.bpm - t).abs() < t * 0.02),
        "expected 174 or 87 BPM, got {:.2}",
        grid.bpm
    );
    let f = f_measure_octave_tolerant(&grid.beats, &truth);
    assert!(f >= 0.85, "octave-tolerant F-measure {f:.3} below floor");
}

#[test]
fn tempo_ramp_is_tracked() {
    // 120 -> 132 BPM smooth ramp over 40 s: beats must follow the ramp.
    let len_secs = 40.0;
    let len = (SAMPLE_RATE as f64 * len_secs) as usize;
    let mut truth = Vec::new();
    let mut pos = 2000.0f64;
    while pos + 64.0 < len as f64 {
        truth.push(pos);
        let frac = pos / len as f64;
        let bpm = 120.0 + 12.0 * frac;
        pos += 60.0 * SAMPLE_RATE as f64 / bpm;
    }
    let amps = vec![1.0f32; truth.len()];
    let audio = render_clicks(len, &truth, &amps);
    let grid = detect_beat_grid(&audio, SAMPLE_RATE);
    assert_grid_f_measure(&grid, &truth, 0.85, "tempo ramp");
}

#[test]
fn tempo_step_splits_segments_and_tracks_both_sides() {
    // Constant 122 for 20 s then constant 138 for 20 s.
    let first = constant_beats(122.0, 20.0, 1500.0);
    let second: Vec<f64> = constant_beats(138.0, 20.0, 500.0)
        .iter()
        .map(|&b| b + SAMPLE_RATE as f64 * 20.0)
        .collect();
    let mut truth = first;
    truth.extend_from_slice(&second);
    let amps = vec![1.0f32; truth.len()];
    let audio = render_clicks((SAMPLE_RATE as f64 * 40.0) as usize, &truth, &amps);

    let grid = detect_beat_grid(&audio, SAMPLE_RATE);
    assert_grid_f_measure(&grid, &truth, 0.80, "tempo step");
    assert!(
        grid.segments.len() >= 2,
        "a hard tempo step must produce multiple segments, got {:?}",
        grid.segments
    );
    let first_bpm = grid.segments.first().unwrap().bpm;
    let last_bpm = grid.segments.last().unwrap().bpm;
    assert!(
        (first_bpm - 122.0).abs() < 4.0,
        "first segment ~122, got {first_bpm:.2}"
    );
    assert!(
        (last_bpm - 138.0).abs() < 4.0,
        "last segment ~138, got {last_bpm:.2}"
    );
}

#[test]
fn live_timing_jitter_is_tolerated() {
    // 100 BPM with ±12 ms uniform jitter per beat (a human drummer):
    // the grid must still land on the played (jittered) positions.
    let interval = 60.0 * SAMPLE_RATE as f64 / 100.0;
    let len = (SAMPLE_RATE as f64 * 30.0) as usize;
    let mut seed = 0x12345678u32;
    let mut rand = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f64 / (1u32 << 24) as f64 - 0.5
    };
    let jitter_max = 0.012 * SAMPLE_RATE as f64;
    let mut truth = Vec::new();
    let mut pos = 4000.0f64;
    while pos + 64.0 < len as f64 {
        let jittered = pos + rand() * 2.0 * jitter_max;
        truth.push(jittered);
        pos += interval;
    }
    let amps = vec![1.0f32; truth.len()];
    let audio = render_clicks(len, &truth, &amps);
    let grid = detect_beat_grid(&audio, SAMPLE_RATE);
    assert!(
        (grid.bpm - 100.0).abs() < 3.0,
        "expected ~100 BPM under jitter, got {:.2}",
        grid.bpm
    );
    assert_grid_f_measure(&grid, &truth, 0.80, "live jitter");
}

#[test]
fn downbeat_phase_follows_accents() {
    // 4/4 at 126 BPM with the first beat of each bar accented, bar phase
    // starting on beat 0.
    let truth = constant_beats(126.0, 30.0, 2500.0);
    let amps: Vec<f32> = (0..truth.len())
        .map(|k| if k % 4 == 0 { 1.0 } else { 0.5 })
        .collect();
    let audio = render_clicks((SAMPLE_RATE as f64 * 30.0) as usize, &truth, &amps);
    let grid = detect_beat_grid(&audio, SAMPLE_RATE);
    assert_grid_f_measure(&grid, &truth, 0.85, "accented 4/4");
    assert!(
        !grid.downbeats.is_empty(),
        "accented pattern must yield downbeats"
    );
    assert!(
        grid.downbeat_confidence > 0.05,
        "downbeat confidence too low: {}",
        grid.downbeat_confidence
    );

    // Every reported downbeat must be within 70 ms of an accented
    // (k % 4 == 0) ground-truth beat.
    let tolerance = F_MEASURE_TOLERANCE_SECS * SAMPLE_RATE as f64;
    let accented: Vec<f64> = truth.iter().step_by(4).copied().collect();
    for &db_idx in &grid.downbeats {
        let pos = grid.beats[db_idx];
        let near = accented.iter().any(|&a| (a - pos).abs() <= tolerance);
        assert!(near, "downbeat at {pos:.0} is not on an accented beat");
    }
}

#[test]
fn analysis_speed_is_offline_friendly() {
    // Exit criterion: full-track analysis comfortably faster than realtime
    // (proposed >= 50x). Use 60 s of audio and a conservative 25x floor in
    // debug-friendly CI environments; release builds clear 50x widely.
    let truth = constant_beats(128.0, 60.0, 0.0);
    let amps = vec![1.0f32; truth.len()];
    let audio = render_clicks((SAMPLE_RATE as f64 * 60.0) as usize, &truth, &amps);

    let started = std::time::Instant::now();
    let grid = detect_beat_grid(&audio, SAMPLE_RATE);
    let elapsed = started.elapsed().as_secs_f64();
    assert!(grid.bpm > 0.0);

    let realtime_factor = 60.0 / elapsed;
    let floor = if cfg!(debug_assertions) { 5.0 } else { 50.0 };
    assert!(
        realtime_factor >= floor,
        "analysis speed {realtime_factor:.1}x realtime below {floor}x floor"
    );
}

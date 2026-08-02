//! Rigid beat-grid fitting for quantized (DAW-produced) material.
//!
//! The DP beat tracker follows local onset evidence, which is correct for
//! live material but wanders on hard quantized tracks (gospel stabs,
//! micro-house, broken techno): corpus diagnostics show gross phase
//! misplacement (60–180 ms mean signed error) on exactly the tracks where
//! tracking is hardest — while the detected *tempo* is right on all of
//! them. Commercial DJ software solves this by fitting one rigid grid
//! (constant BPM + phase anchor) and this module does the same: a small
//! BPM search around the tracked tempo × a full phase circle, scored by
//! mean kick-band onset strength at the grid points (kicks define the
//! beat in dance music; full-band novelty is pulled around by hats,
//! snares, and vocal onsets).
//!
//! Adoption is decided by the fit's own decisiveness, not by a raw score
//! comparison against the tracked beats — tracked beats snap to the
//! detected onsets, so they always win a raw comparison even when they
//! are chasing jitter. On rigid material one phase decisively out-scores
//! all competing phases (`phase_lock` high); on genuinely non-rigid
//! material (live drummers, tempo rides) every phase scores about the
//! same and the tracked grid is kept. A smeared-score sanity floor
//! guards against adopting a fit that misses the kicks outright.

use crate::analysis::beat::BeatGrid;
use crate::core::preanalysis::TempoSegment;

/// Onset-envelope hop in seconds (~5 ms).
const HOP_SECS: f64 = 0.005;
/// Low-pass corner for the kick band, Hz (two cascaded biquads).
const KICK_BAND_HZ: f64 = 150.0;
/// BPM search half-width around the seed tempo, as a fraction. The seed
/// comes from the tracker (exact on the whole corpus), so the window only
/// absorbs residual median bias.
const BPM_SEARCH_FRAC: f64 = 0.005;
/// Coarse search resolution.
const BPM_STEPS: usize = 201;
const PHASE_STEPS: usize = 256;
/// Active-region gate relative to the loudest 1 s kick-band RMS window:
/// leading/trailing regions quieter than this carry no grid beats.
const ACTIVE_GATE: f32 = 0.05;
/// Minimum number of grid beats for a fit to be meaningful.
const MIN_BEATS: usize = 16;
/// Minimum phase decisiveness to adopt the rigid grid — the annotator's
/// own "trust without ear-verification" threshold.
const MIN_PHASE_LOCK: f32 = 0.3;
/// Sanity floor: under a timing-tolerant (smeared) objective the rigid
/// grid must reach at least this fraction of the tracked beats' score,
/// so a decisive-but-wrong fit (e.g. seeded off an octave-wrong tempo on
/// exotic material) cannot replace beats that demonstrably hit the kicks.
const ADOPT_MIN_SMEARED_RATIO: f64 = 0.5;
/// Half-width of the triangular timing tolerance used for that sanity
/// comparison, in seconds (~one vinyl-tight beat placement).
const SMEAR_RADIUS_SECS: f64 = 0.025;

/// A fitted rigid grid.
#[derive(Debug, Clone)]
pub struct RigidGridFit {
    /// Fitted tempo in BPM.
    pub bpm: f64,
    /// Grid anchor in seconds (first grid point, mod period).
    pub phase_secs: f64,
    /// Phase decisiveness in [0, 1]: winner vs the best phase at least an
    /// eighth-period away. Low values mean competing phases score nearly
    /// as well (offbeat-heavy or weakly periodic material).
    pub phase_lock: f32,
    /// Mean kick-band onset strength at the grid points (the objective).
    pub score: f64,
    /// Beat times in seconds over the active region, ascending.
    pub beats_secs: Vec<f64>,
}

/// Fits a rigid grid to a mono signal around `seed_bpm`.
///
/// Returns `None` when the signal is too short/quiet or the active region
/// carries fewer than 16 grid beats. A returned fit is a
/// *candidate*: callers decide adoption (see [`refine_grid_rigid`]).
pub fn fit_rigid_grid(samples: &[f32], sample_rate: u32, seed_bpm: f64) -> Option<RigidGridFit> {
    if sample_rate == 0 || seed_bpm <= 0.0 || samples.is_empty() {
        return None;
    }
    let sr = sample_rate as f64;
    let duration = samples.len() as f64 / sr;
    let KickEnvelope {
        energy,
        onset,
        frame_secs,
    } = kick_onset_envelope(samples, sr);
    if onset.is_empty() {
        return None;
    }

    // Active region from a slow (1 s) mean of the kick-band RMS energy
    // (not the onset strength, whose spiky startup would skew the gate).
    let env_window = (1.0 / HOP_SECS) as usize;
    let slow = moving_mean(&energy, env_window);
    let peak = slow.iter().copied().fold(0.0f32, f32::max);
    if peak <= 0.0 {
        return None;
    }
    let gate = peak * ACTIVE_GATE;
    let first_active = slow.iter().position(|&v| v > gate).unwrap_or(0);
    let last_active = slow.len() - 1 - slow.iter().rev().position(|&v| v > gate).unwrap_or(0);
    let (active_start, active_end) = (
        first_active as f64 * frame_secs,
        last_active as f64 * frame_secs,
    );

    // Coarse BPM × phase search, then a refined pass around the winner.
    let fit = |bpms: &[f64], phases: &[f64]| -> (f64, f64, f64) {
        let mut best = (seed_bpm, 0.0, f64::MIN);
        for &bpm in bpms {
            let period = 60.0 / bpm;
            for &phase in phases {
                let score = grid_score(&onset, frame_secs, phase, period, duration);
                if score > best.2 {
                    best = (bpm, phase, score);
                }
            }
        }
        best
    };
    let base_period = 60.0 / seed_bpm;
    let coarse_bpms: Vec<f64> = (0..BPM_STEPS)
        .map(|i| {
            seed_bpm * (1.0 - BPM_SEARCH_FRAC)
                + seed_bpm * 2.0 * BPM_SEARCH_FRAC * i as f64 / (BPM_STEPS - 1) as f64
        })
        .collect();
    let coarse_phases: Vec<f64> = (0..PHASE_STEPS)
        .map(|i| base_period * i as f64 / PHASE_STEPS as f64)
        .collect();
    let (bpm0, phase0, _) = fit(&coarse_bpms, &coarse_phases);
    let bpm_step = seed_bpm * 2.0 * BPM_SEARCH_FRAC / (BPM_STEPS - 1) as f64;
    let phase_step = base_period / PHASE_STEPS as f64;
    let fine_bpms: Vec<f64> = (0..21)
        .map(|i| bpm0 + bpm_step * (i as f64 - 10.0) / 10.0)
        .collect();
    let fine_phases: Vec<f64> = (0..33)
        .map(|i| phase0 + phase_step * (i as f64 - 16.0) / 16.0)
        .collect();
    let (bpm, phase, score) = fit(&fine_bpms, &fine_phases);
    if !score.is_finite() || score <= 0.0 {
        return None;
    }

    // Phase decisiveness: winner vs the best phase at least an
    // eighth-period away at the same BPM.
    let period = 60.0 / bpm;
    let mut rival = f64::MIN;
    for i in 0..PHASE_STEPS {
        let p = period * i as f64 / PHASE_STEPS as f64;
        let dist = (p - phase.rem_euclid(period)).abs();
        let dist = dist.min(period - dist);
        if dist >= period / 8.0 {
            rival = rival.max(grid_score(&onset, frame_secs, p, period, duration));
        }
    }
    let phase_lock = (1.0 - rival / score).max(0.0) as f32;

    // Beat times over the active region.
    let mut beats_secs: Vec<f64> = Vec::new();
    let mut t = phase.rem_euclid(period);
    while t < duration {
        if t >= active_start - period * 0.5 && t <= active_end + period * 0.5 {
            beats_secs.push(t);
        }
        t += period;
    }
    if beats_secs.len() < MIN_BEATS {
        return None;
    }

    Some(RigidGridFit {
        bpm,
        phase_secs: phase.rem_euclid(period),
        phase_lock,
        score,
        beats_secs,
    })
}

/// Replaces a tracked grid with a rigid fit when the material supports it.
///
/// Fits a rigid grid seeded at `grid.bpm` and adopts it only when the
/// fit's phase is decisive (phase lock at or above the annotator's
/// trust threshold) and it clears the
/// smeared-score sanity floor against the tracked beats. Returns the
/// (possibly refreshed) grid plus whether the rigid fit was adopted.
pub fn refine_grid_rigid(samples: &[f32], sample_rate: u32, grid: BeatGrid) -> (BeatGrid, bool) {
    if grid.bpm <= 0.0 || grid.beats.len() < MIN_BEATS {
        return (grid, false);
    }
    let Some(fit) = fit_rigid_grid(samples, sample_rate, grid.bpm) else {
        return (grid, false);
    };
    if fit.phase_lock < MIN_PHASE_LOCK {
        return (grid, false);
    }

    // Sanity floor under a timing-tolerant objective: smear the onset
    // envelope so honest ±few-ms placements score alike, then require the
    // rigid grid to reach a fraction of the tracked beats' score.
    let sr = sample_rate as f64;
    let KickEnvelope {
        onset, frame_secs, ..
    } = kick_onset_envelope(samples, sr);
    let radius = (SMEAR_RADIUS_SECS / frame_secs).round().max(1.0) as usize;
    let smeared = triangular_smear(&onset, radius);
    let tracked_secs: Vec<f64> = grid.beats.iter().map(|&b| b / sr).collect();
    let tracked_score = mean_env_at(&smeared, frame_secs, &tracked_secs);
    let rigid_score = mean_env_at(&smeared, frame_secs, &fit.beats_secs);
    if rigid_score < tracked_score * ADOPT_MIN_SMEARED_RATIO {
        return (grid, false);
    }

    // Downbeat rotation by kick-band accent (mod 4), as in the annotator.
    let mut rotation_scores = [0.0f64; 4];
    for (i, &b) in fit.beats_secs.iter().enumerate() {
        rotation_scores[i % 4] += sample_env(&onset, frame_secs, b);
    }
    for (r, s) in rotation_scores.iter_mut().enumerate() {
        let n = (fit.beats_secs.len() + 3 - r) / 4;
        *s /= n.max(1) as f64;
    }
    let best_rotation = (0..4)
        .max_by(|&a, &b| rotation_scores[a].total_cmp(&rotation_scores[b]))
        .unwrap_or(0);
    let mut sorted = rotation_scores;
    sorted.sort_by(|a, b| b.total_cmp(a));
    let downbeat_confidence = if sorted[0] > 0.0 {
        ((sorted[0] - sorted[1]) / sorted[0]).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };

    let beats: Vec<f64> = fit.beats_secs.iter().map(|&t| t * sr).collect();
    let downbeats: Vec<usize> = (0..beats.len())
        .filter(|i| i % 4 == best_rotation)
        .collect();
    let bpm_ratio = fit.bpm / grid.bpm;
    let tempo_candidates = grid
        .tempo_candidates
        .iter()
        .map(|c| crate::core::preanalysis::TempoCandidate {
            bpm: c.bpm * bpm_ratio,
            salience: c.salience,
        })
        .collect();

    (
        BeatGrid {
            beats,
            downbeats,
            segments: vec![TempoSegment {
                start_beat: 0,
                bpm: fit.bpm,
            }],
            bpm: fit.bpm,
            confidence: grid.confidence.max(fit.phase_lock),
            downbeat_confidence,
            sample_rate,
            tempo_candidates,
        },
        true,
    )
}

/// Kick-band envelopes at the analysis hop.
pub(crate) struct KickEnvelope {
    /// Per-hop RMS energy of the low-passed kick band.
    pub(crate) energy: Vec<f32>,
    /// Half-wave rectified log-energy difference (onset strength).
    pub(crate) onset: Vec<f32>,
    /// Duration of one envelope frame in seconds.
    pub(crate) frame_secs: f64,
}

/// Kick-band onset envelope: 4th-order low-pass → hop RMS → half-wave
/// rectified log-energy difference.
pub(crate) fn kick_onset_envelope(samples: &[f32], sr: f64) -> KickEnvelope {
    let low = lowpass4(samples, sr, KICK_BAND_HZ);
    let hop = (sr * HOP_SECS).round().max(1.0) as usize;
    let energy = hop_rms(&low, hop);
    let onset = onset_strength(&energy);
    KickEnvelope {
        energy,
        onset,
        frame_secs: hop as f64 / sr,
    }
}

/// Mean onset strength sampled at every grid point of a rigid grid.
fn grid_score(onset: &[f32], frame_secs: f64, phase: f64, period: f64, duration: f64) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    let mut t = phase.rem_euclid(period);
    while t < duration {
        sum += sample_env(onset, frame_secs, t);
        n += 1;
        t += period;
    }
    if n == 0 { f64::MIN } else { sum / n as f64 }
}

/// Mean onset strength at arbitrary beat times.
fn mean_env_at(onset: &[f32], frame_secs: f64, beats_secs: &[f64]) -> f64 {
    if beats_secs.is_empty() {
        return f64::MIN;
    }
    beats_secs
        .iter()
        .map(|&t| sample_env(onset, frame_secs, t))
        .sum::<f64>()
        / beats_secs.len() as f64
}

/// Triangular smearing of the envelope over `±radius` frames, so scoring
/// tolerates small timing differences instead of rewarding exact
/// onset-chasing.
fn triangular_smear(env: &[f32], radius: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; env.len()];
    for (i, out_v) in out.iter_mut().enumerate() {
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(env.len());
        let mut acc = 0.0f32;
        let mut weight_sum = 0.0f32;
        for (j, &v) in env.iter().enumerate().take(hi).skip(lo) {
            let w = 1.0 - (j as f32 - i as f32).abs() / (radius as f32 + 1.0);
            acc += v * w;
            weight_sum += w;
        }
        if weight_sum > 0.0 {
            *out_v = acc / weight_sum;
        }
    }
    out
}

/// Linear interpolation of the envelope at time `t`.
fn sample_env(env: &[f32], frame_secs: f64, t: f64) -> f64 {
    let pos = t / frame_secs;
    let i = pos.floor() as usize;
    if pos < 0.0 || i + 1 >= env.len() {
        return 0.0;
    }
    let frac = pos - i as f64;
    env[i] as f64 * (1.0 - frac) + env[i + 1] as f64 * frac
}

/// Two cascaded RBJ low-pass biquads (Q = 0.707) → 4th-order response.
fn lowpass4(input: &[f32], sr: f64, corner_hz: f64) -> Vec<f32> {
    let w0 = 2.0 * std::f64::consts::PI * corner_hz / sr;
    let (sin_w0, cos_w0) = (w0.sin(), w0.cos());
    let alpha = sin_w0 / (2.0 * 0.707);
    let b0 = (1.0 - cos_w0) / 2.0;
    let b1 = 1.0 - cos_w0;
    let b2 = b0;
    let a0 = 1.0 + alpha;
    let (b0, b1, b2, a1, a2) = (
        b0 / a0,
        b1 / a0,
        b2 / a0,
        -2.0 * cos_w0 / a0,
        (1.0 - alpha) / a0,
    );
    let mut out = input.to_vec();
    for _ in 0..2 {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for v in out.iter_mut() {
            let x0 = *v as f64;
            let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            *v = y0 as f32;
        }
    }
    out
}

/// Per-hop RMS of the input.
fn hop_rms(input: &[f32], hop: usize) -> Vec<f32> {
    input
        .chunks(hop)
        .map(|c| {
            (c.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / c.len() as f64).sqrt() as f32
        })
        .collect()
}

/// Half-wave rectified log-energy difference.
fn onset_strength(env: &[f32]) -> Vec<f32> {
    let eps = 1e-6f64;
    let mut out = vec![0.0f32; env.len()];
    for i in 1..env.len() {
        let d = ((env[i] as f64 + eps).ln() - (env[i - 1] as f64 + eps).ln()).max(0.0);
        out[i] = d as f32;
    }
    out
}

/// Trailing moving mean over `window` frames.
fn moving_mean(input: &[f32], window: usize) -> Vec<f32> {
    let w = window.max(1);
    let mut out = vec![0.0f32; input.len()];
    let mut sum = 0.0f64;
    for i in 0..input.len() {
        sum += input[i] as f64;
        if i >= w {
            sum -= input[i - w] as f64;
        }
        out[i] = (sum / w.min(i + 1) as f64) as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::beat::detect_beats;

    const SR: u32 = 44_100;

    /// Kick-like click train: short low-frequency thumps every beat, with
    /// per-beat timing jitter in samples.
    fn kick_train(bpm: f64, seconds: f64, jitter: &[i64]) -> Vec<f32> {
        let len = (SR as f64 * seconds) as usize;
        let mut out = vec![0.0f32; len];
        let period = 60.0 * SR as f64 / bpm;
        let mut k = 0usize;
        let mut pos = 0.0f64;
        while (pos as usize) < len {
            let at = (pos as i64 + jitter[k % jitter.len().max(1)]).max(0) as usize;
            for i in 0..2000.min(len.saturating_sub(at)) {
                let t = i as f64 / SR as f64;
                let envl = (-t * 40.0).exp();
                out[at + i] += (0.9 * envl * (2.0 * std::f64::consts::PI * 60.0 * t).sin()) as f32;
            }
            pos += period;
            k += 1;
        }
        out
    }

    #[test]
    fn fit_recovers_exact_grid_from_jittered_kicks() {
        // ±8 ms alternating jitter on a 128 BPM kick train: the rigid fit
        // must recover the underlying period, not follow the jitter.
        let jitter = [353i64, -353]; // ±8 ms at 44.1k
        let samples = kick_train(128.0, 30.0, &jitter);
        let fit = fit_rigid_grid(&samples, SR, 128.0).expect("fit");
        assert!((fit.bpm - 128.0).abs() < 0.05, "fitted {} vs 128", fit.bpm);
        assert!(fit.phase_lock > 0.3, "phase_lock {}", fit.phase_lock);
        // Grid intervals are exactly one period.
        let period = 60.0 / fit.bpm;
        for w in fit.beats_secs.windows(2) {
            assert!(((w[1] - w[0]) - period).abs() < 1e-9);
        }
    }

    #[test]
    fn refine_adopts_rigid_grid_on_quantized_material() {
        let jitter = [353i64, -353];
        let samples = kick_train(128.0, 30.0, &jitter);
        let tracked = detect_beats(&samples, SR);
        assert!(tracked.bpm > 0.0);
        let (grid, adopted) = refine_grid_rigid(&samples, SR, tracked);
        assert!(adopted, "rigid grid should win on a jittered kick train");
        assert!((grid.bpm - 128.0).abs() < 0.05, "bpm {}", grid.bpm);
        assert_eq!(grid.segments.len(), 1);
        // Rigid beats: constant interval throughout.
        let period = 60.0 * SR as f64 / grid.bpm;
        for w in grid.beats.windows(2) {
            assert!(((w[1] - w[0]) - period).abs() < 1e-6);
        }
        assert!(!grid.downbeats.is_empty());
    }

    #[test]
    fn refine_keeps_tracked_grid_on_tempo_ramp() {
        // 120 → 132 BPM ramp: no rigid grid explains the kicks better
        // than the tracked curve, so the tracked grid must survive.
        let len = (SR as f64 * 40.0) as usize;
        let mut samples = vec![0.0f32; len];
        let mut pos = 0.0f64;
        while (pos as usize) < len {
            let at = pos as usize;
            for i in 0..2000.min(len - at) {
                let t = i as f64 / SR as f64;
                let envl = (-t * 40.0).exp();
                samples[at + i] +=
                    (0.9 * envl * (2.0 * std::f64::consts::PI * 60.0 * t).sin()) as f32;
            }
            let frac = pos / len as f64;
            let bpm = 120.0 + 12.0 * frac;
            pos += 60.0 * SR as f64 / bpm;
        }
        let tracked = detect_beats(&samples, SR);
        assert!(tracked.bpm > 0.0);
        let tracked_beats = tracked.beats.clone();
        let (grid, adopted) = refine_grid_rigid(&samples, SR, tracked);
        assert!(!adopted, "a tempo ramp must not adopt a rigid grid");
        assert_eq!(grid.beats, tracked_beats);
    }

    #[test]
    fn degenerate_inputs_do_not_fit() {
        assert!(fit_rigid_grid(&[], SR, 128.0).is_none());
        assert!(fit_rigid_grid(&vec![0.0; SR as usize * 30], SR, 128.0).is_none());
        assert!(fit_rigid_grid(&kick_train(128.0, 30.0, &[0]), 0, 128.0).is_none());
        assert!(fit_rigid_grid(&kick_train(128.0, 30.0, &[0]), SR, 0.0).is_none());
        // Too short for MIN_BEATS at 128 BPM.
        assert!(fit_rigid_grid(&kick_train(128.0, 5.0, &[0]), SR, 128.0).is_none());
    }
}

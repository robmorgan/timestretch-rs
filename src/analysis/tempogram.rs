//! Autocorrelation tempogram and tempo-path tracking.
//!
//! Estimates a time-varying tempo curve from an onset novelty function:
//! windowed autocorrelation with harmonic reinforcement produces a
//! tempogram (tempo salience per time window), a wide log-normal prior
//! favors perceptually plausible octaves without hard folding, and a
//! Viterbi pass over the tempogram columns extracts a smooth tempo path
//! that may drift or step (live drummers, tempo rides, older recordings).
//!
//! The output is a per-frame beat period curve consumed by the
//! dynamic-programming beat tracker in [`crate::analysis::beat`].

/// Options controlling tempo estimation.
#[derive(Debug, Clone, Copy)]
pub struct TempoTrackingOptions {
    /// Lower edge of the tempo search range in BPM.
    pub min_bpm: f64,
    /// Upper edge of the tempo search range in BPM.
    pub max_bpm: f64,
    /// Center of the log-normal tempo prior in BPM. The prior is wide
    /// (see [`Self::prior_octave_sigma`]) — it breaks octave ties, it does
    /// not fold tempos into a range.
    pub prior_center_bpm: f64,
    /// Standard deviation of the tempo prior in octaves.
    pub prior_octave_sigma: f64,
    /// Optional soft hint range `(low, high)` in BPM: candidates inside get
    /// a small salience bonus. Used by the EDM/DJ analysis path (100–160);
    /// `None` for general-purpose detection.
    pub hint_range: Option<(f64, f64)>,
}

impl Default for TempoTrackingOptions {
    fn default() -> Self {
        Self {
            min_bpm: 50.0,
            max_bpm: 220.0,
            prior_center_bpm: 120.0,
            prior_octave_sigma: 1.0,
            hint_range: None,
        }
    }
}

/// Salience bonus multiplier applied inside [`TempoTrackingOptions::hint_range`].
const HINT_RANGE_BONUS: f32 = 1.15;

/// Harmonic reinforcement weights: salience at lag `l` also credits
/// autocorrelation at `2l` and `3l`, disambiguating the beat level from
/// its subdivisions (a hi-hat pattern peaks at half the beat period, but
/// the beat lag collects its own peak plus the bar-level harmonics).
const HARMONIC_WEIGHTS: [(usize, f32); 3] = [(1, 1.0), (2, 0.5), (3, 0.33)];

/// Tempogram analysis window length in seconds. Long enough to average
/// over several bars, short enough to follow drift.
const TEMPOGRAM_WINDOW_SECS: f64 = 8.0;

/// Tempogram column hop as a fraction of the window.
const TEMPOGRAM_HOP_FRACTION: f64 = 0.125;

/// Viterbi transition penalty weight on squared log period ratio between
/// adjacent columns. 2% drift per column costs ~0.01 (negligible against
/// salience ~1); an octave jump costs ~11.5 (needs sustained evidence).
const TRANSITION_LAMBDA: f32 = 24.0;

/// Moving-average window for novelty conditioning, in seconds.
const NOVELTY_MEAN_WINDOW_SECS: f64 = 0.5;

/// Smoothing radius for the conditioned novelty, in seconds. Onset peaks a
/// frame or two off the exact lag (timing jitter, fractional periods) must
/// still correlate; a short triangular kernel makes the autocorrelation
/// tolerant without blurring distinct onsets together.
const NOVELTY_SMOOTH_SECS: f64 = 0.03;

/// A time-varying tempo estimate over a novelty curve.
#[derive(Debug, Clone)]
pub struct TempoTrack {
    /// Beat period in novelty frames, one entry per novelty frame.
    pub period_frames: Vec<f32>,
    /// Mean normalized salience along the chosen tempo path in [0, 1] —
    /// how clearly periodic the material is.
    pub path_salience: f32,
}

impl TempoTrack {
    /// Beat period in frames at `frame`, clamped to the curve's ends.
    #[inline]
    pub fn period_at(&self, frame: usize) -> f32 {
        match self.period_frames.get(frame) {
            Some(&p) => p,
            None => self.period_frames.last().copied().unwrap_or(0.0),
        }
    }
}

/// Conditions a raw onset detection function into a novelty curve:
/// subtracts a moving average (removes the loudness baseline), half-wave
/// rectifies, and normalizes to a maximum of 1.
pub(crate) fn condition_novelty(odf: &[f32], frame_rate: f64) -> Vec<f32> {
    if odf.is_empty() {
        return Vec::new();
    }
    let half_win = ((NOVELTY_MEAN_WINDOW_SECS * frame_rate * 0.5).round() as usize).max(1);

    // Prefix sums for O(1) windowed means.
    let mut prefix = Vec::with_capacity(odf.len() + 1);
    prefix.push(0.0f64);
    for &v in odf {
        prefix.push(prefix.last().unwrap() + v as f64);
    }

    let rectified: Vec<f32> = (0..odf.len())
        .map(|i| {
            let lo = i.saturating_sub(half_win);
            let hi = (i + half_win + 1).min(odf.len());
            let mean = (prefix[hi] - prefix[lo]) / (hi - lo) as f64;
            (odf[i] as f64 - mean).max(0.0) as f32
        })
        .collect();

    // Short triangular smoothing (see NOVELTY_SMOOTH_SECS).
    let radius = ((NOVELTY_SMOOTH_SECS * frame_rate).round() as usize).max(1);
    let mut novelty = vec![0.0f32; rectified.len()];
    for (i, out) in novelty.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        let mut weight_sum = 0.0f32;
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(rectified.len());
        for (j, &v) in rectified.iter().enumerate().take(hi).skip(lo) {
            let w = 1.0 - (j as f32 - i as f32).abs() / (radius as f32 + 1.0);
            acc += v * w;
            weight_sum += w;
        }
        if weight_sum > 0.0 {
            *out = acc / weight_sum;
        }
    }

    let max = novelty.iter().copied().fold(0.0f32, f32::max);
    if max > 1e-12 {
        let inv = 1.0 / max;
        for v in &mut novelty {
            *v *= inv;
        }
    }
    novelty
}

/// Windowed, normalized autocorrelation of `novelty` starting at `start`
/// over `win_len` frames, for all lags in `1..=max_lag`.
///
/// Uses normalized cross-correlation (each shifted copy normalized by its
/// own energy) so a crescendo does not bias short lags.
fn window_autocorrelation(
    novelty: &[f32],
    start: usize,
    win_len: usize,
    max_lag: usize,
) -> Vec<f32> {
    let end = (start + win_len).min(novelty.len());
    let base = &novelty[start..end];
    let mut ac = vec![0.0f32; max_lag + 1];

    let base_energy: f64 = base.iter().map(|&v| (v as f64) * (v as f64)).sum();
    if base_energy <= 1e-12 {
        return ac;
    }

    for (lag, ac_val) in ac.iter_mut().enumerate().skip(1) {
        if start + lag >= novelty.len() {
            break;
        }
        let shifted_end = (start + lag + win_len).min(novelty.len());
        let shifted = &novelty[start + lag..shifted_end];
        let n = base.len().min(shifted.len());
        if n == 0 {
            break;
        }
        let mut dot = 0.0f64;
        let mut shifted_energy = 0.0f64;
        for i in 0..n {
            dot += base[i] as f64 * shifted[i] as f64;
            shifted_energy += (shifted[i] as f64) * (shifted[i] as f64);
        }
        let norm = (base_energy * shifted_energy).sqrt();
        if norm > 1e-12 {
            *ac_val = (dot / norm) as f32;
        }
    }
    ac
}

/// Log-normal prior weight for a tempo in BPM.
fn prior_weight(bpm: f64, opts: &TempoTrackingOptions) -> f32 {
    let octaves = (bpm / opts.prior_center_bpm).log2();
    let sigma = opts.prior_octave_sigma.max(1e-3);
    (-0.5 * (octaves / sigma).powi(2)).exp() as f32
}

/// One tempogram column: harmonic-reinforced, prior-weighted salience per
/// lag in `lag_min..=lag_max`, computed from a raw autocorrelation that
/// extends to `3 * lag_max` where the signal allows.
fn salience_column(
    ac: &[f32],
    lag_min: usize,
    lag_max: usize,
    frame_rate: f64,
    opts: &TempoTrackingOptions,
) -> Vec<f32> {
    (lag_min..=lag_max)
        .map(|lag| {
            let mut s = 0.0f32;
            for &(mult, w) in &HARMONIC_WEIGHTS {
                let idx = lag * mult;
                if idx < ac.len() {
                    s += w * ac[idx].max(0.0);
                }
            }
            let bpm = 60.0 * frame_rate / lag as f64;
            let mut weighted = s * prior_weight(bpm, opts);
            if let Some((lo, hi)) = opts.hint_range {
                if bpm >= lo && bpm <= hi {
                    weighted *= HINT_RANGE_BONUS;
                }
            }
            weighted
        })
        .collect()
}

/// Viterbi path over tempogram columns. States are lag indices relative to
/// `lag_min`; transitions penalize squared log period ratio, so the path
/// follows drift but resists spurious octave hops.
fn viterbi_path(columns: &[Vec<f32>], lag_min: usize) -> Vec<usize> {
    let num_states = match columns.first() {
        Some(c) => c.len(),
        None => return Vec::new(),
    };
    if num_states == 0 {
        return Vec::new();
    }

    // Precompute log lags for the transition penalty.
    let log_lag: Vec<f32> = (0..num_states)
        .map(|s| ((lag_min + s) as f32).ln())
        .collect();

    let mut score = columns[0].clone();
    let mut backptr: Vec<Vec<u32>> = Vec::with_capacity(columns.len());
    backptr.push((0..num_states as u32).collect());

    for col in &columns[1..] {
        let mut next_score = vec![f32::NEG_INFINITY; num_states];
        let mut next_back = vec![0u32; num_states];
        for (s, &emission) in col.iter().enumerate() {
            let mut best = f32::NEG_INFINITY;
            let mut best_prev = 0u32;
            for (p, &prev_score) in score.iter().enumerate() {
                let d = log_lag[s] - log_lag[p];
                let cand = prev_score - TRANSITION_LAMBDA * d * d;
                if cand > best {
                    best = cand;
                    best_prev = p as u32;
                }
            }
            next_score[s] = best + emission;
            next_back[s] = best_prev;
        }
        score = next_score;
        backptr.push(next_back);
    }

    // Backtrack from the best final state.
    let mut state = score
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut path = vec![0usize; columns.len()];
    for t in (0..columns.len()).rev() {
        path[t] = state;
        state = backptr[t][state] as usize;
    }
    path
}

/// Parabolic peak refinement of the chosen lag within a salience column.
/// Returns a fractional lag (absolute, not state-relative).
fn refine_lag(column: &[f32], state: usize, lag_min: usize) -> f64 {
    let lag = (lag_min + state) as f64;
    if state == 0 || state + 1 >= column.len() {
        return lag;
    }
    let (a, b, c) = (
        column[state - 1] as f64,
        column[state] as f64,
        column[state + 1] as f64,
    );
    let denom = a - 2.0 * b + c;
    if denom.abs() < 1e-12 {
        return lag;
    }
    let delta = (0.5 * (a - c) / denom).clamp(-0.5, 0.5);
    lag + delta
}

/// Estimates a time-varying tempo track from a raw onset detection function.
///
/// `odf` is a per-frame detection function (e.g. [`crate::analysis::transient::TransientMap::flux`])
/// at `frame_rate` frames per second. Returns `None` when the signal is too
/// short or has no usable periodicity.
pub fn estimate_tempo_track(
    odf: &[f32],
    frame_rate: f64,
    opts: &TempoTrackingOptions,
) -> Option<TempoTrack> {
    if frame_rate <= 0.0 || opts.min_bpm <= 0.0 || opts.max_bpm <= opts.min_bpm {
        return None;
    }
    let novelty = condition_novelty(odf, frame_rate);

    let lag_min = ((60.0 * frame_rate / opts.max_bpm).ceil() as usize).max(1);
    let lag_max = (60.0 * frame_rate / opts.min_bpm).floor() as usize;
    if lag_max <= lag_min {
        return None;
    }
    // Need at least two beat periods of the slowest candidate tempo to see
    // any periodicity at all.
    if novelty.len() < lag_min * 4 || novelty.len() < 2 {
        return None;
    }

    let ideal_window = (TEMPOGRAM_WINDOW_SECS * frame_rate) as usize;
    // The window should cover the longest lag twice, but never exceed the
    // signal itself (short inputs analyze in a single whole-signal window).
    let win_len = ideal_window
        .max(lag_max.saturating_mul(2))
        .min(novelty.len());
    let col_hop = ((win_len as f64 * TEMPOGRAM_HOP_FRACTION) as usize).max(1);

    let ac_extent = (lag_max * HARMONIC_WEIGHTS.last().unwrap().0).min(win_len.saturating_sub(1));

    let mut columns = Vec::new();
    let mut col_centers = Vec::new();
    let mut start = 0usize;
    loop {
        let ac = window_autocorrelation(&novelty, start, win_len, ac_extent);
        columns.push(salience_column(&ac, lag_min, lag_max, frame_rate, opts));
        col_centers.push(start + win_len / 2);
        if start + win_len >= novelty.len() {
            break;
        }
        start += col_hop;
    }

    let path = viterbi_path(&columns, lag_min);
    if path.is_empty() {
        return None;
    }

    // Path salience: mean chosen-state salience normalized by each column's
    // maximum, as a periodicity-clarity measure.
    let mut salience_sum = 0.0f64;
    let mut salient_cols = 0usize;
    for (col, &state) in columns.iter().zip(path.iter()) {
        let col_max = col.iter().copied().fold(0.0f32, f32::max);
        if col_max > 1e-9 {
            salience_sum += (col[state] / col_max) as f64;
            salient_cols += 1;
        }
    }
    if salient_cols == 0 {
        return None;
    }
    let path_salience = (salience_sum / salient_cols as f64) as f32;

    // Refine each column's lag, then interpolate a per-frame period curve
    // between column centers.
    let refined: Vec<f64> = columns
        .iter()
        .zip(path.iter())
        .map(|(col, &state)| refine_lag(col, state, lag_min))
        .collect();

    let mut period_frames = vec![0.0f32; novelty.len()];
    for (i, period) in period_frames.iter_mut().enumerate() {
        *period = interpolate_at(&col_centers, &refined, i) as f32;
    }

    Some(TempoTrack {
        period_frames,
        path_salience,
    })
}

/// Piecewise-linear interpolation of `values` (sampled at `centers`, which
/// are strictly increasing) at position `x`, clamped at the ends.
fn interpolate_at(centers: &[usize], values: &[f64], x: usize) -> f64 {
    debug_assert_eq!(centers.len(), values.len());
    match centers {
        [] => 0.0,
        [_] => values[0],
        _ => {
            if x <= centers[0] {
                return values[0];
            }
            if x >= *centers.last().unwrap() {
                return *values.last().unwrap();
            }
            let idx = centers.partition_point(|&c| c <= x);
            let (c0, c1) = (centers[idx - 1], centers[idx]);
            let (v0, v1) = (values[idx - 1], values[idx]);
            if c1 == c0 {
                return v0;
            }
            let t = (x - c0) as f64 / (c1 - c0) as f64;
            v0 + (v1 - v0) * t
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic novelty: unit impulses every `period` frames with optional
    /// weaker impulses at the half-period (subdivision content).
    fn impulse_novelty(len: usize, period: f64, offbeat: f32) -> Vec<f32> {
        let mut novelty = vec![0.0f32; len];
        let mut pos = 0.0f64;
        while (pos as usize) < len {
            novelty[pos as usize] = 1.0;
            let half = pos + period / 2.0;
            if offbeat > 0.0 && (half as usize) < len {
                novelty[half as usize] = offbeat;
            }
            pos += period;
        }
        novelty
    }

    const FRAME_RATE: f64 = 86.132_812_5; // 44100 / 512

    fn bpm_of(period_frames: f32) -> f64 {
        60.0 * FRAME_RATE / period_frames as f64
    }

    #[test]
    fn constant_tempo_120() {
        let period = 60.0 * FRAME_RATE / 120.0; // ~43.07 frames
        let novelty = impulse_novelty(3000, period, 0.0);
        let track =
            estimate_tempo_track(&novelty, FRAME_RATE, &TempoTrackingOptions::default()).unwrap();
        let mid = track.period_at(1500);
        assert!(
            (bpm_of(mid) - 120.0).abs() < 2.0,
            "expected ~120 BPM, got {:.2}",
            bpm_of(mid)
        );
        assert!(track.path_salience > 0.5);
    }

    #[test]
    fn slow_tempo_90_not_folded() {
        let period = 60.0 * FRAME_RATE / 90.0;
        let novelty = impulse_novelty(3000, period, 0.0);
        let track =
            estimate_tempo_track(&novelty, FRAME_RATE, &TempoTrackingOptions::default()).unwrap();
        let bpm = bpm_of(track.period_at(1500));
        assert!(
            (bpm - 90.0).abs() < 2.0,
            "90 BPM must not octave-fold, got {:.2}",
            bpm
        );
    }

    #[test]
    fn fast_tempo_174_within_octave_family() {
        // A bare impulse train at 174 is indistinguishable from an 87 BPM
        // track articulated in eighths — the octave choice on such material
        // is prior-driven by design (explicit policy: soft prior, no hard
        // fold). Require the estimate to land in the right octave family;
        // callers that know the genre can raise `min_bpm` or hint.
        let period = 60.0 * FRAME_RATE / 174.0;
        let novelty = impulse_novelty(3000, period, 0.0);
        let track =
            estimate_tempo_track(&novelty, FRAME_RATE, &TempoTrackingOptions::default()).unwrap();
        let bpm = bpm_of(track.period_at(1500));
        let in_family = [87.0, 174.0]
            .iter()
            .any(|&target| (bpm - target).abs() < 4.0);
        assert!(in_family, "expected 174 or its half, got {:.2}", bpm);
    }

    #[test]
    fn fast_tempo_174_with_narrowed_range() {
        // With a floor above the half-tempo (DnB-aware caller), the tracker
        // must lock the true 174.
        let period = 60.0 * FRAME_RATE / 174.0;
        let novelty = impulse_novelty(3000, period, 0.0);
        let opts = TempoTrackingOptions {
            min_bpm: 100.0,
            ..TempoTrackingOptions::default()
        };
        let track = estimate_tempo_track(&novelty, FRAME_RATE, &opts).unwrap();
        let bpm = bpm_of(track.period_at(1500));
        assert!(
            (bpm - 174.0).abs() < 4.0,
            "expected ~174 BPM with narrowed range, got {:.2}",
            bpm
        );
    }

    #[test]
    fn subdivisions_do_not_halve_the_period() {
        // Beat at 120 BPM with strong offbeat eighths: the tracker must not
        // report 240 BPM (out of range here, but it must also not sit at the
        // eighth-note lag when 240 is in range — use a 100 BPM base so the
        // double is inside the default range).
        let period = 60.0 * FRAME_RATE / 100.0;
        let novelty = impulse_novelty(3000, period, 0.8);
        let track =
            estimate_tempo_track(&novelty, FRAME_RATE, &TempoTrackingOptions::default()).unwrap();
        let bpm = bpm_of(track.period_at(1500));
        assert!(
            (bpm - 100.0).abs() < 3.0,
            "offbeats must not halve the period: got {:.2} BPM",
            bpm
        );
    }

    #[test]
    fn tempo_ramp_is_followed() {
        // Ramp 120 -> 132 BPM over ~70 seconds.
        let len = 6000usize;
        let mut novelty = vec![0.0f32; len];
        let mut pos = 0.0f64;
        while (pos as usize) < len {
            novelty[pos as usize] = 1.0;
            let frac = pos / len as f64;
            let bpm = 120.0 + 12.0 * frac;
            pos += 60.0 * FRAME_RATE / bpm;
        }
        let track =
            estimate_tempo_track(&novelty, FRAME_RATE, &TempoTrackingOptions::default()).unwrap();
        let early = bpm_of(track.period_at(600));
        let late = bpm_of(track.period_at(len - 600));
        assert!(
            (early - 120.0).abs() < 4.0,
            "early tempo should be ~120, got {:.2}",
            early
        );
        assert!(
            (late - 132.0).abs() < 4.0,
            "late tempo should be ~132, got {:.2}",
            late
        );
    }

    #[test]
    fn silence_returns_none() {
        let novelty = vec![0.0f32; 3000];
        assert!(
            estimate_tempo_track(&novelty, FRAME_RATE, &TempoTrackingOptions::default()).is_none()
        );
    }

    #[test]
    fn too_short_returns_none() {
        let novelty = vec![1.0f32; 10];
        assert!(
            estimate_tempo_track(&novelty, FRAME_RATE, &TempoTrackingOptions::default()).is_none()
        );
    }

    #[test]
    fn hint_range_breaks_octave_tie() {
        // A pattern with equal salience at 70 and 140 BPM (impulses at 140
        // with every other one stronger is ambiguous); the EDM hint should
        // pull the estimate into 100-160.
        let period_140 = 60.0 * FRAME_RATE / 140.0;
        let mut novelty = vec![0.0f32; 3000];
        let mut pos = 0.0f64;
        let mut k = 0usize;
        while (pos as usize) < novelty.len() {
            novelty[pos as usize] = if k % 2 == 0 { 1.0 } else { 0.85 };
            pos += period_140;
            k += 1;
        }
        let opts = TempoTrackingOptions {
            hint_range: Some((100.0, 160.0)),
            ..TempoTrackingOptions::default()
        };
        let track = estimate_tempo_track(&novelty, FRAME_RATE, &opts).unwrap();
        let bpm = bpm_of(track.period_at(1500));
        assert!(
            (bpm - 140.0).abs() < 4.0,
            "hint should keep the tactus at 140, got {:.2}",
            bpm
        );
    }

    #[test]
    fn interpolate_at_clamps_and_blends() {
        let centers = vec![10usize, 20, 30];
        let values = vec![1.0f64, 2.0, 4.0];
        assert_eq!(interpolate_at(&centers, &values, 0), 1.0);
        assert_eq!(interpolate_at(&centers, &values, 35), 4.0);
        assert!((interpolate_at(&centers, &values, 15) - 1.5).abs() < 1e-12);
        assert!((interpolate_at(&centers, &values, 25) - 3.0).abs() < 1e-12);
    }
}

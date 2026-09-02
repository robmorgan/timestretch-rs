//! General-purpose beat tracking, BPM detection, and grid snapping.
//!
//! The tracker follows a time-varying tempo: an autocorrelation tempogram
//! with a Viterbi tempo path ([`crate::analysis::tempogram`]) estimates a
//! per-frame beat period, and a dynamic-programming beat tracker
//! (Ellis-style) places beats on the onset novelty curve along that path.
//! Constant-tempo tracks collapse to a single [`TempoSegment`]; live or
//! drifting material produces a piecewise grid. Downbeats are estimated
//! from beat-synchronous accent features with a 4/4 prior and an explicit
//! confidence, never hard-coded.

use crate::analysis::tempogram::{
    TempoTrack, TempoTrackingOptions, condition_novelty, estimate_tempo_track,
};
use crate::analysis::transient::{TransientMap, detect_transients};

pub use crate::core::preanalysis::{TempoCandidate, TempoSegment};

/// FFT size for beat detection (balances frequency resolution and speed).
const BEAT_FFT_SIZE: usize = 2048;
/// Hop size for beat detection analysis frames.
const BEAT_HOP_SIZE: usize = 512;
/// Transient sensitivity for the onset pass backing beat detection.
const BEAT_SENSITIVITY: f32 = 0.4;
/// Hard bound for dynamically generated beat/subdivision grid points.
const MAX_GRID_POINTS: usize = 1_000_000;

/// DP transition tightness: cost is `TIGHTNESS * ln(interval / period)^2`,
/// with novelty normalized to a maximum of 1. At 100, a 10% interval
/// deviation costs about one full-strength onset peak.
const DP_TIGHTNESS: f32 = 100.0;

/// Beats-per-bar prior for downbeat estimation. 4/4 is a prior, not a
/// hard-coded truth: the phase decision carries its own confidence.
const BEATS_PER_BAR: usize = 4;

/// Minimum number of tracked beats required for downbeat phase estimation.
const MIN_BEATS_FOR_DOWNBEATS: usize = 8;

/// Relative BPM deviation that opens a new tempo segment.
const SEGMENT_BPM_DEVIATION: f64 = 0.03;
/// Consecutive deviating beats required to open a new tempo segment.
const SEGMENT_DEVIATION_RUN: usize = 4;

/// Weight of low-band flux in the downbeat accent feature (bass emphasis
/// marks bar starts across most popular genres).
const DOWNBEAT_LOW_FLUX_WEIGHT: f32 = 0.6;
/// Weight of overall novelty in the downbeat accent feature.
/// Weight of sustained sub-bass ENERGY in the downbeat accent, with the
/// measurement window in analysis frames: skip ~3 frames (past the kick's
/// attack, which every beat shares), then average ~21 (≈ 250 ms at the
/// 2048/512 configuration — a bass note's sustained body). Flux features
/// tie on four-on-floor material (every beat carries an equal kick) and
/// the log-difference onset feature actively favors the snare (a bass
/// fill before the kick suppresses its jump), electing the snare as beat
/// one; the bassline's emphasis on the true one is carried by sustained
/// ENERGY, which difference features cannot see. Weighted above the flux
/// terms because where they disagree, energy is the trustworthy one.
const DOWNBEAT_NOVELTY_WEIGHT: f32 = 0.4;
const DOWNBEAT_ENERGY_WEIGHT: f32 = 1.5;
const DOWNBEAT_ENERGY_SKIP_FRAMES: usize = 3;
const DOWNBEAT_ENERGY_FRAMES: usize = 21;

/// Maximum distance, in novelty frames, for snapping a tracked beat to a
/// detected transient onset (sub-hop precision from the phase-refined
/// onset positions).
const BEAT_ONSET_SNAP_FRAMES: f64 = 3.0;

/// Beat grid for a track: tracked beat positions, downbeats, and a
/// piecewise-constant tempo model.
///
/// Positions are fractional sample indices on the analyzed mono signal.
/// Constant-tempo material yields a single segment; tempo drift or steps
/// yield several. An empty `beats` with `bpm == 0.0` means no tempo was
/// detected.
#[derive(Debug, Clone)]
pub struct BeatGrid {
    /// Fractional-sample positions of every tracked beat, ascending.
    pub beats: Vec<f64>,
    /// Indices into `beats` marking downbeats (bar starts), ascending.
    pub downbeats: Vec<usize>,
    /// Piecewise-constant tempo segments covering `beats`, ascending by
    /// `start_beat`; never empty when `beats` is non-empty.
    pub segments: Vec<TempoSegment>,
    /// Representative tempo in BPM (median of multi-beat-baseline
    /// intervals, which cancels per-beat frame-grid quantization); 0.0
    /// when undetected.
    pub bpm: f64,
    /// Overall grid confidence in [0, 1]: periodicity clarity, beat-level
    /// onset support, and interval regularity against the tempo curve.
    pub confidence: f32,
    /// Confidence of the downbeat phase decision in [0, 1].
    pub downbeat_confidence: f32,
    /// Sample rate the positions refer to.
    pub sample_rate: u32,
    /// Ranked tempo hypotheses, highest salience first: the committed
    /// [`bpm`](Self::bpm) plus its in-range half/double alternatives.
    /// Empty when no tempo was detected.
    pub tempo_candidates: Vec<TempoCandidate>,
    /// True when two independent estimators (kick-band rigid fit and the
    /// DP tracker) genuinely disagreed about beat phase on plausibly
    /// quantized material — the grid's phase should not be trusted for
    /// display or quantization even if it looks internally consistent.
    /// [`confidence`](Self::confidence) is capped alongside this flag.
    pub phase_untrusted: bool,
}

impl BeatGrid {
    /// An empty grid (no tempo detected) at `sample_rate`.
    pub fn empty(sample_rate: u32) -> Self {
        BeatGrid {
            beats: Vec::new(),
            downbeats: Vec::new(),
            segments: Vec::new(),
            bpm: 0.0,
            confidence: 0.0,
            downbeat_confidence: 0.0,
            sample_rate,
            tempo_candidates: Vec::new(),
            phase_untrusted: false,
        }
    }

    /// Returns the representative interval between beats in samples.
    #[inline]
    pub fn beat_interval_samples(&self) -> f64 {
        if self.bpm <= 0.0 {
            return 0.0;
        }
        60.0 * self.sample_rate as f64 / self.bpm
    }

    /// Tempo in BPM at a sample position, from the segment model.
    /// Returns the representative BPM outside the tracked range and 0.0
    /// for an empty grid.
    pub fn bpm_at(&self, position: f64) -> f64 {
        if self.segments.is_empty() {
            return self.bpm;
        }
        let idx = self.index_of_beat_at_or_before(position);
        let beat_idx = match idx {
            Some(i) => i,
            None => return self.segments[0].bpm,
        };
        let seg = self
            .segments
            .iter()
            .rev()
            .find(|s| s.start_beat <= beat_idx)
            .unwrap_or(&self.segments[0]);
        seg.bpm
    }

    /// Index of the nearest beat to `position`, or `None` for an empty grid.
    pub fn nearest_beat_index(&self, position: f64) -> Option<usize> {
        if self.beats.is_empty() {
            return None;
        }
        let idx = self.beats.partition_point(|&b| b < position);
        let mut best = idx.min(self.beats.len() - 1);
        if idx > 0 {
            let before = idx - 1;
            if (position - self.beats[before]).abs() <= (self.beats[best] - position).abs() {
                best = before;
            }
        }
        Some(best)
    }

    /// Index of the last beat at or before `position`, or `None` when
    /// `position` precedes the first beat (or the grid is empty).
    fn index_of_beat_at_or_before(&self, position: f64) -> Option<usize> {
        let idx = self.beats.partition_point(|&b| b <= position);
        idx.checked_sub(1)
    }

    /// Snaps a sample position to the nearest beat.
    #[inline]
    pub fn snap_to_grid(&self, position: usize) -> usize {
        match self.nearest_beat_index(position as f64) {
            Some(i) => self.beats[i].round().max(0.0) as usize,
            None => position,
        }
    }

    /// Snaps a fractional sample position to the nearest beat.
    #[inline]
    pub fn snap_to_grid_fractional(&self, position: f64) -> f64 {
        match self.nearest_beat_index(position) {
            Some(i) => self.beats[i],
            None => position,
        }
    }

    /// Beat positions rounded to integer sample indices.
    pub fn beats_rounded(&self) -> Vec<usize> {
        self.beats
            .iter()
            .map(|&b| b.round().max(0.0) as usize)
            .collect()
    }

    /// Fractional-sample positions of the downbeats.
    pub fn downbeat_positions(&self) -> Vec<f64> {
        self.downbeats
            .iter()
            .filter_map(|&i| self.beats.get(i).copied())
            .collect()
    }
}

/// Detects beats in a mono audio signal with general-purpose defaults
/// (wide 50–220 BPM range, no genre hint).
pub fn detect_beats(samples: &[f32], sample_rate: u32) -> BeatGrid {
    detect_beats_with_options(samples, sample_rate, &TempoTrackingOptions::default())
}

/// Detects beats with explicit tempo-tracking options (range, prior,
/// genre hint).
pub fn detect_beats_with_options(
    samples: &[f32],
    sample_rate: u32,
    options: &TempoTrackingOptions,
) -> BeatGrid {
    let transients = detect_transients(
        samples,
        sample_rate,
        BEAT_FFT_SIZE,
        BEAT_HOP_SIZE,
        BEAT_SENSITIVITY,
    );
    detect_beats_from_transients_with_options(&transients, sample_rate, options)
}

/// Beat tracking from an already-computed transient map with explicit
/// tempo options. The map's detection function drives tracking, so callers
/// with a detection pass in hand (`analyze`) don't run a second one.
pub(crate) fn detect_beats_from_transients_with_options(
    transients: &TransientMap,
    sample_rate: u32,
    options: &TempoTrackingOptions,
) -> BeatGrid {
    let hop = transients.hop_size.max(1);
    let frame_rate = sample_rate as f64 / hop as f64;
    if transients.flux.is_empty() || frame_rate <= 0.0 {
        return BeatGrid::empty(sample_rate);
    }
    // No (or a single) detected onset means the energy gate classified the
    // material as sustained/tonal — the residual detection function is
    // windowing ripple, and normalizing it would hallucinate a tempo (a DC
    // or pure-tone input must report no beats).
    if transients.onsets.len() < 2 {
        return BeatGrid::empty(sample_rate);
    }

    let track = match estimate_tempo_track(&transients.flux, frame_rate, options) {
        Some(track) => track,
        None => return BeatGrid::empty(sample_rate),
    };
    // Metrical-level check (ROADMAP Stage 10): drum & bass tracks the 2/3
    // sub-level of its true ~174 tempo — the half-time feel puts real
    // autocorrelation mass at the 3/2-of-chosen lag while the 120-BPM
    // prior systematically favors the sub-level. Measured separation on
    // the full corpus (2026-08-17): the 3/2 candidate's salience is
    // 0.76–0.82 when the higher level is the true tempo (both DnB rows)
    // and ≤ 0.60 everywhere else (EDM ≤ 0.38). When the 3/2 alternative
    // clears the mid-gap threshold, re-track with the prior centered at
    // that level and adopt the result if it converges there — a second
    // evidence pass, not a fold.
    let track = match metrical_level_check(&transients.flux, frame_rate, options, track) {
        Some(track) => track,
        None => return BeatGrid::empty(sample_rate),
    };
    let novelty = condition_novelty(&transients.flux, frame_rate);

    let beat_frames = track_beats_dp(&novelty, &track);
    if beat_frames.len() < 2 {
        return BeatGrid::empty(sample_rate);
    }

    let beats = refine_beat_positions(&beat_frames, &novelty, transients, hop);

    let segments = segment_tempo(&beats, sample_rate);
    let bpm = median_bpm(&beats, sample_rate);

    let (downbeats, downbeat_confidence) =
        estimate_downbeats(&beat_frames, &novelty, transients, hop);

    let confidence = grid_confidence(&beat_frames, &novelty, &track);

    // Absolute-BPM candidates from the octave saliences, anchored on the
    // refined representative tempo so ×2/÷2 are exact for the UI.
    let mut tempo_candidates: Vec<TempoCandidate> = track
        .octave_saliences
        .iter()
        .filter(|(_, salience)| *salience > 0.0)
        .map(|&(ratio, salience)| TempoCandidate {
            bpm: bpm * ratio,
            salience,
        })
        .collect();
    tempo_candidates.sort_by(|a, b| b.salience.total_cmp(&a.salience));

    BeatGrid {
        beats,
        downbeats,
        segments,
        bpm,
        confidence,
        downbeat_confidence,
        sample_rate,
        tempo_candidates,
        phase_untrusted: false,
    }
}

/// Threshold on the 3/2 metrical candidate's salience above which the
/// higher level is re-evaluated. Measured corpus separation
/// (2026-08-17): true-higher-level rows sit at 0.76–0.82, everything
/// else at or below 0.60 — the threshold is the mid-gap.
const METRICAL_32_SALIENCE_MIN: f32 = 0.70;
/// The re-tracked tempo must land within this relative tolerance of
/// 1.5× the first pass to be adopted (convergence guard: the second
/// pass must confirm the intended level, not wander elsewhere).
const METRICAL_32_CONVERGE_TOL: f64 = 0.05;

/// Second-pass metrical-level check: when the 3/2 candidate's salience
/// clears the measured threshold, re-track with the prior centered at
/// that level and adopt the result only if it converges there.
fn metrical_level_check(
    flux: &[f32],
    frame_rate: f64,
    options: &TempoTrackingOptions,
    track: TempoTrack,
) -> Option<TempoTrack> {
    let s32 = track
        .octave_saliences
        .iter()
        .find(|(ratio, _)| (*ratio - 1.5).abs() < 1e-6)
        .map(|&(_, s)| s)
        .unwrap_or(0.0);
    if s32 < METRICAL_32_SALIENCE_MIN {
        return Some(track);
    }
    let bpm = track_representative_bpm(&track, frame_rate);
    if bpm <= 0.0 {
        return Some(track);
    }
    let retrack_opts = TempoTrackingOptions {
        prior_center_bpm: bpm * 1.5,
        ..*options
    };
    let Some(retracked) = estimate_tempo_track(flux, frame_rate, &retrack_opts) else {
        return Some(track);
    };
    let re_bpm = track_representative_bpm(&retracked, frame_rate);
    if re_bpm > 0.0 && ((re_bpm / (bpm * 1.5)) - 1.0).abs() <= METRICAL_32_CONVERGE_TOL {
        Some(retracked)
    } else {
        Some(track)
    }
}

/// Median-period representative tempo of a track, in BPM (0.0 = none).
fn track_representative_bpm(track: &TempoTrack, frame_rate: f64) -> f64 {
    let mut periods: Vec<f32> = track
        .period_frames
        .iter()
        .copied()
        .filter(|p| *p > 0.0)
        .collect();
    if periods.is_empty() {
        return 0.0;
    }
    periods.sort_by(|a, b| a.total_cmp(b));
    let median = periods[periods.len() / 2] as f64;
    if median > 0.0 {
        60.0 * frame_rate / median
    } else {
        0.0
    }
}

/// Dynamic-programming beat tracking (Ellis-style) along a time-varying
/// tempo curve. Returns beat positions as novelty frame indices, ascending.
fn track_beats_dp(novelty: &[f32], track: &TempoTrack) -> Vec<usize> {
    let n = novelty.len();
    if n == 0 {
        return Vec::new();
    }

    let mut score = vec![0.0f32; n];
    let mut backlink = vec![usize::MAX; n];

    for i in 0..n {
        let period = track.period_at(i) as f64;
        if period <= 1.0 {
            score[i] = novelty[i];
            continue;
        }
        let lo = (i as f64 - 2.0 * period).max(0.0) as usize;
        let hi_excl = (i as f64 - 0.5 * period).ceil().max(0.0) as usize;

        let mut best = 0.0f32;
        let mut best_j = usize::MAX;
        let window = lo..hi_excl.min(i);
        for (j, &prev_score) in score[window.clone()].iter().enumerate() {
            let j = j + window.start;
            let interval = (i - j) as f64;
            let dev = (interval / period).ln() as f32;
            let cand = prev_score - DP_TIGHTNESS * dev * dev;
            if cand > best {
                best = cand;
                best_j = j;
            }
        }
        // `best` floors at 0: a chain may start anywhere without inheriting
        // the (negative) cost of a nonexistent predecessor.
        score[i] = novelty[i] + best;
        backlink[i] = best_j;
    }

    // Choose the endpoint: the best cumulative score within the last two
    // periods (so trailing silence doesn't orphan the chain).
    let tail_period = track.period_at(n.saturating_sub(1)) as f64;
    let tail_start = (n as f64 - 2.0 * tail_period).max(0.0) as usize;
    let mut end = tail_start;
    for (i, &s) in score.iter().enumerate().skip(tail_start) {
        if s > score[end] {
            end = i;
        }
    }
    if score[end] <= 0.0 {
        return Vec::new();
    }

    let mut beats = Vec::new();
    let mut cursor = end;
    loop {
        beats.push(cursor);
        let prev = backlink[cursor];
        if prev == usize::MAX {
            break;
        }
        cursor = prev;
    }
    beats.reverse();
    beats
}

/// Converts beat frames to fractional sample positions: parabolic
/// interpolation on the novelty peak, then snapping to a phase-refined
/// transient onset when one is close (see [`BEAT_ONSET_SNAP_FRAMES`]).
///
/// Frame-to-sample conversion adds the map's detection latency so beats
/// land on the audible attack, matching the (already-compensated) onset
/// positions they snap to.
fn refine_beat_positions(
    beat_frames: &[usize],
    novelty: &[f32],
    transients: &TransientMap,
    hop: usize,
) -> Vec<f64> {
    let snap_tolerance_samples = BEAT_ONSET_SNAP_FRAMES * hop as f64;
    let latency = transients.latency_samples as f64;
    beat_frames
        .iter()
        .map(|&frame| {
            let mut pos = frame as f64;
            if frame > 0 && frame + 1 < novelty.len() {
                let (a, b, c) = (
                    novelty[frame - 1] as f64,
                    novelty[frame] as f64,
                    novelty[frame + 1] as f64,
                );
                let denom = a - 2.0 * b + c;
                if denom.abs() > 1e-12 {
                    pos += (0.5 * (a - c) / denom).clamp(-0.5, 0.5);
                }
            }
            let sample_pos = pos * hop as f64 + latency;

            // Prefer the sub-hop position of a coincident detected onset.
            let onsets = &transients.onsets_fractional;
            if !onsets.is_empty() {
                let idx = onsets.partition_point(|&o| o < sample_pos);
                let mut best: Option<f64> = None;
                for cand in [idx.checked_sub(1), Some(idx)].into_iter().flatten() {
                    if let Some(&o) = onsets.get(cand) {
                        let dist = (o - sample_pos).abs();
                        if dist <= snap_tolerance_samples
                            && best.is_none_or(|b: f64| dist < (b - sample_pos).abs())
                        {
                            best = Some(o);
                        }
                    }
                }
                if let Some(o) = best {
                    return o.max(0.0);
                }
            }
            sample_pos.max(0.0)
        })
        .collect()
}

/// Maximum baseline length (in beats) for BPM estimation.
const BPM_BASELINE_BEATS: usize = 16;

/// Representative BPM of a beat sequence (0.0 with fewer than two beats).
///
/// Median of K-beat-baseline intervals `(beats[i+K] - beats[i]) / K`.
/// The median of *adjacent* intervals is biased: refined beat positions
/// carry a small cyclic error tied to where each beat falls on the
/// analysis frame grid (period `1 / frac(period_frames)` beats, ~3 at
/// 125 BPM with a 512 hop), so adjacent intervals repeat an asymmetric
/// sawtooth whose median sits ~0.2% off the true interval (a flat 125.0
/// read as 125.2). Position errors telescope over a K-beat span, shrinking
/// the sawtooth by 1/K, while the median stays robust to dropped beats
/// and breakdown re-locks. K is capped at a quarter of the available
/// intervals so a single bad interval contaminates at most ~a third of
/// the spans — the median still lands on clean ones.
fn median_bpm(beats: &[f64], sample_rate: u32) -> f64 {
    if beats.len() < 2 {
        return 0.0;
    }
    let n = beats.len();
    let k = BPM_BASELINE_BEATS.min(((n - 1) / 4).max(1));
    let mut spans: Vec<f64> = (0..n - k)
        .map(|i| (beats[i + k] - beats[i]) / k as f64)
        .collect();
    spans.sort_by(|a, b| a.total_cmp(b));
    let median = spans[spans.len() / 2];
    if median <= 0.0 {
        return 0.0;
    }
    60.0 * sample_rate as f64 / median
}

/// Splits the beat sequence into piecewise-constant tempo segments.
///
/// A new segment opens when [`SEGMENT_DEVIATION_RUN`] consecutive beat
/// intervals deviate from the current segment's median interval by more
/// than [`SEGMENT_BPM_DEVIATION`]. Constant-tempo tracks yield one segment.
fn segment_tempo(beats: &[f64], sample_rate: u32) -> Vec<TempoSegment> {
    if beats.len() < 2 {
        return Vec::new();
    }
    let intervals: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();

    // Interval range `start..end` covers beats `start..=end`; use the
    // baseline-median estimator so per-segment BPMs don't carry the
    // adjacent-interval quantization bias (see `median_bpm`).
    let segment_bpm = |range: std::ops::Range<usize>| -> f64 {
        median_bpm(&beats[range.start..=range.end], sample_rate)
    };

    let mut segments = Vec::new();
    let mut seg_start = 0usize; // interval index
    let mut deviation_run = 0usize;

    for i in 0..intervals.len() {
        // Median of the segment so far, updated lazily: use a running
        // reference from the segment's first few intervals, refreshed on
        // each split. Cheap and robust for segmentation purposes.
        let ref_end = (seg_start + 8).min(i + 1).max(seg_start + 1);
        let mut reference: Vec<f64> = intervals[seg_start..ref_end].to_vec();
        reference.sort_by(|a, b| a.total_cmp(b));
        let ref_interval = reference[reference.len() / 2];
        if ref_interval <= 0.0 {
            continue;
        }

        let deviation = (intervals[i] / ref_interval - 1.0).abs();
        if deviation > SEGMENT_BPM_DEVIATION {
            deviation_run += 1;
            if deviation_run >= SEGMENT_DEVIATION_RUN {
                let split = i + 1 - deviation_run;
                if split > seg_start {
                    segments.push(TempoSegment {
                        start_beat: seg_start,
                        bpm: segment_bpm(seg_start..split),
                    });
                    seg_start = split;
                }
                deviation_run = 0;
            }
        } else {
            deviation_run = 0;
        }
    }
    segments.push(TempoSegment {
        start_beat: seg_start,
        bpm: segment_bpm(seg_start..intervals.len()),
    });

    // Merge adjacent segments whose tempo agrees: a transient deviation run
    // (a breakdown, a dropped beat) splits the sequence, but if the tempo
    // on both sides is the same there was no tempo change to model.
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        let end = segments
            .get(i + 1)
            .map(|s| s.start_beat)
            .unwrap_or(intervals.len());
        match spans.last_mut() {
            Some((start, span_end)) => {
                let span_bpm = segment_bpm(*start..*span_end);
                if span_bpm > 0.0
                    && seg.bpm > 0.0
                    && (seg.bpm / span_bpm - 1.0).abs() <= SEGMENT_BPM_DEVIATION
                {
                    *span_end = end;
                } else {
                    spans.push((seg.start_beat, end));
                }
            }
            None => spans.push((seg.start_beat, end)),
        }
    }
    spans
        .into_iter()
        .map(|(start, end)| TempoSegment {
            start_beat: start,
            bpm: segment_bpm(start..end),
        })
        .collect()
}

/// Quartile contrast of a per-beat feature in [0, 1]: `1 - q25/q75`.
/// ~0 for flat or noisy-flat profiles, high when the top quarter of
/// beats genuinely stands out.
fn quartile_contrast_f32(values: &[f32]) -> f32 {
    if values.len() < 4 {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let q25 = sorted[sorted.len() / 4];
    let q75 = sorted[(sorted.len() * 3) / 4];
    if q75 > 1e-12 {
        (1.0 - q25 / q75).max(0.0)
    } else {
        0.0
    }
}

/// Estimates the downbeat phase from beat-synchronous accent features:
/// low-band flux (bass emphasis) plus overall novelty, scored across the
/// [`BEATS_PER_BAR`] phase hypotheses. Returns downbeat indices into the
/// beat sequence and the decision confidence.
fn estimate_downbeats(
    beat_frames: &[usize],
    novelty: &[f32],
    transients: &TransientMap,
    _hop: usize,
) -> (Vec<usize>, f32) {
    if beat_frames.len() < MIN_BEATS_FOR_DOWNBEATS {
        let downbeats = (0..beat_frames.len()).step_by(BEATS_PER_BAR).collect();
        return (downbeats, 0.0);
    }

    // Per-beat accent: novelty + low-band flux at the beat frame.
    let low_flux: Vec<f32> = beat_frames
        .iter()
        .map(|&frame| {
            transients
                .per_frame_band_flux
                .get(frame)
                .map(|bands| bands[0] + bands[1])
                .unwrap_or(0.0)
        })
        .collect();
    let max_low = low_flux.iter().copied().fold(0.0f32, f32::max);
    let low_norm = if max_low > 1e-12 { max_low } else { 1.0 };

    // Sustained sub-bass energy over the frames just after each beat.
    let low_energy: Vec<f32> = beat_frames
        .iter()
        .map(|&frame| {
            let e = &transients.per_frame_low_energy;
            let start = frame + DOWNBEAT_ENERGY_SKIP_FRAMES;
            let end = (start + DOWNBEAT_ENERGY_FRAMES).min(e.len());
            let span = &e[start.min(end)..end];
            if span.is_empty() {
                0.0
            } else {
                span.iter().sum::<f32>() / span.len() as f32
            }
        })
        .collect();
    let max_energy = low_energy.iter().copied().fold(0.0f32, f32::max);
    let energy_norm = if max_energy > 1e-12 { max_energy } else { 1.0 };
    // Self-gating: an uninformative energy profile — flat, or noise
    // around silence (post-attack windows of a pure percussion track) —
    // carries no phase information, and an un-gated term would only
    // dilute the confidence margin (or inject normalized noise). The
    // weight scales with the profile's quartile contrast, which reads
    // ~0 for both flat and noisy-flat profiles but stays high when a
    // bassline lifts the top quarter of beats.
    let energy_weight = DOWNBEAT_ENERGY_WEIGHT * quartile_contrast_f32(&low_energy);

    let accents: Vec<f32> = beat_frames
        .iter()
        .zip(low_flux.iter().zip(low_energy.iter()))
        .map(|(&frame, (&low, &energy))| {
            let nov = novelty.get(frame).copied().unwrap_or(0.0);
            DOWNBEAT_NOVELTY_WEIGHT * nov
                + DOWNBEAT_LOW_FLUX_WEIGHT * (low / low_norm)
                + energy_weight * (energy / energy_norm)
        })
        .collect();

    let mut phase_scores = [0.0f64; BEATS_PER_BAR];
    let mut phase_counts = [0usize; BEATS_PER_BAR];
    for (k, &a) in accents.iter().enumerate() {
        phase_scores[k % BEATS_PER_BAR] += a as f64;
        phase_counts[k % BEATS_PER_BAR] += 1;
    }
    for (score, &count) in phase_scores.iter_mut().zip(phase_counts.iter()) {
        if count > 0 {
            *score /= count as f64;
        }
    }

    let best_phase = (0..BEATS_PER_BAR)
        .max_by(|&a, &b| phase_scores[a].total_cmp(&phase_scores[b]))
        .unwrap_or(0);
    let best = phase_scores[best_phase];
    let second = (0..BEATS_PER_BAR)
        .filter(|&p| p != best_phase)
        .map(|p| phase_scores[p])
        .fold(f64::NEG_INFINITY, f64::max);
    let confidence = if best > 1e-12 {
        (((best - second) / best).clamp(0.0, 1.0)) as f32
    } else {
        0.0
    };

    let downbeats = (best_phase..beat_frames.len())
        .step_by(BEATS_PER_BAR)
        .collect();
    (downbeats, confidence)
}

/// Grid confidence: tempogram path clarity, beat-level onset support, and
/// interval regularity against the tracked tempo curve.
fn grid_confidence(beat_frames: &[usize], novelty: &[f32], track: &TempoTrack) -> f32 {
    if beat_frames.len() < 2 || novelty.is_empty() {
        return 0.0;
    }

    // Beat support: mean novelty at beats relative to the global peak (1.0
    // after conditioning). Strong grids have beats on strong onsets.
    let support = beat_frames
        .iter()
        .map(|&f| novelty.get(f).copied().unwrap_or(0.0))
        .sum::<f32>()
        / beat_frames.len() as f32;

    // Regularity: mean absolute log deviation of intervals from the local
    // tempo period, mapped so 0 deviation -> 1 and 10% -> ~0.
    let mut dev_sum = 0.0f64;
    let mut dev_count = 0usize;
    for w in beat_frames.windows(2) {
        let period = track.period_at(w[0]) as f64;
        if period > 1.0 {
            let interval = (w[1] - w[0]) as f64;
            dev_sum += (interval / period).ln().abs();
            dev_count += 1;
        }
    }
    let regularity = if dev_count > 0 {
        (1.0 - (dev_sum / dev_count as f64) / 0.1).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };

    (0.4 * track.path_salience + 0.3 * support + 0.3 * regularity).clamp(0.0, 1.0)
}

/// Generate a grid of beat subdivision positions (e.g., 1/16th notes) from BPM.
///
/// Returns sample positions (as `f64` for sub-sample precision) for every
/// subdivision within the given duration.
///
/// # Parameters
/// - `bpm`: Tempo in beats per minute.
/// - `sample_rate`: Audio sample rate in Hz.
/// - `total_samples`: Total duration in samples; the grid stops before this.
/// - `subdivision`: Number of subdivisions per beat (e.g., 16 for 1/16th notes).
///
/// # Example
///
/// ```
/// use timestretch::analysis::beat::generate_subdivision_grid;
///
/// // 120 BPM, 44100 Hz, 1 second, 1/16th notes
/// let grid = generate_subdivision_grid(120.0, 44100, 44100, 16);
/// // 120 BPM = 2 beats/sec => 32 subdivisions per second
/// // 32 * (22050/16) = 44100.0, which is not < 44100, so 32 entries (0..31)
/// assert_eq!(grid.len(), 32);
/// ```
pub fn generate_subdivision_grid(
    bpm: f64,
    sample_rate: u32,
    total_samples: usize,
    subdivision: u32,
) -> Vec<f64> {
    if bpm <= 0.0 || subdivision == 0 || total_samples == 0 {
        return Vec::new();
    }
    let beat_interval_samples = 60.0 * sample_rate as f64 / bpm;
    let sub_interval = beat_interval_samples / subdivision as f64;
    if sub_interval <= 0.0 {
        return Vec::new();
    }
    let estimated_count = (total_samples as f64 / sub_interval).ceil() as usize + 1;
    let max_points = estimated_count.min(MAX_GRID_POINTS);
    let mut grid = Vec::with_capacity(max_points);
    let mut pos = 0.0;
    for _ in 0..max_points {
        if pos >= total_samples as f64 {
            break;
        }
        grid.push(pos);
        pos += sub_interval;
    }
    grid
}

/// Snap a transient position to the nearest beat subdivision.
///
/// Returns `Some(snapped_position)` if a subdivision is within `tolerance_samples`,
/// or `None` if no subdivision is close enough (transient should be suppressed).
///
/// Uses binary search for efficient lookup in sorted grids.
///
/// # Parameters
/// - `position`: The transient position in samples.
/// - `grid`: Sorted grid of subdivision positions (from [`generate_subdivision_grid`]).
/// - `tolerance_samples`: Maximum distance (in samples) for snapping.
///
/// # Example
///
/// ```
/// use timestretch::analysis::beat::snap_to_subdivision;
///
/// let grid = vec![0.0, 1000.0, 2000.0, 3000.0];
/// // Position 1005 is 5 samples from grid point 1000
/// assert_eq!(snap_to_subdivision(1005.0, &grid, 10.0), Some(1000.0));
/// // Position 1500 is 500 samples from any grid point — too far
/// assert_eq!(snap_to_subdivision(1500.0, &grid, 10.0), None);
/// ```
pub fn snap_to_subdivision(position: f64, grid: &[f64], tolerance_samples: f64) -> Option<f64> {
    if grid.is_empty() {
        return None;
    }

    // Binary search for the insertion point
    let idx = grid.partition_point(|&g| g < position);

    let mut best_dist = f64::MAX;
    let mut best_pos = position;

    // Check the grid point before and at the insertion point
    for &check_idx in &[idx.saturating_sub(1), idx] {
        if check_idx < grid.len() {
            let dist = (grid[check_idx] - position).abs();
            if dist < best_dist {
                best_dist = dist;
                best_pos = grid[check_idx];
            }
        }
    }

    if best_dist <= tolerance_samples {
        Some(best_pos)
    } else {
        None // No nearby subdivision — suppress this transient
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: a grid with evenly spaced beats for snap tests.
    fn make_grid(beats: Vec<f64>, bpm: f64, sample_rate: u32) -> BeatGrid {
        let n = beats.len();
        BeatGrid {
            beats,
            downbeats: (0..n).step_by(BEATS_PER_BAR).collect(),
            segments: vec![TempoSegment { start_beat: 0, bpm }],
            bpm,
            confidence: 1.0,
            downbeat_confidence: 1.0,
            sample_rate,
            tempo_candidates: Vec::new(),
            phase_untrusted: false,
        }
    }

    /// Click train at `bpm` with a stronger accent every `accent_every`
    /// beats (0 = no accents).
    fn click_train(
        sample_rate: u32,
        bpm: f64,
        seconds: f64,
        accent_every: usize,
        phase_offset_samples: usize,
    ) -> Vec<f32> {
        let len = (sample_rate as f64 * seconds) as usize;
        let mut samples = vec![0.0f32; len];
        let interval = 60.0 * sample_rate as f64 / bpm;
        let mut pos = phase_offset_samples as f64;
        let mut k = 0usize;
        while (pos as usize) + 24 < len {
            let base = pos as usize;
            let amp = if accent_every > 0 && k % accent_every == 0 {
                1.0
            } else {
                0.55
            };
            for j in 0..20 {
                // A short decaying burst with some low-frequency content.
                let t = j as f32 / 20.0;
                samples[base + j] = amp * (1.0 - t) * if j % 2 == 0 { 1.0 } else { -0.7 };
            }
            pos += interval;
            k += 1;
        }
        samples
    }

    #[test]
    fn detect_beats_120_click_train() {
        let sample_rate = 44100u32;
        let grid = detect_beats(&click_train(sample_rate, 120.0, 12.0, 0, 0), sample_rate);
        assert!(
            (grid.bpm - 120.0).abs() < 2.0,
            "expected ~120 BPM, got {:.2}",
            grid.bpm
        );
        assert!(
            grid.beats.len() > 15,
            "expected beats, got {}",
            grid.beats.len()
        );
        assert_eq!(grid.segments.len(), 1, "constant tempo => one segment");
        assert!((grid.segments[0].bpm - 120.0).abs() < 2.0);
        assert!(grid.confidence > 0.4, "confidence {}", grid.confidence);

        // Beats must land ON the clicks, not merely near them: a constant
        // frame-timestamp bias (window start vs. audible attack) once put
        // every beat ~30 ms early, which the old 10%-of-interval tolerance
        // (50 ms) never caught. Sharp isolated clicks trip the unwindowed
        // energy channel a frame or two before the Hann-weighted flux, so
        // their residual runs a few ms earlier than on kick-shaped onsets
        // (see detect_beats_land_on_kick_attacks for the tighter bound).
        let interval = 60.0 * sample_rate as f64 / 120.0;
        let tolerance_ms = 15.0;
        let mut offsets_ms: Vec<f64> = Vec::new();
        for &b in &grid.beats[1..grid.beats.len() - 1] {
            let nearest = (b / interval).round() * interval;
            let off_ms = (b - nearest) / sample_rate as f64 * 1000.0;
            assert!(
                off_ms.abs() < tolerance_ms,
                "beat at {b:.1} is {off_ms:+.1} ms from click at {nearest:.1}"
            );
            offsets_ms.push(off_ms);
        }
        // The mean offset catches a systematic bias smaller than the
        // per-beat tolerance.
        let mean_ms = offsets_ms.iter().sum::<f64>() / offsets_ms.len() as f64;
        assert!(
            mean_ms.abs() < 12.0,
            "beats are systematically {mean_ms:+.1} ms off the clicks"
        );
    }

    /// EDM-shaped kick train (50 Hz decaying body + click attack) with
    /// exactly known onsets: detected beats must sit on the audible attack.
    /// Guards the frame-timestamp latency compensation on material where
    /// the energy channel (not just spectral flux) drives detection.
    #[test]
    fn detect_beats_land_on_kick_attacks() {
        let sr = 44_100u32;
        let interval = 60.0 * sr as f64 / 125.0;
        let len = (sr as f64 * 40.0) as usize;
        let mut x = vec![0.0f32; len];
        let mut truth = Vec::new();
        let mut pos = 0.5 * sr as f64;
        while pos + 4000.0 < len as f64 {
            let base = pos as usize;
            truth.push(pos);
            for j in 0..3000usize {
                let t = j as f32 / sr as f32;
                x[base + j] += 0.9 * (-t * 18.0).exp() * (std::f32::consts::TAU * 50.0 * t).sin();
            }
            for j in 0..64usize {
                x[base + j] += 0.5 * (1.0 - j as f32 / 64.0) * if j % 2 == 0 { 1.0 } else { -0.8 };
            }
            pos += interval;
        }

        let grid = detect_beats(&x, sr);
        // Tight bound: the inlier-mean estimator cancels the per-beat
        // frame-grid sawtooth that used to read a flat 125.0 as ~125.23.
        // Residual error is endpoint beat jitter over the track span.
        assert!(
            (grid.bpm - 125.0).abs() < 0.1,
            "BPM {:.3} (expected 125.000 ± 0.1)",
            grid.bpm
        );
        assert!(grid.beats.len() > 30, "beats {}", grid.beats.len());

        // First/last beats sit at the DP chain edges where the tracker has
        // no neighbor on one side; measure the interior.
        let offsets_ms: Vec<f64> = grid.beats[1..grid.beats.len() - 1]
            .iter()
            .map(|&b| {
                let nearest = truth
                    .iter()
                    .map(|&t| b - t)
                    .min_by(|a, c| a.abs().total_cmp(&c.abs()))
                    .unwrap();
                nearest / sr as f64 * 1000.0
            })
            .collect();
        // Per-beat: bounded jitter (occasional one-frame stragglers from
        // the sub-hop refinement are ~±1.5 hops). Mean: the actual bias
        // guard — the uncompensated window-start convention sat at −30 ms.
        for &off in &offsets_ms {
            assert!(off.abs() < 25.0, "beat {off:+.1} ms from its kick");
        }
        let mean_ms = offsets_ms.iter().sum::<f64>() / offsets_ms.len() as f64;
        assert!(
            mean_ms.abs() < 5.0,
            "beats are systematically {mean_ms:+.1} ms off the kicks"
        );
    }

    #[test]
    fn detect_beats_90_not_octave_folded() {
        let sample_rate = 44100u32;
        let grid = detect_beats(&click_train(sample_rate, 90.0, 12.0, 0, 0), sample_rate);
        assert!(
            (grid.bpm - 90.0).abs() < 2.0,
            "90 BPM must not fold into the EDM range, got {:.2}",
            grid.bpm
        );
    }

    #[test]
    fn detect_beats_silence_is_empty() {
        let grid = detect_beats(&vec![0.0f32; 44100 * 4], 44100);
        assert_eq!(grid.bpm, 0.0);
        assert!(grid.beats.is_empty());
        assert!(grid.segments.is_empty());
        assert_eq!(grid.confidence, 0.0);
    }

    #[test]
    fn detect_beats_too_short_is_empty() {
        let grid = detect_beats(&[0.0f32; 100], 44100);
        assert_eq!(grid.bpm, 0.0);
        assert!(grid.beats.is_empty());
    }

    #[test]
    fn downbeats_follow_accents() {
        let sample_rate = 44100u32;
        // Accent every 4 beats, starting on beat 0.
        let grid = detect_beats(&click_train(sample_rate, 120.0, 16.0, 4, 0), sample_rate);
        assert!(grid.beats.len() >= MIN_BEATS_FOR_DOWNBEATS);
        assert!(
            !grid.downbeats.is_empty(),
            "accented pattern should produce downbeats"
        );
        assert!(
            grid.downbeat_confidence > 0.05,
            "accented pattern should give a confident phase, got {}",
            grid.downbeat_confidence
        );
        // Downbeats should be 4 beats apart.
        for w in grid.downbeats.windows(2) {
            assert_eq!(w[1] - w[0], BEATS_PER_BAR);
        }
        // The downbeat should fall on the accented (stronger) clicks: check
        // the first downbeat is within a beat interval of a multiple of the
        // bar length.
        let bar = 4.0 * 60.0 * sample_rate as f64 / 120.0;
        let first_downbeat = grid.beats[grid.downbeats[0]];
        let nearest_bar = (first_downbeat / bar).round() * bar;
        assert!(
            (first_downbeat - nearest_bar).abs() < bar * 0.1,
            "first downbeat {:.0} not on an accent (nearest bar {:.0})",
            first_downbeat,
            nearest_bar
        );
    }

    #[test]
    fn tempo_ramp_produces_multiple_segments() {
        let sample_rate = 44100u32;
        // Two constant halves: 120 BPM then 132 BPM.
        let mut samples = click_train(sample_rate, 120.0, 10.0, 0, 0);
        samples.extend(click_train(sample_rate, 132.0, 10.0, 0, 0));
        let grid = detect_beats(&samples, sample_rate);
        assert!(
            grid.segments.len() >= 2,
            "tempo step should split segments, got {:?}",
            grid.segments
        );
        let first = grid.segments.first().unwrap().bpm;
        let last = grid.segments.last().unwrap().bpm;
        assert!(
            (first - 120.0).abs() < 3.0,
            "first segment should be ~120, got {:.2}",
            first
        );
        assert!(
            (last - 132.0).abs() < 3.0,
            "last segment should be ~132, got {:.2}",
            last
        );
    }

    #[test]
    fn tempo_candidates_ranked_with_octave_alternative() {
        let sample_rate = 44100u32;
        let samples = click_train(sample_rate, 120.0, 20.0, 0, 0);
        let grid = detect_beats(&samples, sample_rate);
        assert!(grid.bpm > 0.0);
        assert!(!grid.tempo_candidates.is_empty());

        // Sorted by salience, and on a clean click train the committed
        // tempo ranks first.
        assert!(
            grid.tempo_candidates
                .windows(2)
                .all(|w| w[0].salience >= w[1].salience)
        );
        let first = &grid.tempo_candidates[0];
        assert!(
            (first.bpm - grid.bpm).abs() < 1e-9,
            "committed tempo should rank first: {:.2} vs {:.2}",
            first.bpm,
            grid.bpm
        );

        // The half-tempo alternative is in range (60 BPM) and exactly
        // half the committed BPM; the double (240) exceeds max_bpm and
        // must not be offered.
        let half = grid
            .tempo_candidates
            .iter()
            .find(|c| (c.bpm - grid.bpm * 0.5).abs() < 1e-9)
            .expect("half-tempo candidate");
        assert!(half.salience > 0.0 && half.salience < first.salience);
        // The double (240) exceeds max_bpm and must not be offered; the
        // 3/2 metrical alternative (180) IS in range and may appear, but
        // on a clean click train its salience must sit far below the
        // metrical-level re-check threshold (no false second pass).
        assert!(
            grid.tempo_candidates
                .iter()
                .all(|c| (c.bpm - grid.bpm * 2.0).abs() > 1e-6),
            "240 BPM double is outside the default range: {:?}",
            grid.tempo_candidates
        );
        if let Some(three_halves) = grid
            .tempo_candidates
            .iter()
            .find(|c| (c.bpm - grid.bpm * 1.5).abs() < 1e-6)
        {
            assert!(
                three_halves.salience < METRICAL_32_SALIENCE_MIN,
                "click train must not tempt the metrical re-check: {:?}",
                three_halves
            );
        }
    }

    #[test]
    fn empty_grid_has_no_tempo_candidates() {
        assert!(BeatGrid::empty(44100).tempo_candidates.is_empty());
        let silent = vec![0.0f32; 44100 * 4];
        assert!(
            detect_beats(&silent, 44100).tempo_candidates.is_empty(),
            "silence must not offer tempo candidates"
        );
    }

    // --- BeatGrid helpers ---

    #[test]
    fn test_beat_grid_snap() {
        let grid = make_grid(vec![0.0, 22050.0, 44100.0, 66150.0], 120.0, 44100);
        assert_eq!(grid.snap_to_grid(100), 0);
        assert_eq!(grid.snap_to_grid(22000), 22050);
        assert_eq!(grid.snap_to_grid(33000), 22050);
    }

    #[test]
    fn test_beat_interval_samples_120bpm() {
        let grid = make_grid(vec![0.0], 120.0, 44100);
        assert!((grid.beat_interval_samples() - 22050.0).abs() < 1.0);
    }

    #[test]
    fn test_beat_interval_samples_128bpm_48khz() {
        let grid = make_grid(vec![0.0], 128.0, 48000);
        assert!((grid.beat_interval_samples() - 22500.0).abs() < 1.0);
    }

    #[test]
    fn test_snap_to_grid_empty_beats() {
        let grid = BeatGrid::empty(44100);
        assert_eq!(grid.snap_to_grid(1000), 1000);
    }

    #[test]
    fn test_snap_to_grid_before_first_beat() {
        let grid = make_grid(vec![1000.0, 2000.0, 3000.0], 120.0, 44100);
        assert_eq!(grid.snap_to_grid(500), 1000);
    }

    #[test]
    fn test_snap_to_grid_after_last_beat() {
        let grid = make_grid(vec![1000.0, 2000.0, 3000.0], 120.0, 44100);
        assert_eq!(grid.snap_to_grid(10000), 3000);
    }

    #[test]
    fn test_snap_to_grid_exact_beat() {
        let grid = make_grid(vec![0.0, 22050.0, 44100.0], 120.0, 44100);
        assert_eq!(grid.snap_to_grid(22050), 22050);
    }

    #[test]
    fn test_snap_to_grid_fractional_basic() {
        let grid = make_grid(vec![0.0, 22050.5, 44100.25], 120.0, 44100);
        let snapped = grid.snap_to_grid_fractional(22000.0);
        assert!(
            (snapped - 22050.5).abs() < 1e-9,
            "Should snap to 22050.5, got {}",
            snapped
        );
    }

    #[test]
    fn test_snap_to_grid_fractional_empty() {
        let grid = BeatGrid::empty(44100);
        let snapped = grid.snap_to_grid_fractional(1000.0);
        assert!((snapped - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_beat_index() {
        let grid = make_grid(vec![0.0, 1000.0, 2000.0], 120.0, 44100);
        assert_eq!(grid.nearest_beat_index(-50.0), Some(0));
        assert_eq!(grid.nearest_beat_index(400.0), Some(0));
        assert_eq!(grid.nearest_beat_index(600.0), Some(1));
        assert_eq!(grid.nearest_beat_index(2600.0), Some(2));
        assert_eq!(BeatGrid::empty(44100).nearest_beat_index(100.0), None);
    }

    /// The exact failure mode the baseline estimator exists for: beats at
    /// a flat 480 ms period with a cyclic 3-beat position error
    /// (frame-grid quantization sawtooth). Adjacent intervals cycle
    /// {483.3, 477.5, 479.2} ms, whose median (479.2 → 125.2 BPM) misreads
    /// the tempo; multi-beat baselines telescope the error away.
    #[test]
    fn median_bpm_immune_to_cyclic_position_error() {
        let sr = 44_100u32;
        let interval = 60.0 * sr as f64 / 125.0; // 21168 samples
        let err_ms = [-3.4, -0.1, -2.6]; // repeating per-beat error
        let beats: Vec<f64> = (0..64)
            .map(|k| k as f64 * interval + err_ms[k % 3] * sr as f64 / 1000.0)
            .collect();
        let bpm = median_bpm(&beats, sr);
        assert!(
            (bpm - 125.0).abs() < 0.02,
            "cyclic position error must not bias the BPM, got {bpm:.3}"
        );
    }

    #[test]
    fn median_bpm_robust_to_interval_outliers() {
        let sr = 44_100u32;
        let interval = 60.0 * sr as f64 / 120.0;
        // Flat 120 BPM with one dropped beat (double-length gap) mid-track.
        let mut beats: Vec<f64> = (0..40).map(|k| k as f64 * interval).collect();
        for b in beats.iter_mut().skip(20) {
            *b += interval;
        }
        let bpm = median_bpm(&beats, sr);
        assert!(
            (bpm - 120.0).abs() < 3.0,
            "a single dropped beat must not derail the median, got {bpm:.3}"
        );
    }

    #[test]
    fn test_bpm_at_segments() {
        let grid = BeatGrid {
            beats: vec![0.0, 22050.0, 44100.0, 64150.0, 84200.0],
            downbeats: vec![0, 4],
            segments: vec![
                TempoSegment {
                    start_beat: 0,
                    bpm: 120.0,
                },
                TempoSegment {
                    start_beat: 2,
                    bpm: 132.0,
                },
            ],
            bpm: 126.0,
            confidence: 1.0,
            downbeat_confidence: 1.0,
            sample_rate: 44100,
            tempo_candidates: Vec::new(),
            phase_untrusted: false,
        };
        assert!((grid.bpm_at(10000.0) - 120.0).abs() < 1e-9);
        assert!((grid.bpm_at(70000.0) - 132.0).abs() < 1e-9);
        // Before the first beat: first segment's tempo.
        assert!((grid.bpm_at(-10.0) - 120.0).abs() < 1e-9);
        // Empty grid: representative bpm (0.0).
        assert_eq!(BeatGrid::empty(44100).bpm_at(0.0), 0.0);
    }

    #[test]
    fn test_beats_rounded_and_downbeat_positions() {
        let grid = make_grid(vec![0.4, 1000.6, 2000.0, 3000.0, 4000.0], 120.0, 44100);
        assert_eq!(grid.beats_rounded(), vec![0, 1001, 2000, 3000, 4000]);
        assert_eq!(grid.downbeat_positions(), vec![0.4, 4000.0]);
    }

    // --- generate_subdivision_grid ---

    #[test]
    fn test_generate_subdivision_grid_120bpm_1sec() {
        let grid = generate_subdivision_grid(120.0, 44100, 44100, 16);
        assert_eq!(
            grid.len(),
            32,
            "Expected 32 subdivision positions, got {}",
            grid.len()
        );
        assert!((grid[0] - 0.0).abs() < 1e-10, "First position should be 0");
        let expected_interval = 60.0 * 44100.0 / 120.0 / 16.0;
        for i in 1..grid.len() {
            let interval = grid[i] - grid[i - 1];
            assert!(
                (interval - expected_interval).abs() < 1e-6,
                "Interval {} at position {} should be {}, got {}",
                i,
                grid[i],
                expected_interval,
                interval
            );
        }
    }

    #[test]
    fn test_generate_subdivision_grid_zero_bpm() {
        assert!(generate_subdivision_grid(0.0, 44100, 44100, 16).is_empty());
    }

    #[test]
    fn test_generate_subdivision_grid_zero_subdivision() {
        assert!(generate_subdivision_grid(120.0, 44100, 44100, 0).is_empty());
    }

    #[test]
    fn test_generate_subdivision_grid_zero_samples() {
        assert!(generate_subdivision_grid(120.0, 44100, 0, 16).is_empty());
    }

    #[test]
    fn test_generate_subdivision_grid_quarter_notes() {
        let grid = generate_subdivision_grid(128.0, 48000, 96000, 1);
        assert_eq!(
            grid.len(),
            5,
            "Expected 5 beat positions, got {}",
            grid.len()
        );
    }

    // --- snap_to_subdivision ---

    #[test]
    fn test_snap_to_subdivision_exact_on_grid() {
        let grid = vec![0.0, 1000.0, 2000.0, 3000.0];
        assert_eq!(snap_to_subdivision(1000.0, &grid, 220.0), Some(1000.0));
    }

    #[test]
    fn test_snap_to_subdivision_within_tolerance() {
        let grid = vec![0.0, 1000.0, 2000.0, 3000.0];
        let tolerance = 44100.0 * 0.005;
        assert_eq!(snap_to_subdivision(1132.0, &grid, tolerance), Some(1000.0));
    }

    #[test]
    fn test_snap_to_subdivision_outside_tolerance() {
        let grid = vec![0.0, 1000.0, 2000.0, 3000.0];
        let tolerance = 44100.0 * 0.005;
        assert_eq!(snap_to_subdivision(1441.0, &grid, tolerance), None);
    }

    #[test]
    fn test_snap_to_subdivision_empty_grid() {
        assert_eq!(snap_to_subdivision(1000.0, &[], 220.0), None);
    }

    #[test]
    fn test_snap_to_subdivision_snaps_to_nearest() {
        let grid = vec![0.0, 1000.0, 2000.0];
        assert_eq!(snap_to_subdivision(1800.0, &grid, 250.0), Some(2000.0));
    }

    #[test]
    fn test_snap_to_subdivision_first_position() {
        let grid = vec![0.0, 1000.0, 2000.0];
        assert_eq!(snap_to_subdivision(50.0, &grid, 100.0), Some(0.0));
    }

    #[test]
    fn test_snap_to_subdivision_last_position() {
        let grid = vec![0.0, 1000.0, 2000.0];
        assert_eq!(snap_to_subdivision(1990.0, &grid, 100.0), Some(2000.0));
    }

    // --- Integration: snapping transients to a beat grid ---

    #[test]
    fn test_snap_transients_to_beat_grid_integration() {
        let sample_rate = 44100u32;
        let bpm = 128.0;
        let num_samples = sample_rate as usize * 2;
        let beat_interval = (60.0 * sample_rate as f64 / bpm) as usize;

        let mut samples = vec![0.0f32; num_samples];
        let mut true_beat_positions = Vec::new();
        for beat in 0..5 {
            let pos = beat * beat_interval;
            if pos >= num_samples {
                break;
            }
            true_beat_positions.push(pos);
            for j in 0..20.min(num_samples - pos) {
                samples[pos + j] = if j < 5 { 1.0 } else { -0.5 };
            }
        }

        let transients =
            crate::analysis::transient::detect_transients(&samples, sample_rate, 2048, 512, 0.4);

        let grid = generate_subdivision_grid(bpm, sample_rate, num_samples, 16);
        let tolerance = sample_rate as f64 * 0.005;

        let snapped: Vec<usize> = transients
            .onsets
            .iter()
            .filter_map(|&onset| {
                snap_to_subdivision(onset as f64, &grid, tolerance).map(|s| s.round() as usize)
            })
            .collect();

        let tolerance_2ms = (sample_rate as f64 * 0.002) as usize;
        for &snapped_pos in &snapped {
            let near_beat = true_beat_positions.iter().any(|&beat| {
                snapped_pos.abs_diff(beat) <= tolerance_2ms || {
                    let sub_interval = beat_interval as f64 / 16.0;
                    let nearest_sub = (snapped_pos as f64 / sub_interval).round() * sub_interval;
                    (snapped_pos as f64 - nearest_sub).abs() <= tolerance_2ms as f64
                }
            });
            assert!(
                near_beat,
                "Snapped position {} should be near a beat subdivision",
                snapped_pos
            );
        }
    }
}

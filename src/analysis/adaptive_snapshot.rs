//! Shared adaptive analysis snapshot for hybrid and dual-plane paths.

use crate::analysis::beat::{default_subdivision_for_preset, detect_beats, snap_to_subdivision};
use crate::analysis::frequency::{compute_band_energy, FrequencyBands};
use crate::analysis::transient::{
    detect_transients_with_options, TransientDetectionOptions, TransientMap,
};
use crate::core::types::StretchParams;
use std::collections::BTreeMap;

/// Maximum FFT size for transient detection (smaller = faster, less frequency resolution).
const TRANSIENT_MAX_FFT: usize = 2048;
/// Maximum hop size for transient detection.
const TRANSIENT_MAX_HOP: usize = 512;
/// Minimum input length (in samples) for beat detection to be worthwhile.
const MIN_SAMPLES_FOR_BEAT_DETECTION: usize = 44_100;
/// Minimum distance (samples) between merged onset/beat positions.
const DEDUP_DISTANCE: usize = 512;
/// Sentinel strength value marking a beat-only segmentation anchor.
pub(crate) const BEAT_ANCHOR_STRENGTH: f32 = f32::NEG_INFINITY;
/// Hard bound for dynamic subdivision-grid generation.
const MAX_SUBDIVISION_GRID_POINTS: usize = 1_000_000;
/// Minimum transient anchors required before enabling live beat-grid merging.
const MIN_LIVE_BEAT_ANCHORS: usize = 2;
/// Minimum anchor strength considered "reliable" for live beat-grid merging.
const MIN_LIVE_BEAT_ANCHOR_STRENGTH: f32 = 0.2;

/// Shared adaptive analysis result for a mono horizon.
#[derive(Debug, Clone)]
pub(crate) struct AdaptiveAnalysisSnapshot {
    pub transient_map: TransientMap,
    pub onsets: Vec<usize>,
    pub strengths: Vec<f32>,
    pub transient_confidence: f32,
    pub beat_confidence: f32,
    pub tonal_confidence: f32,
    pub noise_confidence: f32,
    pub lane_bias: [f32; 3],
    pub ratio_bias: f64,
    pub transient_mask: Vec<f32>,
}

/// Builds a shared adaptive snapshot for mono content.
pub(crate) fn analyze_adaptive_snapshot_mono(
    input: &[f32],
    params: &StretchParams,
) -> AdaptiveAnalysisSnapshot {
    let transient_map = if input.is_empty() {
        empty_transient_map(params.hop_size.max(1))
    } else {
        detect_transients_with_options(
            input,
            params.sample_rate,
            params.fft_size.min(TRANSIENT_MAX_FFT),
            params.hop_size.min(TRANSIENT_MAX_HOP),
            params.transient_sensitivity,
            TransientDetectionOptions::from_stretch_params(params),
        )
    };

    let confident_pre = params.pre_analysis.as_ref().filter(|artifact| {
        artifact.sample_rate == params.sample_rate
            && artifact.is_confident(params.beat_snap_confidence_threshold)
            && !artifact.beat_positions.is_empty()
    });

    let mut onsets = transient_map.onsets.clone();
    let mut strengths = if transient_map.strengths.len() == transient_map.onsets.len() {
        transient_map.strengths.clone()
    } else {
        vec![1.0; transient_map.onsets.len()]
    };

    let mut detected_beat_grid = None;

    // Optionally merge live/offline beats into transient anchors.
    if params.beat_aware && input.len() >= MIN_SAMPLES_FOR_BEAT_DETECTION {
        let use_live_beats =
            confident_pre.is_some() || should_use_live_beat_aware_anchors(&strengths);
        if use_live_beats {
            let beats = if let Some(artifact) = confident_pre {
                artifact.beat_positions.as_slice()
            } else {
                let grid = detected_beat_grid
                    .get_or_insert_with(|| detect_beats(input, params.sample_rate));
                grid.beats.as_slice()
            };
            let (merged_onsets, merged_strengths) =
                merge_onsets_and_beats(&onsets, &strengths, beats, input.len());
            onsets = merged_onsets;
            strengths = merged_strengths;
        }
    }

    // Optionally snap transient anchors to beat subdivisions.
    let snap_bpm = params
        .bpm
        .or_else(|| confident_pre.map(|artifact| artifact.bpm))
        .filter(|bpm| bpm.is_finite() && *bpm > 0.0);

    if let Some(bpm) = snap_bpm {
        let tolerance_samples =
            params.sample_rate as f64 * (params.beat_snap_tolerance_ms / 1000.0).max(0.001);
        let subdivision = default_subdivision_for_preset(params.preset);
        let phase_offset = confident_pre
            .map(|artifact| artifact.downbeat_offset_samples)
            .unwrap_or(0);
        let beat_grid = generate_subdivision_grid_with_phase(
            bpm,
            params.sample_rate,
            input.len(),
            subdivision,
            phase_offset,
        );

        let strict_suppression = confident_pre.is_some();
        let had_transients = strengths.iter().copied().any(strength_marks_transient);
        let mut snapped: BTreeMap<usize, f32> = BTreeMap::new();
        for (i, &onset) in onsets.iter().enumerate() {
            let strength = strengths.get(i).copied().unwrap_or(1.0);
            let is_transient = strength_marks_transient(strength);
            let chosen = if is_transient {
                match snap_to_subdivision(onset as f64, &beat_grid, tolerance_samples) {
                    Some(snapped) => snapped.round() as usize,
                    None if strict_suppression => continue,
                    None => onset,
                }
            } else {
                onset
            };

            snapped
                .entry(chosen)
                .and_modify(|existing| *existing = merge_anchor_strength(*existing, strength))
                .or_insert(strength);
        }

        let snapped_has_transients = snapped.values().copied().any(strength_marks_transient);
        if !snapped.is_empty() && (!had_transients || snapped_has_transients) {
            onsets = snapped.keys().copied().collect();
            strengths = snapped.values().copied().collect();
        }
    }

    let mut transient_onsets = Vec::with_capacity(onsets.len());
    let mut transient_strengths = Vec::with_capacity(strengths.len());
    for (i, &onset) in onsets.iter().enumerate() {
        let strength = strengths.get(i).copied().unwrap_or(1.0);
        if strength_marks_transient(strength) {
            transient_onsets.push(onset);
            transient_strengths.push(strength);
        }
    }
    let transient_mask = build_transient_mask(
        input.len(),
        params.sample_rate,
        params.transient_region_secs,
        &transient_onsets,
        &transient_strengths,
    );

    let duration_secs = input.len() as f32 / params.sample_rate.max(1) as f32;
    let transient_density = transient_onsets.len() as f32 / duration_secs.max(1e-3);
    let transient_confidence = (transient_density / 8.0).clamp(0.0, 1.0);

    let beat_confidence = if input.len() >= params.sample_rate as usize {
        if let Some(artifact) = confident_pre {
            beat_confidence_from_stats(artifact.beat_positions.len(), artifact.bpm)
                * artifact.confidence.clamp(0.0, 1.0)
        } else {
            let grid =
                detected_beat_grid.get_or_insert_with(|| detect_beats(input, params.sample_rate));
            beat_confidence_from_stats(grid.beats.len(), grid.bpm)
        }
    } else {
        0.0
    };

    let analysis_fft = params.fft_size.min(2048).max(256);
    let tonal_confidence = if input.len() >= analysis_fft {
        let (sub, low, mid, high) = compute_band_energy(
            &input[..analysis_fft],
            analysis_fft,
            params.sample_rate,
            &FrequencyBands::default(),
        );
        let total = sub + low + mid + high + 1e-9;
        ((sub + low + mid) / total).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let noise_confidence = (1.0 - tonal_confidence).clamp(0.0, 1.0);
    let lane_bias = [transient_confidence, tonal_confidence, noise_confidence];
    let ratio_bias =
        ((beat_confidence as f64 - transient_confidence as f64) * 0.04).clamp(-0.08, 0.08);

    AdaptiveAnalysisSnapshot {
        transient_map,
        onsets,
        strengths,
        transient_confidence,
        beat_confidence,
        tonal_confidence,
        noise_confidence,
        lane_bias,
        ratio_bias,
        transient_mask,
    }
}

#[inline]
fn empty_transient_map(hop_size: usize) -> TransientMap {
    TransientMap {
        onsets: Vec::new(),
        onsets_fractional: Vec::new(),
        strengths: Vec::new(),
        flux: Vec::new(),
        hop_size: hop_size.max(1),
        per_frame_band_flux: Vec::new(),
    }
}

#[inline]
fn beat_confidence_from_stats(beat_count: usize, bpm: f64) -> f32 {
    if bpm <= 0.0 || beat_count < 2 {
        return 0.0;
    }
    let beat_count = (beat_count.min(16) as f32) / 16.0;
    let bpm_center_error = ((bpm - 128.0).abs() / 96.0).clamp(0.0, 1.0) as f32;
    (beat_count * (1.0 - bpm_center_error)).clamp(0.0, 1.0)
}

/// Returns true when an anchor strength encodes a real transient.
#[inline]
pub(crate) fn strength_marks_transient(strength: f32) -> bool {
    strength.is_finite() && strength >= 0.0
}

/// Merges two anchor strengths at the same position.
///
/// Transients always win over beat-only anchors. If both are transients,
/// the stronger one is kept.
#[inline]
fn merge_anchor_strength(existing: f32, candidate: f32) -> f32 {
    let existing_is_transient = strength_marks_transient(existing);
    let candidate_is_transient = strength_marks_transient(candidate);

    match (existing_is_transient, candidate_is_transient) {
        (false, true) => candidate,
        (true, false) => existing,
        (true, true) => existing.max(candidate),
        (false, false) => existing,
    }
}

#[inline]
pub(crate) fn should_use_live_beat_aware_anchors(strengths: &[f32]) -> bool {
    strengths
        .iter()
        .copied()
        .filter(|&s| strength_marks_transient(s) && s >= MIN_LIVE_BEAT_ANCHOR_STRENGTH)
        .count()
        >= MIN_LIVE_BEAT_ANCHORS
}

/// Merges transient onsets with beat grid positions, deduplicating nearby entries.
///
/// Beat positions that fall within `DEDUP_DISTANCE` samples of an existing
/// transient onset are dropped to avoid creating overly short segments.
///
/// The returned strengths are aligned with returned onset positions:
/// - finite `>= 0.0`: transient anchor with that strength
/// - non-finite (`BEAT_ANCHOR_STRENGTH`): beat-only anchor
pub(crate) fn merge_onsets_and_beats(
    onsets: &[usize],
    strengths: &[f32],
    beats: &[usize],
    input_len: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut merged_positions: Vec<usize> = Vec::with_capacity(onsets.len() + beats.len());
    let mut merged_strengths: Vec<f32> = Vec::with_capacity(onsets.len() + beats.len());

    for (i, &onset) in onsets.iter().enumerate() {
        if onset >= input_len {
            continue;
        }
        merged_positions.push(onset);
        merged_strengths.push(strengths.get(i).copied().unwrap_or(1.0));
    }

    for &beat in beats {
        if beat >= input_len {
            continue;
        }
        let too_close = merged_positions
            .iter()
            .any(|&pos| pos.abs_diff(beat) < DEDUP_DISTANCE);
        if !too_close {
            merged_positions.push(beat);
            merged_strengths.push(BEAT_ANCHOR_STRENGTH);
        }
    }

    let mut pairs: Vec<(usize, f32)> = merged_positions.into_iter().zip(merged_strengths).collect();
    pairs.sort_unstable_by_key(|(pos, _)| *pos);

    let mut out_onsets = Vec::with_capacity(pairs.len());
    let mut out_strengths = Vec::with_capacity(pairs.len());
    for (pos, strength) in pairs {
        if let Some(last_pos) = out_onsets.last().copied() {
            if last_pos == pos {
                if let Some(last_strength) = out_strengths.last_mut() {
                    *last_strength = merge_anchor_strength(*last_strength, strength);
                }
                continue;
            }
        }
        out_onsets.push(pos);
        out_strengths.push(strength);
    }

    (out_onsets, out_strengths)
}

/// Generates a subdivision grid with a phase/downbeat offset.
pub(crate) fn generate_subdivision_grid_with_phase(
    bpm: f64,
    sample_rate: u32,
    total_samples: usize,
    subdivision: u32,
    phase_offset_samples: usize,
) -> Vec<f64> {
    if bpm <= 0.0 || subdivision == 0 || total_samples == 0 {
        return Vec::new();
    }

    let beat_interval_samples = 60.0 * sample_rate as f64 / bpm;
    let sub_interval = beat_interval_samples / subdivision as f64;
    if sub_interval <= 0.0 {
        return Vec::new();
    }

    let phase = (phase_offset_samples as f64).rem_euclid(sub_interval);
    let estimated_count = (total_samples as f64 / sub_interval).ceil() as usize + 1;
    let max_points = estimated_count.min(MAX_SUBDIVISION_GRID_POINTS);
    let mut grid = Vec::with_capacity(max_points);
    let mut pos = phase;
    for _ in 0..max_points {
        if pos >= total_samples as f64 {
            break;
        }
        grid.push(pos);
        pos += sub_interval;
    }
    grid
}

/// Builds a smoothed transient mask in input timeline space.
pub(crate) fn build_transient_mask(
    input_len: usize,
    sample_rate: u32,
    transient_region_secs: f64,
    onsets: &[usize],
    strengths: &[f32],
) -> Vec<f32> {
    if input_len == 0 {
        return Vec::new();
    }

    let mut mask = vec![0.0f32; input_len];
    if onsets.is_empty() {
        return mask;
    }

    let region_samples = (transient_region_secs * f64::from(sample_rate)).round() as usize;
    let half_width = (region_samples / 2).max(8) as isize;

    for (idx, &onset) in onsets.iter().enumerate() {
        let center = onset.min(input_len.saturating_sub(1)) as isize;
        let strength = strengths.get(idx).copied().unwrap_or(1.0).clamp(0.0, 1.0);
        let start = (center - half_width).max(0) as usize;
        let end = (center + half_width).min(input_len.saturating_sub(1) as isize) as usize;
        for i in start..=end {
            let dist = (i as isize - center).unsigned_abs() as f32 / half_width as f32;
            let tri = (1.0 - dist).max(0.0) * strength;
            mask[i] = mask[i].max(tri);
        }
    }

    for i in 1..input_len {
        let prev = mask[i - 1];
        let cur = mask[i];
        mask[i] = (0.7 * prev + 0.3 * cur).clamp(0.0, 1.0);
    }
    mask
}

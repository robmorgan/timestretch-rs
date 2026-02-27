//! Hybrid stretcher combining WSOLA (transients) with phase vocoder (tonal content).

use crate::analysis::beat::{default_subdivision_for_preset, detect_beats, snap_to_subdivision};
use crate::analysis::frequency::freq_to_bin;
use crate::analysis::hpss::{hpss, HpssParams};
use crate::analysis::transient::detect_transients;
use crate::core::fft::{COMPLEX_ZERO, WINDOW_SUM_EPSILON, WINDOW_SUM_FLOOR_RATIO};
use crate::core::types::StretchParams;
use crate::core::window::{generate_window, WindowType};
use crate::error::StretchError;
use crate::stretch::multi_resolution::MultiResolutionStretcher;
use crate::stretch::phase_vocoder::PhaseVocoder;
use crate::stretch::wsola::Wsola;
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::BTreeMap;

/// Minimum segment length (samples) to use phase vocoder or WSOLA; shorter segments
/// fall back to linear resampling.
const MIN_SEGMENT_FOR_STRETCH: usize = 256;
/// Minimum WSOLA segment size (samples) when clamping for short segments.
const MIN_WSOLA_SEGMENT: usize = 64;
/// Minimum WSOLA search range (samples) when clamping for short segments.
const MIN_WSOLA_SEARCH: usize = 16;
/// Maximum FFT size for transient detection (smaller = faster, less frequency resolution).
const TRANSIENT_MAX_FFT: usize = 2048;
/// Maximum hop size for transient detection.
const TRANSIENT_MAX_HOP: usize = 512;
/// Minimum input length (in samples) for beat detection to be worthwhile.
/// Below this, beat detection is too unreliable to improve segmentation.
const MIN_SAMPLES_FOR_BEAT_DETECTION: usize = 44100; // ~1 second at 44.1kHz
/// FFT size used for sub-bass band splitting. Needs good low-frequency resolution.
const BAND_SPLIT_FFT_SIZE: usize = 4096;
/// Hop size for the band-splitting overlap-add filter (75% overlap).
const BAND_SPLIT_HOP: usize = BAND_SPLIT_FFT_SIZE / 4;
/// Minimum distance (samples) between merged onset/beat positions.
/// Positions closer than this are considered duplicates.
const DEDUP_DISTANCE: usize = 512;
/// Sentinel strength value marking a beat-only segmentation anchor.
///
/// Beat-only anchors create tonal boundaries but do not create transient regions.
const BEAT_ANCHOR_STRENGTH: f32 = f32::NEG_INFINITY;
/// Base direct-copy attack length for transient segments.
const TRANSIENT_ATTACK_COPY_SECS: f64 = 0.008;
/// Minimum WSOLA search time for transient decays to keep low-end alignment stable.
const TRANSIENT_DECAY_SEARCH_FLOOR_SECS: f64 = 0.012;
/// WSOLA search range boost for transient decays.
const TRANSIENT_DECAY_SEARCH_BOOST: f64 = 2.0;
/// RMS threshold (linear) below which audio is considered silence for the
/// leading-silence bypass in tonal segments. Approximately -66 dB.
const LEADING_SILENCE_RMS_THRESHOLD: f32 = 5e-4;
/// Crest-factor threshold used to detect sparse impulse-like content.
const IMPULSIVE_CREST_THRESHOLD: f32 = 20.0;
/// Samples above this fraction of peak are considered "strong".
const IMPULSIVE_STRONG_SAMPLE_FRACTION: f32 = 0.5;
/// Maximum number of strong samples for the signal to be treated as sparse/impulsive.
const IMPULSIVE_MAX_STRONG_SAMPLES: usize = 8;

/// Transient-aware hybrid stretcher.
///
/// Uses WSOLA for transient regions (kicks, snares, hats) and phase vocoder
/// for tonal regions (pads, bass, vocals). Crossfades between segments.
pub struct HybridStretcher {
    params: StretchParams,
}

/// A segment of audio classified as either transient or tonal.
#[derive(Debug)]
struct Segment {
    start: usize,
    end: usize,
    is_transient: bool,
    /// Per-segment stretch ratio. Defaults to the global ratio but may
    /// differ when elastic beat distribution is active.
    stretch_ratio: f64,
}

/// Timeline bookkeeping for exact-length hybrid rendering.
///
/// The core invariant is:
/// `cumulative_synthesis_len - boundary_overlap_len = expected_concat_len`
/// and after correction:
/// `expected_concat_len + duration_correction_frames = final_output_len`.
#[derive(Debug, Clone)]
struct TimelineBookkeeping {
    target_output_len: usize,
    cumulative_synthesis_len: usize,
    boundary_overlap_len: usize,
    expected_concat_len: usize,
    final_output_len: usize,
    duration_correction_frames: isize,
}

impl TimelineBookkeeping {
    fn from_lengths(
        target_output_len: usize,
        segment_target_lens: &[usize],
        boundary_overlaps: &[usize],
        final_output_len: usize,
    ) -> Self {
        let cumulative_synthesis_len = segment_target_lens.iter().sum::<usize>();
        let boundary_overlap_len = boundary_overlaps.iter().sum::<usize>();
        let expected_concat_len = cumulative_synthesis_len.saturating_sub(boundary_overlap_len);
        let duration_correction_frames = final_output_len as isize - expected_concat_len as isize;
        Self {
            target_output_len,
            cumulative_synthesis_len,
            boundary_overlap_len,
            expected_concat_len,
            final_output_len,
            duration_correction_frames,
        }
    }

    fn is_consistent(&self) -> bool {
        let recomputed_concat = self
            .cumulative_synthesis_len
            .saturating_sub(self.boundary_overlap_len);
        let corrected_concat =
            (self.expected_concat_len as isize + self.duration_correction_frames).max(0) as usize;
        recomputed_concat == self.expected_concat_len
            && corrected_concat == self.final_output_len
            && self.final_output_len == self.target_output_len
    }
}

/// Returns true when an anchor strength encodes a real transient.
#[inline]
fn strength_marks_transient(strength: f32) -> bool {
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
fn is_sparse_impulsive(signal: &[f32]) -> bool {
    // Restrict this heuristic to long one-shot buffers; applying it to small
    // streaming chunks can skew the effective ratio.
    if signal.len() < MIN_SAMPLES_FOR_BEAT_DETECTION {
        return false;
    }

    let peak = signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak <= 1e-6 {
        return false;
    }

    let rms = (signal.iter().map(|&s| s * s).sum::<f32>() / signal.len() as f32).sqrt();
    if rms <= 1e-9 {
        return false;
    }
    if peak / rms < IMPULSIVE_CREST_THRESHOLD {
        return false;
    }

    let strong_threshold = peak * IMPULSIVE_STRONG_SAMPLE_FRACTION;
    let strong_count = signal
        .iter()
        .filter(|&&sample| sample.abs() >= strong_threshold)
        .count();
    strong_count <= IMPULSIVE_MAX_STRONG_SAMPLES
}

impl HybridStretcher {
    /// Creates a new hybrid stretcher.
    pub fn new(params: StretchParams) -> Self {
        Self { params }
    }

    /// Updates the global stretch ratio used for subsequent processing.
    pub fn set_stretch_ratio(&mut self, ratio: f64) {
        self.params.stretch_ratio = ratio;
    }

    /// Stretches a mono audio signal using the hybrid algorithm.
    pub fn process(&self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        if input.is_empty() {
            return Ok(vec![]);
        }

        // Sparse impulses are poorly represented by tonal PV processing.
        // Keep their transient peak by using direct resampling instead.
        if is_sparse_impulsive(input) {
            let out_len = (input.len() as f64 * self.params.stretch_ratio).round() as usize;
            return Ok(crate::core::resample::resample_linear(
                input,
                out_len.max(1),
            ));
        }

        let min_size = self.params.fft_size.max(self.params.wsola_segment_size);
        if input.len() < min_size {
            // Fall back to WSOLA for very short input
            let mut wsola = Wsola::new(
                input.len().min(self.params.wsola_segment_size),
                self.params.wsola_search_range.min(input.len() / 4),
                self.params.stretch_ratio,
            );
            return wsola.process(input);
        }

        self.process_hybrid(input)
    }

    /// Core hybrid processing: transient detection + segmented WSOLA/PV.
    fn process_hybrid(&self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        // Step 1: Detect transients
        let transients = detect_transients(
            input,
            self.params.sample_rate,
            self.params.fft_size.min(TRANSIENT_MAX_FFT),
            self.params.hop_size.min(TRANSIENT_MAX_HOP),
            self.params.transient_sensitivity,
        );

        // Step 1b: Optionally detect beat grid and merge with transient onsets.
        // Prefer offline pre-analysis when confidence is sufficient.
        let confident_pre = self.params.pre_analysis.as_ref().filter(|artifact| {
            artifact.sample_rate == self.params.sample_rate
                && artifact.is_confident(self.params.beat_snap_confidence_threshold)
                && !artifact.beat_positions.is_empty()
        });

        let mut onsets = transients.onsets.clone();
        let mut strengths = if transients.strengths.len() == transients.onsets.len() {
            transients.strengths.clone()
        } else {
            vec![1.0; transients.onsets.len()]
        };

        if self.params.beat_aware && input.len() >= MIN_SAMPLES_FOR_BEAT_DETECTION {
            let live_beats;
            let beats = if let Some(artifact) = confident_pre {
                &artifact.beat_positions
            } else {
                live_beats = detect_beats(input, self.params.sample_rate).beats;
                &live_beats
            };
            let (merged_onsets, merged_strengths) =
                merge_onsets_and_beats(&onsets, &strengths, beats, input.len());
            onsets = merged_onsets;
            strengths = merged_strengths;
        }

        // Step 1c: When BPM is known, snap transient positions to the nearest
        // beat subdivision. Beat-only anchors are left unchanged; they exist
        // only to create tonal timing boundaries.
        let snap_bpm = self
            .params
            .bpm
            .or_else(|| confident_pre.map(|artifact| artifact.bpm))
            .filter(|bpm| bpm.is_finite() && *bpm > 0.0);

        if let Some(bpm) = snap_bpm {
            let tolerance_samples = self.params.sample_rate as f64
                * (self.params.beat_snap_tolerance_ms / 1000.0).max(0.001);
            let subdivision = default_subdivision_for_preset(self.params.preset);
            let phase_offset = confident_pre
                .map(|artifact| artifact.downbeat_offset_samples)
                .unwrap_or(0);
            let beat_grid = generate_subdivision_grid_with_phase(
                bpm,
                self.params.sample_rate,
                input.len(),
                subdivision,
                phase_offset,
            );

            // Keep transient strength association after snapping and preserve
            // beat-only boundaries as non-transient anchors.
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

        // Step 2: Segment audio at transient/beat boundaries
        let mut segments = self.segment_audio(input.len(), &onsets, &strengths);

        // Step 2b: Compute elastic per-segment ratios if enabled.
        // Guard: only when elastic_timing is on AND ratio != 1.0 (identity).
        if self.params.elastic_timing
            && (self.params.stretch_ratio - 1.0).abs() > 1e-6
            && segments.len() > 1
        {
            compute_elastic_ratios(
                &mut segments,
                self.params.stretch_ratio,
                self.params.elastic_anchor,
            );
        }

        self.render_with_segments(input, &segments, Some(&transients))
    }

    /// Stretches using an externally supplied shared onset map.
    ///
    /// Useful for stereo coherence: both channels can be segmented from the
    /// same onset/timing map to avoid channel divergence.
    pub fn process_with_onsets(
        &self,
        input: &[f32],
        onsets: &[usize],
        strengths: &[f32],
    ) -> Result<Vec<f32>, StretchError> {
        if input.is_empty() {
            return Ok(vec![]);
        }

        let min_size = self.params.fft_size.max(self.params.wsola_segment_size);
        if input.len() < min_size {
            let mut wsola = Wsola::new(
                input.len().min(self.params.wsola_segment_size),
                self.params.wsola_search_range.min(input.len() / 4),
                self.params.stretch_ratio,
            );
            return wsola.process(input);
        }

        let mut segments = self.segment_audio(input.len(), onsets, strengths);
        if self.params.elastic_timing
            && (self.params.stretch_ratio - 1.0).abs() > 1e-6
            && segments.len() > 1
        {
            compute_elastic_ratios(
                &mut segments,
                self.params.stretch_ratio,
                self.params.elastic_anchor,
            );
        }

        self.render_with_segments(input, &segments, None)
    }

    /// Renders pre-segmented hybrid content with exact-length timeline control.
    fn render_with_segments(
        &self,
        input: &[f32],
        segments: &[Segment],
        transients: Option<&crate::analysis::transient::TransientMap>,
    ) -> Result<Vec<f32>, StretchError> {
        // Step 3: Build explicit timeline bookkeeping for exact output length.
        let target_output_len = self.params.output_length(input.len());
        let base_segment_target_lens = compute_base_segment_target_lengths(segments);
        let crossfade_plan = match self.params.crossfade_mode {
            crate::core::types::CrossfadeMode::Fixed(secs) => {
                let crossfade = compute_fixed_crossfade_len(
                    self.params.sample_rate,
                    secs,
                    &base_segment_target_lens,
                );
                vec![crossfade; segments.len().saturating_sub(1)]
            }
            crate::core::types::CrossfadeMode::Adaptive => {
                compute_adaptive_crossfade_lens(segments, self.params.sample_rate)
            }
        };

        // Crossfades shorten concatenated output. Compensate by adding each
        // boundary overlap to the segment on the right side of that boundary.
        let mut segment_target_lens =
            compensate_segment_targets_for_crossfades(&base_segment_target_lens, &crossfade_plan);
        let desired_synthesis_len =
            target_output_len.saturating_add(crossfade_plan.iter().sum::<usize>());
        reconcile_total_segment_targets(&mut segment_target_lens, desired_synthesis_len);

        // Step 4: Process each segment with appropriate algorithm
        // Reuse a single PV instance for tonal segments (avoids FFT planner recreation)
        let mut pv = PhaseVocoder::with_options(
            self.params.fft_size,
            self.params.hop_size,
            self.params.stretch_ratio,
            self.params.sample_rate,
            self.params.sub_bass_cutoff,
            self.params.window_type,
            self.params.phase_locking_mode,
        );

        // Multi-resolution: 3-band filterbank stretcher (sub-bass / mid / high)
        let mut multi_res = if self.params.multi_resolution {
            Some(MultiResolutionStretcher::new(
                self.params.fft_size,
                self.params.stretch_ratio,
                self.params.sample_rate,
                self.params.sub_bass_cutoff,
            ))
        } else {
            None
        };

        let mut output_segments: Vec<Vec<f32>> = Vec::with_capacity(segments.len());

        for (segment_idx, segment) in segments.iter().enumerate() {
            let seg_data = &input[segment.start..segment.end];
            let stretched_raw = self.stretch_segment(
                seg_data,
                segment.is_transient,
                segment.stretch_ratio,
                &mut pv,
                &mut multi_res,
            );
            let stretched = force_segment_length(stretched_raw, segment_target_lens[segment_idx]);
            output_segments.push(stretched);

            // Reset PV phase state after transient segments so stale phase
            // from the previous tonal region doesn't contaminate the next one.
            // Use per-band reset when band flux data is available to avoid
            // disrupting phase tracking in bands where no transient occurred.
            if segment.is_transient {
                let reset_mask = transients
                    .map(|t| compute_band_reset_mask(segment.start, t))
                    .unwrap_or([true; 4]);
                if reset_mask == [true; 4] {
                    // Full reset (fallback for beat-merged onsets or when all bands active)
                    pv.reset_phase_state();
                    if let Some(ref mut mr) = multi_res {
                        mr.reset_phase_state();
                    }
                } else {
                    pv.reset_phase_state_bands(reset_mask, self.params.sample_rate);
                    if let Some(ref mut mr) = multi_res {
                        mr.reset_phase_state_bands(reset_mask, self.params.sample_rate);
                    }
                }
            }

            // Restore global ratio on PV for next segment (elastic may have changed it)
            pv.set_stretch_ratio(self.params.stretch_ratio);
            if let Some(ref mut mr) = multi_res {
                mr.set_stretch_ratio(self.params.stretch_ratio);
            }
        }

        // Step 5: Concatenate with crossfades
        // Single segment fast path avoids crossfade overhead
        if output_segments.len() == 1 {
            let single = output_segments.into_iter().next().unwrap_or_default();
            return Ok(enforce_exact_output_length(single, target_output_len));
        }

        let (output_raw, actual_crossfades) = match self.params.crossfade_mode {
            crate::core::types::CrossfadeMode::Fixed(_) => {
                concatenate_with_crossfade_report(&output_segments, &crossfade_plan)
            }
            crate::core::types::CrossfadeMode::Adaptive => {
                concatenate_with_adaptive_crossfade_report(&output_segments, &crossfade_plan)
            }
        };

        // Step 6: Enforce exact target duration and verify timeline invariants.
        let output = enforce_exact_output_length(output_raw, target_output_len);
        let timeline = TimelineBookkeeping::from_lengths(
            target_output_len,
            &segment_target_lens,
            &actual_crossfades,
            output.len(),
        );
        debug_assert!(
            timeline.is_consistent(),
            "hybrid timeline invariant failure: {:?}",
            timeline
        );

        Ok(output)
    }

    /// Stretches a single segment using the appropriate algorithm.
    ///
    /// - Very short segments (<256 samples) fall back to linear resampling
    /// - Transient segments use onset-aligned stretching (direct-copy attack + WSOLA decay)
    /// - Tonal segments long enough for FFT use the phase vocoder
    /// - When multi-resolution is enabled, tonal segments use the 3-band
    ///   [`MultiResolutionStretcher`] (sub-bass/mid/high with different FFT sizes)
    /// - Everything else (short tonal) uses WSOLA
    /// - On error, falls back to linear resampling
    ///
    /// `seg_ratio` is the per-segment stretch ratio (may differ from `self.params.stretch_ratio`
    /// when elastic beat distribution is active).
    fn stretch_segment(
        &self,
        seg_data: &[f32],
        is_transient: bool,
        seg_ratio: f64,
        pv: &mut PhaseVocoder,
        multi_res: &mut Option<MultiResolutionStretcher>,
    ) -> Vec<f32> {
        let out_len = (seg_data.len() as f64 * seg_ratio).round() as usize;

        if seg_data.len() < MIN_SEGMENT_FOR_STRETCH {
            return crate::core::resample::resample_linear(seg_data, out_len.max(1));
        }

        // Onset-aligned transient stretching: copy attack, WSOLA the decay
        if is_transient {
            return self.stretch_transient_segment_with_ratio(seg_data, seg_ratio);
        }

        // Bypass leading near-silence so the phase vocoder doesn't smear a
        // distant onset backward through its analysis window.  The silent
        // prefix is linearly resampled (perfect for silence) and only the
        // remainder is PV-processed.
        let hop = self.params.hop_size;
        if seg_data.len() > hop {
            let mut silence_end = 0usize;
            let mut pos = 0usize;
            while pos + hop <= seg_data.len() {
                let rms = (seg_data[pos..pos + hop].iter().map(|&s| s * s).sum::<f32>()
                    / hop as f32)
                    .sqrt();
                if rms >= LEADING_SILENCE_RMS_THRESHOLD {
                    break;
                }
                silence_end = pos + hop;
                pos += hop;
            }

            if silence_end > 0 && silence_end < seg_data.len() {
                let silent_out_len = (silence_end as f64 * seg_ratio).round() as usize;
                let mut result = crate::core::resample::resample_linear(
                    &seg_data[..silence_end],
                    silent_out_len.max(1),
                );
                let remainder = &seg_data[silence_end..];
                let rem_out_len = out_len.saturating_sub(silent_out_len).max(1);
                let rem_stretched = if remainder.len() < MIN_SEGMENT_FOR_STRETCH {
                    crate::core::resample::resample_linear(remainder, rem_out_len)
                } else {
                    pv.set_stretch_ratio(seg_ratio);
                    if let Some(ref mut mr) = multi_res {
                        mr.set_stretch_ratio(seg_ratio);
                    }
                    self.stretch_tonal_core(remainder, seg_ratio, pv, multi_res)
                };
                result.extend_from_slice(&rem_stretched);
                return result;
            }
        }

        // Set the PV ratio for this segment
        pv.set_stretch_ratio(seg_ratio);
        if let Some(ref mut mr) = multi_res {
            mr.set_stretch_ratio(seg_ratio);
        }

        self.stretch_tonal_core(seg_data, seg_ratio, pv, multi_res)
    }

    /// Tonal stretching core shared by [`stretch_segment`] and the
    /// leading-silence bypass path. Assumes PV ratio is already set.
    fn stretch_tonal_core(
        &self,
        seg_data: &[f32],
        seg_ratio: f64,
        pv: &mut PhaseVocoder,
        multi_res: &mut Option<MultiResolutionStretcher>,
    ) -> Vec<f32> {
        let out_len = (seg_data.len() as f64 * seg_ratio).round() as usize;
        let use_phase_vocoder = seg_data.len() >= self.params.fft_size;

        if self.params.hpss_enabled && use_phase_vocoder {
            if let Some(result) = self.stretch_tonal_hpss(seg_data, seg_ratio, pv) {
                return result;
            }
        }

        if let Some(multi) = multi_res.as_mut() {
            let result = multi.process(seg_data);
            return result.unwrap_or_else(|_| {
                crate::core::resample::resample_linear(seg_data, out_len.max(1))
            });
        }

        if self.params.band_split && use_phase_vocoder && seg_data.len() >= BAND_SPLIT_FFT_SIZE {
            return self.stretch_tonal_band_split(seg_data, seg_ratio, pv);
        }

        let result = if use_phase_vocoder {
            pv.process(seg_data)
        } else {
            self.stretch_with_wsola_ratio(seg_data, seg_ratio)
        };

        result.unwrap_or_else(|_| crate::core::resample::resample_linear(seg_data, out_len.max(1)))
    }

    /// Stretches a tonal segment using HPSS separation.
    ///
    /// Separates the segment into harmonic and percussive components,
    /// PV-stretches the harmonic part, WSOLA-stretches the percussive part,
    /// and sums the results. Returns `None` if processing fails.
    fn stretch_tonal_hpss(
        &self,
        seg_data: &[f32],
        seg_ratio: f64,
        pv: &mut PhaseVocoder,
    ) -> Option<Vec<f32>> {
        let hpss_params = HpssParams::default();
        let (harmonic, percussive) = hpss(
            seg_data,
            self.params.fft_size,
            self.params.hop_size,
            &hpss_params,
        );

        // PV-stretch harmonic component
        let harmonic_stretched = if harmonic.len() >= self.params.fft_size {
            pv.process(&harmonic).ok()?
        } else {
            let out_len = (harmonic.len() as f64 * seg_ratio).round() as usize;
            crate::core::resample::resample_linear(&harmonic, out_len.max(1))
        };

        // WSOLA-stretch percussive component
        let percussive_out_len = (percussive.len() as f64 * seg_ratio).round() as usize;
        let percussive_stretched = self
            .stretch_with_wsola_ratio(&percussive, seg_ratio)
            .unwrap_or_else(|_| {
                crate::core::resample::resample_linear(&percussive, percussive_out_len.max(1))
            });

        // Sum the two components, zero-padding shorter to match longer.
        // Zero-padding preserves phase coherence — resampling to a common
        // length would shift phases and cause destructive interference.
        let out_len = harmonic_stretched.len().max(percussive_stretched.len());
        let zeros = std::iter::repeat(0.0f32);
        let output: Vec<f32> = harmonic_stretched
            .iter()
            .copied()
            .chain(zeros.clone())
            .zip(percussive_stretched.iter().copied().chain(zeros))
            .take(out_len)
            .map(|(h, p)| h + p)
            .collect();

        Some(output)
    }

    /// Stretches a tonal segment with per-segment sub-bass band splitting.
    ///
    /// Separates sub-bass from the remainder, PV-stretches each independently
    /// with rigid phase locking for sub-bass, and sums the results. Both outputs
    /// are resampled to the target length before summing.
    fn stretch_tonal_band_split(
        &self,
        seg_data: &[f32],
        seg_ratio: f64,
        pv: &mut PhaseVocoder,
    ) -> Vec<f32> {
        let target_len = (seg_data.len() as f64 * seg_ratio).round().max(1.0) as usize;

        let (sub_bass, remainder) = separate_sub_bass(
            seg_data,
            self.params.sub_bass_cutoff,
            self.params.sample_rate,
        );

        // PV-stretch sub-bass with dedicated PV instance (rigid phase locking)
        let sub_bass_stretched = if sub_bass.len() >= self.params.fft_size {
            let mut sub_pv = PhaseVocoder::with_all_options(
                self.params.fft_size,
                self.params.hop_size,
                seg_ratio,
                self.params.sample_rate,
                self.params.sub_bass_cutoff,
                self.params.window_type,
                self.params.phase_locking_mode,
                self.params.envelope_preservation,
                self.params.envelope_order,
            );
            sub_pv
                .process(&sub_bass)
                .unwrap_or_else(|_| crate::core::resample::resample_linear(&sub_bass, target_len))
        } else {
            crate::core::resample::resample_linear(&sub_bass, target_len)
        };

        // PV-stretch remainder through the shared PV
        let remainder_stretched = if remainder.len() >= self.params.fft_size {
            pv.process(&remainder)
                .unwrap_or_else(|_| crate::core::resample::resample_linear(&remainder, target_len))
        } else {
            crate::core::resample::resample_linear(&remainder, target_len)
        };

        // Sum the two bands, zero-padding shorter to match longer.
        // Zero-padding preserves phase coherence — resampling to a common
        // length would shift phases and cause destructive interference.
        let out_len = sub_bass_stretched.len().max(remainder_stretched.len());
        let zeros = std::iter::repeat(0.0f32);
        sub_bass_stretched
            .iter()
            .copied()
            .chain(zeros.clone())
            .zip(remainder_stretched.iter().copied().chain(zeros))
            .take(out_len)
            .map(|(s, r)| s + r)
            .collect()
    }

    /// Onset-aligned transient stretching with a specific ratio.
    ///
    /// Same as `stretch_transient_segment` but uses the provided ratio instead
    /// of `self.params.stretch_ratio`.
    fn stretch_transient_segment_with_ratio(&self, seg_data: &[f32], ratio: f64) -> Vec<f32> {
        let out_len = (seg_data.len() as f64 * ratio).round() as usize;
        if out_len == 0 {
            return vec![];
        }

        // Attack portion: longer direct copy keeps kick onset and early low-end
        // phase relationship intact before WSOLA handles the decay.
        let attack_samples = ((self.params.sample_rate as f64 * TRANSIENT_ATTACK_COPY_SECS)
            as usize)
            .min(seg_data.len())
            .max(1);
        // Crossfade duration between attack and decay (2ms)
        let crossfade_len = ((self.params.sample_rate as f64 * 0.002) as usize)
            .min(attack_samples / 2)
            .max(1);

        if seg_data.len() <= attack_samples * 2 || out_len <= attack_samples {
            return self
                .stretch_with_wsola_ratio(seg_data, ratio)
                .unwrap_or_else(|_| {
                    crate::core::resample::resample_linear(seg_data, out_len.max(1))
                });
        }

        let attack = &seg_data[..attack_samples];
        let decay = &seg_data[attack_samples..];

        let decay_energy: f32 = decay.iter().map(|&s| s * s).sum();
        let decay_rms = (decay_energy / decay.len().max(1) as f32).sqrt();
        if decay_rms < 1e-4 {
            return self
                .stretch_with_wsola_ratio(seg_data, ratio)
                .unwrap_or_else(|_| {
                    crate::core::resample::resample_linear(seg_data, out_len.max(1))
                });
        }

        let decay_out_len = out_len
            .saturating_sub(attack_samples)
            .saturating_add(crossfade_len);
        if decay_out_len < MIN_WSOLA_SEGMENT {
            let decay_stretched =
                crate::core::resample::resample_linear(decay, decay_out_len.max(1));
            let mut output = Vec::with_capacity(attack_samples + decay_stretched.len());
            output.extend_from_slice(attack);
            output.extend_from_slice(&decay_stretched);
            return output;
        }

        let decay_stretched = {
            let seg_size = self
                .params
                .wsola_segment_size
                .min(decay.len() / 2)
                .max(MIN_WSOLA_SEGMENT);
            let boosted_search = ((self.params.effective_wsola_search_range() as f64)
                * TRANSIENT_DECAY_SEARCH_BOOST)
                .round() as usize;
            let transient_search_floor =
                (self.params.sample_rate as f64 * TRANSIENT_DECAY_SEARCH_FLOOR_SECS) as usize;
            let search = boosted_search
                .max(transient_search_floor)
                .max(MIN_WSOLA_SEARCH)
                .min(seg_size.saturating_sub(1));
            let mut wsola = Wsola::new(seg_size, search, decay_out_len as f64 / decay.len() as f64);
            wsola.process(decay).unwrap_or_else(|_| {
                crate::core::resample::resample_linear(decay, decay_out_len.max(1))
            })
        };

        let crossfade_len = crossfade_len.min(attack_samples).min(decay_stretched.len());

        if crossfade_len == 0 || decay_stretched.is_empty() {
            let mut output = Vec::with_capacity(attack_samples + decay_stretched.len());
            output.extend_from_slice(attack);
            output.extend_from_slice(&decay_stretched);
            return output;
        }

        let pre_fade = attack_samples - crossfade_len;
        let mut output = Vec::with_capacity(out_len);
        output.extend_from_slice(&attack[..pre_fade]);

        for i in 0..crossfade_len {
            let t = i as f32 / crossfade_len as f32;
            let fade_out = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
            let fade_in = 1.0 - fade_out;
            output.push(attack[pre_fade + i] * fade_out + decay_stretched[i] * fade_in);
        }

        if crossfade_len < decay_stretched.len() {
            output.extend_from_slice(&decay_stretched[crossfade_len..]);
        }

        output
    }

    /// Stretches a segment using WSOLA with a specific ratio.
    fn stretch_with_wsola_ratio(
        &self,
        seg_data: &[f32],
        ratio: f64,
    ) -> Result<Vec<f32>, StretchError> {
        let seg_size = self
            .params
            .wsola_segment_size
            .min(seg_data.len() / 2)
            .max(MIN_WSOLA_SEGMENT);
        let search = self
            .params
            .effective_wsola_search_range()
            .min(seg_size / 2)
            .max(MIN_WSOLA_SEARCH);
        let mut wsola = Wsola::new(seg_size, search, ratio);
        wsola.process(seg_data)
    }

    /// Segments audio into transient and tonal regions.
    ///
    /// Uses adaptive transient region sizing based on onset strengths:
    /// strong transients (kicks) get the full `transient_region_secs`,
    /// weak transients (hi-hats) get a smaller region (~5ms minimum).
    fn segment_audio(&self, input_len: usize, onsets: &[usize], strengths: &[f32]) -> Vec<Segment> {
        if onsets.is_empty() {
            return vec![Segment {
                start: 0,
                end: input_len,
                is_transient: false,
                stretch_ratio: self.params.stretch_ratio,
            }];
        }

        let global_ratio = self.params.stretch_ratio;
        // For stretches >1.0, give transients proportionally more input context
        // so onset/early-decay structure survives longer output durations.
        let transient_ratio_scale = global_ratio.max(1.0).min(1.6);
        let max_transient_size = ((self.params.sample_rate as f64
            * self.params.transient_region_secs
            * transient_ratio_scale)
            .round()) as usize;
        // Minimum 5ms region for weak transients
        let min_transient_size = (self.params.sample_rate as f64 * 0.005) as usize;

        let mut segments = Vec::new();
        let mut pos = 0;

        for (i, &onset) in onsets.iter().enumerate() {
            if onset <= pos {
                continue;
            }

            // Tonal region before this boundary
            let tonal_end = onset.min(input_len);
            if tonal_end > pos {
                segments.push(Segment {
                    start: pos,
                    end: tonal_end,
                    is_transient: false,
                    stretch_ratio: global_ratio,
                });
            }

            let strength = strengths.get(i).copied().unwrap_or(1.0);
            let is_transient_anchor = strength_marks_transient(strength);
            if !is_transient_anchor {
                // Beat-only anchor: create a tonal boundary without a transient segment.
                pos = tonal_end;
                continue;
            }

            // Adaptive transient region: scale by onset strength
            // region = min + (max - min) * (0.3 + 0.7 * strength)
            let scale = 0.3 + 0.7 * strength as f64;
            let transient_size = min_transient_size
                + ((max_transient_size - min_transient_size) as f64 * scale) as usize;

            let trans_end = (onset + transient_size).min(input_len);
            if trans_end > onset {
                segments.push(Segment {
                    start: onset,
                    end: trans_end,
                    is_transient: true,
                    stretch_ratio: global_ratio,
                });
            }

            pos = trans_end;
        }

        // Remaining tonal region
        if pos < input_len {
            segments.push(Segment {
                start: pos,
                end: input_len,
                is_transient: false,
                stretch_ratio: global_ratio,
            });
        }

        segments
    }
}

/// Computes target output lengths for each segment before crossfade compensation.
fn compute_base_segment_target_lengths(segments: &[Segment]) -> Vec<usize> {
    segments
        .iter()
        .map(|seg| ((seg.end - seg.start) as f64 * seg.stretch_ratio).round() as usize)
        .collect()
}

/// Computes fixed crossfade length (in samples), clamped for safety.
fn compute_fixed_crossfade_len(
    sample_rate: u32,
    crossfade_secs: f64,
    segment_target_lens: &[usize],
) -> usize {
    if segment_target_lens.len() <= 1 {
        return 0;
    }

    let mut crossfade_samples = (sample_rate as f64 * crossfade_secs) as usize;
    // Ensure crossfade spans at least 2 cycles at low frequency to avoid pops.
    let min_crossfade_samples = (2.0 * sample_rate as f64 / CROSSFADE_MIN_FREQ_HZ_TONAL) as usize;
    crossfade_samples = crossfade_samples.max(min_crossfade_samples);

    // Cap at 25% of the shortest segment output length.
    let shortest = segment_target_lens.iter().copied().min().unwrap_or(0);
    let max_crossfade = shortest / 4;
    crossfade_samples.min(max_crossfade)
}

/// Compensates segment targets so crossfade overlap does not shorten total output.
///
/// For each boundary `i` (between segment `i` and `i+1`), the overlap is added to
/// segment `i+1`. During concatenation, that overlap is subtracted once, restoring
/// the original sum of base segment lengths.
fn compensate_segment_targets_for_crossfades(
    base_segment_target_lens: &[usize],
    crossfade_lens: &[usize],
) -> Vec<usize> {
    if base_segment_target_lens.is_empty() {
        return Vec::new();
    }

    let mut compensated = base_segment_target_lens.to_vec();
    for (boundary_idx, &overlap) in crossfade_lens.iter().enumerate() {
        if let Some(len) = compensated.get_mut(boundary_idx + 1) {
            *len = len.saturating_add(overlap);
        }
    }
    compensated
}

/// Reconciles segment target lengths so their total matches `desired_total`.
fn reconcile_total_segment_targets(segment_target_lens: &mut [usize], desired_total: usize) {
    if segment_target_lens.is_empty() {
        return;
    }

    let current_total: usize = segment_target_lens.iter().sum();
    if current_total == desired_total {
        return;
    }

    if current_total < desired_total {
        let add = desired_total - current_total;
        if let Some(last) = segment_target_lens.last_mut() {
            *last = last.saturating_add(add);
        }
        return;
    }

    let mut remove = current_total - desired_total;
    for len in segment_target_lens.iter_mut().rev() {
        if remove == 0 {
            break;
        }
        let take = (*len).min(remove);
        *len -= take;
        remove -= take;
    }
}

/// Forces a segment to an exact length with minimal correction artifacts.
fn force_segment_length(mut segment: Vec<f32>, target_len: usize) -> Vec<f32> {
    if segment.len() == target_len {
        return segment;
    }
    if target_len == 0 {
        return Vec::new();
    }
    if segment.is_empty() {
        return vec![0.0; target_len];
    }

    let _frame_err = segment.len().abs_diff(target_len);
    // Length correction intentionally avoids re-resampling the already-stretched
    // segment, which would shift local spectral content.
    if segment.len() > target_len {
        segment.truncate(target_len);
    } else {
        let pad = *segment.last().unwrap_or(&0.0);
        segment.resize(target_len, pad);
    }
    segment
}

/// Enforces exact output length for end-to-end tempo fidelity.
fn enforce_exact_output_length(output: Vec<f32>, target_len: usize) -> Vec<f32> {
    force_segment_length(output, target_len)
}

/// Adaptive crossfade durations in seconds, by segment transition type.
/// Tonal→Transient: moderate transition to preserve onset while avoiding pops.
const CROSSFADE_TONAL_TO_TRANSIENT_SECS: f64 = 0.011;
/// Transient→Tonal: short crossfade to preserve attack crispness.
const CROSSFADE_TRANSIENT_TO_TONAL_SECS: f64 = 0.009;
/// Tonal→Tonal: longer crossfade for smooth blending.
const CROSSFADE_TONAL_TO_TONAL_SECS: f64 = 0.017;
/// Transient→Transient: minimal blending, but enough to avoid pops.
const CROSSFADE_TRANSIENT_TO_TRANSIENT_SECS: f64 = 0.005;
/// Lowest frequency (Hz) for fixed/tonal crossfade minimum duration.
const CROSSFADE_MIN_FREQ_HZ_TONAL: f64 = 60.0;
/// Lowest frequency (Hz) for transient-boundary crossfades.
///
/// Using a higher floor frequency shortens transient boundary overlaps so kick
/// attacks are not smeared by long blends.
const CROSSFADE_MIN_FREQ_HZ_TRANSIENT: f64 = 180.0;

/// Computes per-boundary crossfade lengths based on segment transitions.
///
/// Returns a vector of crossfade lengths in samples, one per boundary
/// (length = segments.len() - 1).
///
/// Each crossfade is at least 2 cycles of a transition-dependent minimum
/// frequency (shorter at transient boundaries, longer for tonal boundaries),
/// and at most 25% of the shorter adjacent segment's output length.
fn compute_adaptive_crossfade_lens(segments: &[Segment], sample_rate: u32) -> Vec<usize> {
    if segments.len() <= 1 {
        return vec![];
    }

    let mut lens = Vec::with_capacity(segments.len() - 1);
    for i in 1..segments.len() {
        let prev = &segments[i - 1];
        let cur = &segments[i];

        let secs = match (prev.is_transient, cur.is_transient) {
            (false, true) => CROSSFADE_TONAL_TO_TRANSIENT_SECS,
            (true, false) => CROSSFADE_TRANSIENT_TO_TONAL_SECS,
            (false, false) => CROSSFADE_TONAL_TO_TONAL_SECS,
            (true, true) => CROSSFADE_TRANSIENT_TO_TRANSIENT_SECS,
        };
        let mut crossfade_samples = (sample_rate as f64 * secs) as usize;

        // Enforce transition-dependent minimum:
        // transient boundaries can use shorter overlaps than tonal boundaries.
        let min_freq_hz = if prev.is_transient || cur.is_transient {
            CROSSFADE_MIN_FREQ_HZ_TRANSIENT
        } else {
            CROSSFADE_MIN_FREQ_HZ_TONAL
        };
        let min_crossfade_samples = (2.0 * sample_rate as f64 / min_freq_hz) as usize;
        crossfade_samples = crossfade_samples.max(min_crossfade_samples);

        // Cap at 25% of the shorter adjacent segment's output length
        let prev_out_len = ((prev.end - prev.start) as f64 * prev.stretch_ratio).round() as usize;
        let cur_out_len = ((cur.end - cur.start) as f64 * cur.stretch_ratio).round() as usize;
        let shortest_segment_len = prev_out_len.min(cur_out_len);
        let max_crossfade = shortest_segment_len / 4;
        crossfade_samples = crossfade_samples.min(max_crossfade);

        lens.push(crossfade_samples);
    }

    lens
}

/// Concatenates segments with per-boundary crossfade lengths and reports
/// the actual overlap used at each boundary.
fn concatenate_with_adaptive_crossfade_report(
    segments: &[Vec<f32>],
    crossfade_lens: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    concatenate_with_boundary_crossfades(segments, crossfade_lens)
}

/// Concatenates segments using one overlap value per boundary.
///
/// Returns `(output, actual_overlaps)`, where `actual_overlaps[i]` is the
/// overlap length used between segment `i` and `i+1` after runtime clamping.
fn concatenate_with_boundary_crossfades(
    segments: &[Vec<f32>],
    crossfade_lens: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    match segments.len() {
        0 => return (vec![], vec![]),
        1 => return (segments[0].clone(), vec![]),
        _ => {}
    }

    let total: usize = segments.iter().map(|s| s.len()).sum();
    let overlap_total: usize = crossfade_lens.iter().sum();
    let mut output = Vec::with_capacity(total.saturating_sub(overlap_total));
    let mut actual_overlaps = Vec::with_capacity(segments.len().saturating_sub(1));

    for (idx, segment) in segments.iter().enumerate() {
        if idx == 0 {
            output.extend_from_slice(segment);
            continue;
        }

        let requested = crossfade_lens.get(idx - 1).copied().unwrap_or(0);
        let fade_len = requested.min(output.len()).min(segment.len());
        actual_overlaps.push(fade_len);
        let output_start = output.len() - fade_len;

        // Crossfade overlap region with raised cosine
        for i in 0..fade_len {
            let t = i as f32 / fade_len as f32;
            let fade_out = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
            let fade_in = 1.0 - fade_out;
            output[output_start + i] = output[output_start + i] * fade_out + segment[i] * fade_in;
        }

        // Append non-overlapping part
        if fade_len < segment.len() {
            output.extend_from_slice(&segment[fade_len..]);
        }
    }

    (output, actual_overlaps)
}

/// Threshold for considering a band's flux significant enough to warrant phase reset.
const BAND_FLUX_RESET_THRESHOLD: f32 = 0.1;

/// Computes a per-band phase reset mask for a transient segment.
///
/// Looks up the per-frame band flux at the onset position to determine which
/// frequency bands had significant transient energy. Returns `[true; 4]` if
/// no band flux data is available (fallback to full reset).
fn compute_band_reset_mask(
    segment_start: usize,
    transients: &crate::analysis::transient::TransientMap,
) -> [bool; 4] {
    if transients.per_frame_band_flux.is_empty() || transients.hop_size == 0 {
        return [true; 4]; // No band data — full reset
    }

    let frame_idx = segment_start / transients.hop_size;
    if frame_idx >= transients.per_frame_band_flux.len() {
        return [true; 4];
    }

    let band_flux = transients.per_frame_band_flux[frame_idx];

    // Normalize by the max band flux to get relative energy
    let max_flux = band_flux.iter().copied().fold(0.0f32, f32::max);
    if max_flux < 1e-10 {
        return [true; 4]; // Near-silent — full reset is safe
    }

    [
        band_flux[0] / max_flux > BAND_FLUX_RESET_THRESHOLD,
        band_flux[1] / max_flux > BAND_FLUX_RESET_THRESHOLD,
        band_flux[2] / max_flux > BAND_FLUX_RESET_THRESHOLD,
        band_flux[3] / max_flux > BAND_FLUX_RESET_THRESHOLD,
    ]
}

/// Minimum per-segment stretch ratio for elastic distribution.
const ELASTIC_MIN_RATIO: f64 = 0.5;
/// Maximum per-segment stretch ratio for elastic distribution.
const ELASTIC_MAX_RATIO: f64 = 4.0;

/// Computes per-segment stretch ratios for elastic beat distribution.
///
/// Blends transient segment ratios between the global ratio and 1.0 based on
/// the `anchor` parameter:
/// - `anchor = 0.0`: transients get the global ratio (beats at target tempo)
/// - `anchor = 1.0`: transients stay at ratio 1.0 (beats at original tempo)
///
/// Tonal segments absorb the remaining stretch so the total output duration
/// matches what the global ratio would produce.
///
/// If no tonal segments exist, all segments keep the global ratio.
fn compute_elastic_ratios(segments: &mut [Segment], global_ratio: f64, anchor: f64) {
    if segments.is_empty() {
        return;
    }

    let total_input: f64 = segments.iter().map(|s| (s.end - s.start) as f64).sum();
    if total_input < 1.0 {
        return;
    }

    let total_target_output = total_input * global_ratio;

    // Blend between global ratio (anchor=0) and identity (anchor=1).
    // anchor=0.0 → transients at target tempo (DJ beatmatch)
    // anchor=1.0 → transients at original tempo (creative effects)
    let anchor = anchor.clamp(0.0, 1.0);
    let transient_ratio = global_ratio * (1.0 - anchor) + 1.0 * anchor;

    let transient_input: f64 = segments
        .iter()
        .filter(|s| s.is_transient)
        .map(|s| (s.end - s.start) as f64)
        .sum();
    let tonal_input: f64 = segments
        .iter()
        .filter(|s| !s.is_transient)
        .map(|s| (s.end - s.start) as f64)
        .sum();

    if tonal_input < 1.0 {
        // All transient — no tonal segments to absorb slack; keep global ratio
        return;
    }

    // Output consumed by transient segments at their ratio
    let transient_output = transient_input * transient_ratio;

    // Remaining output for tonal segments
    let tonal_output = total_target_output - transient_output;
    let tonal_ratio = (tonal_output / tonal_input).clamp(ELASTIC_MIN_RATIO, ELASTIC_MAX_RATIO);

    // If the clamped tonal ratio can't absorb all the slack, redistribute
    // back to transients proportionally.
    let actual_output = transient_input * transient_ratio + tonal_input * tonal_ratio;
    let correction = if actual_output > 1.0 {
        total_target_output / actual_output
    } else {
        1.0
    };

    for segment in segments.iter_mut() {
        if segment.is_transient {
            segment.stretch_ratio =
                (transient_ratio * correction).clamp(ELASTIC_MIN_RATIO, ELASTIC_MAX_RATIO);
        } else {
            segment.stretch_ratio =
                (tonal_ratio * correction).clamp(ELASTIC_MIN_RATIO, ELASTIC_MAX_RATIO);
        }
    }
}

/// Separates sub-bass from the remainder of the signal using FFT-based filtering.
///
/// Uses overlap-add with a Hann window to split the input into two bands:
/// - Sub-bass: everything below `cutoff_hz`
/// - Remainder: everything at or above `cutoff_hz`
///
/// Both outputs have the same length as the input.
fn separate_sub_bass(input: &[f32], cutoff_hz: f32, sample_rate: u32) -> (Vec<f32>, Vec<f32>) {
    let fft_size = BAND_SPLIT_FFT_SIZE;
    let hop = BAND_SPLIT_HOP;
    let cutoff_bin = freq_to_bin(cutoff_hz, fft_size, sample_rate);

    if cutoff_bin == 0 || input.len() < fft_size {
        return (vec![0.0; input.len()], input.to_vec());
    }

    let window = generate_window(WindowType::Hann, fft_size);
    let mut planner = FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(fft_size);
    let fft_inv = planner.plan_fft_inverse(fft_size);
    let norm = 1.0 / fft_size as f32;

    let mut sub_bass = vec![0.0f32; input.len()];
    let mut remainder = vec![0.0f32; input.len()];
    let mut window_sum = vec![0.0f32; input.len()];

    let num_frames = if input.len() <= fft_size {
        1
    } else {
        (input.len() - fft_size) / hop + 1
    };

    let mut fft_buf = vec![COMPLEX_ZERO; fft_size];
    let mut fft_buf2 = vec![COMPLEX_ZERO; fft_size];

    for frame in 0..num_frames {
        let pos = frame * hop;
        let frame_end = (pos + fft_size).min(input.len());
        let frame_len = frame_end - pos;

        window_and_transform(&input[pos..frame_end], &window, &mut fft_buf, &fft_fwd);
        split_bands(&mut fft_buf, &mut fft_buf2, fft_size, cutoff_bin);
        fft_inv.process(&mut fft_buf);
        fft_inv.process(&mut fft_buf2);

        // Overlap-add with synthesis window
        for i in 0..frame_len {
            let out_idx = pos + i;
            sub_bass[out_idx] += fft_buf[i].re * norm * window[i];
            remainder[out_idx] += fft_buf2[i].re * norm * window[i];
            window_sum[out_idx] += window[i] * window[i];
        }
    }

    normalize_band_split(&mut sub_bass, &mut remainder, &window_sum);
    (sub_bass, remainder)
}

/// Windows an input frame into the FFT buffer and transforms to frequency domain.
fn window_and_transform(
    input_frame: &[f32],
    window: &[f32],
    fft_buf: &mut [Complex<f32>],
    fft_fwd: &std::sync::Arc<dyn rustfft::Fft<f32>>,
) {
    let windowed = input_frame
        .iter()
        .zip(window.iter())
        .map(|(&s, &w)| Complex::new(s * w, 0.0));
    for (slot, val) in fft_buf
        .iter_mut()
        .zip(windowed.chain(std::iter::repeat(COMPLEX_ZERO)))
    {
        *slot = val;
    }
    fft_fwd.process(fft_buf);
}

/// Width of the raised-cosine crossover transition band in bins.
const CROSSOVER_TRANSITION_BINS: usize = 5;

/// Splits an FFT spectrum into sub-bass and remainder bands.
///
/// Uses a raised-cosine transition band around `cutoff_bin` to avoid
/// ringing artifacts from a brick-wall filter. The transition spans
/// `CROSSOVER_TRANSITION_BINS` bins on each side of the cutoff.
///
/// `fft_buf` is narrowed to sub-bass only (bins below cutoff).
/// `fft_buf2` receives the remainder (bins at or above cutoff).
/// Both respect conjugate symmetry for real-valued signals.
fn split_bands(
    fft_buf: &mut [Complex<f32>],
    fft_buf2: &mut [Complex<f32>],
    fft_size: usize,
    cutoff_bin: usize,
) {
    fft_buf2.copy_from_slice(fft_buf);
    let half = fft_size / 2;
    let width = CROSSOVER_TRANSITION_BINS;
    let trans_start = cutoff_bin.saturating_sub(width);
    let trans_end = (cutoff_bin + width).min(half);

    for bin in 0..=half {
        let sub_gain = if bin <= trans_start {
            1.0f32
        } else if bin >= trans_end {
            0.0
        } else {
            // Raised-cosine taper from 1 -> 0 across transition band
            let t = (bin - trans_start) as f32 / (trans_end - trans_start) as f32;
            0.5 * (1.0 + (std::f32::consts::PI * t).cos())
        };
        let rem_gain = 1.0 - sub_gain;

        // Apply gains to positive-frequency bin
        fft_buf[bin] *= sub_gain;
        fft_buf2[bin] *= rem_gain;

        // Mirror for negative-frequency bin (conjugate symmetry)
        if bin > 0 && bin < half {
            fft_buf[fft_size - bin] *= sub_gain;
            fft_buf2[fft_size - bin] *= rem_gain;
        }
    }
}

/// Normalizes two band-split output buffers by the accumulated window sum.
fn normalize_band_split(sub_bass: &mut [f32], remainder: &mut [f32], window_sum: &[f32]) {
    let max_ws = window_sum.iter().copied().fold(0.0f32, f32::max);
    let min_ws = (max_ws * WINDOW_SUM_FLOOR_RATIO).max(WINDOW_SUM_EPSILON);
    for ((&ws, sb), rem) in window_sum
        .iter()
        .zip(sub_bass.iter_mut())
        .zip(remainder.iter_mut())
    {
        let ws = ws.max(min_ws);
        *sb /= ws;
        *rem /= ws;
    }
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

    // Add beat positions that aren't too close to existing anchors.
    for &beat in beats {
        if beat >= input_len {
            continue;
        }
        let too_close = merged_positions.iter().any(|&pos| {
            let dist = pos.abs_diff(beat);
            dist < DEDUP_DISTANCE
        });
        if !too_close {
            merged_positions.push(beat);
            merged_strengths.push(BEAT_ANCHOR_STRENGTH);
        }
    }

    let mut pairs: Vec<(usize, f32)> = merged_positions.into_iter().zip(merged_strengths).collect();
    pairs.sort_unstable_by_key(|(pos, _)| *pos);

    // Collapse exact duplicate positions while preserving transient priority.
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
fn generate_subdivision_grid_with_phase(
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
    let mut grid = Vec::new();
    let mut pos = phase;
    while pos < total_samples as f64 {
        grid.push(pos);
        pos += sub_interval;
    }
    grid
}

/// Concatenates segments with raised-cosine crossfade.
#[cfg(test)]
fn concatenate_with_crossfade(segments: &[Vec<f32>], crossfade_len: usize) -> Vec<f32> {
    let crossfade_lens = vec![crossfade_len; segments.len().saturating_sub(1)];
    concatenate_with_boundary_crossfades(segments, &crossfade_lens).0
}

/// Concatenates segments with provided crossfade lengths and reports boundary usage.
fn concatenate_with_crossfade_report(
    segments: &[Vec<f32>],
    crossfade_lens: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    concatenate_with_boundary_crossfades(segments, crossfade_lens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::EdmPreset;
    use std::f32::consts::PI;

    #[test]
    fn test_hybrid_stretcher_sine() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let stretcher = HybridStretcher::new(params);
        let output = stretcher.process(&input).unwrap();

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.5).abs() < 0.3,
            "Length ratio {} too far from 1.5",
            len_ratio
        );
    }

    #[test]
    fn test_hybrid_stretcher_with_transients() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let mut input = vec![0.0f32; num_samples];

        // Add clicks every 0.5 seconds
        for beat in 0..4 {
            let pos = (beat as f64 * 0.5 * sample_rate as f64) as usize;
            for j in 0..20.min(num_samples - pos) {
                input[pos + j] = if j < 5 { 0.8 } else { -0.3 };
            }
        }

        // Add some tonal content
        for (i, sample) in input.iter_mut().enumerate().take(num_samples) {
            *sample += 0.3 * (2.0 * PI * 200.0 * i as f32 / sample_rate as f32).sin();
        }

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(EdmPreset::HouseLoop);

        let stretcher = HybridStretcher::new(params);
        let output = stretcher.process(&input).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_hybrid_stretcher_empty() {
        let params = StretchParams::new(1.5);
        let stretcher = HybridStretcher::new(params);
        let output = stretcher.process(&[]).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_concatenate_crossfade() {
        let a = vec![1.0; 100];
        let b = vec![0.5; 100];
        let result = concatenate_with_crossfade(&[a, b], 20);
        // Total should be about 180 (200 - 20 overlap)
        assert!((result.len() as i64 - 180).unsigned_abs() < 5);
        // Middle of crossfade should be between 0.5 and 1.0
        let mid = result[90];
        assert!((0.4..=1.1).contains(&mid), "Crossfade mid = {}", mid);
    }

    #[test]
    fn test_merge_onsets_and_beats_empty() {
        let (onsets, strengths) = merge_onsets_and_beats(&[], &[], &[], 44100);
        assert!(onsets.is_empty());
        assert!(strengths.is_empty());
    }

    #[test]
    fn test_merge_onsets_and_beats_no_overlap() {
        let onsets = vec![1000, 5000];
        let strengths = vec![0.8, 0.6];
        let beats = vec![10000, 20000];
        let (merged_onsets, merged_strengths) =
            merge_onsets_and_beats(&onsets, &strengths, &beats, 44100);
        assert_eq!(merged_onsets, vec![1000, 5000, 10000, 20000]);
        assert_eq!(merged_strengths[0], 0.8);
        assert_eq!(merged_strengths[1], 0.6);
        assert!(!strength_marks_transient(merged_strengths[2]));
        assert!(!strength_marks_transient(merged_strengths[3]));
    }

    #[test]
    fn test_merge_onsets_and_beats_dedup_close() {
        // Beat at 1100 is within 512 samples of onset at 1000 — should be deduped
        let onsets = vec![1000, 5000];
        let strengths = vec![0.7, 0.9];
        let beats = vec![1100, 20000];
        let (merged_onsets, merged_strengths) =
            merge_onsets_and_beats(&onsets, &strengths, &beats, 44100);
        assert_eq!(merged_onsets, vec![1000, 5000, 20000]);
        assert_eq!(merged_strengths[0], 0.7);
        assert_eq!(merged_strengths[1], 0.9);
        assert!(!strength_marks_transient(merged_strengths[2]));
    }

    #[test]
    fn test_merge_onsets_and_beats_out_of_bounds() {
        // Beat at 50000 exceeds input_len of 44100 — should be dropped
        let onsets = vec![1000];
        let strengths = vec![0.5];
        let beats = vec![50000];
        let (merged_onsets, merged_strengths) =
            merge_onsets_and_beats(&onsets, &strengths, &beats, 44100);
        assert_eq!(merged_onsets, vec![1000]);
        assert_eq!(merged_strengths, strengths);
    }

    #[test]
    fn test_beat_aware_stretcher_with_kicks() {
        // Generate a 2-second signal with regular kicks (120 BPM = every 0.5s)
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let mut input = vec![0.0f32; num_samples];
        let beat_interval = sample_rate as usize / 2; // 0.5 seconds

        // Add kick-like transients at beat positions
        for beat in 0..4 {
            let pos = beat * beat_interval;
            for j in 0..20.min(num_samples - pos) {
                input[pos + j] = if j < 5 { 0.9 } else { -0.4 };
            }
        }

        // Add tonal content between kicks
        for (i, sample) in input.iter_mut().enumerate() {
            *sample += 0.3 * (2.0 * PI * 200.0 * i as f32 / sample_rate as f32).sin();
        }

        // Beat-aware mode (enabled by preset)
        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(EdmPreset::HouseLoop);
        assert!(params.beat_aware);

        let stretcher = HybridStretcher::new(params);
        let output_aware = stretcher.process(&input).unwrap();

        // Non-beat-aware mode
        let params_no_beat = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_beat_aware(false);

        let stretcher_no_beat = HybridStretcher::new(params_no_beat);
        let output_no_beat = stretcher_no_beat.process(&input).unwrap();

        // Both should produce valid output
        assert!(!output_aware.is_empty());
        assert!(!output_no_beat.is_empty());

        // Both should have reasonable length ratios
        let ratio_aware = output_aware.len() as f64 / input.len() as f64;
        let ratio_no_beat = output_no_beat.len() as f64 / input.len() as f64;
        assert!(
            (ratio_aware - 1.5).abs() < 0.4,
            "Beat-aware ratio {} too far from 1.5",
            ratio_aware
        );
        assert!(
            (ratio_no_beat - 1.5).abs() < 0.4,
            "Non-beat-aware ratio {} too far from 1.5",
            ratio_no_beat
        );
    }

    #[test]
    fn test_beat_aware_disabled_for_short_input() {
        // For input shorter than MIN_SAMPLES_FOR_BEAT_DETECTION, beat detection
        // should be skipped even with beat_aware enabled.
        let sample_rate = 44100u32;
        let num_samples = 20000; // Less than 44100 (1 second threshold)

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_beat_aware(true);

        let stretcher = HybridStretcher::new(params);
        let output = stretcher.process(&input).unwrap();

        // Should still produce valid output without crashing
        assert!(!output.is_empty());
    }

    #[test]
    fn test_beat_aware_flag_default() {
        // Default: beat_aware should be false
        let params = StretchParams::new(1.0);
        assert!(!params.beat_aware);

        // With preset: beat_aware should be true
        let params = StretchParams::new(1.0).with_preset(EdmPreset::DjBeatmatch);
        assert!(params.beat_aware);

        // Can be overridden after preset
        let params = StretchParams::new(1.0)
            .with_preset(EdmPreset::DjBeatmatch)
            .with_beat_aware(false);
        assert!(!params.beat_aware);
    }

    #[test]
    fn test_band_split_flag_default() {
        // Default: band_split should be false
        let params = StretchParams::new(1.0);
        assert!(!params.band_split);

        // With preset: band_split should be true
        let params = StretchParams::new(1.0).with_preset(EdmPreset::HouseLoop);
        assert!(params.band_split);

        // Can be overridden after preset
        let params = StretchParams::new(1.0)
            .with_preset(EdmPreset::HouseLoop)
            .with_band_split(false);
        assert!(!params.band_split);
    }

    #[test]
    fn test_separate_sub_bass_preserves_energy() {
        // A 60 Hz sine (sub-bass) should end up mostly in the sub-bass band
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 60.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let (sub_bass, remainder) = separate_sub_bass(&input, 120.0, sample_rate);
        assert_eq!(sub_bass.len(), input.len());
        assert_eq!(remainder.len(), input.len());

        let sub_rms = (sub_bass.iter().map(|x| x * x).sum::<f32>() / sub_bass.len() as f32).sqrt();
        let rem_rms =
            (remainder.iter().map(|x| x * x).sum::<f32>() / remainder.len() as f32).sqrt();

        // Sub-bass should have most of the energy for a 60 Hz signal
        assert!(
            sub_rms > rem_rms * 2.0,
            "60 Hz signal should be in sub-bass band: sub_rms={}, rem_rms={}",
            sub_rms,
            rem_rms
        );
    }

    #[test]
    fn test_separate_sub_bass_passes_high_freq() {
        // A 1000 Hz sine should end up mostly in the remainder band
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let (sub_bass, remainder) = separate_sub_bass(&input, 120.0, sample_rate);

        let sub_rms = (sub_bass.iter().map(|x| x * x).sum::<f32>() / sub_bass.len() as f32).sqrt();
        let rem_rms =
            (remainder.iter().map(|x| x * x).sum::<f32>() / remainder.len() as f32).sqrt();

        // Remainder should have most of the energy for a 1000 Hz signal
        assert!(
            rem_rms > sub_rms * 2.0,
            "1000 Hz signal should be in remainder band: sub_rms={}, rem_rms={}",
            sub_rms,
            rem_rms
        );
    }

    #[test]
    fn test_separate_sub_bass_reconstruction() {
        // Sub-bass + remainder should approximately reconstruct the original.
        // Hann-window overlap-add filtering introduces some leakage at the
        // crossover, so we use a lenient SNR threshold.
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        // Mix of sub-bass (60 Hz) and mid (1000 Hz)
        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * PI * 60.0 * t).sin() + 0.5 * (2.0 * PI * 1000.0 * t).sin()
            })
            .collect();

        let (sub_bass, remainder) = separate_sub_bass(&input, 120.0, sample_rate);

        // Reconstruct
        let reconstructed: Vec<f32> = sub_bass
            .iter()
            .zip(remainder.iter())
            .map(|(s, r)| s + r)
            .collect();

        // Check RMS of reconstruction vs input (energy preservation)
        let start = BAND_SPLIT_FFT_SIZE;
        let end = input.len() - BAND_SPLIT_FFT_SIZE;
        let input_rms = (input[start..end]
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            / (end - start) as f64)
            .sqrt();
        let recon_rms = (reconstructed[start..end]
            .iter()
            .map(|x| (*x as f64).powi(2))
            .sum::<f64>()
            / (end - start) as f64)
            .sqrt();

        let rms_ratio = recon_rms / input_rms;
        assert!(
            (0.7..=1.5).contains(&rms_ratio),
            "Reconstruction RMS ratio should be near 1.0, got {:.3} (input={:.4}, recon={:.4})",
            rms_ratio,
            input_rms,
            recon_rms
        );
    }

    #[test]
    fn test_band_split_stretch_produces_output() {
        // Band-split stretch should produce valid output for a mixed signal
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * PI * 60.0 * t).sin() + 0.5 * (2.0 * PI * 440.0 * t).sin()
            })
            .collect();

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_band_split(true);

        let stretcher = HybridStretcher::new(params);
        let output = stretcher.process(&input).unwrap();

        assert!(!output.is_empty());
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.5).abs() < 0.4,
            "Band-split stretch ratio {} too far from 1.5",
            len_ratio
        );

        // Check no NaN/Inf in output
        assert!(
            output.iter().all(|s| s.is_finite()),
            "Output must be all finite"
        );
    }

    #[test]
    fn test_band_split_vs_no_band_split_similar_length() {
        // Both modes should produce similar output lengths
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params_split = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_band_split(true);

        let params_no_split = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_band_split(false);

        let stretcher_split = HybridStretcher::new(params_split);
        let stretcher_no_split = HybridStretcher::new(params_no_split);

        let output_split = stretcher_split.process(&input).unwrap();
        let output_no_split = stretcher_no_split.process(&input).unwrap();

        let ratio_split = output_split.len() as f64 / input.len() as f64;
        let ratio_no_split = output_no_split.len() as f64 / input.len() as f64;

        // Both should be within 30% of 1.5
        assert!(
            (ratio_split - 1.5).abs() < 0.4,
            "Band-split ratio {} too far from 1.5",
            ratio_split
        );
        assert!(
            (ratio_no_split - 1.5).abs() < 0.4,
            "Non-split ratio {} too far from 1.5",
            ratio_no_split
        );
    }

    #[test]
    fn test_band_split_compression() {
        // Band-split should also work for compression (ratio < 1.0)
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * PI * 60.0 * t).sin() + 0.5 * (2.0 * PI * 440.0 * t).sin()
            })
            .collect();

        let params = StretchParams::new(0.75)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_band_split(true);

        let stretcher = HybridStretcher::new(params);
        let output = stretcher.process(&input).unwrap();

        assert!(!output.is_empty());
        assert!(output.iter().all(|s| s.is_finite()));
        // Output should be shorter than input
        assert!(
            output.len() < input.len(),
            "Compression should produce shorter output"
        );
    }

    #[test]
    fn test_band_split_with_preset() {
        // Presets with multi_resolution=true use multi-resolution instead of
        // band_split to avoid redundant sub-bass processing paths.
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        for preset in [
            EdmPreset::DjBeatmatch,
            EdmPreset::HouseLoop,
            EdmPreset::Halftime,
            EdmPreset::Ambient,
            EdmPreset::VocalChop,
        ] {
            let params = StretchParams::new(1.5)
                .with_sample_rate(sample_rate)
                .with_channels(1)
                .with_preset(preset);

            // band_split and multi_resolution are mutually exclusive
            assert!(
                params.band_split || params.multi_resolution,
                "Preset {:?} should enable band_split or multi_resolution",
                preset
            );
            assert!(
                !(params.band_split && params.multi_resolution),
                "Preset {:?} should not enable both band_split and multi_resolution",
                preset
            );

            let stretcher = HybridStretcher::new(params);
            let output = stretcher.process(&input).unwrap();
            assert!(
                !output.is_empty(),
                "Preset {:?} produced empty output",
                preset
            );
            assert!(
                output.iter().all(|s| s.is_finite()),
                "Preset {:?} produced NaN/Inf",
                preset
            );
        }
    }

    #[test]
    fn test_band_split_short_input_fallback() {
        // Input shorter than BAND_SPLIT_FFT_SIZE should skip band splitting
        let sample_rate = 44100u32;
        let num_samples = BAND_SPLIT_FFT_SIZE - 100;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_band_split(true);

        let stretcher = HybridStretcher::new(params);
        let output = stretcher.process(&input).unwrap();

        // Should still produce valid output via hybrid path
        assert!(!output.is_empty());
    }

    #[test]
    fn test_separate_sub_bass_zero_cutoff() {
        // With 0 Hz cutoff, all energy should go to remainder
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 60.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let (sub_bass, remainder) = separate_sub_bass(&input, 0.0, sample_rate);

        let sub_rms = (sub_bass.iter().map(|x| x * x).sum::<f32>() / sub_bass.len() as f32).sqrt();
        assert!(
            sub_rms < 0.01,
            "Zero cutoff should produce no sub-bass, got RMS={}",
            sub_rms
        );

        let rem_rms =
            (remainder.iter().map(|x| x * x).sum::<f32>() / remainder.len() as f32).sqrt();
        assert!(
            rem_rms > 0.1,
            "Remainder should have the energy, got RMS={}",
            rem_rms
        );
    }

    // --- segment_audio internals ---

    #[test]
    fn test_segment_audio_no_onsets() {
        let params = StretchParams::new(1.5).with_sample_rate(44100);
        let stretcher = HybridStretcher::new(params);
        let segments = stretcher.segment_audio(44100, &[], &[]);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 0);
        assert_eq!(segments[0].end, 44100);
        assert!(!segments[0].is_transient);
    }

    #[test]
    fn test_segment_audio_single_onset() {
        let params = StretchParams::new(1.5).with_sample_rate(44100);
        let stretcher = HybridStretcher::new(params);
        let input_len = 44100;
        let onset = 10000;
        let segments = stretcher.segment_audio(input_len, &[onset], &[]);

        // Should have: tonal [0, onset), transient [onset, onset+transient_size), tonal [onset+transient_size, end)
        assert!(segments.len() >= 2, "Should have at least 2 segments");
        assert_eq!(segments[0].start, 0);
        assert_eq!(segments[0].end, onset);
        assert!(!segments[0].is_transient);

        assert_eq!(segments[1].start, onset);
        assert!(segments[1].is_transient);
    }

    #[test]
    fn test_segment_audio_onset_at_zero() {
        // Onset at position 0: onset <= pos (0 <= 0) so it's skipped.
        // Entire input becomes a single tonal segment.
        let params = StretchParams::new(1.5).with_sample_rate(44100);
        let stretcher = HybridStretcher::new(params);
        let segments = stretcher.segment_audio(44100, &[0], &[]);

        assert_eq!(segments.len(), 1);
        assert!(!segments[0].is_transient);
        assert_eq!(segments[0].start, 0);
        assert_eq!(segments[0].end, 44100);
    }

    #[test]
    fn test_segment_audio_onset_near_end() {
        // Onset near end of input: transient region clamped to input_len
        let params = StretchParams::new(1.5).with_sample_rate(44100);
        let stretcher = HybridStretcher::new(params);
        let input_len = 44100;
        let onset = 44090; // Very near end
        let segments = stretcher.segment_audio(input_len, &[onset], &[]);

        // Last transient segment should be clamped to input_len
        let last_transient = segments.iter().find(|s| s.is_transient).unwrap();
        assert!(last_transient.end <= input_len);
    }

    #[test]
    fn test_segment_audio_overlapping_onsets() {
        // Two onsets where second falls within transient region of first
        let params = StretchParams::new(1.5).with_sample_rate(44100);
        let stretcher = HybridStretcher::new(params);
        let transient_size = (44100.0 * 0.010) as usize; // ~441 (default transient region)
        let onset1 = 10000;
        let onset2 = onset1 + transient_size / 2; // Within transient region of onset1

        let segments = stretcher.segment_audio(44100, &[onset1, onset2], &[]);

        // Second onset should be skipped (onset2 <= pos after first transient)
        let transient_count = segments.iter().filter(|s| s.is_transient).count();
        // Should have 1 or 2 transient segments depending on overlap
        assert!(
            transient_count >= 1,
            "Should have at least 1 transient segment"
        );
    }

    #[test]
    fn test_segment_audio_adaptive_strength() {
        // Weak strength should produce smaller transient region than strong
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_transient_region_secs(0.030); // 30ms max
        let stretcher = HybridStretcher::new(params);

        let weak_segs = stretcher.segment_audio(44100, &[10000], &[0.1]);
        let strong_segs = stretcher.segment_audio(44100, &[10000], &[1.0]);

        let weak_trans = weak_segs.iter().find(|s| s.is_transient).unwrap();
        let strong_trans = strong_segs.iter().find(|s| s.is_transient).unwrap();

        let weak_size = weak_trans.end - weak_trans.start;
        let strong_size = strong_trans.end - strong_trans.start;

        assert!(
            strong_size > weak_size,
            "Strong transient region ({}) should be larger than weak ({})",
            strong_size,
            weak_size
        );
    }

    #[test]
    fn test_segment_audio_beat_only_anchor_is_tonal_boundary() {
        let params = StretchParams::new(1.2).with_sample_rate(44100);
        let stretcher = HybridStretcher::new(params);
        let input_len = 44100;
        let onsets = vec![10000, 20000];
        let strengths = vec![BEAT_ANCHOR_STRENGTH, 0.9];

        let segments = stretcher.segment_audio(input_len, &onsets, &strengths);
        assert!(
            segments.len() >= 3,
            "Expected tonal split + transient region, got {segments:?}"
        );
        assert_eq!(segments[0].start, 0);
        assert_eq!(segments[0].end, 10000);
        assert!(!segments[0].is_transient);
        assert_eq!(segments[1].start, 10000);
        assert_eq!(segments[1].end, 20000);
        assert!(!segments[1].is_transient);
        assert_eq!(segments[2].start, 20000);
        assert!(segments[2].is_transient);
    }

    // --- concatenate_with_crossfade edge cases ---

    #[test]
    fn test_crossfade_empty_segments() {
        let result = concatenate_with_crossfade(&[], 20);
        assert!(result.is_empty());
    }

    #[test]
    fn test_crossfade_single_segment() {
        let seg = vec![1.0, 2.0, 3.0];
        let result = concatenate_with_crossfade(std::slice::from_ref(&seg), 20);
        assert_eq!(result, seg);
    }

    #[test]
    fn test_crossfade_zero_length() {
        // crossfade_len=0 → no overlap, just concatenation
        let a = vec![1.0; 10];
        let b = vec![2.0; 10];
        let result = concatenate_with_crossfade(&[a, b], 0);
        assert_eq!(result.len(), 20);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[10] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_larger_than_segment() {
        // crossfade_len > segment length → clamped to min of output.len() and segment.len()
        let a = vec![1.0; 5];
        let b = vec![0.5; 5];
        let result = concatenate_with_crossfade(&[a, b], 100);
        // Crossfade len clamped to min(5, 5) = 5
        // Total length = 5 (output from a) - 5 (overlap) + 5 (b) = 5
        assert!(
            result.len() >= 5,
            "Output should have at least 5 samples, got {}",
            result.len()
        );
        assert!(result.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_crossfade_raised_cosine_midpoint() {
        // At midpoint (t=0.5), raised cosine should be ~0.5 for both fade_in and fade_out
        let a = vec![1.0; 100];
        let b = vec![0.0; 100];
        let crossfade_len = 50;
        let result = concatenate_with_crossfade(&[a, b], crossfade_len);

        // Crossfade starts at output[100-50]=output[50]
        // At midpoint i=25: t=0.5, fade_out = 0.5*(1+cos(0.5*PI)) ≈ 0.5, fade_in ≈ 0.5
        // output = 1.0 * 0.5 + 0.0 * 0.5 = 0.5
        let mid_idx = 50 + 25;
        assert!(
            (result[mid_idx] - 0.5).abs() < 0.05,
            "Midpoint of crossfade should be ~0.5, got {}",
            result[mid_idx]
        );
    }

    #[test]
    fn test_crossfade_three_segments() {
        let a = vec![1.0; 100];
        let b = vec![0.5; 100];
        let c = vec![0.0; 100];
        let crossfade_len = 20;
        let result = concatenate_with_crossfade(&[a, b, c], crossfade_len);
        // Total ≈ 300 - 2*20 = 260
        assert!(
            (result.len() as i64 - 260).unsigned_abs() < 5,
            "Three segments should produce ~260 samples, got {}",
            result.len()
        );
    }

    #[test]
    fn test_crossfade_compensation_restores_base_total() {
        let base = vec![100usize, 200, 300];
        let overlaps = vec![12usize, 24];
        let compensated = compensate_segment_targets_for_crossfades(&base, &overlaps);
        let final_len = compensated.iter().sum::<usize>() - overlaps.iter().sum::<usize>();
        assert_eq!(
            final_len,
            base.iter().sum::<usize>(),
            "Crossfade compensation should preserve base total after overlap subtraction"
        );
    }

    #[test]
    fn test_adaptive_crossfade_shorter_on_transient_boundaries() {
        let segments = vec![
            Segment {
                start: 0,
                end: 10000,
                is_transient: false,
                stretch_ratio: 1.0,
            },
            Segment {
                start: 10000,
                end: 20000,
                is_transient: false,
                stretch_ratio: 1.0,
            },
            Segment {
                start: 20000,
                end: 30000,
                is_transient: true,
                stretch_ratio: 1.0,
            },
            Segment {
                start: 30000,
                end: 40000,
                is_transient: false,
                stretch_ratio: 1.0,
            },
        ];

        let lens = compute_adaptive_crossfade_lens(&segments, 44100);
        assert_eq!(lens.len(), 3);
        assert!(
            lens[1] < lens[0],
            "Tonal→transient crossfade should be shorter than tonal→tonal"
        );
        assert!(
            lens[2] < lens[0],
            "Transient→tonal crossfade should be shorter than tonal→tonal"
        );
    }

    #[test]
    fn test_reconcile_total_segment_targets_hits_desired_sum() {
        let mut targets = vec![100usize, 200, 300];
        reconcile_total_segment_targets(&mut targets, 700);
        assert_eq!(targets.iter().sum::<usize>(), 700);

        reconcile_total_segment_targets(&mut targets, 450);
        assert_eq!(targets.iter().sum::<usize>(), 450);
    }

    #[test]
    fn test_timeline_bookkeeping_invariants() {
        let segment_targets = vec![500usize, 720, 680];
        let overlaps = vec![20usize, 30];
        let expected = segment_targets.iter().sum::<usize>() - overlaps.iter().sum::<usize>();
        let target = expected;
        let timeline =
            TimelineBookkeeping::from_lengths(target, &segment_targets, &overlaps, expected);
        assert!(timeline.is_consistent(), "Timeline invariants should hold");
        assert_eq!(timeline.expected_concat_len, expected);
    }

    // --- merge_onsets_and_beats: dedup distance boundary ---

    #[test]
    fn test_merge_dedup_distance_exactly_512() {
        // Beat exactly 512 samples from onset → just barely too close, should be deduped
        let onsets = vec![1000];
        let strengths = vec![1.0];
        let beats = vec![1512]; // exactly 512 away
        let (merged_onsets, _) = merge_onsets_and_beats(&onsets, &strengths, &beats, 44100);
        // DEDUP_DISTANCE = 512, condition is dist < 512, so 512 is NOT too close
        assert_eq!(merged_onsets, vec![1000, 1512]);
    }

    #[test]
    fn test_merge_dedup_distance_511() {
        // Beat 511 samples from onset → too close, should be deduped
        let onsets = vec![1000];
        let strengths = vec![1.0];
        let beats = vec![1511]; // 511 away (< 512)
        let (merged_onsets, _) = merge_onsets_and_beats(&onsets, &strengths, &beats, 44100);
        assert_eq!(merged_onsets, vec![1000]); // beat deduped
    }

    // --- separate_sub_bass: short input fallback ---

    #[test]
    fn test_separate_sub_bass_short_input() {
        // Input shorter than BAND_SPLIT_FFT_SIZE → sub_bass is zeros, remainder is input
        let input = vec![0.5f32; 100];
        let (sub, rem) = separate_sub_bass(&input, 120.0, 44100);
        assert_eq!(sub.len(), 100);
        assert_eq!(rem.len(), 100);
        // Sub should be all zeros
        assert!(sub.iter().all(|&s| s.abs() < 1e-10));
        // Remainder should equal input
        for (i, (&r, &inp)) in rem.iter().zip(input.iter()).enumerate() {
            assert!(
                (r - inp).abs() < 1e-6,
                "Sample {}: remainder {} != input {}",
                i,
                r,
                inp
            );
        }
    }

    // --- stretch_segment fallback paths ---

    #[test]
    fn test_hybrid_very_short_segment_fallback() {
        // Input shorter than MIN_SEGMENT_FOR_STRETCH → linear resampling fallback
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_band_split(false);
        let stretcher = HybridStretcher::new(params);

        // 200 samples < MIN_SEGMENT_FOR_STRETCH=256 → falls back to WSOLA first,
        // but if it's a single segment it goes through stretch_segment
        let input: Vec<f32> = (0..200)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let output = stretcher.process(&input).unwrap();
        assert!(!output.is_empty());
        assert!(output.iter().all(|s| s.is_finite()));
    }

    // --- BPM-aware transient snapping tests ---

    #[test]
    fn test_bpm_snapping_no_bpm_is_noop() {
        // Without BPM set, the hybrid stretcher should work exactly as before
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let mut input = vec![0.0f32; num_samples];

        // Add clicks every 0.5 seconds
        for beat in 0..4 {
            let pos = (beat as f64 * 0.5 * sample_rate as f64) as usize;
            for j in 0..20.min(num_samples - pos) {
                input[pos + j] = if j < 5 { 0.8 } else { -0.3 };
            }
        }
        for (i, sample) in input.iter_mut().enumerate() {
            *sample += 0.3 * (2.0 * PI * 200.0 * i as f32 / sample_rate as f32).sin();
        }

        // No BPM set (default)
        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1);
        assert!(params.bpm.is_none());

        let stretcher = HybridStretcher::new(params);
        let output = stretcher.process(&input).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_bpm_snapping_with_bpm_set() {
        // With BPM set, the hybrid stretcher should produce valid output
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let bpm = 120.0;
        let beat_interval = (60.0 * sample_rate as f64 / bpm) as usize;

        let mut input = vec![0.0f32; num_samples];

        // Add kicks at beat positions
        for beat in 0..5 {
            let pos = beat * beat_interval;
            if pos >= num_samples {
                break;
            }
            for j in 0..20.min(num_samples - pos) {
                input[pos + j] = if j < 5 { 0.9 } else { -0.4 };
            }
        }
        for (i, sample) in input.iter_mut().enumerate() {
            *sample += 0.3 * (2.0 * PI * 200.0 * i as f32 / sample_rate as f32).sin();
        }

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_bpm(bpm);
        assert_eq!(params.bpm, Some(bpm));

        let stretcher = HybridStretcher::new(params);
        let output = stretcher.process(&input).unwrap();
        assert!(!output.is_empty());

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.5).abs() < 0.3,
            "Length ratio {} too far from 1.5",
            len_ratio
        );
    }

    #[test]
    fn test_bpm_snapping_with_preset_and_bpm() {
        // Combine a preset with BPM for full integration
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let bpm = 128.0;
        let beat_interval = (60.0 * sample_rate as f64 / bpm) as usize;

        let mut input = vec![0.0f32; num_samples];
        for beat in 0..5 {
            let pos = beat * beat_interval;
            if pos >= num_samples {
                break;
            }
            for j in 0..20.min(num_samples - pos) {
                input[pos + j] = if j < 5 { 0.9 } else { -0.4 };
            }
        }
        for (i, sample) in input.iter_mut().enumerate() {
            *sample += 0.3 * (2.0 * PI * 200.0 * i as f32 / sample_rate as f32).sin();
        }

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(EdmPreset::DjBeatmatch)
            .with_bpm(bpm);
        assert_eq!(params.bpm, Some(bpm));

        let stretcher = HybridStretcher::new(params);
        let output = stretcher.process(&input).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_bpm_snapping_halftime_uses_eighth_notes() {
        // Halftime preset should use subdivision=8 (1/8th notes)
        use crate::analysis::beat::default_subdivision_for_preset;
        let sub = default_subdivision_for_preset(Some(EdmPreset::Halftime));
        assert_eq!(sub, 8, "Halftime should use 1/8th note subdivisions");
    }

    #[test]
    fn test_bpm_snapping_ambient_uses_quarter_notes() {
        // Ambient preset should use subdivision=4 (quarter notes)
        use crate::analysis::beat::default_subdivision_for_preset;
        let sub = default_subdivision_for_preset(Some(EdmPreset::Ambient));
        assert_eq!(sub, 4, "Ambient should use quarter note subdivisions");
    }

    #[test]
    fn test_bpm_snapping_backward_compatible_output() {
        // Output with BPM snapping should be similar length to output without it
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let bpm = 128.0;
        let beat_interval = (60.0 * sample_rate as f64 / bpm) as usize;

        let mut input = vec![0.0f32; num_samples];
        for beat in 0..5 {
            let pos = beat * beat_interval;
            if pos >= num_samples {
                break;
            }
            for j in 0..20.min(num_samples - pos) {
                input[pos + j] = if j < 5 { 0.9 } else { -0.4 };
            }
        }
        for (i, sample) in input.iter_mut().enumerate() {
            *sample += 0.3 * (2.0 * PI * 200.0 * i as f32 / sample_rate as f32).sin();
        }

        // Without BPM
        let params_no_bpm = StretchParams::new(1.3)
            .with_sample_rate(sample_rate)
            .with_channels(1);
        let stretcher_no_bpm = HybridStretcher::new(params_no_bpm);
        let output_no_bpm = stretcher_no_bpm.process(&input).unwrap();

        // With BPM
        let params_bpm = StretchParams::new(1.3)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_bpm(bpm);
        let stretcher_bpm = HybridStretcher::new(params_bpm);
        let output_bpm = stretcher_bpm.process(&input).unwrap();

        // Both should produce output of similar length (within 20%)
        let ratio = output_bpm.len() as f64 / output_no_bpm.len() as f64;
        assert!(
            (0.8..=1.2).contains(&ratio),
            "BPM-snapped output length ({}) should be similar to non-snapped ({}), ratio={}",
            output_bpm.len(),
            output_no_bpm.len(),
            ratio
        );
    }
}

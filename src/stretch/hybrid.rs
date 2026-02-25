//! Hybrid stretcher combining WSOLA (transients) with phase vocoder (tonal content).

use crate::analysis::beat::detect_beats;
use crate::analysis::frequency::freq_to_bin;
use crate::analysis::transient::detect_transients;
use crate::core::fft::{COMPLEX_ZERO, WINDOW_SUM_EPSILON, WINDOW_SUM_FLOOR_RATIO};
use crate::core::types::StretchParams;
use crate::core::window::{generate_window, WindowType};
use crate::error::StretchError;
use crate::stretch::phase_vocoder::PhaseVocoder;
use crate::stretch::wsola::Wsola;
use rustfft::{num_complex::Complex, FftPlanner};

/// Crossfade duration in seconds between algorithm segments (5ms raised-cosine).
const CROSSFADE_SECS: f64 = 0.005;
/// Transient region duration in seconds (~10ms around each onset).
const TRANSIENT_REGION_SECS: f64 = 0.010;
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
/// Crossover frequency (Hz) for multi-resolution FFT processing.
/// Below this, a larger FFT is used; above this, a smaller FFT provides
/// better temporal resolution for transient-rich high-frequency content.
const MULTI_RES_CROSSOVER_HZ: f32 = 4000.0;
/// FFT size for the high-frequency band in multi-resolution mode.
/// Half the default 4096, giving 2x better temporal resolution at the cost
/// of 2x worse frequency resolution (acceptable above 4 kHz).
const MULTI_RES_HIGH_FFT_SIZE: usize = 2048;

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
}

impl HybridStretcher {
    /// Creates a new hybrid stretcher.
    pub fn new(params: StretchParams) -> Self {
        Self { params }
    }

    /// Stretches a mono audio signal using the hybrid algorithm.
    pub fn process(&self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        if input.is_empty() {
            return Ok(vec![]);
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

        // Band-split mode: separate sub-bass for independent PV processing
        if self.params.band_split && input.len() >= BAND_SPLIT_FFT_SIZE {
            return self.process_band_split(input);
        }

        self.process_hybrid(input)
    }

    /// Processes audio with sub-bass band splitting.
    ///
    /// Separates audio below `sub_bass_cutoff` Hz and processes it exclusively
    /// through the phase vocoder with rigid phase locking. The remaining audio
    /// goes through the normal hybrid algorithm. The two results are summed.
    fn process_band_split(&self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        let (sub_bass, remainder) =
            separate_sub_bass(input, self.params.sub_bass_cutoff, self.params.sample_rate);

        // Process sub-bass exclusively through PV (rigid phase locking
        // handles phase coherence for bins below cutoff)
        let sub_bass_stretched = if sub_bass.len() >= self.params.fft_size {
            let mut pv = PhaseVocoder::with_all_options(
                self.params.fft_size,
                self.params.hop_size,
                self.params.stretch_ratio,
                self.params.sample_rate,
                self.params.sub_bass_cutoff,
                self.params.window_type,
                self.params.phase_locking_mode,
                self.params.envelope_preservation,
                self.params.envelope_order,
            );
            pv.process(&sub_bass)?
        } else {
            let out_len = self.params.output_length(sub_bass.len()).max(1);
            crate::core::resample::resample_linear(&sub_bass, out_len)
        };

        // Process remainder through normal hybrid (transient detection + PV/WSOLA)
        let remainder_stretched = self.process_hybrid(&remainder)?;

        // Sum the two bands (zero-pad the shorter one to the longer length)
        let out_len = sub_bass_stretched.len().max(remainder_stretched.len());
        let zeros = std::iter::repeat(0.0f32);
        let output: Vec<f32> = sub_bass_stretched
            .iter()
            .copied()
            .chain(zeros.clone())
            .zip(remainder_stretched.iter().copied().chain(zeros))
            .take(out_len)
            .map(|(s, r)| s + r)
            .collect();

        Ok(output)
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
        // Use a reference to avoid cloning when beat-aware is disabled.
        let merged;
        let onsets: &[usize] =
            if self.params.beat_aware && input.len() >= MIN_SAMPLES_FOR_BEAT_DETECTION {
                let grid = detect_beats(input, self.params.sample_rate);
                merged = merge_onsets_and_beats(&transients.onsets, &grid.beats, input.len());
                &merged
            } else {
                &transients.onsets
            };

        // Step 2: Segment audio at transient/beat boundaries
        let segments = self.segment_audio(input.len(), onsets);

        // Step 3: Process each segment with appropriate algorithm
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

        // Create a second PV with smaller FFT for multi-resolution high-band processing
        let mut pv_high = if self.params.multi_resolution {
            // Compute high-band hop to maintain the same overlap ratio as the main PV
            let high_hop = (self.params.hop_size as f64 * MULTI_RES_HIGH_FFT_SIZE as f64
                / self.params.fft_size as f64)
                .round() as usize;
            Some(PhaseVocoder::with_options(
                MULTI_RES_HIGH_FFT_SIZE,
                high_hop,
                self.params.stretch_ratio,
                self.params.sample_rate,
                self.params.sub_bass_cutoff,
                self.params.window_type,
                self.params.phase_locking_mode,
            ))
        } else {
            None
        };

        let mut output_segments: Vec<Vec<f32>> = Vec::with_capacity(segments.len());

        for segment in &segments {
            let seg_data = &input[segment.start..segment.end];
            let stretched =
                self.stretch_segment(seg_data, segment.is_transient, &mut pv, &mut pv_high);
            output_segments.push(stretched);

            // Reset PV phase state after transient segments so stale phase
            // from the previous tonal region doesn't contaminate the next one.
            if segment.is_transient {
                pv.reset_phase_state();
                if let Some(ref mut pv_h) = pv_high {
                    pv_h.reset_phase_state();
                }
            }
        }

        // Step 4: Concatenate with crossfades
        // Single segment fast path avoids crossfade overhead
        if output_segments.len() == 1 {
            return Ok(output_segments.into_iter().next().unwrap_or_default());
        }

        let crossfade_samples = (self.params.sample_rate as f64 * CROSSFADE_SECS) as usize;
        let output = concatenate_with_crossfade(&output_segments, crossfade_samples);

        Ok(output)
    }

    /// Stretches a single segment using the appropriate algorithm.
    ///
    /// - Very short segments (<256 samples) fall back to linear resampling
    /// - Tonal segments long enough for FFT use the phase vocoder
    /// - When multi-resolution is enabled, tonal segments are split into low/high
    ///   bands and processed with different FFT sizes
    /// - Everything else (transients, short tonal) uses WSOLA
    /// - On error, falls back to linear resampling
    fn stretch_segment(
        &self,
        seg_data: &[f32],
        is_transient: bool,
        pv: &mut PhaseVocoder,
        pv_high: &mut Option<PhaseVocoder>,
    ) -> Vec<f32> {
        if seg_data.len() < MIN_SEGMENT_FOR_STRETCH {
            let out_len = self.params.output_length(seg_data.len());
            return crate::core::resample::resample_linear(seg_data, out_len.max(1));
        }

        let use_phase_vocoder = !is_transient && seg_data.len() >= self.params.fft_size;

        // Multi-resolution path: split tonal segments into low/high bands
        if use_phase_vocoder && pv_high.is_some() {
            let result =
                self.process_tonal_multi_resolution(seg_data, pv, pv_high.as_mut().unwrap());
            return result.unwrap_or_else(|_| {
                let out_len = self.params.output_length(seg_data.len());
                crate::core::resample::resample_linear(seg_data, out_len.max(1))
            });
        }

        // For segments too short for the main FFT but long enough for the
        // high-band FFT, use the smaller PV when multi-resolution is enabled.
        if !is_transient
            && seg_data.len() >= MULTI_RES_HIGH_FFT_SIZE
            && seg_data.len() < self.params.fft_size
            && pv_high.is_some()
        {
            let result = pv_high.as_mut().unwrap().process(seg_data);
            return result.unwrap_or_else(|_| {
                let out_len = self.params.output_length(seg_data.len());
                crate::core::resample::resample_linear(seg_data, out_len.max(1))
            });
        }

        let result = if use_phase_vocoder {
            pv.process(seg_data)
        } else {
            self.stretch_with_wsola(seg_data)
        };

        result.unwrap_or_else(|_| {
            let out_len = self.params.output_length(seg_data.len());
            crate::core::resample::resample_linear(seg_data, out_len.max(1))
        })
    }

    /// Processes a tonal segment using multi-resolution FFT.
    ///
    /// Splits the segment into low (0-4 kHz) and high (4 kHz+) frequency bands,
    /// processes the low band with the main PV (large FFT for frequency resolution)
    /// and the high band with a smaller PV (better temporal resolution), then sums
    /// the results.
    fn process_tonal_multi_resolution(
        &self,
        seg_data: &[f32],
        pv_low: &mut PhaseVocoder,
        pv_high: &mut PhaseVocoder,
    ) -> Result<Vec<f32>, StretchError> {
        let (low_band, high_band) =
            separate_bands(seg_data, MULTI_RES_CROSSOVER_HZ, self.params.sample_rate);

        // Process low band (0-4kHz) with the main 4096 PV
        let low_stretched = if low_band.len() >= self.params.fft_size {
            pv_low.process(&low_band)?
        } else {
            let out_len = self.params.output_length(low_band.len()).max(1);
            crate::core::resample::resample_linear(&low_band, out_len)
        };

        // Process high band (4kHz+) with the smaller 2048 PV
        let high_stretched = if high_band.len() >= MULTI_RES_HIGH_FFT_SIZE {
            pv_high.process(&high_band)?
        } else {
            let out_len = self.params.output_length(high_band.len()).max(1);
            crate::core::resample::resample_linear(&high_band, out_len)
        };

        // Sum the two bands (zero-pad the shorter one to the longer length)
        let out_len = low_stretched.len().max(high_stretched.len());
        let zeros = std::iter::repeat(0.0f32);
        let output: Vec<f32> = low_stretched
            .iter()
            .copied()
            .chain(zeros.clone())
            .zip(high_stretched.iter().copied().chain(zeros))
            .take(out_len)
            .map(|(l, h)| l + h)
            .collect();

        Ok(output)
    }

    /// Stretches a segment using WSOLA with clamped parameters.
    fn stretch_with_wsola(&self, seg_data: &[f32]) -> Result<Vec<f32>, StretchError> {
        let seg_size = self
            .params
            .wsola_segment_size
            .min(seg_data.len() / 2)
            .max(MIN_WSOLA_SEGMENT);
        let search = self
            .params
            .wsola_search_range
            .min(seg_size / 2)
            .max(MIN_WSOLA_SEARCH);
        let mut wsola = Wsola::new(seg_size, search, self.params.stretch_ratio);
        wsola.process(seg_data)
    }

    /// Segments audio into transient and tonal regions.
    fn segment_audio(&self, input_len: usize, onsets: &[usize]) -> Vec<Segment> {
        if onsets.is_empty() {
            return vec![Segment {
                start: 0,
                end: input_len,
                is_transient: false,
            }];
        }

        let mut segments = Vec::new();
        let transient_size = (self.params.sample_rate as f64 * TRANSIENT_REGION_SECS) as usize;

        let mut pos = 0;

        for &onset in onsets {
            if onset <= pos {
                continue;
            }

            // Tonal region before this onset
            let tonal_end = onset.min(input_len);
            if tonal_end > pos {
                segments.push(Segment {
                    start: pos,
                    end: tonal_end,
                    is_transient: false,
                });
            }

            // Transient region
            let trans_end = (onset + transient_size).min(input_len);
            if trans_end > onset {
                segments.push(Segment {
                    start: onset,
                    end: trans_end,
                    is_transient: true,
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
            });
        }

        segments
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

/// Separates audio into low and high frequency bands using FFT-based overlap-add filtering.
///
/// Uses the same approach as [`separate_sub_bass()`] but with a configurable crossover
/// frequency (typically ~4000 Hz for multi-resolution processing).
///
/// Returns `(low_band, high_band)` where low contains everything below `crossover_hz`
/// and high contains everything at or above `crossover_hz`. Both outputs have the same
/// length as the input.
fn separate_bands(input: &[f32], crossover_hz: f32, sample_rate: u32) -> (Vec<f32>, Vec<f32>) {
    let fft_size = BAND_SPLIT_FFT_SIZE;
    let hop = BAND_SPLIT_HOP;
    let cutoff_bin = freq_to_bin(crossover_hz, fft_size, sample_rate);

    if cutoff_bin == 0 || input.len() < fft_size {
        return (input.to_vec(), vec![0.0; input.len()]);
    }

    let window = generate_window(WindowType::Hann, fft_size);
    let mut planner = FftPlanner::new();
    let fft_fwd = planner.plan_fft_forward(fft_size);
    let fft_inv = planner.plan_fft_inverse(fft_size);
    let norm = 1.0 / fft_size as f32;

    let mut low_band = vec![0.0f32; input.len()];
    let mut high_band = vec![0.0f32; input.len()];
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
            low_band[out_idx] += fft_buf[i].re * norm * window[i];
            high_band[out_idx] += fft_buf2[i].re * norm * window[i];
            window_sum[out_idx] += window[i] * window[i];
        }
    }

    normalize_band_split(&mut low_band, &mut high_band, &window_sum);
    (low_band, high_band)
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
fn merge_onsets_and_beats(onsets: &[usize], beats: &[usize], input_len: usize) -> Vec<usize> {
    let mut merged: Vec<usize> = Vec::with_capacity(onsets.len() + beats.len());
    merged.extend_from_slice(onsets);

    // Add beat positions that aren't too close to existing transient onsets
    for &beat in beats {
        if beat >= input_len {
            continue;
        }
        let too_close = merged.iter().any(|&pos| {
            let dist = pos.abs_diff(beat);
            dist < DEDUP_DISTANCE
        });
        if !too_close {
            merged.push(beat);
        }
    }

    merged.sort_unstable();
    merged
}

/// Concatenates segments with raised-cosine crossfade.
fn concatenate_with_crossfade(segments: &[Vec<f32>], crossfade_len: usize) -> Vec<f32> {
    match segments.len() {
        0 => return vec![],
        1 => return segments[0].clone(),
        _ => {}
    }

    // Estimate total length
    let total: usize = segments.iter().map(|s| s.len()).sum();
    let overlap_total = crossfade_len * (segments.len() - 1);
    let mut output = Vec::with_capacity(total.saturating_sub(overlap_total));

    for (idx, segment) in segments.iter().enumerate() {
        if idx == 0 {
            output.extend_from_slice(segment);
        } else {
            let fade_len = crossfade_len.min(output.len()).min(segment.len());
            let output_start = output.len() - fade_len;

            // Crossfade overlap region
            for i in 0..fade_len {
                let t = i as f32 / fade_len as f32;
                // Raised cosine crossfade
                let fade_out = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
                let fade_in = 1.0 - fade_out;
                output[output_start + i] =
                    output[output_start + i] * fade_out + segment[i] * fade_in;
            }

            // Append non-overlapping part
            if fade_len < segment.len() {
                output.extend_from_slice(&segment[fade_len..]);
            }
        }
    }

    output
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
        let result = merge_onsets_and_beats(&[], &[], 44100);
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_onsets_and_beats_no_overlap() {
        let onsets = vec![1000, 5000];
        let beats = vec![10000, 20000];
        let result = merge_onsets_and_beats(&onsets, &beats, 44100);
        assert_eq!(result, vec![1000, 5000, 10000, 20000]);
    }

    #[test]
    fn test_merge_onsets_and_beats_dedup_close() {
        // Beat at 1100 is within 512 samples of onset at 1000 — should be deduped
        let onsets = vec![1000, 5000];
        let beats = vec![1100, 20000];
        let result = merge_onsets_and_beats(&onsets, &beats, 44100);
        assert_eq!(result, vec![1000, 5000, 20000]);
    }

    #[test]
    fn test_merge_onsets_and_beats_out_of_bounds() {
        // Beat at 50000 exceeds input_len of 44100 — should be dropped
        let onsets = vec![1000];
        let beats = vec![50000];
        let result = merge_onsets_and_beats(&onsets, &beats, 44100);
        assert_eq!(result, vec![1000]);
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
        // EDM presets should enable band_split by default
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

            assert!(
                params.band_split,
                "Preset {:?} should enable band_split",
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
        let segments = stretcher.segment_audio(44100, &[]);
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
        let segments = stretcher.segment_audio(input_len, &[onset]);

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
        let segments = stretcher.segment_audio(44100, &[0]);

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
        let segments = stretcher.segment_audio(input_len, &[onset]);

        // Last transient segment should be clamped to input_len
        let last_transient = segments.iter().find(|s| s.is_transient).unwrap();
        assert!(last_transient.end <= input_len);
    }

    #[test]
    fn test_segment_audio_overlapping_onsets() {
        // Two onsets where second falls within transient region of first
        let params = StretchParams::new(1.5).with_sample_rate(44100);
        let stretcher = HybridStretcher::new(params);
        let transient_size = (44100.0 * TRANSIENT_REGION_SECS) as usize; // ~441
        let onset1 = 10000;
        let onset2 = onset1 + transient_size / 2; // Within transient region of onset1

        let segments = stretcher.segment_audio(44100, &[onset1, onset2]);

        // Second onset should be skipped (onset2 <= pos after first transient)
        let transient_count = segments.iter().filter(|s| s.is_transient).count();
        // Should have 1 or 2 transient segments depending on overlap
        assert!(
            transient_count >= 1,
            "Should have at least 1 transient segment"
        );
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

    // --- merge_onsets_and_beats: dedup distance boundary ---

    #[test]
    fn test_merge_dedup_distance_exactly_512() {
        // Beat exactly 512 samples from onset → just barely too close, should be deduped
        let onsets = vec![1000];
        let beats = vec![1512]; // exactly 512 away
        let result = merge_onsets_and_beats(&onsets, &beats, 44100);
        // DEDUP_DISTANCE = 512, condition is dist < 512, so 512 is NOT too close
        assert_eq!(result, vec![1000, 1512]);
    }

    #[test]
    fn test_merge_dedup_distance_511() {
        // Beat 511 samples from onset → too close, should be deduped
        let onsets = vec![1000];
        let beats = vec![1511]; // 511 away (< 512)
        let result = merge_onsets_and_beats(&onsets, &beats, 44100);
        assert_eq!(result, vec![1000]); // beat deduped
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
}

//! Hybrid stretcher combining WSOLA (transients) with phase vocoder (tonal content).

use crate::analysis::beat::detect_beats;
use crate::analysis::frequency::freq_to_bin;
use crate::analysis::transient::detect_transients;
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
        let (sub_bass, remainder) = separate_sub_bass(
            input,
            self.params.sub_bass_cutoff,
            self.params.sample_rate,
        );

        // Process sub-bass exclusively through PV (rigid phase locking
        // handles phase coherence for bins below cutoff)
        let sub_bass_stretched = if sub_bass.len() >= self.params.fft_size {
            let mut pv = PhaseVocoder::new(
                self.params.fft_size,
                self.params.hop_size,
                self.params.stretch_ratio,
                self.params.sample_rate,
                self.params.sub_bass_cutoff,
            );
            pv.process(&sub_bass)?
        } else {
            let out_len = self.params.output_length(sub_bass.len()).max(1);
            crate::core::resample::resample_linear(&sub_bass, out_len)
        };

        // Process remainder through normal hybrid (transient detection + PV/WSOLA)
        let remainder_stretched = self.process_hybrid(&remainder)?;

        // Sum the two bands, using the longer length (zero-pad the shorter)
        let out_len = sub_bass_stretched.len().max(remainder_stretched.len());
        let mut output = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let sub = sub_bass_stretched.get(i).copied().unwrap_or(0.0);
            let rem = remainder_stretched.get(i).copied().unwrap_or(0.0);
            output.push(sub + rem);
        }

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
        let onsets: &[usize] = if self.params.beat_aware
            && input.len() >= MIN_SAMPLES_FOR_BEAT_DETECTION
        {
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
        let mut pv = PhaseVocoder::new(
            self.params.fft_size,
            self.params.hop_size,
            self.params.stretch_ratio,
            self.params.sample_rate,
            self.params.sub_bass_cutoff,
        );
        let mut output_segments: Vec<Vec<f32>> = Vec::with_capacity(segments.len());

        for segment in &segments {
            let seg_data = &input[segment.start..segment.end];
            let stretched = self.stretch_segment(seg_data, segment.is_transient, &mut pv);
            output_segments.push(stretched);
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
    /// - Everything else (transients, short tonal) uses WSOLA
    /// - On error, falls back to linear resampling
    fn stretch_segment(
        &self,
        seg_data: &[f32],
        is_transient: bool,
        pv: &mut PhaseVocoder,
    ) -> Vec<f32> {
        if seg_data.len() < MIN_SEGMENT_FOR_STRETCH {
            let out_len = self.params.output_length(seg_data.len());
            return crate::core::resample::resample_linear(seg_data, out_len.max(1));
        }

        let use_phase_vocoder = !is_transient && seg_data.len() >= self.params.fft_size;
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
        // No sub-bass to separate, or input too short for FFT
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

    let mut fft_buf = vec![Complex::new(0.0f32, 0.0); fft_size];
    let mut fft_buf2 = vec![Complex::new(0.0f32, 0.0); fft_size];

    for frame in 0..num_frames {
        let pos = frame * hop;
        let frame_end = (pos + fft_size).min(input.len());
        let frame_len = frame_end - pos;

        // Window and transform
        for i in 0..fft_size {
            fft_buf[i] = if i < frame_len {
                Complex::new(input[pos + i] * window[i], 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        fft_fwd.process(&mut fft_buf);

        // Split into sub-bass and remainder in the frequency domain.
        // For a real-valued signal, the FFT has conjugate symmetry:
        //   bin 0 = DC, bins 1..N/2 = positive freqs,
        //   bin N/2 = Nyquist, bins N/2+1..N-1 = negative freqs (mirror).
        // We need to keep both positive and negative frequency bins consistent.
        fft_buf2.copy_from_slice(&fft_buf);
        let zero = Complex::new(0.0, 0.0);

        // Sub-bass: keep bins 0..cutoff_bin and their mirrors, zero everything else
        for bin in cutoff_bin..=fft_size / 2 {
            fft_buf[bin] = zero;
            if bin > 0 && bin < fft_size / 2 {
                fft_buf[fft_size - bin] = zero;
            }
        }
        // Remainder: keep bins cutoff_bin..N/2 and their mirrors, zero sub-bass
        for bin in 0..cutoff_bin {
            fft_buf2[bin] = zero;
            if bin > 0 {
                fft_buf2[fft_size - bin] = zero;
            }
        }

        // Inverse FFT both
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

    // Normalize by window sum
    let max_ws = window_sum.iter().cloned().fold(0.0f32, f32::max);
    let min_ws = (max_ws * 0.1).max(1e-6);
    for i in 0..input.len() {
        let ws = window_sum[i].max(min_ws);
        sub_bass[i] /= ws;
        remainder[i] /= ws;
    }

    (sub_bass, remainder)
}

/// Merges transient onsets with beat grid positions, deduplicating nearby entries.
///
/// Beat positions that fall within `DEDUP_DISTANCE` samples of an existing
/// transient onset are dropped to avoid creating overly short segments.
fn merge_onsets_and_beats(onsets: &[usize], beats: &[usize], input_len: usize) -> Vec<usize> {
    /// Minimum distance (samples) between merged positions.
    /// Positions closer than this are considered duplicates.
    const DEDUP_DISTANCE: usize = 512;

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

        let sub_rms =
            (sub_bass.iter().map(|x| x * x).sum::<f32>() / sub_bass.len() as f32).sqrt();
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

        let sub_rms =
            (sub_bass.iter().map(|x| x * x).sum::<f32>() / sub_bass.len() as f32).sqrt();
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
        let input_rms = (input[start..end].iter().map(|x| (*x as f64).powi(2)).sum::<f64>()
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
            assert!(!output.is_empty(), "Preset {:?} produced empty output", preset);
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

        let sub_rms =
            (sub_bass.iter().map(|x| x * x).sum::<f32>() / sub_bass.len() as f32).sqrt();
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
}

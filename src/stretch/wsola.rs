//! WSOLA (Waveform Similarity Overlap-Add) time stretching.

use crate::error::StretchError;
use rustfft::{num_complex::Complex, FftPlanner};

/// Minimum energy threshold to avoid division by near-zero in correlation normalization.
const ENERGY_EPSILON: f64 = 1e-12;
/// Minimum number of candidates to justify FFT-based correlation over direct computation.
const FFT_CANDIDATE_THRESHOLD: usize = 64;
/// Minimum overlap length for FFT-based correlation to be worthwhile.
const FFT_OVERLAP_THRESHOLD: usize = 32;

/// WSOLA (Waveform Similarity Overlap-Add) time stretching.
///
/// Preserves transient quality better than phase vocoder by operating
/// in the time domain and finding optimal overlap positions via
/// cross-correlation.
pub struct Wsola {
    segment_size: usize,
    overlap_size: usize,
    search_range: usize,
    stretch_ratio: f64,
    planner: FftPlanner<f32>,
}

impl Wsola {
    /// Creates a new WSOLA processor.
    pub fn new(segment_size: usize, search_range: usize, stretch_ratio: f64) -> Self {
        let overlap_size = segment_size / 2;
        Self {
            segment_size,
            overlap_size,
            search_range,
            stretch_ratio,
            planner: FftPlanner::new(),
        }
    }

    /// Returns the segment size in samples.
    #[inline]
    pub fn segment_size(&self) -> usize {
        self.segment_size
    }

    /// Returns the search range in samples.
    #[inline]
    pub fn search_range(&self) -> usize {
        self.search_range
    }

    /// Returns the stretch ratio.
    #[inline]
    pub fn stretch_ratio(&self) -> f64 {
        self.stretch_ratio
    }

    /// Stretches a mono audio signal using WSOLA.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        if input.len() < self.segment_size {
            return Err(StretchError::InputTooShort {
                provided: input.len(),
                minimum: self.segment_size,
            });
        }

        let advance_input = self.segment_size - self.overlap_size;
        let advance_output_f = advance_input as f64 * self.stretch_ratio;

        if advance_output_f < 1.0 {
            return Err(StretchError::InvalidRatio(
                "Stretch ratio too small for segment size".to_string(),
            ));
        }

        // Target output length based on stretch ratio
        let target_output_len = (input.len() as f64 * self.stretch_ratio).round() as usize;

        // Estimate buffer size (add margin for search variations)
        let estimated_output_len = target_output_len + self.segment_size * 2;
        let mut output = vec![0.0f32; estimated_output_len];

        // Copy first segment
        let first_len = self.segment_size.min(input.len());
        output[..first_len].copy_from_slice(&input[..first_len]);

        let mut input_pos: f64 = advance_input as f64;
        // Track output position fractionally to avoid cumulative rounding error
        let mut output_pos_f: f64 = advance_output_f;
        let mut actual_output_len = first_len;

        while (input_pos as usize) + self.segment_size <= input.len() {
            // For compression (ratio < 1.0), stop once we've produced enough output
            if actual_output_len >= target_output_len {
                break;
            }

            let nominal_pos = input_pos as usize;
            let output_pos = output_pos_f.round() as usize;

            // Ensure we have room in the output buffer
            let needed = output_pos + self.segment_size;
            if needed > output.len() {
                output.resize(needed, 0.0);
            }

            // Search for best matching position around nominal position
            let best_pos = self.find_best_position(input, &output, nominal_pos, output_pos);

            // Overlap-add with cross-fade
            self.overlap_add(input, &mut output, best_pos, output_pos);
            actual_output_len = (output_pos + self.segment_size).max(actual_output_len);

            input_pos += advance_input as f64;
            output_pos_f += advance_output_f;
        }

        // Trim output to target length for accurate ratio
        let final_len = actual_output_len.min(target_output_len);
        output.truncate(final_len);
        Ok(output)
    }

    /// Finds the best matching position within the search range using FFT-accelerated
    /// cross-correlation for large search ranges, falling back to direct computation
    /// for small ranges.
    fn find_best_position(
        &mut self,
        input: &[f32],
        output: &[f32],
        nominal_pos: usize,
        output_pos: usize,
    ) -> usize {
        let search_start = nominal_pos.saturating_sub(self.search_range);
        let search_end =
            (nominal_pos + self.search_range).min(input.len().saturating_sub(self.segment_size));

        if search_start >= search_end {
            return nominal_pos.min(input.len().saturating_sub(self.segment_size));
        }

        let overlap_len = self
            .overlap_size
            .min(output.len().saturating_sub(output_pos));
        if overlap_len == 0 {
            return nominal_pos;
        }

        let num_candidates = search_end - search_start + 1;

        // Use FFT-based correlation when search range is large enough to benefit
        if num_candidates > FFT_CANDIDATE_THRESHOLD && overlap_len >= FFT_OVERLAP_THRESHOLD {
            self.find_best_position_fft(
                input,
                output,
                search_start,
                search_end,
                output_pos,
                overlap_len,
            )
        } else {
            self.find_best_position_direct(
                input,
                output,
                search_start,
                search_end,
                output_pos,
                overlap_len,
            )
        }
    }

    /// Direct time-domain cross-correlation search (used for small search ranges).
    fn find_best_position_direct(
        &self,
        input: &[f32],
        output: &[f32],
        search_start: usize,
        search_end: usize,
        output_pos: usize,
        overlap_len: usize,
    ) -> usize {
        let mut best_pos = search_start;
        let mut best_corr = f64::NEG_INFINITY;

        for pos in search_start..=search_end {
            if pos + overlap_len > input.len() {
                break;
            }

            let corr = normalized_cross_correlation(
                &output[output_pos..output_pos + overlap_len],
                &input[pos..pos + overlap_len],
            );

            if corr > best_corr {
                best_corr = corr;
                best_pos = pos;
            }
        }

        best_pos
    }

    /// FFT-accelerated cross-correlation search.
    ///
    /// Computes cross-correlation between the output overlap region (reference)
    /// and all candidate positions in the input search region simultaneously.
    fn find_best_position_fft(
        &mut self,
        input: &[f32],
        output: &[f32],
        search_start: usize,
        search_end: usize,
        output_pos: usize,
        overlap_len: usize,
    ) -> usize {
        let ref_signal = &output[output_pos..output_pos + overlap_len];
        let search_region_len = search_end - search_start + overlap_len;

        // Clamp to available input
        let actual_region_end = (search_start + search_region_len).min(input.len());
        let actual_region_len = actual_region_end - search_start;
        if actual_region_len < overlap_len {
            return search_start;
        }
        let search_signal = &input[search_start..actual_region_end];

        // Compute raw cross-correlation via FFT
        let corr_buf = self.fft_cross_correlate(ref_signal, search_signal);

        // Compute reference energy (constant for all candidates)
        let ref_energy: f64 = ref_signal.iter().map(|&s| (s as f64) * (s as f64)).sum();
        if ref_energy < ENERGY_EPSILON {
            return search_start;
        }

        // Find best candidate using normalized correlation
        let num_candidates = actual_region_len.saturating_sub(overlap_len) + 1;
        let best_pos = find_best_candidate(
            search_signal,
            &corr_buf,
            ref_energy,
            num_candidates,
            overlap_len,
            search_start,
        );

        // Clamp to valid range
        best_pos.min(search_end)
    }

    /// Computes cross-correlation between two signals using FFT.
    ///
    /// Returns the raw (unnormalized) correlation buffer in the frequency domain.
    fn fft_cross_correlate(
        &mut self,
        ref_signal: &[f32],
        search_signal: &[f32],
    ) -> Vec<Complex<f32>> {
        let conv_len = search_signal.len() + ref_signal.len() - 1;
        let fft_size = conv_len.next_power_of_two();

        let fft_fwd = self.planner.plan_fft_forward(fft_size);
        let fft_inv = self.planner.plan_fft_inverse(fft_size);

        // Zero-pad signals into FFT buffers
        let mut ref_buf = vec![Complex::new(0.0f32, 0.0); fft_size];
        for (i, &s) in ref_signal.iter().enumerate() {
            ref_buf[i] = Complex::new(s, 0.0);
        }

        let mut search_buf = vec![Complex::new(0.0f32, 0.0); fft_size];
        for (i, &s) in search_signal.iter().enumerate() {
            search_buf[i] = Complex::new(s, 0.0);
        }

        // Forward FFT, multiply conj(Ref) * Search, inverse FFT
        fft_fwd.process(&mut ref_buf);
        fft_fwd.process(&mut search_buf);

        let mut corr_buf = vec![Complex::new(0.0f32, 0.0); fft_size];
        for i in 0..fft_size {
            corr_buf[i] = ref_buf[i].conj() * search_buf[i];
        }

        fft_inv.process(&mut corr_buf);
        corr_buf
    }

    /// Overlap-adds a segment from input into output with raised-cosine crossfade.
    fn overlap_add(&self, input: &[f32], output: &mut [f32], input_pos: usize, output_pos: usize) {
        let segment_end = (input_pos + self.segment_size).min(input.len());
        let segment_len = segment_end - input_pos;

        for i in 0..segment_len {
            let out_idx = output_pos + i;
            if out_idx >= output.len() {
                break;
            }

            if i < self.overlap_size {
                // Crossfade region
                let fade = i as f32 / self.overlap_size as f32;
                let fade_in = fade;
                let fade_out = 1.0 - fade;
                output[out_idx] = output[out_idx] * fade_out + input[input_pos + i] * fade_in;
            } else {
                // Non-overlap region: just copy
                output[out_idx] = input[input_pos + i];
            }
        }
    }
}

/// Finds the best correlation candidate using prefix-sum energy normalization.
///
/// Scans `num_candidates` lag positions in `corr_buf`, normalizing each by
/// the windowed energy of `search_signal` and the reference energy.
fn find_best_candidate(
    search_signal: &[f32],
    corr_buf: &[Complex<f32>],
    ref_energy: f64,
    num_candidates: usize,
    overlap_len: usize,
    search_start: usize,
) -> usize {
    let norm = 1.0 / corr_buf.len() as f64;

    // Running energy of search signal windows via prefix sums
    let mut prefix_sq = vec![0.0f64; search_signal.len() + 1];
    for i in 0..search_signal.len() {
        prefix_sq[i + 1] = prefix_sq[i] + (search_signal[i] as f64) * (search_signal[i] as f64);
    }

    let mut best_pos = search_start;
    let mut best_ncorr = f64::NEG_INFINITY;

    for k in 0..num_candidates {
        let raw_corr = corr_buf[k].re as f64 * norm;
        let window_energy = prefix_sq[k + overlap_len] - prefix_sq[k];
        let denom = (ref_energy * window_energy).sqrt();

        let ncorr = if denom > ENERGY_EPSILON {
            raw_corr / denom
        } else {
            0.0
        };

        if ncorr > best_ncorr {
            best_ncorr = ncorr;
            best_pos = search_start + k;
        }
    }

    best_pos
}

/// Normalized cross-correlation between two signals.
#[inline]
fn normalized_cross_correlation(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut sum_ab = 0.0f64;
    let mut sum_a2 = 0.0f64;
    let mut sum_b2 = 0.0f64;

    for i in 0..len {
        let va = a[i] as f64;
        let vb = b[i] as f64;
        sum_ab += va * vb;
        sum_a2 += va * va;
        sum_b2 += vb * vb;
    }

    let denom = (sum_a2 * sum_b2).sqrt();
    if denom < ENERGY_EPSILON {
        return 0.0;
    }

    sum_ab / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_wsola_identity() {
        let sample_rate = 44100;
        let segment_size = 882; // ~20ms
        let search_range = 441; // ~10ms

        // 440 Hz sine wave, 1 second
        let input: Vec<f32> = (0..sample_rate)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut wsola = Wsola::new(segment_size, search_range, 1.0);
        let output = wsola.process(&input).unwrap();

        // Length should be approximately the same
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.0).abs() < 0.05,
            "Length ratio {} too far from 1.0",
            len_ratio
        );
    }

    #[test]
    fn test_wsola_stretch_2x() {
        let sample_rate = 44100;
        let segment_size = 882;
        let search_range = 441;

        let input: Vec<f32> = (0..sample_rate)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut wsola = Wsola::new(segment_size, search_range, 2.0);
        let output = wsola.process(&input).unwrap();

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 2.0).abs() < 0.1,
            "Length ratio {} too far from 2.0",
            len_ratio
        );
    }

    #[test]
    fn test_wsola_compress() {
        let sample_rate = 44100;
        let segment_size = 882;
        let search_range = 441;

        // Use a longer input for more stable ratio
        let input: Vec<f32> = (0..sample_rate * 2)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut wsola = Wsola::new(segment_size, search_range, 0.75);
        let output = wsola.process(&input).unwrap();

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 0.75).abs() < 0.1,
            "Length ratio {} too far from 0.75",
            len_ratio
        );

        // Test 0.5 ratio with tighter tolerance
        let mut wsola_half = Wsola::new(segment_size, search_range, 0.5);
        let output_half = wsola_half.process(&input).unwrap();
        let half_ratio = output_half.len() as f64 / input.len() as f64;
        assert!(
            (half_ratio - 0.5).abs() < 0.1,
            "Half compression ratio {} too far from 0.5",
            half_ratio
        );
    }

    #[test]
    fn test_wsola_extreme_compression() {
        let sample_rate = 44100;
        let segment_size = 882;
        let search_range = 441;

        // 3 seconds for stable measurement
        let input: Vec<f32> = (0..sample_rate * 3)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        // Test ratio 0.33 (3x speedup)
        let mut wsola = Wsola::new(segment_size, search_range, 0.33);
        let output = wsola.process(&input).unwrap();
        let ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (ratio - 0.33).abs() < 0.1,
            "Compression ratio {} too far from 0.33",
            ratio
        );

        // Test ratio 0.25 (4x speedup)
        let mut wsola = Wsola::new(segment_size, search_range, 0.25);
        let output = wsola.process(&input).unwrap();
        let ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (ratio - 0.25).abs() < 0.1,
            "Compression ratio {} too far from 0.25",
            ratio
        );
    }

    #[test]
    fn test_wsola_dj_ratios() {
        let sample_rate = 44100;
        let segment_size = 882;
        let search_range = 441;

        // 2 seconds of audio
        let input: Vec<f32> = (0..sample_rate * 2)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        // DJ-typical ratios: ±1-8%
        for &ratio in &[0.92, 0.96, 1.02, 1.04, 1.08] {
            let mut wsola = Wsola::new(segment_size, search_range, ratio);
            let output = wsola.process(&input).unwrap();
            let actual_ratio = output.len() as f64 / input.len() as f64;
            assert!(
                (actual_ratio - ratio).abs() < 0.05,
                "DJ ratio {}: actual {} too far from target",
                ratio,
                actual_ratio
            );
        }
    }

    #[test]
    fn test_wsola_extreme_compress() {
        let sample_rate = 44100;
        let segment_size = 882;
        let search_range = 441;

        let input: Vec<f32> = (0..sample_rate * 4)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        // Test ratios from 0.25 to 0.5
        for &ratio in &[0.5, 0.4, 0.3, 0.25] {
            let mut wsola = Wsola::new(segment_size, search_range, ratio);
            let output = wsola.process(&input).unwrap();
            let actual_ratio = output.len() as f64 / input.len() as f64;
            assert!(
                (actual_ratio - ratio).abs() < 0.1,
                "Ratio {}: actual {:.3} too far from target",
                ratio,
                actual_ratio
            );
        }
    }

    #[test]
    fn test_wsola_input_too_short() {
        let mut wsola = Wsola::new(882, 441, 1.0);
        let result = wsola.process(&[0.0; 100]);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalized_cross_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = normalized_cross_correlation(&a, &b);
        assert!(
            (c - 1.0).abs() < 1e-6,
            "Self-correlation should be 1.0, got {}",
            c
        );

        let neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let c_neg = normalized_cross_correlation(&a, &neg);
        assert!(
            (c_neg - (-1.0)).abs() < 1e-6,
            "Negated correlation should be -1.0, got {}",
            c_neg
        );
    }
}

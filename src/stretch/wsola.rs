//! WSOLA (Waveform Similarity Overlap-Add) time stretching.

use crate::core::fft::COMPLEX_ZERO;
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
    /// Reusable FFT buffer for reference signal in cross-correlation.
    fft_ref_buf: Vec<Complex<f32>>,
    /// Reusable FFT buffer for search signal in cross-correlation.
    fft_search_buf: Vec<Complex<f32>>,
    /// Reusable FFT buffer for correlation result.
    fft_corr_buf: Vec<Complex<f32>>,
    /// Reusable prefix-sum buffer for energy normalization.
    prefix_sq_buf: Vec<f64>,
    /// Reusable output buffer for overlap-add accumulation.
    output_buf: Vec<f32>,
}

impl std::fmt::Debug for Wsola {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wsola")
            .field("segment_size", &self.segment_size)
            .field("overlap_size", &self.overlap_size)
            .field("search_range", &self.search_range)
            .field("stretch_ratio", &self.stretch_ratio)
            .finish()
    }
}

impl Wsola {
    /// Creates a new WSOLA processor.
    ///
    /// For small stretch ratios (within ±15% of unity), uses a smaller overlap
    /// region (`segment_size / 4`) to reduce transient smearing. Larger ratios
    /// use the standard `segment_size / 2` overlap for better continuity.
    pub fn new(segment_size: usize, search_range: usize, stretch_ratio: f64) -> Self {
        let overlap_size = if (stretch_ratio - 1.0).abs() < 0.15 {
            segment_size / 4
        } else {
            segment_size / 2
        };
        Self {
            segment_size,
            overlap_size,
            search_range,
            stretch_ratio,
            planner: FftPlanner::new(),
            fft_ref_buf: Vec::new(),
            fft_search_buf: Vec::new(),
            fft_corr_buf: Vec::new(),
            prefix_sq_buf: Vec::new(),
            output_buf: Vec::new(),
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

        // Take the reusable buffer out of self to avoid borrow conflicts
        // (find_best_position borrows &mut self while output is also needed)
        let mut output = std::mem::take(&mut self.output_buf);

        // Grow if needed, zero the portion we'll use; never shrink
        let estimated_output_len = target_output_len + self.segment_size * 2;
        if output.len() < estimated_output_len {
            output.resize(estimated_output_len, 0.0);
        } else {
            for s in &mut output[..estimated_output_len] {
                *s = 0.0;
            }
        }

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

        // Return a copy of the result; put the buffer back for reuse
        let final_len = actual_output_len.min(target_output_len);
        let result = output[..final_len].to_vec();
        self.output_buf = output;
        Ok(result)
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

        // Compute raw cross-correlation via FFT (results stored in self.fft_corr_buf)
        self.fft_cross_correlate(ref_signal, search_signal);

        // Compute reference energy (constant for all candidates)
        let ref_energy: f64 = ref_signal.iter().map(|&s| (s as f64) * (s as f64)).sum();
        if ref_energy < ENERGY_EPSILON {
            return search_start;
        }

        // Find best candidate using normalized correlation
        let num_candidates = actual_region_len.saturating_sub(overlap_len) + 1;

        // Reuse prefix_sq_buf for energy normalization
        self.prefix_sq_buf.clear();
        self.prefix_sq_buf.reserve(search_signal.len() + 1);
        self.prefix_sq_buf.push(0.0f64);
        let mut accum = 0.0f64;
        for &s in search_signal {
            accum += (s as f64) * (s as f64);
            self.prefix_sq_buf.push(accum);
        }

        let best_pos = find_best_candidate(
            &self.prefix_sq_buf,
            &self.fft_corr_buf,
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
    /// Uses pre-allocated buffers that grow as needed but never shrink,
    /// eliminating per-call heap allocations in the hot path.
    fn fft_cross_correlate(&mut self, ref_signal: &[f32], search_signal: &[f32]) {
        let conv_len = search_signal.len() + ref_signal.len() - 1;
        let fft_size = conv_len.next_power_of_two();

        let fft_fwd = self.planner.plan_fft_forward(fft_size);
        let fft_inv = self.planner.plan_fft_inverse(fft_size);

        // Resize and fill reusable buffers (grow-only, never shrink).
        // Zero-fill first, then copy signal data — avoids per-element branch
        // which inhibits auto-vectorization.
        self.fft_ref_buf.resize(fft_size, COMPLEX_ZERO);
        for slot in self.fft_ref_buf.iter_mut() {
            *slot = COMPLEX_ZERO;
        }
        for (slot, &s) in self.fft_ref_buf.iter_mut().zip(ref_signal.iter()) {
            *slot = Complex::new(s, 0.0);
        }

        self.fft_search_buf.resize(fft_size, COMPLEX_ZERO);
        for slot in self.fft_search_buf.iter_mut() {
            *slot = COMPLEX_ZERO;
        }
        for (slot, &s) in self.fft_search_buf.iter_mut().zip(search_signal.iter()) {
            *slot = Complex::new(s, 0.0);
        }

        // Forward FFT
        fft_fwd.process(&mut self.fft_ref_buf);
        fft_fwd.process(&mut self.fft_search_buf);

        // Multiply conj(Ref) * Search into corr_buf (index-based for auto-vectorization)
        self.fft_corr_buf.resize(fft_size, COMPLEX_ZERO);
        for i in 0..fft_size {
            self.fft_corr_buf[i] = self.fft_ref_buf[i].conj() * self.fft_search_buf[i];
        }

        // Inverse FFT in-place
        fft_inv.process(&mut self.fft_corr_buf);
    }

    /// Overlap-adds a segment from input into output with raised-cosine crossfade.
    ///
    /// Split into two separate loops (crossfade region vs copy region) so each
    /// loop body is branch-free and amenable to auto-vectorization.
    #[inline]
    fn overlap_add(&self, input: &[f32], output: &mut [f32], input_pos: usize, output_pos: usize) {
        let segment_end = (input_pos + self.segment_size).min(input.len());
        let segment_len = segment_end - input_pos;
        let out_avail = output.len().saturating_sub(output_pos);
        let len = segment_len.min(out_avail);

        let overlap_len = self.overlap_size.min(len);
        let inv_overlap = 1.0 / self.overlap_size as f32;

        // Crossfade region: raised-cosine fade for smoother transitions
        for i in 0..overlap_len {
            let t = i as f32 * inv_overlap;
            let fade_in = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
            let fade_out = 1.0 - fade_in;
            output[output_pos + i] =
                output[output_pos + i] * fade_out + input[input_pos + i] * fade_in;
        }

        // Non-overlap region: direct copy
        let copy_start = overlap_len;
        output[output_pos + copy_start..output_pos + len]
            .copy_from_slice(&input[input_pos + copy_start..input_pos + len]);
    }
}

/// Finds the best correlation candidate using prefix-sum energy normalization.
///
/// Scans `num_candidates` lag positions in `corr_buf`, normalizing each by
/// the windowed energy (via pre-computed `prefix_sq`) and the reference energy.
fn find_best_candidate(
    prefix_sq: &[f64],
    corr_buf: &[Complex<f32>],
    ref_energy: f64,
    num_candidates: usize,
    overlap_len: usize,
    search_start: usize,
) -> usize {
    let norm = 1.0 / corr_buf.len() as f64;

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
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let (sum_ab, sum_a2, sum_b2) =
        a.iter()
            .zip(b.iter())
            .fold((0.0f64, 0.0f64, 0.0f64), |(ab, a2, b2), (&va, &vb)| {
                let va = va as f64;
                let vb = vb as f64;
                (ab + va * vb, a2 + va * va, b2 + vb * vb)
            });

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

    // --- normalized_cross_correlation edge cases ---

    #[test]
    fn test_ncc_zero_energy_signals() {
        // Both zero → denom < ENERGY_EPSILON → returns 0.0
        let a = vec![0.0f32; 8];
        let b = vec![0.0f32; 8];
        assert!((normalized_cross_correlation(&a, &b)).abs() < 1e-10);
    }

    #[test]
    fn test_ncc_one_zero_one_nonzero() {
        // One signal zero, one non-zero → denom < ENERGY_EPSILON → 0.0
        let a = vec![0.0f32; 4];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        assert!((normalized_cross_correlation(&a, &b)).abs() < 1e-10);
    }

    #[test]
    fn test_ncc_orthogonal_signals() {
        // Sine and cosine over one period are orthogonal
        let n = 128;
        let a: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * i as f32 / n as f32).sin())
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * i as f32 / n as f32).cos())
            .collect();
        let c = normalized_cross_correlation(&a, &b);
        assert!(
            c.abs() < 0.1,
            "Orthogonal signals should have near-zero correlation, got {}",
            c
        );
    }

    #[test]
    fn test_ncc_empty_input() {
        let c = normalized_cross_correlation(&[], &[]);
        assert!((c).abs() < 1e-10);
    }

    #[test]
    fn test_ncc_mismatched_lengths() {
        // Uses min of two lengths
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let c = normalized_cross_correlation(&a, &b);
        // Should correlate [1.0,2.0] with [1.0,2.0] = 1.0
        assert!(
            (c - 1.0).abs() < 1e-6,
            "Truncated correlation should be 1.0, got {}",
            c
        );
    }

    // --- FFT cross-correlation ---

    #[test]
    fn test_fft_cross_correlate_self_correlation() {
        // Self-correlation via FFT should have max at lag 0
        let mut wsola = Wsola::new(100, 50, 1.0);
        let signal: Vec<f32> = (0..64)
            .map(|i| (2.0 * PI * 4.0 * i as f32 / 64.0).sin())
            .collect();

        wsola.fft_cross_correlate(&signal, &signal);
        // Lag 0 should have the highest real value
        let max_lag = wsola
            .fft_corr_buf
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.re.partial_cmp(&b.1.re).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_lag, 0, "Self-correlation peak should be at lag 0");
    }

    #[test]
    fn test_fft_cross_correlate_shifted_signal() {
        // Shift signal by known amount; peak should appear at that lag
        let mut wsola = Wsola::new(100, 50, 1.0);
        let n = 128;
        let shift = 10;
        let ref_sig: Vec<f32> = (0..64)
            .map(|i| (2.0 * PI * 3.0 * i as f32 / 64.0).sin())
            .collect();
        // Search signal: ref_sig shifted right by `shift` samples
        let mut search = vec![0.0f32; n];
        for (i, &v) in ref_sig.iter().enumerate() {
            if i + shift < n {
                search[i + shift] = v;
            }
        }

        wsola.fft_cross_correlate(&ref_sig, &search);
        let norm = 1.0 / wsola.fft_corr_buf.len() as f32;
        // Peak should be at or near `shift`
        let best_lag = (0..wsola.fft_corr_buf.len())
            .max_by(|&a, &b| {
                (wsola.fft_corr_buf[a].re * norm)
                    .partial_cmp(&(wsola.fft_corr_buf[b].re * norm))
                    .unwrap()
            })
            .unwrap();
        assert!(
            (best_lag as i64 - shift as i64).unsigned_abs() <= 2,
            "Expected peak near lag {}, got {}",
            shift,
            best_lag
        );
    }

    // --- find_best_candidate ---

    /// Helper to compute prefix sum of squared values for energy normalization.
    fn compute_prefix_sq(signal: &[f32]) -> Vec<f64> {
        let mut prefix_sq = Vec::with_capacity(signal.len() + 1);
        prefix_sq.push(0.0f64);
        let mut accum = 0.0f64;
        for &s in signal {
            accum += (s as f64) * (s as f64);
            prefix_sq.push(accum);
        }
        prefix_sq
    }

    #[test]
    fn test_find_best_candidate_identical_signals() {
        // When search_signal starts with ref, best candidate should be at position 0
        let ref_signal = vec![1.0f32, 0.5, -0.3, 0.8];
        let overlap_len = ref_signal.len();
        // Build a search_signal that starts with ref_signal
        let mut search_signal = ref_signal.clone();
        search_signal.extend_from_slice(&[0.0; 8]); // padding

        let mut wsola = Wsola::new(100, 50, 1.0);
        wsola.fft_cross_correlate(&ref_signal, &search_signal);
        let ref_energy: f64 = ref_signal.iter().map(|&s| (s as f64) * (s as f64)).sum();
        let num_candidates = search_signal.len() - overlap_len + 1;
        let prefix_sq = compute_prefix_sq(&search_signal);

        let best = find_best_candidate(
            &prefix_sq,
            &wsola.fft_corr_buf,
            ref_energy,
            num_candidates,
            overlap_len,
            0, // search_start
        );

        assert_eq!(
            best, 0,
            "Best candidate should be at position 0 (exact match)"
        );
    }

    #[test]
    fn test_find_best_candidate_zero_energy_search() {
        // All zero search signal → all ncorr = 0.0 → first candidate
        let ref_signal = vec![1.0f32, 2.0, 3.0];
        let search_signal = vec![0.0f32; 16];
        let overlap_len = ref_signal.len();

        let mut wsola = Wsola::new(100, 50, 1.0);
        wsola.fft_cross_correlate(&ref_signal, &search_signal);
        let ref_energy: f64 = ref_signal.iter().map(|&s| (s as f64) * (s as f64)).sum();
        let num_candidates = search_signal.len() - overlap_len + 1;
        let prefix_sq = compute_prefix_sq(&search_signal);

        let best = find_best_candidate(
            &prefix_sq,
            &wsola.fft_corr_buf,
            ref_energy,
            num_candidates,
            overlap_len,
            100, // search_start=100
        );

        // With all zero energy, ncorr=0.0 for all candidates. First one wins.
        assert_eq!(best, 100);
    }

    // --- FFT vs direct threshold boundary ---

    #[test]
    fn test_wsola_fft_threshold_boundary() {
        // Create a scenario that exercises the FFT path by using a large search range.
        // FFT_CANDIDATE_THRESHOLD=64, so search_range > 32 should produce >64 candidates.
        let sample_rate = 44100usize;
        let segment_size = 882;
        // search_range = 400: nominal_pos ± 400 → ~800 candidates (> 64 threshold)
        let search_range = 400;

        let input: Vec<f32> = (0..sample_rate * 2)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut wsola = Wsola::new(segment_size, search_range, 1.5);
        let output = wsola.process(&input).unwrap();

        // Should produce valid stretched output using the FFT path
        assert!(!output.is_empty());
        assert!(output.iter().all(|s| s.is_finite()));
        let ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (ratio - 1.5).abs() < 0.1,
            "Large search range ratio {} too far from 1.5",
            ratio
        );
    }

    #[test]
    fn test_wsola_direct_path_small_search_range() {
        // Small search_range → fewer than FFT_CANDIDATE_THRESHOLD candidates → direct path
        let sample_rate = 44100usize;
        let segment_size = 882;
        let search_range = 20; // ~40 candidates (< 64 threshold)

        let input: Vec<f32> = (0..sample_rate * 2)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut wsola = Wsola::new(segment_size, search_range, 1.5);
        let output = wsola.process(&input).unwrap();

        assert!(!output.is_empty());
        assert!(output.iter().all(|s| s.is_finite()));
        let ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (ratio - 1.5).abs() < 0.15,
            "Small search range ratio {} too far from 1.5",
            ratio
        );
    }

    // --- overlap_add crossfade ---

    #[test]
    fn test_overlap_add_crossfade_raised_cosine() {
        // Verify the overlap region uses a raised-cosine crossfade.
        // Use ratio 2.0 to get the standard segment_size/2 overlap.
        let segment_size = 100;
        let wsola = Wsola::new(segment_size, 10, 2.0);
        let overlap_size = wsola.overlap_size;
        assert_eq!(overlap_size, 50);

        // Pre-fill output with 1.0
        let mut output = vec![1.0f32; 200];
        // Input segment is all 0.0
        let input = vec![0.0f32; 200];

        wsola.overlap_add(&input, &mut output, 0, 50);

        // In the overlap region:
        //   t = i/overlap_size
        //   fade_in = 0.5 * (1 - cos(PI * t))
        //   fade_out = 1 - fade_in
        //   output = 1.0 * fade_out + 0.0 * fade_in = fade_out
        for i in 0..overlap_size {
            let t = i as f32 / overlap_size as f32;
            let fade_in = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
            let expected = 1.0 - fade_in;
            assert!(
                (output[50 + i] - expected).abs() < 1e-5,
                "Overlap sample {}: expected {}, got {}",
                i,
                expected,
                output[50 + i]
            );
        }

        // After overlap, should be the input (0.0)
        for i in overlap_size..segment_size {
            assert!(
                (output[50 + i] - 0.0).abs() < 1e-5,
                "Post-overlap sample {}: expected 0.0, got {}",
                i,
                output[50 + i]
            );
        }
    }

    #[test]
    fn test_overlap_add_out_of_bounds_clamping() {
        // Segment extends beyond output buffer — should not panic
        let wsola = Wsola::new(100, 10, 1.0);
        let input = vec![0.5f32; 200];
        let mut output = vec![0.0f32; 60]; // Only 60 samples available
        wsola.overlap_add(&input, &mut output, 0, 10);
        // Should write up to output[59] without panicking
        assert!(output.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_overlap_add_input_truncated() {
        // Input shorter than segment_size — should handle gracefully
        let wsola = Wsola::new(100, 10, 1.0);
        let input = vec![0.5f32; 30]; // Only 30 samples
        let mut output = vec![0.0f32; 100];
        wsola.overlap_add(&input, &mut output, 0, 0);
        // Only first 30 samples should be written
        assert!((output[0] - 0.0).abs() < 1e-5); // fade_in at i=0 → 0*0.5 = 0
        assert!(output.iter().all(|s| s.is_finite()));
    }

    // --- WSOLA ratio too small ---

    #[test]
    fn test_wsola_ratio_too_small_for_segment() {
        // stretch_ratio so small that advance_output < 1.0 → InvalidRatio error
        let mut wsola = Wsola::new(882, 441, 0.001);
        let input = vec![0.0f32; 4410];
        let result = wsola.process(&input);
        assert!(result.is_err(), "Extremely small ratio should return error");
    }
}

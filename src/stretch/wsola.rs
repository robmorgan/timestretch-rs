//! WSOLA (Waveform Similarity Overlap-Add) time stretching.

use crate::error::StretchError;

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
        }
    }

    /// Stretches a mono audio signal using WSOLA.
    pub fn process(&self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        if input.len() < self.segment_size {
            return Err(StretchError::InputTooShort {
                provided: input.len(),
                minimum: self.segment_size,
            });
        }

        let advance_input = self.segment_size - self.overlap_size;
        let advance_output = (advance_input as f64 * self.stretch_ratio).round() as usize;

        if advance_output == 0 {
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
        let mut output_pos = advance_output;
        let mut actual_output_len = first_len;

        while (input_pos as usize) + self.segment_size <= input.len() {
            let nominal_pos = input_pos as usize;

            // Search for best matching position around nominal position
            let best_pos = self.find_best_position(
                input,
                &output,
                nominal_pos,
                output_pos,
            );

            // Overlap-add with cross-fade
            if output_pos + self.segment_size <= output.len() {
                self.overlap_add(
                    input,
                    &mut output,
                    best_pos,
                    output_pos,
                );
                actual_output_len =
                    (output_pos + self.segment_size).max(actual_output_len);
            }

            input_pos += advance_input as f64;
            output_pos += advance_output;
        }

        // Trim output to target length for better ratio accuracy
        let final_len = actual_output_len.min(target_output_len + self.overlap_size);
        output.truncate(final_len);
        Ok(output)
    }

    /// Finds the best matching position within the search range.
    fn find_best_position(
        &self,
        input: &[f32],
        output: &[f32],
        nominal_pos: usize,
        output_pos: usize,
    ) -> usize {
        let search_start = nominal_pos.saturating_sub(self.search_range);
        let search_end = (nominal_pos + self.search_range)
            .min(input.len().saturating_sub(self.segment_size));

        if search_start >= search_end {
            return nominal_pos.min(input.len().saturating_sub(self.segment_size));
        }

        let mut best_pos = nominal_pos;
        let mut best_corr = f64::NEG_INFINITY;

        // Compare the overlap region of the candidate against what's already in output
        let overlap_len = self.overlap_size.min(output.len().saturating_sub(output_pos));
        if overlap_len == 0 {
            return nominal_pos;
        }

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

    /// Overlap-adds a segment from input into output with raised-cosine crossfade.
    fn overlap_add(
        &self,
        input: &[f32],
        output: &mut [f32],
        input_pos: usize,
        output_pos: usize,
    ) {
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
    if denom < 1e-12 {
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

        let wsola = Wsola::new(segment_size, search_range, 1.0);
        let output = wsola.process(&input).unwrap();

        // Length should be approximately the same
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.0).abs() < 0.15,
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

        let wsola = Wsola::new(segment_size, search_range, 2.0);
        let output = wsola.process(&input).unwrap();

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 2.0).abs() < 0.3,
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

        let wsola = Wsola::new(segment_size, search_range, 0.75);
        let output = wsola.process(&input).unwrap();

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 0.75).abs() < 0.15,
            "Length ratio {} too far from 0.75",
            len_ratio
        );

        // Test 0.5 ratio with accuracy check
        let wsola_half = Wsola::new(segment_size, search_range, 0.5);
        let output_half = wsola_half.process(&input).unwrap();
        let half_ratio = output_half.len() as f64 / input.len() as f64;
        assert!(
            (half_ratio - 0.5).abs() < 0.1,
            "Half ratio {} too far from 0.5",
            half_ratio
        );
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
            let wsola = Wsola::new(segment_size, search_range, ratio);
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
        let wsola = Wsola::new(882, 441, 1.0);
        let result = wsola.process(&[0.0; 100]);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalized_cross_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = normalized_cross_correlation(&a, &b);
        assert!((c - 1.0).abs() < 1e-6, "Self-correlation should be 1.0, got {}", c);

        let neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let c_neg = normalized_cross_correlation(&a, &neg);
        assert!((c_neg - (-1.0)).abs() < 1e-6, "Negated correlation should be -1.0, got {}", c_neg);
    }
}

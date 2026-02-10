/// WSOLA (Waveform Similarity Overlap-Add) time stretching.
///
/// Preserves transient quality better than phase vocoder by operating
/// directly on the time-domain waveform.
pub struct Wsola {
    segment_size: usize,
    overlap_size: usize,
    search_range: usize,
    stretch_ratio: f64,
}

impl Wsola {
    /// Create a new WSOLA processor.
    pub fn new(segment_size: usize, overlap_ratio: f32, search_range: usize, stretch_ratio: f64) -> Self {
        let overlap_size = (segment_size as f32 * overlap_ratio) as usize;
        Self {
            segment_size,
            overlap_size,
            search_range,
            stretch_ratio,
        }
    }

    /// Process mono audio samples.
    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        if input.len() < self.segment_size {
            // Input shorter than one segment, just return it scaled
            let out_len = (input.len() as f64 * self.stretch_ratio) as usize;
            return crate::core::resample::resample_linear(input, out_len.max(1));
        }

        let advance = self.segment_size - self.overlap_size;
        if advance == 0 {
            return input.to_vec();
        }

        let output_advance = (advance as f64 * self.stretch_ratio) as usize;
        let output_advance = output_advance.max(1);

        let expected_output_len = (input.len() as f64 * self.stretch_ratio) as usize;
        let mut output = Vec::with_capacity(expected_output_len + self.segment_size);

        // First segment: copy directly
        let first_end = self.segment_size.min(input.len());
        output.extend_from_slice(&input[..first_end]);

        let mut input_pos = advance;
        let mut output_pos = output_advance;

        while input_pos + self.segment_size <= input.len() {
            // Find best matching position within search range
            let best_offset = self.find_best_match(input, input_pos, &output, output_pos);
            let actual_pos = (input_pos as isize + best_offset) as usize;

            if actual_pos + self.segment_size > input.len() {
                break;
            }

            // Overlap-add with crossfade
            let segment = &input[actual_pos..actual_pos + self.segment_size];
            let overlap_start = output_pos;

            // Ensure output is long enough
            let needed = output_pos + self.segment_size;
            if output.len() < needed {
                output.resize(needed, 0.0);
            }

            // Crossfade the overlap region
            for i in 0..self.overlap_size.min(output.len().saturating_sub(overlap_start)) {
                let idx = overlap_start + i;
                if idx < output.len() {
                    let t = i as f32 / self.overlap_size as f32;
                    output[idx] = output[idx] * (1.0 - t) + segment[i] * t;
                }
            }

            // Copy non-overlapping part
            for i in self.overlap_size..self.segment_size {
                let idx = overlap_start + i;
                if idx < output.len() {
                    output[idx] = segment[i];
                } else {
                    output.push(segment[i]);
                }
            }

            input_pos = actual_pos + advance;
            output_pos += output_advance;
        }

        // Trim to expected length
        output.truncate(expected_output_len);
        output
    }

    /// Find the best matching offset using normalized cross-correlation.
    fn find_best_match(&self, input: &[f32], input_pos: usize, output: &[f32], output_pos: usize) -> isize {
        let search = self.search_range as isize;
        let mut best_offset: isize = 0;
        let mut best_corr = f32::NEG_INFINITY;

        let compare_len = self.overlap_size.min(256); // Limit correlation length for speed

        for offset in -search..=search {
            let pos = input_pos as isize + offset;
            if pos < 0 || pos as usize + compare_len > input.len() {
                continue;
            }
            if output_pos + compare_len > output.len() {
                continue;
            }

            let corr = normalized_cross_correlation(
                &input[pos as usize..pos as usize + compare_len],
                &output[output_pos..output_pos + compare_len],
            );

            if corr > best_corr {
                best_corr = corr;
                best_offset = offset;
            }
        }

        best_offset
    }
}

/// Compute normalized cross-correlation between two signals.
fn normalized_cross_correlation(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut sum_ab = 0.0f64;
    let mut sum_aa = 0.0f64;
    let mut sum_bb = 0.0f64;

    for i in 0..len {
        let av = a[i] as f64;
        let bv = b[i] as f64;
        sum_ab += av * bv;
        sum_aa += av * av;
        sum_bb += bv * bv;
    }

    let denom = (sum_aa * sum_bb).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    (sum_ab / denom) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn generate_sine(freq: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_wsola_identity() {
        let input = generate_sine(440.0, 44100, 44100);
        let wsola = Wsola::new(960, 0.5, 441, 1.0);
        let output = wsola.process(&input);

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.0).abs() < 0.05,
            "Identity WSOLA should preserve length, got ratio {len_ratio}"
        );
    }

    #[test]
    fn test_wsola_stretch() {
        let input = generate_sine(440.0, 44100, 44100);
        let wsola = Wsola::new(960, 0.5, 441, 2.0);
        let output = wsola.process(&input);

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 2.0).abs() < 0.1,
            "2x WSOLA should ~double length, got ratio {len_ratio}"
        );
    }

    #[test]
    fn test_wsola_compress() {
        let input = generate_sine(440.0, 44100, 44100);
        let wsola = Wsola::new(960, 0.5, 441, 0.5);
        let output = wsola.process(&input);

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 0.5).abs() < 0.1,
            "0.5x WSOLA should ~halve length, got ratio {len_ratio}"
        );
    }

    #[test]
    fn test_wsola_empty() {
        let wsola = Wsola::new(960, 0.5, 441, 1.5);
        let output = wsola.process(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn test_wsola_short_input() {
        let input = vec![0.5; 100]; // Shorter than segment size
        let wsola = Wsola::new(960, 0.5, 441, 1.5);
        let output = wsola.process(&input);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_wsola_silence() {
        let input = vec![0.0; 44100];
        let wsola = Wsola::new(960, 0.5, 441, 1.5);
        let output = wsola.process(&input);
        let max_val = output.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(
            max_val < 1e-6,
            "WSOLA on silence should produce silence, max={max_val}"
        );
    }

    #[test]
    fn test_normalized_cross_correlation_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = normalized_cross_correlation(&a, &a);
        assert!(
            (corr - 1.0).abs() < 1e-5,
            "Self-correlation should be 1.0, got {corr}"
        );
    }

    #[test]
    fn test_normalized_cross_correlation_opposite() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        let corr = normalized_cross_correlation(&a, &b);
        assert!(
            (corr + 1.0).abs() < 1e-5,
            "Opposite signals should have -1.0 correlation, got {corr}"
        );
    }

    #[test]
    fn test_normalized_cross_correlation_zeros() {
        let a = vec![0.0; 5];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = normalized_cross_correlation(&a, &b);
        assert_eq!(corr, 0.0, "Correlation with zeros should be 0");
    }
}

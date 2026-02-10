use crate::analysis::transient::detect_transients;
use crate::stretch::params::AlgorithmParams;
use crate::stretch::phase_vocoder::PhaseVocoder;
use crate::stretch::wsola::Wsola;

/// Hybrid time stretcher that uses WSOLA for transient regions
/// and phase vocoder for tonal regions.
pub struct HybridStretcher {
    params: AlgorithmParams,
}

impl HybridStretcher {
    /// Create a new hybrid stretcher.
    pub fn new(params: AlgorithmParams) -> Self {
        Self { params }
    }

    /// Process mono audio samples using the hybrid algorithm.
    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        if !self.params.use_hybrid {
            // Pure phase vocoder mode
            let pv = PhaseVocoder::new(
                self.params.fft_size,
                self.params.hop_size,
                self.params.stretch_ratio,
                self.params.sample_rate,
                self.params.sub_bass_cutoff,
            );
            return pv.process(input);
        }

        // Detect transients
        let markers = detect_transients(
            input,
            self.params.sample_rate,
            self.params.transient_sensitivity,
            self.params.fft_size.min(2048),
        );

        if markers.positions.is_empty() {
            // No transients detected: use pure phase vocoder
            let pv = PhaseVocoder::new(
                self.params.fft_size,
                self.params.hop_size,
                self.params.stretch_ratio,
                self.params.sample_rate,
                self.params.sub_bass_cutoff,
            );
            return pv.process(input);
        }

        // Build segments: classify each as transient or tonal
        let segments = self.build_segments(input.len(), &markers.positions);

        let expected_output_len = (input.len() as f64 * self.params.stretch_ratio) as usize;
        let mut output = Vec::with_capacity(expected_output_len);

        let pv = PhaseVocoder::new(
            self.params.fft_size,
            self.params.hop_size,
            self.params.stretch_ratio,
            self.params.sample_rate,
            self.params.sub_bass_cutoff,
        );

        let wsola = Wsola::new(
            self.params.wsola_segment_size,
            self.params.wsola_overlap,
            self.params.wsola_search_range,
            self.params.stretch_ratio,
        );

        for segment in &segments {
            let seg_input = &input[segment.start..segment.end];
            let stretched = if segment.is_transient {
                wsola.process(seg_input)
            } else {
                pv.process(seg_input)
            };

            if output.is_empty() {
                output.extend_from_slice(&stretched);
            } else {
                // Crossfade between segments
                crossfade_append(&mut output, &stretched, self.params.crossfade_len);
            }
        }

        // Trim to expected length
        output.truncate(expected_output_len);
        output
    }
}

/// A segment of audio classified as transient or tonal.
#[derive(Debug)]
struct Segment {
    start: usize,
    end: usize,
    is_transient: bool,
}

impl HybridStretcher {
    /// Build segments from transient positions.
    /// Transient segments extend from a transient marker to a short window after it.
    /// Everything else is tonal.
    fn build_segments(&self, input_len: usize, transient_positions: &[usize]) -> Vec<Segment> {
        let transient_window = self.params.wsola_segment_size * 4; // ~80ms around transients
        let mut segments = Vec::new();
        let mut pos = 0;

        for &t_pos in transient_positions {
            let t_start = t_pos.saturating_sub(self.params.wsola_segment_size);
            let t_end = (t_pos + transient_window).min(input_len);

            if t_start > pos {
                // Tonal segment before this transient
                segments.push(Segment {
                    start: pos,
                    end: t_start,
                    is_transient: false,
                });
            }

            if t_end > pos.max(t_start) {
                // Transient segment
                segments.push(Segment {
                    start: pos.max(t_start),
                    end: t_end,
                    is_transient: true,
                });
            }

            pos = t_end;
        }

        // Remaining tonal segment
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

/// Append new audio to output with a raised-cosine crossfade.
fn crossfade_append(output: &mut Vec<f32>, new_data: &[f32], crossfade_len: usize) {
    if new_data.is_empty() {
        return;
    }
    let cf_len = crossfade_len.min(output.len()).min(new_data.len());
    if cf_len == 0 {
        output.extend_from_slice(new_data);
        return;
    }

    let out_start = output.len() - cf_len;
    for i in 0..cf_len {
        let t = (i as f32 + 0.5) / cf_len as f32;
        // Raised cosine crossfade
        let fade_out = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
        let fade_in = 1.0 - fade_out;
        output[out_start + i] = output[out_start + i] * fade_out + new_data[i] * fade_in;
    }

    if cf_len < new_data.len() {
        output.extend_from_slice(&new_data[cf_len..]);
    }
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
    fn test_hybrid_identity() {
        let params = AlgorithmParams::from_preset(None, 1.0, 44100, 4096, 1024, 0.5, true);
        let stretcher = HybridStretcher::new(params);
        let input = generate_sine(440.0, 44100, 44100);
        let output = stretcher.process(&input);

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.0).abs() < 0.15,
            "Hybrid identity should preserve length, ratio={len_ratio}"
        );
    }

    #[test]
    fn test_hybrid_stretch() {
        let params = AlgorithmParams::from_preset(None, 2.0, 44100, 4096, 1024, 0.5, true);
        let stretcher = HybridStretcher::new(params);
        let input = generate_sine(440.0, 44100, 44100);
        let output = stretcher.process(&input);

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 2.0).abs() < 0.5,
            "Hybrid 2x should ~double length, ratio={len_ratio}"
        );
    }

    #[test]
    fn test_hybrid_empty() {
        let params = AlgorithmParams::from_preset(None, 1.5, 44100, 4096, 1024, 0.5, true);
        let stretcher = HybridStretcher::new(params);
        assert!(stretcher.process(&[]).is_empty());
    }

    #[test]
    fn test_hybrid_pv_only_mode() {
        let params = AlgorithmParams::from_preset(Some(crate::core::types::EdmPreset::Ambient), 1.5, 44100, 8192, 2048, 0.3, false);
        let stretcher = HybridStretcher::new(params);
        let input = generate_sine(440.0, 44100, 44100);
        let output = stretcher.process(&input);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_crossfade_append_basic() {
        let mut output = vec![1.0; 100];
        let new_data = vec![0.0; 100];
        crossfade_append(&mut output, &new_data, 20);
        // After crossfade, the junction should be smooth
        assert_eq!(output.len(), 180); // 100 - 20 + 100
        // Check crossfade region is between 0 and 1
        for i in 80..100 {
            assert!(output[i] >= -0.1 && output[i] <= 1.1);
        }
    }

    #[test]
    fn test_crossfade_append_empty() {
        let mut output = vec![1.0; 10];
        crossfade_append(&mut output, &[], 5);
        assert_eq!(output.len(), 10);
    }
}

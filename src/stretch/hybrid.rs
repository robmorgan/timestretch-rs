//! Hybrid stretcher combining WSOLA (transients) with phase vocoder (tonal content).

use crate::analysis::transient::detect_transients;
use crate::core::types::StretchParams;
use crate::error::StretchError;
use crate::stretch::phase_vocoder::PhaseVocoder;
use crate::stretch::wsola::Wsola;

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
            let wsola = Wsola::new(
                input.len().min(self.params.wsola_segment_size),
                self.params.wsola_search_range.min(input.len() / 4),
                self.params.stretch_ratio,
            );
            return wsola.process(input);
        }

        // Step 1: Detect transients
        let transients = detect_transients(
            input,
            self.params.sample_rate,
            self.params.fft_size.min(2048), // Use smaller FFT for transient detection
            self.params.hop_size.min(512),
            self.params.transient_sensitivity,
        );

        // Step 2: Segment audio at transient boundaries
        let segments = self.segment_audio(input.len(), &transients.onsets);

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

        let crossfade_samples =
            (self.params.sample_rate as f64 * 0.005) as usize; // 5ms crossfade
        let output = concatenate_with_crossfade(&output_segments, crossfade_samples);

        Ok(output)
    }

    /// Stretches a single segment using the appropriate algorithm.
    ///
    /// Transient segments use WSOLA to preserve attack characteristics.
    /// Tonal segments use the phase vocoder for smooth stretching.
    /// Very short segments fall back to linear resampling.
    fn stretch_segment(
        &self,
        seg_data: &[f32],
        is_transient: bool,
        pv: &mut PhaseVocoder,
    ) -> Vec<f32> {
        if seg_data.len() < 256 {
            let out_len =
                (seg_data.len() as f64 * self.params.stretch_ratio).round() as usize;
            return crate::core::resample::resample_linear(seg_data, out_len.max(1));
        }

        let result = if is_transient {
            self.stretch_with_wsola(seg_data)
        } else if seg_data.len() >= self.params.fft_size {
            pv.process(seg_data)
        } else {
            self.stretch_with_wsola(seg_data)
        };

        result.unwrap_or_else(|_| {
            let out_len =
                (seg_data.len() as f64 * self.params.stretch_ratio).round() as usize;
            crate::core::resample::resample_linear(seg_data, out_len.max(1))
        })
    }

    /// Stretches a segment using WSOLA with clamped parameters.
    fn stretch_with_wsola(&self, seg_data: &[f32]) -> Result<Vec<f32>, StretchError> {
        let seg_size = self.params.wsola_segment_size.min(seg_data.len() / 2).max(64);
        let search = self.params.wsola_search_range.min(seg_size / 2).max(16);
        let wsola = Wsola::new(seg_size, search, self.params.stretch_ratio);
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
        // Transient region size: ~10ms around each onset
        let transient_size =
            (self.params.sample_rate as f64 * 0.010) as usize;

        let mut pos = 0;

        for &onset in onsets {
            if onset <= pos {
                continue;
            }

            // Tonal region before this onset
            if onset > pos {
                let tonal_end = onset.min(input_len);
                if tonal_end > pos {
                    segments.push(Segment {
                        start: pos,
                        end: tonal_end,
                        is_transient: false,
                    });
                }
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
}

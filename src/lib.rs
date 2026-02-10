#![forbid(unsafe_code)]
#![doc = "Pure Rust audio time stretching library optimized for EDM."]

pub mod analysis;
pub mod core;
pub mod error;
pub mod io;
pub mod stream;
pub mod stretch;

pub use core::types::{AudioBuffer, Channels, EdmPreset, Sample, StretchParams};
pub use error::StretchError;
pub use stream::StreamProcessor;

/// Stretches audio samples by the given parameters.
///
/// This is the main entry point for one-shot (non-streaming) time stretching.
/// For stereo input, provide interleaved L/R samples.
///
/// # Example
/// ```no_run
/// use timestretch::{StretchParams, EdmPreset};
///
/// let input = vec![0.0f32; 44100]; // 1 second of silence
/// let params = StretchParams::new(1.5)
///     .with_sample_rate(44100)
///     .with_channels(1)
///     .with_preset(EdmPreset::HouseLoop);
/// let output = timestretch::stretch(&input, &params).unwrap();
/// ```
pub fn stretch(input: &[f32], params: &StretchParams) -> Result<Vec<f32>, StretchError> {
    // Validate parameters
    stretch::params::validate_params(params)
        .map_err(StretchError::InvalidRatio)?;

    if input.is_empty() {
        return Ok(vec![]);
    }

    let num_channels = params.channels.count();

    // Process each channel separately
    let mut channel_outputs: Vec<Vec<f32>> = Vec::new();

    for ch in 0..num_channels {
        // Deinterleave
        let channel_data: Vec<f32> = input
            .iter()
            .skip(ch)
            .step_by(num_channels)
            .copied()
            .collect();

        // Use hybrid stretcher
        let stretcher = stretch::hybrid::HybridStretcher::new(params.clone());
        let stretched = stretcher.process(&channel_data)?;
        channel_outputs.push(stretched);
    }

    // Re-interleave
    if channel_outputs.is_empty() {
        return Ok(vec![]);
    }

    let min_len = channel_outputs.iter().map(|c| c.len()).min().unwrap_or(0);
    let mut output = Vec::with_capacity(min_len * num_channels);

    for i in 0..min_len {
        for ch_out in &channel_outputs {
            output.push(ch_out[i]);
        }
    }

    Ok(output)
}

/// Stretches an AudioBuffer and returns a new AudioBuffer.
pub fn stretch_buffer(
    buffer: &AudioBuffer,
    params: &StretchParams,
) -> Result<AudioBuffer, StretchError> {
    let mut effective_params = params.clone();
    effective_params.sample_rate = buffer.sample_rate;
    effective_params.channels = buffer.channels;

    let output_data = stretch(&buffer.data, &effective_params)?;

    Ok(AudioBuffer::new(
        output_data,
        buffer.sample_rate,
        buffer.channels,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stretch_empty() {
        let params = StretchParams::new(1.5);
        let output = stretch(&[], &params).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_stretch_mono_sine() {
        let sample_rate = 44100u32;
        let duration = 2.0;
        let num_samples = (sample_rate as f64 * duration) as usize;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = stretch(&input, &params).unwrap();
        assert!(!output.is_empty());

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.5).abs() < 0.5,
            "Length ratio {} too far from 1.5",
            len_ratio
        );
    }

    #[test]
    fn test_stretch_stereo() {
        let sample_rate = 44100u32;
        let num_frames = 44100;
        let mut input = vec![0.0f32; num_frames * 2];

        for i in 0..num_frames {
            let t = i as f32 / sample_rate as f32;
            input[i * 2] = (2.0 * std::f32::consts::PI * 440.0 * t).sin(); // L
            input[i * 2 + 1] = (2.0 * std::f32::consts::PI * 880.0 * t).sin(); // R
        }

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(2);

        let output = stretch(&input, &params).unwrap();
        assert!(!output.is_empty());
        // Output should have even number of samples (stereo)
        assert_eq!(output.len() % 2, 0);
    }

    #[test]
    fn test_stretch_invalid_ratio() {
        let params = StretchParams::new(0.0);
        assert!(stretch(&[0.0; 44100], &params).is_err());
    }

    #[test]
    fn test_stretch_buffer() {
        let buffer = AudioBuffer::from_mono(
            (0..44100)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
                .collect(),
            44100,
        );

        let params = StretchParams::new(1.5);
        let output = stretch_buffer(&buffer, &params).unwrap();
        assert_eq!(output.sample_rate, 44100);
        assert_eq!(output.channels, Channels::Mono);
        assert!(!output.data.is_empty());
    }

    #[test]
    fn test_stretch_dj_beatmatch_preset() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        // Small ratio change: 126 BPM -> 128 BPM
        let ratio = 126.0 / 128.0; // ~0.984
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(EdmPreset::DjBeatmatch);

        let output = stretch(&input, &params).unwrap();
        assert!(!output.is_empty());
    }
}

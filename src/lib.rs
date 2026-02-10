#![forbid(unsafe_code)]
//! Pure Rust audio time stretching library optimized for electronic dance music.
//!
//! `timestretch` changes the tempo of audio without altering its pitch, using a
//! hybrid algorithm that combines WSOLA (for transients) with a phase vocoder
//! (for tonal content). It ships with five EDM-tuned presets and a streaming API
//! for real-time use.
//!
//! # Quick Start
//!
//! ```
//! use timestretch::{StretchParams, EdmPreset};
//!
//! // 1 second of 440 Hz sine at 44.1 kHz
//! let input: Vec<f32> = (0..44100)
//!     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
//!     .collect();
//!
//! let params = StretchParams::new(1.5)
//!     .with_sample_rate(44100)
//!     .with_channels(1)
//!     .with_preset(EdmPreset::HouseLoop);
//!
//! let output = timestretch::stretch(&input, &params).unwrap();
//! assert!(output.len() > input.len()); // ~1.5x longer
//! ```
//!
//! # Streaming
//!
//! For real-time use, feed audio in chunks via [`StreamProcessor`]:
//!
//! ```
//! use timestretch::{StreamProcessor, StretchParams, EdmPreset};
//!
//! let params = StretchParams::new(1.0)
//!     .with_preset(EdmPreset::DjBeatmatch)
//!     .with_sample_rate(44100)
//!     .with_channels(1);
//!
//! let mut processor = StreamProcessor::new(params);
//! // processor.process(&chunk) for each audio buffer
//! // processor.set_stretch_ratio(1.05) to change on the fly
//! ```

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
/// # Errors
///
/// Returns [`StretchError::InvalidRatio`] if the stretch ratio is out of range
/// (must be between 0.01 and 100.0).
///
/// # Example
///
/// ```
/// use timestretch::{StretchParams, EdmPreset};
///
/// let input: Vec<f32> = (0..44100)
///     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
///     .collect();
///
/// let params = StretchParams::new(1.5)
///     .with_sample_rate(44100)
///     .with_channels(1)
///     .with_preset(EdmPreset::HouseLoop);
/// let output = timestretch::stretch(&input, &params).unwrap();
/// ```
pub fn stretch(input: &[f32], params: &StretchParams) -> Result<Vec<f32>, StretchError> {
    // Validate parameters
    stretch::params::validate_params(params).map_err(StretchError::InvalidRatio)?;

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

/// Stretches an [`AudioBuffer`] and returns a new `AudioBuffer`.
///
/// The sample rate and channel layout are taken from the input buffer,
/// overriding whatever is set in `params`.
///
/// # Errors
///
/// Returns [`StretchError::InvalidRatio`] if the stretch ratio is out of range.
///
/// # Example
///
/// ```
/// use timestretch::{AudioBuffer, StretchParams, EdmPreset};
///
/// let buffer = AudioBuffer::from_mono(
///     (0..44100)
///         .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
///         .collect(),
///     44100,
/// );
/// let params = StretchParams::new(1.5).with_preset(EdmPreset::HouseLoop);
/// let output = timestretch::stretch_buffer(&buffer, &params).unwrap();
/// assert_eq!(output.sample_rate, 44100);
/// ```
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

/// Shifts the pitch of audio without changing its duration.
///
/// `pitch_factor` > 1.0 raises the pitch; < 1.0 lowers it. For example,
/// `pitch_factor = 2.0` raises the pitch by one octave. This works by
/// time-stretching the audio and then resampling back to the original length
/// using cubic interpolation.
///
/// # Errors
///
/// Returns [`StretchError::InvalidRatio`] if the pitch factor is out of range.
///
/// # Example
///
/// ```
/// use timestretch::StretchParams;
///
/// let input: Vec<f32> = (0..44100)
///     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
///     .collect();
///
/// let params = StretchParams::new(1.0) // ratio is overridden internally
///     .with_sample_rate(44100)
///     .with_channels(1);
/// let output = timestretch::pitch_shift(&input, &params, 1.5).unwrap();
/// // Output has the same length but higher pitch
/// ```
pub fn pitch_shift(
    input: &[f32],
    params: &StretchParams,
    pitch_factor: f64,
) -> Result<Vec<f32>, StretchError> {
    if !(0.01..=100.0).contains(&pitch_factor) {
        return Err(StretchError::InvalidRatio(format!(
            "Pitch factor must be between 0.01 and 100.0, got {}",
            pitch_factor
        )));
    }

    if input.is_empty() {
        return Ok(vec![]);
    }

    // Step 1: Time-stretch by 1/pitch_factor to compensate for the resampling
    let stretch_ratio = 1.0 / pitch_factor;
    let mut stretch_params = params.clone();
    stretch_params.stretch_ratio = stretch_ratio;
    let stretched = stretch(input, &stretch_params)?;

    if stretched.is_empty() {
        return Ok(vec![]);
    }

    // Step 2: Resample each channel to original length using cubic interpolation
    let num_channels = params.channels.count();
    let num_input_frames = input.len() / num_channels;

    let mut channel_outputs: Vec<Vec<f32>> = Vec::with_capacity(num_channels);
    for ch in 0..num_channels {
        let channel_data: Vec<f32> = stretched
            .iter()
            .skip(ch)
            .step_by(num_channels)
            .copied()
            .collect();
        let resampled = core::resample::resample_cubic(&channel_data, num_input_frames);
        channel_outputs.push(resampled);
    }

    // Re-interleave
    let mut output = Vec::with_capacity(num_input_frames * num_channels);
    for i in 0..num_input_frames {
        for ch_out in &channel_outputs {
            if i < ch_out.len() {
                output.push(ch_out[i]);
            } else {
                output.push(0.0);
            }
        }
    }

    Ok(output)
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
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
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
    fn test_pitch_shift_preserves_length() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = pitch_shift(&input, &params, 1.5).unwrap();
        // Output should have the same length as input
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_pitch_shift_empty() {
        let params = StretchParams::new(1.0);
        let output = pitch_shift(&[], &params, 1.5).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_pitch_shift_invalid_factor() {
        let params = StretchParams::new(1.0);
        assert!(pitch_shift(&[0.0; 44100], &params, 0.0).is_err());
        assert!(pitch_shift(&[0.0; 44100], &params, -1.0).is_err());
        assert!(pitch_shift(&[0.0; 44100], &params, 200.0).is_err());
    }

    #[test]
    fn test_pitch_shift_stereo() {
        let sample_rate = 44100u32;
        let num_frames = 44100;
        let mut input = vec![0.0f32; num_frames * 2];
        for i in 0..num_frames {
            let t = i as f32 / sample_rate as f32;
            input[i * 2] = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            input[i * 2 + 1] = (2.0 * std::f32::consts::PI * 880.0 * t).sin();
        }

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(2);

        let output = pitch_shift(&input, &params, 0.8).unwrap();
        assert_eq!(output.len(), input.len());
        assert_eq!(output.len() % 2, 0);
    }

    #[test]
    fn test_stretch_dj_beatmatch_preset() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
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

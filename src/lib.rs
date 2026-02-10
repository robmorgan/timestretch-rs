#![forbid(unsafe_code)]
#![doc = "Pure Rust audio time stretching library optimized for electronic dance music."]

pub mod analysis;
pub mod core;
pub mod io;
pub mod stream;
pub mod stretch;

pub use core::types::{AudioBuffer, EdmPreset, Frame, Sample, StretchError, StretchParams};
pub use stream::processor::StreamProcessor;

/// Stretch audio by the given parameters.
///
/// This is the main entry point for one-shot (non-streaming) time stretching.
///
/// # Arguments
/// * `input` - Input audio samples (interleaved if stereo)
/// * `params` - Stretch parameters controlling ratio, quality, and algorithm selection
///
/// # Returns
/// Stretched audio samples.
///
/// # Errors
/// Returns `StretchError` if parameters are invalid.
///
/// # Example
/// ```
/// use timestretch::{StretchParams, EdmPreset};
///
/// let input = vec![0.0f32; 44100]; // 1 second of silence
/// let params = StretchParams::new(1.5).unwrap()
///     .with_preset(EdmPreset::HouseLoop)
///     .with_sample_rate(44100)
///     .with_channels(1);
/// let output = timestretch::stretch(&input, &params).unwrap();
/// ```
pub fn stretch(input: &[f32], params: &StretchParams) -> Result<Vec<f32>, StretchError> {
    params.validate()?;

    if input.is_empty() {
        return Ok(Vec::new());
    }

    if params.channels == 1 {
        stretch_mono(input, params)
    } else {
        stretch_stereo(input, params)
    }
}

fn stretch_mono(input: &[f32], params: &StretchParams) -> Result<Vec<f32>, StretchError> {
    let algo_params = stretch::params::AlgorithmParams::from_preset(
        params.preset,
        params.stretch_ratio,
        params.sample_rate,
        params.fft_size,
        params.effective_hop_size(),
        params.transient_sensitivity,
        params.use_hybrid,
    );

    let stretcher = stretch::hybrid::HybridStretcher::new(algo_params);
    Ok(stretcher.process(input))
}

fn stretch_stereo(input: &[f32], params: &StretchParams) -> Result<Vec<f32>, StretchError> {
    if input.len() % 2 != 0 {
        return Err(StretchError::InvalidInput(
            "Stereo input must have an even number of samples".to_string(),
        ));
    }

    // Deinterleave
    let num_frames = input.len() / 2;
    let mut left = Vec::with_capacity(num_frames);
    let mut right = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        left.push(input[i * 2]);
        right.push(input[i * 2 + 1]);
    }

    // Stretch each channel
    let mono_params = StretchParams {
        channels: 1,
        ..params.clone()
    };
    let left_out = stretch_mono(&left, &mono_params)?;
    let right_out = stretch_mono(&right, &mono_params)?;

    // Reinterleave
    let out_frames = left_out.len().min(right_out.len());
    let mut output = Vec::with_capacity(out_frames * 2);
    for i in 0..out_frames {
        output.push(left_out[i]);
        output.push(right_out[i]);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stretch_empty() {
        let params = StretchParams::new(1.5).unwrap();
        let output = stretch(&[], &params).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_stretch_invalid_params() {
        let mut params = StretchParams::new(1.0).unwrap();
        params.fft_size = 1000;
        assert!(stretch(&[0.0; 100], &params).is_err());
    }

    #[test]
    fn test_stretch_mono_sine() {
        let sample_rate = 44100;
        let input: Vec<f32> = (0..44100)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let params = StretchParams::new(1.5)
            .unwrap()
            .with_sample_rate(sample_rate as u32)
            .with_channels(1);
        let output = stretch(&input, &params).unwrap();

        assert!(!output.is_empty(), "Output should not be empty");
    }

    #[test]
    fn test_stretch_stereo() {
        let sample_rate = 44100u32;
        let num_frames = 44100;
        let mut input = Vec::with_capacity(num_frames * 2);
        for i in 0..num_frames {
            let l =
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin();
            let r =
                (2.0 * std::f32::consts::PI * 880.0 * i as f32 / sample_rate as f32).sin();
            input.push(l);
            input.push(r);
        }

        let params = StretchParams::new(1.2)
            .unwrap()
            .with_sample_rate(sample_rate)
            .with_channels(2);
        let output = stretch(&input, &params).unwrap();
        assert_eq!(output.len() % 2, 0, "Stereo output must have even length");
    }

    #[test]
    fn test_stretch_stereo_odd_input() {
        let params = StretchParams::new(1.0).unwrap().with_channels(2);
        assert!(stretch(&[0.0; 3], &params).is_err());
    }
}

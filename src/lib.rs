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

pub use analysis::beat::BeatGrid;
pub use core::types::{AudioBuffer, Channels, EdmPreset, FrameIter, Sample, StretchParams};
pub use core::window::WindowType;
pub use error::StretchError;
pub use stream::StreamProcessor;

/// Creates params adjusted for the given buffer's sample rate and channels,
/// then wraps the processing result in a new AudioBuffer.
fn process_buffer(
    buffer: &AudioBuffer,
    params: &StretchParams,
    process_fn: impl FnOnce(&[f32], &StretchParams) -> Result<Vec<f32>, StretchError>,
) -> Result<AudioBuffer, StretchError> {
    let mut effective_params = params.clone();
    effective_params.sample_rate = buffer.sample_rate;
    effective_params.channels = buffer.channels;

    let output_data = process_fn(&buffer.data, &effective_params)?;
    Ok(AudioBuffer::new(
        output_data,
        buffer.sample_rate,
        buffer.channels,
    ))
}

/// Deinterleaves multi-channel audio into separate per-channel vectors.
#[inline]
fn deinterleave(input: &[f32], num_channels: usize) -> Vec<Vec<f32>> {
    (0..num_channels)
        .map(|ch| {
            input
                .iter()
                .skip(ch)
                .step_by(num_channels)
                .copied()
                .collect()
        })
        .collect()
}

/// Interleaves per-channel vectors into a single buffer, truncating to the shortest channel.
#[inline]
fn interleave(channels: &[Vec<f32>]) -> Vec<f32> {
    let min_len = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    (0..min_len)
        .flat_map(|i| channels.iter().map(move |ch| ch[i]))
        .collect()
}

/// Validates that input is non-empty and contains only finite samples.
///
/// Returns `Ok(false)` if input is empty (caller should return `Ok(vec![])`),
/// `Ok(true)` if input is valid, or `Err` if it contains NaN/Inf.
#[inline]
fn validate_input(input: &[f32]) -> Result<bool, StretchError> {
    if input.is_empty() {
        return Ok(false);
    }
    if input.iter().any(|s| !s.is_finite()) {
        return Err(StretchError::NonFiniteInput);
    }
    Ok(true)
}

/// Extracts a mono signal from interleaved audio (takes the first channel).
#[inline]
fn extract_mono(samples: &[f32], num_channels: usize) -> Vec<f32> {
    if num_channels <= 1 {
        samples.to_vec()
    } else {
        samples.iter().step_by(num_channels).copied().collect()
    }
}

/// Minimum RMS threshold to avoid division by zero during normalization.
const NORMALIZE_RMS_FLOOR: f32 = 1e-8;

/// Computes the RMS (root mean square) of a signal.
#[inline]
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

/// Scales output so its RMS matches `target_rms`, if the output has sufficient energy.
#[inline]
fn normalize_rms(output: &mut [f32], target_rms: f32) {
    let output_rms = compute_rms(output);
    if output_rms < NORMALIZE_RMS_FLOOR || target_rms < NORMALIZE_RMS_FLOOR {
        return;
    }
    let gain = target_rms / output_rms;
    for s in output.iter_mut() {
        *s *= gain;
    }
}

/// Validates that a BPM value is positive, returning a descriptive error otherwise.
#[inline]
fn validate_bpm(bpm: f64, label: &str) -> Result<(), StretchError> {
    if bpm <= 0.0 {
        return Err(StretchError::BpmDetectionFailed(format!(
            "{} BPM must be positive, got {}",
            label, bpm
        )));
    }
    Ok(())
}

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
    stretch::params::validate_params(params).map_err(StretchError::InvalidRatio)?;

    if !validate_input(input)? {
        return Ok(vec![]);
    }

    let num_channels = params.channels.count();
    let channels = deinterleave(input, num_channels);

    let input_rms = if params.normalize {
        compute_rms(input)
    } else {
        0.0
    };

    let mut channel_outputs: Vec<Vec<f32>> = Vec::with_capacity(num_channels);
    for channel_data in &channels {
        let stretcher = stretch::hybrid::HybridStretcher::new(params.clone());
        let stretched = stretcher.process(channel_data)?;
        channel_outputs.push(stretched);
    }

    let mut output = interleave(&channel_outputs);

    if params.normalize {
        normalize_rms(&mut output, input_rms);
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
    process_buffer(buffer, params, stretch)
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
    use stretch::params::{RATIO_MAX, RATIO_MIN};
    if !(RATIO_MIN..=RATIO_MAX).contains(&pitch_factor) {
        return Err(StretchError::InvalidRatio(format!(
            "Pitch factor must be between {} and {}, got {}",
            RATIO_MIN, RATIO_MAX, pitch_factor
        )));
    }

    if !validate_input(input)? {
        return Ok(vec![]);
    }

    let input_rms = if params.normalize {
        compute_rms(input)
    } else {
        0.0
    };

    // Step 1: Time-stretch by 1/pitch_factor to compensate for the resampling
    // Disable normalization for the inner stretch — we normalize the final result.
    let stretch_ratio = 1.0 / pitch_factor;
    let mut stretch_params = params.clone();
    stretch_params.stretch_ratio = stretch_ratio;
    stretch_params.normalize = false;
    let stretched = stretch(input, &stretch_params)?;

    if stretched.is_empty() {
        return Ok(vec![]);
    }

    // Step 2: Resample each channel to original length using cubic interpolation
    let num_channels = params.channels.count();
    let num_input_frames = input.len() / num_channels;
    let channels = deinterleave(&stretched, num_channels);

    let channel_outputs: Vec<Vec<f32>> = channels
        .iter()
        .map(|ch| core::resample::resample_cubic(ch, num_input_frames))
        .collect();

    let mut output = interleave(&channel_outputs);

    if params.normalize {
        normalize_rms(&mut output, input_rms);
    }

    Ok(output)
}

/// Shifts the pitch of an [`AudioBuffer`] without changing its duration.
///
/// Convenience wrapper around [`pitch_shift`] that takes and returns
/// an `AudioBuffer`. Sample rate and channel layout are taken from the buffer.
///
/// # Errors
///
/// Returns [`StretchError::InvalidRatio`] if the pitch factor is out of range.
pub fn pitch_shift_buffer(
    buffer: &AudioBuffer,
    params: &StretchParams,
    pitch_factor: f64,
) -> Result<AudioBuffer, StretchError> {
    process_buffer(buffer, params, |data, p| pitch_shift(data, p, pitch_factor))
}

/// Detects the BPM of a mono audio signal.
///
/// Uses transient detection and inter-onset interval analysis optimized
/// for 4/4 EDM (house/techno) with expected BPM range 100-160. Returns
/// the estimated BPM, or 0.0 if no tempo can be detected.
///
/// For stereo audio, extract the left channel first (or mix to mono).
///
/// # Example
///
/// ```
/// // Generate a click train at ~120 BPM
/// let sample_rate = 44100u32;
/// let beat_interval = (60.0 * sample_rate as f64 / 120.0) as usize;
/// let mut audio = vec![0.0f32; sample_rate as usize * 4];
/// for pos in (0..audio.len()).step_by(beat_interval) {
///     for j in 0..10.min(audio.len() - pos) {
///         audio[pos + j] = if j < 5 { 0.9 } else { -0.4 };
///     }
/// }
///
/// let bpm = timestretch::detect_bpm(&audio, sample_rate);
/// // BPM detection may or may not succeed on synthetic clicks
/// // For real EDM audio with kicks, this is very reliable
/// ```
pub fn detect_bpm(samples: &[f32], sample_rate: u32) -> f64 {
    analysis::beat::detect_beats(samples, sample_rate).bpm
}

/// Detects beats and returns a [`BeatGrid`] with BPM and beat positions.
///
/// This provides more detail than [`detect_bpm`], including the sample
/// positions of detected beats and a grid-snapping utility.
///
/// # Example
///
/// ```
/// let audio = vec![0.0f32; 44100 * 4];
/// let grid = timestretch::detect_beat_grid(&audio, 44100);
/// println!("BPM: {}, beats: {}", grid.bpm, grid.beats.len());
/// ```
pub fn detect_beat_grid(samples: &[f32], sample_rate: u32) -> BeatGrid {
    analysis::beat::detect_beats(samples, sample_rate)
}

/// Detects the BPM of an [`AudioBuffer`].
///
/// For stereo buffers, uses the left channel for detection.
/// Returns 0.0 if no tempo can be detected.
pub fn detect_bpm_buffer(buffer: &AudioBuffer) -> f64 {
    let mono = extract_mono(&buffer.data, buffer.channels.count());
    detect_bpm(&mono, buffer.sample_rate)
}

/// Detects beats in an [`AudioBuffer`] and returns a [`BeatGrid`].
///
/// For stereo buffers, uses the left channel for detection.
/// This is the buffer-based equivalent of [`detect_beat_grid`].
pub fn detect_beat_grid_buffer(buffer: &AudioBuffer) -> BeatGrid {
    let mono = extract_mono(&buffer.data, buffer.channels.count());
    detect_beat_grid(&mono, buffer.sample_rate)
}

/// Stretches audio from one BPM to another.
///
/// Computes the stretch ratio as `source_bpm / target_bpm` and applies
/// time stretching. For example, stretching from 126 BPM to 128 BPM
/// produces a ratio of 126/128 ≈ 0.984 (slightly shorter/faster).
///
/// # Errors
///
/// Returns [`StretchError::BpmDetectionFailed`] if either BPM value is invalid,
/// or [`StretchError::InvalidRatio`] if the computed ratio is out of range.
///
/// # Example
///
/// ```
/// use timestretch::{StretchParams, EdmPreset};
///
/// let input: Vec<f32> = (0..88200)
///     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
///     .collect();
///
/// let params = StretchParams::new(1.0) // ratio will be overridden
///     .with_sample_rate(44100)
///     .with_channels(1)
///     .with_preset(EdmPreset::DjBeatmatch);
///
/// let output = timestretch::stretch_to_bpm(&input, 126.0, 128.0, &params).unwrap();
/// // Output is slightly shorter (126/128 ≈ 0.984x)
/// assert!(output.len() < input.len());
/// ```
pub fn stretch_to_bpm(
    input: &[f32],
    source_bpm: f64,
    target_bpm: f64,
    params: &StretchParams,
) -> Result<Vec<f32>, StretchError> {
    validate_bpm(source_bpm, "source")?;
    validate_bpm(target_bpm, "target")?;

    let ratio = source_bpm / target_bpm;
    let mut adjusted_params = params.clone();
    adjusted_params.stretch_ratio = ratio;

    stretch(input, &adjusted_params)
}

/// Stretches audio to a target BPM, auto-detecting the source BPM.
///
/// Uses beat detection to estimate the current tempo, then computes the
/// stretch ratio needed to reach `target_bpm`. Best suited for audio
/// with a clear rhythmic pattern (kicks, hi-hats).
///
/// For mono input, pass samples directly. For stereo, pass interleaved
/// L/R samples and set channels to 2 — BPM detection uses the left channel.
///
/// # Errors
///
/// Returns [`StretchError::BpmDetectionFailed`] if no tempo can be detected
/// (e.g. the input is too short, contains only silence, or lacks rhythmic content).
///
/// # Example
///
/// ```
/// use timestretch::{StretchParams, EdmPreset};
///
/// // Generate a click train at ~120 BPM for auto-detection
/// let sample_rate = 44100u32;
/// let beat_interval = (60.0 * sample_rate as f64 / 120.0) as usize;
/// let mut audio = vec![0.0f32; sample_rate as usize * 4];
/// for pos in (0..audio.len()).step_by(beat_interval) {
///     for j in 0..20.min(audio.len() - pos) {
///         audio[pos + j] = if j < 5 { 0.9 } else { -0.4 };
///     }
/// }
///
/// let params = StretchParams::new(1.0)
///     .with_sample_rate(sample_rate)
///     .with_channels(1)
///     .with_preset(EdmPreset::DjBeatmatch);
///
/// // Auto-detect BPM and stretch to 128 BPM
/// match timestretch::stretch_to_bpm_auto(&audio, 128.0, &params) {
///     Ok(output) => println!("Stretched {} -> {} samples", audio.len(), output.len()),
///     Err(e) => println!("BPM detection failed: {}", e),
/// }
/// ```
pub fn stretch_to_bpm_auto(
    input: &[f32],
    target_bpm: f64,
    params: &StretchParams,
) -> Result<Vec<f32>, StretchError> {
    validate_bpm(target_bpm, "target")?;

    // Reject non-finite samples before expensive beat detection
    if !validate_input(input)? {
        return Ok(vec![]);
    }

    // Extract mono signal for beat detection
    let mono = extract_mono(input, params.channels.count());

    let beat_grid = analysis::beat::detect_beats(&mono, params.sample_rate);

    if beat_grid.bpm <= 0.0 {
        return Err(StretchError::BpmDetectionFailed(
            "could not detect BPM from input audio (too short or no rhythmic content)".to_string(),
        ));
    }

    stretch_to_bpm(input, beat_grid.bpm, target_bpm, params)
}

/// Stretches an [`AudioBuffer`] from one BPM to another.
///
/// Convenience wrapper around [`stretch_to_bpm`] that takes and returns
/// an `AudioBuffer`. Sample rate and channel layout are taken from the buffer.
///
/// # Errors
///
/// Returns [`StretchError::BpmDetectionFailed`] if either BPM value is invalid.
pub fn stretch_bpm_buffer(
    buffer: &AudioBuffer,
    source_bpm: f64,
    target_bpm: f64,
    params: &StretchParams,
) -> Result<AudioBuffer, StretchError> {
    process_buffer(buffer, params, |data, p| {
        stretch_to_bpm(data, source_bpm, target_bpm, p)
    })
}

/// Stretches an [`AudioBuffer`] to a target BPM, auto-detecting the source BPM.
///
/// Convenience wrapper around [`stretch_to_bpm_auto`] that takes and returns
/// an `AudioBuffer`. Sample rate and channel layout are taken from the buffer.
///
/// # Errors
///
/// Returns [`StretchError::BpmDetectionFailed`] if no tempo can be detected.
pub fn stretch_bpm_buffer_auto(
    buffer: &AudioBuffer,
    target_bpm: f64,
    params: &StretchParams,
) -> Result<AudioBuffer, StretchError> {
    process_buffer(buffer, params, |data, p| {
        stretch_to_bpm_auto(data, target_bpm, p)
    })
}

/// Reads a WAV file, stretches it, and writes the result to another WAV file.
///
/// The output is written as 32-bit float WAV. The sample rate and channel
/// layout are read from the input file and passed through automatically.
///
/// # Errors
///
/// Returns [`StretchError::IoError`] if the files cannot be read or written,
/// [`StretchError::InvalidFormat`] if the input is not a valid WAV file,
/// or [`StretchError::InvalidRatio`] if the stretch ratio is out of range.
pub fn stretch_wav_file(
    input_path: &str,
    output_path: &str,
    params: &StretchParams,
) -> Result<AudioBuffer, StretchError> {
    let buffer = io::wav::read_wav_file(input_path)?;
    let result = stretch_buffer(&buffer, params)?;
    io::wav::write_wav_file_float(output_path, &result)?;
    Ok(result)
}

/// Reads a WAV file, stretches it from one BPM to another, and writes the result.
///
/// The output is written as 32-bit float WAV. The sample rate and channel
/// layout are read from the input file and passed through automatically.
/// This is a convenience function combining WAV I/O with BPM-based stretching.
///
/// # Errors
///
/// Returns [`StretchError::IoError`] if the files cannot be read or written,
/// [`StretchError::InvalidFormat`] if the input is not a valid WAV file,
/// [`StretchError::BpmDetectionFailed`] if either BPM value is invalid.
pub fn stretch_to_bpm_wav_file(
    input_path: &str,
    output_path: &str,
    source_bpm: f64,
    target_bpm: f64,
    params: &StretchParams,
) -> Result<AudioBuffer, StretchError> {
    let buffer = io::wav::read_wav_file(input_path)?;
    let result = stretch_bpm_buffer(&buffer, source_bpm, target_bpm, params)?;
    io::wav::write_wav_file_float(output_path, &result)?;
    Ok(result)
}

/// Reads a WAV file, pitch-shifts it, and writes the result to another WAV file.
///
/// The output is written as 32-bit float WAV. The sample rate and channel
/// layout are read from the input file.
///
/// # Errors
///
/// Returns [`StretchError::IoError`] if the files cannot be read or written,
/// or [`StretchError::InvalidRatio`] if the pitch factor is out of range.
pub fn pitch_shift_wav_file(
    input_path: &str,
    output_path: &str,
    params: &StretchParams,
    pitch_factor: f64,
) -> Result<AudioBuffer, StretchError> {
    let buffer = io::wav::read_wav_file(input_path)?;
    let result = pitch_shift_buffer(&buffer, params, pitch_factor)?;
    io::wav::write_wav_file_float(output_path, &result)?;
    Ok(result)
}

/// Returns the stretch ratio needed to change from one BPM to another.
///
/// This is a simple utility: `source_bpm / target_bpm`. Use it when you
/// want to compute the ratio yourself before calling [`stretch()`].
///
/// # Example
///
/// ```
/// let ratio = timestretch::bpm_ratio(126.0, 128.0);
/// assert!((ratio - 0.984375).abs() < 1e-6);
/// ```
#[inline]
pub fn bpm_ratio(source_bpm: f64, target_bpm: f64) -> f64 {
    source_bpm / target_bpm
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time assertions that key public types are Send + Sync.
    // This is critical for real-time audio where processing often runs
    // on a dedicated thread.
    const _: () = {
        fn assert_send_sync<T: Send + Sync>() {}
        fn check() {
            assert_send_sync::<AudioBuffer>();
            assert_send_sync::<StretchParams>();
            assert_send_sync::<StreamProcessor>();
            assert_send_sync::<StretchError>();
            assert_send_sync::<BeatGrid>();
        }
        let _ = check;
    };

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

    #[test]
    fn test_bpm_ratio() {
        let ratio = bpm_ratio(126.0, 128.0);
        assert!((ratio - 0.984375).abs() < 1e-6);

        // Same BPM = ratio 1.0
        assert!((bpm_ratio(120.0, 120.0) - 1.0).abs() < 1e-10);

        // Double BPM = half length
        assert!((bpm_ratio(120.0, 240.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stretch_to_bpm_basic() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(EdmPreset::DjBeatmatch);

        // 126 -> 128 BPM: should produce slightly shorter output
        let output = stretch_to_bpm(&input, 126.0, 128.0, &params).unwrap();
        let expected_ratio = 126.0 / 128.0;
        let actual_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (actual_ratio - expected_ratio).abs() < 0.3,
            "BPM stretch ratio: expected ~{:.3}, got {:.3}",
            expected_ratio,
            actual_ratio
        );
    }

    #[test]
    fn test_stretch_to_bpm_speedup() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        // 120 -> 150 BPM: significant speedup (ratio 0.8)
        let output = stretch_to_bpm(&input, 120.0, 150.0, &params).unwrap();
        assert!(
            output.len() < input.len(),
            "Should be shorter when speeding up"
        );
    }

    #[test]
    fn test_stretch_to_bpm_slowdown() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        // 120 -> 90 BPM: slow down (ratio 1.333)
        let output = stretch_to_bpm(&input, 120.0, 90.0, &params).unwrap();
        assert!(
            output.len() > input.len(),
            "Should be longer when slowing down"
        );
    }

    #[test]
    fn test_stretch_to_bpm_invalid_bpm() {
        let params = StretchParams::new(1.0);
        let input = vec![0.0f32; 44100];

        // Zero source BPM
        assert!(stretch_to_bpm(&input, 0.0, 128.0, &params).is_err());
        // Negative source BPM
        assert!(stretch_to_bpm(&input, -120.0, 128.0, &params).is_err());
        // Zero target BPM
        assert!(stretch_to_bpm(&input, 120.0, 0.0, &params).is_err());
        // Negative target BPM
        assert!(stretch_to_bpm(&input, 120.0, -128.0, &params).is_err());
    }

    #[test]
    fn test_stretch_to_bpm_same_bpm() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        // Same BPM: ratio 1.0, output length ~ input length
        let output = stretch_to_bpm(&input, 128.0, 128.0, &params).unwrap();
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.0).abs() < 0.1,
            "Same BPM should preserve length, got ratio {}",
            len_ratio
        );
    }

    #[test]
    fn test_stretch_to_bpm_empty() {
        let params = StretchParams::new(1.0);
        let output = stretch_to_bpm(&[], 120.0, 128.0, &params).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_stretch_to_bpm_auto_silence() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        // Silence has no beats, should return BpmDetectionFailed
        let silence = vec![0.0f32; 44100 * 4];
        let result = stretch_to_bpm_auto(&silence, 128.0, &params);
        assert!(result.is_err());
        match result {
            Err(StretchError::BpmDetectionFailed(_)) => {} // expected
            other => panic!("Expected BpmDetectionFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_stretch_to_bpm_auto_invalid_target() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let input = vec![0.0f32; 44100];

        assert!(stretch_to_bpm_auto(&input, 0.0, &params).is_err());
        assert!(stretch_to_bpm_auto(&input, -128.0, &params).is_err());
    }

    #[test]
    fn test_stretch_bpm_buffer() {
        let sample_rate = 44100u32;
        let buffer = AudioBuffer::from_mono(
            (0..sample_rate as usize * 2)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
                .collect(),
            sample_rate,
        );

        let params = StretchParams::new(1.0).with_preset(EdmPreset::DjBeatmatch);
        let output = stretch_bpm_buffer(&buffer, 126.0, 128.0, &params).unwrap();
        assert_eq!(output.sample_rate, sample_rate);
        assert_eq!(output.channels, Channels::Mono);
        assert!(output.data.len() < buffer.data.len()); // Speeding up
    }

    #[test]
    fn test_stretch_rejects_nan() {
        let mut input = vec![0.0f32; 44100];
        input[1000] = f32::NAN;
        let params = StretchParams::new(1.5).with_channels(1);
        assert!(matches!(
            stretch(&input, &params),
            Err(StretchError::NonFiniteInput)
        ));
    }

    #[test]
    fn test_stretch_rejects_infinity() {
        let mut input = vec![0.0f32; 44100];
        input[500] = f32::INFINITY;
        let params = StretchParams::new(1.5).with_channels(1);
        assert!(matches!(
            stretch(&input, &params),
            Err(StretchError::NonFiniteInput)
        ));

        input[500] = f32::NEG_INFINITY;
        assert!(matches!(
            stretch(&input, &params),
            Err(StretchError::NonFiniteInput)
        ));
    }

    #[test]
    fn test_pitch_shift_rejects_nan() {
        let mut input = vec![0.0f32; 44100];
        input[100] = f32::NAN;
        let params = StretchParams::new(1.0).with_channels(1);
        assert!(matches!(
            pitch_shift(&input, &params, 1.5),
            Err(StretchError::NonFiniteInput)
        ));
    }

    #[test]
    fn test_from_tempo_stretch() {
        // Verify from_tempo integrates with stretch()
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let params = StretchParams::from_tempo(126.0, 128.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let output = stretch(&input, &params).unwrap();
        // Compressing: output should be shorter than input
        assert!(output.len() < input.len());
    }

    #[test]
    fn test_detect_bpm_silence() {
        // Silence should return 0 BPM
        let silence = vec![0.0f32; 44100 * 4];
        let bpm = detect_bpm(&silence, 44100);
        assert!(bpm == 0.0, "Silence should return 0 BPM, got {}", bpm);
    }

    #[test]
    fn test_detect_bpm_empty() {
        let bpm = detect_bpm(&[], 44100);
        assert!(bpm == 0.0, "Empty input should return 0 BPM, got {}", bpm);
    }

    #[test]
    fn test_detect_bpm_short_input() {
        // Very short input should not crash
        let short = vec![0.5f32; 100];
        let bpm = detect_bpm(&short, 44100);
        // Should return 0 or some value, but not crash
        assert!(bpm >= 0.0);
    }

    #[test]
    fn test_detect_beat_grid_returns_grid() {
        let sample_rate = 44100u32;
        // Create a click train at ~120 BPM
        let beat_interval = (60.0 * sample_rate as f64 / 120.0) as usize;
        let num_samples = sample_rate as usize * 4;
        let mut audio = vec![0.0f32; num_samples];

        for pos in (0..num_samples).step_by(beat_interval) {
            for j in 0..20.min(num_samples - pos) {
                audio[pos + j] = if j < 5 { 0.9 } else { -0.4 };
            }
            // Add tone between clicks for transient detector
            let tone_start = pos + 20;
            let tone_end = (pos + beat_interval / 2).min(num_samples);
            for (i, sample) in audio[tone_start..tone_end].iter_mut().enumerate() {
                let idx = tone_start + i;
                *sample += 0.2
                    * (2.0 * std::f32::consts::PI * 200.0 * idx as f32 / sample_rate as f32).sin();
            }
        }

        let grid = detect_beat_grid(&audio, sample_rate);
        assert_eq!(grid.sample_rate, sample_rate);
        // Beat grid should have reasonable interval if beats were detected
        if grid.bpm > 0.0 {
            let interval = grid.beat_interval_samples();
            assert!(interval > 0.0, "Beat interval should be positive");
        }
    }

    #[test]
    fn test_detect_bpm_with_click_train() {
        let sample_rate = 44100u32;
        let target_bpm = 120.0;
        let beat_interval = (60.0 * sample_rate as f64 / target_bpm) as usize;
        let num_samples = sample_rate as usize * 6; // 6 seconds

        let mut audio = vec![0.0f32; num_samples];

        // Create strong clicks at beat positions
        for pos in (0..num_samples).step_by(beat_interval) {
            for j in 0..10.min(num_samples - pos) {
                audio[pos + j] = if j < 5 { 0.95 } else { -0.5 };
            }
        }

        // Add background tone
        for (i, sample) in audio.iter_mut().enumerate() {
            *sample +=
                0.15 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sample_rate as f32).sin();
        }

        let bpm = detect_bpm(&audio, sample_rate);
        // BPM detection is heuristic; may succeed or detect a harmonic (e.g., 240 BPM)
        // but should return something in the EDM range if it finds beats
        if bpm > 0.0 {
            assert!(
                (100.0..=160.0).contains(&bpm),
                "BPM {} should be in EDM range 100-160",
                bpm
            );
        }
    }

    #[test]
    fn test_pitch_shift_buffer() {
        let buffer = AudioBuffer::from_mono(
            (0..44100)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
                .collect(),
            44100,
        );

        let params = StretchParams::new(1.0);
        let output = pitch_shift_buffer(&buffer, &params, 1.5).unwrap();
        assert_eq!(output.sample_rate, 44100);
        assert_eq!(output.channels, Channels::Mono);
        // Pitch shift preserves length
        assert_eq!(output.data.len(), buffer.data.len());
    }

    #[test]
    fn test_pitch_shift_buffer_stereo() {
        let sample_rate = 44100u32;
        let num_frames = 44100;
        let mut data = vec![0.0f32; num_frames * 2];
        for i in 0..num_frames {
            let t = i as f32 / sample_rate as f32;
            data[i * 2] = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            data[i * 2 + 1] = (2.0 * std::f32::consts::PI * 880.0 * t).sin();
        }

        let buffer = AudioBuffer::new(data, sample_rate, Channels::Stereo);
        let params = StretchParams::new(1.0);
        let output = pitch_shift_buffer(&buffer, &params, 0.8).unwrap();
        assert_eq!(output.data.len(), buffer.data.len());
        assert_eq!(output.channels, Channels::Stereo);
    }

    #[test]
    fn test_detect_bpm_buffer_silence() {
        let buffer = AudioBuffer::from_mono(vec![0.0f32; 44100 * 4], 44100);
        let bpm = detect_bpm_buffer(&buffer);
        assert!(bpm == 0.0, "Silence should return 0 BPM, got {}", bpm);
    }

    #[test]
    fn test_detect_bpm_buffer_stereo() {
        // Stereo buffer with silence should return 0 BPM and not crash
        let data = vec![0.0f32; 44100 * 4 * 2]; // 4 seconds stereo
        let buffer = AudioBuffer::new(data, 44100, Channels::Stereo);
        let bpm = detect_bpm_buffer(&buffer);
        assert!(bpm == 0.0, "Silence should return 0 BPM, got {}", bpm);
    }

    #[test]
    fn test_detect_beat_grid_buffer_mono() {
        let buffer = AudioBuffer::from_mono(vec![0.0f32; 44100 * 4], 44100);
        let grid = detect_beat_grid_buffer(&buffer);
        assert_eq!(grid.sample_rate, 44100);
    }

    #[test]
    fn test_detect_beat_grid_buffer_stereo() {
        let data = vec![0.0f32; 44100 * 4 * 2]; // 4 seconds stereo
        let buffer = AudioBuffer::new(data, 44100, Channels::Stereo);
        let grid = detect_beat_grid_buffer(&buffer);
        assert_eq!(grid.sample_rate, 44100);
        // Silence should yield 0 BPM
        assert!(
            grid.bpm == 0.0,
            "Silence should return 0 BPM, got {}",
            grid.bpm
        );
    }

    #[test]
    fn test_stretch_wav_file() {
        // Create a temp WAV file
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let buf = AudioBuffer::from_mono(input, 44100);
        let dir = std::env::temp_dir();
        let in_path = dir.join("timestretch_test_in.wav");
        let out_path = dir.join("timestretch_test_out.wav");
        io::wav::write_wav_file_float(in_path.to_str().unwrap(), &buf).unwrap();

        let params = StretchParams::new(1.5).with_channels(1);
        let result = stretch_wav_file(
            in_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
            &params,
        )
        .unwrap();

        assert!(!result.is_empty());
        assert_eq!(result.channels, Channels::Mono);

        // Verify the output file was written
        let reloaded = io::wav::read_wav_file(out_path.to_str().unwrap()).unwrap();
        assert_eq!(reloaded.data.len(), result.data.len());

        // Clean up
        let _ = std::fs::remove_file(&in_path);
        let _ = std::fs::remove_file(&out_path);
    }

    #[test]
    fn test_pitch_shift_wav_file() {
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let buf = AudioBuffer::from_mono(input, 44100);
        let dir = std::env::temp_dir();
        let in_path = dir.join("timestretch_pitch_in.wav");
        let out_path = dir.join("timestretch_pitch_out.wav");
        io::wav::write_wav_file_float(in_path.to_str().unwrap(), &buf).unwrap();

        let params = StretchParams::new(1.0).with_channels(1);
        let result = pitch_shift_wav_file(
            in_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
            &params,
            1.5,
        )
        .unwrap();

        assert!(!result.is_empty());
        // Pitch shift preserves length
        assert_eq!(result.data.len(), buf.data.len());

        // Clean up
        let _ = std::fs::remove_file(&in_path);
        let _ = std::fs::remove_file(&out_path);
    }

    #[test]
    fn test_stretch_wav_file_missing_input() {
        let params = StretchParams::new(1.5);
        let result = stretch_wav_file("/nonexistent/path/input.wav", "/tmp/output.wav", &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_stretch_to_bpm_wav_file() {
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let buf = AudioBuffer::from_mono(input, 44100);
        let dir = std::env::temp_dir();
        let in_path = dir.join("timestretch_bpm_in.wav");
        let out_path = dir.join("timestretch_bpm_out.wav");
        io::wav::write_wav_file_float(in_path.to_str().unwrap(), &buf).unwrap();

        let params = StretchParams::new(1.0).with_channels(1);
        let result = stretch_to_bpm_wav_file(
            in_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
            126.0,
            128.0,
            &params,
        )
        .unwrap();

        // Ratio should be 126/128 ≈ 0.984, so output slightly shorter
        assert!(result.data.len() < 44100);
        assert!(!result.is_empty());

        // Verify output was written
        let reloaded = io::wav::read_wav_file(out_path.to_str().unwrap()).unwrap();
        assert_eq!(reloaded.data.len(), result.data.len());
    }

    #[test]
    fn test_normalize_preserves_rms() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                0.8 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let input_rms = compute_rms(&input);

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_normalize(true);

        let output = stretch(&input, &params).unwrap();
        let output_rms = compute_rms(&output);

        // With normalization, output RMS should be very close to input RMS
        assert!(
            (output_rms - input_rms).abs() < input_rms * 0.05,
            "Normalized RMS mismatch: input={:.4}, output={:.4}",
            input_rms,
            output_rms
        );
    }

    #[test]
    fn test_normalize_off_by_default() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                0.8 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        // Without normalization, stretch should still work
        let output = stretch(&input, &params).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_normalize_with_silence() {
        // Normalization should not amplify silence
        let silence = vec![0.0f32; 44100];
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_normalize(true);

        let output = stretch(&silence, &params).unwrap();
        let max_val = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(
            max_val < 1e-4,
            "Normalized silence should stay silent, got max={}",
            max_val
        );
    }

    #[test]
    fn test_normalize_with_compression() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                0.6 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let input_rms = compute_rms(&input);

        // Compression (ratio < 1.0)
        let params = StretchParams::new(0.75)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_normalize(true);

        let output = stretch(&input, &params).unwrap();
        let output_rms = compute_rms(&output);

        assert!(
            (output_rms - input_rms).abs() < input_rms * 0.1,
            "Normalized compression RMS mismatch: input={:.4}, output={:.4}",
            input_rms,
            output_rms
        );
    }

    #[test]
    fn test_stretch_with_window_type() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        // Stretch with Blackman-Harris window
        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_window_type(core::window::WindowType::BlackmanHarris);

        let output = stretch(&input, &params).unwrap();
        assert!(!output.is_empty());

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.5).abs() < 0.5,
            "BH stretch ratio {} too far from 1.5",
            len_ratio
        );
    }

    #[test]
    fn test_pitch_shift_with_normalize() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                0.7 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let input_rms = compute_rms(&input);

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_normalize(true);

        let output = pitch_shift(&input, &params, 1.5).unwrap();
        let output_rms = compute_rms(&output);

        assert!(
            (output_rms - input_rms).abs() < input_rms * 0.1,
            "Normalized pitch shift RMS mismatch: input={:.4}, output={:.4}",
            input_rms,
            output_rms
        );
    }
}

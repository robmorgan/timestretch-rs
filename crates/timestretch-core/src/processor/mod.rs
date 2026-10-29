//! Reference-validated kernels and native schedule.
// Diagnostics and component entry points are reachable only through the QA feature.
#![cfg_attr(not(feature = "qa"), allow(dead_code, unused_imports))]
mod analysis;
mod buffering;
mod fft;
#[cfg(any(test, feature = "qa"))]
mod frame;
mod marks;
mod math;
mod phase_vocoder;
#[cfg(feature = "qa")]
#[doc(hidden)]
pub mod qa;
mod resample;
mod scheduling;
mod selector;
mod splitter;
mod synthesis;
mod tables;
mod transient;

pub use analysis::{AnalysisState, MarkAnalysis};
#[cfg(any(test, feature = "qa"))]
pub use frame::FrameProcessor;
pub use marks::{PitchMark, PitchMarks};
pub use phase_vocoder::{PhaseVocoder, VocoderState};
pub use resample::{Interpolation, Resampler, ResamplerState};
pub use scheduling::{NativeProcessor, RatioAdjustment, SchedulerState, StartupOptimization};
pub use splitter::Splitter;
pub use synthesis::{GrainSynthesis, SynthesisState};

use crate::StretchError;

fn invalid(message: &'static str) -> StretchError {
    StretchError::InvalidFormat(message)
}

fn check_channels(channels: usize) -> Result<(), StretchError> {
    if !(1..=2).contains(&channels) {
        return Err(invalid("timestretch supports mono or stereo"));
    }
    Ok(())
}

fn input_frames(input: &[&[f32]], channels: usize) -> Result<usize, StretchError> {
    if input.len() != channels {
        return Err(invalid("planar input channel count mismatch"));
    }
    let frames = input[0].len();
    if input.iter().any(|x| x.len() != frames) {
        return Err(invalid("planar input lengths differ"));
    }
    if input.iter().any(|x| x.iter().any(|v| !v.is_finite())) {
        return Err(StretchError::NonFiniteInput);
    }
    Ok(frames)
}

fn output_capacity(
    output: &[&mut [f32]],
    channels: usize,
    needed: usize,
) -> Result<(), StretchError> {
    if output.len() != channels || output.iter().any(|x| x.len() < needed) {
        return Err(invalid("planar output capacity is insufficient"));
    }
    Ok(())
}

fn check_capacity(capacity: usize) -> Result<(), StretchError> {
    if !(16..=1_048_576).contains(&capacity) {
        return Err(invalid("timestretch capacity must be 16..=1048576 frames"));
    }
    Ok(())
}

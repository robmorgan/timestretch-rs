//! Optional track analysis, independent of the time/pitch processor.
#![forbid(unsafe_code)]
pub mod analysis;
mod core;
pub mod error;
pub use analysis::*;
pub use core::preanalysis::*;
pub use core::types::TransientThresholdPolicy;
pub use error::StretchError;

/// Detect the tempo of a mono signal.
pub fn detect_bpm(samples: &[f32], sample_rate: u32) -> f64 {
    analysis::beat::detect_beats(samples, sample_rate).bpm
}
/// Detect a beat grid with the general-purpose tracker.
pub fn detect_beat_grid(samples: &[f32], sample_rate: u32) -> BeatGrid {
    analysis::beat::detect_beats(samples, sample_rate)
}
/// Detect a beat grid with explicit tracking options.
pub fn detect_beat_grid_with_options(
    samples: &[f32],
    sample_rate: u32,
    options: &TempoTrackingOptions,
) -> BeatGrid {
    analysis::beat::detect_beats_with_options(samples, sample_rate, options)
}

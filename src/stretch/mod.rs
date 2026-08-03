//! Batch time-stretching DSP retained by the engine: the phase vocoder
//! (the offline wide-ratio path and analysis tooling) and its support
//! modules. The hybrid combiner, multi-resolution stretcher, WSOLA
//! driver and stereo mid/side wrapper were deleted with the old engine
//! at ROADMAP Stage 9.

pub mod envelope;
pub mod params;
pub mod phase_locking;
pub mod phase_vocoder;

pub use phase_locking::PhaseLockingMode;
pub use phase_vocoder::PhaseVocoder;

//! Batch time-stretching DSP retained by the engine: the phase vocoder
//! and its support modules. The PV backs the wide keylock stage (live and
//! offline; ratios inside the engine's rate range run the shipped wide
//! stage) and is used directly only for ratios beyond the engine range,
//! plus analysis tooling. The hybrid combiner, multi-resolution
//! stretcher, WSOLA driver and stereo mid/side wrapper were deleted with
//! the old engine at ROADMAP Stage 9.

pub mod envelope;
pub mod params;
pub mod phase_locking;
pub mod phase_vocoder;

pub use phase_locking::PhaseLockingMode;
pub use phase_vocoder::PhaseVocoder;

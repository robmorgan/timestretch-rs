//! Time stretching algorithms: phase vocoder, WSOLA, and the hybrid combiner.

pub mod envelope;
pub mod hybrid;
pub mod multi_resolution;
pub mod params;
pub mod phase_locking;
pub mod phase_vocoder;
pub mod stereo;
pub mod wsola;

pub use hybrid::HybridStretcher;
pub use multi_resolution::MultiResolutionStretcher;
pub use phase_locking::PhaseLockingMode;
pub use phase_vocoder::PhaseVocoder;
pub use stereo::StereoMode;
pub use wsola::Wsola;

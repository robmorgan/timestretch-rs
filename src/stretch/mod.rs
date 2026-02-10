//! Time stretching algorithms: phase vocoder, WSOLA, and the hybrid combiner.

pub mod hybrid;
pub mod params;
pub mod phase_vocoder;
pub mod wsola;

pub use hybrid::HybridStretcher;
pub use phase_vocoder::PhaseVocoder;
pub use wsola::Wsola;

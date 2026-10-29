//! File I/O: WAV audio and the `.tsa` analysis container.

#[cfg(feature = "analysis")]
pub mod tsa;
pub mod wav;

#[cfg(feature = "analysis")]
pub use tsa::*;
pub use wav::*;

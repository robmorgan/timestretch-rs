//! Native timestretch kernels. Applications use `timestretch::StreamSession`.
#![forbid(unsafe_code)]
mod error;
pub use error::StretchError;
#[cfg(any(test, feature = "qa"))]
mod core;
#[cfg(feature = "qa")]
pub mod processor;
#[cfg(not(feature = "qa"))]
mod processor;
#[doc(hidden)]
pub use processor::{NativeProcessor, RatioAdjustment};
mod numerics;
pub use numerics::Numerics;

#[doc(hidden)]
pub mod memory;

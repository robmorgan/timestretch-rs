//! timestretch time/pitch processing through one host-driven session.
//!
//! ```
//! use timestretch::{render, Backend, Parameters, SessionConfig, TempoRate};
//! let config = SessionConfig {
//!     channels: 1,
//!     backend: Backend::Portable,
//!     parameters: Parameters::keylock(TempoRate::new(0.75)?),
//!     ..SessionConfig::default()
//! };
//! let audio = render(&vec![0.0; 4410], config)?;
//! assert!(audio.samples.len() > 4410);
//! # Ok::<(), timestretch::SessionError>(())
//! ```
//!
//! Preparation allocates. `StreamSession` processing is synchronous and bounded
//! by a work quota; the host chooses its thread and scheduling policy. Offline
//! rendering drives exactly the same adapter. Source frames, collected output,
//! and acknowledged playback progress are distinct. See `examples/realtime_stream.rs`.
#![forbid(unsafe_code)]
pub mod error;
mod render;
mod session;
mod session_types;
mod support;
pub use error::StretchError;
pub use render::{RenderReport, RenderedAudio, render};
pub use session::{PreparedSession, StreamSession};
pub use session_types::*;
#[cfg(feature = "io")]
mod audio;
#[cfg(feature = "io")]
pub use audio::{AudioBuffer, Channels, FrameIter, Sample};
#[cfg(feature = "io")]
pub use support::window::WindowType;
#[cfg(feature = "io")]
pub mod io;
#[cfg(all(feature = "analysis", feature = "io"))]
pub use io::tsa::*;
#[cfg(feature = "analysis")]
pub use timestretch_analysis::*;

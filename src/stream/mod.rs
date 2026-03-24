//! Streaming (chunked) audio processing for real-time use.

pub mod processor;
mod transient_scheduler;

// --- Persistent streaming hybrid engine (scaffolding) ----------------------
// These modules are internal-only for now. They will be wired into
// `StreamProcessor` incrementally behind a new `StreamingEngine` variant.
#[allow(dead_code)]
pub(crate) mod analyzer;
#[allow(dead_code)]
pub(crate) mod mixer;
#[allow(dead_code)]
pub(crate) mod render;
#[allow(dead_code)]
pub(crate) mod router;

pub use processor::{StreamProcessor, TransientResetStats};

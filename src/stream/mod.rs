//! Streaming (chunked) audio processing for real-time use.

pub mod processor;
mod transient_scheduler;

pub use processor::{StreamProcessor, StreamingEngine, TransientResetStats};

#![forbid(unsafe_code)]
//! Tiny hard-RT facade crate.
//!
//! Re-exports the dual-plane RT API from `timestretch`:
//! - `RtProcessor::prepare()`
//! - `RtProcessor::process_block()`
//! - `RtProcessor::flush()`

pub use timestretch::dual_plane::{
    DualPlaneProcessor, LatencyProfile, QualityTier, RenderHints, RtConfig, RtControlSender,
    RtProcessor, TimeWarpMap, WarpAnchor,
};

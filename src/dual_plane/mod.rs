//! Dual-plane streaming architecture.
//!
//! The callback-facing real-time plane only executes bounded kernels.
//! Adaptive analysis and policy decisions run on an asynchronous plane and
//! publish lock-free snapshots to the RT core.

pub mod analysis_plane;
pub mod engine;
pub mod hints;
pub mod quality;
pub mod rt;
pub mod warp_map;

pub use engine::DualPlaneProcessor;
pub use hints::RenderHints;
pub use quality::{LatencyProfile, QualityGovernor, QualityTier, RtGovernorConfig};
pub use rt::{RtConfig, RtControlSender, RtProcessor};
pub use warp_map::{TimeWarpMap, WarpAnchor};

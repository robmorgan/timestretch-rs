//! Decision router that maps analysis events to rendering operations.
//!
//! # Responsibility
//!
//! [`HybridRouter`] receives [`AnalysisEvent`]s from the analyzer and decides,
//! for each timeline region, whether the audio should be rendered by the tonal
//! (phase-vocoder) path, the transient (WSOLA / pass-through) path, or a blend
//! of both.
//!
//! # Design invariants
//!
//! * **Timeline-based decisions.** Routing is expressed as a sequence of
//!   [`RoutedHybridOp`]s keyed by absolute input position, *not* by chunk
//!   index. This decouples the routing logic from callback granularity.
//! * **Deterministic.** Given the same analysis stream, the router always
//!   produces the same operations regardless of how input was chunked.
//! * **Zero allocation in steady state.** The ops buffer is pre-allocated and
//!   reused across callbacks.

use super::analyzer::AnalysisEvent;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which renderer(s) should handle a region of the input timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderPath {
    /// Tonal phase-vocoder only.
    Tonal,
    /// Transient (WSOLA / passthrough) only.
    Transient,
    /// Blend both paths with the given tonal weight in `[0.0, 1.0]`.
    Blend { tonal_weight: f32 },
}

/// A routing decision for a contiguous region of input audio.
///
/// Produced by [`HybridRouter::route`] and consumed by the tonal/transient
/// renderers and the [`super::mixer::TimelineMixer`].
#[derive(Debug, Clone)]
pub struct RoutedHybridOp {
    /// Absolute per-channel input-sample position where this region starts.
    pub input_start: u64,
    /// Length of the region in input samples (per channel).
    pub input_len: usize,
    /// Which renderer(s) to use.
    pub path: RenderPath,
}

/// Routes analysis events to rendering operations.
///
/// Constructed once per stream lifetime. Stateless beyond a small amount of
/// hysteresis to prevent rapid toggling between paths.
pub struct HybridRouter {
    /// Minimum consecutive frames before the router switches paths.
    _hysteresis_frames: usize,
}

impl HybridRouter {
    /// Create a new router.
    pub fn new(hysteresis_frames: usize) -> Self {
        Self {
            _hysteresis_frames: hysteresis_frames,
        }
    }

    /// Convert a batch of analysis events into routing operations.
    ///
    /// The returned ops are sorted by `input_start` and are non-overlapping.
    /// This is a placeholder — the real implementation will apply hysteresis
    /// and tonality thresholds.
    pub fn route(&mut self, _events: &[AnalysisEvent]) -> &[RoutedHybridOp] {
        // Placeholder: no ops emitted yet.
        &[]
    }
}

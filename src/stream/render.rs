//! Persistent tonal and transient renderers for the streaming hybrid engine.
//!
//! # Responsibility
//!
//! This module provides two renderer structs that maintain persistent internal
//! state across callbacks:
//!
//! * [`PersistentTonalRenderer`] — wraps a phase-vocoder whose phase
//!   accumulators, overlap buffers, and hop counters survive between calls.
//! * [`PersistentTransientRenderer`] — wraps a WSOLA / passthrough path with
//!   persistent correlation state.
//!
//! Both renderers accept [`RoutedHybridOp`]s and produce timeline-addressed
//! output patches ([`ScheduledPatch`]) for the mixer.
//!
//! # Design invariants
//!
//! * **Persistent state.** Unlike the legacy hybrid-streaming path, these
//!   renderers are *not* re-created or re-run from scratch each callback.
//!   Phase continuity is maintained across arbitrary callback boundaries.
//! * **No crossfade needed at chunk seams.** Because state persists, the
//!   phase-discontinuity crossfade that `HYBRID_STREAM_CROSSFADE_SAMPLES`
//!   exists to paper over is unnecessary.
//! * **Timeline-addressed output.** Each [`ScheduledPatch`] is tagged with an
//!   absolute output-sample position so the mixer can interleave and overlap
//!   patches from the two renderers without knowing callback boundaries.

use super::router::RoutedHybridOp;
use crate::core::types::StretchParams;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A rendered audio fragment placed on the output timeline.
///
/// Produced by the renderers and consumed by [`super::mixer::TimelineMixer`].
#[derive(Debug, Clone)]
pub struct ScheduledPatch {
    /// Absolute per-channel output-sample position where this patch starts.
    pub output_pos: u64,
    /// Rendered audio samples (single channel, non-interleaved).
    ///
    /// The mixer is responsible for interleaving channels back together.
    pub samples: Vec<f32>,
    /// Which renderer produced this patch, for diagnostic / blending purposes.
    pub source: PatchSource,
}

/// Identifies which renderer produced a [`ScheduledPatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchSource {
    /// Phase-vocoder tonal renderer.
    Tonal,
    /// WSOLA / passthrough transient renderer.
    Transient,
}

/// Persistent phase-vocoder renderer for tonal content.
///
/// Maintains overlap buffers and phase accumulators across callbacks so that
/// phase continuity is preserved without re-rendering historical context.
pub struct PersistentTonalRenderer {
    _params: StretchParams,
    /// Absolute output-sample counter (per-channel).
    _output_pos: u64,
}

impl PersistentTonalRenderer {
    /// Create a new tonal renderer pre-allocated for the given parameters.
    pub fn new(params: &StretchParams) -> Self {
        Self {
            _params: params.clone(),
            _output_pos: 0,
        }
    }

    /// Render tonal regions described by the given operations.
    ///
    /// Returns an empty slice in this scaffolding version.
    pub fn render(&mut self, _ops: &[RoutedHybridOp]) -> &[ScheduledPatch] {
        &[]
    }
}

/// Persistent WSOLA / passthrough renderer for transient content.
///
/// Preserves correlation-search state across callbacks so that transient
/// onsets are not re-detected or double-rendered.
pub struct PersistentTransientRenderer {
    _params: StretchParams,
    /// Absolute output-sample counter (per-channel).
    _output_pos: u64,
}

impl PersistentTransientRenderer {
    /// Create a new transient renderer pre-allocated for the given parameters.
    pub fn new(params: &StretchParams) -> Self {
        Self {
            _params: params.clone(),
            _output_pos: 0,
        }
    }

    /// Render transient regions described by the given operations.
    ///
    /// Returns an empty slice in this scaffolding version.
    pub fn render(&mut self, _ops: &[RoutedHybridOp]) -> &[ScheduledPatch] {
        &[]
    }
}

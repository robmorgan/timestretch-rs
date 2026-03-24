//! Real-time spectral analysis front-end for the persistent streaming hybrid engine.
//!
//! # Responsibility
//!
//! `StreamAnalyzer` consumes interleaved PCM input one callback at a time and
//! produces a stream of [`AnalysisEvent`]s that describe the spectral and
//! transient character of each analysis frame.
//!
//! # Design invariants
//!
//! * **No allocations after construction.** All scratch buffers are pre-allocated
//!   in [`StreamAnalyzer::new`].
//! * **Timeline-addressed output.** Every [`AnalysisEvent`] carries an absolute
//!   input-sample position so downstream stages can correlate analysis with
//!   rendered audio independently of callback boundaries.
//! * **Purely informational.** The analyzer never mutates audio; it only reads
//!   input and emits metadata.

use crate::core::types::StretchParams;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single analysis observation anchored to the input timeline.
///
/// Produced by [`StreamAnalyzer::analyze`] and consumed by [`super::router::HybridRouter`].
#[derive(Debug, Clone)]
pub struct AnalysisEvent {
    /// Absolute sample offset (per-channel) in the input stream where this
    /// analysis window is centred.
    pub input_pos: u64,
    /// `true` when a transient onset is detected within this window.
    pub is_transient: bool,
    /// Estimated tonality ratio in `[0.0, 1.0]`.
    ///
    /// Values near 1.0 indicate strongly tonal content (sustained notes);
    /// values near 0.0 indicate noise-like or percussive content.
    pub tonality: f32,
    /// Short-term RMS energy of the analysis window, linear scale.
    pub energy: f32,
}

/// Real-time spectral analyzer for the persistent streaming hybrid engine.
///
/// Constructed once per stream lifetime; fed input audio each callback via
/// [`analyze`](StreamAnalyzer::analyze).
pub struct StreamAnalyzer {
    _params: StretchParams,
    /// Absolute input-sample counter (per-channel).
    _input_pos: u64,
}

impl StreamAnalyzer {
    /// Create a new analyzer pre-allocated for the given parameters.
    pub fn new(params: &StretchParams) -> Self {
        Self {
            _params: params.clone(),
            _input_pos: 0,
        }
    }
}

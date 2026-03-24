//! Timeline-based output mixer for the persistent streaming hybrid engine.
//!
//! # Responsibility
//!
//! [`TimelineMixer`] collects [`ScheduledPatch`]es from both renderers,
//! resolves overlaps according to the [`RoutedHybridOp`] blend weights, and
//! writes the final interleaved output into the caller's buffer.
//!
//! # Design invariants
//!
//! * **Sorted timeline merge.** Patches are merged in output-position order.
//!   Overlapping regions (from `RenderPath::Blend` ops) are mixed with the
//!   weight specified by the router — no fixed crossfade length.
//! * **Output-position monotonicity.** The mixer tracks the last emitted
//!   output position and asserts that it never moves backward.
//! * **Channel interleaving.** Renderers produce mono patches; the mixer
//!   interleaves them into the caller-supplied buffer according to the
//!   channel count in [`StretchParams`].

use super::render::ScheduledPatch;
use crate::core::types::StretchParams;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Collects rendered patches and produces final interleaved output.
///
/// Constructed once per stream lifetime. Holds a small amount of state to
/// track the output cursor and handle partial-patch carry-over between
/// callbacks.
pub struct TimelineMixer {
    _params: StretchParams,
    /// Absolute per-channel output-sample position of the next sample to emit.
    _output_cursor: u64,
}

impl TimelineMixer {
    /// Create a new mixer for the given stream parameters.
    pub fn new(params: &StretchParams) -> Self {
        Self {
            _params: params.clone(),
            _output_cursor: 0,
        }
    }

    /// Mix scheduled patches into the output buffer.
    ///
    /// Returns the number of interleaved samples written.
    /// This is a placeholder — always returns 0 in the scaffolding version.
    pub fn mix(&mut self, _patches: &[ScheduledPatch], _output: &mut [f32]) -> usize {
        0
    }
}

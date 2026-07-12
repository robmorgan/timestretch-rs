//! Fixed per-profile stage chains.
//!
//! A profile is a compile-time-known chain of stages, not a tuning surface:
//! selecting one picks the whole signal path. Stage 1 ships only the tape
//! chain; the keylock chain (band split + correctors) lands in Stages 2–3.

use crate::engine::stage::Stage;

/// Which fixed stage chain the engine runs after the varispeed head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineProfile {
    /// Pure varispeed: pitch follows tempo, like a tape deck or a turntable
    /// without keylock. Zero pipeline delay; genuinely useful DJ behavior
    /// (and the walking skeleton for everything else).
    #[default]
    Tape,
}

/// Builds the stage chain for a profile. Tape is the empty chain: the
/// varispeed head is the whole signal path.
pub(crate) fn build_stages(profile: EngineProfile) -> Vec<Box<dyn Stage>> {
    match profile {
        EngineProfile::Tape => Vec::new(),
    }
}

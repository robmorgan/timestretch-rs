//! Fixed per-profile stage chains.
//!
//! A profile is a compile-time-known chain of stages, not a tuning surface:
//! selecting one picks the whole signal path.

use crate::engine::stage::Stage;
use crate::engine::stages::keylock::KeylockStage;

/// Which fixed stage chain the engine runs after the varispeed head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineProfile {
    /// Pure varispeed: pitch follows tempo, like a tape deck or a turntable
    /// without keylock. Zero pipeline delay; genuinely useful DJ behavior
    /// (and the walking skeleton for everything else).
    #[default]
    Tape,
    /// Keylock: band split at ~150 Hz; the low band passes un-corrected
    /// through a matching delay (pitch follows tempo — inaudible at DJ
    /// ratios), the high band is pitch-corrected by a small-FFT phase
    /// vocoder at the delay-matched transposition. Pipeline delay ≈ 9.6 ms
    /// at 44.1 kHz.
    Keylock,
}

/// Builds the stage chain for a profile. Tape is the empty chain: the
/// varispeed head is the whole signal path.
pub(crate) fn build_stages(
    profile: EngineProfile,
    sample_rate: u32,
    channels: usize,
) -> Vec<Box<dyn Stage>> {
    match profile {
        EngineProfile::Tape => Vec::new(),
        EngineProfile::Keylock => vec![Box::new(KeylockStage::new(sample_rate, channels))],
    }
}

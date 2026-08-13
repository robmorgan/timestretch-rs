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
    /// Keylock: band split at 120 Hz; the low band passes un-corrected
    /// through a matching delay (pitch follows tempo — inaudible at DJ
    /// ratios), the high band is pitch-corrected by the time-domain SOLA
    /// corrector at the delay-matched transposition. Full keylock through
    /// ±20%, fading to plain varispeed beyond ±35%. Pipeline delay
    /// ≈ 12.7 ms at 44.1 kHz (the primary deck contract).
    Keylock,
    /// Wide-range Master Tempo (CDJ "WIDE" range setting): a big-FFT
    /// identity-locked phase-vocoder corrector keylocks the FULL spectrum
    /// across the engine's whole tempo range (rates 0.25–2.0) with no
    /// correction fade. Pipeline delay ≈ 48.6 ms at 44.1 kHz — a
    /// deliberately different latency contract from [`Self::Keylock`],
    /// reported honestly via the graph; switching profiles is a
    /// seek-priced rebuild, not a live morph (ROADMAP Stage 11).
    WideKeylock,
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
        EngineProfile::WideKeylock => {
            // Stage 19: the direct-ratio wide PV is the graph HEAD for
            // this profile (it owns the tempo axis), so the stage chain
            // is empty like Tape's. The superseded `WideKeylockStage`
            // (Stage 11 topology) lives in git history.
            let _ = (sample_rate, channels);
            Vec::new()
        }
    }
}

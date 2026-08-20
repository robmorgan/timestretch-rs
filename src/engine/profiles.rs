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
    /// Keylock: band split at 120 Hz; each band is pitch-corrected by its
    /// own time-domain corrector at the delay-matched transposition — the
    /// high band by the SOLA corrector, the low band by a period-aligned
    /// SOLA-class bass corrector (ROADMAP Stage 21) that engages beyond
    /// ~±1–2% deviation (mild nudges keep the seam-rigid pitch-follow
    /// bass). Full keylock through ±20%, fading to plain varispeed beyond
    /// ±35%. Pipeline delay ≈ 12.7 ms at 44.1 kHz (the primary deck
    /// contract).
    Keylock,
    /// Wide-range Master Tempo (CDJ "WIDE" range setting): a big-FFT
    /// identity-locked phase-vocoder corrector keylocks the FULL spectrum
    /// across the engine's whole tempo range (rates 0.25–2.0) with no
    /// correction fade. The direct-ratio PV head buffers source-side
    /// LOOKAHEAD rather than delaying output, so like tape the reported
    /// pipeline delay is 0 ms and the first delivered frame is source
    /// frame 0 (ROADMAP Stage 19); switching profiles is a seek-priced
    /// rebuild, not a live morph.
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

//! The keylock chain as one composed stage: band split → (low: delay,
//! high: PV corrector) → re-sum.
//!
//! The low band is deliberately **not** keylocked — its pitch follows tempo
//! (inaudible at DJ ratios; the Stage 2 falsification experiment tests
//! exactly this) — so it only needs a pure delay matched to the corrector's
//! constant latency for the bands to re-sum in time.

use crate::engine::stage::{BlockBuf, Stage, StageCtx, BLOCK_FRAMES};
use crate::engine::stages::band_split::{TwoBandSplit, KEYLOCK_CROSSOVER_HZ};
use crate::engine::stages::delay::FixedDelay;
use crate::engine::stages::pv_corrector::PvCorrector;

/// Two-band keylock stage.
#[derive(Debug)]
pub(crate) struct KeylockStage {
    split: TwoBandSplit,
    low_delay: FixedDelay,
    corrector: PvCorrector,
    low_scratch: [f32; BLOCK_FRAMES],
    high_scratch: [f32; BLOCK_FRAMES],
}

impl KeylockStage {
    pub(crate) fn new(sample_rate: u32, channels: usize) -> Self {
        let corrector = PvCorrector::new(sample_rate, channels);
        Self {
            split: TwoBandSplit::new(KEYLOCK_CROSSOVER_HZ, sample_rate, channels),
            low_delay: FixedDelay::new(corrector.latency_frames(), channels),
            corrector,
            low_scratch: [0.0; BLOCK_FRAMES],
            high_scratch: [0.0; BLOCK_FRAMES],
        }
    }
}

impl Stage for KeylockStage {
    fn process(&mut self, block: &mut BlockBuf, ctx: &StageCtx) {
        // Delay-matched transposition: cancel the pitch shift embedded in
        // THIS audio (the varispeed rate at the block's timeline position),
        // not the control target.
        let transposition = if ctx.embedded_rate.is_finite() && ctx.embedded_rate > 0.0 {
            1.0 / ctx.embedded_rate
        } else {
            1.0
        };
        self.corrector.set_transposition(transposition);

        for ch in 0..block.channels() {
            self.split.process_channel(
                ch,
                block.channel(ch),
                &mut self.low_scratch,
                &mut self.high_scratch,
            );
            self.low_delay.process_channel(ch, &mut self.low_scratch);
            self.corrector.process_channel(ch, &mut self.high_scratch);
            let out = block.channel_mut(ch);
            for (o, (&l, &h)) in out
                .iter_mut()
                .zip(self.low_scratch.iter().zip(self.high_scratch.iter()))
            {
                *o = l + h;
            }
        }
    }

    fn latency_frames(&self) -> usize {
        // The low band's delay is constructed equal to the corrector's
        // constant latency; either is the chain's pipeline delay.
        debug_assert_eq!(
            self.low_delay.latency_frames(),
            self.corrector.latency_frames()
        );
        self.low_delay.latency_frames()
    }

    fn reset(&mut self) {
        self.split.reset();
        self.low_delay.reset();
        self.corrector.reset();
    }
}

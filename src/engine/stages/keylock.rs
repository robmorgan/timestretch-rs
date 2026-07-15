//! The keylock chain as one composed stage: band split → (low: delay,
//! high: SOLA corrector) → re-sum, fading to plain varispeed at extreme
//! rates.
//!
//! The low band is deliberately **not** keylocked — its pitch follows tempo
//! (validated by the Stage 2 falsification listen) — so it only needs a
//! pure delay matched to the corrector's constant nominal lag.
//!
//! The high band is corrected by the time-domain `SolaCorrector` across
//! the ENTIRE corrected range. The chain originally carried a small-FFT
//! phase-vocoder corrector for wide deviations, selected by threshold with
//! a phase-aligned handoff; three owner listening passes (2026-07) rejected
//! the PV at every boundary it was placed behind (phasey/robotic mids and
//! highs at 5%, 9.7%, and 14.1%), and it was deleted at Stage 9. Beyond
//! [`CORRECTION_FADE_START_DEV`] the correction fades toward raw varispeed
//! (deck-stop/spinback territory, where pitch SHOULD follow tempo), with
//! SOLA's transposition clamp soft-flattening the corrected copy so it
//! stays a single coherent pitch through the fade.

use crate::engine::stage::{BlockBuf, Stage, StageCtx, BLOCK_FRAMES};
use crate::engine::stages::band_split::{TwoBandSplit, KEYLOCK_CROSSOVER_HZ};
use crate::engine::stages::delay::FixedDelay;
use crate::engine::stages::sola::SolaCorrector;

/// The keylock chain's constant delay, in frames (12.7 ms at 44.1 kHz,
/// inside the ≤ 15 ms pipeline budget). This is SOLA's nominal elastic
/// lag: it must cover the hard drift trigger plus the splice search range
/// and sinc margins (320 + 160 + 36, with headroom). Historically equal to
/// the deleted PV corrector's latency so the two were interchangeable;
/// kept at that figure — the deck integration and every latency gate are
/// anchored on it, and SOLA needs the headroom regardless.
pub(crate) const KEYLOCK_LATENCY_FRAMES: usize = 560;

/// Beyond this rate deviation the corrector starts fading out toward
/// plain varispeed (pitch follows tempo). Inside the fade the output
/// carries BOTH pitches at complementary weights — audibly doubled/
/// chorused (owner listening 2026-07-14: a full mix "falls apart"
/// approaching −15% with the fade at 0.12, while −11.5% — pure SOLA —
/// was acceptable) — so the fade must not overlap rates a DJ actually
/// plays at: it starts past the ±20% secondary DJ range and serves true
/// extremes only (deck-stop/spinback territory, where pitch SHOULD
/// follow tempo). Within the fade SOLA's transposition clamp (1.35)
/// additionally soft-limits the correction, keeping the corrected copy
/// a single coherent pitch that gradually goes flat rather than grainy.
pub const CORRECTION_FADE_START_DEV: f64 = 0.205;
/// …and fully out here: pure varispeed beyond ~±35%.
pub const CORRECTION_FADE_END_DEV: f64 = 0.35;

/// Two-band keylock stage.
#[derive(Debug)]
pub(crate) struct KeylockStage {
    split: TwoBandSplit,
    low_delay: FixedDelay,
    /// Delayed copy of the RAW high band, kept warm for the extreme-rate
    /// fade-out (aligned with the corrector's constant nominal lag).
    raw_high_delay: FixedDelay,
    sola: SolaCorrector,
    /// Per-channel low-band scratch.
    low: Vec<[f32; BLOCK_FRAMES]>,
    /// Per-channel high band feeding the corrector (in/out in place).
    high: Vec<[f32; BLOCK_FRAMES]>,
    /// Per-channel raw (uncorrected) high band, delayed to alignment.
    high_raw: Vec<[f32; BLOCK_FRAMES]>,
}

impl KeylockStage {
    pub(crate) fn new(sample_rate: u32, channels: usize) -> Self {
        let sola = SolaCorrector::new(channels, KEYLOCK_LATENCY_FRAMES);
        debug_assert_eq!(sola.latency_frames(), KEYLOCK_LATENCY_FRAMES);
        Self {
            split: TwoBandSplit::new(KEYLOCK_CROSSOVER_HZ, sample_rate, channels),
            low_delay: FixedDelay::new(KEYLOCK_LATENCY_FRAMES, channels),
            raw_high_delay: FixedDelay::new(KEYLOCK_LATENCY_FRAMES, channels),
            sola,
            low: vec![[0.0; BLOCK_FRAMES]; channels],
            high: vec![[0.0; BLOCK_FRAMES]; channels],
            high_raw: vec![[0.0; BLOCK_FRAMES]; channels],
        }
    }
}

impl Stage for KeylockStage {
    fn process(&mut self, block: &mut BlockBuf, ctx: &StageCtx<'_>) {
        // Delay-matched transposition: cancel the pitch shift embedded in
        // THIS audio (the varispeed rate at the block's timeline position),
        // not the control target.
        let transposition = if ctx.embedded_rate.is_finite() && ctx.embedded_rate > 0.0 {
            1.0 / ctx.embedded_rate
        } else {
            1.0
        };
        self.sola.set_transposition(transposition);
        // SOLA reads elastically off the nominal lag; give it the local
        // rate slope so its synthesis rate tracks the audio actually under
        // its cursor.
        self.sola.set_rate_slope(ctx.embedded_rate_slope);

        // Split every channel; keep the delayed low band and a delayed raw
        // copy of the high band for the extreme-rate fade-out.
        for ch in 0..block.channels() {
            let (low, high) = (&mut self.low[ch], &mut self.high[ch]);
            self.split.process_channel(ch, block.channel(ch), low, high);
            self.high_raw[ch].copy_from_slice(high);
            self.raw_high_delay
                .process_channel(ch, &mut self.high_raw[ch]);
            self.low_delay.process_channel(ch, low);
        }
        self.sola.process_block(&mut self.high, ctx.onsets);

        // Extreme-rate correction weight: 1 inside the DJ range, fading to
        // plain varispeed (pitch follows tempo) beyond it.
        let deviation = (ctx.embedded_rate - 1.0).abs();
        let correction = ((CORRECTION_FADE_END_DEV - deviation)
            / (CORRECTION_FADE_END_DEV - CORRECTION_FADE_START_DEV))
            .clamp(0.0, 1.0) as f32;

        for ch in 0..block.channels() {
            let out = block.channel_mut(ch);
            for (i, sample) in out.iter_mut().enumerate() {
                let high =
                    correction * self.high[ch][i] + (1.0 - correction) * self.high_raw[ch][i];
                *sample = self.low[ch][i] + high;
            }
        }
    }

    fn latency_frames(&self) -> usize {
        // The low band's delay is constructed equal to the corrector's
        // constant nominal lag; either is the chain's delay.
        debug_assert_eq!(self.low_delay.latency_frames(), KEYLOCK_LATENCY_FRAMES);
        KEYLOCK_LATENCY_FRAMES
    }

    fn reset(&mut self) {
        self.split.reset();
        self.low_delay.reset();
        self.raw_high_delay.reset();
        self.sola.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44_100;

    fn run_blocks(stage: &mut KeylockStage, input: &[f32], rate: f64) -> Vec<f32> {
        let mut block = BlockBuf::new(1);
        let ctx = StageCtx {
            embedded_rate: rate,
            embedded_rate_slope: 0.0,
            onsets: &[],
            modulation_hold: false,
            has_artifact: false,
        };
        let mut out = Vec::with_capacity(input.len());
        for chunk in input.chunks_exact(BLOCK_FRAMES) {
            block.channel_mut(0).copy_from_slice(chunk);
            stage.process(&mut block, &ctx);
            out.extend_from_slice(block.channel(0));
        }
        out
    }

    fn sine(freq: f64, len: usize, amp: f32) -> Vec<f32> {
        (0..len)
            .map(|i| amp * (2.0 * std::f64::consts::PI * freq * i as f64 / SR as f64).sin() as f32)
            .collect()
    }

    #[test]
    fn seam_level_recovers_after_a_nudge() {
        // A platter nudge drifts SOLA's elastic cursor; if that drift PARKS
        // after the nudge, the high band stays time-shifted against the low
        // band's fixed delay and the crossover overlap region cancels —
        // audible as "filtered / lost bass" at rest. Seam-region level must
        // return to its pre-nudge baseline shortly after the gesture.
        // Dominant high content (880 Hz) over a quieter seam tone (170 Hz):
        // splice correlation aligns to the dominant period grid, so any
        // parked drift leaves the seam tone at an arbitrary inter-band
        // phase — the realistic case (a pure seam tone self-aligns and
        // cannot reproduce the bug).
        let mut stage = KeylockStage::new(SR, 1);
        let seam_hz = 170.0;
        let mut phase_seam = 0.0f64;
        let mut phase_hi = 0.0f64;
        let mut block = BlockBuf::new(1);
        let mut collected = Vec::new();
        let total_secs = 8.0;
        let total_blocks = (total_secs * SR as f64 / BLOCK_FRAMES as f64) as usize;
        for bi in 0..total_blocks {
            let t = (bi * BLOCK_FRAMES) as f64 / SR as f64;
            // Rest 2 s, nudge to +4% over 0.2 s, hold 0.1 s, back over
            // 0.2 s, rest for the remainder.
            let rate = if t < 2.0 {
                1.0
            } else if t < 2.2 {
                1.0 + 0.04 * (t - 2.0) / 0.2
            } else if t < 2.3 {
                1.04
            } else if t < 2.5 {
                1.04 - 0.04 * (t - 2.3) / 0.2
            } else {
                1.0
            };
            for s in block.channel_mut(0).iter_mut() {
                phase_seam += 2.0 * std::f64::consts::PI * seam_hz * rate / SR as f64;
                phase_hi += 2.0 * std::f64::consts::PI * 880.0 * rate / SR as f64;
                *s = 0.15 * phase_seam.sin() as f32 + 0.5 * phase_hi.sin() as f32;
            }
            let ctx = StageCtx {
                embedded_rate: rate,
                embedded_rate_slope: 0.0,
                onsets: &[],
                modulation_hold: false,
                has_artifact: false,
            };
            stage.process(&mut block, &ctx);
            collected.extend_from_slice(block.channel(0));
        }

        // Goertzel power at the seam frequency, before vs well after.
        let goertzel = |lo: usize, hi: usize| -> f64 {
            let w = 2.0 * std::f64::consts::PI * seam_hz / SR as f64;
            let coeff = 2.0 * w.cos();
            let (mut s1, mut s2) = (0.0f64, 0.0f64);
            for &x in &collected[lo..hi] {
                let s0 = x as f64 + coeff * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            (s1 * s1 + s2 * s2 - coeff * s1 * s2) / ((hi - lo) as f64 / 2.0).powi(2)
        };
        let sr = SR as usize;
        let baseline = goertzel(sr, 2 * sr); // settled, pre-nudge
        let after = goertzel(6 * sr, 8 * sr); // 3.5 s past the gesture
        let loss_db = 10.0 * (after / baseline).log10();
        println!(
            "seam 170 Hz power: baseline {baseline:.5}, after nudge {after:.5} ({loss_db:+.2} dB)"
        );
        assert!(
            loss_db > -1.5,
            "seam level did not recover after the nudge: {loss_db:+.2} dB \
             (parked SOLA drift de-phasing the bands)"
        );
    }

    #[test]
    fn keylock_holds_pitch_across_the_corrected_range() {
        // SOLA carries the whole corrected range; test a DJ-range rate at
        // full correction and a wide rate still inside the fade start
        // ([`CORRECTION_FADE_START_DEV`] is 0.205 — 1.15 stays inside it).
        for (rate, label) in [(1.03f64, "dj"), (1.15, "wide")] {
            let mut stage = KeylockStage::new(SR, 1);
            // The stage receives already-varispeeded audio: a 440 Hz source
            // arrives pitched to 440 * rate; the corrector must return it
            // to 440.
            let shifted = sine(440.0 * rate, SR as usize * 3, 0.6);
            let out = run_blocks(&mut stage, &shifted, rate);
            let scan = &out[SR as usize..SR as usize * 2];
            let (mut first, mut last, mut count) = (None, None, 0usize);
            for i in 1..scan.len() {
                let (a, b) = (scan[i - 1] as f64, scan[i] as f64);
                if a <= 0.0 && b > 0.0 {
                    let t = (i - 1) as f64 + a / (a - b);
                    if first.is_none() {
                        first = Some(t);
                    }
                    last = Some(t);
                    count += 1;
                }
            }
            let freq = (count - 1) as f64 * SR as f64 / (last.unwrap() - first.unwrap());
            let cents = 1_200.0 * (freq / 440.0).log2();
            assert!(
                cents.abs() < 12.0,
                "{label} rate: pitch off by {cents:.1} cents ({freq:.2} Hz)"
            );
        }
    }
}

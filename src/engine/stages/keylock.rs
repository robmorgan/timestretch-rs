//! The keylock chain as one composed stage: band split → (low: delay,
//! high: corrector) → re-sum, with the high-band corrector selected by
//! transposition magnitude.
//!
//! The low band is deliberately **not** keylocked — its pitch follows tempo
//! (validated by the Stage 2 falsification listen) — so it only needs a
//! pure delay matched to the correctors' shared constant latency.
//!
//! Two high-band correctors run warm at all times:
//!
//! - below ~5% transposition, the time-domain `SolaCorrector` —
//!   near-zero intrinsic latency, transparent transients;
//! - above, the small-FFT `PvCorrector` — the wide-range path.
//!
//! Selection has hysteresis and a minimum dwell so a gesture hovering at
//! the threshold cannot chatter, and the handoff is a linear crossfade
//! between two same-latency, same-pitch outputs — riding through the
//! threshold mid-gesture is inaudible by construction.

use crate::engine::stage::{BlockBuf, Stage, StageCtx, BLOCK_FRAMES};
use crate::engine::stages::band_split::{TwoBandSplit, KEYLOCK_CROSSOVER_HZ};
use crate::engine::stages::delay::FixedDelay;
use crate::engine::stages::pv_corrector::PvCorrector;
use crate::engine::stages::sola::SolaCorrector;

/// SOLA↔PV selection threshold on |T − 1|. A tuning constant (settled with
/// corpus evidence in Stage 7); the hysteresis pair below brackets it.
pub const SOLA_PV_THRESHOLD: f64 = 0.05;
/// Engage SOLA when the transposition deviation falls below this…
const SOLA_ENGAGE_DEV: f64 = 0.045;
/// …and release back to the PV when it rises above this.
const SOLA_RELEASE_DEV: f64 = 0.055;
/// Corrector handoff crossfade length, in frames (~1.1 ms at 44.1 kHz).
/// Deliberately short: the two correctors emit the same pitch but their
/// relative timing sweeps at `|1 − T|` samples per frame (the PV's latency
/// is structural, SOLA's is elastic), so a fade that starts phase-aligned
/// must complete before the sweep re-opens the phase gap (~3 samples move
/// across this fade at the release bound).
const HANDOFF_FRAMES: usize = 48;
/// Minimum blocks between selection flips (~93 ms at 44.1 kHz). Chatter
/// costs only a short fade — both correctors always run — so the dwell is
/// modest; the hysteresis pair carries most of the anti-chatter duty.
const MIN_DWELL_BLOCKS: u32 = 128;
/// A handoff fade (either direction) starts only while the two correctors'
/// outputs measure at least this correlated. Their relative phase sweeps
/// continuously (see `HANDOFF_FRAMES`), passing through alignment once per
/// `period / |1 − T|` frames — ~20–100 ms for high-band content — so the
/// gate opens promptly; it merely picks the aligned instant. The window
/// below is sized so the in-window phase rotation (≤ ~40° at the 7.5%
/// torture extreme and 1 kHz) cannot smear an aligned instant below the
/// threshold — the failure mode of longer windows.
const HANDOFF_ALIGNMENT_MIN: f64 = 0.85;
/// Samples of output history the alignment estimate correlates over.
const ALIGN_WINDOW: usize = 96;
/// Blocks after which a pending handoff is forced without alignment
/// (~250 ms; a backstop for noise-like content, not the normal path).
const HANDOFF_FORCE_BLOCKS: u32 = 344;

/// Two-band keylock stage with corrector selection.
#[derive(Debug)]
pub(crate) struct KeylockStage {
    split: TwoBandSplit,
    low_delay: FixedDelay,
    pv: PvCorrector,
    sola: SolaCorrector,
    /// Per-channel low-band scratch.
    low: Vec<[f32; BLOCK_FRAMES]>,
    /// Per-channel high band feeding the PV corrector.
    high_pv: Vec<[f32; BLOCK_FRAMES]>,
    /// Per-channel high band feeding the SOLA corrector (lockstep API).
    high_sola: Vec<[f32; BLOCK_FRAMES]>,
    /// True when the selector wants SOLA (deviation inside the threshold).
    sola_selected: bool,
    /// Corrector mix: 0.0 = all PV, 1.0 = all SOLA. NaN until first block.
    mix: f32,
    blocks_since_flip: u32,
    /// Blocks the current handoff has waited for alignment.
    handoff_wait_blocks: u32,
    /// True once the pending engage has hard-recentered SOLA.
    engage_recentered: bool,
    /// Rolling channel-mix history of each corrector's output.
    sola_history: [f32; ALIGN_WINDOW],
    pv_history: [f32; ALIGN_WINDOW],
    /// Alignment measured after the previous block's synthesis (selection
    /// runs before synthesis, so it reads one block behind — the phase
    /// sweeps ≤ ~2.5 samples per block, well inside the gate margin).
    last_alignment: f64,
}

impl KeylockStage {
    pub(crate) fn new(sample_rate: u32, channels: usize) -> Self {
        let pv = PvCorrector::new(sample_rate, channels);
        let sola = SolaCorrector::new(channels, pv.latency_frames());
        debug_assert_eq!(pv.latency_frames(), sola.latency_frames());
        Self {
            split: TwoBandSplit::new(KEYLOCK_CROSSOVER_HZ, sample_rate, channels),
            low_delay: FixedDelay::new(pv.latency_frames(), channels),
            sola,
            pv,
            low: vec![[0.0; BLOCK_FRAMES]; channels],
            high_pv: vec![[0.0; BLOCK_FRAMES]; channels],
            high_sola: vec![[0.0; BLOCK_FRAMES]; channels],
            sola_selected: true,
            mix: f32::NAN,
            blocks_since_flip: 0,
            handoff_wait_blocks: 0,
            engage_recentered: false,
            sola_history: [0.0; ALIGN_WINDOW],
            pv_history: [0.0; ALIGN_WINDOW],
            last_alignment: 0.0,
        }
    }

    /// Current corrector mix (0 = PV, 1 = SOLA) — observability for tests.
    #[cfg(test)]
    pub(crate) fn sola_mix(&self) -> f32 {
        self.mix
    }

    /// Updates selection state for this block's transposition and returns
    /// the (start, end) mix for a per-sample handoff ramp.
    ///
    /// A PV→SOLA engage starts immediately: SOLA is inaudible at mix 0, so
    /// its cursor is hard-recentered onto the PV's exact timing first — the
    /// fade starts phase-aligned by construction. A SOLA→PV release first
    /// asks SOLA for a recentering splice (correlation-matched, landing on
    /// the dominant content's period grid nearest zero drift, i.e.
    /// phase-aligned with the PV) and starts fading the moment it lands.
    /// Once moving, a ramp always completes.
    fn update_selection(&mut self, transposition: f64) -> (f32, f32) {
        let deviation = (transposition - 1.0).abs();
        self.blocks_since_flip = self.blocks_since_flip.saturating_add(1);
        if self.blocks_since_flip >= MIN_DWELL_BLOCKS {
            if self.sola_selected && deviation > SOLA_RELEASE_DEV {
                self.sola_selected = false;
                self.blocks_since_flip = 0;
            } else if !self.sola_selected && deviation < SOLA_ENGAGE_DEV {
                self.sola_selected = true;
                self.blocks_since_flip = 0;
            }
        }

        let target = if self.sola_selected { 1.0f32 } else { 0.0 };
        if self.mix.is_nan() {
            // First block: snap — there is nothing to fade from yet.
            self.mix = target;
        }
        let start = self.mix;
        if start == target {
            self.handoff_wait_blocks = 0;
            self.engage_recentered = false;
            return (start, start);
        }
        let at_extreme = start == 0.0 || start == 1.0;
        if at_extreme {
            // Prep once: bound SOLA's elastic time error before fading, so
            // the low band's fixed delay stays honest across the handoff.
            if start == 0.0 && !self.engage_recentered {
                // SOLA is inaudible: hard recenter is free.
                self.sola.recenter_hard();
                self.engage_recentered = true;
            } else if start == 1.0 {
                // SOLA is live: recenter via a correlation-matched splice.
                self.sola.request_recenter_splice();
            }
            // Fade only at a phase-aligned instant (or on force timeout).
            self.handoff_wait_blocks = self.handoff_wait_blocks.saturating_add(1);
            let ready = self.sola.is_recentered() && self.last_alignment >= HANDOFF_ALIGNMENT_MIN;
            if !ready && self.handoff_wait_blocks < HANDOFF_FORCE_BLOCKS {
                return (start, start);
            }
            self.handoff_wait_blocks = 0;
        }
        let step = BLOCK_FRAMES as f32 / HANDOFF_FRAMES as f32;
        let end = if target > start {
            (start + step).min(target)
        } else {
            (start - step).max(target)
        };
        self.mix = end;
        (start, end)
    }

    /// Slides this block's corrector outputs into the alignment histories
    /// and stores their normalized correlation for the next block's gate.
    fn update_alignment(&mut self) {
        self.sola_history.copy_within(BLOCK_FRAMES.., 0);
        self.pv_history.copy_within(BLOCK_FRAMES.., 0);
        let tail = ALIGN_WINDOW - BLOCK_FRAMES;
        for i in 0..BLOCK_FRAMES {
            let (mut a, mut b) = (0.0f32, 0.0f32);
            for ch in 0..self.high_sola.len() {
                a += self.high_sola[ch][i];
                b += self.high_pv[ch][i];
            }
            self.sola_history[tail + i] = a;
            self.pv_history[tail + i] = b;
        }
        let (mut dot, mut a_sq, mut b_sq) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..ALIGN_WINDOW {
            let (a, b) = (self.sola_history[i] as f64, self.pv_history[i] as f64);
            dot += a * b;
            a_sq += a * a;
            b_sq += b * b;
        }
        let norm = (a_sq * b_sq).sqrt();
        self.last_alignment = if norm < 1e-9 {
            1.0 // silence hands off for free
        } else {
            dot / norm
        };
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
        self.pv.set_transposition(transposition);
        self.sola.set_transposition(transposition);

        // Selection first: a recenter (hard or spliced) must move the SOLA
        // cursor BEFORE this block is synthesized, so a starting fade mixes
        // post-recenter audio from its very first sample.
        let (mix_start, mix_end) = self.update_selection(transposition);

        // Split every channel, keep the delayed low band, and fan the high
        // band out to BOTH correctors (both stay warm so a handoff never
        // waits on state convergence).
        for ch in 0..block.channels() {
            let (low, high) = (&mut self.low[ch], &mut self.high_pv[ch]);
            self.split.process_channel(ch, block.channel(ch), low, high);
            self.high_sola[ch].copy_from_slice(high);
            self.low_delay.process_channel(ch, low);
            self.pv.process_channel(ch, high);
        }
        self.sola.process_block(&mut self.high_sola);
        self.update_alignment();

        // Re-sum with a per-sample handoff ramp. Linear (amplitude) mix:
        // the two correctors emit the same pitch at the same latency, so
        // their outputs are correlated and must sum amplitude-complementary.
        for ch in 0..block.channels() {
            let out = block.channel_mut(ch);
            for (i, sample) in out.iter_mut().enumerate() {
                let g = mix_start + (mix_end - mix_start) * (i as f32 / BLOCK_FRAMES as f32);
                let high = g * self.high_sola[ch][i] + (1.0 - g) * self.high_pv[ch][i];
                *sample = self.low[ch][i] + high;
            }
        }
    }

    fn latency_frames(&self) -> usize {
        // The low band's delay is constructed equal to the correctors'
        // shared constant latency; any of the three is the chain's delay.
        debug_assert_eq!(self.low_delay.latency_frames(), self.pv.latency_frames());
        self.low_delay.latency_frames()
    }

    fn reset(&mut self) {
        self.split.reset();
        self.low_delay.reset();
        self.pv.reset();
        self.sola.reset();
        self.sola_selected = true;
        self.mix = f32::NAN;
        self.blocks_since_flip = 0;
        self.handoff_wait_blocks = 0;
        self.engage_recentered = false;
        self.sola_history.fill(0.0);
        self.pv_history.fill(0.0);
        self.last_alignment = 0.0;
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
    fn selects_sola_inside_threshold_and_pv_outside() {
        let mut stage = KeylockStage::new(SR, 1);
        // 600 blocks: dwell (344) + alignment wait (bounded by one phase
        // sweep cycle) + the short ramp.
        let input = sine(500.0, BLOCK_FRAMES * 600, 0.5);
        run_blocks(&mut stage, &input, 1.02); // T ≈ 0.98, dev 2% < 4.5%
        assert_eq!(stage.sola_mix(), 1.0, "small deviation must select SOLA");

        let mut stage = KeylockStage::new(SR, 1);
        run_blocks(&mut stage, &input, 1.10); // T ≈ 0.91, dev 9% > 5.5%
        assert_eq!(stage.sola_mix(), 0.0, "large deviation must select PV");
    }

    #[test]
    fn handoff_ramps_and_hysteresis_dwells() {
        let mut stage = KeylockStage::new(SR, 1);
        let chunk = sine(500.0, BLOCK_FRAMES * 400, 0.5);
        // Start inside the threshold: SOLA.
        run_blocks(&mut stage, &chunk, 1.02);
        assert_eq!(stage.sola_mix(), 1.0);

        // Cross out: after the dwell, the mix must ramp toward PV — and a
        // value hovering between the hysteresis bounds must NOT flip back.
        run_blocks(&mut stage, &chunk, 1.10);
        assert_eq!(stage.sola_mix(), 0.0, "must hand off to PV");
        run_blocks(
            &mut stage,
            &sine(500.0, BLOCK_FRAMES * 40, 0.5),
            1.052, // dev ~4.9%: inside release, outside engage — no flip
        );
        assert_eq!(stage.sola_mix(), 0.0, "hysteresis band must hold PV");
    }

    #[test]
    fn keylock_holds_pitch_in_both_corrector_modes() {
        for (rate, label) in [(1.03f64, "sola"), (1.10, "pv")] {
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
                "{label} mode: pitch off by {cents:.1} cents ({freq:.2} Hz)"
            );
        }
    }
}

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
//! - across the DJ range (rate deviation below ~9%), the time-domain
//!   `SolaCorrector` — near-zero intrinsic latency, transparent
//!   transients, measured-flat top end;
//! - beyond, the small-FFT `PvCorrector` — the wide-range path.
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

/// SOLA↔PV selection threshold on the tempo-rate deviation `|1 − rate|`
/// (the DJ-facing number; symmetric where transposition space is not —
/// rate 0.92 is a 8.7% transposition). A tuning constant, raised three
/// times on owner listening: 5% → 9% (2026-07-12, PV audibly muffled
/// above 5%), 9% → 12% (2026-07-14, PV phasey/robotic at −9.7% on a full
/// mix), and 12% → 22% (2026-07-14 again: the same artifact reappeared at
/// −14.1%, i.e. at every boundary the PV was put behind). The threshold
/// now equals `CORRECTION_FADE_END_DEV` (the fade end): **SOLA owns
/// the entire corrected range**, and the PV engages only past the fade end, where
/// the correction weight is zero and the handoff is inaudible by
/// construction. Measured on the adversarial multitone the two tie in
/// the deep fade region (fade attenuation dominating) and SOLA wins on
/// the up side (−3.9 vs −6.7 dB at 1.141); on real music SOLA's
/// correlation-matched splices land far cleaner than the fixture
/// suggests. The PV is now a candidate for removal from the live chain
/// entirely (it costs the chain's FFT WCET while never audible) — an
/// architectural cleanup awaiting the Stage 7 corpus settlement.
pub const SOLA_PV_THRESHOLD: f64 = CORRECTION_FADE_END_DEV;
/// Engage SOLA when the rate deviation falls below this…
const SOLA_ENGAGE_DEV: f64 = CORRECTION_FADE_END_DEV - 0.005;
/// …and release back to the PV when it rises above this.
const SOLA_RELEASE_DEV: f64 = CORRECTION_FADE_END_DEV + 0.005;
/// Corrector handoff crossfade length, in frames (~1.1 ms at 44.1 kHz).
/// Deliberately short: the two correctors emit the same pitch but their
/// relative timing sweeps at `|1 − T|` samples per frame (the PV's latency
/// is structural, SOLA's is elastic), so a fade that starts phase-aligned
/// must complete before the sweep re-opens the phase gap (~7 samples move
/// across this fade at the 12.5% release bound — acceptable because the
/// extreme-rate correction fade is already blending raw high band in
/// there, masking the handoff).
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
/// a single coherent pitch that gradually goes flat rather than
/// grainy.
const CORRECTION_FADE_START_DEV: f64 = 0.205;
/// …and fully out here: pure varispeed beyond ~±35%.
const CORRECTION_FADE_END_DEV: f64 = 0.35;
/// With SOLA fully carrying the output and the ride settled this close to
/// unity (rate space), the PV's streaming pipeline is flushed once: the PV
/// accumulates an envelope sag riding through unity (known inherited
/// defect — see ROADMAP Stage 3 notes), and shedding that state while the
/// PV is inaudible means a later engage starts clean.
const PV_FLUSH_DEV: f64 = 0.02;
/// The flush re-arms once the ride moves back out beyond this.
const PV_FLUSH_REARM_DEV: f64 = 0.04;
/// Blocks after a flush during which a release handoff must not start:
/// the PV is re-priming its latency of silence, and the alignment gate
/// reads silence as aligned (a fade would swap in a silent corrector).
/// Covers the prime (560 frames) plus one render hop, in blocks.
const PV_FLUSH_COOLDOWN_BLOCKS: u32 = 24;

/// Two-band keylock stage with corrector selection.
#[derive(Debug)]
pub(crate) struct KeylockStage {
    split: TwoBandSplit,
    low_delay: FixedDelay,
    /// Delayed copy of the RAW high band, kept warm for the extreme-rate
    /// corrector fade-out (aligned with the correctors' constant latency).
    raw_high_delay: FixedDelay,
    pv: PvCorrector,
    sola: SolaCorrector,
    /// Per-channel low-band scratch.
    low: Vec<[f32; BLOCK_FRAMES]>,
    /// Per-channel high band feeding the PV corrector.
    high_pv: Vec<[f32; BLOCK_FRAMES]>,
    /// Per-channel high band feeding the SOLA corrector (lockstep API).
    high_sola: Vec<[f32; BLOCK_FRAMES]>,
    /// Per-channel raw (uncorrected) high band, delayed to alignment.
    high_raw: Vec<[f32; BLOCK_FRAMES]>,
    /// True when the selector wants SOLA (deviation inside the threshold).
    sola_selected: bool,
    /// Corrector mix: 0.0 = all PV, 1.0 = all SOLA. NaN until first block.
    mix: f32,
    blocks_since_flip: u32,
    /// Blocks the current handoff has waited for alignment.
    handoff_wait_blocks: u32,
    /// True once the pending engage has hard-recentered SOLA.
    engage_recentered: bool,
    /// Near-unity PV flush is armed (one-shot, re-armed away from unity).
    pv_flush_armed: bool,
    /// Blocks remaining in the post-flush handoff cooldown.
    pv_flush_cooldown: u32,
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
            raw_high_delay: FixedDelay::new(pv.latency_frames(), channels),
            sola,
            pv,
            low: vec![[0.0; BLOCK_FRAMES]; channels],
            high_pv: vec![[0.0; BLOCK_FRAMES]; channels],
            high_sola: vec![[0.0; BLOCK_FRAMES]; channels],
            high_raw: vec![[0.0; BLOCK_FRAMES]; channels],
            sola_selected: true,
            mix: f32::NAN,
            blocks_since_flip: 0,
            handoff_wait_blocks: 0,
            engage_recentered: false,
            pv_flush_armed: true,
            pv_flush_cooldown: 0,
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
        // Rate-space deviation: |1 − rate| = |1 − 1/T|.
        let deviation = (1.0 / transposition - 1.0).abs();
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
            let ready = self.sola.is_recentered()
                && self.last_alignment >= HANDOFF_ALIGNMENT_MIN
                && (start == 0.0 || self.pv_flush_cooldown == 0);
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
    fn process(&mut self, block: &mut BlockBuf, ctx: &StageCtx<'_>) {
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
        // SOLA reads elastically off the nominal lag; give it the local
        // rate slope so its synthesis rate tracks the audio actually under
        // its cursor (the PV reads at a constant lag and needs no slope).
        self.sola.set_rate_slope(ctx.embedded_rate_slope);
        // Artifact events for this block: PV schedules per-band phase
        // resets; SOLA steers splice timing around the same events.
        self.pv
            .begin_block(ctx.onsets, ctx.modulation_hold, ctx.has_artifact);

        // Selection first: a recenter (hard or spliced) must move the SOLA
        // cursor BEFORE this block is synthesized, so a starting fade mixes
        // post-recenter audio from its very first sample.
        let (mix_start, mix_end) = self.update_selection(transposition);

        // Near-unity PV flush: one-shot while SOLA alone is audible.
        let rate_deviation = (ctx.embedded_rate - 1.0).abs();
        self.pv_flush_cooldown = self.pv_flush_cooldown.saturating_sub(1);
        if mix_start == 1.0
            && mix_end == 1.0
            && self.pv_flush_armed
            && rate_deviation < PV_FLUSH_DEV
        {
            self.pv.flush_streaming_pipeline();
            self.pv_flush_armed = false;
            self.pv_flush_cooldown = PV_FLUSH_COOLDOWN_BLOCKS;
        } else if !self.pv_flush_armed && rate_deviation > PV_FLUSH_REARM_DEV {
            self.pv_flush_armed = true;
        }

        // Split every channel, keep the delayed low band, and fan the high
        // band out to BOTH correctors (both stay warm so a handoff never
        // waits on state convergence).
        for ch in 0..block.channels() {
            let (low, high) = (&mut self.low[ch], &mut self.high_pv[ch]);
            self.split.process_channel(ch, block.channel(ch), low, high);
            self.high_sola[ch].copy_from_slice(high);
            // Keep an aligned raw copy warm for the extreme-rate fade-out.
            self.high_raw[ch].copy_from_slice(high);
            self.raw_high_delay
                .process_channel(ch, &mut self.high_raw[ch]);
            self.low_delay.process_channel(ch, low);
            self.pv.process_channel(ch, high);
        }
        self.sola.process_block(&mut self.high_sola, ctx.onsets);

        // Extreme-rate correction weight: 1 inside the DJ range, fading to
        // plain varispeed (pitch follows tempo) beyond it.
        let deviation = (ctx.embedded_rate - 1.0).abs();
        let correction = ((CORRECTION_FADE_END_DEV - deviation)
            / (CORRECTION_FADE_END_DEV - CORRECTION_FADE_START_DEV))
            .clamp(0.0, 1.0) as f32;
        self.update_alignment();

        // Re-sum with a per-sample handoff ramp. Linear (amplitude) mix:
        // the two correctors emit the same pitch at the same latency, so
        // their outputs are correlated and must sum amplitude-complementary.
        for ch in 0..block.channels() {
            let out = block.channel_mut(ch);
            for (i, sample) in out.iter_mut().enumerate() {
                let g = mix_start + (mix_end - mix_start) * (i as f32 / BLOCK_FRAMES as f32);
                let corrected = g * self.high_sola[ch][i] + (1.0 - g) * self.high_pv[ch][i];
                let high = correction * corrected + (1.0 - correction) * self.high_raw[ch][i];
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
        self.raw_high_delay.reset();
        self.pv.reset();
        self.sola.reset();
        self.sola_selected = true;
        self.mix = f32::NAN;
        self.blocks_since_flip = 0;
        self.handoff_wait_blocks = 0;
        self.engage_recentered = false;
        self.pv_flush_armed = true;
        self.pv_flush_cooldown = 0;
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
    fn selects_sola_inside_threshold_and_pv_outside() {
        let mut stage = KeylockStage::new(SR, 1);
        // 600 blocks: dwell (344) + alignment wait (bounded by one phase
        // sweep cycle) + the short ramp.
        let input = sine(500.0, BLOCK_FRAMES * 600, 0.5);
        run_blocks(&mut stage, &input, 1.06); // rate dev 6%: deep inside engage
        assert_eq!(stage.sola_mix(), 1.0, "DJ-range deviation must select SOLA");

        let mut stage = KeylockStage::new(SR, 1);
        run_blocks(&mut stage, &input, 1.40); // rate dev 40% > 35.5% release
        assert_eq!(stage.sola_mix(), 0.0, "extreme deviation must select PV");
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
        run_blocks(&mut stage, &chunk, 1.40);
        assert_eq!(stage.sola_mix(), 0.0, "must hand off to PV");
        run_blocks(
            &mut stage,
            &sine(500.0, BLOCK_FRAMES * 40, 0.5),
            1.35, // rate dev 35%: inside release, outside engage — no flip
        );
        assert_eq!(stage.sola_mix(), 0.0, "hysteresis band must hold PV");
    }

    #[test]
    fn keylock_holds_pitch_in_both_corrector_modes() {
        // SOLA carries the whole corrected range; test a DJ-range rate at
        // full correction and a wide rate where correction still dominates
        // ([`CORRECTION_FADE_START_DEV`] is 0.12 — 1.10 stays inside it).
        for (rate, label) in [(1.03f64, "sola-dj"), (1.10, "sola-wide")] {
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

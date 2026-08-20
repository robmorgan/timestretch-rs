//! The keylock chain as one composed stage: band split → (low: delay,
//! high: SOLA corrector) → re-sum, fading to plain varispeed at extreme
//! rates.
//!
//! The low band is corrected by its own time-domain SOLA-class corrector
//! (`BassSola`, ROADMAP Stage 21 — the Stage 2 "un-keylocked low band won"
//! verdict rejected a VOCODER bass; the time-domain corrector won the
//! blind re-match in all four ±8% conditions). It blends against the
//! pitch-follow delayed copy with the SAME per-frame weight as the high
//! band, so the live keylock toggle and the extreme-rate fade move both
//! bands together.
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

use crate::engine::stage::{BLOCK_FRAMES, BlockBuf, Stage, StageCtx};
use crate::engine::stages::band_split::{KEYLOCK_CROSSOVER_HZ, TwoBandSplit};
use crate::engine::stages::bass_sola::BassSola;
use crate::engine::stages::delay::FixedDelay;
use crate::engine::stages::sola::SolaCorrector;

/// The keylock chain's constant delay, in frames (12.7 ms at 44.1 kHz,
/// inside the ≤ 15 ms pipeline budget). This is SOLA's nominal elastic
/// lag: it must cover the slowdown-side FORCE trigger (the write-head-side
/// cap `SLOWDOWN_FORCE_CAP`, since Stage 18 the binding figure — the
/// stretched hard trigger no longer fits this side) plus the splice search
/// range and sinc margins, asserted at SOLA construction. Historically
/// equal to the deleted PV corrector's latency so the two were
/// interchangeable; kept at that figure — the deck integration and every
/// latency gate are anchored on it, and SOLA needs the headroom
/// regardless.
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

/// Frames the live keylock toggle takes to crossfade between the corrected
/// and raw high band (~11.6 ms at 44.1 kHz). Long enough to be click-free,
/// short enough to feel instant on a deck switch; during the ramp the high
/// band audibly morphs between the two pitches (inherent to any instant
/// keylock toggle — the CDJ master-tempo behavior).
pub const KEYLOCK_TOGGLE_FADE_FRAMES: usize = 512;

/// Two-band keylock stage.
#[derive(Debug)]
pub(crate) struct KeylockStage {
    split: TwoBandSplit,
    low_delay: FixedDelay,
    /// Low-band corrector (Stage 21). Holds the same nominal lag as the
    /// delay, so band alignment and the latency contract are unchanged.
    bass_sola: BassSola,
    /// Delayed copy of the RAW high band, kept warm for the extreme-rate
    /// fade-out (aligned with the corrector's constant nominal lag).
    raw_high_delay: FixedDelay,
    sola: SolaCorrector,
    /// Per-channel low band feeding the corrector (in/out in place).
    low: Vec<[f32; BLOCK_FRAMES]>,
    /// Per-channel raw (pitch-follow) low band, delayed to alignment.
    low_raw: Vec<[f32; BLOCK_FRAMES]>,
    /// Per-channel high band feeding the corrector (in/out in place).
    high: Vec<[f32; BLOCK_FRAMES]>,
    /// Per-channel raw (uncorrected) high band, delayed to alignment.
    high_raw: Vec<[f32; BLOCK_FRAMES]>,
    /// Smoothed keylock-toggle weight chasing `ctx.keylock` at
    /// [`KEYLOCK_TOGGLE_FADE_FRAMES`]. NaN = snap to the target on the next
    /// block (stream start / post-reset: no fade-in from stale state).
    enable: f32,
    /// Smoothed extreme-rate correction weight chasing the per-block
    /// deviation target at the same slew bound as `enable`. The fade
    /// crossfades two DIFFERENTLY-PITCHED copies of the high band, so a
    /// per-block step here is a click source under fast tempo gestures
    /// (Stage 13 review, finding D6). NaN = snap, as `enable`.
    correction: f32,
}

impl KeylockStage {
    pub(crate) fn new(sample_rate: u32, channels: usize) -> Self {
        let sola = SolaCorrector::new(channels, KEYLOCK_LATENCY_FRAMES);
        debug_assert_eq!(sola.latency_frames(), KEYLOCK_LATENCY_FRAMES);
        let bass_sola = BassSola::new(channels, KEYLOCK_LATENCY_FRAMES);
        debug_assert_eq!(bass_sola.latency_frames(), KEYLOCK_LATENCY_FRAMES);
        Self {
            split: TwoBandSplit::new(KEYLOCK_CROSSOVER_HZ, sample_rate, channels),
            low_delay: FixedDelay::new(KEYLOCK_LATENCY_FRAMES, channels),
            bass_sola,
            raw_high_delay: FixedDelay::new(KEYLOCK_LATENCY_FRAMES, channels),
            sola,
            low: vec![[0.0; BLOCK_FRAMES]; channels],
            low_raw: vec![[0.0; BLOCK_FRAMES]; channels],
            high: vec![[0.0; BLOCK_FRAMES]; channels],
            high_raw: vec![[0.0; BLOCK_FRAMES]; channels],
            enable: f32::NAN,
            correction: f32::NAN,
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
        self.bass_sola.set_transposition(transposition);
        for ch in 0..block.channels() {
            let (low, high) = (&mut self.low[ch], &mut self.high[ch]);
            self.split.process_channel(ch, block.channel(ch), low, high);
            self.high_raw[ch].copy_from_slice(high);
            self.raw_high_delay
                .process_channel(ch, &mut self.high_raw[ch]);
            self.low_raw[ch].copy_from_slice(low);
            self.low_delay.process_channel(ch, &mut self.low_raw[ch]);
        }
        self.bass_sola.process_block(&mut self.low, ctx.onsets);
        self.sola.process_block(&mut self.high, ctx.onsets);

        // Extreme-rate correction weight: 1 inside the DJ range, fading to
        // plain varispeed (pitch follows tempo) beyond it.
        let deviation = (ctx.embedded_rate - 1.0).abs();
        let correction_target = ((CORRECTION_FADE_END_DEV - deviation)
            / (CORRECTION_FADE_END_DEV - CORRECTION_FADE_START_DEV))
            .clamp(0.0, 1.0) as f32;

        // Live keylock toggle and extreme-rate fade: chase both targets per
        // sample so a mid-play switch OR a fast tempo gesture through the
        // fade band is a click-free crossfade (the fade blends two
        // differently-pitched copies, so a per-block weight step is a
        // discontinuity between unrelated waveforms). The per-frame weights
        // are shared across channels so the image stays stable through a
        // fade.
        let target = (ctx.keylock.clamp(0.0, 1.0)) as f32;
        if self.enable.is_nan() {
            self.enable = target;
        }
        if self.correction.is_nan() {
            self.correction = correction_target;
        }
        let step = 1.0 / KEYLOCK_TOGGLE_FADE_FRAMES as f32;
        let mut weight_w = [0.0f32; BLOCK_FRAMES];
        let mut enable = self.enable;
        let mut correction = self.correction;
        for w in &mut weight_w {
            enable += (target - enable).clamp(-step, step);
            correction += (correction_target - correction).clamp(-step, step);
            *w = correction * enable;
        }
        self.enable = enable;
        self.correction = correction;

        for ch in 0..block.channels() {
            let out = block.channel_mut(ch);
            for (i, sample) in out.iter_mut().enumerate() {
                let weight = weight_w[i];
                let high = weight * self.high[ch][i] + (1.0 - weight) * self.high_raw[ch][i];
                let low = weight * self.low[ch][i] + (1.0 - weight) * self.low_raw[ch][i];
                *sample = low + high;
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
        self.bass_sola.reset();
        self.raw_high_delay.reset();
        self.sola.reset();
        self.enable = f32::NAN;
        self.correction = f32::NAN;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44_100;

    fn run_blocks(stage: &mut KeylockStage, input: &[f32], rate: f64) -> Vec<f32> {
        run_blocks_keylock(stage, input, rate, 1.0)
    }

    fn run_blocks_keylock(
        stage: &mut KeylockStage,
        input: &[f32],
        rate: f64,
        keylock: f64,
    ) -> Vec<f32> {
        let mut block = BlockBuf::new(1);
        let ctx = StageCtx {
            embedded_rate: rate,
            embedded_rate_slope: 0.0,
            onsets: &[],
            modulation_hold: false,
            has_artifact: false,
            keylock,
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
    fn seam_survives_a_sustained_mild_ride() {
        // ROADMAP Stage 15: a DJ riding the fader gently (±0.6% here) used
        // to let SOLA's drift sawtooth to the full 192-frame trigger,
        // comb-filtering the crossover seam at −7 dB for as long as the
        // ride lasted (rest recovery needs ~150 ms of stillness a moving
        // fader never provides). The mild-motion bounded recenter caps the
        // sawtooth at ~96 frames: measured worst −4.4 dB / riding mean
        // −1.5 dB on this fixture. Gates pin those with margin, plus a
        // splice-count bound so the fix cannot degenerate into churn.
        let seam_hz = 135.0;
        let srf = SR as f64;
        let mut stage = KeylockStage::new(SR, 1);
        let mut block = BlockBuf::new(1);
        let mut collected = Vec::new();
        let (mut ps, mut ph) = (0.0f64, 0.0f64);
        let total_blocks = (30.0 * srf / BLOCK_FRAMES as f64) as usize;
        for bi in 0..total_blocks {
            let t = (bi * BLOCK_FRAMES) as f64 / srf;
            let rate = if t < 2.0 {
                1.0
            } else {
                1.0 + 0.006 * (2.0 * std::f64::consts::PI * 0.3 * (t - 2.0)).sin()
            };
            for s in block.channel_mut(0).iter_mut() {
                ps += 2.0 * std::f64::consts::PI * seam_hz * rate / srf;
                ph += 2.0 * std::f64::consts::PI * 880.0 * rate / srf;
                *s = 0.15 * ps.sin() as f32 + 0.5 * ph.sin() as f32;
            }
            let ctx = StageCtx {
                embedded_rate: rate,
                embedded_rate_slope: 0.0,
                onsets: &[],
                modulation_hold: false,
                has_artifact: false,
                keylock: 1.0,
            };
            stage.process(&mut block, &ctx);
            collected.extend_from_slice(block.channel(0));
        }
        let goertzel = |lo: usize, hi: usize| -> f64 {
            let w = 2.0 * std::f64::consts::PI * seam_hz / srf;
            let coeff = 2.0 * w.cos();
            let (mut s1, mut s2) = (0.0f64, 0.0f64);
            for &x in &collected[lo..hi] {
                let s0 = x as f64 + coeff * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            (s1 * s1 + s2 * s2 - coeff * s1 * s2).sqrt() / ((hi - lo) as f64 / 2.0)
        };
        let win = (0.25 * srf) as usize;
        let baseline = (goertzel(2 * win, 3 * win) + goertzel(3 * win, 4 * win)) / 2.0;
        let mut worst = 0.0f64;
        let mut tail = Vec::new();
        for k in 8..collected.len() / win {
            let db = 20.0 * (goertzel(k * win, (k + 1) * win) / baseline).log10();
            worst = worst.min(db);
            if k * win > collected.len() - 12 * win {
                tail.push(db);
            }
        }
        let tail_mean = tail.iter().sum::<f64>() / tail.len() as f64;
        let splices = stage.sola.splice_count();
        println!(
            "mild ride seam: worst {worst:+.2} dB, tail {tail_mean:+.2} dB, {splices} splices"
        );
        assert!(
            worst > -6.0,
            "seam comb too deep during mild ride: worst {worst:+.2} dB (pre-Stage-15: −7)"
        );
        assert!(
            tail_mean > -3.0,
            "seam still de-phased while riding: tail mean {tail_mean:+.2} dB (pre-Stage-15: −2.5 \
             and not recovering)"
        );
        assert!(
            splices < 400,
            "mild-motion recentering degenerated into splice churn: {splices} over 30 s"
        );
    }

    #[test]
    fn fade_band_rate_steps_are_click_free() {
        // The extreme-rate fade crossfades the corrected and raw high
        // bands — two DIFFERENTLY-PITCHED copies. Rate gestures that jump
        // across the fade band (dev 0.205→0.35) used to step the fade
        // weight once per 32-frame block, splicing between unrelated
        // waveforms mid-tone (Stage 13 review, finding D6). Toggle the
        // rate between the band's edges repeatedly (so some step lands at
        // adverse phase) and bound the output's sample-to-sample delta by
        // the tone's own slew: measured ~0.9x the bound with the
        // per-sample chase, up to ~3.5x with per-block steps.
        let freq = 2_000.0;
        let amp = 0.5f32;
        let mut stage = KeylockStage::new(SR, 1);
        let input = sine(freq, 4 * SR as usize, amp);
        let mut block = BlockBuf::new(1);
        let mut out = Vec::with_capacity(input.len());
        for (bi, chunk) in input.chunks_exact(BLOCK_FRAMES).enumerate() {
            let t = (bi * BLOCK_FRAMES) as f64 / SR as f64;
            // Warm up inside the corrected range, then square-wave across
            // the fade band every ~15 ms.
            let rate = if t < 1.0 || (t / 0.015) as usize % 2 == 0 {
                1.22
            } else {
                1.34
            };
            let ctx = StageCtx {
                embedded_rate: rate,
                embedded_rate_slope: 0.0,
                onsets: &[],
                modulation_hold: false,
                has_artifact: false,
                keylock: 1.0,
            };
            block.channel_mut(0).copy_from_slice(chunk);
            stage.process(&mut block, &ctx);
            out.extend_from_slice(block.channel(0));
        }
        // Skip warm-up + latency; scan the toggling region.
        let start = SR as usize + KEYLOCK_LATENCY_FRAMES;
        let max_delta = out[start..]
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        let tone_slew = amp * 2.0 * std::f32::consts::PI * freq as f32 / SR as f32;
        assert!(
            max_delta < 1.5 * tone_slew,
            "fade-band rate steps click: max sample delta {max_delta:.4} vs tone slew bound \
             {tone_slew:.4}"
        );
    }

    #[test]
    fn rest_recenter_never_degenerates_into_noop_splices() {
        // Regression (autoresearch #62): on highly periodic content the
        // rest-recenter splice search used to let the zero-jump candidate
        // win on correlation (identical audio), freezing the parked drift
        // in a limit cycle of ~500 no-op fades per second — the crossover
        // seam stayed de-phased forever. Reproduced deterministically at
        // 48 kHz with a 135 Hz seam tone under a dominant 880 Hz; the
        // bounded rest splice (residual < REST_SPLICE_DRIFT) plus the
        // rest trim must recover the seam level after the nudge.
        let sr = 48_000u32;
        let srf = sr as f64;
        let seam_hz = 135.0;
        let mut stage = KeylockStage::new(sr, 1);
        let (mut ps, mut ph) = (0.0f64, 0.0f64);
        let mut block = BlockBuf::new(1);
        let mut collected = Vec::new();
        let total_blocks = (8.0 * srf / BLOCK_FRAMES as f64) as usize;
        for bi in 0..total_blocks {
            let t = (bi * BLOCK_FRAMES) as f64 / srf;
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
                ps += 2.0 * std::f64::consts::PI * seam_hz * rate / srf;
                ph += 2.0 * std::f64::consts::PI * 880.0 * rate / srf;
                *s = 0.15 * ps.sin() as f32 + 0.5 * ph.sin() as f32;
            }
            let ctx = StageCtx {
                embedded_rate: rate,
                embedded_rate_slope: 0.0,
                onsets: &[],
                modulation_hold: false,
                has_artifact: false,
                keylock: 1.0,
            };
            stage.process(&mut block, &ctx);
            collected.extend_from_slice(block.channel(0));
        }
        let goertzel = |lo: usize, hi: usize| -> f64 {
            let w = 2.0 * std::f64::consts::PI * seam_hz / srf;
            let coeff = 2.0 * w.cos();
            let (mut s1, mut s2) = (0.0f64, 0.0f64);
            for &x in &collected[lo..hi] {
                let s0 = x as f64 + coeff * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            (s1 * s1 + s2 * s2 - coeff * s1 * s2) / ((hi - lo) as f64 / 2.0).powi(2)
        };
        let sru = sr as usize;
        let baseline = goertzel(sru, 2 * sru);
        let after = goertzel(6 * sru, 8 * sru);
        let loss_db = 10.0 * (after / baseline).log10();
        // And the splice count must be bounded: the old limit cycle fired
        // ~500/s indefinitely (thousands over this run).
        let splices = stage.sola.splice_count();
        println!("48k seam: post-nudge {loss_db:+.2} dB, {splices} splices");
        assert!(
            loss_db > -1.5,
            "48 kHz seam level did not recover: {loss_db:+.2} dB"
        );
        assert!(
            splices < 400,
            "rest splices degenerated: {splices} splices over 8 s"
        );
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
                keylock: 1.0,
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
            let freq = measure_freq(&out[SR as usize..SR as usize * 2]);
            let cents = 1_200.0 * (freq / 440.0).log2();
            assert!(
                cents.abs() < 12.0,
                "{label} rate: pitch off by {cents:.1} cents ({freq:.2} Hz)"
            );
        }
    }

    /// Zero-crossing frequency estimate over a slice (same method as the
    /// pitch-hold test).
    fn measure_freq(scan: &[f32]) -> f64 {
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
        (count - 1) as f64 * SR as f64 / (last.unwrap() - first.unwrap())
    }

    #[test]
    fn keylock_disabled_is_delay_matched_varispeed() {
        // With the toggle off from the first block the stage must pass the
        // varispeeded audio through untouched in pitch and level (the LR8
        // split re-sums to allpass, so compare frequency and RMS at the
        // chain's constant delay, not samples).
        let rate = 1.06f64;
        let mut stage = KeylockStage::new(SR, 1);
        let shifted = sine(440.0 * rate, SR as usize * 3, 0.6);
        let out = run_blocks_keylock(&mut stage, &shifted, rate, 0.0);

        let scan = &out[SR as usize..SR as usize * 2];
        let freq = measure_freq(scan);
        let cents = 1200.0 * (freq / (440.0 * rate)).log2();
        assert!(
            cents.abs() < 3.0,
            "bypassed output re-pitched: off by {cents:.1} cents ({freq:.2} Hz)"
        );
        let rms = |xs: &[f32]| {
            (xs.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / xs.len() as f64).sqrt()
        };
        let level_db = 20.0 * (rms(scan) / rms(&shifted[SR as usize..SR as usize * 2])).log10();
        assert!(
            level_db.abs() < 0.5,
            "bypassed output level off by {level_db:+.2} dB"
        );
    }

    #[test]
    fn keylock_toggle_is_click_free_and_converges() {
        // Toggle off then back on mid-stream at a DJ rate. The output must
        // never step harder than the signal's own slew (no click at either
        // seam), and after each fade the pitch must settle on the mode's
        // target (source pitch corrected vs varispeed pitch).
        let rate = 1.06f64;
        let secs = SR as usize;
        let shifted = sine(440.0 * rate, secs * 6, 0.6);
        let mut stage = KeylockStage::new(SR, 1);

        // One continuous block stream; the toggle flips at 2 s and 4 s.
        let mut out = Vec::with_capacity(shifted.len());
        let mut block = BlockBuf::new(1);
        for (bi, chunk) in shifted.chunks_exact(BLOCK_FRAMES).enumerate() {
            let start = bi * BLOCK_FRAMES;
            let keylock = if (secs * 2..secs * 4).contains(&start) {
                0.0
            } else {
                1.0
            };
            let ctx = StageCtx {
                embedded_rate: rate,
                embedded_rate_slope: 0.0,
                onsets: &[],
                modulation_hold: false,
                has_artifact: false,
                keylock,
            };
            block.channel_mut(0).copy_from_slice(chunk);
            stage.process(&mut block, &ctx);
            out.extend_from_slice(block.channel(0));
        }

        // No click: the largest sample-to-sample step may not exceed the
        // signal's own maximum slew by more than a small margin. A hard
        // switch (no ramp) fails this at both toggle points.
        let max_step = out
            .windows(2)
            .skip(secs / 2) // past cold-start convergence
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        let signal_slew = 0.6 * (2.0 * std::f64::consts::PI * 440.0 * rate / SR as f64) as f32;
        assert!(
            max_step < signal_slew * 1.5,
            "toggle clicked: max step {max_step:.4} vs signal slew {signal_slew:.4}"
        );

        // Converged pitch in each phase (measured well past each fade).
        let corrected = measure_freq(&out[secs..secs * 2]);
        let bypassed = measure_freq(&out[secs * 3..secs * 4]);
        let recorrected = measure_freq(&out[secs * 5..]);
        let cents = |f: f64, target: f64| 1200.0 * (f / target).log2();
        assert!(
            cents(corrected, 440.0).abs() < 12.0,
            "keylock phase off: {corrected:.2} Hz"
        );
        assert!(
            cents(bypassed, 440.0 * rate).abs() < 12.0,
            "bypass phase not at varispeed pitch: {bypassed:.2} Hz"
        );
        assert!(
            cents(recorrected, 440.0).abs() < 12.0,
            "re-enabled phase off: {recorrected:.2} Hz"
        );
    }

    fn zero_crossing_hz(x: &[f32]) -> f64 {
        let mut crossings = 0u32;
        for i in 1..x.len() {
            if x[i - 1] < 0.0 && x[i] >= 0.0 {
                crossings += 1;
            }
        }
        crossings as f64 * SR as f64 / x.len() as f64
    }

    #[test]
    fn low_band_is_keylocked_at_dj_rates() {
        // ROADMAP Stage 21: a 60 Hz fundamental at ±8% must come out at
        // ~60 Hz with keylock on (the varispeed's embedded pitch shift
        // cancelled), where the shipped pre-Stage-21 chain followed
        // pitch (64.8 / 55.2 Hz).
        for rate in [1.08, 0.92] {
            let mut stage = KeylockStage::new(SR, 1);
            // The stage receives varispeed OUTPUT: the source's 60 Hz
            // arrives already pitch-scaled by the embedded rate.
            let input = sine(60.0 * rate, 8 * SR as usize, 0.5);
            let out = run_blocks(&mut stage, &input, rate);
            let f = zero_crossing_hz(&out[2 * SR as usize..]);
            assert!(
                (f - 60.0).abs() / 60.0 < 0.01,
                "rate {rate}: low band at {f:.2} Hz, expected ~60 (keylocked)"
            );
        }
    }

    #[test]
    fn low_band_follows_pitch_with_keylock_off() {
        // Toggle off: the low band blends to the delayed raw copy and its
        // pitch follows tempo, exactly the pre-Stage-21 contract.
        let rate = 1.08;
        let mut stage = KeylockStage::new(SR, 1);
        let input = sine(60.0 * rate, 8 * SR as usize, 0.5);
        let out = run_blocks_keylock(&mut stage, &input, rate, 0.0);
        let f = zero_crossing_hz(&out[2 * SR as usize..]);
        let expect = 60.0 * rate;
        assert!(
            (f - expect).abs() / expect < 0.01,
            "keylock off: low band at {f:.2} Hz, expected ~{expect:.2} (pitch follows)"
        );
    }
}

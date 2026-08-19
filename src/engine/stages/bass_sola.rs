//! Time-domain SOLA-class corrector for the keylock chain's LOW band
//! (ROADMAP Stage 21).
//!
//! The sub-120 Hz scope line (pitch follows tempo) stood since the Stage 2
//! falsification listen — but that rejection was of a VOCODER bass. The
//! Stage 18 exit listen heard the uncorrected bass blind ("bass out of
//! key", ±8% on bass-forward material) once splice granulation stopped
//! masking it, and the Stage 21 kill experiment's time-domain prototype
//! WON the blind re-match in all four conditions (2026-08-19): the
//! detuned bass read as the artifact, the corrected bass read "hitting
//! well / open, clean". This module is that prototype built out to the
//! chain's contracts.
//!
//! Mechanism: the low band is quasi-periodic and band-limited — the ideal
//! SOLA signal class. A ring reader consumes at the transposition rate
//! (cancelling the varispeed pitch shift); the accumulating lag drift is
//! repaid in PERIOD-length jumps, NCC-aligned and hidden under long
//! raised-cosine crossfades. Contract points, each carried over from the
//! high-band corrector at bass scale:
//!
//! - **Lockstep channels**: one splice decision (on the channel mean)
//!   applied to every channel — independent per-channel splices would
//!   decorrelate the low band (the Stage 14 lesson, at splice scale).
//! - **Onset protection**: fades never overlap a sub-bass/low-flux
//!   onset's window (a kick splice reads as a flam); the masked window
//!   just AFTER a hit is preferred splice ground. Gated on per-band flux
//!   so a hi-hat never blocks a bass splice.
//! - **RT budget**: the period estimator is an incremental NCC sweep, a
//!   few lags per block, publishing a completed estimate every ~24 ms —
//!   there is no per-splice full sweep on the audio path.
//! - **Rest recentering** (the seam contract): parked drift time-shifts
//!   the low band against the high band's nominal lag and cancels the
//!   crossover seam. Period jumps cannot repay sub-period drift, so at
//!   rest the residual is recentered outright in quiet moments (hidden
//!   by the fade) or trimmed continuously at an inaudible micro-detune.
//!
//! Holds the chain's nominal lag ([`super::keylock::KEYLOCK_LATENCY_FRAMES`])
//! so the band re-sum stays aligned; the read cursor wobbles elastically
//! by up to ± one bass period around it, like the high-band corrector at
//! larger scale.

use crate::engine::stage::{BLOCK_FRAMES, OnsetEvent};

/// Ring capacity: nominal lag + one max period + correlation window +
/// margins, with headroom (power of two).
const RING_LEN: usize = 8_192;
const RING_MASK: usize = RING_LEN - 1;
/// Bass period search range, frames at 44.1 kHz: ~34–176 Hz.
const PERIOD_MIN: usize = 250;
const PERIOD_MAX: usize = 1_300;
/// NCC window for period estimation and splice alignment: 2+ periods at
/// the 120 Hz band edge (the Stage 3 seam lesson, scaled to this band).
const CORR_WIN: usize = 768;
/// Coarse sweep lag step; a fine pass refines around the coarse peak.
const SWEEP_STEP: usize = 4;
/// Sweep lags evaluated per block: bounds the estimator's per-callback
/// cost (this × CORR_WIN mults) and completes a full sweep in ~33 blocks
/// (~24 ms) — faster than bass periods move.
const SWEEP_LAGS_PER_BLOCK: usize = 8;
/// Minimum NCC peak for the band to count as periodic at all.
const PERIODICITY_MIN: f64 = 0.5;
/// Raised-cosine splice crossfade (~8.7 ms): long enough that residual
/// misalignment reads as a slow level dip, not a phase flick.
const XFADE: usize = 384;
/// Cubic-interp read margin from either ring edge, frames.
const MARGIN: f64 = 8.0;
/// Splice-decision slack kept between a landing and the hard bounds.
const SLACK: f64 = 64.0;
/// Frames between splice attempts after a declined one.
const RETRY_COOLDOWN: usize = 64;
/// Opportunistic splice: repay early when hidden (quiet, or in the
/// masked window after a hit) once this fraction of a period drained.
const OPPORTUNISTIC_DRIFT_FRAC: f64 = 0.25;
/// "Quiet" = short-term energy below this fraction of the rolling mean.
const QUIET_RATIO: f32 = 0.5;
/// Onset protection window (frames around the event a fade must not
/// touch): a kick's sub-bass body rings long past its attack.
const ONSET_PROTECT_PRE: f64 = 128.0;
const ONSET_PROTECT_POST: f64 = 1_536.0;
/// Only onsets with sub-bass or low flux at least this strong protect
/// (beats and flux-less artifacts publish 1.0 and always qualify).
const ONSET_FLUX_MIN: f32 = 0.25;
/// Masked window after a qualifying hit: the best splice hiding place.
const MASKED_WINDOW_START: f64 = 1_024.0;
const MASKED_WINDOW_END: f64 = 3_072.0;
/// Transposition clamp, matching the high-band corrector's soft limit:
/// beyond it the extreme-rate fade owns the output anyway.
const TRANSPOSITION_CLAMP: f64 = 1.35;
/// Correction engagement ramp on |1 - transposition|: below the start,
/// the bass follows pitch exactly (read at the write rate — lag pinned,
/// the crossover seam stays rigid, and the detune is inaudible at sub
/// frequencies: ≤ ~17 cents); full correction by the end, where the
/// out-of-key evidence lives (±8% ≈ 133 cents). This is the low-band
/// analog of the high-band mild-motion contract (Stage 15): under mild
/// rides the seam must not comb, and period-sized jumps cannot keep
/// drift tight enough — so mild rides do not correct at all.
const CORRECT_RAMP_START: f64 = 0.010;
const CORRECT_RAMP_END: f64 = 0.020;
/// Rest detection (|1 - transposition| below this) and the micro-trim
/// slew used to repay sub-period drift at rest: ~3.5 cents on the sub
/// band, inaudible, converging a worst-case half-period park in ~4 s.
const REST_DEV: f64 = 0.003;
const REST_TRIM_MAX: f64 = 0.002;
/// At rest, drift below this is close enough (seam phase error at the
/// 120 Hz band edge stays under ~4°).
const REST_DRIFT_DONE: f64 = 4.0;

/// Incremental period-sweep state.
#[derive(Debug)]
struct Sweep {
    next_lag: usize,
    best_lag: usize,
    best_corr: f64,
}

/// SOLA-class low-band corrector; one instance corrects every channel in
/// lockstep.
#[derive(Debug)]
pub(crate) struct BassSola {
    /// Per audio channel ring.
    rings: Vec<Vec<f32>>,
    /// Channel-mean ring driving every decision (period, NCC, energy).
    mono: Vec<f32>,
    /// Absolute frames written (ring index = write & RING_MASK).
    write: u64,
    /// Absolute read cursor (fractional), shared by all channels.
    read: f64,
    /// Crossfade source cursor while a splice fade is active.
    old_read: f64,
    /// Remaining fade frames (0 = no fade in flight).
    fade_left: usize,
    /// Short-term energy (fast) and rolling mean (slow), on the mean.
    env_fast: f32,
    env_slow: f32,
    /// Frames until the next splice attempt after a declined one.
    retry_cooldown: usize,
    sweep: Sweep,
    /// Last completed period estimate (None until the band shows
    /// periodicity above [`PERIODICITY_MIN`]).
    period: Option<usize>,
    nominal_lag: usize,
    transposition: f64,
    /// Splice audit trail for tests: (read stage frame, jump size).
    #[cfg(test)]
    splice_log: Vec<(f64, f64)>,
}

impl BassSola {
    pub(crate) fn new(num_channels: usize, nominal_lag: usize) -> Self {
        Self {
            rings: (0..num_channels).map(|_| vec![0.0; RING_LEN]).collect(),
            mono: vec![0.0; RING_LEN],
            // Pre-fill the nominal lag with silence so read starts at 0;
            // ingested frame k sits at ring position nominal_lag + k, so
            // stage frame = ring position - nominal_lag.
            write: nominal_lag as u64,
            read: 0.0,
            old_read: 0.0,
            fade_left: 0,
            env_fast: 0.0,
            env_slow: 0.0,
            retry_cooldown: 0,
            sweep: Sweep {
                next_lag: PERIOD_MIN,
                best_lag: 0,
                best_corr: -1.0,
            },
            period: None,
            nominal_lag,
            transposition: 1.0,
            #[cfg(test)]
            splice_log: Vec::new(),
        }
    }

    pub(crate) fn set_transposition(&mut self, transposition: f64) {
        self.transposition = if transposition.is_finite() && transposition > 0.0 {
            transposition.clamp(1.0 / TRANSPOSITION_CLAMP, TRANSPOSITION_CLAMP)
        } else {
            1.0
        };
    }

    pub(crate) fn latency_frames(&self) -> usize {
        self.nominal_lag
    }

    pub(crate) fn reset(&mut self) {
        for ring in &mut self.rings {
            ring.iter_mut().for_each(|s| *s = 0.0);
        }
        self.mono.iter_mut().for_each(|s| *s = 0.0);
        self.write = self.nominal_lag as u64;
        self.read = 0.0;
        self.old_read = 0.0;
        self.fade_left = 0;
        self.env_fast = 0.0;
        self.env_slow = 0.0;
        self.retry_cooldown = 0;
        self.sweep = Sweep {
            next_lag: PERIOD_MIN,
            best_lag: 0,
            best_corr: -1.0,
        };
        self.period = None;
        self.transposition = 1.0;
        #[cfg(test)]
        self.splice_log.clear();
    }

    /// Cubic (Catmull-Rom) read from one channel ring; the band is ~370x
    /// oversampled at 44.1 kHz so interpolation error is negligible.
    #[inline]
    fn sample(ring: &[f32], pos: f64) -> f32 {
        let i = pos.floor();
        let frac = (pos - i) as f32;
        let i = i as i64;
        let at = |k: i64| ring[((i + k) as u64 & RING_MASK as u64) as usize];
        let (p0, p1, p2, p3) = (at(-1), at(0), at(1), at(2));
        let a = 0.5 * (3.0 * (p1 - p2) + p3 - p0);
        let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
        let c = 0.5 * (p2 - p0);
        p1 + frac * (c + frac * (b + frac * a))
    }

    /// Normalized cross-correlation between the mono windows ENDING at
    /// `a` and `b`: cursors can sit near the write head, so only history
    /// behind them is guaranteed written.
    fn ncc(&self, a: u64, b: u64) -> f64 {
        let (mut dot, mut ea, mut eb) = (0.0f64, 0.0f64, 0.0f64);
        for k in 1..=CORR_WIN as u64 {
            let x = self.mono[((a.wrapping_sub(k)) & RING_MASK as u64) as usize] as f64;
            let y = self.mono[((b.wrapping_sub(k)) & RING_MASK as u64) as usize] as f64;
            dot += x * y;
            ea += x * x;
            eb += y * y;
        }
        dot / (ea * eb).sqrt().max(1e-12)
    }

    /// Advances the incremental period sweep by up to `budget` coarse
    /// lags (anchored at the write head), publishing a completed estimate
    /// when the sweep wraps.
    fn advance_sweep(&mut self, budget: usize) {
        let anchor = self.write;
        for _ in 0..budget {
            let lag = self.sweep.next_lag;
            let c = self.ncc(anchor, anchor.wrapping_sub(lag as u64));
            if c > self.sweep.best_corr {
                self.sweep.best_corr = c;
                self.sweep.best_lag = lag;
            }
            self.sweep.next_lag += SWEEP_STEP;
            if self.sweep.next_lag > PERIOD_MAX {
                // Fine pass around the coarse peak, then publish.
                let coarse = self.sweep.best_lag;
                let mut best = (coarse, self.sweep.best_corr);
                let lo = coarse.saturating_sub(SWEEP_STEP - 1).max(PERIOD_MIN);
                for lag in lo..=(coarse + SWEEP_STEP - 1).min(PERIOD_MAX) {
                    let c = self.ncc(anchor, anchor.wrapping_sub(lag as u64));
                    if c > best.1 {
                        best = (lag, c);
                    }
                }
                self.period = (best.1 > PERIODICITY_MIN).then_some(best.0);
                self.sweep = Sweep {
                    next_lag: PERIOD_MIN,
                    best_lag: 0,
                    best_corr: -1.0,
                };
                return;
            }
        }
    }

    /// Whether a fade span starting at ring position `start` overlaps any
    /// qualifying onset's protection window (stage frame = ring position
    /// minus the silent prefill).
    fn span_hits_onset(&self, onsets: &[OnsetEvent], start: f64, t: f64) -> bool {
        let base = self.nominal_lag as f64;
        let span = XFADE as f64 * t.max(1.0);
        onsets.iter().any(|event| {
            (event.band_flux[0].max(event.band_flux[1]) >= ONSET_FLUX_MIN)
                && start - base < event.stage_frame + ONSET_PROTECT_POST
                && start - base + span > event.stage_frame - ONSET_PROTECT_PRE
        })
    }

    /// Whether the read cursor sits in the masked window just after a
    /// qualifying hit — the best hiding place for a splice.
    fn in_masked_window(&self, onsets: &[OnsetEvent]) -> bool {
        let read_stage = self.read - self.nominal_lag as f64;
        onsets.iter().any(|event| {
            (event.band_flux[0].max(event.band_flux[1]) >= ONSET_FLUX_MIN)
                && (MASKED_WINDOW_START..MASKED_WINDOW_END)
                    .contains(&(read_stage - event.stage_frame))
        })
    }

    /// Processes one fixed block for every channel in lockstep: ingests,
    /// advances the period sweep, emits at the transposition rate, and
    /// splices when the elastic lag calls for it.
    pub(crate) fn process_block(&mut self, io: &mut [[f32; BLOCK_FRAMES]], onsets: &[OnsetEvent]) {
        debug_assert_eq!(io.len(), self.rings.len());
        // Correction engagement: below the ramp the effective
        // transposition is unity (pitch follows, lag pinned); above it,
        // the full delay-matched transposition.
        let dev = (1.0 - self.transposition).abs();
        let c =
            ((dev - CORRECT_RAMP_START) / (CORRECT_RAMP_END - CORRECT_RAMP_START)).clamp(0.0, 1.0);
        let t = 1.0 + c * (self.transposition - 1.0);
        let nominal = self.nominal_lag as f64;
        let channels = self.rings.len();

        self.advance_sweep(SWEEP_LAGS_PER_BLOCK);

        for i in 0..BLOCK_FRAMES {
            // Ingest every channel and the decision mean.
            let widx = (self.write & RING_MASK as u64) as usize;
            let mut mean = 0.0f32;
            for (ch, ring) in self.rings.iter_mut().enumerate() {
                let x = io[ch][i];
                ring[widx] = x;
                mean += x;
            }
            mean /= channels as f32;
            self.mono[widx] = mean;
            self.write += 1;
            self.env_fast += 0.02 * (mean * mean - self.env_fast);
            self.env_slow += 0.0005 * (mean * mean - self.env_slow);

            // Rest micro-trim: while correction is disengaged, repay
            // residual sub-period drift by a bounded read-rate detune —
            // the seam contract.
            let mut advance = t;
            if c == 0.0 {
                let drift = self.write as f64 - self.read - nominal;
                if drift.abs() > REST_DRIFT_DONE {
                    advance += (drift * 0.001).clamp(-REST_TRIM_MAX, REST_TRIM_MAX);
                }
            }

            // Emit (all channels share the cursor and fade weights).
            if self.fade_left > 0 {
                let k = self.fade_left as f32 / XFADE as f32;
                let w_old = 0.5 - 0.5 * (core::f32::consts::PI * k).cos();
                for (ch, ring) in self.rings.iter().enumerate() {
                    let a = Self::sample(ring, self.old_read);
                    let b = Self::sample(ring, self.read);
                    io[ch][i] = w_old * a + (1.0 - w_old) * b;
                }
                self.old_read += advance;
                self.fade_left -= 1;
            } else {
                for (ch, ring) in self.rings.iter().enumerate() {
                    io[ch][i] = Self::sample(ring, self.read);
                }
            }
            self.read += advance;

            if self.fade_left > 0 {
                continue; // one splice in flight at a time
            }
            self.splice_decision(t, nominal, onsets);
        }
    }

    /// Considers one splice at the current cursor state.
    fn splice_decision(&mut self, t: f64, nominal: f64, onsets: &[OnsetEvent]) {
        // Positive drift = read is late (lag grew, transposing down);
        // negative = read is catching the write head (transposing up).
        // The lag change per frame is `v = 1 - t`; repayment jumps must
        // go AGAINST v only — a magnitude trigger ping-pongs (a period
        // jump lands the drift on the far side of the corridor,
        // re-triggers, and the counter-jump gets clamped and breaks
        // alignment).
        let lag = self.write as f64 - self.read;
        let drift = lag - nominal;
        let v = 1.0 - t;
        // Hard bounds keep the cursor (and a full fade + corr window)
        // inside valid history regardless of periodicity.
        let fade_travel = XFADE as f64 * t.max(1.0);
        let hard_low = MARGIN + fade_travel + 16.0;
        let hard_high = (RING_LEN - CORR_WIN) as f64 - fade_travel - 16.0;
        let forced = lag < hard_low || lag > hard_high;
        let at_rest = v.abs() < REST_DEV;
        let quiet = self.env_fast < QUIET_RATIO * self.env_slow;
        debug_assert!(REST_DEV < CORRECT_RAMP_START); // rest ⊂ disengaged
        if !forced {
            if self.retry_cooldown > 0 {
                self.retry_cooldown -= 1;
                return;
            }
            if at_rest {
                // Rest recentering: sub-period residue is the trim's job;
                // a QUIET gap lets the whole residual go at once (the
                // fade hides an unaligned jump only when nothing rings).
                if quiet && drift.abs() > SLACK && !self.span_hits_onset(onsets, self.read, t) {
                    self.recenter_splice(drift, hard_low, hard_high);
                }
                return;
            }
            let p = self.period.unwrap_or(PERIOD_MIN) as f64;
            let hidden = quiet || self.in_masked_window(onsets);
            // Directional thresholds. Draining DOWN (v < 0) the hard
            // floor sits only ~120 frames below nominal, so the splice
            // must come EARLY — still ~p/4 above nominal — or every
            // splice arrives forced (and forced splices skip onset
            // protection). Draining UP there are thousands of frames of
            // room, so waiting half a period keeps splices sparse.
            let due = if v < 0.0 {
                drift < p * 0.25 || (hidden && drift < p * 0.5)
            } else {
                drift > p * 0.5 || (hidden && drift > p * OPPORTUNISTIC_DRIFT_FRAC)
            };
            if !due {
                return;
            }
        }

        // Repay whole periods against the drain direction. The jump count
        // comes from the ROOM on the landing side, not from rounding the
        // drift: the corridor is asymmetric (only ~150 frames below
        // nominal before the hard floor, thousands above), and a jump
        // that has to be clamped is a jump that breaks period alignment.
        let jump = match self.period {
            Some(p) => {
                let p = p as f64;
                let n = if v > 0.0 {
                    ((lag - (hard_low + SLACK)) / p).floor()
                } else {
                    let n_max = ((hard_high - SLACK - lag) / p).floor();
                    (-drift / p).round().max(1.0).min(n_max)
                };
                if n < 1.0 {
                    if forced {
                        drift // no aligned jump fits: recenter outright
                    } else {
                        self.retry_cooldown = RETRY_COOLDOWN;
                        return;
                    }
                } else if v > 0.0 {
                    n * p
                } else {
                    -n * p
                }
            }
            // No periodicity worth aligning to (silence): recenter
            // outright — but only when the corridor demands it; splicing
            // mid-note without alignment is the one guaranteed-audible
            // move.
            None => {
                if forced {
                    drift
                } else {
                    self.retry_cooldown = RETRY_COOLDOWN;
                    return;
                }
            }
        };

        // Onset protection: neither fade span may cover a protected hit.
        // Forced splices proceed regardless (the alternative is reading
        // unwritten memory).
        let target =
            (self.read + jump).clamp(self.write as f64 - hard_high, self.write as f64 - hard_low);
        if !forced
            && (self.span_hits_onset(onsets, self.read, t)
                || self.span_hits_onset(onsets, target, t))
        {
            self.retry_cooldown = RETRY_COOLDOWN / 2;
            return;
        }
        if (target - self.read).abs() < 1.0 {
            return;
        }
        #[cfg(test)]
        self.splice_log
            .push((self.read - self.nominal_lag as f64, target - self.read));
        self.old_read = self.read;
        self.read = target;
        self.fade_left = XFADE;
    }

    /// Unconditional recenter splice (rest path): jump the whole drift.
    fn recenter_splice(&mut self, drift: f64, hard_low: f64, hard_high: f64) {
        let target =
            (self.read + drift).clamp(self.write as f64 - hard_high, self.write as f64 - hard_low);
        if (target - self.read).abs() < 1.0 {
            return;
        }
        #[cfg(test)]
        self.splice_log
            .push((self.read - self.nominal_lag as f64, target - self.read));
        self.old_read = self.read;
        self.read = target;
        self.fade_left = XFADE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAG: usize = 560;

    fn run(bass: &mut BassSola, input: &[f32], t: f64, onsets: &[OnsetEvent]) -> Vec<f32> {
        bass.set_transposition(t);
        let mut out = Vec::with_capacity(input.len());
        for chunk in input.chunks(BLOCK_FRAMES) {
            let mut block = [[0.0f32; BLOCK_FRAMES]];
            block[0][..chunk.len()].copy_from_slice(chunk);
            bass.process_block(&mut block, onsets);
            out.extend_from_slice(&block[0][..chunk.len()]);
        }
        out
    }

    fn sine(freq: f64, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 44_100.0).sin() as f32 * 0.5)
            .collect()
    }

    fn dominant_freq_zc(x: &[f32]) -> f64 {
        let mut crossings = 0u32;
        for i in 1..x.len() {
            if x[i - 1] < 0.0 && x[i] >= 0.0 {
                crossings += 1;
            }
        }
        crossings as f64 * 44_100.0 / x.len() as f64
    }

    #[test]
    fn unity_transposition_is_a_pure_delay() {
        let mut bass = BassSola::new(1, LAG);
        let input = sine(60.0, 44_100);
        let out = run(&mut bass, &input, 1.0, &[]);
        for i in 20_000..40_000 {
            assert!(
                (out[i] - input[i - LAG]).abs() < 1e-3,
                "unity path deviates at {i}"
            );
        }
    }

    #[test]
    fn transposition_moves_the_fundamental_and_stays_smooth() {
        for (t, f_in) in [(1.087, 55.0), (0.926, 55.0), (1.087, 110.0)] {
            let mut bass = BassSola::new(1, LAG);
            let input = sine(f_in, 6 * 44_100);
            let out = run(&mut bass, &input, t, &[]);
            let settled = &out[44_100..];
            let f_out = dominant_freq_zc(settled);
            let expect = f_in * t;
            assert!(
                (f_out - expect).abs() / expect < 0.01,
                "t={t}: fundamental {f_out:.2} Hz, expected {expect:.2}"
            );
            // Splices must be inaudible on a steady tone: adjacent-sample
            // steps bounded by the tone's own max slope with slack.
            let max_step = settled
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold(0.0f32, f32::max);
            let tone_slope =
                (2.0 * std::f64::consts::PI * expect / 44_100.0) as f32 * 0.5 * 1.5 + 1e-3;
            assert!(
                max_step < tone_slope,
                "t={t}: splice step {max_step} vs tone slope bound {tone_slope}"
            );
        }
    }

    #[test]
    fn stereo_channels_stay_identical_through_splices() {
        // Lockstep contract: identical inputs must produce bit-identical
        // outputs on both channels — one splice decision for the pair.
        let mut bass = BassSola::new(2, LAG);
        bass.set_transposition(1.08);
        let input = sine(60.0, 4 * 44_100);
        for chunk in input.chunks(BLOCK_FRAMES) {
            let mut block = [[0.0f32; BLOCK_FRAMES]; 2];
            block[0][..chunk.len()].copy_from_slice(chunk);
            block[1][..chunk.len()].copy_from_slice(chunk);
            bass.process_block(&mut block, &[]);
            assert_eq!(block[0], block[1], "channels diverged");
        }
        assert!(
            !bass.splice_log.is_empty(),
            "the run must actually have spliced"
        );
    }

    #[test]
    fn splices_avoid_protected_onsets_when_drift_allows() {
        // Kicks every ~0.47 s (128 BPM) with sub-bass flux; at ±8% the
        // drift budget between kicks is ample, so no discretionary fade
        // may overlap a protection window.
        let mut bass = BassSola::new(1, LAG);
        let input = sine(55.0, 6 * 44_100);
        let kick_spacing = 20_672.0;
        // k starts at 1: a kick at stage frame 0 collides with the one
        // unavoidable cold-start forced splice (the first drain cycle has
        // no post-jump headroom yet) — a start-of-stream corner, not the
        // steady state under test.
        let onsets: Vec<OnsetEvent> = (1..14)
            .map(|k| OnsetEvent {
                stage_frame: k as f64 * kick_spacing,
                strength: 1.0,
                beat: false,
                band_flux: [1.0, 0.6, 0.1, 0.1],
            })
            .collect();
        let _ = run(&mut bass, &input, 1.087, &onsets);
        assert!(!bass.splice_log.is_empty(), "must splice at ±8%");
        let span = XFADE as f64 * 1.087;
        for &(read_stage, jump) in &bass.splice_log {
            for span_start in [read_stage, read_stage + jump] {
                for k in 1..14 {
                    let onset = k as f64 * kick_spacing;
                    let overlaps = span_start < onset + ONSET_PROTECT_POST
                        && span_start + span > onset - ONSET_PROTECT_PRE;
                    assert!(
                        !overlaps,
                        "fade at stage frame {span_start:.0} overlaps kick at {onset:.0}"
                    );
                }
            }
        }
    }

    #[test]
    fn rest_recenters_parked_drift() {
        // Ride at +8% long enough to park drift, then rest: the parked
        // lag must converge back toward nominal (the seam contract).
        let mut bass = BassSola::new(1, LAG);
        let _ = run(&mut bass, &sine(60.0, 4 * 44_100), 1.08, &[]);
        let parked = (bass.write as f64 - bass.read - LAG as f64).abs();
        let _ = run(&mut bass, &sine(60.0, 8 * 44_100), 1.0, &[]);
        let settled = (bass.write as f64 - bass.read - LAG as f64).abs();
        assert!(
            settled < REST_DRIFT_DONE + 1.0,
            "rest drift {settled:.1} (parked {parked:.1}) did not recenter"
        );
    }
}

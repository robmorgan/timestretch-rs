//! PROTO (ROADMAP Stage 21 kill experiment, env-gated off by default):
//! a time-domain SOLA-class corrector for the keylock chain's LOW band.
//!
//! The sub-120 Hz scope line (pitch follows tempo) was set by the Stage 2
//! falsification listen — but that rejection was of a VOCODER bass; a
//! time-domain corrector was never tried, and the Stage 18 exit listen
//! heard the uncorrected bass blind ("bass out of key", ±8% on
//! bass-forward material) once the splice granulation stopped masking it.
//!
//! Mechanism: the low band is quasi-periodic and band-limited, the ideal
//! SOLA signal class. A ring reader consumes at the transposition rate
//! (cancelling the varispeed pitch shift); the accumulating lag drift is
//! repaid in PERIOD-length jumps, aligned by normalized cross-correlation
//! with a period-scaled window and hidden under long raised-cosine
//! crossfades, preferring quiet moments (kick protection: don't splice on
//! top of an onset if the drift budget allows waiting).
//!
//! Holds the chain's nominal lag ([`KEYLOCK_LATENCY_FRAMES`]) so the band
//! re-sum stays aligned; the read cursor wobbles elastically around it by
//! up to ± one bass period, the same contract as the high-band corrector
//! at larger scale. Not RT-vetted (period search is a full NCC sweep) and
//! not wired to the extreme-rate fade — this exists to answer one blind
//! question, not to ship.

use crate::engine::stage::BLOCK_FRAMES;

/// Ring capacity: nominal lag + one max period + correlation window +
/// interp margins, with headroom (power of two).
const RING_LEN: usize = 8_192;
const RING_MASK: usize = RING_LEN - 1;
/// Bass period search range, frames at 44.1 kHz: ~34–176 Hz.
const PERIOD_MIN: usize = 250;
const PERIOD_MAX: usize = 1_300;
/// NCC window for period estimation and splice alignment: 2+ periods at
/// the 120 Hz band edge (the Stage 3 seam lesson, scaled to this band).
const CORR_WIN: usize = 768;
/// Raised-cosine splice crossfade (~8.7 ms): long enough that a residual
/// misalignment reads as a slow level dip, not a phase flick.
const XFADE: usize = 384;
/// Cubic-interp read margin from either ring edge, frames.
const MARGIN: f64 = 8.0;
/// Opportunistic splice: repay early when the band is quiet and at least
/// this fraction of a period of drift has accumulated.
const OPPORTUNISTIC_DRIFT_FRAC: f64 = 0.25;
/// "Quiet" = short-term energy below this fraction of the rolling mean.
const QUIET_RATIO: f32 = 0.5;

#[derive(Debug)]
struct BassChannel {
    ring: Vec<f32>,
    /// Absolute frames written (ring index = write & RING_MASK).
    write: u64,
    /// Absolute read cursor (fractional).
    read: f64,
    /// Crossfade source cursor while a splice fade is active.
    old_read: f64,
    /// Remaining fade frames (0 = no fade in flight).
    fade_left: usize,
    /// Short-term energy (fast) and rolling mean (slow), for quiet gating.
    env_fast: f32,
    env_slow: f32,
    /// Frames until the next splice attempt after a declined one (the
    /// period sweep is not per-sample cheap).
    retry_cooldown: usize,
}

#[derive(Debug)]
pub(crate) struct BassSola {
    channels: Vec<BassChannel>,
    nominal_lag: usize,
    transposition: f64,
}

impl BassSola {
    pub(crate) fn from_env(num_channels: usize, nominal_lag: usize) -> Option<Self> {
        if std::env::var("TIMESTRETCH_PROTO_BASSLOCK").as_deref() != Ok("1") {
            return None;
        }
        Some(Self::new(num_channels, nominal_lag))
    }

    fn new(num_channels: usize, nominal_lag: usize) -> Self {
        Self {
            channels: (0..num_channels)
                .map(|_| BassChannel::new(nominal_lag))
                .collect(),
            nominal_lag,
            transposition: 1.0,
        }
    }

    pub(crate) fn set_transposition(&mut self, transposition: f64) {
        self.transposition = if transposition.is_finite() && transposition > 0.0 {
            transposition
        } else {
            1.0
        };
    }

    pub(crate) fn process_channel(&mut self, ch: usize, io: &mut [f32; BLOCK_FRAMES]) {
        let t = self.transposition;
        let nominal = self.nominal_lag as f64;
        self.channels[ch].process(io, t, nominal);
    }

    pub(crate) fn reset(&mut self) {
        let lag = self.nominal_lag;
        for ch in &mut self.channels {
            *ch = BassChannel::new(lag);
        }
        self.transposition = 1.0;
    }
}

impl BassChannel {
    fn new(nominal_lag: usize) -> Self {
        Self {
            ring: vec![0.0; RING_LEN],
            // Pre-fill the nominal lag with silence so read starts at 0.
            write: nominal_lag as u64,
            read: 0.0,
            old_read: 0.0,
            fade_left: 0,
            env_fast: 0.0,
            env_slow: 0.0,
            retry_cooldown: 0,
        }
    }

    #[inline]
    fn sample(&self, pos: f64) -> f32 {
        // 4-point cubic (Catmull-Rom); the band is ~370x oversampled at
        // 44.1 kHz so interpolation error is negligible.
        let i = pos.floor();
        let frac = (pos - i) as f32;
        let i = i as u64;
        let idx = |k: u64| self.ring[((i + k) & RING_MASK as u64) as usize];
        // pos >= 1.0 is guaranteed by the margin clamps.
        let (p0, p1, p2, p3) = (
            self.ring[((i.wrapping_sub(1)) & RING_MASK as u64) as usize],
            idx(0),
            idx(1),
            idx(2),
        );
        let a = 0.5 * (3.0 * (p1 - p2) + p3 - p0);
        let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
        let c = 0.5 * (p2 - p0);
        p1 + frac * (c + frac * (b + frac * a))
    }

    /// Normalized cross-correlation between the windows ENDING at `a`
    /// and `b` (integer ring positions, backward CORR_WIN frames): the
    /// read cursor can sit within a fade-travel of the write head, so
    /// only history behind it is guaranteed written.
    fn ncc(&self, a: u64, b: u64) -> f64 {
        let (mut dot, mut ea, mut eb) = (0.0f64, 0.0f64, 0.0f64);
        for k in 1..=CORR_WIN as u64 {
            let x = self.ring[((a.wrapping_sub(k)) & RING_MASK as u64) as usize] as f64;
            let y = self.ring[((b.wrapping_sub(k)) & RING_MASK as u64) as usize] as f64;
            dot += x * y;
            ea += x * x;
            eb += y * y;
        }
        dot / (ea * eb).sqrt().max(1e-12)
    }

    /// Dominant period near the read cursor via NCC sweep (coarse step 4,
    /// fine ±4), or None when the band carries no periodicity worth
    /// aligning to (silence — any splice point is as good as another).
    fn estimate_period(&self) -> Option<usize> {
        let anchor = self.read as u64;
        let mut best = (0usize, -1.0f64);
        let mut lag = PERIOD_MIN;
        while lag <= PERIOD_MAX {
            let c = self.ncc(anchor, anchor.wrapping_sub(lag as u64));
            if c > best.1 {
                best = (lag, c);
            }
            lag += 4;
        }
        for lag in best.0.saturating_sub(3)..=(best.0 + 3).min(PERIOD_MAX) {
            let c = self.ncc(anchor, anchor.wrapping_sub(lag as u64));
            if c > best.1 {
                best = (lag, c);
            }
        }
        (best.1 > 0.5).then_some(best.0)
    }

    fn process(&mut self, io: &mut [f32; BLOCK_FRAMES], t: f64, nominal: f64) {
        for s in io.iter_mut() {
            // Ingest.
            self.ring[(self.write & RING_MASK as u64) as usize] = *s;
            self.write += 1;
            let x = *s;
            self.env_fast += 0.02 * (x * x - self.env_fast);
            self.env_slow += 0.0005 * (x * x - self.env_slow);

            // Emit.
            let out = if self.fade_left > 0 {
                let k = self.fade_left as f32 / XFADE as f32;
                // Raised cosine: old fades out, new fades in.
                let w_old = 0.5 - 0.5 * (core::f32::consts::PI * k).cos();
                let a = self.sample(self.old_read);
                let b = self.sample(self.read);
                self.old_read += t;
                self.fade_left -= 1;
                w_old * a + (1.0 - w_old) * b
            } else {
                self.sample(self.read)
            };
            self.read += t;
            *s = out;

            if self.fade_left > 0 {
                continue; // one splice in flight at a time
            }

            // Lag bookkeeping. Positive drift = read is late (lag grew,
            // transposing down); negative = read is catching the write
            // head (transposing up). The lag change per frame is
            // `v = 1 - t`, and repayment jumps must go AGAINST v only:
            // a magnitude-only trigger ping-pongs (a period jump lands
            // the drift on the far side of the corridor, re-triggers,
            // and the counter-jump gets clamped and breaks alignment).
            let lag = self.write as f64 - self.read;
            let drift = lag - nominal;
            let v = 1.0 - t;
            // Hard bounds keep the cursor (and a full fade + corr window)
            // inside valid history regardless of periodicity.
            let fade_travel = XFADE as f64 * t.max(1.0);
            let hard_low = MARGIN + fade_travel + 16.0;
            let hard_high = (RING_LEN - CORR_WIN) as f64 - fade_travel - 16.0;
            const SLACK: f64 = 64.0;
            let forced = lag < hard_low || lag > hard_high;
            if !forced {
                if v.abs() < 1e-9 {
                    continue; // unity: nothing drains, nothing to repay
                }
                if self.retry_cooldown > 0 {
                    self.retry_cooldown -= 1;
                    continue;
                }
                // Cheap precheck before the NCC sweep: enough drift has
                // drained toward the exiting side to be worth a look.
                let toward_exit = if v < 0.0 { -drift } else { drift };
                let quiet = self.env_fast < QUIET_RATIO * self.env_slow;
                let due = toward_exit > PERIOD_MIN as f64 * 0.5
                    || (quiet && toward_exit > PERIOD_MIN as f64 * OPPORTUNISTIC_DRIFT_FRAC);
                if !due {
                    continue;
                }
            }

            // Repay whole periods against the drain direction. The jump
            // count comes from the ROOM on the landing side, not from
            // rounding the drift: the corridor is asymmetric (only ~150
            // frames below nominal before the hard floor, thousands
            // above), and a jump that has to be clamped is a jump that
            // breaks period alignment.
            let period = self.estimate_period();
            let jump = match period {
                Some(p) => {
                    let p = p as f64;
                    let n = if v > 0.0 {
                        // Lag grows; jump forward as many periods as fit
                        // while still landing above the hard floor.
                        ((lag - (hard_low + SLACK)) / p).floor()
                    } else {
                        // Lag shrinks; jump back into history, capped by
                        // the ring end.
                        let n_max = ((hard_high - SLACK - lag) / p).floor();
                        (-drift / p).round().max(1.0).min(n_max)
                    };
                    if n < 1.0 {
                        if forced {
                            // No aligned jump fits: recenter outright and
                            // let the fade absorb it (degenerate input —
                            // period longer than the corridor allows).
                            drift
                        } else {
                            self.retry_cooldown = 64;
                            continue;
                        }
                    } else if v > 0.0 {
                        n * p
                    } else {
                        -n * p
                    }
                }
                // No periodicity worth aligning to (silence): recenter
                // outright (read += drift lowers lag by drift, back to
                // nominal) — but only when it must; splicing mid-note
                // without alignment is the one guaranteed-audible move.
                None => {
                    if forced {
                        drift
                    } else {
                        self.retry_cooldown = 64;
                        continue;
                    }
                }
            };
            let target = (self.read + jump)
                .clamp(self.write as f64 - hard_high, self.write as f64 - hard_low);
            if (target - self.read).abs() < 1.0 {
                continue;
            }
            self.old_read = self.read;
            self.read = target;
            self.fade_left = XFADE;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAG: usize = 560;

    fn run(bass: &mut BassSola, input: &[f32], t: f64) -> Vec<f32> {
        bass.set_transposition(t);
        let mut out = Vec::with_capacity(input.len());
        for chunk in input.chunks(BLOCK_FRAMES) {
            let mut block = [0.0f32; BLOCK_FRAMES];
            block[..chunk.len()].copy_from_slice(chunk);
            bass.process_channel(0, &mut block);
            out.extend_from_slice(&block[..chunk.len()]);
        }
        out
    }

    fn sine(freq: f64, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / 44_100.0).sin() as f32 * 0.5)
            .collect()
    }

    fn mk() -> BassSola {
        BassSola::new(1, LAG)
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
        let mut bass = mk();
        let input = sine(60.0, 44_100);
        let out = run(&mut bass, &input, 1.0);
        // After the lag, output must track the input delayed by LAG.
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
            let mut bass = mk();
            let input = sine(f_in, 6 * 44_100);
            let out = run(&mut bass, &input, t);
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
}

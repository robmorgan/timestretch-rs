//! Time-domain SOLA pitch corrector for small transpositions.
//!
//! At DJ transpositions (|T − 1| below ~5%) a splice-based corrector is
//! transparent on transients and needs no FFT: the output reads from a ring
//! of recent band audio at rate `T` (windowed-sinc interpolation), and when
//! the read cursor drifts from its nominal lag the corrector splices — a
//! correlation-matched jump with an equal-power crossfade, placed at
//! low-energy moments when possible (onset snapping deepens in Stage 4).
//!
//! The nominal lag is the keylock chain's constant latency (see
//! `KEYLOCK_LATENCY_FRAMES`): the SOLA algorithm itself only needs the
//! sinc margin (~0.5 ms), but the elastic drift triggers and splice
//! search need real headroom behind the cursor, and the low band's
//! matching delay is anchored on the same figure.
//!
//! Splice decisions are made once per block on the channel mix and applied
//! to every channel identically, keeping the stereo image intact.

use crate::engine::stage::{OnsetEvent, BLOCK_FRAMES};

/// Half-width of the SOLA read kernel in zero-crossings (64-tap kernel).
///
/// Twice the shared streaming prototype's 16: the corrector's random-access
/// reads sit at arbitrary fractional offsets for minutes at a time, so
/// passband flatness matters more than for the varispeed's forward
/// resampler — 16 half-taps droop ~1.5 dB in the top octave (measured as
/// the keylock chain's HF loss vs the old engine), 32 half-taps are flat
/// past 0.9× Nyquist. Integer offsets remain an exact delta, preserving
/// bit-exact unity passthrough.
const READ_HALF_TAPS: usize = 32;
/// Table oversampling (entries per zero-crossing, linearly interpolated).
const READ_PHASES: usize = 512;
/// Kaiser beta (~−90 dB stopband, same family as the shared prototype).
const READ_KAISER_BETA: f64 = 9.0;

/// Windowed-sinc interpolation table for the SOLA reader.
#[derive(Debug)]
struct ReadInterpTable {
    taps: Vec<f32>,
}

impl ReadInterpTable {
    fn new() -> Self {
        fn bessel_i0(x: f64) -> f64 {
            let mut sum = 1.0f64;
            let mut term = 1.0f64;
            let half_x = x * 0.5;
            for k in 1..=25 {
                term *= (half_x / k as f64) * (half_x / k as f64);
                sum += term;
                if term < sum * 1e-16 {
                    break;
                }
            }
            sum
        }
        let entries = READ_HALF_TAPS * READ_PHASES;
        let mut taps = vec![0.0f32; entries + 2];
        let bessel_beta = bessel_i0(READ_KAISER_BETA);
        for (i, tap) in taps.iter_mut().enumerate().take(entries + 1) {
            let u = i as f64 / READ_PHASES as f64;
            let sinc_val = if u < 1e-12 {
                1.0
            } else {
                let pi_u = std::f64::consts::PI * u;
                pi_u.sin() / pi_u
            };
            let t = u / READ_HALF_TAPS as f64;
            let window = if t <= 1.0 {
                bessel_i0(READ_KAISER_BETA * (1.0 - t * t).max(0.0).sqrt()) / bessel_beta
            } else {
                0.0
            };
            *tap = (sinc_val * window) as f32;
        }
        Self { taps }
    }

    #[inline]
    fn weight(&self, u_abs: f64) -> f32 {
        if u_abs >= READ_HALF_TAPS as f64 {
            return 0.0;
        }
        let x = u_abs * READ_PHASES as f64;
        let i = x as usize;
        let frac = (x - i as f64) as f32;
        let a = self.taps[i];
        let b = self.taps[i + 1];
        a + (b - a) * frac
    }
}

/// Ring capacity per channel (power of two), in frames.
const RING_LEN: usize = 4_096;
const RING_MASK: usize = RING_LEN - 1;

/// Drift from the nominal lag that triggers a splice, in frames.
const DRIFT_TRIGGER: f64 = 192.0;
/// Drift at which a splice is forced even through a transient.
const HARD_TRIGGER: f64 = 320.0;
/// Correlation search half-range around the nominal jump, in frames.
const SEARCH_RANGE: isize = 160;
/// Correlation window length, in frames. Sized to cover 2+ periods at the
/// band's bottom edge (~150 Hz): a shorter window aligns splices only to
/// the dominant mid/high period grid and repeatedly chops seam-region
/// content mid-period — audible as bass thinning against the un-corrected
/// low band during tempo gestures.
const CORR_WINDOW: usize = 320;
/// Splice crossfade length, in frames (~2.2 ms). Longer fades measurably
/// worsen ride pitch stability (each fade interpolates between copies at
/// a sub-period offset, and more estimation windows straddle a fade), so
/// seam-content preservation leans on the correlation window instead.
const XFADE_FRAMES: usize = 96;
/// Sinc half-width plus slack the read cursor keeps behind the write head.
const MIN_READ_MARGIN: f64 = (READ_HALF_TAPS + 4) as f64;
/// Sub-sample splice refinement half-range, in frames: after the integer
/// correlation search, the landing is refined on a fractional grid within
/// this radius (see `plan_splice`).
const FINE_SEARCH_RADIUS: f64 = 0.5;
/// Fractional steps per frame in the sub-sample refinement grid.
const FINE_SEARCH_STEPS: isize = 4;
/// Candidate-region energy above this multiple of the rolling average reads
/// as a transient: postpone the splice until [`HARD_TRIGGER`]. Online
/// fallback — artifact onsets take precedence when present.
const TRANSIENT_POSTPONE_RATIO: f64 = 3.0;
/// Protection window around an artifact onset, in frames: a splice fade
/// must not overlap `[onset - PRE, onset + POST]` in either the outgoing
/// or incoming read span (the attack must come from one uncut read).
const ONSET_PROTECT_PRE: f64 = 96.0;
const ONSET_PROTECT_POST: f64 = 512.0;
/// With drift beyond this, a masked window right after an onset/beat is
/// taken opportunistically even though the trigger has not been reached.
const OPPORTUNISTIC_DRIFT: f64 = 96.0;
/// The masked window after an onset where a splice hides best.
const MASKED_WINDOW_START: f64 = 768.0;
const MASKED_WINDOW_END: f64 = 2_048.0;
/// Below this transposition deviation the corrector is effectively at rest
/// and actively bleeds elastic drift back to zero (see `process_block`):
/// parked drift keeps the high band time-shifted against the low band's
/// fixed delay, comb-filtering the crossover overlap indefinitely.
const REST_DEV: f64 = 0.003;
/// Blocks the deviation must stay below [`REST_DEV`] before rest actions
/// engage (~150 ms): a fast ride sweeps through unity in tens of
/// milliseconds, and firing rest splices/trim there costs measurable
/// pitch stability exactly where the ride should be purest.
const REST_DWELL_BLOCKS: u32 = 206;
/// At-rest drift beyond this recenters via a clean splice immediately.
const REST_SPLICE_DRIFT: f64 = 48.0;
/// Maximum read-rate trim used to bleed small at-rest drift. Kept under
/// the pure-tone pitch JND (~1.4 cents): a tempo ride sweeps through
/// unity — engaging the trim — exactly when pitch should be purest, so
/// this must stay inaudible. The rest splice above removes bulk drift;
/// the trim only polishes the sub-[`REST_SPLICE_DRIFT`] residue.
const REST_TRIM_MAX: f64 = 0.0008;

/// One channel's ring state.
#[derive(Debug)]
struct SolaChannel {
    ring: Vec<f32>,
}

/// Elastic time-domain pitch corrector across all channels.
#[derive(Debug)]
pub(crate) struct SolaCorrector {
    channels: Vec<SolaChannel>,
    table: ReadInterpTable,
    /// Frames written since reset (shared across channels).
    write_abs: u64,
    /// Absolute fractional read cursor (shared: channels are lockstep).
    read_pos: f64,
    /// Crossfade-out cursor while a splice fade is active.
    xfade_from: f64,
    /// Remaining crossfade frames (0 = not fading).
    xfade_remaining: usize,
    transposition: f64,
    /// Local slope of the embedded rate (rate per stage frame): the read
    /// cursor sits `settled_drift` frames off nominal, consuming audio
    /// embedded at `rate - slope * drift`, and the synthesis rate tracks
    /// that instead of the nominal transposition (ride pitch accuracy).
    rate_slope: f64,
    nominal_lag: f64,
    /// Rolling average of block RMS on the channel mix (transient gate).
    energy_avg: f64,
    /// Splices executed since reset (observability for tests/QA).
    splice_count: u64,
    /// Consecutive blocks with the transposition inside [`REST_DEV`].
    rest_blocks: u32,
}

impl SolaCorrector {
    pub(crate) fn new(num_channels: usize, nominal_lag_frames: usize) -> Self {
        assert!(
            nominal_lag_frames as f64 + HARD_TRIGGER + (SEARCH_RANGE as f64) + MIN_READ_MARGIN
                < (RING_LEN - CORR_WINDOW - BLOCK_FRAMES) as f64,
            "SOLA ring too small for nominal lag {nominal_lag_frames}"
        );
        Self {
            channels: (0..num_channels)
                .map(|_| SolaChannel {
                    ring: vec![0.0; RING_LEN],
                })
                .collect(),
            table: ReadInterpTable::new(),
            write_abs: 0,
            // Starting the cursor a full lag behind zero realizes the
            // constant delay exactly: the first `nominal_lag` reads land on
            // never-written (zero) ring slots and emit the priming silence.
            read_pos: -(nominal_lag_frames as f64),
            xfade_from: 0.0,
            xfade_remaining: 0,
            transposition: 1.0,
            rate_slope: 0.0,
            nominal_lag: nominal_lag_frames as f64,
            energy_avg: 0.0,
            splice_count: 0,
            rest_blocks: 0,
        }
    }

    pub(crate) fn set_transposition(&mut self, transposition: f64) {
        self.transposition = if transposition.is_finite() {
            transposition.clamp(0.75, 1.35)
        } else {
            1.0
        };
    }

    pub(crate) fn set_rate_slope(&mut self, slope: f64) {
        self.rate_slope = if slope.is_finite() { slope } else { 0.0 };
    }

    pub(crate) fn latency_frames(&self) -> usize {
        self.nominal_lag as usize
    }

    #[cfg(test)]
    pub(crate) fn splice_count(&self) -> u64 {
        self.splice_count
    }

    /// Current drift of the read cursor from its nominal lag, in frames
    /// (0 = perfectly centered). The keylock handoff prefers to switch
    /// correctors when this is small.
    pub(crate) fn lag_error_frames(&self) -> f64 {
        (self.write_abs as f64 - self.read_pos) - self.nominal_lag
    }

    pub(crate) fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.ring.fill(0.0);
        }
        self.write_abs = 0;
        self.read_pos = -self.nominal_lag;
        self.xfade_from = 0.0;
        self.xfade_remaining = 0;
        self.energy_avg = 0.0;
        self.splice_count = 0;
        self.rest_blocks = 0;
        self.rate_slope = 0.0;
    }

    /// Whether a fade read-span starting at `start` overlaps any artifact
    /// onset's protection window (positions share the write/stage axis).
    fn span_hits_onset(&self, onsets: &[OnsetEvent], start: f64) -> bool {
        // Padded by the sub-sample refinement radius: the fine search may
        // move a vetted candidate by up to half a frame either way.
        let span = XFADE_FRAMES as f64 * self.transposition.max(1.0);
        onsets.iter().any(|event| {
            start - FINE_SEARCH_RADIUS < event.stage_frame + ONSET_PROTECT_POST
                && start + span + FINE_SEARCH_RADIUS > event.stage_frame - ONSET_PROTECT_PRE
        })
    }

    /// Whether the current read position sits in the masked window just
    /// after an onset/beat — the best hiding place for a splice.
    fn in_masked_window(&self, onsets: &[OnsetEvent]) -> bool {
        onsets.iter().any(|event| {
            let since = self.read_pos - event.stage_frame;
            (MASKED_WINDOW_START..MASKED_WINDOW_END).contains(&since)
        })
    }

    /// Processes one fixed block for every channel in lockstep: writes the
    /// inputs into the rings, splices if the elastic lag calls for it, then
    /// synthesizes the outputs in place. `onsets` (stage-timeline artifact
    /// events; empty when no artifact) steer splice timing: fades never
    /// overlap an onset's protection window, and pending splices are taken
    /// opportunistically in the masked window after a hit.
    pub(crate) fn process_block(&mut self, io: &mut [[f32; BLOCK_FRAMES]], onsets: &[OnsetEvent]) {
        debug_assert_eq!(io.len(), self.channels.len());

        // 1) Ingest.
        let mut block_energy = 0.0f64;
        for (ch, input) in self.channels.iter_mut().zip(io.iter()) {
            for (i, &sample) in input.iter().enumerate() {
                ch.ring[(self.write_abs as usize + i) & RING_MASK] = sample;
                block_energy += (sample as f64) * (sample as f64);
            }
        }
        self.write_abs += BLOCK_FRAMES as u64;
        let block_rms = (block_energy / (BLOCK_FRAMES * io.len()) as f64).sqrt();
        self.energy_avg = 0.98 * self.energy_avg + 0.02 * block_rms;

        // 2) Splice management (block-granular: drift accrues < 2 frames
        //    per block at the clamp bounds). `lag_error_frames` here — after
        //    ingest, before synthesis — reads one block high; the SETTLED
        //    drift (what actually parks between blocks) subtracts it.
        let deviation = (self.transposition - 1.0).abs();
        let settled_drift = self.lag_error_frames() - BLOCK_FRAMES as f64;
        let at_rest = if deviation < REST_DEV {
            self.rest_blocks = self.rest_blocks.saturating_add(1);
            self.rest_blocks >= REST_DWELL_BLOCKS
        } else {
            self.rest_blocks = 0;
            false
        };
        if self.xfade_remaining == 0 {
            let drift = self.lag_error_frames();
            if drift.abs() > DRIFT_TRIGGER {
                // Onset protection applies inside the candidate search; the
                // HARD trigger forces through it eventually.
                self.try_splice(drift, onsets);
            } else if drift.abs() > OPPORTUNISTIC_DRIFT && self.in_masked_window(onsets) {
                // Beat-synchronous placement: a pending correction hides
                // best right after a hit.
                self.try_splice(drift, onsets);
            } else if at_rest && settled_drift.abs() > REST_SPLICE_DRIFT {
                // At sustained rest a parked drift comb-filters the
                // crossover overlap against the low band's fixed delay;
                // recenter cleanly.
                self.try_splice(drift, onsets);
            }
        }

        // 3) Synthesis. At sustained rest, a small read-rate trim bleeds
        // residual drift to zero (kept under the pitch JND).
        // d(drift)/dframe = −trim: positive parked drift needs a faster
        // read (positive trim) to converge.
        let trim = if at_rest {
            (settled_drift * 0.001).clamp(-REST_TRIM_MAX, REST_TRIM_MAX)
        } else {
            0.0
        };
        // Slope-tracked transposition: the cursor reads `settled_drift`
        // frames off the nominal lag, where the embedded rate differs by
        // `slope * drift` — invert THAT rate, or a ride detunes by up to a
        // couple of cents whenever drift is parked. Zero slope (constant
        // rate) leaves the nominal transposition bit-exact.
        let t = if self.rate_slope != 0.0 {
            let rate_at_cursor =
                1.0 / self.transposition - (self.rate_slope * settled_drift).clamp(-0.02, 0.02);
            if rate_at_cursor > 0.5 {
                (1.0 / rate_at_cursor).clamp(0.75, 1.35) + trim
            } else {
                self.transposition + trim
            }
        } else {
            self.transposition + trim
        };
        for i in 0..BLOCK_FRAMES {
            if self.xfade_remaining > 0 {
                // Raised-cosine amplitude-complementary crossfade between
                // the outgoing and incoming read positions (both advance at
                // T). Amplitude- rather than power-complementary because the
                // splice is correlation-matched: the two signals are nearly
                // identical, and an equal-power fade would bulge to ~1.41x
                // mid-fade on correlated content (same choice as `Wsola`).
                let progress =
                    1.0 - (self.xfade_remaining as f64 - 1.0) / (XFADE_FRAMES as f64 - 1.0);
                let g_in = (0.5 - 0.5 * (std::f64::consts::PI * progress).cos()) as f32;
                let g_out = 1.0 - g_in;
                for (ch, out) in self.channels.iter().zip(io.iter_mut()) {
                    let a = sinc_read(&ch.ring, self.xfade_from, &self.table);
                    let b = sinc_read(&ch.ring, self.read_pos, &self.table);
                    out[i] = g_out * a + g_in * b;
                }
                self.xfade_from += t;
                self.read_pos += t;
                self.xfade_remaining -= 1;
            } else {
                for (ch, out) in self.channels.iter().zip(io.iter_mut()) {
                    out[i] = sinc_read(&ch.ring, self.read_pos, &self.table);
                }
                self.read_pos += t;
            }
        }
        let newest_read = if self.xfade_remaining > 0 {
            self.read_pos.max(self.xfade_from)
        } else {
            self.read_pos
        };
        debug_assert!(
            self.write_abs as f64 - newest_read >= MIN_READ_MARGIN - 1.0,
            "SOLA read overtook the write head"
        );
    }

    /// Drift-triggered splice: onset protection and the transient postpone
    /// apply until the drift is critical.
    fn try_splice(&mut self, drift: f64, onsets: &[OnsetEvent]) {
        let force = drift.abs() >= HARD_TRIGGER;
        let _ = self.plan_splice(drift, force, onsets);
    }

    /// Plans and starts a correlation-matched splice toward the nominal
    /// lag; unless `force`, a landing region that reads as a transient
    /// postpones it, and candidates whose fade would overlap an artifact
    /// onset's protection window are excluded from the search.
    fn plan_splice(&mut self, drift: f64, force: bool, onsets: &[OnsetEvent]) -> bool {
        // Outgoing span: the fade also reads from the current cursor.
        if !force && self.span_hits_onset(onsets, self.read_pos) {
            return false;
        }
        // Jump the read cursor so the lag returns to nominal: with
        // lag' = write − (read + jump), jump = lag − nominal = drift.
        let nominal_jump = drift;
        // …searching around that jump for the offset whose audio best
        // continues what the cursor is currently playing.
        let (mut best_jump, mut best_score) = (nominal_jump, f64::MIN);
        let base = self.read_pos;
        let lo = nominal_jump as isize - SEARCH_RANGE;
        let hi = nominal_jump as isize + SEARCH_RANGE;
        for jump in lo..=hi {
            let candidate = base + jump as f64;
            if !self.readable_span(candidate, CORR_WINDOW + XFADE_FRAMES) {
                continue;
            }
            if !force && self.span_hits_onset(onsets, candidate) {
                continue;
            }
            // Mild distance penalty: periodic content scores every
            // period-grid candidate identically, and un-penalized ties
            // resolve to the search edge — parking the elastic drift ~a
            // full search range off nominal instead of converging to it.
            let distance = (jump as f64 - nominal_jump).abs() / SEARCH_RANGE as f64;
            let score = self.mix_correlation(base, candidate, CORR_WINDOW) - 0.02 * distance;
            if score > best_score {
                best_score = score;
                best_jump = jump as f64;
            }
        }
        if best_score == f64::MIN {
            return false; // every candidate excluded; retried next block
        }
        // Sub-sample refinement: the integer grid misses the correlation
        // peak by up to half a sample — at the band's top octave that is
        // most of a period, and the residual phase error at each splice
        // scrambles HF coherence (measured as steady top-octave loss at
        // sustained rates). Only sub-sample structure distinguishes the
        // fractional candidates, so a short window (the fade span, where
        // the two copies actually interfere) suffices; the read cursor is
        // fractional anyway, so the refined jump costs nothing downstream.
        if self.readable_span(base + best_jump - 1.0, CORR_WINDOW + XFADE_FRAMES)
            && self.readable_span(base + best_jump + 1.0, CORR_WINDOW + XFADE_FRAMES)
        {
            const GRID: usize = 2 * FINE_SEARCH_STEPS as usize + 1;
            let step = FINE_SEARCH_RADIUS / FINE_SEARCH_STEPS as f64;
            let mut cs = [0.0f64; GRID];
            let (mut best_k, mut best_c) = (0usize, f64::MIN);
            for (k, c) in cs.iter_mut().enumerate() {
                let frac = (k as f64 - FINE_SEARCH_STEPS as f64) * step;
                *c = self.mix_correlation_frac(base, base + best_jump + frac, XFADE_FRAMES);
                if *c > best_c {
                    best_c = *c;
                    best_k = k;
                }
            }
            let mut best_frac = (best_k as f64 - FINE_SEARCH_STEPS as f64) * step;
            // The grid steps sample the correlation ripple ~20x denser than
            // even top-octave content, so a parabolic vertex through the
            // peak's neighbours is well-conditioned: residual sub-sample
            // error drops another order of magnitude (each splice's fade
            // spreads that error as a pitch wobble — ride cents accuracy).
            if best_k > 0 && best_k + 1 < GRID {
                let (cm, c0, cp) = (cs[best_k - 1], cs[best_k], cs[best_k + 1]);
                let denom = cm - 2.0 * c0 + cp;
                if denom < -1e-12 {
                    best_frac += step * (0.5 * (cm - cp) / denom).clamp(-0.5, 0.5);
                }
            }
            best_jump += best_frac;
        }
        let target = base + best_jump;
        if !self.readable_span(target, CORR_WINDOW + XFADE_FRAMES) {
            return false; // nothing readable yet; retried next block
        }

        // Transient postpone: if the landing region is a local energy burst
        // (an onset we would smear), wait — unless forced.
        if !force
            && self.energy_avg > 1e-6
            && self.region_rms(target, CORR_WINDOW) > TRANSIENT_POSTPONE_RATIO * self.energy_avg
        {
            return false;
        }

        self.xfade_from = self.read_pos;
        self.read_pos = target;
        self.xfade_remaining = XFADE_FRAMES;
        self.splice_count += 1;
        true
    }

    /// Whether `span` frames starting at `pos` (plus sinc margins) are
    /// inside the ring's valid window.
    fn readable_span(&self, pos: f64, span: usize) -> bool {
        let end = pos + span as f64 * self.transposition.max(1.0);
        let newest_ok = end <= self.write_abs as f64 - MIN_READ_MARGIN;
        let oldest_ok = pos
            >= (self.write_abs as f64 - RING_LEN as f64) + MIN_READ_MARGIN + BLOCK_FRAMES as f64;
        let started = pos >= MIN_READ_MARGIN;
        newest_ok && oldest_ok && started
    }

    /// Normalized cross-correlation between two ring regions on the channel
    /// mix (mono decision keeps channels phase-coherent).
    fn mix_correlation(&self, a_pos: f64, b_pos: f64, len: usize) -> f64 {
        let a0 = a_pos.floor() as usize;
        let b0 = b_pos.floor() as usize;
        let (mut dot, mut a_sq, mut b_sq) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..len {
            let (mut a, mut b) = (0.0f64, 0.0f64);
            for ch in &self.channels {
                a += ch.ring[(a0 + i) & RING_MASK] as f64;
                b += ch.ring[(b0 + i) & RING_MASK] as f64;
            }
            dot += a * b;
            a_sq += a * a;
            b_sq += b * b;
        }
        let norm = (a_sq * b_sq).sqrt();
        if norm < 1e-12 {
            0.0
        } else {
            dot / norm
        }
    }

    /// Like [`Self::mix_correlation`], but `b_pos` is honoured at fractional
    /// precision (sinc-interpolated reads) so sub-sample offsets rank
    /// correctly. `a_pos` stays on the integer grid like the coarse search.
    fn mix_correlation_frac(&self, a_pos: f64, b_pos: f64, len: usize) -> f64 {
        let a0 = a_pos.floor() as usize;
        let (mut dot, mut a_sq, mut b_sq) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..len {
            let (mut a, mut b) = (0.0f64, 0.0f64);
            for ch in &self.channels {
                a += ch.ring[(a0 + i) & RING_MASK] as f64;
                b += sinc_read(&ch.ring, b_pos + i as f64, &self.table) as f64;
            }
            dot += a * b;
            a_sq += a * a;
            b_sq += b * b;
        }
        let norm = (a_sq * b_sq).sqrt();
        if norm < 1e-12 {
            0.0
        } else {
            dot / norm
        }
    }

    /// RMS of the channel mix over `len` frames starting at `pos`.
    fn region_rms(&self, pos: f64, len: usize) -> f64 {
        let p0 = pos.floor() as usize;
        let mut acc = 0.0f64;
        for i in 0..len {
            let mut mix = 0.0f64;
            for ch in &self.channels {
                mix += ch.ring[(p0 + i) & RING_MASK] as f64;
            }
            let mix = mix / self.channels.len() as f64;
            acc += mix * mix;
        }
        (acc / len as f64).sqrt()
    }
}

/// Windowed-sinc random-access read from a ring at a fractional position.
/// Exact passthrough at integer positions (the kernel is a delta there).
#[inline]
fn sinc_read(ring: &[f32], pos: f64, table: &ReadInterpTable) -> f32 {
    let center = pos.floor();
    let frac = pos - center;
    let center = center as isize;
    let half = READ_HALF_TAPS as isize;
    let mut acc = 0.0f64;
    let mut wsum = 0.0f64;
    for j in (1 - half)..=half {
        let w = table.weight((j as f64 - frac).abs()) as f64;
        if w != 0.0 {
            let idx = (center + j) as usize & RING_MASK;
            acc += ring[idx] as f64 * w;
            wsum += w;
        }
    }
    if wsum.abs() > 1e-12 {
        (acc / wsum) as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44_100.0;
    const LAG: usize = 560;

    fn sine(freq: f64, len: usize, amp: f32) -> Vec<f32> {
        (0..len)
            .map(|i| amp * (2.0 * std::f64::consts::PI * freq * i as f64 / SR).sin() as f32)
            .collect()
    }

    fn run(corrector: &mut SolaCorrector, input: &[f32], t: f64) -> Vec<f32> {
        corrector.set_transposition(t);
        let mut out = Vec::with_capacity(input.len());
        let mut block = [[0.0f32; BLOCK_FRAMES]; 1];
        for chunk in input.chunks_exact(BLOCK_FRAMES) {
            block[0].copy_from_slice(chunk);
            corrector.process_block(&mut block, &[]);
            out.extend_from_slice(&block[0]);
        }
        out
    }

    fn measure_freq(window: &[f32]) -> f64 {
        let (mut first, mut last, mut count) = (None, None, 0usize);
        for i in 1..window.len() {
            let (a, b) = (window[i - 1] as f64, window[i] as f64);
            if a <= 0.0 && b > 0.0 {
                let t = (i - 1) as f64 + a / (a - b);
                if first.is_none() {
                    first = Some(t);
                }
                last = Some(t);
                count += 1;
            }
        }
        match (first, last) {
            (Some(f), Some(l)) if count >= 2 => (count - 1) as f64 * SR / (l - f),
            _ => 0.0,
        }
    }

    #[test]
    fn unity_is_pure_delay_with_no_splices() {
        let mut corrector = SolaCorrector::new(1, LAG);
        let input = sine(700.0, 44_100, 0.5);
        let out = run(&mut corrector, &input, 1.0);
        assert_eq!(corrector.splice_count(), 0, "unity must never splice");
        for i in 8_192..40_000 {
            assert!(
                (out[i] - input[i - LAG]).abs() < 1e-4,
                "unity SOLA deviates at {i}"
            );
        }
    }

    #[test]
    fn shifts_pitch_by_transposition() {
        let mut corrector = SolaCorrector::new(1, LAG);
        let input = sine(600.0, 44_100 * 4, 0.5);
        let out = run(&mut corrector, &input, 1.04);
        let f = measure_freq(&out[44_100..88_200]);
        assert!(
            (f - 624.0).abs() < 2.0,
            "expected ~624 Hz at T=1.04, measured {f:.1}"
        );
        assert!(corrector.splice_count() > 0, "non-unity must splice");
    }

    #[test]
    fn splices_are_click_free_on_tone() {
        let freq = 330.0;
        let amp = 0.5;
        let mut corrector = SolaCorrector::new(1, LAG);
        let input = sine(freq, 44_100 * 6, amp);
        let out = run(&mut corrector, &input, 1.05);
        // Pitch shifts to freq * 1.05; correlation-matched splices on a
        // periodic tone must land within a small fraction of the slew.
        let bound = amp * 2.0 * std::f32::consts::PI * (freq * 1.05) as f32 / SR as f32 * 1.35;
        let mut worst = (0usize, 0.0f32);
        for (i, w) in out[4_096..].windows(2).enumerate() {
            let d = (w[1] - w[0]).abs();
            if d > worst.1 {
                worst = (i + 4_096, d);
            }
        }
        println!(
            "sola tone splice: {} splices, max diff {:.5} (bound {bound:.5})",
            corrector.splice_count(),
            worst.1
        );
        assert!(
            worst.1 <= bound,
            "splice click at {}: {:.5} > {bound:.5}",
            worst.0,
            worst.1
        );
    }

    #[test]
    fn lag_stays_bounded_over_long_runs() {
        let mut corrector = SolaCorrector::new(1, LAG);
        let input = sine(500.0, 44_100 * 20, 0.4);
        let _ = run(&mut corrector, &input, 1.05);
        let error = corrector.lag_error_frames().abs();
        assert!(
            error <= HARD_TRIGGER + BLOCK_FRAMES as f64,
            "lag error {error:.0} frames escaped the elastic band"
        );
    }

    #[test]
    fn splice_fades_avoid_artifact_onsets() {
        // Feed a tone with onsets declared every 4410 frames; at T=1.05 the
        // corrector must splice regularly, and no fade span may overlap an
        // onset's protection window.
        let mut corrector = SolaCorrector::new(1, LAG);
        corrector.set_transposition(1.05);
        let input = sine(500.0, 44_100 * 8, 0.4);
        let mut block = [[0.0f32; BLOCK_FRAMES]; 1];
        let mut fade_spans: Vec<(f64, f64)> = Vec::new();
        let mut events: Vec<OnsetEvent> = Vec::new();
        let mut was_fading = false;
        for (bi, chunk) in input.chunks_exact(BLOCK_FRAMES).enumerate() {
            let stage_now = (bi * BLOCK_FRAMES) as f64;
            // Publish onsets near the block, like the graph cursor would.
            events.clear();
            let mut onset = ((stage_now - 2_048.0).max(0.0) / 4_410.0).floor() * 4_410.0;
            while onset <= stage_now + 2_048.0 {
                if onset > 0.0 {
                    events.push(OnsetEvent {
                        stage_frame: onset,
                        strength: 0.9,
                        beat: false,
                    });
                }
                onset += 4_410.0;
            }
            block[0].copy_from_slice(chunk);
            corrector.process_block(&mut block, &events);
            let fading = corrector.xfade_remaining > 0;
            if fading && !was_fading {
                // Record the fade's read spans (outgoing + incoming),
                // rewinding the cursors to the fade's first frame — they
                // have already advanced within this block.
                let elapsed = (XFADE_FRAMES - corrector.xfade_remaining) as f64 * 1.05;
                let span = XFADE_FRAMES as f64 * 1.05;
                let out_start = corrector.xfade_from - elapsed;
                let in_start = corrector.read_pos - elapsed;
                fade_spans.push((out_start, out_start + span));
                fade_spans.push((in_start, in_start + span));
            }
            was_fading = fading;
        }
        assert!(
            corrector.splice_count() > 10,
            "fixture must splice ({} splices)",
            corrector.splice_count()
        );
        for &(lo, hi) in &fade_spans {
            let mut onset = 4_410.0;
            while onset < 44_100.0 * 8.0 {
                assert!(
                    hi <= onset - ONSET_PROTECT_PRE || lo >= onset + ONSET_PROTECT_POST,
                    "fade span [{lo:.0}, {hi:.0}] overlaps onset at {onset:.0}"
                );
                onset += 4_410.0;
            }
        }
    }

    #[test]
    fn stereo_channels_splice_in_lockstep() {
        let mut corrector = SolaCorrector::new(2, LAG);
        corrector.set_transposition(1.05);
        let mono = sine(440.0, 44_100 * 4, 0.5);
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut block = [[0.0f32; BLOCK_FRAMES]; 2];
        for chunk in mono.chunks_exact(BLOCK_FRAMES) {
            for (i, &s) in chunk.iter().enumerate() {
                block[0][i] = s;
                block[1][i] = -0.8 * s;
            }
            corrector.process_block(&mut block, &[]);
            left.extend_from_slice(&block[0]);
            right.extend_from_slice(&block[1]);
        }
        // The exact -0.8 relationship must survive every splice.
        for i in 4_096..left.len() {
            assert!(
                (right[i] + 0.8 * left[i]).abs() < 1e-4,
                "stereo lockstep broken at {i}: L={} R={}",
                left[i],
                right[i]
            );
        }
    }
}

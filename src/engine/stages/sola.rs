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

use crate::core::resample::{bessel_i0, dot_f32_f64, fill_row_lerp, polyphase_rows};
use crate::engine::stage::{BLOCK_FRAMES, OnsetEvent};

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

/// Windowed-sinc interpolation table for the SOLA reader, stored as
/// contiguous polyphase rows (one full kernel per fractional phase; see
/// [`polyphase_rows`]) so a read's weight row fills as a vectorizable lerp
/// of two rows.
#[derive(Debug)]
struct ReadInterpTable {
    rows: Vec<f32>,
}

impl ReadInterpTable {
    fn new() -> Self {
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
        Self {
            rows: polyphase_rows(&taps, READ_HALF_TAPS, READ_PHASES),
        }
    }

    /// Fills the full kernel row for fractional phase `frac` (weights for
    /// tap offsets `1-READ_HALF_TAPS..=READ_HALF_TAPS`) and returns the
    /// weight sum. One row serves every channel and — for runs of reads at
    /// integer strides — every frame sharing the phase, so the row fill
    /// cost is paid once instead of per read.
    #[inline]
    fn fill_row(&self, frac: f64, row: &mut [f32; READ_TAPS]) -> f64 {
        fill_row_lerp(&self.rows, READ_TAPS, READ_PHASES, frac, row)
    }
}

/// Full kernel width of the SOLA reader, in taps.
const READ_TAPS: usize = 2 * READ_HALF_TAPS;

/// One windowed-sinc read kernel: the weight row and normalization for a
/// specific fractional phase, applicable to any channel's ring at the
/// matching integer start index.
struct SincReadKernel {
    row: [f32; READ_TAPS],
    wsum: f64,
    /// First tap's ring index (unmasked; masked per segment in `read`).
    start: isize,
}

impl SincReadKernel {
    #[inline]
    fn at(table: &ReadInterpTable, pos: f64) -> Self {
        let center = pos.floor();
        let frac = pos - center;
        let mut row = [0.0f32; READ_TAPS];
        let wsum = table.fill_row(frac, &mut row);
        Self {
            row,
            wsum,
            start: center as isize + 1 - READ_HALF_TAPS as isize,
        }
    }

    /// Normalized kernel dot against one ring, splitting at the ring wrap so
    /// both segments are contiguous slice dot products.
    #[inline]
    fn read(&self, ring: &[f32]) -> f32 {
        if self.wsum.abs() <= 1e-12 {
            return 0.0;
        }
        (ring_dot(ring, self.start, &self.row) / self.wsum) as f32
    }
}

/// Dot product of `weights` against the ring starting at (masked) `start`,
/// split into at most two contiguous runs at the wrap point.
#[inline]
fn ring_dot(ring: &[f32], start: isize, weights: &[f32]) -> f64 {
    let s = (start as usize) & RING_MASK;
    let n = weights.len();
    if s + n <= RING_LEN {
        dot_f32_f64(&ring[s..s + n], weights)
    } else {
        let first = RING_LEN - s;
        dot_f32_f64(&ring[s..], &weights[..first])
            + dot_f32_f64(&ring[..n - first], &weights[first..])
    }
}

/// Four-lane f64 dot product (see `dot_f32_f64`; correlation windows are
/// premixed into f64 scratch).
#[inline]
fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f64; 4];
    let mut a_chunks = a.chunks_exact(4);
    let mut b_chunks = b.chunks_exact(4);
    for (x, y) in (&mut a_chunks).zip(&mut b_chunks) {
        acc[0] += x[0] * y[0];
        acc[1] += x[1] * y[1];
        acc[2] += x[2] * y[2];
        acc[3] += x[3] * y[3];
    }
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        acc[0] += x * y;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// Four-lane dot product of premixed f64 samples against f32 kernel weights.
#[inline]
fn dot_f64_f32(samples: &[f64], weights: &[f32]) -> f64 {
    debug_assert_eq!(samples.len(), weights.len());
    let mut acc = [0.0f64; 4];
    let mut s_chunks = samples.chunks_exact(4);
    let mut w_chunks = weights.chunks_exact(4);
    for (s, w) in (&mut s_chunks).zip(&mut w_chunks) {
        acc[0] += s[0] * w[0] as f64;
        acc[1] += s[1] * w[1] as f64;
        acc[2] += s[2] * w[2] as f64;
        acc[3] += s[3] * w[3] as f64;
    }
    for (s, w) in s_chunks.remainder().iter().zip(w_chunks.remainder()) {
        acc[0] += s * *w as f64;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// Sums all channels' rings into `out` starting at (masked) ring index
/// `start`: the splice search scores the channel mix, and premixing once
/// turns every candidate's correlation into flat dot products.
fn mix_channels(channels: &[SolaChannel], start: isize, out: &mut [f64]) {
    out.fill(0.0);
    for ch in channels {
        for (k, m) in out.iter_mut().enumerate() {
            *m += ch.ring[(start as usize).wrapping_add(k) & RING_MASK] as f64;
        }
    }
}

/// Ring capacity per channel (power of two), in frames.
const RING_LEN: usize = 4_096;
const RING_MASK: usize = RING_LEN - 1;

/// Drift from the nominal lag that triggers a splice, in frames.
const DRIFT_TRIGGER: f64 = 192.0;
/// Under MILD transposition deviations (below [`MILD_DEV`]) drift beyond
/// this fires a BOUNDED recenter splice (the rest-splice mechanism, no
/// dwell). Rationale (ROADMAP Stage 15): a sustained 0.5–1% ride lets
/// drift sawtooth to the full [`DRIFT_TRIGGER`] — a ~2–4 ms high-band
/// shift against the low band's rigid delay, comb-filtering the
/// crossover seam at −7 dB for as long as the ride lasts. Recentring at
/// 96 frames caps the shift at ~2 ms for ~3–5 splices/s — well under the
/// ~18/s the shipped ±8% path already carries. The landing bound matters:
/// an UNbounded splice on periodic content parks on the dominant-period
/// grid (or zero-jump limit-cycles, autoresearch #62) and makes the seam
/// WORSE — measured −5.5 dB persistent vs −2 dB with the bound. At
/// strong deviations nothing changes: combing there is brief, and
/// tighter triggers would multiply splice granulation.
const MILD_DRIFT_TRIGGER: f64 = 96.0;
/// Deviation ceiling for the mild-motion bounded recenter.
const MILD_DEV: f64 = 0.012;
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
/// Quiet-gap opportunism: with drift beyond [`OPPORTUNISTIC_DRIFT`], a
/// splice whose outgoing and incoming regions both sit below this multiple
/// of the rolling average energy is taken early — small corrections in
/// quiet moments instead of letting drift accrue to a forced splice that
/// may land on an attack.
const QUIET_SPLICE_RATIO: f64 = 0.6;
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
    /// Unforced splice attempts that failed since the last splice. A
    /// forced splice (which tears through onset protection) is honored
    /// only when this is nonzero, so a regime change that shrinks the
    /// force threshold under already-parked drift (a ride starting while
    /// the cadence stretch had the band widened) cannot skip protection
    /// on its very first block.
    failed_attempts: u32,
    /// Consecutive blocks with the transposition inside [`REST_DEV`].
    rest_blocks: u32,
    /// Premixed base window for the splice search ([`CORR_WINDOW`] frames;
    /// preallocated: `plan_splice` runs on the audio thread).
    corr_a: Vec<f64>,
    /// Premixed candidate span for the coarse search (every candidate's
    /// window is a slice of this).
    corr_b: Vec<f64>,
    /// Premixed span for the sub-sample refinement's sinc reads.
    frac_mix: Vec<f64>,
}

/// Splice-cadence stretch (ROADMAP Stage 18): at full stretch the elastic
/// drift triggers double, so splices fire half as often and land with
/// larger, better period-aligned jumps — the fix for the Stage 16 blind
/// verdict, where the ~20-splices/s cadence granulated sustained tonal
/// material ("roboty"; harmonic-15 purity 22 dB at −8% tempo, 63 dB at
/// half cadence; blind owner A/B: half cadence won or tied all 6
/// conditions, every "robot" rating on the shipped cadence).
const CADENCE_SCALE_MAX: f64 = 2.0;
/// Slowdowns read faster than the ring writes (T > 1), so the elastic
/// drift bound eats the nominal-lag headroom toward the write head — the
/// binding constraint the Stage 18 prototype hit: full stretch is green
/// through the primary DJ window but stalls the cursor near the ±20–25%
/// edge (measured: sine pitch −16 cents at T = 1.25). The stretch
/// therefore tapers off past the primary window and is fully released
/// well before the measured failure. Speed-ups (T < 1) drift toward the
/// ring's old side, where headroom is ~6x the hard trigger — no taper.
const CADENCE_TAPER_START: f64 = 1.09;
const CADENCE_TAPER_END: f64 = 1.15;
/// Write-head-side cap on the FORCED splice threshold (negative/slowdown
/// drift). The elastic band is asymmetric by physics: speed-up drift
/// parks toward the ring's old side (~6x the hard trigger of headroom),
/// but slowdown drift eats the nominal-lag gap to the write head — the
/// stretched hard trigger (640) sits PAST the head (lag 560), so under
/// blocked-splice pressure (dense onset protection, loud landings) the
/// backstop could never fire before the read margin was violated. The
/// cap keeps the force reachable with real margin:
/// `560 − MIN_READ_MARGIN − 2·BLOCK_FRAMES` of runway, leaving 64 frames
/// of unforced attempts between the stretched normal trigger (384) and
/// the force. Asserted against the nominal lag at construction.
const SLOWDOWN_FORCE_CAP: f64 = 448.0;

/// Cadence stretch for the current transposition: [`CADENCE_SCALE_MAX`]
/// through the primary DJ window, linearly released to 1 across the
/// taper band on slowdowns.
fn cadence_scale(transposition: f64) -> f64 {
    if transposition <= CADENCE_TAPER_START {
        return CADENCE_SCALE_MAX;
    }
    if transposition >= CADENCE_TAPER_END {
        return 1.0;
    }
    let t = (transposition - CADENCE_TAPER_START) / (CADENCE_TAPER_END - CADENCE_TAPER_START);
    CADENCE_SCALE_MAX + (1.0 - CADENCE_SCALE_MAX) * t
}

impl SolaCorrector {
    pub(crate) fn new(num_channels: usize, nominal_lag_frames: usize) -> Self {
        // Ring / old-audio side (binds speed-up drift, +HARD·scale).
        assert!(
            nominal_lag_frames as f64
                + HARD_TRIGGER * CADENCE_SCALE_MAX
                + (SEARCH_RANGE as f64)
                + MIN_READ_MARGIN
                < (RING_LEN - CORR_WINDOW - BLOCK_FRAMES) as f64,
            "SOLA ring too small for nominal lag {nominal_lag_frames}"
        );
        // Write-head side (binds slowdown drift): the forced splice must
        // fire while the read cursor still has margin.
        assert!(
            SLOWDOWN_FORCE_CAP + MIN_READ_MARGIN + 2.0 * BLOCK_FRAMES as f64
                <= nominal_lag_frames as f64,
            "SOLA nominal lag {nominal_lag_frames} cannot cover the slowdown force cap"
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
            failed_attempts: 0,
            rest_blocks: 0,
            corr_a: vec![0.0; CORR_WINDOW],
            corr_b: vec![0.0; 2 * SEARCH_RANGE as usize + CORR_WINDOW],
            frac_mix: vec![0.0; XFADE_FRAMES + 2 * READ_TAPS],
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

    /// Stage 18 cadence stretch, gated on STEADY transposition: sustained
    /// tonal content — where splice granulation is audible (Stage 16
    /// blind verdict) — is a parked-fader regime, while rides need small
    /// drift for pitch accuracy: the slope-tracked synthesis correction
    /// is proportional to the parked drift and clamps, so a doubled
    /// drift band doubled ride cents p95 straight past its A/B-matrix
    /// gate (measured 5.65 vs bound 1.5 before this gate was added).
    /// A per-block scalar from existing state — no allocation.
    fn cadence(&self) -> f64 {
        if self.rate_slope == 0.0 {
            cadence_scale(self.transposition)
        } else {
            1.0
        }
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
        self.failed_attempts = 0;
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
            let cadence = self.cadence();
            let drift = self.lag_error_frames();
            if drift.abs() > DRIFT_TRIGGER * cadence {
                // Onset protection applies inside the candidate search; the
                // HARD trigger forces through it eventually.
                self.try_splice(drift, onsets);
            } else if drift.abs() > OPPORTUNISTIC_DRIFT * cadence && self.in_masked_window(onsets) {
                // Beat-synchronous placement: a pending correction hides
                // best right after a hit.
                self.try_splice(drift, onsets);
            } else if drift.abs() > OPPORTUNISTIC_DRIFT * cadence
                && self.energy_avg > 1e-6
                && self.region_rms(self.read_pos, CORR_WINDOW)
                    < QUIET_SPLICE_RATIO * self.energy_avg
                && self.region_rms(self.read_pos + drift, CORR_WINDOW)
                    < QUIET_SPLICE_RATIO * self.energy_avg
            {
                // Quiet-gap placement: correct early where nothing is
                // playing loudly, so fewer splices are ever forced near
                // attacks. (A drift-graded gate was tried and won on a
                // 3-track corpus but washed out on 4 — the fixed strict
                // gate is simpler and at least as good; see autoresearch
                // log #43/#51.)
                self.try_splice(drift, onsets);
            } else if deviation < MILD_DEV && settled_drift.abs() > MILD_DRIFT_TRIGGER {
                // Mild-motion bounded recenter (ROADMAP Stage 15): during
                // sustained gentle rides the seam combs for the whole ride
                // unless drift is kept small; the bounded landing (must
                // actually recenter, or skip and let drift ride to the
                // normal trigger) avoids the periodic-content parking that
                // an unbounded early splice causes.
                self.try_rest_splice(drift, onsets);
            } else if at_rest && settled_drift.abs() > REST_SPLICE_DRIFT {
                // At sustained rest a parked drift comb-filters the
                // crossover overlap against the low band's fixed delay;
                // recenter cleanly (bounded: the landing must actually
                // recenter, or the splice is skipped and the trim bleeds
                // the drift instead).
                self.try_rest_splice(drift, onsets);
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
                let out_kernel = SincReadKernel::at(&self.table, self.xfade_from);
                let in_kernel = SincReadKernel::at(&self.table, self.read_pos);
                for (ch, out) in self.channels.iter().zip(io.iter_mut()) {
                    let a = out_kernel.read(&ch.ring);
                    let b = in_kernel.read(&ch.ring);
                    out[i] = g_out * a + g_in * b;
                }
                self.xfade_from += t;
                self.read_pos += t;
                self.xfade_remaining -= 1;
            } else {
                let kernel = SincReadKernel::at(&self.table, self.read_pos);
                for (ch, out) in self.channels.iter().zip(io.iter_mut()) {
                    out[i] = kernel.read(&ch.ring);
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
        // Asymmetric force threshold: slowdown (negative) drift is capped
        // by the write-head headroom ([`SLOWDOWN_FORCE_CAP`] — the
        // stretched hard trigger does not fit that side); speed-up keeps
        // the full stretched band against the ring's old-audio headroom.
        let stretched = HARD_TRIGGER * self.cadence();
        let threshold = if drift < 0.0 {
            stretched.min(SLOWDOWN_FORCE_CAP)
        } else {
            stretched
        };
        let force = drift.abs() >= threshold && self.failed_attempts > 0;
        if self.plan_splice(drift, force, f64::INFINITY, onsets) {
            self.failed_attempts = 0;
        } else {
            self.failed_attempts = self.failed_attempts.saturating_add(1);
        }
    }

    /// Rest-recenter splice: only candidates that actually bring the lag
    /// back under the rest threshold qualify. Without this bound, highly
    /// periodic content lets the zero-jump candidate win on correlation
    /// (it is literally the same audio) and the corrector enters a limit
    /// cycle of no-op splices — hundreds of pointless fades per second
    /// with the drift (and its seam de-phasing) parked forever, while the
    /// per-splice sub-sample refinement cancels the rest trim's bleed.
    /// When no candidate qualifies the splice is skipped and the trim
    /// converges the drift unimpeded.
    fn try_rest_splice(&mut self, drift: f64, onsets: &[OnsetEvent]) {
        if self.plan_splice(drift, false, REST_SPLICE_DRIFT, onsets) {
            self.failed_attempts = 0;
        }
    }

    /// Plans and starts a correlation-matched splice toward the nominal
    /// lag; unless `force`, a landing region that reads as a transient
    /// postpones it, and candidates whose fade would overlap an artifact
    /// onset's protection window are excluded from the search.
    fn plan_splice(
        &mut self,
        drift: f64,
        force: bool,
        max_residual: f64,
        onsets: &[OnsetEvent],
    ) -> bool {
        // Outgoing span: the fade also reads from the current cursor.
        if !force && self.span_hits_onset(onsets, self.read_pos) {
            return false;
        }
        // Jump the read cursor so the lag returns to nominal: with
        // lag' = write − (read + jump), jump = lag − nominal = drift.
        let nominal_jump = drift;
        // …searching around that jump for the offset whose audio best
        // continues what the cursor is currently playing. The base window
        // and the full candidate span are premixed once; every candidate's
        // normalized correlation is then two flat dot products.
        let (mut best_jump, mut best_score) = (nominal_jump, f64::MIN);
        let base = self.read_pos;
        let lo = nominal_jump as isize - SEARCH_RANGE;
        let hi = nominal_jump as isize + SEARCH_RANGE;
        let a0 = base.floor() as isize;
        mix_channels(&self.channels, a0, &mut self.corr_a);
        let a_sq = dot_f64(&self.corr_a, &self.corr_a);
        mix_channels(&self.channels, a0 + lo, &mut self.corr_b);
        for jump in lo..=hi {
            // Residual-drift bound (rest splices): the landing must leave
            // the lag within `max_residual` of nominal or the splice is
            // pointless damage.
            if (drift - jump as f64).abs() > max_residual {
                continue;
            }
            let candidate = base + jump as f64;
            if !self.readable_span(candidate, CORR_WINDOW + XFADE_FRAMES) {
                continue;
            }
            if !force && self.span_hits_onset(onsets, candidate) {
                continue;
            }
            let off = (jump - lo) as usize;
            let b_win = &self.corr_b[off..off + CORR_WINDOW];
            let b_sq = dot_f64(b_win, b_win);
            let norm = (a_sq * b_sq).sqrt();
            let corr = if norm < 1e-12 {
                0.0
            } else {
                dot_f64(&self.corr_a, b_win) / norm
            };
            // Mild distance penalty: periodic content scores every
            // period-grid candidate identically, and un-penalized ties
            // resolve to the search edge — parking the elastic drift ~a
            // full search range off nominal instead of converging to it.
            let distance = (jump as f64 - nominal_jump).abs() / SEARCH_RANGE as f64;
            let score = corr - 0.02 * distance;
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
            // The base window is a prefix of the coarse search's premix; the
            // candidates' sinc reads all draw from one premixed span, and
            // each candidate's fractional phase is constant across its
            // window, so one kernel row serves all its reads.
            let a_win = &self.corr_a[..XFADE_FRAMES];
            let a_sq = dot_f64(a_win, a_win);
            let span_lo = (base + best_jump - FINE_SEARCH_RADIUS).floor() as isize + 1
                - READ_HALF_TAPS as isize;
            mix_channels(&self.channels, span_lo, &mut self.frac_mix);
            let mut row = [0.0f32; READ_TAPS];
            let mut cs = [0.0f64; GRID];
            let (mut best_k, mut best_c) = (0usize, f64::MIN);
            for (k, c) in cs.iter_mut().enumerate() {
                let frac_off = (k as f64 - FINE_SEARCH_STEPS as f64) * step;
                let b_pos = base + best_jump + frac_off;
                let b0 = b_pos.floor();
                let wsum = self.table.fill_row(b_pos - b0, &mut row);
                let base_off = (b0 as isize + 1 - READ_HALF_TAPS as isize - span_lo) as usize;
                let (mut dot, mut b_sq) = (0.0f64, 0.0f64);
                for (i, &a) in a_win.iter().enumerate() {
                    let seg = &self.frac_mix[base_off + i..base_off + i + READ_TAPS];
                    let b = if wsum.abs() <= 1e-12 {
                        0.0
                    } else {
                        dot_f64_f32(seg, &row) / wsum
                    };
                    dot += a * b;
                    b_sq += b * b;
                }
                let norm = (a_sq * b_sq).sqrt();
                *c = if norm < 1e-12 { 0.0 } else { dot / norm };
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
        // The elastic band is the cadence-stretched hard trigger inside
        // the primary window (Stage 18).
        assert!(
            error <= HARD_TRIGGER * CADENCE_SCALE_MAX + BLOCK_FRAMES as f64,
            "lag error {error:.0} frames escaped the elastic band"
        );
    }

    #[test]
    fn cadence_scale_tapers_with_transposition() {
        // Full stretch through the primary DJ window (both directions)…
        assert_eq!(cadence_scale(0.9), CADENCE_SCALE_MAX);
        assert_eq!(cadence_scale(1.0), CADENCE_SCALE_MAX);
        assert_eq!(cadence_scale(CADENCE_TAPER_START), CADENCE_SCALE_MAX);
        // …released to shipped cadence before the slowdown range edge
        // (the measured T=1.25 cursor stall must run at scale 1)…
        assert_eq!(cadence_scale(CADENCE_TAPER_END), 1.0);
        assert_eq!(cadence_scale(1.25), 1.0);
        // …monotone in between.
        let mid = cadence_scale((CADENCE_TAPER_START + CADENCE_TAPER_END) * 0.5);
        assert!(mid > 1.0 && mid < CADENCE_SCALE_MAX);
    }

    #[test]
    fn cadence_stretch_halves_splice_rate_in_window() {
        // At T=1.05 (inside the primary window) drift accrues at
        // |T−1| frames per output frame; with the stretched trigger the
        // splice count must sit near accrual / (DRIFT_TRIGGER * 2) —
        // meaningfully below the shipped-cadence rate, but still actively
        // correcting.
        let secs = 20;
        let frames = 44_100 * secs;
        let mut corrector = SolaCorrector::new(1, LAG);
        let input = sine(500.0, frames, 0.4);
        let _ = run(&mut corrector, &input, 1.05);
        let accrual = frames as f64 * 0.05;
        let scale1_rate = accrual / DRIFT_TRIGGER;
        let count = corrector.splice_count() as f64;
        assert!(
            count < 0.7 * scale1_rate,
            "cadence stretch inactive: {count} splices vs ~{scale1_rate:.0} at shipped cadence"
        );
        assert!(
            count > 0.3 * scale1_rate,
            "implausibly few splices ({count}) — corrector may be stalling"
        );
    }

    #[test]
    fn ride_slope_restores_shipped_cadence() {
        // With a nonzero rate slope (a ride in progress) the cadence
        // stretch must be inert: the slope-tracked pitch correction is
        // proportional to parked drift and clamps, so the widened band
        // measurably detunes rides (A/B matrix: cents p95 5.65 vs bound
        // 1.5 before this gate). Splice rate must sit near the shipped
        // cadence, not the halved one.
        let secs = 20;
        let frames = 44_100 * secs;
        let mut corrector = SolaCorrector::new(1, LAG);
        corrector.set_rate_slope(1e-4);
        let input = sine(500.0, frames, 0.4);
        corrector.set_transposition(1.05);
        let mut block = [[0.0f32; BLOCK_FRAMES]; 1];
        for chunk in input.chunks_exact(BLOCK_FRAMES) {
            block[0].copy_from_slice(chunk);
            corrector.process_block(&mut block, &[]);
        }
        let scale1_rate = frames as f64 * 0.05 / DRIFT_TRIGGER;
        let count = corrector.splice_count() as f64;
        assert!(
            count > 0.7 * scale1_rate,
            "cadence stretch active during a ride: {count} splices vs \
             ~{scale1_rate:.0} expected at shipped cadence"
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
                        band_flux: [1.0; 4],
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

//! Time-domain SOLA pitch corrector for small transpositions.
//!
//! At DJ transpositions (|T − 1| below ~5%) a splice-based corrector is
//! transparent on transients and needs no FFT: the output reads from a ring
//! of recent band audio at rate `T` (windowed-sinc interpolation), and when
//! the read cursor drifts from its nominal lag the corrector splices — a
//! correlation-matched jump with an equal-power crossfade, placed at
//! low-energy moments when possible (onset snapping deepens in Stage 4).
//!
//! The nominal lag equals the PV corrector's constant latency, so the two
//! correctors are interchangeable mid-stream: handoff is latency-neutral
//! and the low band's matching delay never changes. The SOLA algorithm
//! itself only needs the sinc margin (~0.5 ms) — the shared lag exists for
//! the chain, not the splicer.
//!
//! Splice decisions are made once per block on the channel mix and applied
//! to every channel identically, keeping the stereo image intact.

use std::sync::Arc;

use crate::core::resample::{SincInterpTable, STREAM_SINC_HALF_TAPS};
use crate::engine::stage::BLOCK_FRAMES;

/// Ring capacity per channel (power of two), in frames.
const RING_LEN: usize = 4_096;
const RING_MASK: usize = RING_LEN - 1;

/// Drift from the nominal lag that triggers a splice, in frames.
const DRIFT_TRIGGER: f64 = 192.0;
/// Drift at which a splice is forced even through a transient.
const HARD_TRIGGER: f64 = 320.0;
/// Correlation search half-range around the nominal jump, in frames.
const SEARCH_RANGE: isize = 160;
/// Correlation window length, in frames.
const CORR_WINDOW: usize = 96;
/// Splice crossfade length, in frames (equal-power).
const XFADE_FRAMES: usize = 96;
/// Sinc half-width plus slack the read cursor keeps behind the write head.
const MIN_READ_MARGIN: f64 = (STREAM_SINC_HALF_TAPS + 4) as f64;
/// Candidate-region energy above this multiple of the rolling average reads
/// as a transient: postpone the splice until [`HARD_TRIGGER`].
const TRANSIENT_POSTPONE_RATIO: f64 = 3.0;

/// One channel's ring state.
#[derive(Debug)]
struct SolaChannel {
    ring: Vec<f32>,
}

/// Elastic time-domain pitch corrector across all channels.
#[derive(Debug)]
pub(crate) struct SolaCorrector {
    channels: Vec<SolaChannel>,
    table: Arc<SincInterpTable>,
    /// Frames written since reset (shared across channels).
    write_abs: u64,
    /// Absolute fractional read cursor (shared: channels are lockstep).
    read_pos: f64,
    /// Crossfade-out cursor while a splice fade is active.
    xfade_from: f64,
    /// Remaining crossfade frames (0 = not fading).
    xfade_remaining: usize,
    transposition: f64,
    nominal_lag: f64,
    /// Rolling average of block RMS on the channel mix (transient gate).
    energy_avg: f64,
    /// Splices executed since reset (observability for tests/QA).
    splice_count: u64,
    /// A recentering splice is pending (keylock release handoff prep).
    recenter_requested: bool,
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
            table: SincInterpTable::new_stream_default(),
            write_abs: 0,
            // Starting the cursor a full lag behind zero realizes the
            // constant delay exactly: the first `nominal_lag` reads land on
            // never-written (zero) ring slots and emit the priming silence.
            read_pos: -(nominal_lag_frames as f64),
            xfade_from: 0.0,
            xfade_remaining: 0,
            transposition: 1.0,
            nominal_lag: nominal_lag_frames as f64,
            energy_avg: 0.0,
            splice_count: 0,
            recenter_requested: false,
        }
    }

    pub(crate) fn set_transposition(&mut self, transposition: f64) {
        self.transposition = if transposition.is_finite() {
            transposition.clamp(0.75, 1.35)
        } else {
            1.0
        };
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

    /// Hard-recenters the read cursor to the exact nominal lag, discarding
    /// any elastic drift and in-flight crossfade. Audibly discontinuous —
    /// only call while this corrector's output is faded out (the keylock
    /// stage uses it to phase-align a PV→SOLA handoff for free).
    pub(crate) fn recenter_hard(&mut self) {
        self.read_pos = self.write_abs as f64 - self.nominal_lag;
        self.xfade_remaining = 0;
        self.recenter_requested = false;
    }

    /// Requests an audible-path recentering splice: at the next opportunity
    /// the corrector splices its elastic drift away (correlation-matched,
    /// so the jump lands on the dominant content's period grid nearest
    /// zero — leaving the output phase-aligned with a fixed-latency
    /// reference like the PV corrector). Keylock release-handoff prep.
    pub(crate) fn request_recenter_splice(&mut self) {
        self.recenter_requested = true;
    }

    /// Whether the cursor is recentered enough to hand off: no pending
    /// recenter request and no in-flight splice fade.
    pub(crate) fn is_recentered(&self) -> bool {
        !self.recenter_requested && self.xfade_remaining == 0
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
        self.recenter_requested = false;
    }

    /// Processes one fixed block for every channel in lockstep: writes the
    /// inputs into the rings, splices if the elastic lag calls for it, then
    /// synthesizes the outputs in place.
    pub(crate) fn process_block(&mut self, io: &mut [[f32; BLOCK_FRAMES]]) {
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
        //    per block at the clamp bounds).
        if self.xfade_remaining == 0 {
            let drift = self.lag_error_frames();
            if self.recenter_requested {
                // Handoff prep: splice the drift away now, bypassing the
                // transient postpone (a pending handoff outranks it).
                if drift.abs() < 8.0 || self.try_splice_forced(drift) {
                    self.recenter_requested = false;
                }
            } else if drift.abs() > DRIFT_TRIGGER {
                self.try_splice(drift);
            }
        }

        // 3) Synthesis.
        let t = self.transposition;
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
        debug_assert!(
            self.write_abs as f64 - self.read_pos.max(self.xfade_from) >= MIN_READ_MARGIN - 1.0,
            "SOLA read overtook the write head"
        );
    }

    /// Drift-triggered splice: transient postpone applies until the drift
    /// is critical.
    fn try_splice(&mut self, drift: f64) {
        let force = drift.abs() >= HARD_TRIGGER;
        let _ = self.plan_splice(drift, force);
    }

    /// Handoff-requested splice: always forced. Returns whether it ran.
    fn try_splice_forced(&mut self, drift: f64) -> bool {
        self.plan_splice(drift, true)
    }

    /// Plans and starts a correlation-matched splice toward the nominal
    /// lag; unless `force`, a landing region that reads as a transient
    /// postpones it.
    fn plan_splice(&mut self, drift: f64, force: bool) -> bool {
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
            let score = self.mix_correlation(base, candidate, CORR_WINDOW);
            if score > best_score {
                best_score = score;
                best_jump = jump as f64;
            }
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
fn sinc_read(ring: &[f32], pos: f64, table: &SincInterpTable) -> f32 {
    let center = pos.floor();
    let frac = pos - center;
    let center = center as isize;
    let half = STREAM_SINC_HALF_TAPS as isize;
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
            corrector.process_block(&mut block);
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
            corrector.process_block(&mut block);
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

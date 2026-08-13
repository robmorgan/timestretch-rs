//! Direct-ratio wide head (ROADMAP Stage 19): the phase vocoder OWNS the
//! tempo axis for the WideKeylock profile.
//!
//! The Stage 14 attribution chain (three blind sessions + probes,
//! 2026-08-13) convicted the Stage 11 topology — varispeed tempo prepass
//! → PV stretch → post-resampler transpose — as the wide path's "roboty
//! background noise" floor, with resets, M/S, shared PV code, and
//! streaming chunking each cleared. This head is the configuration that
//! auditioned clean: the PV runs at the direct tempo ratio on
//! source-rate audio — no varispeed prepass, no post-resampler — so
//! pitch is preserved structurally (there is no pitch axis to disturb
//! during a rate gesture; the Stage 19 dynamics kill-experiment measured
//! instant full-range ratio steps click-free, pitch unmoved).
//!
//! Because the PV consumes source frames at the tempo rate, this head —
//! not the varispeed — is the profile's demand inverter: it ingests
//! interleaved source frames, windows them per processed channel, renders
//! one FFT hop at a time, and emits fixed-cap output like the varispeed
//! head so the graph's feed loop and timeline bookkeeping carry over.
//!
//! Contract changes vs the varispeed head, recorded deliberately:
//! - **Retarget landing is hop-quantized** (256 source frames ≈ 5.8 ms):
//!   a rate change applies to the next rendered hop. Emission COUNTS stay
//!   exact (the pending-output FIFO honors per-feed caps to the frame),
//!   so timeline accounting is unchanged; only the audible boundary of
//!   the new rate is hop-granular. The log-slew smooths it regardless.
//! - **Output-frame latency varies with ratio** (the analysis window
//!   spans [`WIDE_FFT`] SOURCE frames = `WIDE_FFT * ratio` output
//!   frames). The head reports its lookahead in source frames, which is
//!   constant; the profile's output-side latency figure is derived at
//!   the graph level.
//!
//! Stereo runs the corrected path in mid/side (the Stage 14 measurement:
//! M/S is source-faithful; per-channel processing manufactures ~16 dB of
//! side energy by decorrelation). A deliberate bounded width treatment,
//! if the owner preference for a wider render holds, layers on top later
//! — it does not come back as an accident of uncoupled processing.

use crate::core::window::WindowType;
use crate::stretch::{PhaseLockingMode, PhaseVocoder};

/// FFT and hop mirror the wide stage (FFT 2048, hop FFT/8 — the
/// mandatory wide overlap; see `wide_keylock.rs`).
pub(crate) const WIDE_FFT: usize = 2_048;
pub(crate) const WIDE_HOP: usize = 256;
/// Tempo ratio clamp (output/input length ratio; rate = 1/ratio). Same
/// range the wide profile ships today.
const RATIO_MIN: f64 = 0.5;
const RATIO_MAX: f64 = 2.0;
/// Per-hop log-slew bound on the ratio, scaled from the wide stage's
/// 0.05 ln per 32-frame block to the 256-frame hop cadence: the same
/// ~30 ms full-range settle, applied where this head quantizes anyway.
const RATIO_SLEW_LN_PER_HOP: f64 = 0.05 * (WIDE_HOP as f64 / 32.0);
/// Upper bound on frames a single hop can emit (hop × RATIO_MAX).
pub(crate) const MAX_OUT_PER_HOP: usize = (WIDE_HOP as f64 * RATIO_MAX) as usize;

/// One processed channel: the PV, its source window, and its share of
/// the pending (rendered, not yet emitted) output.
struct HeadChannel {
    pv: PhaseVocoder,
    /// Source-rate analysis window (grows to [`WIDE_FFT`], drains by
    /// [`WIDE_HOP`] per rendered hop).
    window: Vec<f32>,
    /// Rendered output awaiting emission (drained by `feed_capped`'s
    /// cap-honoring pop).
    pending: Vec<f32>,
    /// PV render scratch.
    chunk: Vec<f32>,
}

/// Direct-ratio PV head: demand inverter and corrector in one, for the
/// WideKeylock profile.
pub(crate) struct WidePvHead {
    channels: Vec<HeadChannel>,
    /// Audio channel count (the processed domain equals it: M/S for
    /// stereo is a reversible 2-channel transform).
    num_channels: usize,
    /// Slewed tempo ratio currently applied to rendered hops.
    ratio: f64,
    /// Per-channel emission buffers exposed via [`Self::output`]
    /// (deinterleaved, like the varispeed head's).
    out: Vec<Vec<f32>>,
    /// Source frames ingested since reset.
    ingested: u64,
    sample_rate: u32,
}

impl WidePvHead {
    pub(crate) fn new(sample_rate: u32, num_channels: usize) -> Self {
        let mk = || HeadChannel {
            pv: PhaseVocoder::with_options(
                WIDE_FFT,
                WIDE_HOP,
                1.0,
                sample_rate,
                100.0,
                WindowType::Hann,
                PhaseLockingMode::Identity,
            ),
            window: Vec::with_capacity(WIDE_FFT + 2 * WIDE_HOP),
            pending: Vec::with_capacity(8 * MAX_OUT_PER_HOP),
            chunk: Vec::with_capacity(4 * WIDE_FFT),
        };
        Self {
            channels: (0..num_channels).map(|_| mk()).collect(),
            num_channels,
            ratio: 1.0,
            out: (0..num_channels)
                .map(|_| Vec::with_capacity(8 * MAX_OUT_PER_HOP))
                .collect(),
            ingested: 0,
            sample_rate,
        }
    }

    /// Ingests interleaved source frames at `rate` (tempo rate; ratio is
    /// its inverse), renders any full hops, and emits up to `max_out`
    /// frames per channel into [`Self::output`]. Returns frames emitted.
    pub(crate) fn feed_capped(&mut self, interleaved: &[f32], rate: f64, max_out: usize) -> usize {
        let target_ratio = if rate.is_finite() && rate > 0.0 {
            (1.0 / rate).clamp(RATIO_MIN, RATIO_MAX)
        } else {
            1.0
        };
        let frames = interleaved.len() / self.num_channels.max(1);
        self.ingested += frames as u64;

        // Ingest: stereo encodes to M/S; other counts pass per channel.
        let stereo = self.num_channels == 2;
        for f in 0..frames {
            if stereo {
                let l = interleaved[f * 2];
                let r = interleaved[f * 2 + 1];
                self.channels[0].window.push(0.5 * (l + r));
                self.channels[1].window.push(0.5 * (l - r));
            } else {
                for ch in 0..self.num_channels {
                    self.channels[ch]
                        .window
                        .push(interleaved[f * self.num_channels + ch]);
                }
            }
        }

        // Render every full hop, slewing the ratio once per hop.
        while self.channels[0].window.len() >= WIDE_FFT {
            let step = (target_ratio.ln() - self.ratio.ln())
                .clamp(-RATIO_SLEW_LN_PER_HOP, RATIO_SLEW_LN_PER_HOP);
            self.ratio = (self.ratio.ln() + step).exp();
            for ch in &mut self.channels {
                ch.pv.set_stretch_ratio(self.ratio);
                let result = ch
                    .pv
                    .process_streaming_into(&ch.window[..WIDE_FFT], &mut ch.chunk);
                debug_assert!(result.is_ok(), "wide pv head render failed: {result:?}");
                if result.is_ok() {
                    ch.pending.extend_from_slice(&ch.chunk);
                }
                let remaining = ch.window.len() - WIDE_HOP;
                ch.window.copy_within(WIDE_HOP.., 0);
                ch.window.truncate(remaining);
            }
        }

        // Emit up to the cap (channels are lockstep by construction).
        let available = self
            .channels
            .iter()
            .map(|c| c.pending.len())
            .min()
            .unwrap_or(0);
        let emit = available.min(max_out);
        for (ch, state) in self.channels.iter_mut().enumerate() {
            self.out[ch].clear();
            self.out[ch].extend_from_slice(&state.pending[..emit]);
            state.pending.copy_within(emit.., 0);
            let rest = state.pending.len() - emit;
            state.pending.truncate(rest);
        }
        // Decode M/S back to L/R at the emission boundary.
        if stereo && emit > 0 {
            for i in 0..emit {
                let (m, s) = (self.out[0][i], self.out[1][i]);
                self.out[0][i] = m + s;
                self.out[1][i] = m - s;
            }
        }
        emit
    }

    /// Most recent emission for `ch` (deinterleaved), valid until the
    /// next [`Self::feed_capped`].
    pub(crate) fn output(&self, ch: usize) -> &[f32] {
        &self.out[ch]
    }

    /// Source position corresponding to the analysis frontier: frames
    /// ingested minus the un-analyzed window tail.
    pub(crate) fn source_pos(&self) -> f64 {
        self.ingested as f64 - self.channels[0].window.len() as f64
    }

    /// Constant SOURCE-frame lookahead (the analysis window).
    pub(crate) fn lookahead_frames(&self) -> usize {
        WIDE_FFT
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub(crate) fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.pv.reset_phase_state();
            ch.pv.reset_streaming_state();
            ch.window.clear();
            ch.pending.clear();
            ch.chunk.clear();
        }
        for o in &mut self.out {
            o.clear();
        }
        self.ratio = 1.0;
        self.ingested = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44_100;

    fn drive(head: &mut WidePvHead, input: &[f32], rate: f64, chunk_frames: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for chunk in input.chunks(chunk_frames) {
            let mut emitted = head.feed_capped(chunk, rate, usize::MAX);
            out.extend_from_slice(&head.output(0)[..emitted]);
            // Drain anything the cap withheld (none with MAX, but keeps
            // the pattern honest for capped tests).
            while emitted > 0 {
                emitted = head.feed_capped(&[], rate, usize::MAX);
                out.extend_from_slice(&head.output(0)[..emitted]);
            }
        }
        out
    }

    #[test]
    fn output_length_tracks_ratio() {
        for rate in [0.7f64, 1.0, 1.5, 2.0] {
            let mut head = WidePvHead::new(SR, 1);
            let n = SR as usize * 4;
            let input: Vec<f32> = (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR as f64).sin() as f32)
                .collect();
            let out = drive(&mut head, &input, rate, 32);
            let expected = n as f64 / rate;
            let err = (out.len() as f64 - expected).abs();
            assert!(
                err < 2.0 * WIDE_FFT as f64 * (1.0 / rate).max(1.0),
                "rate {rate}: output {} vs expected {expected:.0} (err {err:.0})",
                out.len()
            );
        }
    }

    #[test]
    fn pitch_is_preserved_and_click_free_under_instant_steps() {
        let mut head = WidePvHead::new(SR, 1);
        let n = SR as usize * 8;
        let amp = 0.4f32;
        let input: Vec<f32> = (0..n)
            .map(|i| amp * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SR as f64).sin() as f32)
            .collect();
        // Instant rate flips every second of source time.
        let mut out = Vec::new();
        for (k, chunk) in input.chunks(SR as usize).enumerate() {
            let rate = if k % 2 == 0 { 0.7 } else { 2.0 };
            for block in chunk.chunks(32) {
                let emitted = head.feed_capped(block, rate, usize::MAX);
                out.extend_from_slice(&head.output(0)[..emitted]);
            }
        }
        // Click bound: source-pitch tone slew x3 (the soak criterion) —
        // pitch is preserved, so the output tone is still 440 Hz.
        let bound = amp * (2.0 * std::f64::consts::PI * 440.0 / SR as f64) as f32 * 3.0;
        let mut worst = 0.0f32;
        for w in out[8_192..out.len() - 4_096].windows(2) {
            worst = worst.max((w[1] - w[0]).abs());
        }
        assert!(
            worst <= bound,
            "instant retargets click through the wide head: {worst:.5} > {bound:.5}"
        );
        // Zero-crossing pitch over a mid window.
        let mid = &out[out.len() / 2 - 32_768..out.len() / 2 + 32_768];
        let (mut first, mut last, mut count) = (None, None, 0usize);
        for i in 1..mid.len() {
            let (a, b) = (mid[i - 1] as f64, mid[i] as f64);
            if a <= 0.0 && b > 0.0 {
                let t = (i - 1) as f64 + a / (a - b);
                if first.is_none() {
                    first = Some(t);
                }
                last = Some(t);
                count += 1;
            }
        }
        let f = match (first, last) {
            (Some(a), Some(b)) if count >= 2 => (count - 1) as f64 * SR as f64 / (b - a),
            _ => 0.0,
        };
        assert!(
            (f - 440.0).abs() < 2.0,
            "pitch must survive instant retargets: measured {f:.1} Hz"
        );
    }

    #[test]
    fn emission_caps_are_honored_to_the_frame() {
        let mut head = WidePvHead::new(SR, 1);
        let input = vec![0.1f32; WIDE_FFT + 4 * WIDE_HOP];
        let emitted = head.feed_capped(&input, 1.0, 7);
        assert!(emitted <= 7, "cap violated: emitted {emitted}");
        let more = head.feed_capped(&[], 1.0, usize::MAX);
        assert!(more > 0, "withheld frames must drain on the next feed");
    }

    #[test]
    fn stereo_ms_round_trip_is_faithful_on_identical_channels() {
        let mut head = WidePvHead::new(SR, 2);
        let n = SR as usize * 2;
        let mono: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 330.0 * i as f64 / SR as f64).sin() as f32 * 0.4)
            .collect();
        let interleaved: Vec<f32> = mono.iter().flat_map(|&v| [v, v]).collect();
        let mut l = Vec::new();
        let mut r = Vec::new();
        for chunk in interleaved.chunks(64) {
            let emitted = head.feed_capped(chunk, 1.3, usize::MAX);
            l.extend_from_slice(&head.output(0)[..emitted]);
            r.extend_from_slice(&head.output(1)[..emitted]);
        }
        // Identical channels: side is exactly zero in, so L == R out.
        for (i, (&a, &b)) in l.iter().zip(&r).enumerate() {
            assert_eq!(a, b, "identical channels diverged at {i}");
        }
    }
}

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
use crate::engine::stages::keylock::KEYLOCK_TOGGLE_FADE_FRAMES;
use crate::engine::stages::varispeed::{FEED_CHUNK_FRAMES, VarispeedHead};
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
    /// False until the first hop renders: the first render snaps the
    /// ratio (no slew from unity) and runs the mirror-padded OLA warmup
    /// so emission starts fully overlapped at source frame 0 (the same
    /// start contract as the batch `process()` path).
    started: bool,
    /// Warmup scratch: one mirrored analysis window.
    warm_window: Vec<f32>,
    /// Raw (pitch-follows) arm for the live keylock toggle: a plain
    /// varispeed fed the same source at the same rate. Content-aligned
    /// with the corrected arm by construction (both start at source
    /// frame 0); emission is lockstep on the min pending.
    raw: VarispeedHead,
    /// Per-audio-channel pending raw-arm output.
    raw_pending: Vec<Vec<f32>>,
    /// Interleaved source-side delay ahead of the raw arm: the PV renders
    /// source frame `p` only once `p + WIDE_FFT` is ingested, so the raw
    /// varispeed is fed [`Self::raw_input_delay_frames`] behind ingest.
    /// This keeps the two arms' PRODUCTION contemporaneous (near-zero raw
    /// backlog — a rate change reaches the audible arm within a block)
    /// while their CONTENT stays aligned at emission.
    raw_delay: Vec<f32>,
    /// Keylock target (0.0 = raw varispeed, 1.0 = corrected).
    keylock_target: f64,
    /// Smoothed toggle weight (NaN = snap to target on next emission).
    keylock_weight: f32,
}

/// Pending-surplus bound before the inaudible arm is resynced (dropped
/// forward) to the audible one: the two arms' cumulative production can
/// drift a bounded amount per retarget (the PV slews hop-quantized, the
/// varispeed ramps per feed), and unchecked the surplus would grow over
/// hours of gestures. Resync only ever drops frames from an arm whose
/// toggle weight makes it inaudible.
const ARM_SURPLUS_MAX: usize = 8_192;

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
            started: false,
            warm_window: vec![0.0; WIDE_FFT],
            raw: VarispeedHead::new(num_channels),
            raw_delay: Vec::with_capacity((WIDE_FFT + 4 * FEED_CHUNK_FRAMES) * num_channels),
            raw_pending: (0..num_channels)
                .map(|_| Vec::with_capacity(2 * ARM_SURPLUS_MAX))
                .collect(),
            keylock_target: 1.0,
            keylock_weight: f32::NAN,
        }
    }

    /// Source frames the raw arm's input runs behind ingest (the PV's
    /// analysis-window lead minus the varispeed's own kernel lookahead).
    fn raw_input_delay_frames(&self) -> usize {
        WIDE_FFT.saturating_sub(self.raw.lookahead_frames())
    }

    /// Keylock (pitch-correction) toggle target; smoothed at emission.
    pub(crate) fn set_keylock(&mut self, target: f64) {
        self.keylock_target = target.clamp(0.0, 1.0);
    }

    /// Frames rendered but not yet emitted (the graph keeps feeding while
    /// this can still satisfy demand even if the source ring is dry).
    pub(crate) fn pending_frames(&self) -> usize {
        let pv = self
            .channels
            .iter()
            .map(|c| c.pending.len())
            .min()
            .unwrap_or(0);
        let raw = self.raw_pending.iter().map(Vec::len).min().unwrap_or(0);
        pv.min(raw)
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

        // Raw arm: the same source at the same rate through a plain
        // varispeed; content-aligned with the corrected arm from source
        // frame 0. Its production is buffered and emitted lockstep.
        self.raw_delay.extend_from_slice(interleaved);
        let delay_samples = self.raw_input_delay_frames() * self.num_channels.max(1);
        while self.raw_delay.len() > delay_samples {
            let take = (self.raw_delay.len() - delay_samples)
                .min(FEED_CHUNK_FRAMES * self.num_channels.max(1));
            let raw_produced = self
                .raw
                .feed_capped(&self.raw_delay[..take], rate, usize::MAX);
            self.raw_delay.copy_within(take.., 0);
            let rest = self.raw_delay.len() - take;
            self.raw_delay.truncate(rest);
            for ch in 0..self.num_channels {
                let out = self.raw.output(ch);
                self.raw_pending[ch].extend_from_slice(&out[..raw_produced]);
            }
        }

        // Render every full hop, slewing the ratio once per hop.
        while self.channels[0].window.len() >= WIDE_FFT {
            if !self.started {
                self.started = true;
                // Stream start: snap to the commanded ratio (slew is for
                // CHANGES, not cold start) and run the mirror-padded OLA
                // warmup — render the windows that would precede source
                // frame 0 under mirror extension, discarding their
                // output, so the first kept hop is fully overlapped.
                self.ratio = target_ratio;
                let warmup_hops = WIDE_FFT / WIDE_HOP - 1;
                for k in (1..=warmup_hops).rev() {
                    let shift = k * WIDE_HOP;
                    if k == 1 {
                        // The earlier warmup hops exist only to converge
                        // the OLA weights; their phase state is seeded
                        // from MIRRORED content, whose per-bin offsets
                        // would persist in the accumulators and partially
                        // cancel low bins forever (the pre-Stage-14
                        // batch path's sub-bass imbalance — measured
                        // 2.47 vs ideal 0.54 two-tone balance at rate
                        // 2.0). Re-seed at the LAST warmup window, which
                        // is mostly real content: the dominant overlap
                        // into the kept stream is then phase-coherent.
                        for ch in &mut self.channels {
                            ch.pv.reset_phase_state();
                        }
                    }
                    for ch in &mut self.channels {
                        for (i, w) in self.warm_window.iter_mut().enumerate() {
                            // Window position i maps to source i - shift;
                            // negative indices mirror around frame 0.
                            let src = i as isize - shift as isize;
                            let idx = if src < 0 {
                                (-src) as usize
                            } else {
                                src as usize
                            };
                            *w = ch.window[idx.min(WIDE_FFT - 1)];
                        }
                        ch.pv.set_stretch_ratio(self.ratio);
                        let result = ch
                            .pv
                            .process_streaming_into(&self.warm_window, &mut ch.chunk);
                        debug_assert!(result.is_ok(), "wide head warmup failed: {result:?}");
                        // Warmup output is discarded: it belongs to the
                        // mirrored pre-roll, not the stream.
                    }
                }
            }
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

        // Inaudible-arm resync: bounded per-retarget production drift
        // between the arms must not accumulate; drop surplus only from
        // an arm the toggle weight silences.
        let pv_avail = self
            .channels
            .iter()
            .map(|c| c.pending.len())
            .min()
            .unwrap_or(0);
        let raw_avail = self.raw_pending.iter().map(Vec::len).min().unwrap_or(0);
        let w = if self.keylock_weight.is_nan() {
            self.keylock_target as f32
        } else {
            self.keylock_weight
        };
        if w >= 0.999 && raw_avail > pv_avail + ARM_SURPLUS_MAX {
            let drop = raw_avail - pv_avail;
            for pending in &mut self.raw_pending {
                pending.copy_within(drop.., 0);
                let rest = pending.len() - drop;
                pending.truncate(rest);
            }
        } else if w <= 0.001 && pv_avail > raw_avail + ARM_SURPLUS_MAX {
            let drop = pv_avail - raw_avail;
            for chn in &mut self.channels {
                chn.pending.copy_within(drop.., 0);
                let rest = chn.pending.len() - drop;
                chn.pending.truncate(rest);
            }
        }

        // Emit up to the cap, lockstep across both arms and all channels.
        let pv_avail = self
            .channels
            .iter()
            .map(|c| c.pending.len())
            .min()
            .unwrap_or(0);
        let raw_avail = self.raw_pending.iter().map(Vec::len).min().unwrap_or(0);
        let emit = pv_avail.min(raw_avail).min(max_out);
        for (ch, state) in self.channels.iter_mut().enumerate() {
            self.out[ch].clear();
            self.out[ch].extend_from_slice(&state.pending[..emit]);
            state.pending.copy_within(emit.., 0);
            let rest = state.pending.len() - emit;
            state.pending.truncate(rest);
        }
        // Decode M/S back to L/R before the toggle blend (the raw arm is
        // L/R throughout).
        if stereo && emit > 0 {
            for i in 0..emit {
                let (m, s) = (self.out[0][i], self.out[1][i]);
                self.out[0][i] = m + s;
                self.out[1][i] = m - s;
            }
        }
        // Toggle crossfade, per frame, weights shared across channels.
        let target = self.keylock_target as f32;
        let mut weight = if self.keylock_weight.is_nan() {
            target
        } else {
            self.keylock_weight
        };
        let step = 1.0 / KEYLOCK_TOGGLE_FADE_FRAMES as f32;
        for i in 0..emit {
            weight += (target - weight).clamp(-step, step);
            for ch in 0..self.num_channels {
                let raw = self.raw_pending[ch][i];
                self.out[ch][i] = weight * self.out[ch][i] + (1.0 - weight) * raw;
            }
        }
        if emit > 0 {
            self.keylock_weight = weight;
        }
        for pending in &mut self.raw_pending {
            pending.copy_within(emit.., 0);
            let rest = pending.len() - emit;
            pending.truncate(rest);
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
        self.started = false;
        self.raw.reset();
        self.raw_delay.clear();
        for pending in &mut self.raw_pending {
            pending.clear();
        }
        self.keylock_weight = f32::NAN;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44_100;

    fn goertzel(seg: &[f32], freq: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI * freq / SR as f64;
        let c = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &x in seg {
            let s0 = x as f64 + c * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        ((s1 * s1 + s2 * s2 - c * s1 * s2).max(0.0)).sqrt() / (seg.len() as f64 / 2.0)
    }

    #[test]
    fn two_tone_balance_survives_all_wide_ratios() {
        // The pre-Stage-14 batch path's sub-bass imbalance came from
        // mirror-pad phase seeding; the head's warmup re-seeds from real
        // content, so the 100/1000 Hz balance must sit at the ideal
        // (~0.54) at every wide rate, mid-stream.
        for rate in [2.0f64, 0.8, 1.5, 0.5] {
            let mut head = WidePvHead::new(SR, 1);
            let n = SR as usize * 6;
            let input: Vec<f32> = (0..n)
                .map(|i| {
                    let t = i as f64 / SR as f64;
                    (0.65 * (2.0 * std::f64::consts::PI * 100.0 * t).sin()
                        + 0.35 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin())
                        as f32
                })
                .collect();
            let mut out = Vec::new();
            for chunk in input.chunks(32) {
                let emitted = head.feed_capped(chunk, rate, 386);
                out.extend_from_slice(&head.output(0)[..emitted]);
            }
            let mid = &out[out.len() / 2 - 16_384..out.len() / 2 + 16_384];
            let balance = goertzel(mid, 1_000.0) / goertzel(mid, 100.0);
            assert!(
                (0.4..0.7).contains(&balance),
                "rate {rate}: two-tone balance {balance:.3} off the ideal (~0.54)"
            );
        }
    }

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

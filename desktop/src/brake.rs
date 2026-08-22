//! Post-engine brake resampler for the Wide range's sub-floor fader zone.
//!
//! The wide chain's tempo rate bottoms out at 0.5 (-50%, its PV head's
//! ratio clamp); a CDJ-3000's WIDE fader reaches -100% — a platter stop,
//! not a stretch. While the fader sits below the floor, the engine stays
//! pinned at rate 0.5 and the audio callback reads its output through
//! this variable-rate linear resampler at factor
//! `b = desired_rate / 0.5 ∈ [0, 1]`: combined tempo is `0.5·b` and
//! pitch falls with `b` from the keylocked pitch — the vinyl-brake
//! sound, continuous at the -50% boundary (`b = 1` is passthrough) down
//! to a frozen, gated-silent stop at `b = 0`.
//!
//! Handing off to a raw source-varispeed voice instead would jump pitch
//! an octave at the boundary (keylocked rate 0.5 plays original pitch;
//! a raw 0.5x read does not). Like [`crate::scrub::ScrubVoice`], this is
//! pure math over slices — no I/O — so it stays unit-testable off the
//! audio thread.

const CHANNELS: usize = 2;
/// Engine tempo rate the brake multiplies (the wide chain's floor).
const ENGINE_FLOOR_RATE: f64 = 0.5;
/// One-pole time constant (seconds) easing the brake factor toward the
/// fader target — the vinyl-brake glide (same family as the scrub voice's
/// `SETTLE_TAU_SECS`).
const BRAKE_TAU_SECS: f64 = 0.10;
/// Below this audible source rate (`0.25·b`, source frames per output
/// frame) the output fades out — a near-stopped platter is silent
/// (matches `scrub::AUDIBLE_RATE`).
const AUDIBLE_RATE: f64 = 0.02;
/// Gate fade time constant (seconds), matching the scrub voice.
const GATE_SMOOTH_SECS: f64 = 0.005;

/// Variable-rate reader over the engine's output stream.
///
/// Engine frames are pulled on demand into a small FIFO and consumed at
/// the smoothed brake rate; at `b = 0` nothing is pulled, so the engine —
/// and therefore the published playhead — freezes in place.
pub struct BrakeResampler {
    /// Interleaved stereo engine frames awaiting consumption.
    fifo: Vec<f32>,
    /// Fractional read position into `fifo`, in frames.
    frac: f64,
    /// Smoothed brake factor `b`, 0..1.
    rate: f64,
    /// Smoothed audibility gate, 0..1.
    gate: f64,
    rate_alpha: f64,
    gate_alpha: f64,
    engaged: bool,
}

impl BrakeResampler {
    pub fn new(sample_rate: u32) -> Self {
        let sr = sample_rate.max(1) as f64;
        Self {
            fifo: Vec::with_capacity(8192),
            frac: 0.0,
            rate: 1.0,
            gate: 1.0,
            rate_alpha: 1.0 - (-1.0 / (BRAKE_TAU_SECS * sr)).exp(),
            gate_alpha: 1.0 - (-1.0 / (GATE_SMOOTH_SECS * sr)).exp(),
            engaged: false,
        }
    }

    /// Whether the resampler currently owns the engine read. While `true`
    /// the caller must route every block through [`render`](Self::render)
    /// even at `b = 1`, so the FIFO drains before the direct path resumes.
    pub fn engaged(&self) -> bool {
        self.engaged
    }

    /// Drop buffered pre-seek frames (rate/gate keep smoothing). Called on
    /// a warm-start seek so the resampler doesn't replay stale audio.
    pub fn reset(&mut self) {
        self.fifo.clear();
        self.frac = 0.0;
    }

    /// Render one block into `out` (interleaved stereo, overwritten),
    /// easing toward brake factor `target_b` and pulling exactly the
    /// engine frames the block consumes via `pull` (called with an
    /// interleaved buffer to fill, at most once per block, never longer
    /// than `out`).
    pub fn render(&mut self, target_b: f64, out: &mut [f32], mut pull: impl FnMut(&mut [f32])) {
        let target = if target_b.is_finite() {
            target_b.clamp(0.0, 1.0)
        } else {
            1.0
        };
        if !self.engaged {
            // Engage on the first sub-unity block: empty FIFO, rate and
            // gate at unity — the first pulled frame is the frame the
            // direct path would have played, so the handoff is seamless.
            self.engaged = true;
            self.rate = 1.0;
            self.gate = 1.0;
            self.frac = 0.0;
            self.fifo.clear();
        }

        // Rail snaps: the one-pole never quite reaches its target (see the
        // scrub_mix note in audio_engine.rs) — without these the brake
        // neither fully stops nor ever hands back to the direct path.
        if target >= 1.0 && self.rate > 0.999 {
            self.rate = 1.0;
        } else if target <= 0.0 && self.rate < 1e-3 {
            self.rate = 0.0;
        }

        // Frozen: consume nothing (playhead pins), keep the FIFO so
        // raising the fader resumes from the exact frame.
        if self.rate == 0.0 && target <= 0.0 {
            out.fill(0.0);
            return;
        }

        let out_frames = out.len() / CHANNELS;

        // Pre-simulate the block's rate ramp with the exact render-loop
        // arithmetic (the ScrubVoice::begin_settle trick) to count the
        // engine frames the block needs, and pull them in one call.
        let (mut sim_rate, mut sim_pos) = (self.rate, self.frac);
        for _ in 0..out_frames {
            sim_rate += (target - sim_rate) * self.rate_alpha;
            sim_pos += sim_rate;
        }
        // +2: the interpolator reads frames floor(pos) and floor(pos)+1.
        let needed = (sim_pos.floor() as usize + 2) * CHANNELS;
        if needed > self.fifo.len() {
            let start = self.fifo.len();
            self.fifo.resize(needed, 0.0);
            pull(&mut self.fifo[start..]);
        }

        for frame in out.chunks_exact_mut(CHANNELS) {
            self.rate += (target - self.rate) * self.rate_alpha;
            let gate_target = (ENGINE_FLOOR_RATE * self.rate / AUDIBLE_RATE).min(1.0);
            self.gate += (gate_target - self.gate) * self.gate_alpha;

            let base = self.frac.floor() as usize;
            let interp = (self.frac - base as f64) as f32;
            let gain = self.gate as f32;
            for (ch, sample) in frame.iter_mut().enumerate() {
                let a = self.fifo[base * CHANNELS + ch];
                let b = self.fifo[(base + 1) * CHANNELS + ch];
                *sample = (a + (b - a) * interp) * gain;
            }

            self.frac += self.rate;
        }

        // Retire consumed whole frames.
        let whole = self.frac.floor() as usize;
        self.fifo.drain(..whole * CHANNELS);
        self.frac -= whole as f64;

        // Hand back to the direct path once the rate has rail-snapped to
        // unity. The residual FIFO holds at most a frame or two of
        // sub-sample offset; dropping it skips < 2 samples — inaudible.
        if target >= 1.0 && self.rate == 1.0 {
            self.engaged = false;
            self.fifo.clear();
            self.frac = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;
    const BLOCK: usize = 512;

    /// Pull source: an endless frame-index ramp on the left channel, its
    /// negation on the right — interpolation errors show up as asymmetry,
    /// skipped/replayed engine frames as ramp discontinuities.
    struct RampEngine {
        next_frame: usize,
        pulls: usize,
    }

    impl RampEngine {
        fn new() -> Self {
            Self {
                next_frame: 0,
                pulls: 0,
            }
        }

        fn pull(&mut self, buf: &mut [f32]) {
            self.pulls += 1;
            for frame in buf.chunks_exact_mut(CHANNELS) {
                frame[0] = self.next_frame as f32;
                frame[1] = -(self.next_frame as f32);
                self.next_frame += 1;
            }
        }
    }

    fn render_block(br: &mut BrakeResampler, engine: &mut RampEngine, target: f64) -> Vec<f32> {
        let mut out = vec![0.0f32; BLOCK * CHANNELS];
        br.render(target, &mut out, |buf| engine.pull(buf));
        out
    }

    #[test]
    fn unity_is_passthrough_and_disengages() {
        let mut br = BrakeResampler::new(SR);
        let mut engine = RampEngine::new();
        let out = render_block(&mut br, &mut engine, 1.0);
        for (i, frame) in out.chunks_exact(CHANNELS).enumerate() {
            assert_eq!(frame[0], i as f32);
            assert_eq!(frame[1], -(i as f32));
        }
        assert!(!br.engaged(), "unity target must hand back immediately");
    }

    #[test]
    fn half_rate_halves_the_ramp_slope() {
        let mut br = BrakeResampler::new(SR);
        let mut engine = RampEngine::new();
        // Let the one-pole converge (~5 tau of blocks).
        for _ in 0..100 {
            render_block(&mut br, &mut engine, 0.5);
        }
        let out = render_block(&mut br, &mut engine, 0.5);
        for pair in out.chunks_exact(CHANNELS).collect::<Vec<_>>().windows(2) {
            let slope = pair[1][0] - pair[0][0];
            // The ramp has climbed to ~50k frames, where an f32 ulp is
            // ~0.004 — allow a couple of ulps of interpolation quantization.
            assert!((slope - 0.5).abs() < 0.01, "slope {slope} != 0.5");
        }
        assert!(br.engaged());
    }

    #[test]
    fn full_stop_pulls_nothing_and_is_silent() {
        let mut br = BrakeResampler::new(SR);
        let mut engine = RampEngine::new();
        for _ in 0..200 {
            render_block(&mut br, &mut engine, 0.0);
        }
        let pulls_at_stop = engine.pulls;
        let out = render_block(&mut br, &mut engine, 0.0);
        assert_eq!(engine.pulls, pulls_at_stop, "frozen brake must not pull");
        assert!(out.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn freeze_then_resume_continues_from_the_same_frame() {
        let mut br = BrakeResampler::new(SR);
        let mut engine = RampEngine::new();
        for _ in 0..200 {
            render_block(&mut br, &mut engine, 0.0);
        }
        let frozen_at = engine.next_frame;
        // The retained FIFO must cover the first resumed reads before any
        // new pull — no gap, no replay.
        let mut resumed = Vec::new();
        for _ in 0..50 {
            resumed.extend(render_block(&mut br, &mut engine, 0.5));
        }
        let first_audible = resumed
            .chunks_exact(CHANNELS)
            .find(|f| f[0] != 0.0)
            .expect("resume must become audible")[0];
        assert!(
            (first_audible as usize) < frozen_at,
            "resume read frame {first_audible} but the freeze held frame {frozen_at}"
        );
    }

    #[test]
    fn releasing_the_brake_disengages_via_rail_snap() {
        let mut br = BrakeResampler::new(SR);
        let mut engine = RampEngine::new();
        for _ in 0..50 {
            render_block(&mut br, &mut engine, 0.4);
        }
        assert!(br.engaged());
        let mut blocks = 0;
        while br.engaged() && blocks < 500 {
            render_block(&mut br, &mut engine, 1.0);
            blocks += 1;
        }
        assert!(!br.engaged(), "brake never handed back to the direct path");
    }
}

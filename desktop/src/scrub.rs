//! Varispeed scrub voice for CDJ-style audible waveform dragging.
//!
//! While the zoomed waveform is dragged, the audio callback bypasses the
//! engine and renders this voice instead: a linear-interpolated read of the
//! raw decoded source that chases the pointer-implied position with a
//! smoothed rate, so pitch follows hand speed in either direction (the
//! engine itself is forward-only and tempo-clamped, so it can't make this
//! sound). The voice is pure math over slices — no I/O — so it stays
//! unit-testable off the audio thread.

const CHANNELS: usize = 2;
/// Time (seconds) over which the reader converges on the published target.
const CATCHUP_SECS: f64 = 0.060;
/// One-pole time constant (seconds) smoothing the chase rate; jerky pointer
/// deltas sound like tape, not a zipper.
const RATE_SMOOTH_SECS: f64 = 0.010;
/// Fastest scrub speed in source frames per output frame.
const MAX_RATE: f64 = 32.0;
/// Below this |rate| the voice fades out: a near-stationary read is just a
/// held DC value, and a CDJ in vinyl-hold is silent.
const AUDIBLE_RATE: f64 = 0.02;
/// Gate fade time constant (seconds) as the tape slows to a stop.
const GATE_SMOOTH_SECS: f64 = 0.005;
/// Release-glide time constant (seconds): after the drag drops, the rate
/// eases exponentially toward the settle target (1.0 while playing, 0.0
/// while paused) — the CDJ vinyl release. ~3x this reaches 95% of the way.
const SETTLE_TAU_SECS: f64 = 0.15;
/// The glide is over once the rate is within this of its target.
const SETTLE_EPS_RATE: f64 = 0.02;
/// Trajectory length cap in frames (safety; ~1.1 s suffices from ±32x).
const MAX_SETTLE_FRAMES: u64 = 48_000 * 5;

/// Release-glide trajectory: the rate eases toward `rate_target` for
/// exactly `frames_left` more frames (pre-counted at release so the landing
/// position is known in advance).
struct SettleTraj {
    rate_target: f64,
    frames_left: u64,
}

/// Variable-rate scrub reader over an interleaved stereo source.
pub struct ScrubVoice {
    /// Read position in source frames.
    pos: f64,
    /// Smoothed advance in source frames per output frame (sign = direction).
    rate: f64,
    /// Smoothed audibility gate, 0..1.
    gate: f64,
    catchup_frames: f64,
    rate_alpha: f64,
    gate_alpha: f64,
    settle_alpha: f64,
    settle: Option<SettleTraj>,
}

impl ScrubVoice {
    pub fn new(sample_rate: u32) -> Self {
        let sr = sample_rate.max(1) as f64;
        Self {
            pos: 0.0,
            rate: 0.0,
            gate: 0.0,
            catchup_frames: (CATCHUP_SECS * sr).max(1.0),
            rate_alpha: 1.0 - (-1.0 / (RATE_SMOOTH_SECS * sr)).exp(),
            gate_alpha: 1.0 - (-1.0 / (GATE_SMOOTH_SECS * sr)).exp(),
            settle_alpha: 1.0 - (-1.0 / (SETTLE_TAU_SECS * sr)).exp(),
            settle: None,
        }
    }

    /// Re-anchor the reader at `frame` on scrub engage: no residual motion
    /// or gate from a previous gesture.
    pub fn seed(&mut self, frame: f64) {
        self.pos = frame;
        self.rate = 0.0;
        self.gate = 0.0;
        self.settle = None;
    }

    /// Current read position in source frames.
    pub fn position(&self) -> f64 {
        self.pos
    }

    /// Start the release glide easing the current rate toward `rate_target`
    /// and return the predicted landing frame. The trajectory is simulated
    /// once with the same per-frame arithmetic [`render_settle`] runs, so
    /// the voice lands exactly on the returned frame — the engine can be
    /// warm-started there in parallel for a seamless handoff.
    pub fn begin_settle(&mut self, rate_target: f64, source: &[f32]) -> f64 {
        let total_frames = source.len() / CHANNELS;
        let max_pos = (total_frames.saturating_sub(1)) as f64;
        let mut rate = self.rate;
        let mut pos = self.pos.clamp(0.0, max_pos);
        let mut n: u64 = 0;
        while (rate - rate_target).abs() >= SETTLE_EPS_RATE && n < MAX_SETTLE_FRAMES {
            rate += (rate_target - rate) * self.settle_alpha;
            pos = (pos + rate).clamp(0.0, max_pos);
            n += 1;
            // A boundary ends the glide early — but only when still moving
            // into it, so a clamped start can ease back off the rail.
            if (pos == 0.0 && rate < 0.0) || (pos == max_pos && rate > 0.0) {
                break;
            }
        }
        self.settle = Some(SettleTraj {
            rate_target,
            frames_left: n,
        });
        pos
    }

    /// Render one glide block into `out` (interleaved stereo, overwritten).
    /// Returns `true` once the trajectory is complete. Past the landing the
    /// voice keeps playing at the settle rate — the caller's mix ramp fades
    /// it against the engine (time-aligned at rate 1.0), so there is no cut.
    pub fn render_settle(&mut self, source: &[f32], out: &mut [f32]) -> bool {
        let total_frames = source.len() / CHANNELS;
        let Some(SettleTraj {
            rate_target,
            ref mut frames_left,
        }) = self.settle
        else {
            out.fill(0.0);
            return true;
        };
        if total_frames == 0 {
            out.fill(0.0);
            return true;
        }
        let max_pos = (total_frames - 1) as f64;

        for frame in out.chunks_exact_mut(CHANNELS) {
            // Same op order as the begin_settle simulation: rate, then
            // read at the pre-advance position, then advance + clamp.
            self.rate += (rate_target - self.rate) * self.settle_alpha;
            let gate_target = (self.rate.abs() / AUDIBLE_RATE).min(1.0);
            self.gate += (gate_target - self.gate) * self.gate_alpha;

            let base = self.pos.floor() as usize;
            let frac = (self.pos - base as f64) as f32;
            let next = (base + 1).min(total_frames - 1);
            let gain = self.gate as f32;
            for (ch, sample) in frame.iter_mut().enumerate() {
                let a = source[base * CHANNELS + ch];
                let b = source[next * CHANNELS + ch];
                *sample = (a + (b - a) * frac) * gain;
            }

            self.pos = (self.pos + self.rate).clamp(0.0, max_pos);
            *frames_left = frames_left.saturating_sub(1);
        }

        self.settle.as_ref().is_none_or(|s| s.frames_left == 0)
    }

    /// Render one block into `out` (interleaved stereo, overwritten),
    /// chasing `target_frame` through `source`.
    pub fn render(&mut self, target_frame: f64, source: &[f32], out: &mut [f32]) {
        let total_frames = source.len() / CHANNELS;
        if total_frames == 0 {
            out.fill(0.0);
            return;
        }
        let max_pos = (total_frames - 1) as f64;
        let target = target_frame.clamp(0.0, max_pos);
        let target_rate = ((target - self.pos) / self.catchup_frames).clamp(-MAX_RATE, MAX_RATE);

        for frame in out.chunks_exact_mut(CHANNELS) {
            self.rate += (target_rate - self.rate) * self.rate_alpha;
            let gate_target = (self.rate.abs() / AUDIBLE_RATE).min(1.0);
            self.gate += (gate_target - self.gate) * self.gate_alpha;

            let base = self.pos.floor() as usize;
            let frac = (self.pos - base as f64) as f32;
            let next = (base + 1).min(total_frames - 1);
            let gain = self.gate as f32;
            for (ch, sample) in frame.iter_mut().enumerate() {
                let a = source[base * CHANNELS + ch];
                let b = source[next * CHANNELS + ch];
                *sample = (a + (b - a) * frac) * gain;
            }

            self.pos = (self.pos + self.rate).clamp(0.0, max_pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    /// Interleaved stereo source whose left channel is a frame-index ramp
    /// and right channel its negation — interpolation errors show up as
    /// asymmetry.
    fn ramp_source(frames: usize) -> Vec<f32> {
        let mut s = Vec::with_capacity(frames * CHANNELS);
        for i in 0..frames {
            s.push(i as f32);
            s.push(-(i as f32));
        }
        s
    }

    fn render_blocks(voice: &mut ScrubVoice, target: f64, source: &[f32], blocks: usize) {
        let mut out = vec![0.0f32; 512 * CHANNELS];
        for _ in 0..blocks {
            voice.render(target, source, &mut out);
        }
    }

    #[test]
    fn converges_on_forward_target() {
        let source = ramp_source(SR as usize * 10);
        let mut voice = ScrubVoice::new(SR);
        voice.seed(1_000.0);
        // One second of audio: far longer than catchup + smoothing.
        render_blocks(&mut voice, 50_000.0, &source, SR as usize / 512);
        assert!(
            (voice.pos - 50_000.0).abs() < 1.0,
            "pos {} should have converged on 50000",
            voice.pos
        );
    }

    #[test]
    fn converges_on_reverse_target() {
        let source = ramp_source(SR as usize * 10);
        let mut voice = ScrubVoice::new(SR);
        voice.seed(200_000.0);
        render_blocks(&mut voice, 150_000.0, &source, SR as usize / 512);
        assert!(
            (voice.pos - 150_000.0).abs() < 1.0,
            "pos {} should have converged on 150000",
            voice.pos
        );
    }

    #[test]
    fn rate_is_clamped() {
        let frames = SR as usize * 100;
        let source = ramp_source(frames);
        let mut voice = ScrubVoice::new(SR);
        voice.seed(0.0);
        // Target far beyond what MAX_RATE covers in one block.
        let mut out = vec![0.0f32; 512 * CHANNELS];
        voice.render((frames - 1) as f64, &source, &mut out);
        assert!(voice.rate.abs() <= MAX_RATE + 1e-9);
        assert!(voice.pos <= 512.0 * MAX_RATE);
    }

    #[test]
    fn position_clamps_at_boundaries() {
        let source = ramp_source(1_000);
        let mut voice = ScrubVoice::new(SR);
        voice.seed(500.0);
        render_blocks(&mut voice, -10_000.0, &source, 200);
        assert!(voice.pos >= 0.0 && voice.pos < 1e-6, "pos {}", voice.pos);
        render_blocks(&mut voice, 10_000.0, &source, 200);
        assert!(
            voice.pos <= 999.0 && voice.pos > 999.0 - 1e-6,
            "pos {}",
            voice.pos
        );
    }

    #[test]
    fn interpolates_ramp_exactly_once_gate_opens() {
        let source = ramp_source(SR as usize * 10);
        let mut voice = ScrubVoice::new(SR);
        voice.seed(10_000.0);
        // Settle into a steady forward chase so the gate is fully open.
        render_blocks(&mut voice, 100_000.0, &source, 20);
        let mut out = vec![0.0f32; 256 * CHANNELS];
        let pos_before = voice.pos;
        voice.render(100_000.0, &source, &mut out);
        // On a linear ramp, an interpolated read at p returns exactly p.
        let expected = (pos_before + voice.rate) as f32;
        assert!(
            (out[CHANNELS] - expected).abs() < 2.0,
            "left {} vs expected ~{expected}",
            out[CHANNELS]
        );
        for frame in out.chunks_exact(CHANNELS) {
            assert!(
                (frame[0] + frame[1]).abs() < 1e-3,
                "channels should mirror: {} vs {}",
                frame[0],
                frame[1]
            );
        }
    }

    #[test]
    fn stationary_target_fades_to_silence() {
        let source = vec![1.0f32; 10_000 * CHANNELS]; // constant full-scale
        let mut voice = ScrubVoice::new(SR);
        voice.seed(5_000.0);
        // Hold the target at the current position for half a second.
        render_blocks(&mut voice, 5_000.0, &source, SR as usize / 2 / 512);
        let mut out = vec![1.0f32; 256 * CHANNELS];
        voice.render(5_000.0, &source, &mut out);
        for &s in &out {
            assert!(s.abs() < 1e-3, "expected silence, got {s}");
        }
    }

    #[test]
    fn empty_source_outputs_silence() {
        let mut voice = ScrubVoice::new(SR);
        let mut out = vec![1.0f32; 64 * CHANNELS];
        voice.render(100.0, &[], &mut out);
        assert!(out.iter().all(|&s| s == 0.0));
    }

    /// Drive the voice into a steady chase toward a far target so it still
    /// carries full momentum (a near target would converge and decay the
    /// rate to ~0 before release).
    fn voice_with_momentum(source: &[f32], start: f64, target: f64) -> ScrubVoice {
        let mut voice = ScrubVoice::new(SR);
        voice.seed(start);
        render_blocks(&mut voice, target, source, 30);
        voice
    }

    /// Render settle blocks until done; returns frames rendered until the
    /// completion block (inclusive).
    fn settle_to_done(voice: &mut ScrubVoice, source: &[f32]) -> usize {
        let mut out = vec![0.0f32; 512 * CHANNELS];
        let mut frames = 0;
        for _ in 0..2000 {
            let done = voice.render_settle(source, &mut out);
            frames += 512;
            if done {
                return frames;
            }
        }
        panic!("settle never completed");
    }

    #[test]
    fn settle_lands_exactly_on_prediction() {
        let source = ramp_source(SR as usize * 30);
        for rate_target in [1.0, 0.0] {
            let mut voice = voice_with_momentum(&source, 100_000.0, 1_400_000.0);
            let rate_before = voice.rate;
            assert!(rate_before > 1.5, "need real momentum, got {rate_before}");
            let landing = voice.begin_settle(rate_target, &source);
            let traj_frames = voice.settle.as_ref().unwrap().frames_left;
            // Render exactly the trajectory length in odd-sized blocks to
            // cross block boundaries.
            let mut remaining = traj_frames as usize;
            let mut out = vec![0.0f32; 173 * CHANNELS];
            while remaining >= 173 {
                voice.render_settle(&source, &mut out);
                remaining -= 173;
            }
            let mut tail = vec![0.0f32; remaining * CHANNELS];
            if remaining > 0 {
                voice.render_settle(&source, &mut tail);
            }
            assert_eq!(
                voice.pos, landing,
                "voice must land bit-exactly on the predicted frame (rt {rate_target})"
            );
            assert!(
                (voice.rate - rate_target).abs() < SETTLE_EPS_RATE + 1e-9,
                "rate {} should have eased to {rate_target}",
                voice.rate
            );
        }
    }

    #[test]
    fn settle_toward_play_overshoots_drop_point() {
        let source = ramp_source(SR as usize * 30);
        let mut voice = voice_with_momentum(&source, 100_000.0, 1_400_000.0);
        let drop_pos = voice.pos;
        let landing = voice.begin_settle(1.0, &source);
        // Fast forward momentum must carry the landing well past the drop.
        assert!(
            landing > drop_pos + SR as f64 * 0.05,
            "landing {landing} should overshoot drop {drop_pos}"
        );
        settle_to_done(&mut voice, &source);
        assert!((voice.rate - 1.0).abs() < SETTLE_EPS_RATE + 1e-9);
    }

    #[test]
    fn spin_up_from_hold_reaches_play_rate() {
        let source = ramp_source(SR as usize * 10);
        let mut voice = ScrubVoice::new(SR);
        voice.seed(200_000.0); // rate 0, as after a stationary hold
        let landing = voice.begin_settle(1.0, &source);
        assert!(landing > 200_000.0, "spin-up still travels forward");
        settle_to_done(&mut voice, &source);
        assert!((voice.rate - 1.0).abs() < SETTLE_EPS_RATE + 1e-9);
        // Monotonic rise: never overshoots 1.0 from below.
        assert!(voice.rate <= 1.0 + 1e-9);
    }

    #[test]
    fn settle_clamps_at_track_end() {
        let total = SR as usize; // 1s track
        let source = ramp_source(total);
        let mut voice = voice_with_momentum(&source, 20_000.0, 2_000_000.0);
        let landing = voice.begin_settle(1.0, &source);
        assert_eq!(
            landing,
            (total - 1) as f64,
            "fling into EOF lands on the last frame"
        );
        settle_to_done(&mut voice, &source);
        assert_eq!(voice.pos, landing);
    }

    #[test]
    fn spin_down_fades_to_silence() {
        let source = vec![1.0f32; SR as usize * 20 * CHANNELS];
        let mut voice = voice_with_momentum(&source, 100_000.0, 900_000.0);
        voice.begin_settle(0.0, &source);
        settle_to_done(&mut voice, &source);
        // Past the landing the rate keeps easing to 0 and the gate follows
        // it down (in the app the mix ramp-out cuts the tail much sooner).
        let mut out = vec![1.0f32; 512 * CHANNELS];
        for _ in 0..300 {
            voice.render_settle(&source, &mut out);
        }
        assert!(
            out.iter().all(|&s| s.abs() < 1e-2),
            "spin-down should end silent"
        );
    }
}

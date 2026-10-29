//! ITU-R BS.1770-4 / EBU R128 loudness measurement.
//!
//! Wraps the `ebur128` crate (a pure-Rust port of libebur128): K-weighted,
//! gated integrated loudness, oversampled true peak, and EBU R128
//! loudness range. This is the real meter for track gain decisions; the
//! simplified RMS-based `estimate_lufs` in [`crate::analysis::comparison`]
//! remains a relative A/B utility for the quality benchmarks only.

use crate::core::preanalysis::LoudnessMeasurement;
use ebur128::{EbuR128, Mode};

/// Measures BS.1770-4 loudness on interleaved audio.
///
/// `interleaved` is the original multi-channel signal (not the mono
/// analysis downmix — BS.1770 sums per-channel energies, so a mid downmix
/// reads up to ~3 dB low depending on channel correlation).
///
/// Returns `None` for empty/near-silent input (integrated loudness is
/// undefined below the -70 LUFS absolute gate), zero channels, or an
/// unsupported channel count/sample rate.
pub fn measure_loudness(
    interleaved: &[f32],
    channels: usize,
    sample_rate: u32,
) -> Option<LoudnessMeasurement> {
    if interleaved.is_empty() || channels == 0 || sample_rate == 0 {
        return None;
    }

    let mut meter = EbuR128::new(
        channels as u32,
        sample_rate,
        Mode::I | Mode::LRA | Mode::TRUE_PEAK,
    )
    .ok()?;
    // Drop any trailing partial frame rather than failing the measurement.
    let whole_frames = interleaved.len() / channels * channels;
    meter.add_frames_f32(&interleaved[..whole_frames]).ok()?;

    let integrated_lufs = meter.loudness_global().ok()?;
    if !integrated_lufs.is_finite() {
        return None; // Below the absolute gate: silence or near-silence.
    }
    let loudness_range_lu = meter.loudness_range().ok()?;

    let mut true_peak_linear = 0f64;
    for ch in 0..channels as u32 {
        true_peak_linear = true_peak_linear.max(meter.true_peak(ch).ok()?);
    }
    let true_peak_dbtp = if true_peak_linear > 0.0 {
        20.0 * true_peak_linear.log10()
    } else {
        f64::NEG_INFINITY
    };

    Some(LoudnessMeasurement {
        integrated_lufs,
        true_peak_dbtp,
        loudness_range_lu,
    })
}

/// Frames buffered per channel before handing off to the inner meter.
const CHUNK_FRAMES: usize = 1024;

/// Real-time-safe streaming BS.1770-4 momentary (400 ms) loudness meter.
///
/// Construction allocates the K-weighting filter state, the 400 ms energy
/// window, and a fixed-capacity chunk buffer; after that, `push_stereo`,
/// `process`, `momentary_lufs`, and `reset` never allocate, lock, or
/// panic, so they are safe inside an audio callback. Feed samples as they
/// are rendered and read `momentary_lufs` once per block.
pub struct MomentaryLoudness {
    meter: EbuR128,
    /// Interleaved samples awaiting handoff; capacity is fixed at
    /// `channels * CHUNK_FRAMES` and always holds whole frames.
    pending: Vec<f32>,
    channels: usize,
}

impl MomentaryLoudness {
    /// Finite floor reported for silence or insufficient data (the raw
    /// meter reports -inf, which is hostile to downstream atomic/UI math).
    pub const SILENCE_LUFS: f32 = -100.0;

    /// Allocates a meter for the given stream format. Returns `None` for
    /// zero channels or a rate/channel count the underlying meter rejects
    /// (rate outside 16..=2 822 400 Hz, more than 64 channels).
    pub fn new(sample_rate: u32, channels: usize) -> Option<Self> {
        if channels == 0 {
            return None;
        }
        let meter = EbuR128::new(channels as u32, sample_rate, Mode::M).ok()?;
        Some(Self {
            meter,
            pending: Vec::with_capacity(channels * CHUNK_FRAMES),
            channels,
        })
    }

    /// Hand buffered frames to the inner meter. `pending` always holds
    /// whole frames, so `add_frames_f32` cannot fail; the ring write and
    /// biquad cascade behind it are allocation-free.
    fn flush(&mut self) {
        if !self.pending.is_empty() {
            let _ = self.meter.add_frames_f32(&self.pending);
            self.pending.clear();
        }
    }

    /// Feed one stereo frame. The meter must have been built with
    /// `channels == 2`.
    pub fn push_stereo(&mut self, left: f32, right: f32) {
        debug_assert_eq!(self.channels, 2);
        if self.pending.len() + 2 > self.pending.capacity() {
            self.flush();
        }
        self.pending.push(left);
        self.pending.push(right);
    }

    /// Feed a block of interleaved samples; a trailing partial frame is
    /// dropped.
    pub fn process(&mut self, interleaved: &[f32]) {
        let whole_frames = interleaved.len() / self.channels * self.channels;
        for frame in interleaved[..whole_frames].chunks_exact(self.channels) {
            if self.pending.len() + self.channels > self.pending.capacity() {
                self.flush();
            }
            self.pending.extend_from_slice(frame);
        }
    }

    /// Momentary loudness over the trailing 400 ms in LUFS, clamped to
    /// [`Self::SILENCE_LUFS`]. Flushes buffered frames first (hence
    /// `&mut`).
    pub fn momentary_lufs(&mut self) -> f32 {
        self.flush();
        match self.meter.loudness_momentary() {
            Ok(lufs) if lufs.is_finite() => (lufs as f32).max(Self::SILENCE_LUFS),
            _ => Self::SILENCE_LUFS,
        }
    }

    /// Clear the filter state and the 400 ms window (track change / seek).
    pub fn reset(&mut self) {
        self.pending.clear();
        self.meter.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: u32 = 44_100;

    /// Interleaved stereo sine, both channels identical.
    fn stereo_sine(freq: f32, amplitude: f32, secs: f32) -> Vec<f32> {
        let frames = (secs * SAMPLE_RATE as f32) as usize;
        let mut out = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let s =
                amplitude * (std::f32::consts::TAU * freq * i as f32 / SAMPLE_RATE as f32).sin();
            out.push(s);
            out.push(s);
        }
        out
    }

    #[test]
    fn test_full_scale_stereo_sine_reference_level() {
        // BS.1770 reference point: a full-scale 997 Hz sine on ONE channel
        // reads -3.01 LKFS; driving both stereo channels sums the two
        // channel energies (+3.01 dB), so the dual-channel sine reads
        // ~0.0 LUFS. True peak of a full-scale sine is 0 dBTP.
        let m = measure_loudness(&stereo_sine(997.0, 1.0, 5.0), 2, SAMPLE_RATE).unwrap();
        assert!(
            m.integrated_lufs.abs() < 0.5,
            "integrated {}",
            m.integrated_lufs
        );
        assert!(
            m.true_peak_dbtp.abs() < 0.3,
            "true peak {}",
            m.true_peak_dbtp
        );
        // A steady sine has essentially no loudness range.
        assert!(m.loudness_range_lu < 1.0, "LRA {}", m.loudness_range_lu);
    }

    #[test]
    fn test_level_tracks_amplitude() {
        // -20 dB of amplitude must read ~20 LU lower.
        let loud = measure_loudness(&stereo_sine(997.0, 1.0, 5.0), 2, SAMPLE_RATE).unwrap();
        let quiet = measure_loudness(&stereo_sine(997.0, 0.1, 5.0), 2, SAMPLE_RATE).unwrap();
        let drop = loud.integrated_lufs - quiet.integrated_lufs;
        assert!((drop - 20.0).abs() < 0.5, "drop {}", drop);
    }

    #[test]
    fn test_silence_and_degenerate_inputs_return_none() {
        assert!(measure_loudness(&vec![0.0; SAMPLE_RATE as usize * 4], 2, SAMPLE_RATE).is_none());
        assert!(measure_loudness(&[], 2, SAMPLE_RATE).is_none());
        assert!(measure_loudness(&[0.5; 1024], 0, SAMPLE_RATE).is_none());
        assert!(measure_loudness(&[0.5; 1024], 2, 0).is_none());
    }

    #[test]
    fn test_gain_helpers() {
        let m = LoudnessMeasurement {
            integrated_lufs: -10.0,
            true_peak_dbtp: -0.5,
            loudness_range_lu: 3.0,
        };
        assert!((m.gain_db_to(-14.0) - (-4.0)).abs() < 1e-12);
        assert!((m.gain_linear_to(-14.0) - 10f64.powf(-0.2)).abs() < 1e-12);
        assert!((m.gain_db_to(-10.0)).abs() < 1e-12);
    }

    /// Stereo sine at an arbitrary sample rate (the module-level helper is
    /// pinned to `SAMPLE_RATE`).
    fn stereo_sine_at(rate: u32, freq: f32, amplitude: f32, secs: f32) -> Vec<f32> {
        let frames = (secs * rate as f32) as usize;
        let mut out = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let s = amplitude * (std::f32::consts::TAU * freq * i as f32 / rate as f32).sin();
            out.push(s);
            out.push(s);
        }
        out
    }

    #[test]
    fn test_momentary_reference_level_and_amplitude_tracking() {
        // Dual-channel full-scale 997 Hz sine reads ~0.0 LUFS (see the
        // integrated test above); -20 dB of amplitude reads ~20 LU lower.
        let mut meter = MomentaryLoudness::new(SAMPLE_RATE, 2).unwrap();
        meter.process(&stereo_sine(997.0, 1.0, 1.0));
        let loud = meter.momentary_lufs();
        assert!(loud.abs() < 0.5, "momentary {loud}");

        meter.reset();
        meter.process(&stereo_sine(997.0, 0.1, 1.0));
        let quiet = meter.momentary_lufs();
        assert!((loud - quiet - 20.0).abs() < 0.5, "drop {}", loud - quiet);
    }

    #[test]
    fn test_momentary_streaming_matches_oneshot_oracle() {
        // Frame-by-frame push through the chunk buffer must be transparent:
        // identical to feeding the whole signal to a raw Mode::M meter.
        let signal = stereo_sine(440.0, 0.7, 1.0);
        let mut streaming = MomentaryLoudness::new(SAMPLE_RATE, 2).unwrap();
        for frame in signal.chunks_exact(2) {
            streaming.push_stereo(frame[0], frame[1]);
        }

        let mut oracle = EbuR128::new(2, SAMPLE_RATE, Mode::M).unwrap();
        oracle.add_frames_f32(&signal).unwrap();

        let got = streaming.momentary_lufs() as f64;
        let want = oracle.loudness_momentary().unwrap();
        assert!(
            (got - want).abs() < 0.01,
            "streaming {got} vs oracle {want}"
        );
    }

    #[test]
    fn test_momentary_silence_floor_and_reset() {
        let mut meter = MomentaryLoudness::new(SAMPLE_RATE, 2).unwrap();
        // Fresh meter (no data at all) and pure silence both sit at the floor.
        assert_eq!(meter.momentary_lufs(), MomentaryLoudness::SILENCE_LUFS);
        meter.process(&vec![0.0; SAMPLE_RATE as usize * 2]);
        assert_eq!(meter.momentary_lufs(), MomentaryLoudness::SILENCE_LUFS);

        // A loud signal lifts it; reset drops it straight back to the floor.
        meter.process(&stereo_sine(997.0, 1.0, 1.0));
        assert!(meter.momentary_lufs() > -1.0);
        meter.reset();
        assert_eq!(meter.momentary_lufs(), MomentaryLoudness::SILENCE_LUFS);
    }

    #[test]
    fn test_momentary_sample_rate_independence() {
        let mut a = MomentaryLoudness::new(44_100, 2).unwrap();
        a.process(&stereo_sine_at(44_100, 997.0, 0.5, 1.0));
        let mut b = MomentaryLoudness::new(48_000, 2).unwrap();
        b.process(&stereo_sine_at(48_000, 997.0, 0.5, 1.0));
        let (la, lb) = (a.momentary_lufs(), b.momentary_lufs());
        assert!((la - lb).abs() < 0.2, "44.1k {la} vs 48k {lb}");
    }

    #[test]
    fn test_momentary_degenerate_construction() {
        assert!(MomentaryLoudness::new(SAMPLE_RATE, 0).is_none());
        assert!(MomentaryLoudness::new(0, 2).is_none());
        assert!(MomentaryLoudness::new(SAMPLE_RATE, 65).is_none());
    }
}

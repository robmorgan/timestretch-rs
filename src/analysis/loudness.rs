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
}

//! Multi-resolution phase vocoder using a three-band filterbank.
//!
//! Splits audio into sub-bass (0-200 Hz), mid (200-4000 Hz), and high (4000+ Hz)
//! frequency bands using Linkwitz-Riley crossover filters, then processes each
//! band with a phase vocoder using an FFT size optimized for that frequency range:
//!
//! - **Sub-bass**: Large FFT (default 16384) for precise frequency resolution
//! - **Mid**: Medium FFT (default 4096) for balanced time-frequency trade-off
//! - **High**: Small FFT (default 1024) for sharp temporal resolution
//!
//! The stretched bands are summed to produce the final output.

use crate::core::crossover::ThreeBandSplitter;
use crate::error::StretchError;
use crate::stretch::phase_vocoder::PhaseVocoder;

/// Default crossover frequency between sub-bass and mid bands (Hz).
const DEFAULT_LOW_CROSSOVER: f64 = 200.0;
/// Default crossover frequency between mid and high bands (Hz).
const DEFAULT_HIGH_CROSSOVER: f64 = 4000.0;

/// Multiplier for sub-bass FFT size relative to mid FFT size.
const SUB_BASS_FFT_MULTIPLIER: usize = 4;
/// Divisor for high-band FFT size relative to mid FFT size.
const HIGH_FFT_DIVISOR: usize = 4;

/// Minimum FFT size for any band (must be a power of 2).
const MIN_FFT_SIZE: usize = 256;

/// Multi-resolution phase vocoder using three frequency bands
/// with different FFT sizes for optimal time-frequency resolution.
///
/// Each band is processed independently by its own [`PhaseVocoder`] instance
/// with an FFT size tuned for the frequency content in that range. The results
/// are summed to produce the final stretched audio.
///
/// # Example
///
/// ```
/// use timestretch::stretch::multi_resolution::MultiResolutionStretcher;
///
/// let mut stretcher = MultiResolutionStretcher::new(
///     4096,       // mid FFT size
///     1.5,        // stretch ratio
///     44100,      // sample rate
///     120.0,      // sub-bass cutoff
/// );
///
/// let input = vec![0.0f32; 44100]; // 1 second of silence
/// let output = stretcher.process(&input).unwrap();
/// assert!(output.len() > input.len()); // ~1.5x longer
/// ```
pub struct MultiResolutionStretcher {
    /// Three-band crossover splitter.
    splitter: ThreeBandSplitter,
    /// Phase vocoder for sub-bass band (large FFT).
    sub_bass_pv: PhaseVocoder,
    /// Phase vocoder for mid band (medium FFT).
    mid_pv: PhaseVocoder,
    /// Phase vocoder for high band (small FFT).
    high_pv: PhaseVocoder,
    /// FFT size for sub-bass band.
    sub_bass_fft_size: usize,
    /// FFT size for mid band.
    mid_fft_size: usize,
    /// FFT size for high band.
    high_fft_size: usize,
    /// Stretch ratio.
    stretch_ratio: f64,
    /// Pre-allocated buffer for sub-bass band input.
    sub_bass_buf: Vec<f32>,
    /// Pre-allocated buffer for mid band input.
    mid_buf: Vec<f32>,
    /// Pre-allocated buffer for high band input.
    high_buf: Vec<f32>,
}

impl MultiResolutionStretcher {
    /// Creates a new multi-resolution stretcher.
    ///
    /// The `mid_fft_size` parameter sets the FFT size for the mid band (200-4000 Hz).
    /// Sub-bass uses `mid_fft_size * 4` and high uses `mid_fft_size / 4`.
    ///
    /// # Arguments
    ///
    /// * `mid_fft_size` - FFT size for the mid band (e.g., 4096)
    /// * `stretch_ratio` - Time-stretch ratio (>1.0 = slower, <1.0 = faster)
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `sub_bass_cutoff` - Sub-bass phase lock cutoff for each PV instance
    pub fn new(
        mid_fft_size: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
    ) -> Self {
        let sub_bass_fft = (mid_fft_size * SUB_BASS_FFT_MULTIPLIER).max(MIN_FFT_SIZE);
        let high_fft = (mid_fft_size / HIGH_FFT_DIVISOR).max(MIN_FFT_SIZE);

        let sub_bass_hop = sub_bass_fft / 4;
        let mid_hop = mid_fft_size / 4;
        let high_hop = high_fft / 4;

        let sub_bass_pv = PhaseVocoder::new(
            sub_bass_fft,
            sub_bass_hop,
            stretch_ratio,
            sample_rate,
            sub_bass_cutoff,
        );
        let mid_pv = PhaseVocoder::new(
            mid_fft_size,
            mid_hop,
            stretch_ratio,
            sample_rate,
            sub_bass_cutoff,
        );
        let high_pv = PhaseVocoder::new(
            high_fft,
            high_hop,
            stretch_ratio,
            sample_rate,
            sub_bass_cutoff,
        );

        Self {
            splitter: ThreeBandSplitter::new(
                DEFAULT_LOW_CROSSOVER,
                DEFAULT_HIGH_CROSSOVER,
                sample_rate,
            ),
            sub_bass_pv,
            mid_pv,
            high_pv,
            sub_bass_fft_size: sub_bass_fft,
            mid_fft_size,
            high_fft_size: high_fft,
            stretch_ratio,
            sub_bass_buf: Vec::new(),
            mid_buf: Vec::new(),
            high_buf: Vec::new(),
        }
    }

    /// Creates a new multi-resolution stretcher with custom crossover frequencies.
    ///
    /// # Arguments
    ///
    /// * `mid_fft_size` - FFT size for the mid band
    /// * `stretch_ratio` - Time-stretch ratio
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `sub_bass_cutoff` - Sub-bass phase lock cutoff for each PV instance
    /// * `low_crossover` - Crossover frequency between sub-bass and mid (Hz)
    /// * `high_crossover` - Crossover frequency between mid and high (Hz)
    #[allow(clippy::too_many_arguments)]
    pub fn with_crossover_freqs(
        mid_fft_size: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
        low_crossover: f64,
        high_crossover: f64,
    ) -> Self {
        let mut s = Self::new(mid_fft_size, stretch_ratio, sample_rate, sub_bass_cutoff);
        s.splitter = ThreeBandSplitter::new(low_crossover, high_crossover, sample_rate);
        s
    }

    /// Updates the stretch ratio for all three bands.
    pub fn set_stretch_ratio(&mut self, ratio: f64) {
        self.stretch_ratio = ratio;
        self.sub_bass_pv.set_stretch_ratio(ratio);
        self.mid_pv.set_stretch_ratio(ratio);
        self.high_pv.set_stretch_ratio(ratio);
    }

    /// Resets the phase state of all three phase vocoders.
    pub fn reset_phase_state(&mut self) {
        self.sub_bass_pv.reset_phase_state();
        self.mid_pv.reset_phase_state();
        self.high_pv.reset_phase_state();
        self.splitter.reset();
    }

    /// Resets phase state for specific frequency bands across all PVs.
    pub fn reset_phase_state_bands(&mut self, reset_mask: [bool; 4], sample_rate: u32) {
        self.sub_bass_pv
            .reset_phase_state_bands(reset_mask, sample_rate);
        self.mid_pv.reset_phase_state_bands(reset_mask, sample_rate);
        self.high_pv
            .reset_phase_state_bands(reset_mask, sample_rate);
    }

    /// Returns the sub-bass FFT size.
    #[inline]
    pub fn sub_bass_fft_size(&self) -> usize {
        self.sub_bass_fft_size
    }

    /// Returns the mid-band FFT size.
    #[inline]
    pub fn mid_fft_size(&self) -> usize {
        self.mid_fft_size
    }

    /// Returns the high-band FFT size.
    #[inline]
    pub fn high_fft_size(&self) -> usize {
        self.high_fft_size
    }

    /// Stretches a mono audio signal using multi-resolution processing.
    ///
    /// Splits the input into three frequency bands, stretches each with
    /// an optimally-sized phase vocoder, and sums the results.
    ///
    /// Bands with input shorter than their FFT size fall back to linear
    /// resampling to avoid errors.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        if input.is_empty() {
            return Ok(vec![]);
        }

        let len = input.len();

        // Resize band buffers if needed (grow only, never shrink in hot path)
        if self.sub_bass_buf.len() < len {
            self.sub_bass_buf.resize(len, 0.0);
            self.mid_buf.resize(len, 0.0);
            self.high_buf.resize(len, 0.0);
        }

        // Split input into three bands
        self.splitter.process(
            input,
            &mut self.sub_bass_buf[..len],
            &mut self.mid_buf[..len],
            &mut self.high_buf[..len],
        );

        let out_len_fallback = (len as f64 * self.stretch_ratio).round().max(1.0) as usize;

        // Process each band with its own PV (or fall back to linear resample)
        let sub_bass_out = if len >= self.sub_bass_fft_size {
            self.sub_bass_pv.process(&self.sub_bass_buf[..len])?
        } else {
            crate::core::resample::resample_linear(&self.sub_bass_buf[..len], out_len_fallback)
        };

        let mid_out = if len >= self.mid_fft_size {
            self.mid_pv.process(&self.mid_buf[..len])?
        } else {
            crate::core::resample::resample_linear(&self.mid_buf[..len], out_len_fallback)
        };

        let high_out = if len >= self.high_fft_size {
            self.high_pv.process(&self.high_buf[..len])?
        } else {
            crate::core::resample::resample_linear(&self.high_buf[..len], out_len_fallback)
        };

        // Sum the three bands, zero-padding shorter outputs.
        // Zero-padding preserves phase coherence between bands — resampling
        // to a common length would shift phases and cause destructive
        // interference. The shorter bands are naturally near-silent at their
        // tails due to PV edge effects, so zero-padding is safe.
        let max_len = sub_bass_out.len().max(mid_out.len()).max(high_out.len());

        let mut output = vec![0.0f32; max_len];
        for (i, s) in sub_bass_out.iter().enumerate() {
            output[i] += s;
        }
        for (i, s) in mid_out.iter().enumerate() {
            output[i] += s;
        }
        for (i, s) in high_out.iter().enumerate() {
            output[i] += s;
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that multi-resolution stretcher produces output of approximately correct length.
    #[test]
    fn test_multi_res_output_length() {
        let sample_rate = 44100;
        let stretch_ratio = 1.5;
        let mut stretcher = MultiResolutionStretcher::new(4096, stretch_ratio, sample_rate, 120.0);

        // Generate 2 seconds of 440 Hz sine (long enough for all FFT sizes)
        let len = sample_rate as usize * 2;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let output = stretcher.process(&input).unwrap();

        // Output should be approximately stretch_ratio * input_len
        let expected = (len as f64 * stretch_ratio) as usize;
        let tolerance = expected / 5; // 20% tolerance for PV edge effects
        assert!(
            output.len().abs_diff(expected) < tolerance,
            "Output length {} too far from expected {} (tolerance {})",
            output.len(),
            expected,
            tolerance
        );
    }

    /// Test that stretch ratio 1.0 with multi-resolution produces near-identity output.
    #[test]
    fn test_multi_res_identity_stretch() {
        let sample_rate = 44100;
        let mut stretcher = MultiResolutionStretcher::new(4096, 1.0, sample_rate, 120.0);

        // Generate 2 seconds of 440 Hz sine
        let len = sample_rate as usize * 2;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let output = stretcher.process(&input).unwrap();

        // Output length should be very close to input length for ratio 1.0
        let length_diff = output.len().abs_diff(input.len());
        assert!(
            length_diff < len / 10,
            "Identity stretch length diff too large: {length_diff} (input={}, output={})",
            input.len(),
            output.len()
        );

        // Check that the output has similar energy to the input
        let input_energy: f64 = input.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        let output_energy: f64 = output
            .iter()
            .take(input.len())
            .map(|s| (*s as f64) * (*s as f64))
            .sum();

        let energy_ratio = output_energy / input_energy;
        assert!(
            (0.3..3.0).contains(&energy_ratio),
            "Energy ratio {energy_ratio:.3} too far from 1.0 for identity stretch"
        );
    }

    /// Test that a 100 Hz sine stretched 1.5x preserves frequency.
    #[test]
    fn test_multi_res_preserves_low_freq() {
        let sample_rate = 44100;
        let freq = 100.0f32;
        let stretch_ratio = 1.5;
        let mut stretcher = MultiResolutionStretcher::new(4096, stretch_ratio, sample_rate, 120.0);

        // 2 seconds of 100 Hz sine
        let len = sample_rate as usize * 2;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let output = stretcher.process(&input).unwrap();

        // Measure dominant frequency via zero-crossing rate
        // Skip edges (start/end) where PV artifacts are worst
        let skip = sample_rate as usize / 2;
        let analysis_len = output.len().saturating_sub(skip * 2);
        if analysis_len < sample_rate as usize {
            // Not enough output to analyze
            return;
        }

        let analysis = &output[skip..skip + analysis_len];
        let zero_crossings = analysis
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();

        // Zero crossings per second = 2 * frequency
        let measured_freq =
            zero_crossings as f64 / 2.0 / (analysis_len as f64 / sample_rate as f64);

        // Frequency should be preserved (within 15% tolerance for PV processing)
        let freq_error = (measured_freq - freq as f64).abs() / freq as f64;
        assert!(
            freq_error < 0.15,
            "100 Hz sine frequency not preserved: measured {measured_freq:.1} Hz, error {:.1}%",
            freq_error * 100.0
        );
    }

    /// Test that a high-frequency sine stretched 1.5x preserves frequency.
    #[test]
    fn test_multi_res_preserves_high_freq() {
        let sample_rate = 44100;
        let freq = 8000.0f32;
        let stretch_ratio = 1.5;
        let mut stretcher = MultiResolutionStretcher::new(4096, stretch_ratio, sample_rate, 120.0);

        // 2 seconds of 8000 Hz sine
        let len = sample_rate as usize * 2;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let output = stretcher.process(&input).unwrap();

        // Measure dominant frequency via zero-crossing rate
        let skip = sample_rate as usize / 2;
        let analysis_len = output.len().saturating_sub(skip * 2);
        if analysis_len < sample_rate as usize {
            return;
        }

        let analysis = &output[skip..skip + analysis_len];
        let zero_crossings = analysis
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();

        let measured_freq =
            zero_crossings as f64 / 2.0 / (analysis_len as f64 / sample_rate as f64);

        let freq_error = (measured_freq - freq as f64).abs() / freq as f64;
        assert!(
            freq_error < 0.15,
            "8 kHz sine frequency not preserved: measured {measured_freq:.1} Hz, error {:.1}%",
            freq_error * 100.0
        );
    }

    /// Test that set_stretch_ratio updates all bands.
    #[test]
    fn test_multi_res_set_ratio() {
        let mut stretcher = MultiResolutionStretcher::new(4096, 1.0, 44100, 120.0);
        stretcher.set_stretch_ratio(2.0);

        // Generate enough input
        let len = 44100 * 2;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let output = stretcher.process(&input).unwrap();

        // Should be approximately 2x the input length
        let expected = len * 2;
        let tolerance = expected / 5;
        assert!(
            output.len().abs_diff(expected) < tolerance,
            "After set_stretch_ratio(2.0): output {} vs expected {} (tol {})",
            output.len(),
            expected,
            tolerance
        );
    }

    /// Test that empty input produces empty output.
    #[test]
    fn test_multi_res_empty_input() {
        let mut stretcher = MultiResolutionStretcher::new(4096, 1.5, 44100, 120.0);
        let output = stretcher.process(&[]).unwrap();
        assert!(output.is_empty());
    }

    /// Test that short input (below FFT sizes) still produces output via fallback.
    #[test]
    fn test_multi_res_short_input_fallback() {
        let mut stretcher = MultiResolutionStretcher::new(4096, 1.5, 44100, 120.0);

        // Input shorter than any FFT size -- all bands fall back to linear resample
        let input = vec![0.5f32; 100];
        let output = stretcher.process(&input).unwrap();
        assert!(
            !output.is_empty(),
            "Short input should still produce output via fallback"
        );
    }

    /// Test FFT size getters.
    #[test]
    fn test_multi_res_fft_sizes() {
        let stretcher = MultiResolutionStretcher::new(4096, 1.0, 44100, 120.0);
        assert_eq!(stretcher.sub_bass_fft_size(), 16384);
        assert_eq!(stretcher.mid_fft_size(), 4096);
        assert_eq!(stretcher.high_fft_size(), 1024);
    }

    /// Test that FFT sizes scale proportionally.
    #[test]
    fn test_multi_res_fft_size_scaling() {
        let stretcher = MultiResolutionStretcher::new(2048, 1.0, 44100, 120.0);
        assert_eq!(stretcher.sub_bass_fft_size(), 8192);
        assert_eq!(stretcher.mid_fft_size(), 2048);
        assert_eq!(stretcher.high_fft_size(), 512);
    }

    /// Test that minimum FFT size is enforced.
    #[test]
    fn test_multi_res_min_fft_size() {
        // With mid_fft_size = 512, high would be 128 but MIN_FFT_SIZE = 256
        let stretcher = MultiResolutionStretcher::new(512, 1.0, 44100, 120.0);
        assert_eq!(stretcher.high_fft_size(), 256);
    }
}

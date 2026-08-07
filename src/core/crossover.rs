//! Linkwitz-Riley crossover filters for multi-band signal splitting.
//!
//! Provides 4th-order (24 dB/oct) and 8th-order (48 dB/oct) Linkwitz-Riley
//! crossover filters that split audio into frequency bands with flat magnitude
//! response at the crossover frequency. LR4 crossovers cascade two 2nd-order
//! Butterworth filters; LR8 crossovers cascade four for steeper roll-off.
//! Both ensure that the low and high outputs sum to unity at all frequencies
//! (minus a small phase shift).

use std::f64::consts::PI;

/// Butterworth Q factor (1/sqrt(2)) for maximally-flat magnitude response.
const BUTTERWORTH_Q: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// A single biquad (second-order IIR) filter section.
///
/// Implements the Direct Form I difference equation:
///   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
///
/// Coefficients are pre-normalized by a0.
#[derive(Debug, Clone)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    // Input delay line
    x1: f64,
    x2: f64,
    // Output delay line
    y1: f64,
    y2: f64,
}

impl Biquad {
    /// Creates a 2nd-order Butterworth low-pass biquad.
    fn lowpass(freq: f64, sample_rate: u32) -> Self {
        Self::lowpass_q(freq, sample_rate, BUTTERWORTH_Q)
    }

    /// Creates a 2nd-order low-pass biquad with an explicit Q.
    fn lowpass_q(freq: f64, sample_rate: u32, q: f64) -> Self {
        let w0 = 2.0 * PI * freq / sample_rate as f64;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let a0 = 1.0 + alpha;
        let b0 = (1.0 - cos_w0) / 2.0 / a0;
        let b1 = (1.0 - cos_w0) / a0;
        let b2 = (1.0 - cos_w0) / 2.0 / a0;
        let a1 = -2.0 * cos_w0 / a0;
        let a2 = (1.0 - alpha) / a0;

        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Creates a 2nd-order Butterworth high-pass biquad.
    fn highpass(freq: f64, sample_rate: u32) -> Self {
        Self::highpass_q(freq, sample_rate, BUTTERWORTH_Q)
    }

    /// Creates a 2nd-order high-pass biquad with an explicit Q.
    fn highpass_q(freq: f64, sample_rate: u32, q: f64) -> Self {
        let w0 = 2.0 * PI * freq / sample_rate as f64;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let a0 = 1.0 + alpha;
        let b0 = (1.0 + cos_w0) / 2.0 / a0;
        let b1 = -(1.0 + cos_w0) / a0;
        let b2 = (1.0 + cos_w0) / 2.0 / a0;
        let a1 = -2.0 * cos_w0 / a0;
        let a2 = (1.0 - alpha) / a0;

        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Processes a single sample through the biquad filter.
    #[inline]
    fn process_sample(&mut self, input: f64) -> f64 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;
        output
    }

    /// Resets all delay line state to zero.
    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// 4th-order Linkwitz-Riley (LR4) crossover filter (24 dB/oct slope).
///
/// Splits an input signal into low-pass and high-pass bands at a specified
/// crossover frequency. The LR4 topology cascades two 2nd-order Butterworth
/// filters, producing a flat magnitude sum at the crossover point.
///
/// # Example
///
/// ```
/// use timestretch::core::crossover::LR4Crossover;
///
/// let mut xover = LR4Crossover::new(200.0, 44100);
/// let (low, high) = xover.process_sample(1.0);
/// // low + high approximately equals the input (with phase shift)
/// ```
pub struct LR4Crossover {
    /// Two cascaded 2nd-order Butterworth low-pass filters.
    low_pass: [Biquad; 2],
    /// Two cascaded 2nd-order Butterworth high-pass filters.
    high_pass: [Biquad; 2],
}

impl LR4Crossover {
    /// Creates a new LR4 crossover at the specified frequency.
    ///
    /// # Arguments
    ///
    /// * `crossover_freq` - Crossover frequency in Hz
    /// * `sample_rate` - Audio sample rate in Hz
    pub fn new(crossover_freq: f64, sample_rate: u32) -> Self {
        Self {
            low_pass: [
                Biquad::lowpass(crossover_freq, sample_rate),
                Biquad::lowpass(crossover_freq, sample_rate),
            ],
            high_pass: [
                Biquad::highpass(crossover_freq, sample_rate),
                Biquad::highpass(crossover_freq, sample_rate),
            ],
        }
    }

    /// Processes a single sample, returning (low, high) band outputs.
    ///
    /// The input is split into two complementary frequency bands at the
    /// crossover frequency set during construction.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> (f32, f32) {
        let x = input as f64;

        // Cascade two Butterworth LP stages for 4th-order LR low-pass
        let lp_stage1 = self.low_pass[0].process_sample(x);
        let low = self.low_pass[1].process_sample(lp_stage1);

        // Cascade two Butterworth HP stages for 4th-order LR high-pass
        let hp_stage1 = self.high_pass[0].process_sample(x);
        let high = self.high_pass[1].process_sample(hp_stage1);

        (low as f32, high as f32)
    }

    /// Processes a buffer, splitting into low and high bands.
    ///
    /// Processes up to the minimum shared length of `input`, `low`, and `high`.
    pub fn process(&mut self, input: &[f32], low: &mut [f32], high: &mut [f32]) {
        let len = input.len().min(low.len()).min(high.len());
        for (i, &sample) in input.iter().take(len).enumerate() {
            let (l, h) = self.process_sample(sample);
            low[i] = l;
            high[i] = h;
        }
    }

    /// Resets all filter state to zero.
    ///
    /// Call this when processing a new, discontinuous signal to prevent
    /// transient artifacts from stale state.
    pub fn reset(&mut self) {
        for bq in &mut self.low_pass {
            bq.reset();
        }
        for bq in &mut self.high_pass {
            bq.reset();
        }
    }
}

/// Butterworth 4th-order section Qs (poles at ±22.5° and ±67.5°).
const BW4_Q1: f64 = 0.541_196_100_146_197;
const BW4_Q2: f64 = 1.306_562_964_876_377;

/// Correct 8th-order Linkwitz-Riley (LR8) crossover: a squared 4th-order
/// Butterworth (sections at Q = 0.5412 and Q = 1.3066, each applied twice).
///
/// This splitter's bands are −6 dB and in phase at the crossover, so
/// `low + high` re-sums to a true allpass — the LR defining property.
/// (A naive cascade of four Q = 0.707 sections is **not** a valid LR
/// topology: each band would sit at −12 dB at the crossover, so the
/// in-phase sum would notch −6 dB there.)
#[derive(Debug)]
pub struct LinkwitzRiley8 {
    low_pass: [Biquad; 4],
    high_pass: [Biquad; 4],
}

impl LinkwitzRiley8 {
    /// Creates a new LR8 crossover at the specified frequency.
    pub fn new(crossover_freq: f64, sample_rate: u32) -> Self {
        Self {
            low_pass: [
                Biquad::lowpass_q(crossover_freq, sample_rate, BW4_Q1),
                Biquad::lowpass_q(crossover_freq, sample_rate, BW4_Q2),
                Biquad::lowpass_q(crossover_freq, sample_rate, BW4_Q1),
                Biquad::lowpass_q(crossover_freq, sample_rate, BW4_Q2),
            ],
            high_pass: [
                Biquad::highpass_q(crossover_freq, sample_rate, BW4_Q1),
                Biquad::highpass_q(crossover_freq, sample_rate, BW4_Q2),
                Biquad::highpass_q(crossover_freq, sample_rate, BW4_Q1),
                Biquad::highpass_q(crossover_freq, sample_rate, BW4_Q2),
            ],
        }
    }

    /// Processes a single sample, returning (low, high) band outputs.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> (f32, f32) {
        let x = input as f64;
        let mut low = x;
        for stage in &mut self.low_pass {
            low = stage.process_sample(low);
        }
        let mut high = x;
        for stage in &mut self.high_pass {
            high = stage.process_sample(high);
        }
        (low as f32, high as f32)
    }

    /// Processes a buffer, splitting into low and high bands.
    ///
    /// Processes up to the minimum shared length of `input`, `low`, and `high`.
    pub fn process(&mut self, input: &[f32], low: &mut [f32], high: &mut [f32]) {
        let len = input.len().min(low.len()).min(high.len());
        for (i, &sample) in input.iter().take(len).enumerate() {
            let (l, h) = self.process_sample(sample);
            low[i] = l;
            high[i] = h;
        }
    }

    /// Resets all filter state to zero.
    pub fn reset(&mut self) {
        for bq in &mut self.low_pass {
            bq.reset();
        }
        for bq in &mut self.high_pass {
            bq.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the LR4 crossover preserves total energy (power complementary).
    ///
    /// LR4 is power-complementary: `|LP(jw)|^2 + |HP(jw)|^2 = 1`. The total
    /// energy in the low + high bands should closely match the input energy.
    /// Note: the time-domain sample sum `low[i] + high[i]` does NOT equal
    /// `input[i]` because LR4 is not amplitude-complementary (it has phase shift).
    #[test]
    fn test_lr4_crossover_energy_conservation() {
        let sample_rate = 44100;
        let crossover_freq = 1000.0;
        let mut xover = LR4Crossover::new(crossover_freq, sample_rate);

        // Use a sine sweep covering multiple frequencies
        let len = 16384;
        let input: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                // Mix of frequencies: 200 Hz + 1000 Hz + 5000 Hz
                (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.33
                    + (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.33
                    + (2.0 * std::f32::consts::PI * 5000.0 * t).sin() * 0.33
            })
            .collect();

        let mut low = vec![0.0f32; len];
        let mut high = vec![0.0f32; len];
        xover.process(&input, &mut low, &mut high);

        // Skip settling time for energy measurement
        let settle = 1024;
        let input_energy: f64 = input[settle..].iter().map(|s| (*s as f64).powi(2)).sum();
        let low_energy: f64 = low[settle..].iter().map(|s| (*s as f64).powi(2)).sum();
        let high_energy: f64 = high[settle..].iter().map(|s| (*s as f64).powi(2)).sum();

        let combined_energy = low_energy + high_energy;
        let energy_ratio = combined_energy / input_energy;

        // LR4 is power complementary (`|LP|^2 + |HP|^2 = 1` at each frequency).
        // For correlated input, the total energy may differ from the input due to
        // cross-terms near the crossover, but should remain within a reasonable range.
        assert!(
            (0.7..1.3).contains(&energy_ratio),
            "LR4 energy ratio {energy_ratio:.4} too far from 1.0 (low={low_energy:.2}, high={high_energy:.2}, input={input_energy:.2})"
        );
    }

    /// Verify that reset clears filter state.
    #[test]
    fn test_lr4_reset() {
        let mut xover = LR4Crossover::new(1000.0, 44100);

        // Process some samples to build up state
        for i in 0..100 {
            xover.process_sample((i as f32 * 0.1).sin());
        }

        xover.reset();

        // After reset, processing a zero should produce (near-)zero output
        let (low, high) = xover.process_sample(0.0);
        assert!(low.abs() < 1e-10, "low should be ~0 after reset, got {low}");
        assert!(
            high.abs() < 1e-10,
            "high should be ~0 after reset, got {high}"
        );
    }

    /// LR8 must re-sum to a true allpass: a tone AT the crossover comes
    /// back at unity amplitude (a naive Q = 0.707 cascade would notch
    /// −6 dB here — the defect `LinkwitzRiley8` exists to avoid).
    #[test]
    fn test_linkwitz_riley8_sums_to_allpass_at_crossover() {
        let sample_rate = 44_100;
        let fc = 150.0;
        let mut xover = LinkwitzRiley8::new(fc, sample_rate);
        let len = 32_768;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * fc as f32 * i as f32 / sample_rate as f32).sin())
            .collect();
        let mut low = vec![0.0f32; len];
        let mut high = vec![0.0f32; len];
        xover.process(&input, &mut low, &mut high);

        let settle = 8_192;
        let sum_energy: f64 = (settle..len)
            .map(|i| ((low[i] + high[i]) as f64).powi(2))
            .sum();
        let in_energy: f64 = input[settle..].iter().map(|s| (*s as f64).powi(2)).sum();
        let level_db = 10.0 * (sum_energy / in_energy).log10();
        assert!(
            level_db.abs() < 0.1,
            "LR8 re-sum at crossover: {level_db:+.3} dB (must be allpass)"
        );

        // And each band sits at −6 dB there (the LR defining property).
        let low_energy: f64 = low[settle..].iter().map(|s| (*s as f64).powi(2)).sum();
        let band_db = 10.0 * (low_energy / in_energy).log10();
        assert!(
            (band_db + 6.0).abs() < 0.3,
            "LR8 band level at crossover: {band_db:+.2} dB, expected −6"
        );
    }

    /// Corrected LR8 keeps the 48 dB/oct slope of the legacy one.
    #[test]
    fn test_linkwitz_riley8_slope_matches_lr8() {
        let sample_rate = 44_100;
        let fc = 1_000.0;
        let mut xover = LinkwitzRiley8::new(fc, sample_rate);
        let freq = 4_000.0; // two octaves above: expect ~ -96 dB low-band
        let len = 32_768;
        let input: Vec<f32> = (0..len)
            .map(|i| {
                (2.0 * std::f32::consts::PI * freq as f32 * i as f32 / sample_rate as f32).sin()
            })
            .collect();
        let mut low = vec![0.0f32; len];
        let mut high = vec![0.0f32; len];
        xover.process(&input, &mut low, &mut high);
        let settle = 8_192;
        let low_energy: f64 = low[settle..].iter().map(|s| (*s as f64).powi(2)).sum();
        let in_energy: f64 = input[settle..].iter().map(|s| (*s as f64).powi(2)).sum();
        let db = 10.0 * (low_energy / in_energy).log10();
        // Theoretical LR8 is −96 dB two octaves up; the f32 signal boundary
        // floors the measurement near −78 dB. Anything past −70 dB proves
        // the 48 dB/oct topology (LR4 would sit at −48 dB).
        assert!(
            db < -70.0,
            "LR8 low band only {db:.1} dB down two octaves above fc"
        );
    }
}

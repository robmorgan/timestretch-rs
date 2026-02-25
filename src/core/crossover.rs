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
        let w0 = 2.0 * PI * freq / sample_rate as f64;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * BUTTERWORTH_Q);

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
        let w0 = 2.0 * PI * freq / sample_rate as f64;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * BUTTERWORTH_Q);

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
    /// The `low` and `high` output slices must be at least as long as `input`.
    ///
    /// # Panics
    ///
    /// Panics if `low` or `high` is shorter than `input`.
    pub fn process(&mut self, input: &[f32], low: &mut [f32], high: &mut [f32]) {
        assert!(
            low.len() >= input.len(),
            "low buffer too short: {} < {}",
            low.len(),
            input.len()
        );
        assert!(
            high.len() >= input.len(),
            "high buffer too short: {} < {}",
            high.len(),
            input.len()
        );

        for (i, &sample) in input.iter().enumerate() {
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

/// 8th-order Linkwitz-Riley (LR8) crossover filter (48 dB/oct slope).
///
/// Splits an input signal into low-pass and high-pass bands at a specified
/// crossover frequency. The LR8 topology cascades four 2nd-order Butterworth
/// filters, producing a steeper roll-off than LR4 while maintaining a flat
/// magnitude sum at the crossover point.
///
/// # Example
///
/// ```
/// use timestretch::core::crossover::LR8Crossover;
///
/// let mut xover = LR8Crossover::new(200.0, 44100);
/// let (low, high) = xover.process_sample(1.0);
/// // low + high approximately equals the input (with phase shift)
/// ```
pub struct LR8Crossover {
    /// Four cascaded 2nd-order Butterworth low-pass filters.
    low_pass: [Biquad; 4],
    /// Four cascaded 2nd-order Butterworth high-pass filters.
    high_pass: [Biquad; 4],
}

impl LR8Crossover {
    /// Creates a new LR8 crossover at the specified frequency.
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
                Biquad::lowpass(crossover_freq, sample_rate),
                Biquad::lowpass(crossover_freq, sample_rate),
            ],
            high_pass: [
                Biquad::highpass(crossover_freq, sample_rate),
                Biquad::highpass(crossover_freq, sample_rate),
                Biquad::highpass(crossover_freq, sample_rate),
                Biquad::highpass(crossover_freq, sample_rate),
            ],
        }
    }

    /// Processes a single sample, returning (low, high) band outputs.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> (f32, f32) {
        let x = input as f64;

        // Cascade four Butterworth LP stages for 8th-order LR low-pass
        let mut low = x;
        for stage in &mut self.low_pass {
            low = stage.process_sample(low);
        }

        // Cascade four Butterworth HP stages for 8th-order LR high-pass
        let mut high = x;
        for stage in &mut self.high_pass {
            high = stage.process_sample(high);
        }

        (low as f32, high as f32)
    }

    /// Processes a buffer, splitting into low and high bands.
    ///
    /// # Panics
    ///
    /// Panics if `low` or `high` is shorter than `input`.
    pub fn process(&mut self, input: &[f32], low: &mut [f32], high: &mut [f32]) {
        assert!(
            low.len() >= input.len(),
            "low buffer too short: {} < {}",
            low.len(),
            input.len()
        );
        assert!(
            high.len() >= input.len(),
            "high buffer too short: {} < {}",
            high.len(),
            input.len()
        );

        for (i, &sample) in input.iter().enumerate() {
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

/// Three-band signal splitter using two cascaded Linkwitz-Riley crossovers.
///
/// Splits audio into sub-bass, mid, and high frequency bands using two
/// LR8 (48 dB/oct) crossover points. The first crossover separates sub-bass
/// from everything above, and the second crossover splits the upper portion
/// into mid and high bands. The steeper LR8 roll-off provides better band
/// isolation than LR4, reducing inter-band leakage.
///
/// # Example
///
/// ```
/// use timestretch::core::crossover::ThreeBandSplitter;
///
/// let mut splitter = ThreeBandSplitter::new(200.0, 4000.0, 44100);
/// let input = vec![0.5f32; 1024];
/// let mut sub_bass = vec![0.0f32; 1024];
/// let mut mid = vec![0.0f32; 1024];
/// let mut high = vec![0.0f32; 1024];
/// splitter.process(&input, &mut sub_bass, &mut mid, &mut high);
/// ```
pub struct ThreeBandSplitter {
    /// Crossover splitting sub-bass from mid+high.
    low_mid: LR8Crossover,
    /// Crossover splitting mid from high (applied to the high output of low_mid).
    mid_high: LR8Crossover,
}

impl ThreeBandSplitter {
    /// Creates a new three-band splitter.
    ///
    /// # Arguments
    ///
    /// * `low_freq` - Crossover frequency between sub-bass and mid (e.g., 200 Hz)
    /// * `high_freq` - Crossover frequency between mid and high (e.g., 4000 Hz)
    /// * `sample_rate` - Audio sample rate in Hz
    pub fn new(low_freq: f64, high_freq: f64, sample_rate: u32) -> Self {
        Self {
            low_mid: LR8Crossover::new(low_freq, sample_rate),
            mid_high: LR8Crossover::new(high_freq, sample_rate),
        }
    }

    /// Splits input into three bands: sub_bass, mid, and high.
    ///
    /// All output slices must be at least as long as `input`.
    ///
    /// # Panics
    ///
    /// Panics if any output buffer is shorter than `input`.
    pub fn process(
        &mut self,
        input: &[f32],
        sub_bass: &mut [f32],
        mid: &mut [f32],
        high: &mut [f32],
    ) {
        assert!(
            sub_bass.len() >= input.len(),
            "sub_bass buffer too short: {} < {}",
            sub_bass.len(),
            input.len()
        );
        assert!(
            mid.len() >= input.len(),
            "mid buffer too short: {} < {}",
            mid.len(),
            input.len()
        );
        assert!(
            high.len() >= input.len(),
            "high buffer too short: {} < {}",
            high.len(),
            input.len()
        );

        for (i, &sample) in input.iter().enumerate() {
            // First crossover: sub-bass vs upper
            let (lo, upper) = self.low_mid.process_sample(sample);
            // Second crossover: mid vs high
            let (m, h) = self.mid_high.process_sample(upper);

            sub_bass[i] = lo;
            mid[i] = m;
            high[i] = h;
        }
    }

    /// Resets all filter state to zero.
    pub fn reset(&mut self) {
        self.low_mid.reset();
        self.mid_high.reset();
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

    /// Verify that the three-band splitter preserves total energy.
    #[test]
    fn test_three_band_energy_conservation() {
        let sample_rate = 44100;
        let mut splitter = ThreeBandSplitter::new(200.0, 4000.0, sample_rate);

        // Mix of frequencies spanning all three bands
        let len = 16384;
        let input: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 80.0 * t).sin() * 0.33
                    + (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.33
                    + (2.0 * std::f32::consts::PI * 8000.0 * t).sin() * 0.33
            })
            .collect();

        let mut sub_bass = vec![0.0f32; len];
        let mut mid = vec![0.0f32; len];
        let mut high = vec![0.0f32; len];
        splitter.process(&input, &mut sub_bass, &mut mid, &mut high);

        let settle = 1024;
        let input_energy: f64 = input[settle..].iter().map(|s| (*s as f64).powi(2)).sum();
        let sub_bass_energy: f64 = sub_bass[settle..].iter().map(|s| (*s as f64).powi(2)).sum();
        let mid_energy: f64 = mid[settle..].iter().map(|s| (*s as f64).powi(2)).sum();
        let high_energy: f64 = high[settle..].iter().map(|s| (*s as f64).powi(2)).sum();

        let combined_energy = sub_bass_energy + mid_energy + high_energy;
        let energy_ratio = combined_energy / input_energy;

        // Three-band split: combined energy should be close to input energy.
        // Slightly wider tolerance than 2-band due to cascaded crossovers.
        assert!(
            (0.7..1.3).contains(&energy_ratio),
            "Three-band energy ratio {energy_ratio:.4} too far from 1.0 \
             (sub={sub_bass_energy:.2}, mid={mid_energy:.2}, high={high_energy:.2}, input={input_energy:.2})"
        );
    }

    /// Verify that a pure low-frequency sine ends up mostly in the sub-bass band.
    #[test]
    fn test_three_band_low_freq_routing() {
        let sample_rate = 44100;
        let mut splitter = ThreeBandSplitter::new(200.0, 4000.0, sample_rate);

        let freq = 50.0; // Well below 200 Hz crossover
        let len = 8192;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut sub_bass = vec![0.0f32; len];
        let mut mid = vec![0.0f32; len];
        let mut high = vec![0.0f32; len];
        splitter.process(&input, &mut sub_bass, &mut mid, &mut high);

        // After settling, sub-bass should have most energy
        let settle = 1024;
        let sub_bass_energy: f32 = sub_bass[settle..].iter().map(|s| s * s).sum();
        let mid_energy: f32 = mid[settle..].iter().map(|s| s * s).sum();
        let high_energy: f32 = high[settle..].iter().map(|s| s * s).sum();

        assert!(
            sub_bass_energy > mid_energy * 10.0,
            "50 Hz should be mostly in sub-bass, but sub_bass={sub_bass_energy:.4}, mid={mid_energy:.4}"
        );
        assert!(
            sub_bass_energy > high_energy * 100.0,
            "50 Hz should have negligible high energy"
        );
    }

    /// Verify that a high-frequency sine ends up mostly in the high band.
    #[test]
    fn test_three_band_high_freq_routing() {
        let sample_rate = 44100;
        let mut splitter = ThreeBandSplitter::new(200.0, 4000.0, sample_rate);

        let freq = 10000.0; // Well above 4000 Hz crossover
        let len = 8192;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut sub_bass = vec![0.0f32; len];
        let mut mid = vec![0.0f32; len];
        let mut high = vec![0.0f32; len];
        splitter.process(&input, &mut sub_bass, &mut mid, &mut high);

        let settle = 1024;
        let sub_bass_energy: f32 = sub_bass[settle..].iter().map(|s| s * s).sum();
        let mid_energy: f32 = mid[settle..].iter().map(|s| s * s).sum();
        let high_energy: f32 = high[settle..].iter().map(|s| s * s).sum();

        assert!(
            high_energy > mid_energy * 10.0,
            "10 kHz should be mostly in high band, but high={high_energy:.4}, mid={mid_energy:.4}"
        );
        assert!(
            high_energy > sub_bass_energy * 100.0,
            "10 kHz should have negligible sub-bass energy"
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

    /// Verify that a mid-range sine ends up mostly in the mid band.
    #[test]
    fn test_three_band_mid_freq_routing() {
        let sample_rate = 44100;
        let mut splitter = ThreeBandSplitter::new(200.0, 4000.0, sample_rate);

        let freq = 1000.0; // Between 200 Hz and 4000 Hz
        let len = 8192;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut sub_bass = vec![0.0f32; len];
        let mut mid = vec![0.0f32; len];
        let mut high = vec![0.0f32; len];
        splitter.process(&input, &mut sub_bass, &mut mid, &mut high);

        let settle = 1024;
        let sub_bass_energy: f32 = sub_bass[settle..].iter().map(|s| s * s).sum();
        let mid_energy: f32 = mid[settle..].iter().map(|s| s * s).sum();
        let high_energy: f32 = high[settle..].iter().map(|s| s * s).sum();

        assert!(
            mid_energy > sub_bass_energy * 10.0,
            "1 kHz should be mostly in mid band, but mid={mid_energy:.4}, sub_bass={sub_bass_energy:.4}"
        );
        assert!(
            mid_energy > high_energy * 10.0,
            "1 kHz should be mostly in mid band, but mid={mid_energy:.4}, high={high_energy:.4}"
        );
    }

    /// Verify that the LR8 crossover preserves total energy (power complementary).
    #[test]
    fn test_lr8_crossover_energy_conservation() {
        let sample_rate = 44100;
        let crossover_freq = 1000.0;
        let mut xover = LR8Crossover::new(crossover_freq, sample_rate);

        let len = 16384;
        let input: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.33
                    + (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.33
                    + (2.0 * std::f32::consts::PI * 5000.0 * t).sin() * 0.33
            })
            .collect();

        let mut low = vec![0.0f32; len];
        let mut high = vec![0.0f32; len];
        xover.process(&input, &mut low, &mut high);

        let settle = 1024;
        let input_energy: f64 = input[settle..].iter().map(|s| (*s as f64).powi(2)).sum();
        let low_energy: f64 = low[settle..].iter().map(|s| (*s as f64).powi(2)).sum();
        let high_energy: f64 = high[settle..].iter().map(|s| (*s as f64).powi(2)).sum();

        let combined_energy = low_energy + high_energy;
        let energy_ratio = combined_energy / input_energy;

        assert!(
            (0.7..1.3).contains(&energy_ratio),
            "LR8 energy ratio {energy_ratio:.4} too far from 1.0 (low={low_energy:.2}, high={high_energy:.2}, input={input_energy:.2})"
        );
    }

    /// Verify that LR8 has steeper roll-off than LR4.
    #[test]
    fn test_lr8_steeper_rolloff_than_lr4() {
        let sample_rate = 44100;
        let crossover_freq = 1000.0;
        let mut lr4 = LR4Crossover::new(crossover_freq, sample_rate);
        let mut lr8 = LR8Crossover::new(crossover_freq, sample_rate);

        // Test with a sine at 2x the crossover frequency (2000 Hz)
        let freq = 2000.0;
        let len = 16384;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut lr4_low = vec![0.0f32; len];
        let mut lr4_high = vec![0.0f32; len];
        lr4.process(&input, &mut lr4_low, &mut lr4_high);

        let mut lr8_low = vec![0.0f32; len];
        let mut lr8_high = vec![0.0f32; len];
        lr8.process(&input, &mut lr8_low, &mut lr8_high);

        let settle = 2048;
        let lr4_low_energy: f64 = lr4_low[settle..].iter().map(|s| (*s as f64).powi(2)).sum();
        let lr8_low_energy: f64 = lr8_low[settle..].iter().map(|s| (*s as f64).powi(2)).sum();

        // At 2x crossover freq, LR8 low-pass should have much less energy than LR4
        // LR4 = 24 dB/oct, LR8 = 48 dB/oct, so at 1 octave above: LR4 = -24dB, LR8 = -48dB
        // That's a ~24 dB difference, i.e. LR8 energy should be ~250x less
        assert!(
            lr8_low_energy < lr4_low_energy * 0.1,
            "LR8 low-pass at 2x crossover should be much lower than LR4: LR8={lr8_low_energy:.6}, LR4={lr4_low_energy:.6}"
        );
    }

    /// Verify that LR8 reset clears filter state.
    #[test]
    fn test_lr8_reset() {
        let mut xover = LR8Crossover::new(1000.0, 44100);

        for i in 0..100 {
            xover.process_sample((i as f32 * 0.1).sin());
        }

        xover.reset();

        let (low, high) = xover.process_sample(0.0);
        assert!(low.abs() < 1e-10, "low should be ~0 after reset, got {low}");
        assert!(
            high.abs() < 1e-10,
            "high should be ~0 after reset, got {high}"
        );
    }
}

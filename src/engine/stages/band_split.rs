//! Two-band split for the keylock chain.
//!
//! Splits each channel at the keylock crossover so the low band can bypass
//! pitch correction entirely (pitch follows tempo there — inaudible at DJ
//! ratios, and it deletes the giant sub-bass FFT and its latency) while the
//! high band is corrected at a small transposition.
//!
//! Uses the corrected [`LinkwitzRiley8`] (a true LR8: bands −6 dB and in
//! phase at the crossover, re-summing to allpass) — 48 dB/oct keeps the
//! band overlap region narrow, which matters here because the two bands
//! play at slightly different pitches near the seam.

use crate::core::crossover::LinkwitzRiley8;

/// Low/high crossover frequency for the keylock chain, in Hz.
///
/// A tuning constant: low enough that pitch-following bass stays below the
/// pitch-salience range, high enough to keep kick fundamentals out of the
/// PV. Settled with corpus evidence in Stage 7 (ROADMAP).
pub const KEYLOCK_CROSSOVER_HZ: f64 = 120.0;

/// Per-channel two-band splitter.
#[derive(Debug)]
pub(crate) struct TwoBandSplit {
    crossovers: Vec<LinkwitzRiley8>,
}

impl TwoBandSplit {
    pub(crate) fn new(crossover_hz: f64, sample_rate: u32, channels: usize) -> Self {
        Self {
            crossovers: (0..channels)
                .map(|_| LinkwitzRiley8::new(crossover_hz, sample_rate))
                .collect(),
        }
    }

    /// Splits one channel's samples into `low` and `high` bands.
    pub(crate) fn process_channel(
        &mut self,
        ch: usize,
        input: &[f32],
        low: &mut [f32],
        high: &mut [f32],
    ) {
        self.crossovers[ch].process(input, low, high);
    }

    pub(crate) fn reset(&mut self) {
        for crossover in &mut self.crossovers {
            crossover.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, sample_rate: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    fn energy(samples: &[f32]) -> f64 {
        samples.iter().map(|&s| (s as f64) * (s as f64)).sum()
    }

    #[test]
    fn bass_routes_low_hats_route_high() {
        let mut split = TwoBandSplit::new(KEYLOCK_CROSSOVER_HZ, 44_100, 1);
        let len = 16_384;
        let bass = sine(50.0, 44_100.0, len);
        let (mut low, mut high) = (vec![0.0; len], vec![0.0; len]);
        split.process_channel(0, &bass, &mut low, &mut high);
        assert!(energy(&low[2048..]) > energy(&high[2048..]) * 100.0);

        split.reset();
        let hat = sine(8_000.0, 44_100.0, len);
        split.process_channel(0, &hat, &mut low, &mut high);
        assert!(energy(&high[2048..]) > energy(&low[2048..]) * 1000.0);
    }

    #[test]
    fn bands_resum_to_allpass_of_input() {
        // LR crossovers are magnitude-complementary: low + high re-sums to
        // an allpass version of the input (same energy, phase-rotated).
        let mut split = TwoBandSplit::new(KEYLOCK_CROSSOVER_HZ, 44_100, 1);
        let len = 32_768;
        let input: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f32 / 44_100.0;
                0.4 * (2.0 * std::f32::consts::PI * 55.0 * t).sin()
                    + 0.4 * (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let (mut low, mut high) = (vec![0.0; len], vec![0.0; len]);
        split.process_channel(0, &input, &mut low, &mut high);
        let sum: Vec<f32> = low.iter().zip(high.iter()).map(|(&l, &h)| l + h).collect();
        let in_energy = energy(&input[4096..]);
        let sum_energy = energy(&sum[4096..]);
        let ratio = sum_energy / in_energy;
        assert!(
            (0.9..1.1).contains(&ratio),
            "re-summed energy ratio {ratio:.4} deviates from allpass"
        );
    }
}

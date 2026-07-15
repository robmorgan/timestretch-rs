//! 3-band multi-resolution waveform peaks.
//!
//! The mono mix is split into low/mid/high bands (two 2nd-order Butterworth
//! crossovers at 200 Hz and 2 kHz) and reduced to per-bucket min/max peaks
//! at a base resolution of 150 buckets/s, with a halving pyramid down to
//! ~1024 buckets so every zoom level can paint at roughly one bucket per
//! pixel without rescanning samples.

/// Number of frequency bands (low / mid / high).
pub const NUM_BANDS: usize = 3;

/// Base peak resolution in buckets per second of audio.
const BASE_BUCKETS_PER_SEC: f64 = 150.0;

/// The pyramid stops halving once a level has at most this many buckets;
/// the coarsest level is what the full-track overview texture rasterizes.
const COARSEST_TARGET_BUCKETS: usize = 1024;

/// Low/mid crossover frequency in Hz.
const CROSSOVER_LOW_HZ: f64 = 200.0;
/// Mid/high crossover frequency in Hz.
const CROSSOVER_HIGH_HZ: f64 = 2_000.0;

/// One resolution level: per-band positive/negative peaks per bucket.
pub struct PeakLevel {
    /// Buckets per second of audio at this level.
    pub buckets_per_sec: f64,
    /// Positive peaks, `[band][bucket]`, bands ordered low/mid/high.
    pub pos: [Vec<f32>; NUM_BANDS],
    /// Negative peaks (≤ 0), same layout.
    pub neg: [Vec<f32>; NUM_BANDS],
}

impl PeakLevel {
    pub fn num_buckets(&self) -> usize {
        self.pos[0].len()
    }
}

/// The full pyramid: `levels[0]` is the finest (150 buckets/s), each
/// following level halves the bucket count.
pub struct BandPeaks {
    levels: Vec<PeakLevel>,
}

/// 2nd-order IIR section, transposed direct form II.
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    /// RBJ Butterworth low-pass (Q = 1/sqrt(2)).
    fn lowpass(cutoff_hz: f64, sample_rate: f64) -> Self {
        let (b0, b1, b2, a0, a1, a2) = {
            let w0 = std::f64::consts::TAU * cutoff_hz / sample_rate;
            let alpha = w0.sin() / std::f64::consts::SQRT_2;
            let cos_w0 = w0.cos();
            (
                (1.0 - cos_w0) / 2.0,
                1.0 - cos_w0,
                (1.0 - cos_w0) / 2.0,
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            )
        };
        Self::normalized(b0, b1, b2, a0, a1, a2)
    }

    /// RBJ Butterworth high-pass (Q = 1/sqrt(2)).
    fn highpass(cutoff_hz: f64, sample_rate: f64) -> Self {
        let (b0, b1, b2, a0, a1, a2) = {
            let w0 = std::f64::consts::TAU * cutoff_hz / sample_rate;
            let alpha = w0.sin() / std::f64::consts::SQRT_2;
            let cos_w0 = w0.cos();
            (
                (1.0 + cos_w0) / 2.0,
                -(1.0 + cos_w0),
                (1.0 + cos_w0) / 2.0,
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            )
        };
        Self::normalized(b0, b1, b2, a0, a1, a2)
    }

    fn normalized(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

impl BandPeaks {
    /// Compute the pyramid from interleaved samples (mixed to mono for
    /// display). One O(n) pass over the samples; offline, so the biquads'
    /// phase lag is irrelevant.
    pub fn compute(samples: &[f32], channels: usize, sample_rate: u32) -> Self {
        let channels = channels.max(1);
        let num_frames = samples.len() / channels;
        let sr = sample_rate.max(1) as f64;
        let num_buckets = ((num_frames as f64 * BASE_BUCKETS_PER_SEC / sr).ceil() as usize).max(1);

        let mut pos: [Vec<f32>; NUM_BANDS] = std::array::from_fn(|_| vec![0.0; num_buckets]);
        let mut neg: [Vec<f32>; NUM_BANDS] = std::array::from_fn(|_| vec![0.0; num_buckets]);

        // Crossover network: low = LP200, high = HP2k, mid = LP2k(HP200).
        let mut lp_low = Biquad::lowpass(CROSSOVER_LOW_HZ, sr);
        let mut hp_low = Biquad::highpass(CROSSOVER_LOW_HZ, sr);
        let mut lp_high = Biquad::lowpass(CROSSOVER_HIGH_HZ, sr);
        let mut hp_high = Biquad::highpass(CROSSOVER_HIGH_HZ, sr);

        let inv_channels = 1.0 / channels as f64;
        let bucket_scale = BASE_BUCKETS_PER_SEC / sr;
        for f in 0..num_frames {
            let mut mono = 0.0f64;
            for c in 0..channels {
                mono += samples[f * channels + c] as f64;
            }
            mono *= inv_channels;

            let low = lp_low.process(mono);
            let above_low = hp_low.process(mono);
            let mid = lp_high.process(above_low);
            let high = hp_high.process(above_low);

            let bucket = ((f as f64 * bucket_scale) as usize).min(num_buckets - 1);
            for (band, sample) in [low, mid, high].into_iter().enumerate() {
                let s = sample as f32;
                if s > pos[band][bucket] {
                    pos[band][bucket] = s;
                }
                if s < neg[band][bucket] {
                    neg[band][bucket] = s;
                }
            }
        }

        let mut levels = vec![PeakLevel {
            buckets_per_sec: BASE_BUCKETS_PER_SEC,
            pos,
            neg,
        }];
        while levels.last().unwrap().num_buckets() > COARSEST_TARGET_BUCKETS {
            levels.push(halve(levels.last().unwrap()));
        }
        Self { levels }
    }

    /// The level whose bucket density best matches `px_per_sec`: the
    /// finest level whose buckets are at least one pixel wide (largest
    /// `buckets_per_sec <= px_per_sec`), so bars never go sub-pixel. When
    /// the view is coarser than every level (zoomed way out), the coarsest
    /// level is the best available.
    pub fn level_for(&self, px_per_sec: f32) -> &PeakLevel {
        self.levels
            .iter()
            .filter(|l| l.buckets_per_sec <= px_per_sec as f64)
            .max_by(|a, b| a.buckets_per_sec.total_cmp(&b.buckets_per_sec))
            .unwrap_or_else(|| self.levels.last().unwrap())
    }

    /// The coarsest level (≤ ~1024 buckets), used for the overview texture.
    pub fn coarsest(&self) -> &PeakLevel {
        self.levels.last().unwrap()
    }
}

/// Pairwise reduction: max of positive peaks, min of negative peaks.
fn halve(level: &PeakLevel) -> PeakLevel {
    let n = level.num_buckets().div_ceil(2);
    let mut pos: [Vec<f32>; NUM_BANDS] = std::array::from_fn(|_| Vec::with_capacity(n));
    let mut neg: [Vec<f32>; NUM_BANDS] = std::array::from_fn(|_| Vec::with_capacity(n));
    for band in 0..NUM_BANDS {
        for pair in level.pos[band].chunks(2) {
            pos[band].push(pair.iter().copied().fold(f32::MIN, f32::max));
        }
        for pair in level.neg[band].chunks(2) {
            neg[band].push(pair.iter().copied().fold(f32::MAX, f32::min));
        }
    }
    PeakLevel {
        buckets_per_sec: level.buckets_per_sec / 2.0,
        pos,
        neg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Interleaved stereo test signal: a 60 Hz sine (low band) plus an
    /// 8 kHz sine (high band), 10 s at 44.1 kHz.
    fn test_signal(secs: f64) -> Vec<f32> {
        let sr = 44_100.0;
        let n = (secs * sr) as usize;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f64 / sr;
            let s = (0.8 * (std::f64::consts::TAU * 60.0 * t).sin()
                + 0.3 * (std::f64::consts::TAU * 8_000.0 * t).sin()) as f32;
            out.push(s);
            out.push(s);
        }
        out
    }

    #[test]
    fn base_resolution_is_150_per_sec() {
        let peaks = BandPeaks::compute(&test_signal(10.0), 2, 44_100);
        assert_eq!(peaks.levels[0].num_buckets(), 1500);
        assert_eq!(peaks.levels[0].buckets_per_sec, 150.0);
    }

    #[test]
    fn pyramid_halves_down_to_coarsest_target() {
        // 60 s -> 9000 base buckets -> 4500 -> 2250 -> 1125 -> 563.
        let peaks = BandPeaks::compute(&test_signal(60.0), 2, 44_100);
        let counts: Vec<usize> = peaks.levels.iter().map(|l| l.num_buckets()).collect();
        assert_eq!(counts, vec![9000, 4500, 2250, 1125, 563]);
        assert!(peaks.coarsest().num_buckets() <= COARSEST_TARGET_BUCKETS);
    }

    #[test]
    fn halving_preserves_global_extrema() {
        let peaks = BandPeaks::compute(&test_signal(30.0), 2, 44_100);
        for band in 0..NUM_BANDS {
            let global_max = |l: &PeakLevel| l.pos[band].iter().copied().fold(f32::MIN, f32::max);
            let global_min = |l: &PeakLevel| l.neg[band].iter().copied().fold(f32::MAX, f32::min);
            for pair in peaks.levels.windows(2) {
                assert_eq!(global_max(&pair[0]), global_max(&pair[1]));
                assert_eq!(global_min(&pair[0]), global_min(&pair[1]));
            }
        }
    }

    #[test]
    fn bands_separate_low_and_high_content() {
        let peaks = BandPeaks::compute(&test_signal(5.0), 2, 44_100);
        let level = &peaks.levels[0];
        // Skip the first buckets (filter settling).
        let mid_bucket = level.num_buckets() / 2;
        let low = level.pos[0][mid_bucket];
        let mid = level.pos[1][mid_bucket];
        let high = level.pos[2][mid_bucket];
        assert!(low > 0.6, "60 Hz should land in the low band, got {low}");
        assert!(high > 0.2, "8 kHz should land in the high band, got {high}");
        assert!(
            mid < 0.15,
            "neither test tone is in the mid band, got {mid}"
        );
    }

    #[test]
    fn level_for_picks_finest_level_with_pixel_wide_buckets() {
        let peaks = BandPeaks::compute(&test_signal(60.0), 2, 44_100);
        // Levels: 150, 75, 37.5, 18.75, ~9.4 buckets/s.
        assert_eq!(peaks.level_for(200.0).buckets_per_sec, 150.0);
        assert_eq!(peaks.level_for(150.0).buckets_per_sec, 150.0);
        assert_eq!(peaks.level_for(100.0).buckets_per_sec, 75.0);
        assert_eq!(peaks.level_for(40.0).buckets_per_sec, 37.5);
        // Below the coarsest density: the coarsest level is the closest fit.
        assert_eq!(peaks.level_for(1.0).buckets_per_sec, 9.375);
    }

    #[test]
    fn empty_input_yields_single_bucket() {
        let peaks = BandPeaks::compute(&[], 2, 44_100);
        assert_eq!(peaks.levels[0].num_buckets(), 1);
        assert_eq!(peaks.coarsest().num_buckets(), 1);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientThresholdPolicy {
    /// Sliding median window size in analysis frames.
    ///
    /// Even values are rounded up to the next odd value to keep the median centered.
    pub median_window_frames: usize,
    /// Sliding window (frames) for the median absolute deviation estimate.
    ///
    /// Wider than the median window for a stable dispersion estimate; even
    /// values round up, and values below `median_window_frames` are raised
    /// to it.
    pub mad_window_frames: usize,
    /// Additive threshold floor to suppress near-silence false positives.
    pub threshold_floor: f32,
    /// Base deviation multiplier in the robust threshold:
    /// `threshold = median + k * MAD + floor` with
    /// `k = threshold_k_base + (1 - sensitivity) * threshold_k_slope`.
    pub threshold_k_base: f32,
    /// Sensitivity-to-k slope (see [`Self::threshold_k_base`]). Higher
    /// sensitivity lowers k and admits more onsets.
    pub threshold_k_slope: f32,
    /// Minimum onset spacing in frames for normal sensitivity settings.
    pub min_gap_frames: usize,
    /// Minimum onset spacing in frames when sensitivity is above
    /// `high_sensitivity_split`.
    pub high_sensitivity_min_gap_frames: usize,
    /// Sensitivity split above which high-sensitivity min-gap is used.
    pub high_sensitivity_split: f32,
}

impl TransientThresholdPolicy {
    /// Returns a clamped, internally valid policy.
    #[inline]
    pub fn sanitized(self) -> Self {
        let mut window = self.median_window_frames.clamp(1, 63);
        if window % 2 == 0 {
            window = window.saturating_add(1).min(63);
        }
        let mut mad_window = self.mad_window_frames.clamp(window, 63);
        if mad_window % 2 == 0 {
            mad_window = mad_window.saturating_add(1).min(63);
        }
        Self {
            median_window_frames: window,
            mad_window_frames: mad_window,
            threshold_floor: self.threshold_floor.max(0.0),
            threshold_k_base: self.threshold_k_base.max(0.0),
            threshold_k_slope: self.threshold_k_slope.max(0.0),
            min_gap_frames: self.min_gap_frames.clamp(1, 32),
            high_sensitivity_min_gap_frames: self.high_sensitivity_min_gap_frames.clamp(1, 32),
            high_sensitivity_split: self.high_sensitivity_split.clamp(0.0, 1.0),
        }
    }

    /// Returns the onset-gap policy for a given sensitivity.
    #[inline]
    pub fn min_gap_for_sensitivity(&self, sensitivity: f32) -> usize {
        if sensitivity > self.high_sensitivity_split {
            self.high_sensitivity_min_gap_frames
        } else {
            self.min_gap_frames
        }
    }
}

impl Default for TransientThresholdPolicy {
    fn default() -> Self {
        Self {
            median_window_frames: DEFAULT_TRANSIENT_MEDIAN_WINDOW_FRAMES,
            mad_window_frames: DEFAULT_TRANSIENT_MAD_WINDOW_FRAMES,
            threshold_floor: DEFAULT_TRANSIENT_THRESHOLD_FLOOR,
            threshold_k_base: DEFAULT_TRANSIENT_THRESHOLD_K_BASE,
            threshold_k_slope: DEFAULT_TRANSIENT_THRESHOLD_K_SLOPE,
            min_gap_frames: DEFAULT_TRANSIENT_MIN_GAP_FRAMES,
            high_sensitivity_min_gap_frames: DEFAULT_TRANSIENT_HIGH_SENS_MIN_GAP_FRAMES,
            high_sensitivity_split: DEFAULT_TRANSIENT_HIGH_SENS_SPLIT,
        }
    }
}

const DEFAULT_TRANSIENT_MEDIAN_WINDOW_FRAMES: usize = 11;
const DEFAULT_TRANSIENT_MAD_WINDOW_FRAMES: usize = 21;
const DEFAULT_TRANSIENT_THRESHOLD_FLOOR: f32 = 0.01;
const DEFAULT_TRANSIENT_THRESHOLD_K_BASE: f32 = 1.5;
const DEFAULT_TRANSIENT_THRESHOLD_K_SLOPE: f32 = 4.5;
const DEFAULT_TRANSIENT_MIN_GAP_FRAMES: usize = 6;
const DEFAULT_TRANSIENT_HIGH_SENS_MIN_GAP_FRAMES: usize = 2;
const DEFAULT_TRANSIENT_HIGH_SENS_SPLIT: f32 = 0.6;

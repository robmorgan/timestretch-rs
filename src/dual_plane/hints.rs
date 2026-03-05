//! Snapshot hints produced by the asynchronous analysis plane.

/// Immutable analysis snapshot consumed by the RT plane.
#[derive(Debug, Clone)]
pub struct RenderHints {
    /// Monotonic sequence number from the analysis plane.
    pub sequence: u64,
    /// Input timeline frame that this snapshot is aligned to.
    pub at_input_frame: usize,
    /// Confidence that current horizon is transient-heavy.
    pub transient_confidence: f32,
    /// Confidence in beat-grid lock for current horizon.
    pub beat_confidence: f32,
    /// Confidence that tonal/PV lane should dominate.
    pub tonal_confidence: f32,
    /// Confidence that residual/noise lane should contribute.
    pub noise_confidence: f32,
    /// Lane bias correction `[transient, tonal, residual]`.
    pub lane_bias: [f32; 3],
    /// Short-horizon multiplicative ratio bias applied to warp-map slope.
    pub ratio_bias: f64,
}

impl Default for RenderHints {
    fn default() -> Self {
        Self {
            sequence: 0,
            at_input_frame: 0,
            transient_confidence: 0.0,
            beat_confidence: 0.0,
            tonal_confidence: 0.5,
            noise_confidence: 0.0,
            lane_bias: [0.0, 0.0, 0.0],
            ratio_bias: 0.0,
        }
    }
}

impl RenderHints {
    /// Returns a normalized lane bias vector.
    #[inline]
    pub fn normalized_lane_bias(&self) -> [f32; 3] {
        let t = self.lane_bias[0].max(0.0);
        let o = self.lane_bias[1].max(0.0);
        let r = self.lane_bias[2].max(0.0);
        let sum = t + o + r;
        if sum <= f32::EPSILON {
            return [0.0, 0.0, 0.0];
        }
        [t / sum, o / sum, r / sum]
    }
}

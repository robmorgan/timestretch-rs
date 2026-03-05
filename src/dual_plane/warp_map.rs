//! Continuous input->output warp mapping with beat/marker anchors.

use crate::error::StretchError;

/// Single anchor in a time-warp map.
///
/// Anchors define a monotonic mapping from input timeline to output timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarpAnchor {
    /// Input timeline position in frames.
    pub input_frame: f64,
    /// Output timeline position in frames.
    pub output_frame: f64,
}

impl WarpAnchor {
    #[inline]
    pub fn new(input_frame: f64, output_frame: f64) -> Result<Self, StretchError> {
        if !input_frame.is_finite() || !output_frame.is_finite() {
            return Err(StretchError::InvalidFormat(
                "warp anchor values must be finite".to_string(),
            ));
        }
        if input_frame < 0.0 || output_frame < 0.0 {
            return Err(StretchError::InvalidFormat(
                "warp anchors must be non-negative".to_string(),
            ));
        }
        Ok(Self {
            input_frame,
            output_frame,
        })
    }
}

/// Piecewise-linear time warp map.
///
/// This replaces scalar "ratio" control with a continuous timeline mapping.
#[derive(Debug, Clone)]
pub struct TimeWarpMap {
    anchors: Vec<WarpAnchor>,
}

impl Default for TimeWarpMap {
    fn default() -> Self {
        Self::identity()
    }
}

impl TimeWarpMap {
    /// Identity map (`t_in == t_out`).
    pub fn identity() -> Self {
        Self {
            anchors: vec![
                WarpAnchor {
                    input_frame: 0.0,
                    output_frame: 0.0,
                },
                WarpAnchor {
                    input_frame: 1.0,
                    output_frame: 1.0,
                },
            ],
        }
    }

    /// Constant-ratio map over a finite horizon.
    pub fn from_ratio(ratio: f64, horizon_frames: usize) -> Result<Self, StretchError> {
        if !ratio.is_finite() || ratio <= 0.0 {
            return Err(StretchError::InvalidRatio(format!(
                "warp ratio must be finite and > 0.0, got {ratio}"
            )));
        }
        let horizon = horizon_frames.max(1) as f64;
        Self::from_anchors(vec![
            WarpAnchor {
                input_frame: 0.0,
                output_frame: 0.0,
            },
            WarpAnchor {
                input_frame: horizon,
                output_frame: horizon * ratio,
            },
        ])
    }

    /// Builds a map from explicit anchors.
    pub fn from_anchors(mut anchors: Vec<WarpAnchor>) -> Result<Self, StretchError> {
        if anchors.len() < 2 {
            return Err(StretchError::InvalidFormat(
                "time-warp map needs at least 2 anchors".to_string(),
            ));
        }
        anchors.sort_by(|a, b| a.input_frame.total_cmp(&b.input_frame));
        validate_anchors(&anchors)?;
        Ok(Self { anchors })
    }

    /// Returns anchors in ascending input-frame order.
    #[inline]
    pub fn anchors(&self) -> &[WarpAnchor] {
        &self.anchors
    }

    /// Inserts an anchor and revalidates monotonicity.
    pub fn insert_anchor(&mut self, anchor: WarpAnchor) -> Result<(), StretchError> {
        let mut anchors = self.anchors.clone();
        anchors.push(anchor);
        anchors.sort_by(|a, b| a.input_frame.total_cmp(&b.input_frame));
        validate_anchors(&anchors)?;
        self.anchors = anchors;
        Ok(())
    }

    /// Evaluates output timeline position for an input position.
    pub fn output_for_input(&self, input_frame: f64) -> f64 {
        if !input_frame.is_finite() {
            return 0.0;
        }
        let idx = self.segment_index(input_frame);
        let a = self.anchors[idx];
        let b = self.anchors[idx + 1];
        let slope = slope(a, b);
        a.output_frame + (input_frame - a.input_frame) * slope
    }

    /// Evaluates local warp slope (`d t_out / d t_in`) at a position.
    pub fn local_ratio_at_input(&self, input_frame: f64) -> f64 {
        let idx = self.segment_index(input_frame);
        slope(self.anchors[idx], self.anchors[idx + 1])
    }

    /// Average ratio over an input interval.
    pub fn ratio_over_range(&self, start_input_frame: f64, end_input_frame: f64) -> f64 {
        if !start_input_frame.is_finite() || !end_input_frame.is_finite() {
            return self.local_ratio_at_input(0.0);
        }
        let delta_in = end_input_frame - start_input_frame;
        if delta_in.abs() <= f64::EPSILON {
            return self.local_ratio_at_input(start_input_frame);
        }
        let out0 = self.output_for_input(start_input_frame);
        let out1 = self.output_for_input(end_input_frame);
        ((out1 - out0) / delta_in).abs()
    }

    #[inline]
    fn segment_index(&self, input_frame: f64) -> usize {
        let idx = self
            .anchors
            .partition_point(|a| a.input_frame <= input_frame)
            .saturating_sub(1);
        idx.min(self.anchors.len().saturating_sub(2))
    }
}

#[inline]
fn slope(a: WarpAnchor, b: WarpAnchor) -> f64 {
    let denom = (b.input_frame - a.input_frame).max(f64::EPSILON);
    (b.output_frame - a.output_frame) / denom
}

fn validate_anchors(anchors: &[WarpAnchor]) -> Result<(), StretchError> {
    for anchor in anchors {
        if !anchor.input_frame.is_finite() || !anchor.output_frame.is_finite() {
            return Err(StretchError::InvalidFormat(
                "warp anchors must be finite".to_string(),
            ));
        }
        if anchor.input_frame < 0.0 || anchor.output_frame < 0.0 {
            return Err(StretchError::InvalidFormat(
                "warp anchors must be non-negative".to_string(),
            ));
        }
    }

    for pair in anchors.windows(2) {
        let a = pair[0];
        let b = pair[1];
        if b.input_frame <= a.input_frame {
            return Err(StretchError::InvalidFormat(
                "warp anchors must be strictly increasing in input time".to_string(),
            ));
        }
        if b.output_frame <= a.output_frame {
            return Err(StretchError::InvalidFormat(
                "warp anchors must be strictly increasing in output time".to_string(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{TimeWarpMap, WarpAnchor};

    #[test]
    fn identity_map_returns_unity_ratio() {
        let map = TimeWarpMap::identity();
        assert!((map.local_ratio_at_input(128.0) - 1.0).abs() < 1e-12);
        assert!((map.output_for_input(1024.0) - 1024.0).abs() < 1e-12);
    }

    #[test]
    fn ratio_map_extrapolates_consistently() {
        let map = TimeWarpMap::from_ratio(1.25, 1024).unwrap();
        assert!((map.output_for_input(2048.0) - 2560.0).abs() < 1e-6);
        assert!((map.ratio_over_range(256.0, 1280.0) - 1.25).abs() < 1e-6);
    }

    #[test]
    fn piecewise_anchor_segments_are_respected() {
        let map = TimeWarpMap::from_anchors(vec![
            WarpAnchor::new(0.0, 0.0).unwrap(),
            WarpAnchor::new(100.0, 120.0).unwrap(),
            WarpAnchor::new(200.0, 170.0).unwrap(),
        ])
        .unwrap();
        assert!((map.local_ratio_at_input(50.0) - 1.2).abs() < 1e-6);
        assert!((map.local_ratio_at_input(150.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rejects_non_monotonic_output() {
        let err = TimeWarpMap::from_anchors(vec![
            WarpAnchor::new(0.0, 0.0).unwrap(),
            WarpAnchor::new(100.0, 100.0).unwrap(),
            WarpAnchor::new(200.0, 99.0).unwrap(),
        ])
        .err()
        .unwrap();
        assert!(format!("{err}").contains("warp anchors"));
    }
}

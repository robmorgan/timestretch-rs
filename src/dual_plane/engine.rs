//! High-level dual-plane orchestrator.

use crate::dual_plane::analysis_plane::{AnalysisPlane, AnalysisPlaneConfig};
use crate::dual_plane::hints::RenderHints;
use crate::dual_plane::quality::QualityTier;
use crate::dual_plane::rt::{
    RtConfig, RtControlSender, RtDelayTelemetry, RtProcessor, RtProfileTelemetry,
    RtRuntimeTelemetry,
};
use crate::dual_plane::warp_map::TimeWarpMap;
use crate::error::StretchError;
use std::sync::Arc;

/// Combined dual-plane processor (RT + async analysis).
pub struct DualPlaneProcessor {
    rt: RtProcessor,
    control: RtControlSender,
    analysis: AnalysisPlane,
    analysis_input_frame_cursor: usize,
    num_channels: usize,
    analysis_interleaved_scratch: Vec<f32>,
}

impl DualPlaneProcessor {
    pub fn prepare(rt_config: RtConfig) -> Result<Self, StretchError> {
        let block_frames = rt_config.block_frames;
        let rt = RtProcessor::prepare(rt_config.clone())?;
        let control = rt.control_sender();
        let analysis = AnalysisPlane::start(
            control.clone(),
            AnalysisPlaneConfig::from_params(&rt_config.params),
        );
        let num_channels = rt.num_channels();
        Ok(Self {
            rt,
            control,
            analysis,
            analysis_input_frame_cursor: 0,
            num_channels,
            analysis_interleaved_scratch: vec![0.0; block_frames.saturating_mul(num_channels)],
        })
    }

    /// RT-core slice API with async analysis submission.
    pub fn process(
        &mut self,
        input_slices: &[&[f32]],
        output_slices: &mut [&mut [f32]],
    ) -> (usize, usize) {
        let _ = self.submit_analysis_slices(input_slices);
        self.rt.process(input_slices, output_slices)
    }

    /// Fallible variant of [`DualPlaneProcessor::process`].
    pub fn process_checked(
        &mut self,
        input_slices: &[&[f32]],
        output_slices: &mut [&mut [f32]],
    ) -> Result<(usize, usize), StretchError> {
        let _ = self.submit_analysis_slices(input_slices);
        self.rt.process_checked(input_slices, output_slices)
    }

    /// RT-only processing call.
    pub fn process_block(
        &mut self,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<(), StretchError> {
        self.rt.process_block(input, output)
    }

    /// RT-only processing call into a fixed interleaved output buffer.
    pub fn process_block_into(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<usize, StretchError> {
        self.rt.process_block_into(input, output)
    }

    /// Convenience call: enqueue analysis for this block, then run RT processing.
    pub fn process_block_with_analysis(
        &mut self,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<(), StretchError> {
        let _ = self.submit_analysis_block(input);
        self.process_block(input, output)
    }

    /// Convenience call: enqueue analysis for this block, then run RT processing into a slice.
    pub fn process_block_with_analysis_into(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<usize, StretchError> {
        let _ = self.submit_analysis_block(input);
        self.process_block_into(input, output)
    }

    /// Queues a block for async analysis.
    pub fn submit_analysis_block(&mut self, input: &[f32]) -> bool {
        let at = self.analysis_input_frame_cursor;
        let submitted = self.analysis.submit_block(input, at);
        self.analysis_input_frame_cursor = self
            .analysis_input_frame_cursor
            .saturating_add(input.len() / self.num_channels.max(1));
        submitted
    }

    fn submit_analysis_slices(&mut self, input_slices: &[&[f32]]) -> bool {
        if input_slices.len() != self.num_channels {
            return false;
        }
        let frames = input_slices.first().map_or(0, |ch| ch.len());
        for slice in input_slices {
            if slice.len() != frames {
                return false;
            }
        }

        let needed = frames.saturating_mul(self.num_channels);
        if needed > self.analysis_interleaved_scratch.len() {
            return false;
        }
        for frame in 0..frames {
            for ch in 0..self.num_channels {
                self.analysis_interleaved_scratch[frame * self.num_channels + ch] =
                    input_slices[ch][frame];
            }
        }

        let at = self.analysis_input_frame_cursor;
        let submitted = self
            .analysis
            .submit_block(&self.analysis_interleaved_scratch[..needed], at);
        self.analysis_input_frame_cursor = self.analysis_input_frame_cursor.saturating_add(frames);
        submitted
    }

    /// Publishes a new warp map snapshot.
    pub fn publish_warp_map(&self, map: Arc<TimeWarpMap>) -> bool {
        self.control.publish_warp_map(map)
    }

    /// Publishes a manual render-hint snapshot.
    pub fn publish_hints(&self, hints: Arc<RenderHints>) -> bool {
        self.control.publish_hints(hints)
    }

    pub fn quality_tier(&self) -> QualityTier {
        self.rt.quality_tier()
    }

    pub fn set_latency_profile(&mut self, profile: crate::dual_plane::LatencyProfile) {
        self.rt.set_latency_profile(profile);
    }

    pub fn set_auto_profile_switching(&mut self, enabled: bool) {
        self.rt.set_auto_profile_switching(enabled);
    }

    pub fn profile_telemetry(&self) -> RtProfileTelemetry {
        self.rt.profile_telemetry()
    }

    pub fn runtime_telemetry(&self) -> RtRuntimeTelemetry {
        self.rt.runtime_telemetry()
    }

    pub fn delay_telemetry(&self) -> RtDelayTelemetry {
        self.rt.delay_telemetry()
    }

    pub fn active_ratio(&self) -> f64 {
        self.rt.active_ratio()
    }

    pub fn flush(&mut self, output: &mut Vec<f32>) -> Result<(), StretchError> {
        self.rt.flush(output)
    }

    /// Flushes pending RT state into a fixed interleaved output buffer.
    pub fn flush_into(&mut self, output: &mut [f32]) -> Result<usize, StretchError> {
        self.rt.flush_into(output)
    }

    pub fn rt(&self) -> &RtProcessor {
        &self.rt
    }

    pub fn rt_mut(&mut self) -> &mut RtProcessor {
        &mut self.rt
    }
}

//! Async analysis/decision plane.
//!
//! This worker performs heavier transient/beat/tonal analysis off the callback
//! thread and publishes immutable snapshots to the RT plane.

use crate::analysis::adaptive_snapshot::analyze_adaptive_snapshot_mono;
use crate::core::types::StretchParams;
use crate::dual_plane::hints::RenderHints;
use crate::dual_plane::rt::RtControlSender;
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Async analysis configuration.
#[derive(Debug, Clone)]
pub struct AnalysisPlaneConfig {
    pub params: StretchParams,
    pub channels: usize,
    pub job_queue_capacity: usize,
}

impl AnalysisPlaneConfig {
    pub fn from_params(params: &StretchParams) -> Self {
        Self {
            params: params.clone(),
            channels: params.channels.count().max(1),
            job_queue_capacity: 4,
        }
    }
}

enum AnalysisCommand {
    Analyze {
        interleaved: Vec<f32>,
        at_input_frame: usize,
    },
    Shutdown,
}

/// Asynchronous analysis worker.
pub struct AnalysisPlane {
    tx: SyncSender<AnalysisCommand>,
    worker: Option<JoinHandle<()>>,
}

impl AnalysisPlane {
    pub fn start(control: RtControlSender, cfg: AnalysisPlaneConfig) -> Self {
        let cap = cfg.job_queue_capacity.max(1);
        let (tx, rx) = sync_channel::<AnalysisCommand>(cap);

        let worker = thread::Builder::new()
            .name("timestretch-analysis".to_string())
            .spawn(move || {
                let mut sequence = 0u64;
                while let Ok(cmd) = rx.recv() {
                    let (interleaved, at_input_frame) = match cmd {
                        AnalysisCommand::Analyze {
                            interleaved,
                            at_input_frame,
                        } => (interleaved, at_input_frame),
                        AnalysisCommand::Shutdown => break,
                    };

                    let hints = analyze_block(
                        &interleaved,
                        at_input_frame,
                        sequence,
                        cfg.channels,
                        &cfg.params,
                    );
                    sequence = sequence.saturating_add(1);
                    let _ = control.publish_hints(Arc::new(hints));
                }
            })
            .ok();

        Self { tx, worker }
    }

    /// Enqueues a block for asynchronous analysis.
    ///
    /// Returns `false` if the queue is currently saturated.
    pub fn submit_block(&self, interleaved: &[f32], at_input_frame: usize) -> bool {
        let job = AnalysisCommand::Analyze {
            interleaved: interleaved.to_vec(),
            at_input_frame,
        };
        match self.tx.try_send(job) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => false,
            Err(TrySendError::Disconnected(_)) => false,
        }
    }
}

impl Drop for AnalysisPlane {
    fn drop(&mut self) {
        let _ = self.tx.try_send(AnalysisCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn analyze_block(
    interleaved: &[f32],
    at_input_frame: usize,
    sequence: u64,
    channels: usize,
    params: &StretchParams,
) -> RenderHints {
    let mono = mix_to_mono(interleaved, channels);
    if mono.is_empty() {
        return RenderHints {
            sequence,
            at_input_frame,
            ..RenderHints::default()
        };
    }

    let adaptive = analyze_adaptive_snapshot_mono(&mono, params);

    RenderHints {
        sequence,
        at_input_frame,
        transient_confidence: adaptive.transient_confidence,
        beat_confidence: adaptive.beat_confidence,
        tonal_confidence: adaptive.tonal_confidence,
        noise_confidence: adaptive.noise_confidence,
        lane_bias: adaptive.lane_bias,
        ratio_bias: adaptive.ratio_bias,
        transient_mask: adaptive.transient_mask,
    }
}

fn mix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::analysis::adaptive_snapshot::build_transient_mask;

    #[test]
    fn transient_mask_marks_onset_regions() {
        let mask = build_transient_mask(128, 48_000, 0.001, &[32, 96], &[1.0, 0.5]);
        assert_eq!(mask.len(), 128);
        assert!(mask[32] > 0.9);
        assert!(mask[96] > 0.4);
        assert!(mask[0] < 0.1);
    }
}

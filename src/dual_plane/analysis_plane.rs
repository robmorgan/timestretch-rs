//! Async analysis/decision plane.
//!
//! This worker performs heavier transient/beat/tonal analysis off the callback
//! thread and publishes immutable snapshots to the RT plane.

use crate::analysis::beat::detect_beats;
use crate::analysis::frequency::{compute_band_energy, FrequencyBands};
use crate::analysis::transient::{detect_transients_with_options, TransientDetectionOptions};
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

    let analysis_fft = params.fft_size.min(2048).max(256);
    let analysis_hop = params.hop_size.min(512).max(64);
    let transient_map = detect_transients_with_options(
        &mono,
        params.sample_rate,
        analysis_fft,
        analysis_hop,
        params.transient_sensitivity,
        TransientDetectionOptions::from_stretch_params(params),
    );
    let duration_secs = mono.len() as f32 / params.sample_rate.max(1) as f32;
    let transient_density = transient_map.onsets.len() as f32 / duration_secs.max(1e-3);
    let transient_confidence = (transient_density / 8.0).clamp(0.0, 1.0);

    let beat_confidence = if mono.len() >= params.sample_rate as usize {
        let grid = detect_beats(&mono, params.sample_rate);
        if grid.bpm > 0.0 && grid.beats.len() > 1 {
            let beat_count = (grid.beats.len().min(16) as f32) / 16.0;
            let bpm_center_error = ((grid.bpm - 128.0).abs() / 96.0).clamp(0.0, 1.0) as f32;
            (beat_count * (1.0 - bpm_center_error)).clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let tonal_confidence = if mono.len() >= analysis_fft {
        let (sub, low, mid, high) = compute_band_energy(
            &mono[..analysis_fft],
            analysis_fft,
            params.sample_rate,
            &FrequencyBands::default(),
        );
        let total = sub + low + mid + high + 1e-9;
        ((sub + low + mid) / total).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let noise_confidence = (1.0 - tonal_confidence).clamp(0.0, 1.0);
    let lane_bias = [transient_confidence, tonal_confidence, noise_confidence];

    RenderHints {
        sequence,
        at_input_frame,
        transient_confidence,
        beat_confidence,
        tonal_confidence,
        noise_confidence,
        lane_bias,
        ratio_bias: ((beat_confidence as f64 - transient_confidence as f64) * 0.04)
            .clamp(-0.08, 0.08),
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

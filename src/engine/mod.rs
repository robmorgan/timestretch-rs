//! Real-time-first engine.
//!
//! A stage graph driven from the audio callback, with tempo implemented by
//! a varispeed sinc resampler at the head of the chain. This is the only
//! engine: batch [`stretch`](crate::stretch()) runs the same graph offline
//! (see [`offline`]).
//!
//! # Architecture
//!
//! ```text
//! host thread                    audio thread
//! ───────────                    ────────────
//! SourceProducer ──ring──▶ varispeed head ─▶ [stages…] ─▶ process(out)
//! EngineController ──mailbox──▶ (tempo & future params)
//! ```
//!
//! - **Audio-callback API:** [`EngineProcessor::process`] fills exactly the requested
//!   frames. No `Result`, no allocation, no locks in the audio path;
//!   invariants are enforced at construction and validation in the hot path
//!   is debug-only.
//! - **Control:** [`EngineController`] writes to a lock-free timestamped
//!   parameter mailbox. Tempo retargets are sample-accurate — no control
//!   glide; the resampler ramps over one 32-frame feed chunk only.
//! - **Source:** the host keeps a lock-free ring topped up through
//!   [`SourceProducer`]; underruns render silence and are counted, never
//!   errors.
//!
//! # Example
//!
//! ```
//! use timestretch::engine::{Engine, EngineConfig};
//!
//! let handles = Engine::build(EngineConfig::default()).unwrap();
//! let (controller, mut processor, mut source) =
//!     (handles.controller, handles.processor, handles.source);
//!
//! // Host side: keep the source ring fed (stereo interleaved).
//! let track = vec![0.0f32; 44_100 * 2];
//! source.push(&track);
//! controller.set_tempo_rate(1.04); // +4%, pitch follows in tape mode
//!
//! // Audio callback side: fill exactly what the device asks for.
//! let mut out = vec![0.0f32; 256 * 2];
//! processor.process(&mut out);
//! ```

pub mod control;
pub mod graph;
pub mod offline;
pub mod profiles;
pub mod source;
pub mod stage;
pub mod stages;

pub use control::{EngineController, MAX_TEMPO_RATE, MIN_TEMPO_RATE};
pub use graph::EngineProcessor;
pub use profiles::EngineProfile;
pub use source::SourceProducer;
pub use stage::{BLOCK_FRAMES, BlockBuf, Stage, StageCtx};

use crate::engine::control::EngineShared;
use crate::engine::source::SourceRing;
use crate::error::StretchError;

/// Construction-time engine configuration. All invariants are validated in
/// [`Engine::build`]; nothing here can fail later on the audio thread.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Sample rate in Hz (8 kHz – 192 kHz).
    pub sample_rate: u32,
    /// Interleaved channel count (1–8).
    pub channels: usize,
    /// Stage chain to run after the varispeed head.
    pub profile: EngineProfile,
    /// Tempo rate in effect before the first control write (clamped to
    /// [`MIN_TEMPO_RATE`]..=[`MAX_TEMPO_RATE`]).
    pub initial_tempo_rate: f64,
    /// Largest audio-callback size the engine optimizes for, in frames
    /// (64–8192). Larger `process` calls still work — they are rendered in
    /// chunks of this size.
    pub max_block_frames: usize,
    /// Source ring capacity in frames. Must cover at least four maximum
    /// callbacks at the fastest tempo so the host has real scheduling slack.
    pub source_capacity_frames: usize,
    /// Offline pre-analysis of the track being fed (beatgrid, onsets,
    /// strengths). When present it is the engine's primary transient
    /// control signal: SOLA splices steer around onsets and the PV
    /// schedules per-band phase resets at them. Positions are absolute
    /// track frames — anchor the feed with
    /// [`SourceProducer::set_track_position`]. `None` falls back to online
    /// detection.
    pub pre_analysis: Option<std::sync::Arc<crate::core::preanalysis::PreAnalysisArtifact>>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            channels: 2,
            profile: EngineProfile::Tape,
            initial_tempo_rate: 1.0,
            max_block_frames: 1024,
            source_capacity_frames: 32_768,
            pre_analysis: None,
        }
    }
}

impl EngineConfig {
    fn validate(&self) -> Result<(), StretchError> {
        if !(8_000..=192_000).contains(&self.sample_rate) {
            return Err(StretchError::InvalidFormat(format!(
                "sample rate {} outside 8000..=192000",
                self.sample_rate
            )));
        }
        if !(1..=8).contains(&self.channels) {
            return Err(StretchError::InvalidFormat(format!(
                "channel count {} outside 1..=8",
                self.channels
            )));
        }
        if !(64..=8192).contains(&self.max_block_frames) {
            return Err(StretchError::InvalidFormat(format!(
                "max_block_frames {} outside 64..=8192",
                self.max_block_frames
            )));
        }
        let min_source = (self.max_block_frames as f64 * control::MAX_TEMPO_RATE) as usize * 4;
        if self.source_capacity_frames < min_source {
            return Err(StretchError::InvalidFormat(format!(
                "source_capacity_frames {} below minimum {} for max_block_frames {}",
                self.source_capacity_frames, min_source, self.max_block_frames
            )));
        }
        if !self.initial_tempo_rate.is_finite() {
            return Err(StretchError::InvalidRatio(
                "initial_tempo_rate must be finite".to_string(),
            ));
        }
        Ok(())
    }
}

/// The three engine halves returned by [`Engine::build`]: hand
/// [`EngineHandles::processor`] to the audio thread,
/// [`EngineHandles::source`] to the feed thread, and keep
/// [`EngineHandles::controller`] wherever control lives.
#[derive(Debug)]
pub struct EngineHandles {
    pub controller: EngineController,
    pub processor: EngineProcessor,
    pub source: SourceProducer,
}

/// Constructor namespace for the engine.
#[derive(Debug)]
pub struct Engine;

impl Engine {
    /// Builds an engine with the profile's fixed stage chain.
    pub fn build(config: EngineConfig) -> Result<EngineHandles, StretchError> {
        let stages = profiles::build_stages(config.profile, config.sample_rate, config.channels);
        Self::build_with_stages(config, stages)
    }

    /// Builds an engine with an explicit stage chain (used by tests and
    /// profile development; the public path is [`Engine::build`]).
    pub fn build_with_stages(
        config: EngineConfig,
        stages: Vec<Box<dyn Stage>>,
    ) -> Result<EngineHandles, StretchError> {
        config.validate()?;
        let initial_rate = control::clamp_tempo_rate(config.initial_tempo_rate);
        let shared = EngineShared::new(initial_rate);
        let ring = SourceRing::new(config.source_capacity_frames, config.channels);

        let controller = EngineController::new(shared.clone(), config.sample_rate);
        let processor = EngineProcessor::new(
            shared,
            ring.clone(),
            stages,
            config.channels,
            config.sample_rate,
            initial_rate,
            config.max_block_frames,
            config.pre_analysis.clone(),
            config.profile,
        );
        let source = SourceProducer::new(ring);
        Ok(EngineHandles {
            controller,
            processor,
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation_rejects_bad_values() {
        assert!(
            Engine::build(EngineConfig {
                channels: 0,
                ..EngineConfig::default()
            })
            .is_err()
        );
        assert!(
            Engine::build(EngineConfig {
                sample_rate: 1000,
                ..EngineConfig::default()
            })
            .is_err()
        );
        assert!(
            Engine::build(EngineConfig {
                max_block_frames: 16,
                ..EngineConfig::default()
            })
            .is_err()
        );
        assert!(
            Engine::build(EngineConfig {
                source_capacity_frames: 100,
                ..EngineConfig::default()
            })
            .is_err()
        );
        assert!(Engine::build(EngineConfig::default()).is_ok());
    }

    #[test]
    fn initial_tempo_rate_is_clamped_not_rejected() {
        let handles = Engine::build(EngineConfig {
            initial_tempo_rate: 100.0,
            ..EngineConfig::default()
        })
        .unwrap();
        assert_eq!(handles.processor.current_tempo_rate(), MAX_TEMPO_RATE);
    }
}

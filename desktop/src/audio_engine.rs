use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, Stream, StreamConfig};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::state::SharedStateHandle;

pub struct AudioEngine {
    _stream: Stream,
}

impl AudioEngine {
    /// Queries the default output device's sample rate without opening a
    /// stream. Used at startup to size the UI before any track is loaded.
    pub fn default_sample_rate() -> Option<u32> {
        let host = cpal::default_host();
        let device = host.default_output_device()?;
        let config = device.default_output_config().ok()?;
        Some(config.sample_rate().0)
    }

    /// Create an audio engine whose callback owns an
    /// [`timestretch::engine::EngineProcessor`] and reads audio from it
    /// directly — no intermediate processed-audio ring. The feed thread
    /// keeps the engine's source ring topped up and signals hard restarts
    /// via `reset_request`, which the callback acknowledges by clearing the
    /// flag after resetting.
    pub fn new(
        state: SharedStateHandle,
        stream_active: Arc<AtomicBool>,
        desired_sample_rate: Option<u32>,
        mut processor: timestretch::engine::EngineProcessor,
        reset_request: Arc<AtomicBool>,
    ) -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "No audio output device found".to_string())?;

        let default_config = device
            .default_output_config()
            .map_err(|e| format!("Failed to get default output config: {e}"))?;

        let sample_rate = desired_sample_rate.unwrap_or(default_config.sample_rate().0);
        let config = StreamConfig {
            channels: 2,
            sample_rate: SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    // Acknowledge a pending hard restart before anything
                    // else so seeks work even while output is inactive.
                    if reset_request.load(Ordering::Acquire) {
                        processor.reset();
                        reset_request.store(false, Ordering::Release);
                    }

                    if !stream_active.load(Ordering::Relaxed) {
                        data.fill(0.0);
                        return;
                    }

                    let volume = {
                        let st = state.lock().unwrap();
                        st.volume
                    };

                    processor.process(data);
                    for sample in data.iter_mut() {
                        *sample *= volume;
                    }
                },
                move |err| {
                    log::error!("Audio output error: {err}");
                },
                None,
            )
            .map_err(|e| format!("Failed to build output stream: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("Failed to start audio stream: {e}"))?;

        Ok(AudioEngine { _stream: stream })
    }
}

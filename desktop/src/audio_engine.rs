use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, Stream, StreamConfig};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::delivery::Delivery;
use crate::state::{AtomicVolume, ScrubState};

pub struct AudioEngine {
    _stream: Stream,
    _queue_guard: crate::output_queue::QueueLifetime,
    _source_guard: Arc<Vec<f32>>,
    failed: Arc<AtomicBool>,
}

impl AudioEngine {
    pub fn failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }
    /// Queries the default output device's sample rate without opening a
    /// stream. Used at startup to size the UI before any track is loaded.
    pub fn default_sample_rate() -> Option<u32> {
        let host = cpal::default_host();
        let device = host.default_output_device()?;
        let config = device.default_output_config().ok()?;
        Some(config.sample_rate().0)
    }

    /// Device-only callback over a desktop queue. Library transport actions
    /// are prepared on the render worker. Scrubbing remains an application voice.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        volume: Arc<AtomicVolume>,
        stream_active: Arc<AtomicBool>,
        desired_sample_rate: Option<u32>,
        consumer: crate::output_queue::Consumer,
        scrub: Arc<ScrubState>,
        source_audio: Arc<Vec<f32>>,
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

        let queue_guard = consumer.keep_alive();
        let source_guard = source_audio.clone();
        let failed = Arc::new(AtomicBool::new(false));
        let callback_failed = failed.clone();
        let mut delivery = Delivery::new(
            sample_rate,
            consumer,
            volume,
            stream_active,
            scrub,
            source_audio,
        );

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    delivery.render(data);
                },
                move |_err| {
                    callback_failed.store(true, Ordering::Release);
                },
                None,
            )
            .map_err(|e| format!("Failed to build output stream: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("Failed to start audio stream: {e}"))?;

        Ok(AudioEngine {
            _stream: stream,
            _queue_guard: queue_guard,
            _source_guard: source_guard,
            failed,
        })
    }
}

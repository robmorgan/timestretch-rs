use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, Stream, StreamConfig};
use ringbuf::traits::{Consumer, Observer, Split};
use ringbuf::HeapRb;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::state::SharedStateHandle;

/// Ring buffer capacity: ~2 seconds of stereo audio at 44100 Hz.
const RING_BUFFER_SIZE: usize = 44100 * 2 * 2;

pub type RingProducer = ringbuf::HeapProd<f32>;

pub struct AudioEngine {
    _stream: Stream,
    pub output_sample_rate: u32,
}

impl AudioEngine {
    /// Create a new audio engine with a cpal output stream.
    /// Returns the engine and a ring buffer producer for feeding audio.
    pub fn new(
        state: SharedStateHandle,
        stream_active: Arc<AtomicBool>,
    ) -> Result<(Self, RingProducer), String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "No audio output device found".to_string())?;

        let default_config = device
            .default_output_config()
            .map_err(|e| format!("Failed to get default output config: {e}"))?;

        let sample_rate = default_config.sample_rate().0;
        let config = StreamConfig {
            channels: 2,
            sample_rate: SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let rb = HeapRb::<f32>::new(RING_BUFFER_SIZE);
        let (producer, mut consumer) = rb.split();

        let state_clone = state.clone();
        let stream_active_clone = stream_active.clone();

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if !stream_active_clone.load(Ordering::Relaxed) {
                        data.fill(0.0);
                        return;
                    }

                    let volume = {
                        let st = state_clone.lock().unwrap();
                        st.volume
                    };

                    let available = consumer.occupied_len();
                    let to_read = data.len().min(available);

                    // Read from ring buffer
                    let read = consumer.pop_slice(&mut data[..to_read]);

                    // Apply volume
                    for sample in &mut data[..read] {
                        *sample *= volume;
                    }

                    // Fill remainder with silence
                    for sample in &mut data[read..] {
                        *sample = 0.0;
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

        Ok((
            AudioEngine {
                _stream: stream,
                output_sample_rate: sample_rate,
            },
            producer,
        ))
    }
}

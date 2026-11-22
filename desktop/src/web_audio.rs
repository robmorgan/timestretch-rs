//! Browser audio output. There is no device callback thread here: the page
//! timer calls [`AudioEngine::pump`], which renders [`Delivery`] blocks on the
//! main thread and schedules them back to back on the Web Audio clock, a
//! fixed lead ahead of the playhead. The lead absorbs main-thread stalls and
//! is the price of having no threads: it adds to control and scrub latency.
use crate::delivery::Delivery;
use crate::output_queue::Consumer;
use crate::state::{AtomicVolume, ScrubState};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use web_sys::{AudioContext, AudioContextOptions};

/// Frames rendered and scheduled per block.
const BLOCK_FRAMES: usize = 512;
/// How far ahead of the audio clock blocks are scheduled.
pub const LEAD_SECS: f64 = 0.12;
/// Restart margin after the schedule has fallen behind the clock.
const RESTART_SECS: f64 = 0.02;

pub struct AudioEngine {
    context: AudioContext,
    delivery: Delivery,
    sample_rate: f32,
    /// Context time at which the next block starts.
    next_time: f64,
    interleaved: Vec<f32>,
    left: Vec<f32>,
    right: Vec<f32>,
}

impl AudioEngine {
    /// The rate the browser would pick for a new context.
    pub fn default_sample_rate() -> Option<u32> {
        let context = AudioContext::new().ok()?;
        let rate = context.sample_rate() as u32;
        let _ = context.close();
        Some(rate)
    }

    pub fn new(
        volume: Arc<AtomicVolume>,
        stream_active: Arc<AtomicBool>,
        sample_rate: u32,
        consumer: Consumer,
        scrub: Arc<ScrubState>,
        source_audio: Arc<Vec<f32>>,
    ) -> Result<Self, String> {
        let options = AudioContextOptions::new();
        options.set_sample_rate(sample_rate as f32);
        let context = AudioContext::new_with_context_options(&options)
            .map_err(|e| format!("Failed to create audio context: {e:?}"))?;
        // Browsers start contexts suspended until the page has seen a user
        // gesture; playback is always started by one.
        let _ = context.resume();
        // The context may not honor the requested rate.
        let sample_rate = context.sample_rate();
        Ok(Self {
            delivery: Delivery::new(
                sample_rate as u32,
                consumer,
                volume,
                stream_active,
                scrub,
                source_audio,
            ),
            context,
            sample_rate,
            next_time: 0.0,
            interleaved: vec![0.0; BLOCK_FRAMES * 2],
            left: vec![0.0; BLOCK_FRAMES],
            right: vec![0.0; BLOCK_FRAMES],
        })
    }

    /// Schedule blocks until the lead is full. Cheap when it already is.
    pub fn pump(&mut self) {
        let now = self.context.current_time();
        if self.next_time < now {
            // First block, or the main thread stalled past the lead.
            if self.next_time > 0.0 {
                log::warn!(
                    "Audio schedule fell {:.0} ms behind; restarting",
                    (now - self.next_time) * 1000.0
                );
            }
            self.next_time = now + RESTART_SECS;
        }
        while self.next_time < now + LEAD_SECS {
            if self.schedule_block().is_err() {
                return;
            }
            self.next_time += BLOCK_FRAMES as f64 / self.sample_rate as f64;
        }
    }

    fn schedule_block(&mut self) -> Result<(), wasm_bindgen::JsValue> {
        self.delivery.render(&mut self.interleaved);
        for (frame, (left, right)) in self
            .interleaved
            .chunks_exact(2)
            .zip(self.left.iter_mut().zip(self.right.iter_mut()))
        {
            *left = frame[0];
            *right = frame[1];
        }
        let buffer = self
            .context
            .create_buffer(2, BLOCK_FRAMES as u32, self.sample_rate)?;
        buffer.copy_to_channel(&self.left, 0)?;
        buffer.copy_to_channel(&self.right, 1)?;
        let node = self.context.create_buffer_source()?;
        node.set_buffer(Some(&buffer));
        node.connect_with_audio_node(&self.context.destination())?;
        node.start_with_when(self.next_time)?;
        Ok(())
    }
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        let _ = self.context.close();
    }
}

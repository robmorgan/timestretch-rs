use ringbuf::traits::{Observer, Producer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use timestretch::{StretchParams, StreamProcessor};

use crate::audio_engine::RingProducer;
use crate::state::{AtomicPosition, SharedStateHandle, StopFlag, Transport};

/// Chunk size for feeding into StreamProcessor (in frames, stereo = 2x samples).
const CHUNK_FRAMES: usize = 1024;
const CHANNELS: u32 = 2;

/// Start the processing thread. Returns a stop flag handle.
#[allow(clippy::too_many_arguments)]
pub fn start_processing_thread(
    state: SharedStateHandle,
    source_audio: Arc<Vec<f32>>,
    mut working_audio: Vec<f32>,
    mut producer: RingProducer,
    sample_rate: u32,
    position: Arc<AtomicPosition>,
    stream_active: Arc<AtomicBool>,
    stop_flag: Arc<StopFlag>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut processor = build_processor(&state, sample_rate);
        let mut src_pos: usize = 0; // position in working_audio samples (interleaved)
        let chunk_samples = CHUNK_FRAMES * CHANNELS as usize;

        stream_active.store(true, Ordering::Relaxed);

        loop {
            if stop_flag.is_set() {
                break;
            }

            let (transport, stretch_ratio, seek_req, pitch_changed, preset_changed, pitch_semi) = {
                let mut st = state.lock().unwrap();
                let t = st.transport;
                let r = st.stretch_ratio;
                let s = st.seek_request.take();
                let pc = st.pitch_changed;
                let prc = st.preset_changed;
                let ps = st.pitch_semitones;
                if pc {
                    st.pitch_changed = false;
                }
                if prc {
                    st.preset_changed = false;
                }
                (t, r, s, pc, prc, ps)
            };

            // Handle preset change - rebuild processor
            if preset_changed {
                processor = build_processor(&state, sample_rate);
            }

            // Handle pitch change - re-process entire source
            if pitch_changed {
                {
                    let mut st = state.lock().unwrap();
                    st.pitch_processing = true;
                }

                if pitch_semi.abs() < 0.01 {
                    // No pitch shift, use original
                    working_audio = source_audio.as_ref().clone();
                } else {
                    let factor = 2.0_f64.powf(pitch_semi as f64 / 12.0);
                    let params = StretchParams::new(1.0)
                        .with_sample_rate(sample_rate)
                        .with_channels(CHANNELS);
                    match timestretch::pitch_shift(&source_audio, &params, factor) {
                        Ok(shifted) => {
                            working_audio = shifted;
                        }
                        Err(e) => {
                            log::error!("Pitch shift failed: {e}");
                            working_audio = source_audio.as_ref().clone();
                        }
                    }
                }

                // Reset processor and position
                processor.reset();
                // Clamp src_pos to new working audio length
                let max_pos = working_audio.len();
                if src_pos > max_pos {
                    src_pos = 0;
                }

                {
                    let mut st = state.lock().unwrap();
                    st.pitch_processing = false;
                    st.total_frames = working_audio.len() / CHANNELS as usize;
                }
            }

            // Handle seek
            if let Some(seek_frame) = seek_req {
                src_pos = (seek_frame * CHANNELS as usize).min(working_audio.len());
                processor.reset();
                // Drain ring buffer by waiting for it to empty
                // (the audio thread will consume what's there)
            }

            // Only process if playing
            if transport != Transport::Playing {
                thread::sleep(Duration::from_millis(10));
                continue;
            }

            // Update stretch ratio
            processor.set_stretch_ratio(stretch_ratio);

            // Check if we have audio to process
            if src_pos >= working_audio.len() {
                // End of audio
                // Flush remaining samples
                if let Ok(flushed) = processor.flush() {
                    push_to_ring(&mut producer, &flushed);
                }
                // Signal stop
                {
                    let mut st = state.lock().unwrap();
                    st.transport = Transport::Stopped;
                    st.position_frames = 0;
                }
                src_pos = 0;
                processor.reset();
                continue;
            }

            // Don't push if ring buffer is too full
            let available_space = producer.vacant_len();
            if available_space < chunk_samples * 4 {
                thread::sleep(Duration::from_millis(5));
                continue;
            }

            // Get next chunk
            let end = (src_pos + chunk_samples).min(working_audio.len());
            let chunk = &working_audio[src_pos..end];
            src_pos = end;

            // Update position
            let frame_pos = src_pos / CHANNELS as usize;
            position.store(frame_pos);

            // Process through StreamProcessor
            match processor.process(chunk) {
                Ok(output) => {
                    if !output.is_empty() {
                        push_to_ring(&mut producer, &output);
                    }
                }
                Err(e) => {
                    log::error!("Stream processing error: {e}");
                }
            }
        }

        stream_active.store(false, Ordering::Relaxed);
    })
}

fn build_processor(state: &SharedStateHandle, sample_rate: u32) -> StreamProcessor {
    let st = state.lock().unwrap();
    let mut params = StretchParams::new(st.stretch_ratio)
        .with_sample_rate(sample_rate)
        .with_channels(CHANNELS);
    if let Some(preset) = st.preset.to_edm_preset() {
        params = params.with_preset(preset);
    }
    StreamProcessor::new(params)
}

fn push_to_ring(producer: &mut RingProducer, data: &[f32]) {
    let mut offset = 0;
    while offset < data.len() {
        let pushed = producer.push_slice(&data[offset..]);
        if pushed == 0 {
            // Ring buffer full, yield
            thread::sleep(Duration::from_millis(1));
        }
        offset += pushed;
    }
}

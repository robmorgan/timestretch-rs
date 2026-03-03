use ringbuf::traits::{Observer, Producer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use timestretch::{QualityMode, StretchParams, StreamProcessor};

use crate::audio_engine::RingProducer;
use crate::state::{AtomicPosition, PresetChoice, SharedStateHandle, StopFlag, Transport};

/// Chunk size for feeding into StreamProcessor (in frames, stereo = 2x samples).
const CHUNK_FRAMES: usize = 1024;
const CHANNELS: u32 = 2;
/// Extra callback cushion to absorb scheduling jitter at stream start.
const START_PREROLL_CALLBACKS: usize = 2;

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
    flush_ring: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut processor = build_processor(&state, sample_rate);
        let mut src_pos: usize = 0; // position in working_audio samples (interleaved)
        let chunk_samples = CHUNK_FRAMES * CHANNELS as usize;
        let mut stream_started = false;
        let preroll_target_samples = startup_preroll_target_samples(&processor);

        // Keep output muted until enough stretched samples are buffered.
        stream_active.store(false, Ordering::Relaxed);

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

            // Handle preset change - rebuild processor and flush stale audio
            if preset_changed {
                processor = build_processor(&state, sample_rate);
                flush_ring.store(true, Ordering::Release);
                stream_active.store(false, Ordering::Relaxed);
                stream_started = false;
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
                        .with_channels(CHANNELS)
                        .with_normalize(true);
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

            // Handle seek - flush stale audio and re-enter preroll
            if let Some(seek_frame) = seek_req {
                src_pos = (seek_frame * CHANNELS as usize).min(working_audio.len());
                processor.reset();
                flush_ring.store(true, Ordering::Release);
                stream_active.store(false, Ordering::Relaxed);
                stream_started = false;
            }

            // Only process if playing
            if transport != Transport::Playing {
                thread::sleep(Duration::from_millis(10));
                continue;
            }

            // Update stretch ratio (set_stretch_ratio handles smooth interpolation).
            if let Err(err) = processor.set_stretch_ratio(stretch_ratio) {
                log::error!("Invalid stretch ratio {stretch_ratio}: {err}");
                thread::sleep(Duration::from_millis(5));
                continue;
            }

            // Check if we have audio to process
            if src_pos >= working_audio.len() {
                // End of audio
                // Flush remaining samples
                if let Ok(flushed) = processor.flush() {
                    push_to_ring(&mut producer, &flushed);
                }
                // Short clips can end before reaching the normal preroll target.
                // If we have anything buffered, start the stream so the tail can play.
                if !stream_started && producer.occupied_len() > 0 {
                    stream_started = true;
                    stream_active.store(true, Ordering::Relaxed);
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
                    if !stream_started && producer.occupied_len() >= preroll_target_samples {
                        stream_started = true;
                        stream_active.store(true, Ordering::Relaxed);
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

    if st.preset == PresetChoice::DjBeatmatch {
        let detected_bpm = st.detected_bpm;
        let target_bpm = if st.target_bpm.is_finite() && st.target_bpm > 0.0 {
            st.target_bpm
        } else {
            detected_bpm
        };
        if detected_bpm.is_finite() && detected_bpm > 0.0 && target_bpm.is_finite() && target_bpm > 0.0
        {
            if let Ok(mut processor) = StreamProcessor::try_from_tempo_low_latency(
                detected_bpm,
                target_bpm,
                sample_rate,
                CHANNELS,
            ) {
                if let Err(err) = processor.set_stretch_ratio(st.stretch_ratio) {
                    log::warn!(
                        "failed to apply slider ratio {} to low-latency DJ processor: {}",
                        st.stretch_ratio,
                        err
                    );
                }
                return processor;
            }
        }

        // Fallback to a low-latency profile if BPM metadata is unavailable.
        let params = StretchParams::new(st.stretch_ratio)
            .with_sample_rate(sample_rate)
            .with_channels(CHANNELS)
            .with_quality_mode(QualityMode::LowLatency)
            .with_fft_size(1024)
            .with_hop_size(256);
        return StreamProcessor::new(params);
    }

    let mut params = StretchParams::new(st.stretch_ratio)
        .with_sample_rate(sample_rate)
        .with_channels(CHANNELS)
        .with_normalize(true);
    if let Some(preset) = st.preset.to_edm_preset() {
        params = params.with_preset(preset);
    }
    StreamProcessor::new(params)
}

fn startup_preroll_target_samples(processor: &StreamProcessor) -> usize {
    let latency_frames = processor.latency_samples();
    let callback_cushion_samples = CHUNK_FRAMES * CHANNELS as usize * START_PREROLL_CALLBACKS;
    latency_frames
        .saturating_mul(CHANNELS as usize)
        .max(callback_cushion_samples)
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

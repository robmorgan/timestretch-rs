use ringbuf::traits::{Observer, Producer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use timestretch::{
    DualPlaneProcessor, QualityMode, RtConfig, StretchParams, TimeWarpMap,
};

use crate::audio_engine::RingProducer;
use crate::state::{AtomicPosition, PresetChoice, SharedStateHandle, StopFlag, Transport};

/// Fixed block size for dual-plane RT processing (in frames).
const CHUNK_FRAMES: usize = 1024;
const CHANNELS: u32 = 2;
/// Extra callback cushion to absorb scheduling jitter at stream start.
const START_PREROLL_CALLBACKS: usize = 2;
const RATIO_UPDATE_EPSILON: f64 = 1e-4;
const RATIO_WARP_HORIZON_FRAMES: usize = CHUNK_FRAMES * 16;
const MAX_OUTPUT_FRAMES_PER_CALL: usize = CHUNK_FRAMES * 4;

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
        let (mut processor, mut preroll_target_samples) = build_processor(&state, sample_rate);
        update_quality_tier_state(&state, processor.quality_tier());
        let mut last_ratio = f64::NAN;
        let mut src_pos: usize = 0; // position in working_audio samples (interleaved)
        let chunk_samples = CHUNK_FRAMES * CHANNELS as usize;
        let mut processed_chunk = Vec::with_capacity(chunk_samples * 6);
        let mut input_planar = vec![vec![0.0f32; CHUNK_FRAMES]; CHANNELS as usize];
        let mut output_planar = vec![vec![0.0f32; MAX_OUTPUT_FRAMES_PER_CALL]; CHANNELS as usize];
        let mut stream_started = false;

        // Keep output muted until enough stretched samples are buffered.
        stream_active.store(false, Ordering::Relaxed);

        loop {
            if stop_flag.is_set() {
                break;
            }

            let (
                transport,
                stretch_ratio,
                seek_req,
                pitch_changed,
                preset_changed,
                latency_profile_changed,
                pitch_semi,
            ) = {
                let mut st = state.lock().unwrap();
                let t = st.transport;
                let r = st.stretch_ratio;
                let s = st.seek_request.take();
                let pc = st.pitch_changed;
                let prc = st.preset_changed;
                let lpc = st.latency_profile_changed;
                let ps = st.pitch_semitones;
                if pc {
                    st.pitch_changed = false;
                }
                if prc {
                    st.preset_changed = false;
                }
                if lpc {
                    st.latency_profile_changed = false;
                }
                (t, r, s, pc, prc, lpc, ps)
            };

            // Handle preset change - rebuild processor and flush stale audio
            if preset_changed || latency_profile_changed {
                (processor, preroll_target_samples) = build_processor(&state, sample_rate);
                update_quality_tier_state(&state, processor.quality_tier());
                last_ratio = f64::NAN;
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

                // Rebuild processor to reset RT state.
                (processor, preroll_target_samples) = build_processor(&state, sample_rate);
                update_quality_tier_state(&state, processor.quality_tier());
                last_ratio = f64::NAN;
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
                (processor, preroll_target_samples) = build_processor(&state, sample_rate);
                update_quality_tier_state(&state, processor.quality_tier());
                last_ratio = f64::NAN;
                flush_ring.store(true, Ordering::Release);
                stream_active.store(false, Ordering::Relaxed);
                stream_started = false;
            }

            // Only process if playing
            if transport != Transport::Playing {
                thread::sleep(Duration::from_millis(10));
                continue;
            }

            // Update warp-map ratio snapshot.
            if let Err(err) = maybe_update_ratio_warp(&mut processor, &mut last_ratio, stretch_ratio)
            {
                log::error!("Invalid stretch ratio {stretch_ratio}: {err}");
                thread::sleep(Duration::from_millis(5));
                continue;
            }

            // Check if we have audio to process
            if src_pos >= working_audio.len() {
                // End of audio
                // Flush remaining samples
                let mut flushed = Vec::with_capacity(chunk_samples * 8);
                if let Ok(()) = processor.flush(&mut flushed) {
                    push_to_ring(&mut producer, &flushed);
                }
                update_quality_tier_state(&state, processor.quality_tier());
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
                (processor, preroll_target_samples) = build_processor(&state, sample_rate);
                update_quality_tier_state(&state, processor.quality_tier());
                last_ratio = f64::NAN;
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

            // Process through dual-plane processor.
            processed_chunk.clear();
            match process_input_chunk(
                &mut processor,
                chunk,
                &mut processed_chunk,
                &mut input_planar,
                &mut output_planar,
            ) {
                Ok(()) => {
                    if !processed_chunk.is_empty() {
                        push_to_ring(&mut producer, &processed_chunk);
                    }
                    if !stream_started && producer.occupied_len() >= preroll_target_samples {
                        stream_started = true;
                        stream_active.store(true, Ordering::Relaxed);
                    }
                    update_quality_tier_state(&state, processor.quality_tier());
                }
                Err(e) => {
                    log::error!("Stream processing error: {e}");
                }
            }
        }

        stream_active.store(false, Ordering::Relaxed);
    })
}

fn build_processor(state: &SharedStateHandle, sample_rate: u32) -> (DualPlaneProcessor, usize) {
    let st = state.lock().unwrap();
    let ratio = st.stretch_ratio;
    let latency_profile = st.latency_profile;

    if st.preset == PresetChoice::DjBeatmatch {
        let detected_bpm = st.detected_bpm;
        let target_bpm = if st.target_bpm.is_finite() && st.target_bpm > 0.0 {
            st.target_bpm
        } else {
            detected_bpm
        };
        if detected_bpm.is_finite()
            && detected_bpm > 0.0
            && target_bpm.is_finite()
            && target_bpm > 0.0
        {
            let base_ratio = detected_bpm / target_bpm;
            let params = StretchParams::new(base_ratio)
                .with_sample_rate(sample_rate)
                .with_channels(CHANNELS)
                .with_quality_mode(QualityMode::LowLatency)
                .with_window_type(timestretch::WindowType::Hann)
                .with_fft_size(1024)
                .with_hop_size(256);

            let mut cfg = RtConfig::new(params.clone(), CHUNK_FRAMES);
            cfg.latency_profile = latency_profile;
            let mut processor =
                DualPlaneProcessor::prepare(cfg).expect("valid dual-plane RT config for desktop DJ");
            if let Err(err) = force_ratio_warp(&mut processor, ratio) {
                log::warn!("failed to apply ratio {ratio} to low-latency DJ processor: {err}");
            }
            let preroll = startup_preroll_target_samples(&params);
            return (processor, preroll);
        }

        // Fallback to a low-latency profile if BPM metadata is unavailable.
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(CHANNELS)
            .with_quality_mode(QualityMode::LowLatency)
            .with_fft_size(1024)
            .with_hop_size(256);
        let mut cfg = RtConfig::new(params.clone(), CHUNK_FRAMES);
        cfg.latency_profile = latency_profile;
        let mut processor =
            DualPlaneProcessor::prepare(cfg).expect("valid dual-plane RT config fallback");
        if let Err(err) = force_ratio_warp(&mut processor, ratio) {
            log::warn!("failed to apply ratio {ratio} in fallback processor: {err}");
        }
        let preroll = startup_preroll_target_samples(&params);
        return (processor, preroll);
    }

    let mut params = StretchParams::new(ratio)
        .with_sample_rate(sample_rate)
        .with_channels(CHANNELS)
        .with_normalize(true);
    if let Some(preset) = st.preset.to_edm_preset() {
        params = params.with_preset(preset);
    }
    let mut cfg = RtConfig::new(params.clone(), CHUNK_FRAMES);
    cfg.latency_profile = latency_profile;
    let mut processor = DualPlaneProcessor::prepare(cfg).expect("valid dual-plane RT config");
    if let Err(err) = force_ratio_warp(&mut processor, ratio) {
        log::warn!("failed to apply initial ratio {ratio}: {err}");
    }
    let preroll = startup_preroll_target_samples(&params);
    (processor, preroll)
}

fn startup_preroll_target_samples(params: &StretchParams) -> usize {
    let latency_frames = params.fft_size.saturating_mul(3) / 2;
    let callback_cushion_samples = CHUNK_FRAMES * CHANNELS as usize * START_PREROLL_CALLBACKS;
    latency_frames
        .saturating_mul(CHANNELS as usize)
        .max(callback_cushion_samples)
}

fn maybe_update_ratio_warp(
    processor: &mut DualPlaneProcessor,
    last_ratio: &mut f64,
    ratio: f64,
) -> Result<(), timestretch::StretchError> {
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(timestretch::StretchError::InvalidRatio(format!(
            "stretch ratio must be finite and > 0.0, got {ratio}"
        )));
    }
    if (*last_ratio - ratio).abs() <= RATIO_UPDATE_EPSILON {
        return Ok(());
    }
    force_ratio_warp(processor, ratio)?;
    *last_ratio = ratio;
    Ok(())
}

fn force_ratio_warp(
    processor: &mut DualPlaneProcessor,
    ratio: f64,
) -> Result<(), timestretch::StretchError> {
    let map = Arc::new(TimeWarpMap::from_ratio(ratio, RATIO_WARP_HORIZON_FRAMES)?);
    if !processor.publish_warp_map(map.clone()) {
        processor.rt_mut().set_warp_map_snapshot(map);
    }
    Ok(())
}

fn process_input_chunk(
    processor: &mut DualPlaneProcessor,
    input: &[f32],
    output: &mut Vec<f32>,
    input_planar: &mut [Vec<f32>],
    output_planar: &mut [Vec<f32>],
) -> Result<(), timestretch::StretchError> {
    if input.is_empty() {
        return Ok(());
    }
    if input_planar.len() != CHANNELS as usize || output_planar.len() != CHANNELS as usize {
        return Err(timestretch::StretchError::InvalidFormat(
            "desktop planar scratch channel count mismatch".to_string(),
        ));
    }

    let block_samples = CHUNK_FRAMES * CHANNELS as usize;
    let mut offset = 0usize;
    while offset < input.len() {
        let take = (input.len() - offset).min(block_samples);
        let frames = take / CHANNELS as usize;
        if frames == 0 {
            break;
        }

        if input_planar[0].len() < frames || output_planar[0].is_empty() {
            return Err(timestretch::StretchError::InvalidFormat(
                "desktop planar scratch capacity mismatch".to_string(),
            ));
        }

        for frame in 0..frames {
            let base = offset + frame * CHANNELS as usize;
            input_planar[0][frame] = input[base];
            input_planar[1][frame] = input[base + 1];
        }

        let input_refs = [&input_planar[0][..frames], &input_planar[1][..frames]];
        let (left_out, right_out) = output_planar.split_at_mut(1);
        let mut output_refs = [left_out[0].as_mut_slice(), right_out[0].as_mut_slice()];

        let (_consumed, produced_frames) = processor.process(&input_refs, &mut output_refs);
        if produced_frames > 0 {
            for frame in 0..produced_frames {
                output.push(output_refs[0][frame]);
                output.push(output_refs[1][frame]);
            }
        }

        offset += frames * CHANNELS as usize;
    }
    Ok(())
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

fn update_quality_tier_state(state: &SharedStateHandle, tier: timestretch::QualityTier) {
    state.lock().unwrap().current_quality_tier = tier;
}

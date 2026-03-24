use ringbuf::traits::{Observer, Producer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use timestretch::{EdmPreset, QualityMode, StreamProcessor, StretchError, StretchParams};

use crate::audio_engine::RingProducer;
use crate::state::{
    AtomicPosition, PresetChoice, SharedStateHandle, StopFlag, StreamProfile, Transport,
};

/// Fixed chunk size for desktop stream processing (in frames).
const CHUNK_FRAMES: usize = 1024;
const CHANNELS: u32 = 2;
/// Extra callback cushion to absorb scheduling jitter at stream start.
const START_PREROLL_CALLBACKS: usize = 2;
const RATIO_UPDATE_EPSILON: f64 = 1e-4;

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
        let mut last_ratio = f64::NAN;
        let mut src_pos: usize = 0;
        let chunk_samples = CHUNK_FRAMES * CHANNELS as usize;
        let mut processed_chunk = Vec::with_capacity(chunk_samples * 6);
        let mut stream_started = false;

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

            if preset_changed {
                (processor, preroll_target_samples) = build_processor(&state, sample_rate);
                last_ratio = f64::NAN;
                flush_ring.store(true, Ordering::Release);
                stream_active.store(false, Ordering::Relaxed);
                stream_started = false;
            }

            if pitch_changed {
                {
                    let mut st = state.lock().unwrap();
                    st.pitch_processing = true;
                }

                if pitch_semi.abs() < 0.01 {
                    working_audio = source_audio.as_ref().clone();
                } else {
                    let factor = 2.0_f64.powf(pitch_semi as f64 / 12.0);
                    let params = StretchParams::new(1.0)
                        .with_sample_rate(sample_rate)
                        .with_channels(CHANNELS)
                        .with_normalize(true);
                    match timestretch::pitch_shift(&source_audio, &params, factor) {
                        Ok(shifted) => working_audio = shifted,
                        Err(e) => {
                            log::error!("Pitch shift failed: {e}");
                            working_audio = source_audio.as_ref().clone();
                        }
                    }
                }

                (processor, preroll_target_samples) = build_processor(&state, sample_rate);
                last_ratio = f64::NAN;
                if src_pos > working_audio.len() {
                    src_pos = 0;
                }

                {
                    let mut st = state.lock().unwrap();
                    st.pitch_processing = false;
                    st.total_frames = working_audio.len() / CHANNELS as usize;
                }
            }

            if let Some(seek_frame) = seek_req {
                src_pos = (seek_frame * CHANNELS as usize).min(working_audio.len());
                (processor, preroll_target_samples) = build_processor(&state, sample_rate);
                last_ratio = f64::NAN;
                flush_ring.store(true, Ordering::Release);
                stream_active.store(false, Ordering::Relaxed);
                stream_started = false;
            }

            if transport != Transport::Playing {
                thread::sleep(Duration::from_millis(10));
                continue;
            }

            if let Err(err) = maybe_update_ratio(&mut processor, &mut last_ratio, stretch_ratio) {
                log::error!("Invalid stretch ratio {stretch_ratio}: {err}");
                thread::sleep(Duration::from_millis(5));
                continue;
            }

            if src_pos >= working_audio.len() {
                let mut flushed = Vec::new();
                reserve_flush_output_capacity(&processor, &mut flushed);
                if let Ok(_written) = processor.flush_into(&mut flushed) {
                    push_to_ring(&mut producer, &flushed);
                }
                if !stream_started && producer.occupied_len() > 0 {
                    stream_started = true;
                    stream_active.store(true, Ordering::Relaxed);
                }
                {
                    let mut st = state.lock().unwrap();
                    st.transport = Transport::Stopped;
                    st.position_frames = 0;
                }
                src_pos = 0;
                (processor, preroll_target_samples) = build_processor(&state, sample_rate);
                last_ratio = f64::NAN;
                continue;
            }

            let available_space = producer.vacant_len();
            if available_space < chunk_samples * 4 {
                thread::sleep(Duration::from_millis(5));
                continue;
            }

            let end = (src_pos + chunk_samples).min(working_audio.len());
            let chunk = &working_audio[src_pos..end];
            src_pos = end;

            let frame_pos = src_pos / CHANNELS as usize;
            position.store(frame_pos);

            processed_chunk.clear();
            match process_input_chunk(&mut processor, chunk, &mut processed_chunk) {
                Ok(()) => {
                    if !processed_chunk.is_empty() {
                        push_to_ring(&mut producer, &processed_chunk);
                    }
                    if !stream_started && producer.occupied_len() >= preroll_target_samples {
                        stream_started = true;
                        stream_active.store(true, Ordering::Relaxed);
                    }
                }
                Err(e) => log::error!("Stream processing error: {e}"),
            }
        }

        stream_active.store(false, Ordering::Relaxed);
    })
}

fn build_processor(state: &SharedStateHandle, sample_rate: u32) -> (StreamProcessor, usize) {
    let st = state.lock().unwrap();
    let ratio = st.stretch_ratio;
    let params = desktop_stream_params(&st, sample_rate);
    drop(st);

    let preroll = startup_preroll_target_samples(&params);
    let mut processor = StreamProcessor::new(params);
    if let Err(err) = processor.set_stretch_ratio(ratio) {
        log::warn!("failed to apply initial ratio {ratio}: {err}");
    }

    (processor, preroll)
}

fn desktop_stream_params(
    st: &crate::state::SharedState,
    sample_rate: u32,
) -> StretchParams {
    if st.preset == PresetChoice::DjBeatmatch {
        let detected_bpm = st.detected_bpm;
        let target_bpm = if st.target_bpm.is_finite() && st.target_bpm > 0.0 {
            st.target_bpm
        } else {
            detected_bpm
        };
        let base_ratio = if detected_bpm.is_finite()
            && detected_bpm > 0.0
            && target_bpm.is_finite()
            && target_bpm > 0.0
        {
            detected_bpm / target_bpm
        } else {
            st.stretch_ratio
        };

        let mut params = match st.stream_profile {
            // Lowest-latency profile for responsive live control.
            StreamProfile::Live => StretchParams::new(base_ratio)
                .with_sample_rate(sample_rate)
                .with_channels(CHANNELS)
                .with_quality_mode(QualityMode::LowLatency)
                .with_fft_size(1024)
                .with_hop_size(256)
                .with_normalize(true),
            // Higher-quality realtime profile: larger FFT and full DJ preset
            // reduce robotic PV character at the cost of more latency/CPU.
            StreamProfile::Quality => StretchParams::new(base_ratio)
                .with_sample_rate(sample_rate)
                .with_channels(CHANNELS)
                .with_preset(EdmPreset::DjBeatmatch)
                .with_normalize(true),
        };
        if detected_bpm.is_finite() && detected_bpm > 0.0 {
            params = params.with_bpm(detected_bpm);
        }
        return params;
    }

    let mut params = StretchParams::new(st.stretch_ratio)
        .with_sample_rate(sample_rate)
        .with_channels(CHANNELS)
        .with_normalize(true);
    if let Some(preset) = st.preset.to_edm_preset() {
        params = params.with_preset(preset);
    }
    if st.detected_bpm.is_finite() && st.detected_bpm > 0.0 {
        params = params.with_bpm(st.detected_bpm);
    }
    params
}

fn startup_preroll_target_samples(params: &StretchParams) -> usize {
    let latency_frames = params.fft_size.saturating_mul(3) / 2;
    let callback_cushion_samples = CHUNK_FRAMES * CHANNELS as usize * START_PREROLL_CALLBACKS;
    latency_frames
        .saturating_mul(CHANNELS as usize)
        .max(callback_cushion_samples)
}

fn maybe_update_ratio(
    processor: &mut StreamProcessor,
    last_ratio: &mut f64,
    ratio: f64,
) -> Result<(), StretchError> {
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(StretchError::InvalidRatio(format!(
            "stretch ratio must be finite and > 0.0, got {ratio}"
        )));
    }
    if (*last_ratio - ratio).abs() <= RATIO_UPDATE_EPSILON {
        return Ok(());
    }
    processor.set_stretch_ratio(ratio)?;
    *last_ratio = ratio;
    Ok(())
}

fn process_input_chunk(
    processor: &mut StreamProcessor,
    input: &[f32],
    output: &mut Vec<f32>,
) -> Result<(), StretchError> {
    if input.is_empty() {
        return Ok(());
    }

    reserve_process_output_capacity(processor, input.len(), output);
    processor.process_into(input, output)
}

fn reserve_process_output_capacity(
    processor: &StreamProcessor,
    input_len: usize,
    output: &mut Vec<f32>,
) {
    let ratio_hint = processor
        .current_stretch_ratio()
        .max(processor.target_stretch_ratio())
        .max(1.0);
    let (_, _, _, pending_capacity) = processor.capacities();
    let required = ((input_len as f64) * ratio_hint).ceil() as usize + pending_capacity;
    let available = output.capacity().saturating_sub(output.len());
    if required > available {
        output.reserve(required - available);
    }
}

fn reserve_flush_output_capacity(processor: &StreamProcessor, output: &mut Vec<f32>) {
    let (_, _, _, pending_capacity) = processor.capacities();
    let required = pending_capacity.saturating_mul(2);
    let available = output.capacity().saturating_sub(output.len());
    if required > available {
        output.reserve(required - available);
    }
}

fn push_to_ring(producer: &mut RingProducer, data: &[f32]) {
    let mut offset = 0;
    while offset < data.len() {
        let pushed = producer.push_slice(&data[offset..]);
        if pushed == 0 {
            thread::sleep(Duration::from_millis(1));
        }
        offset += pushed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::SharedState;

    #[test]
    fn desktop_dj_beatmatch_uses_low_latency_stream_profile() {
        let mut state = SharedState::new();
        state.preset = PresetChoice::DjBeatmatch;
        state.stream_profile = StreamProfile::Live;
        state.detected_bpm = 126.0;
        state.target_bpm = 128.0;

        let params = desktop_stream_params(&state, 44_100);

        assert_eq!(params.quality_mode, QualityMode::LowLatency);
        assert_eq!(params.fft_size, 1024);
        assert_eq!(params.hop_size, 256);
        assert!(params.normalize);
        assert!(params.preset.is_none());
        assert_eq!(params.bpm, Some(126.0));
        assert!((params.stretch_ratio - (126.0 / 128.0)).abs() < 1e-9);
    }

    #[test]
    fn desktop_dj_quality_profile_uses_full_preset() {
        let mut state = SharedState::new();
        state.preset = PresetChoice::DjBeatmatch;
        state.stream_profile = StreamProfile::Quality;
        state.detected_bpm = 126.0;
        state.target_bpm = 128.0;

        let params = desktop_stream_params(&state, 44_100);

        assert_eq!(params.quality_mode, QualityMode::Balanced);
        assert_eq!(params.preset, Some(EdmPreset::DjBeatmatch));
        assert_eq!(params.fft_size, 4096);
        assert_eq!(params.hop_size, 1024);
        assert_eq!(params.bpm, Some(126.0));
        assert!(params.normalize);
        assert!((params.stretch_ratio - (126.0 / 128.0)).abs() < 1e-9);
    }

    #[test]
    fn desktop_non_dj_presets_keep_standard_profile() {
        let mut state = SharedState::new();
        state.preset = PresetChoice::HouseLoop;
        state.stretch_ratio = 1.08;
        state.detected_bpm = 124.0;

        let params = desktop_stream_params(&state, 48_000);

        assert_eq!(params.quality_mode, QualityMode::Balanced);
        assert_eq!(params.preset, Some(EdmPreset::HouseLoop));
        assert_eq!(params.fft_size, 4096);
        assert_eq!(params.hop_size, 1024);
        assert_eq!(params.bpm, Some(124.0));
        assert!(params.normalize);
    }
}

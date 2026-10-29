//! Desktop scheduling only. StreamSession owns demand, EOF, seek and timelines.
use crate::output_queue::Producer;
use crate::state::{
    AtomicPosition, ScrubPhase, ScrubState, SharedStateHandle, StopFlag, Transport,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;
use timestretch::{PreparedSession, SessionConfig, SourceAccess};

fn request_priority(
    state: &SharedStateHandle,
    rate: u32,
    reserve: usize,
) -> Option<crate::worker_priority::AudioPriority> {
    use crate::worker_priority::AudioPriority;
    let result = AudioPriority::new(rate, reserve);
    let mut state_guard = state.try_lock().ok();
    match result {
        Ok(guard) => {
            if let Some(st) = state_guard.as_mut() {
                st.worker_scheduling = AudioPriority::label();
                st.worker_scheduling_error = None;
            }
            Some(guard)
        }
        Err(error) => {
            log::warn!("Audio worker priority unavailable: {error}");
            if let Some(st) = state_guard.as_mut() {
                st.worker_scheduling = "platform default (promotion failed)";
                st.worker_scheduling_error = Some(error);
            }
            None
        }
    }
}

pub fn start_deck_thread(
    state: SharedStateHandle,
    source_audio: Arc<Vec<f32>>,
    mut output: Producer,
    position: Arc<AtomicPosition>,
    stream_active: Arc<AtomicBool>,
    stop_flag: Arc<StopFlag>,
    scrub: Arc<ScrubState>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut config = {
            let mut st = state.lock().unwrap();
            let parameters = match st.deck_engine.parameters(st.stretch_ratio, st.pitch) {
                Ok(parameters) => parameters,
                Err(error) => {
                    st.processing_error = Some(error.to_string());
                    st.transport = Transport::Stopped;
                    return;
                }
            };
            SessionConfig {
                source_sample_rate: st.sample_rate,
                output_sample_rate: st.output_sample_rate,
                channels: 2,
                backend: st.backend,
                source_access: SourceAccess::Seekable,
                source_frames: Some((source_audio.len() / 2) as u64),
                parameters,
                ..Default::default()
            }
        };
        let mut session = match PreparedSession::new(config.clone()) {
            Ok(p) => p.activate(),
            Err(e) => {
                let mut st = state.lock().unwrap();
                st.processing_error = Some(e.to_string());
                st.transport = Transport::Stopped;
                return;
            }
        };
        let reserve = crate::render_worker::output_reserve_frames(config.output_sample_rate);
        let mut priority = None;
        let mut priority_requested = false;
        let mut backend_request = config.backend;
        let mut last_params = session.applied_parameters().requested;
        let mut last_loop = None;
        let mut command = 0;
        let mut primed = false;
        let mut finished_published = false;
        let mut scratch = vec![0.0; 2048];
        let mut controls = {
            let mut st = state.lock().unwrap();
            session.set_transport(st.transport);
            let target = st.seek_request.take().unwrap_or(st.position_frames);
            if target > 0 {
                let _ = session.seek(target as u64);
            }
            (
                st.transport,
                st.stretch_ratio,
                st.loop_region,
                st.deck_engine,
                st.backend,
                st.pitch,
            )
        };
        output.publish_epoch(session.epoch());
        'worker: while !stop_flag.is_set() {
            if let Some(receipt) = output.receipt() {
                let _ = session.acknowledge(receipt);
            }
            // UI contention delays new intent/telemetry, never media work.
            // Already observed controls remain valid until the next snapshot.
            let seek = if let Ok(mut st) = state.try_lock() {
                controls = (
                    st.transport,
                    st.stretch_ratio,
                    st.loop_region,
                    st.deck_engine,
                    st.backend,
                    st.pitch,
                );
                st.seek_request.take()
            } else {
                None
            };
            let (transport, ratio, loop_region, mode, backend, pitch) = controls;
            let previous_epoch = session.epoch();
            let result = (|| -> Result<(), timestretch::SessionError> {
                session.set_transport(transport);
                if backend != backend_request {
                    stream_active.store(false, Ordering::Release);
                    // Preparation and destruction use normal thread policy.
                    drop(priority.take());
                    priority_requested = false;
                    config.backend = backend;
                    let prepared = PreparedSession::new(config.clone())?;
                    if let Some(receipt) = output.receipt() {
                        session.acknowledge(receipt)?;
                    }
                    drop(session.replace_prepared(prepared)?);
                    backend_request = backend;
                }
                let params = mode.parameters(ratio, pitch)?;
                let region = loop_region.map(|(a, b)| (a as u64, b as u64));
                if region != last_loop {
                    session.set_loop(region)?;
                    last_loop = region;
                }
                if let Some(target) = seek {
                    session.seek(target as u64)?;
                }
                // Epoch transactions cancel queued controls. Reissue current UI
                // intent afterwards, even if that intent has not changed.
                if params != last_params || previous_epoch != session.epoch() {
                    command += 1;
                    session.schedule(command, session.commit_horizon(), params)?;
                    last_params = params;
                }
                Ok(())
            })();
            if let Err(error) = result {
                stream_active.store(false, Ordering::Release);
                drop(priority.take());
                let mut st = state.lock().unwrap();
                st.processing_error = Some(error.to_string());
                st.transport = Transport::Stopped;
                stream_active.store(false, Ordering::Release);
                break;
            }
            if previous_epoch != session.epoch() {
                stream_active.store(false, Ordering::Release);
                output.publish_epoch(session.epoch());
                primed = false;
                finished_published = false;
            }
            if transport != Transport::Playing || scrub.phase() == ScrubPhase::Active {
                stream_active.store(false, Ordering::Release);
                drop(priority.take());
                priority_requested = false;
            } else {
                if !priority_requested {
                    priority = request_priority(&state, config.output_sample_rate, reserve);
                    priority_requested = true;
                }
                // Work quota and reserve are host policy. Source requests and
                // processing boundaries come entirely from the library.
                if let Err(e) = crate::render_worker::fill_reserve(
                    &mut session,
                    &source_audio,
                    &mut output,
                    &mut scratch,
                    reserve,
                    8,
                ) {
                    stream_active.store(false, Ordering::Release);
                    drop(priority.take());
                    let mut st = state.lock().unwrap();
                    st.processing_error = Some(e.to_string());
                    st.transport = Transport::Stopped;
                    break 'worker;
                }
                if output.occupied_frames() >= reserve
                    || (session.rendered_finished() && output.occupied_frames() > 0)
                {
                    primed = true;
                }
                stream_active.store(primed && !session.finished(), Ordering::Release);
            }
            if let Some(e) = session.fault() {
                stream_active.store(false, Ordering::Release);
                drop(priority.take());
                let mut st = state.lock().unwrap();
                st.processing_error = Some(e.to_string());
                st.transport = Transport::Stopped;
                stream_active.store(false, Ordering::Release);
                break;
            }
            let timing = session.timing();
            if let Ok(mut st) = state.try_lock() {
                st.worker_scheduling = if priority.is_some() {
                    crate::worker_priority::AudioPriority::label()
                } else if priority_requested {
                    "platform default (promotion failed)"
                } else {
                    "inactive"
                };
                st.resolved_backend = session.config().backend;
                st.native_sample_rate = timing.native_sample_rate;
                st.achieved_tempo = session.applied_parameters().tempo();
                st.achieved_pitch = session.applied_parameters().achieved_pitch;
                st.reported_latency_secs = Some(reserve as f64 / config.output_sample_rate as f64);
                st.reported_control_latency_secs = (timing.control_quantum_frames
                    + timing.pending_output_frames
                    + timing.unacknowledged_output_frames)
                    as f64
                    / config.output_sample_rate as f64;
                st.underrun_frames = output.underrun_frames();
                if scrub.phase() == ScrubPhase::Idle {
                    position.store(session.position() as usize);
                }
                if session.finished() && !finished_published {
                    // Rewinding the UI after delivered EOF is product policy.
                    st.transport = Transport::Stopped;
                    st.position_frames = 0;
                    position.store(0);
                    stream_active.store(false, Ordering::Release);
                    finished_published = true;
                }
            }
            thread::sleep(Duration::from_millis(1));
        }
        stream_active.store(false, Ordering::Release);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::SharedState;
    use std::sync::Mutex;
    use std::time::Instant;

    #[test]
    fn ui_lock_cannot_block_running_media_work() {
        let mut initial = SharedState::new();
        initial.transport = Transport::Playing;
        let state = Arc::new(Mutex::new(initial));
        let source = Arc::new((0..88200).map(|i| (i as f32 * 0.031).sin() * 0.2).collect());
        let (producer, mut consumer) = crate::output_queue::channel(32768);
        let active = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(StopFlag::new());
        let worker = start_deck_thread(
            state.clone(),
            source,
            producer,
            Arc::new(AtomicPosition::new()),
            active.clone(),
            stop.clone(),
            Arc::new(ScrubState::new()),
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        while !active.load(Ordering::Acquire) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        let primed = active.load(Ordering::Acquire);
        let ui = state.lock().unwrap();
        let target = crate::render_worker::output_reserve_frames(44100) * 3;
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut received = 0;
        let mut samples = [0.0; 256];
        while primed && received < target && Instant::now() < deadline {
            consumer.sync_epoch();
            let n = consumer.read(&mut samples);
            consumer.acknowledge_frames(n);
            received += n;
            thread::yield_now();
        }
        // Always release the UI and join, including when an assertion fails.
        drop(ui);
        stop.set();
        worker.join().unwrap();
        assert!(primed, "worker failed to prime");
        assert!(
            received >= target,
            "UI lock stalled delivery at {received} frames"
        );
    }
}

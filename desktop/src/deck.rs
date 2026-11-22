//! Desktop scheduling only. StreamSession owns demand, EOF, seek and timelines.
use crate::output_queue::Producer;
use crate::state::{AtomicPosition, ScrubPhase, ScrubState, SharedStateHandle, Transport};
use crate::worker_priority::AudioPriority;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use timestretch::{
    Backend, Parameters, PitchRatio, PreparedSession, SessionConfig, SourceAccess, StreamSession,
};

fn request_priority(state: &SharedStateHandle, rate: u32, reserve: usize) -> Option<AudioPriority> {
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

/// UI intent last observed by the worker.
type Controls = (
    Transport,
    f64,
    Option<(usize, usize)>,
    crate::state::DeckEngine,
    Backend,
    PitchRatio,
);

/// The deck's render pump as a resumable object: [`DeckWorker::step`] is one
/// pass of the worker loop. Natively a dedicated thread steps it (see
/// [`start_deck_thread`]); in the browser, which has no threads, the page's
/// main loop does. It must stay on the thread that created it.
pub struct DeckWorker {
    state: SharedStateHandle,
    source_audio: Arc<Vec<f32>>,
    output: Producer,
    position: Arc<AtomicPosition>,
    stream_active: Arc<AtomicBool>,
    scrub: Arc<ScrubState>,
    config: SessionConfig,
    session: StreamSession,
    reserve: usize,
    priority: Option<AudioPriority>,
    priority_requested: bool,
    backend_request: Backend,
    last_params: Parameters,
    last_loop: Option<(u64, u64)>,
    command: u64,
    primed: bool,
    finished_published: bool,
    scratch: Vec<f32>,
    controls: Controls,
}

impl DeckWorker {
    /// Prepares the session from the current UI state. On failure the error
    /// is published to the shared state and `None` is returned.
    pub fn new(
        state: SharedStateHandle,
        source_audio: Arc<Vec<f32>>,
        mut output: Producer,
        position: Arc<AtomicPosition>,
        stream_active: Arc<AtomicBool>,
        scrub: Arc<ScrubState>,
    ) -> Option<Self> {
        let config = {
            let mut st = state.lock().unwrap();
            let parameters = match st.deck_engine.parameters(st.stretch_ratio, st.pitch) {
                Ok(parameters) => parameters,
                Err(error) => {
                    st.processing_error = Some(error.to_string());
                    st.transport = Transport::Stopped;
                    return None;
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
                return None;
            }
        };
        let reserve = crate::render_worker::output_reserve_frames(config.output_sample_rate);
        let controls = {
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
        Some(Self {
            state,
            source_audio,
            output,
            position,
            stream_active,
            scrub,
            backend_request: config.backend,
            last_params: session.applied_parameters().requested,
            config,
            session,
            reserve,
            priority: None,
            priority_requested: false,
            last_loop: None,
            command: 0,
            primed: false,
            finished_published: false,
            scratch: vec![0.0; 2048],
            controls,
        })
    }

    /// Publishes a fatal processing error and silences delivery.
    fn fail(&mut self, error: timestretch::SessionError) {
        self.stream_active.store(false, Ordering::Release);
        drop(self.priority.take());
        let mut st = self.state.lock().unwrap();
        st.processing_error = Some(error.to_string());
        st.transport = Transport::Stopped;
    }

    /// One pass of the worker loop: acknowledge delivery, apply UI intent,
    /// refill the reserve and publish telemetry. Returns `false` once the
    /// worker has failed and must not be stepped again.
    pub fn step(&mut self) -> bool {
        if let Some(receipt) = self.output.receipt() {
            let _ = self.session.acknowledge(receipt);
        }
        // UI contention delays new intent/telemetry, never media work.
        // Already observed controls remain valid until the next snapshot.
        let seek = if let Ok(mut st) = self.state.try_lock() {
            self.controls = (
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
        let (transport, ratio, loop_region, mode, backend, pitch) = self.controls;
        let previous_epoch = self.session.epoch();
        let result = (|| -> Result<(), timestretch::SessionError> {
            self.session.set_transport(transport);
            if backend != self.backend_request {
                self.stream_active.store(false, Ordering::Release);
                // Preparation and destruction use normal thread policy.
                drop(self.priority.take());
                self.priority_requested = false;
                self.config.backend = backend;
                let prepared = PreparedSession::new(self.config.clone())?;
                if let Some(receipt) = self.output.receipt() {
                    self.session.acknowledge(receipt)?;
                }
                drop(self.session.replace_prepared(prepared)?);
                self.backend_request = backend;
            }
            let params = mode.parameters(ratio, pitch)?;
            let region = loop_region.map(|(a, b)| (a as u64, b as u64));
            if region != self.last_loop {
                self.session.set_loop(region)?;
                self.last_loop = region;
            }
            if let Some(target) = seek {
                self.session.seek(target as u64)?;
            }
            // Epoch transactions cancel queued controls. Reissue current UI
            // intent afterwards, even if that intent has not changed.
            if params != self.last_params || previous_epoch != self.session.epoch() {
                self.command += 1;
                self.session
                    .schedule(self.command, self.session.commit_horizon(), params)?;
                self.last_params = params;
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.fail(error);
            return false;
        }
        if previous_epoch != self.session.epoch() {
            self.stream_active.store(false, Ordering::Release);
            self.output.publish_epoch(self.session.epoch());
            self.primed = false;
            self.finished_published = false;
        }
        if transport != Transport::Playing || self.scrub.phase() == ScrubPhase::Active {
            self.stream_active.store(false, Ordering::Release);
            drop(self.priority.take());
            self.priority_requested = false;
        } else {
            if !self.priority_requested {
                self.priority =
                    request_priority(&self.state, self.config.output_sample_rate, self.reserve);
                self.priority_requested = true;
            }
            // Work quota and reserve are host policy. Source requests and
            // processing boundaries come entirely from the library.
            if let Err(e) = crate::render_worker::fill_reserve(
                &mut self.session,
                &self.source_audio,
                &mut self.output,
                &mut self.scratch,
                self.reserve,
                8,
            ) {
                self.fail(e);
                return false;
            }
            if self.output.occupied_frames() >= self.reserve
                || (self.session.rendered_finished() && self.output.occupied_frames() > 0)
            {
                self.primed = true;
            }
            self.stream_active
                .store(self.primed && !self.session.finished(), Ordering::Release);
        }
        if let Some(e) = self.session.fault() {
            self.fail(e);
            return false;
        }
        let timing = self.session.timing();
        let rate = self.config.output_sample_rate;
        if let Ok(mut st) = self.state.try_lock() {
            st.worker_scheduling = if self.priority.is_some() {
                AudioPriority::label()
            } else if self.priority_requested {
                "platform default (promotion failed)"
            } else {
                "inactive"
            };
            st.resolved_backend = self.session.config().backend;
            st.native_sample_rate = timing.native_sample_rate;
            st.achieved_tempo = self.session.applied_parameters().tempo();
            st.achieved_pitch = self.session.applied_parameters().achieved_pitch;
            st.reported_latency_secs = Some(self.reserve as f64 / rate as f64);
            st.reported_control_latency_secs = (timing.control_quantum_frames
                + timing.pending_output_frames
                + timing.unacknowledged_output_frames)
                as f64
                / rate as f64;
            st.underrun_frames = self.output.underrun_frames();
            if self.scrub.phase() == ScrubPhase::Idle {
                self.position.store(self.session.position() as usize);
            }
            if self.session.finished() && !self.finished_published {
                // Rewinding the UI after delivered EOF is product policy.
                st.transport = Transport::Stopped;
                st.position_frames = 0;
                self.position.store(0);
                self.stream_active.store(false, Ordering::Release);
                self.finished_published = true;
            }
        }
        true
    }
}

/// Runs a [`DeckWorker`] on its own thread until `stop_flag` is set.
#[cfg(not(target_arch = "wasm32"))]
pub fn start_deck_thread(
    state: SharedStateHandle,
    source_audio: Arc<Vec<f32>>,
    output: Producer,
    position: Arc<AtomicPosition>,
    stream_active: Arc<AtomicBool>,
    stop_flag: Arc<crate::state::StopFlag>,
    scrub: Arc<ScrubState>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let active = stream_active.clone();
        if let Some(mut worker) =
            DeckWorker::new(state, source_audio, output, position, stream_active, scrub)
        {
            while !stop_flag.is_set() && worker.step() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        active.store(false, Ordering::Release);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{SharedState, StopFlag};
    use std::sync::Mutex;
    use std::thread;
    use std::time::{Duration, Instant};

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

//! Platform hosting for one playing deck: who steps the [`crate::deck::DeckWorker`] and
//! who delivers audio. Natively that is a worker thread plus a cpal device
//! callback; in the browser both run on the page's main thread, driven by a
//! timer. Media semantics stay in the library either way.
use crate::output_queue::Consumer;
use crate::output_queue::Producer;
use crate::state::{AtomicPosition, AtomicVolume, ScrubState, SharedStateHandle};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Shared handles a playing deck needs, bundled for [`Playback::start`].
pub struct PlaybackParts {
    pub state: SharedStateHandle,
    pub source: Arc<Vec<f32>>,
    pub producer: Producer,
    pub consumer: Consumer,
    pub position: Arc<AtomicPosition>,
    pub stream_active: Arc<AtomicBool>,
    pub scrub: Arc<ScrubState>,
    pub volume: Arc<AtomicVolume>,
    pub output_sample_rate: u32,
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use super::PlaybackParts;
    use crate::audio_engine::AudioEngine;
    use crate::state::StopFlag;
    use std::sync::Arc;
    use std::thread::JoinHandle;

    #[derive(Default)]
    pub struct Playback {
        engine: Option<AudioEngine>,
        worker: Option<JoinHandle<()>>,
        stop_flag: Option<Arc<StopFlag>>,
    }

    impl Playback {
        pub fn default_sample_rate() -> Option<u32> {
            AudioEngine::default_sample_rate()
        }

        pub fn start(&mut self, parts: PlaybackParts) -> Result<(), String> {
            self.stop();
            self.engine = Some(AudioEngine::new(
                parts.volume,
                parts.stream_active.clone(),
                Some(parts.output_sample_rate),
                parts.consumer,
                parts.scrub.clone(),
                parts.source.clone(),
            )?);
            let stop = Arc::new(StopFlag::new());
            self.stop_flag = Some(stop.clone());
            self.worker = Some(crate::deck::start_deck_thread(
                parts.state,
                parts.source,
                parts.producer,
                parts.position,
                parts.stream_active,
                stop,
                parts.scrub,
            ));
            Ok(())
        }

        pub fn stop(&mut self) {
            if let Some(flag) = self.stop_flag.take() {
                flag.set();
            }
            if let Some(handle) = self.worker.take() {
                let _ = handle.join();
            }
            self.engine = None;
        }

        /// Whether an output stream exists to render scrub glides.
        pub fn has_engine(&self) -> bool {
            self.engine.is_some()
        }

        pub fn device_failed(&self) -> bool {
            self.engine.as_ref().is_some_and(AudioEngine::failed)
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use super::PlaybackParts;
    use crate::deck::DeckWorker;
    use crate::web_audio::AudioEngine;
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::JsCast as _;
    use wasm_bindgen::closure::Closure;

    /// How often the page timer steps the deck and tops up Web Audio. The
    /// timer keeps audio running when the UI is not repainting.
    const PUMP_INTERVAL_MS: i32 = 8;

    struct Running {
        engine: AudioEngine,
        /// `None` once the worker has failed; the engine then plays silence.
        worker: Option<DeckWorker>,
    }

    impl Running {
        fn pump(&mut self) {
            if self.worker.as_mut().is_some_and(|worker| !worker.step()) {
                self.worker = None;
            }
            self.engine.pump();
        }
    }

    pub struct Playback {
        running: Rc<RefCell<Option<Running>>>,
    }

    impl Default for Playback {
        fn default() -> Self {
            let running: Rc<RefCell<Option<Running>>> = Rc::default();
            let shared = running.clone();
            let tick = Closure::<dyn FnMut()>::new(move || {
                // `try_borrow_mut`: never re-enter a pump already on the stack.
                if let Ok(mut slot) = shared.try_borrow_mut()
                    && let Some(running) = slot.as_mut()
                {
                    running.pump();
                }
            });
            if let Some(window) = web_sys::window() {
                let _ = window.set_interval_with_callback_and_timeout_and_arguments_0(
                    tick.as_ref().unchecked_ref(),
                    PUMP_INTERVAL_MS,
                );
            }
            // One timer for the lifetime of the page.
            tick.forget();
            Self { running }
        }
    }

    impl Playback {
        pub fn default_sample_rate() -> Option<u32> {
            AudioEngine::default_sample_rate()
        }

        pub fn start(&mut self, parts: PlaybackParts) -> Result<(), String> {
            self.stop();
            let engine = AudioEngine::new(
                parts.volume,
                parts.stream_active.clone(),
                parts.output_sample_rate,
                parts.consumer,
                parts.scrub.clone(),
                parts.source.clone(),
            )?;
            let worker = DeckWorker::new(
                parts.state,
                parts.source,
                parts.producer,
                parts.position,
                parts.stream_active,
                parts.scrub,
            );
            *self.running.borrow_mut() = Some(Running { engine, worker });
            Ok(())
        }

        pub fn stop(&mut self) {
            *self.running.borrow_mut() = None;
        }

        pub fn has_engine(&self) -> bool {
            self.running.borrow().is_some()
        }

        pub fn device_failed(&self) -> bool {
            false
        }

        /// Step once now, in addition to the timer (called every UI frame).
        pub fn pump(&mut self) {
            if let Some(running) = self.running.borrow_mut().as_mut() {
                running.pump();
            }
        }
    }
}

pub use imp::Playback;

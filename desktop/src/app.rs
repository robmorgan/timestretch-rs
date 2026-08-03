use eframe::egui;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::audio_engine::AudioEngine;
use crate::deck;
use crate::decoder;
use crate::state::*;
use crate::waveform::{
    self, BandPeaks, GridMarks, OverviewParams, OverviewTexture, ScrubGesture, ZoomSpan,
    ZoomedParams, ZoomedTiles,
};

const MIN_STRETCH_RATIO: f64 = 0.25;
/// Ratio ceiling on the varispeed-first tempo path (the library bounds the
/// varispeed resampler's step range to `[0.25, 4.0]`).
const MAX_VARISPEED_RATIO: f64 = 4.0;
/// Tempo fader reaches double the track BPM (+100%).
const MAX_TEMPO_FACTOR: f64 = 2.0;
/// Tempo fader width in points (3x egui's ~100pt default) for fine control.
const TEMPO_SLIDER_WIDTH: f32 = 300.0;
/// Auto-loop length ladder: `2^exp` beats. 1/8 beat is still thousands of
/// frames at DJ tempos — far more than the deck feed needs per wrap.
const LOOP_EXP_MIN: i32 = -3;
/// Ladder ceiling: 32 beats (8 bars).
const LOOP_EXP_MAX: i32 = 5;

/// Auto-loop button label for a `2^exp`-beat length: `1/8` … `1/2`, `1` … `32`.
fn loop_len_label(exp: i32) -> String {
    if exp >= 0 {
        format!("{}", 1u32 << exp)
    } else {
        format!("1/{}", 1u32 << -exp)
    }
}

/// Frame span of `beats` (a power of two, possibly fractional) measured on
/// the grid from tracked beat `anchor`: whole-beat spans use the tracked
/// positions so loops stay on the beat through tempo drift, sub-beat spans
/// scale the local beat interval. Falls back to the grid's median interval
/// where the grid runs out (near EOF).
fn grid_loop_span(g: &timestretch::BeatGrid, anchor: usize, beats: f64) -> f64 {
    if beats >= 1.0 {
        match g.beats.get(anchor + beats as usize) {
            Some(&b) => b - g.beats[anchor],
            None => beats * g.beat_interval_samples(),
        }
    } else {
        let local = match g.beats.get(anchor + 1) {
            Some(&b) => b - g.beats[anchor],
            None => g.beat_interval_samples(),
        };
        beats * local
    }
}

/// How far past a stalled anchor the smoother keeps coasting, in seconds.
/// A couple of audio buffers: long enough to glide across normal publish
/// gaps, short enough that an underrun freezes the display quickly.
const PLAYHEAD_MAX_COAST_SECS: f64 = 0.15;
/// Prediction errors within this many seconds of travel are measurement
/// noise: publish-latency jitter, and the engine's splice-aligned source
/// consumption — keylock splices land on beats, so tracking them literally
/// makes the waveform lurch every couple of beats even though the audio is
/// seamless. In-band errors bleed off slowly under a velocity cap instead
/// of being followed.
const PLAYHEAD_NOISE_BAND_SECS: f64 = 0.08;
/// Time constant of the in-band correction.
const PLAYHEAD_NOISE_TAU_SECS: f64 = 0.4;
/// In-band correction speed cap, as a fraction of the nominal velocity:
/// the scroll rate never visibly deviates while an error bleeds off.
const PLAYHEAD_NOISE_MAX_DEVIATION: f64 = 0.04;
/// Out-of-band errors are real events (stall, drift) and correct fast.
const PLAYHEAD_CORRECTION_TAU_SECS: f64 = 0.08;
/// Prediction error beyond this many seconds of travel — in either
/// direction — is a discontinuity (seek, loop wrap) and snaps instead of
/// gliding.
const PLAYHEAD_SNAP_TRAVEL_SECS: f64 = 0.2;

/// Wall-clock smoother for the painted playhead.
///
/// The published playhead only moves when the audio callback consumes a
/// buffer, and the UI samples that on its own unsynchronized cadence — so
/// the raw value stair-steps by whole audio buffers at a beat frequency the
/// eye reads as jitter in the scrolling waveform. While playback runs at a
/// known rate, extrapolate from the last observed step at that rate and
/// correct the residual error away exponentially.
struct PlayheadSmoother {
    /// Raw playhead at the last observed change.
    last_raw: usize,
    /// Extrapolation anchor: the raw playhead when it last changed, and when.
    anchor_frame: f64,
    anchor_time: Instant,
    /// Smoothed position handed to the painters.
    displayed: f64,
    last_tick: Instant,
}

impl PlayheadSmoother {
    fn new(now: Instant) -> Self {
        Self {
            last_raw: 0,
            anchor_frame: 0.0,
            anchor_time: now,
            displayed: 0.0,
            last_tick: now,
        }
    }

    /// Pin the display to `raw` while smoothing is inactive (paused,
    /// stopped, or a scrub gesture owns the position).
    fn reset(&mut self, raw: usize, now: Instant) {
        self.last_raw = raw;
        self.anchor_frame = raw as f64;
        self.anchor_time = now;
        self.displayed = raw as f64;
        self.last_tick = now;
    }

    /// Advance the smoothed playhead: `raw` is the shared atomic's current
    /// value, `rate` the nominal playback speed in source frames per wall
    /// second (negative while a scrub glide moves backward). Returns the
    /// position to paint.
    fn tick(&mut self, raw: usize, rate: f64, now: Instant) -> f64 {
        // Clamp so a long gap between paints (hidden window) can't teleport.
        let dt = (now - self.last_tick).as_secs_f64().min(0.1);
        self.last_tick = now;
        if raw != self.last_raw {
            // Backward steps get no special casing: the error ladder below
            // classifies them by size like any other discrepancy. A splice-
            // sized re-publish (the keylock engine can report a slightly
            // earlier source position around a splice) stays inside the
            // noise band and bleeds off invisibly — snapping the display
            // backward for it was a visible twitch-right-and-recover. Loop
            // wraps and seeks are far bigger and still snap.
            if raw < self.last_raw {
                log::debug!("playhead: backward raw step {} frames", self.last_raw - raw);
            }
            self.last_raw = raw;
            self.anchor_frame = raw as f64;
            self.anchor_time = now;
        }
        let coast = (now - self.anchor_time)
            .as_secs_f64()
            .min(PLAYHEAD_MAX_COAST_SECS);
        let predicted = self.anchor_frame + rate * coast;
        self.displayed += rate * dt;
        let error = predicted - self.displayed;
        let travel = rate.abs().max(1.0);
        if error.abs() > travel * PLAYHEAD_SNAP_TRAVEL_SECS {
            log::debug!("playhead: snap, error {error:.0} frames");
            self.displayed = predicted;
        } else if error.abs() > travel * PLAYHEAD_NOISE_BAND_SECS {
            log::debug!("playhead: out-of-band error {error:.0} frames");
            self.displayed += error * (1.0 - (-dt / PLAYHEAD_CORRECTION_TAU_SECS).exp());
        } else {
            let step = error * (1.0 - (-dt / PLAYHEAD_NOISE_TAU_SECS).exp());
            let cap = travel * dt * PLAYHEAD_NOISE_MAX_DEVIATION;
            self.displayed += step.clamp(-cap, cap);
        }
        self.displayed
    }
}

/// Per-frame snapshot of the shared state, read under a single lock at the
/// top of `update`. The paint path consumes this instead of re-locking:
/// the mutex is shared with the deck thread, and repeated per-frame
/// acquisitions risk blocking long enough to miss the 120 Hz present
/// deadline (one skipped vsync slot reads as a scroll twitch).
#[derive(Clone, Copy)]
struct FrameState {
    transport: Transport,
    sample_rate: u32,
    total_frames: usize,
    /// Raw published playhead this frame (the smoothed one is
    /// `TimeStretchApp::display_pos`).
    position_frames: usize,
    detected_bpm: f64,
    loop_region: Option<(usize, usize)>,
    loop_in: Option<usize>,
}

impl FrameState {
    fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.total_frames as f64 / self.sample_rate as f64
    }

    fn position_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.position_frames as f64 / self.sample_rate as f64
    }
}

pub struct TimeStretchApp {
    state: SharedStateHandle,
    position: Arc<AtomicPosition>,
    stream_active: Arc<AtomicBool>,
    /// Output volume shared lock-free with the audio callback.
    volume_shared: Arc<AtomicVolume>,
    /// Audible-scrub handshake with the audio callback and deck thread.
    scrub: Arc<ScrubState>,
    /// Pointer-implied source frame while a waveform drag is in progress.
    scrub_pos: Option<f64>,
    /// Last consumed glide-landing sequence number; each new one fires the
    /// engine warm-start seek that primes in parallel with the glide.
    landing_seq_seen: u64,
    /// Wall-clock extrapolator that de-jitters the buffer-quantized playhead.
    playhead_smoother: PlayheadSmoother,
    /// Smoothed playhead the painters draw, refreshed every `update`.
    display_pos: f64,

    // Audio engine (lives for the lifetime of the app)
    audio_engine: Option<AudioEngine>,
    output_sample_rate: u32,

    // Source audio data
    source_audio: Option<Arc<Vec<f32>>>,

    // Processing thread
    processing_handle: Option<JoinHandle<()>>,
    stop_flag: Option<Arc<StopFlag>>,

    // File info
    file_name: String,
    file_path: Option<PathBuf>,

    // Waveform
    band_peaks: Option<BandPeaks>,
    /// Pre-rendered overview texture, built lazily from the peaks (needs
    /// an egui context) and dropped on track load.
    overview_texture: Option<OverviewTexture>,
    /// Zoomed-view span state (bars/seconds preset).
    zoom_span: ZoomSpan,
    /// Cached zoomed-view waveform tiles, cleared on track load.
    zoom_tiles: ZoomedTiles,

    /// Beat grid detected on load; drives grid-accurate beat jumps.
    beat_grid: Option<timestretch::BeatGrid>,
    /// Frame-based grid cache for the waveform painters and beat counter.
    grid_marks: GridMarks,

    // UI state
    stretch_ratio: f64,
    /// Target playback BPM the tempo fader binds to (0.0 until a BPM is
    /// detected). Derived view of `stretch_ratio`, kept in sync.
    target_bpm: f64,
    volume: f32,
    preset: PresetChoice,
    deck_engine: DeckEngine,
    target_bpm_text: String,
    /// Auto-loop length as a ladder exponent (`2^exp` beats).
    loop_beats_exp: i32,
    /// Fixed-refresh display pin; `Some` while the display is pinned
    /// (dropping restores the previous mode).
    refresh_pin: Option<crate::display_refresh::RefreshPin>,

    // Error messages
    error_message: Option<String>,
}

impl TimeStretchApp {
    /// Widest stretch ratio the active tempo path supports.
    #[inline]
    fn max_stretch_ratio(&self) -> f64 {
        // The engine's tempo axis is the varispeed resampler.
        MAX_VARISPEED_RATIO
    }

    #[inline]
    fn clamp_stretch_ratio(&self, ratio: f64) -> f64 {
        ratio.clamp(MIN_STRETCH_RATIO, self.max_stretch_ratio())
    }

    /// Sets the playback tempo to `target_bpm` (for a track at `detected_bpm`)
    /// and syncs every derived view — stretch ratio, the fader's `target_bpm`,
    /// and the BPM text box — to the value that actually plays after the
    /// engine's ratio clamp. Single write point for the tempo control.
    fn apply_target_bpm(&mut self, detected_bpm: f64, target_bpm: f64) {
        let ratio = self.clamp_stretch_ratio(detected_bpm / target_bpm.max(1e-6));
        let effective_bpm = detected_bpm / ratio;
        self.stretch_ratio = ratio;
        self.target_bpm = effective_bpm;
        self.target_bpm_text = format!("{effective_bpm:.1}");
        let mut st = self.state.lock().unwrap();
        st.stretch_ratio = ratio;
        st.target_bpm = effective_bpm;
    }

    pub fn new(cc: &eframe::CreationContext<'_>, initial_file: Option<PathBuf>) -> Self {
        // CDJ-style deck: always the dark theme, whatever the system says.
        cc.egui_ctx.set_theme(egui::Theme::Dark);

        let state = Arc::new(Mutex::new(SharedState::new()));
        let stream_active = Arc::new(AtomicBool::new(false));
        let position = Arc::new(AtomicPosition::new());

        // Detect the default output sample rate to size the UI; the real
        // engine is built when a track starts playing.
        let output_sample_rate = AudioEngine::default_sample_rate().unwrap_or_else(|| {
            log::error!("No default audio output device; falling back to 44100 Hz");
            44100
        });
        let audio_engine = None;

        let mut app = Self {
            state,
            position,
            stream_active,
            volume_shared: Arc::new(AtomicVolume::new(0.8)),
            scrub: Arc::new(ScrubState::new()),
            scrub_pos: None,
            landing_seq_seen: 0,
            playhead_smoother: PlayheadSmoother::new(Instant::now()),
            display_pos: 0.0,
            audio_engine,
            output_sample_rate,
            source_audio: None,
            processing_handle: None,
            stop_flag: None,
            file_name: String::new(),
            file_path: None,
            band_peaks: None,
            overview_texture: None,
            zoom_span: ZoomSpan::default(),
            zoom_tiles: ZoomedTiles::default(),
            beat_grid: None,
            grid_marks: GridMarks::empty(),
            stretch_ratio: 1.0,
            target_bpm: 0.0,
            volume: 0.8,
            preset: PresetChoice::DjBeatmatch,
            deck_engine: DeckEngine::Keylock,
            target_bpm_text: String::new(),
            loop_beats_exp: 2,
            error_message: None,
            refresh_pin: None,
        };
        if let Some(path) = initial_file {
            app.load_file(path);
        }
        app
    }

    fn load_file(&mut self, path: PathBuf) {
        // Stop any existing playback
        self.stop_playback();

        self.error_message = None;
        self.file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        log::info!("Loading file: {}", path.display());

        match decoder::decode_file(&path) {
            Ok(decoded) => {
                log::info!(
                    "Decoded: {} frames, {} Hz, {} ch",
                    decoded.num_frames,
                    decoded.sample_rate,
                    decoded.channels
                );

                let sample_rate = decoded.sample_rate;
                let num_frames = decoded.num_frames;
                let channel_layout = timestretch::Channels::from_count(decoded.channels as usize)
                    .unwrap_or(timestretch::Channels::Stereo);
                let bpm_buffer =
                    timestretch::AudioBuffer::new(decoded.samples, sample_rate, channel_layout);

                // Compute the 3-band waveform peak pyramid
                self.band_peaks = Some(BandPeaks::compute(
                    &bpm_buffer.data,
                    channel_layout.count(),
                    sample_rate,
                ));
                self.overview_texture = None;
                self.zoom_tiles.clear();

                // Detect the beat grid from the channel-aware buffer
                // (stereo-safe): BPM for the tempo fader plus beat and
                // downbeat positions for the overlay and beat jumps.
                let grid = timestretch::detect_beat_grid_buffer(&bpm_buffer);
                let bpm = grid.bpm;
                log::info!(
                    "Detected BPM: {bpm:.1} ({} beats, {} segments, confidence {:.2})",
                    grid.beats.len(),
                    grid.segments.len(),
                    grid.confidence
                );
                self.grid_marks = GridMarks::from_grid(&grid);
                self.beat_grid = Some(grid);
                self.target_bpm_text = if bpm > 0.0 {
                    format!("{bpm:.1}")
                } else {
                    String::new()
                };
                // A fresh track starts at its own tempo (fader centered) and
                // unity ratio, regardless of any prior track's settings.
                self.stretch_ratio = 1.0;
                self.target_bpm = bpm.max(0.0);
                let num_channels = bpm_buffer.channels.count();
                let samples = Arc::new(bpm_buffer.into_data());

                // Update shared state
                let analysis_generation = {
                    let mut st = self.state.lock().unwrap();
                    st.sample_rate = sample_rate;
                    st.total_frames = num_frames;
                    st.position_frames = 0;
                    st.detected_bpm = bpm;
                    st.target_bpm = bpm.max(0.0);
                    st.transport = Transport::Stopped;
                    st.stretch_ratio = self.stretch_ratio;
                    st.preset = self.preset;
                    st.pre_analysis = None;
                    st.loop_region = None;
                    st.loop_in = None;
                    st.analysis_generation += 1;
                    st.analysis_generation
                };

                // Analyze-on-load: a matching sidecar is used immediately;
                // otherwise a background thread analyzes once and caches the
                // result next to the file. `detect_bpm_buffer` above keeps
                // the UI BPM instant either way — the artifact upgrades
                // subsequent processor rebuilds when it lands.
                spawn_pre_analysis(
                    self.state.clone(),
                    samples.clone(),
                    num_channels,
                    sample_rate,
                    sidecar_path(&path),
                    analysis_generation,
                );

                self.source_audio = Some(samples);
                self.file_path = Some(path);
                self.output_sample_rate = sample_rate;
                self.position.store(0);
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to load: {e}"));
                log::error!("Failed to load file: {e}");
            }
        }
    }

    /// Starts playback: the audio callback owns the processor and reads
    /// from it; the feed thread keeps the source ring topped up and forwards
    /// tempo control.
    fn start_playback(&mut self) {
        let source = match &self.source_audio {
            Some(s) => s.clone(),
            None => return,
        };

        // Stop any existing processing thread
        self.stop_processing_thread();

        let sample_rate = {
            let st = self.state.lock().unwrap();
            st.sample_rate
        };

        let initial_ratio = {
            let st = self.state.lock().unwrap();
            st.stretch_ratio
        };
        // Always the keylock chain: the Tape/Keylock deck mode is a live
        // engine parameter (delay-matched high-band crossfade), not a
        // profile choice — switching mid-play is instant, with constant
        // pipeline latency in both modes.
        let profile = timestretch::engine::EngineProfile::Keylock;
        // Analyze-on-load artifact: the engine's primary transient control
        // signal (splice guidance + PV phase resets). Arc-shared with the
        // analysis thread's result.
        let pre_analysis = self.state.lock().unwrap().pre_analysis.clone();
        let config = timestretch::engine::EngineConfig {
            sample_rate,
            channels: 2,
            profile,
            initial_tempo_rate: 1.0 / initial_ratio.clamp(0.25, MAX_VARISPEED_RATIO),
            max_block_frames: 2048,
            source_capacity_frames: 65_536,
            pre_analysis,
        };
        let handles = match timestretch::engine::Engine::build(config) {
            Ok(h) => h,
            Err(e) => {
                self.error_message = Some(format!("Engine error: {e}"));
                return;
            }
        };
        let pipeline_latency_secs =
            handles.processor.pipeline_latency_frames() as f64 / sample_rate as f64;
        let warm_start_preroll = handles.processor.warm_start_preroll_frames();

        let reset_request = Arc::new(AtomicBool::new(false));
        self.volume_shared.store(self.volume);
        let engine = match AudioEngine::new(
            self.volume_shared.clone(),
            self.stream_active.clone(),
            Some(sample_rate),
            handles.processor,
            reset_request.clone(),
            self.scrub.clone(),
            source.clone(),
        ) {
            Ok(e) => e,
            Err(e) => {
                self.error_message = Some(format!("Audio error: {e}"));
                return;
            }
        };
        self.audio_engine = Some(engine);

        {
            let mut st = self.state.lock().unwrap();
            st.transport = Transport::Playing;
            st.total_frames = source.len() / 2;
        }

        let stop_flag = Arc::new(StopFlag::new());
        self.stop_flag = Some(stop_flag.clone());

        let handle = deck::start_deck_thread(
            self.state.clone(),
            source,
            handles.source,
            handles.controller,
            self.position.clone(),
            self.stream_active.clone(),
            stop_flag,
            reset_request,
            self.scrub.clone(),
            pipeline_latency_secs,
            warm_start_preroll,
        );

        self.processing_handle = Some(handle);
    }

    fn stop_playback(&mut self) {
        {
            let mut st = self.state.lock().unwrap();
            st.transport = Transport::Stopped;
            st.position_frames = 0;
        }
        self.position.store(0);
        self.stop_processing_thread();
    }

    fn toggle_pause(&mut self) {
        let mut st = self.state.lock().unwrap();
        match st.transport {
            Transport::Playing => st.transport = Transport::Paused,
            Transport::Paused => st.transport = Transport::Playing,
            Transport::Stopped => {
                drop(st);
                self.start_playback();
            }
        }
    }

    fn stop_processing_thread(&mut self) {
        if let Some(flag) = self.stop_flag.take() {
            flag.set();
        }
        if let Some(handle) = self.processing_handle.take() {
            let _ = handle.join();
        }
        self.stream_active.store(false, Ordering::Relaxed);
        self.audio_engine = None;
    }

    fn format_time(secs: f64) -> String {
        let mins = (secs / 60.0) as u32;
        let s = secs % 60.0;
        format!("{mins}:{s:05.2}")
    }
}

impl eframe::App for TimeStretchApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let frame_start = Instant::now();
        // Scrub glide bookkeeping. During a settle the voice owns the
        // moving position, so mirror it into the shared playhead; and each
        // newly published landing fires the parallel engine warm-start.
        let scrub_phase = self.scrub.phase();
        if scrub_phase == ScrubPhase::Settling {
            self.position.store(self.scrub.voice_frame() as usize);
        }
        let (landing_seq, landing) = self.scrub.landing();
        if landing_seq != self.landing_seq_seen {
            self.landing_seq_seen = landing_seq;
            self.request_seek(landing as usize);
        }

        // Sync position from the atomic counter and snapshot everything the
        // paint path needs under one lock.
        let pos_frames = self.position.load();
        let fs = {
            let mut st = self.state.lock().unwrap();
            st.position_frames = pos_frames;
            FrameState {
                transport: st.transport,
                sample_rate: st.sample_rate,
                total_frames: st.total_frames,
                position_frames: pos_frames,
                detected_bpm: st.detected_bpm,
                loop_region: st.loop_region,
                loop_in: st.loop_in,
            }
        };
        let (transport, sample_rate) = (fs.transport, fs.sample_rate);

        // The painted playhead: while the engine is streaming steadily (or
        // a scrub release glide is running), extrapolate between the
        // buffer-quantized published positions so the waveform scrolls at
        // wall-clock speed instead of stair-stepping at the audio-buffer/
        // UI-frame beat frequency. The glide extrapolates with the voice's
        // published rate, and the smoother carries its state across the
        // Settling → Idle handoff so the voice → engine position mismatch
        // bleeds off instead of stepping. A drag (pointer owns the display)
        // and paused/stopped pin the display to the raw value.
        let smoothing_rate = match scrub_phase {
            ScrubPhase::Idle
                if transport == Transport::Playing
                    && self.stream_active.load(Ordering::Relaxed) =>
            {
                Some(sample_rate as f64 / self.stretch_ratio.max(MIN_STRETCH_RATIO))
            }
            ScrubPhase::Settling => Some(self.scrub.voice_rate() * self.output_sample_rate as f64),
            _ => None,
        };
        self.display_pos = match smoothing_rate {
            Some(rate) => self
                .playhead_smoother
                .tick(pos_frames, rate, Instant::now()),
            None => {
                self.playhead_smoother.reset(pos_frames, Instant::now());
                pos_frames as f64
            }
        };
        // Repaint at full display rate while playing: vsync-paced, so the
        // extrapolated playhead lands on screen at even intervals. This was
        // ~90% of the app's CPU when the zoomed view tessellated thousands
        // of rects per frame; with the tile cache a frame is a few textured
        // quads, so the uncapped loop is affordable. A scrub glide animates
        // the playhead the same way even while paused, so it keeps the
        // repaint loop alive too.
        if transport == Transport::Playing || scrub_phase != ScrubPhase::Idle {
            // Experiment lever: TIMESTRETCH_REPAINT_MS paces repaints on a
            // timer instead of every vsync, to test whether a steady
            // sub-max present cadence makes ProMotion settle into a stable
            // rate without a display mode switch.
            match std::env::var("TIMESTRETCH_REPAINT_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
            {
                Some(ms) => ctx.request_repaint_after(std::time::Duration::from_millis(ms)),
                None => ctx.request_repaint(),
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Timestretch Desktop");
            ui.add_space(8.0);

            // Error message
            if let Some(ref err) = self.error_message {
                ui.colored_label(egui::Color32::RED, err);
                ui.add_space(4.0);
            }

            // File panel
            self.file_panel(ui, &fs);
            ui.add_space(8.0);

            // Deck waveforms
            self.deck_panel(ui, &fs);
            ui.add_space(8.0);

            // Transport
            self.transport_panel(ui, &fs);
            ui.add_space(12.0);

            // Controls
            self.controls_panel(ui, &fs);
        });

        // Frame-pacing measurement harness: run with
        // `RUST_LOG=timestretch_desktop=trace` and diff consecutive `us`
        // stamps — gaps of ~2 vsync slots are missed present deadlines,
        // which the eye reads as scroll twitches during pursuit. `cpu_us`
        // is this update's CPU cost (layout + tessellation submission);
        // misses without a slow cpu_us are downstream (wakeup/GPU).
        log::trace!(
            "frame raw={pos_frames} disp={:.1} cpu_us={} us={}",
            self.display_pos,
            frame_start.elapsed().as_micros(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros()
        );
    }

    fn on_exit(&mut self) {
        // Drop the pin explicitly so the display mode is restored even if
        // the app value itself never gets dropped on this exit path.
        self.refresh_pin = None;
        self.stop_processing_thread();
    }
}

impl TimeStretchApp {
    fn file_panel(&mut self, ui: &mut egui::Ui, fs: &FrameState) {
        ui.horizontal(|ui| {
            if ui.button("Load Audio File").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("Audio", &["wav", "mp3", "flac", "ogg"])
                    .pick_file()
            {
                self.load_file(path);
            }

            if !self.file_name.is_empty() {
                ui.separator();
                ui.label(&self.file_name);

                ui.separator();
                ui.label(format!("{} Hz", fs.sample_rate));
                ui.separator();
                ui.label(format!("{:.1}s", fs.duration_secs()));
                ui.separator();
                if fs.detected_bpm > 0.0 {
                    ui.label(format!("{:.1} BPM", fs.detected_bpm));
                } else {
                    ui.label("BPM: --");
                }
            }
        });
    }

    /// CDJ-style deck display: zoomed scrolling waveform on top, beat
    /// counter + zoom controls in a thin row, full-track overview strip
    /// below.
    fn deck_panel(&mut self, ui: &mut egui::Ui, fs: &FrameState) {
        let (total_frames, sample_rate, loop_region, loop_in) =
            (fs.total_frames, fs.sample_rate, fs.loop_region, fs.loop_in);

        // Zoomed scrolling view; dragging scrubs audibly relative to the
        // pointer. During the drag the UI owns the displayed position and
        // publishes the target to the audio callback's scrub voice; the drop
        // triggers a single warm-start seek, so playback resumes at normal
        // speed from wherever the waveform was released.
        let gesture = waveform::paint_zoomed(
            ui,
            ZoomedParams {
                peaks: self.band_peaks.as_ref(),
                marks: &self.grid_marks,
                position_frames: self.display_pos,
                total_frames,
                sample_rate,
                loop_region,
                loop_in,
            },
            &mut self.zoom_span,
            &mut self.zoom_tiles,
        );
        match gesture {
            Some(ScrubGesture::Drag(delta_frames)) if total_frames > 0 => {
                // Re-grabbing a mid-glide platter continues from the voice's
                // gliding position, not the stale engine playhead.
                let base = self.scrub_pos.unwrap_or_else(|| {
                    if self.scrub.phase() == ScrubPhase::Settling {
                        self.scrub.voice_frame()
                    } else {
                        // Anchor the grab to the smoothed position actually
                        // on screen, not the buffer-quantized raw playhead.
                        self.display_pos
                    }
                });
                let target = (base + delta_frames).clamp(0.0, (total_frames - 1) as f64);
                if self.scrub_pos.is_none() {
                    self.scrub.begin(target);
                } else {
                    self.scrub.update_target(target);
                }
                self.scrub_pos = Some(target);
                self.position.store(target as usize);
                self.state.lock().unwrap().position_frames = target as usize;
            }
            Some(ScrubGesture::Release) => {
                if let Some(frame) = self.scrub_pos.take() {
                    if self.audio_engine.is_some() {
                        // Momentum glide: the audio callback eases the voice
                        // toward play speed (or rest), predicts the landing,
                        // and the landing consumer below warm-starts the
                        // engine there in parallel.
                        let playing = self.state.lock().unwrap().transport == Transport::Playing;
                        self.scrub.release(if playing { 1.0 } else { 0.0 });
                    } else {
                        // No audio stream to render a glide — land instantly.
                        self.scrub.cancel();
                        self.request_seek(frame as usize);
                    }
                }
            }
            _ => {}
        }
        ui.add_space(4.0);

        // Counter row: bar.beat readout, beat segments, zoom controls.
        ui.horizontal(|ui| {
            waveform::paint_beat_counter(ui, &self.grid_marks, self.display_pos);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("+").clicked() {
                    self.zoom_span.zoom_in();
                }
                ui.label(
                    egui::RichText::new(self.zoom_span.label(self.grid_marks.is_usable()))
                        .monospace()
                        .weak(),
                );
                if ui.button("−").clicked() {
                    self.zoom_span.zoom_out();
                }
            });
        });
        ui.add_space(4.0);

        // Overview strip; built lazily on first paint after a track load.
        if self.overview_texture.is_none()
            && let Some(peaks) = &self.band_peaks
        {
            self.overview_texture = Some(OverviewTexture::from_peaks(ui.ctx(), peaks));
        }
        let progress = if total_frames > 0 {
            (self.display_pos / total_frames as f64) as f32
        } else {
            0.0
        };
        let seek_frac = waveform::paint_overview(
            ui,
            OverviewParams {
                texture: self.overview_texture.as_ref(),
                marks: &self.grid_marks,
                progress,
                total_frames,
                loop_region,
                loop_in,
            },
        );
        if let Some(frac) = seek_frac
            && total_frames > 0
        {
            self.request_seek((frac as f64 * total_frames as f64) as usize);
        }
    }

    /// Routes a seek to the deck thread; while stopped (no deck thread
    /// running) also moves the visible position directly — the pending
    /// `seek_request` is consumed when playback next starts.
    fn request_seek(&mut self, frame: usize) {
        let stopped = {
            let mut st = self.state.lock().unwrap();
            st.seek_request = Some(frame);
            if st.transport == Transport::Stopped {
                st.position_frames = frame;
            }
            st.transport == Transport::Stopped
        };
        if stopped {
            self.position.store(frame);
        }
    }

    fn transport_panel(&mut self, ui: &mut egui::Ui, fs: &FrameState) {
        let (transport, pos_secs, duration_secs) =
            (fs.transport, fs.position_secs(), fs.duration_secs());

        ui.horizontal(|ui| {
            let play_label = match transport {
                Transport::Playing => "Pause",
                Transport::Paused => "Resume",
                Transport::Stopped => "Play",
            };

            if ui
                .add_enabled(self.source_audio.is_some(), egui::Button::new(play_label))
                .clicked()
            {
                self.toggle_pause();
            }

            if ui
                .add_enabled(transport != Transport::Stopped, egui::Button::new("Stop"))
                .clicked()
            {
                self.stop_playback();
            }

            ui.separator();

            ui.monospace(format!(
                "{} / {}",
                Self::format_time(pos_secs),
                Self::format_time(duration_secs)
            ));
        });

        self.loop_and_jump_panel(ui, fs);
    }

    /// Beat-jump buttons and loop controls (manual in/out/exit plus a
    /// grid-quantized auto-loop with a halve/double length ladder). Jumps
    /// and loop wraps go through the processing thread's warm-start
    /// machinery.
    fn loop_and_jump_panel(&mut self, ui: &mut egui::Ui, fs: &FrameState) {
        let has_audio = self.source_audio.is_some();
        let (pos_frames, total_frames, bpm, sample_rate, loop_region, loop_in) = (
            fs.position_frames,
            fs.total_frames,
            fs.detected_bpm,
            fs.sample_rate,
            fs.loop_region,
            fs.loop_in,
        );

        ui.horizontal(|ui| {
            // Beat jumps: relative seeks by whole beats on the detected
            // grid (tracked positions, not computed intervals, so jumps
            // stay on the beat through tempo drift). Falls back to fixed
            // 60/BPM intervals when only a tempo is known. Disabled until
            // a tempo is detected.
            let grid = self
                .beat_grid
                .as_ref()
                .filter(|g| g.beats.len() >= 2 && g.bpm > 0.0);
            let beat_frames = if bpm > 0.0 {
                (sample_rate as f64 * 60.0 / bpm).round() as i64
            } else {
                0
            };
            let can_jump = has_audio && total_frames > 0 && (grid.is_some() || beat_frames > 0);
            ui.label("Jump:");
            for beats in [-16i64, -4, 4, 16] {
                let label = if beats > 0 {
                    format!("+{beats}")
                } else {
                    format!("{beats}")
                };
                if ui.add_enabled(can_jump, egui::Button::new(label)).clicked() {
                    let target = match grid {
                        Some(g) => {
                            let here = g.nearest_beat_index(pos_frames as f64).unwrap_or(0) as i64;
                            let idx = (here + beats).clamp(0, g.beats.len() as i64 - 1) as usize;
                            g.beats[idx].round().max(0.0) as usize
                        }
                        None => {
                            let delta = beats * beat_frames;
                            (pos_frames as i64 + delta).max(0) as usize
                        }
                    }
                    .min(total_frames.saturating_sub(1));
                    self.state.lock().unwrap().seek_request = Some(target);
                }
            }

            ui.separator();

            // Loop: set in point, then out point to arm; exit to clear.
            ui.label("Loop:");
            if ui.add_enabled(has_audio, egui::Button::new("In")).clicked() {
                self.state.lock().unwrap().loop_in = Some(pos_frames);
            }
            let can_close = has_audio && loop_in.is_some_and(|i| pos_frames > i);
            if ui
                .add_enabled(can_close, egui::Button::new("Out"))
                .clicked()
                && let Some(start) = loop_in
            {
                let mut st = self.state.lock().unwrap();
                st.loop_region = Some((start, pos_frames));
                st.loop_in = None;
            }
            if ui
                .add_enabled(loop_region.is_some(), egui::Button::new("Exit"))
                .clicked()
            {
                self.state.lock().unwrap().loop_region = None;
            }

            // Auto-loop: one click arms a loop of the selected length
            // snapped to the nearest grid beat (fixed 60/BPM intervals
            // when only a tempo is known, like jumps); clicking again
            // releases it. </> halve/double the length, resizing an
            // active loop in place from its start.
            //
            // Loop ends never pass EOF: the deck feed can't reach a loop
            // end beyond the source, and an armed loop suppresses the
            // end-of-stream stop.
            let loop_end_for = |start: usize, beats: f64| -> Option<usize> {
                let end = start as f64
                    + match grid {
                        Some(g) => {
                            let anchor = g.nearest_beat_index(start as f64).unwrap_or(0);
                            grid_loop_span(g, anchor, beats)
                        }
                        None => beats * beat_frames as f64,
                    };
                let end = (end.round() as usize).min(total_frames);
                (end > start).then_some(end)
            };

            let looping = loop_region.is_some();
            let can_loop = if looping { has_audio } else { can_jump };
            let mut new_exp = self.loop_beats_exp;
            if ui
                .add_enabled(
                    can_loop && self.loop_beats_exp > LOOP_EXP_MIN,
                    egui::Button::new("<"),
                )
                .clicked()
            {
                new_exp -= 1;
            }
            let label = format!("Loop {}", loop_len_label(self.loop_beats_exp));
            if ui
                .add_enabled(can_loop, egui::Button::new(label).selected(looping))
                .clicked()
            {
                if looping {
                    self.state.lock().unwrap().loop_region = None;
                } else {
                    let start = match grid {
                        Some(g) => {
                            let i = g.nearest_beat_index(pos_frames as f64).unwrap_or(0);
                            g.beats[i].round().max(0.0) as usize
                        }
                        None => pos_frames,
                    };
                    if let Some(end) = loop_end_for(start, 2f64.powi(self.loop_beats_exp)) {
                        let mut st = self.state.lock().unwrap();
                        st.loop_region = Some((start, end));
                        st.loop_in = None;
                    }
                }
            }
            if ui
                .add_enabled(
                    can_loop && self.loop_beats_exp < LOOP_EXP_MAX,
                    egui::Button::new(">"),
                )
                .clicked()
            {
                new_exp += 1;
            }
            if new_exp != self.loop_beats_exp {
                self.loop_beats_exp = new_exp;
                // Live resize: keep the loop start, requantize the end.
                if let Some((start, _)) = loop_region
                    && let Some(end) = loop_end_for(start, 2f64.powi(new_exp))
                {
                    self.state.lock().unwrap().loop_region = Some((start, end));
                }
            }

            match loop_region {
                Some((s, e)) => {
                    let secs = (e - s) as f64 / sample_rate.max(1) as f64;
                    ui.monospace(format!("looping {secs:.2}s"));
                }
                None if loop_in.is_some() => {
                    ui.monospace("in set…");
                }
                None => {}
            }
        });
    }

    fn controls_panel(&mut self, ui: &mut egui::Ui, fs: &FrameState) {
        egui::Grid::new("controls_grid")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                // Tempo control. With a detected BPM it is a CDJ-style tempo
                // fader in BPM (0.1-BPM steps, centered on the track tempo);
                // otherwise it falls back to a raw stretch-ratio slider.
                let detected = fs.detected_bpm;
                ui.label("Tempo:");
                ui.horizontal(|ui| {
                    if detected > 0.0 {
                        if self.target_bpm <= 0.0 {
                            self.target_bpm = detected;
                        }
                        let min_bpm = detected / self.max_stretch_ratio();
                        let max_bpm = detected * MAX_TEMPO_FACTOR;
                        let old_bpm = self.target_bpm;
                        // Widen the fader (3x the default) so 0.1-BPM steps
                        // are easy to hit by drag.
                        ui.spacing_mut().slider_width = TEMPO_SLIDER_WIDTH;
                        ui.add(
                            egui::Slider::new(&mut self.target_bpm, min_bpm..=max_bpm)
                                .step_by(0.1)
                                .fixed_decimals(1)
                                .suffix(" BPM"),
                        );
                        if (self.target_bpm - old_bpm).abs() > 0.001 {
                            self.apply_target_bpm(detected, self.target_bpm);
                        }
                        let pct = (1.0 / self.stretch_ratio - 1.0) * 100.0;
                        ui.label(egui::RichText::new(format!("{pct:+.1}%")).weak());
                        if ui.button("Reset").clicked() {
                            self.apply_target_bpm(detected, detected);
                        }
                    } else {
                        let old_ratio = self.stretch_ratio;
                        let max_ratio = self.max_stretch_ratio();
                        ui.add(
                            egui::Slider::new(
                                &mut self.stretch_ratio,
                                MIN_STRETCH_RATIO..=max_ratio,
                            )
                            .text("x")
                            .fixed_decimals(2),
                        );
                        if (self.stretch_ratio - old_ratio).abs() > 0.001 {
                            self.state.lock().unwrap().stretch_ratio = self.stretch_ratio;
                        }
                        if ui.button("Reset").clicked() {
                            self.stretch_ratio = 1.0;
                            self.state.lock().unwrap().stretch_ratio = 1.0;
                        }
                    }
                });
                ui.end_row();

                // BPM panel
                let detected_bpm = fs.detected_bpm;
                ui.label("BPM:");
                ui.horizontal(|ui| {
                    if detected_bpm > 0.0 {
                        ui.label(format!("Detected: {detected_bpm:.1}"));
                        ui.separator();
                        ui.label("Target:");
                        let response = ui.add(
                            egui::TextEdit::singleline(&mut self.target_bpm_text)
                                .desired_width(60.0),
                        );
                        if response.lost_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::Enter))
                            && let Ok(target) = self.target_bpm_text.parse::<f64>()
                            && target > 0.0
                        {
                            // Route through the shared sync point so the
                            // tempo fader tracks a typed BPM too.
                            self.apply_target_bpm(detected_bpm, target);
                        }
                    } else {
                        ui.label("Load a file to detect BPM");
                    }
                });
                ui.end_row();

                // EDM Preset
                ui.label("Preset:");
                ui.horizontal(|ui| {
                    let old_preset = self.preset;
                    egui::ComboBox::from_id_salt("preset_combo")
                        .selected_text(self.preset.label())
                        .show_ui(ui, |ui| {
                            for &p in PresetChoice::ALL {
                                ui.selectable_value(&mut self.preset, p, p.label());
                            }
                        });
                    if self.preset != old_preset {
                        let mut st = self.state.lock().unwrap();
                        st.preset = self.preset;
                    }
                });
                ui.end_row();

                // Deck engine mode: Tape (pitch follows tempo) vs Keylock
                // (pitch-preserving). A live engine parameter — the deck
                // thread forwards it and the keylock stage crossfades
                // (~12 ms), so switching mid-play is instant and gapless.
                ui.label("Deck:");
                ui.horizontal(|ui| {
                    let old_deck = self.deck_engine;
                    egui::ComboBox::from_id_salt("deck_engine_combo")
                        .selected_text(self.deck_engine.label())
                        .show_ui(ui, |ui| {
                            for &deck in DeckEngine::ALL {
                                ui.selectable_value(&mut self.deck_engine, deck, deck.label());
                            }
                        });
                    if self.deck_engine != old_deck {
                        self.state.lock().unwrap().deck_engine = self.deck_engine;
                    }
                    match self.deck_engine {
                        DeckEngine::Tape => {
                            ui.label(egui::RichText::new("tape: pitch follows tempo").weak())
                                .on_hover_text(
                                    "Tape mode: pitch follows the fader (delay-matched \
                                     varispeed; same constant ~13 ms pipeline delay as \
                                     keylock, so switching live is seamless).",
                                );
                        }
                        DeckEngine::Keylock => {
                            ui.label(egui::RichText::new("keylock: two-band").weak())
                                .on_hover_text(
                                    "Keylock mode: low band follows tempo, high band \
                                     pitch-corrected at the delay-matched transposition \
                                     (~13 ms pipeline delay).",
                                );
                        }
                    }
                });
                ui.end_row();

                // Fixed-refresh pin: ProMotion's adaptive rate switching
                // skips ~2 vsync slots/s in any windowed macOS app, which
                // reads as waveform scroll twitches; a fixed rate is
                // near-perfectly clean (see examples/metalwave_spike.rs).
                if cfg!(target_os = "macos") {
                    ui.label("Display:");
                    ui.horizontal(|ui| {
                        let mut pinned = self.refresh_pin.is_some();
                        let changed = ui
                            .checkbox(&mut pinned, "Fixed refresh")
                            .on_hover_text(
                                "Pin the display to a fixed refresh rate for perfectly \
                                 even waveform scrolling (ProMotion's adaptive rate \
                                 causes visible micro-stutters). Session-scoped: \
                                 restored on untick, app exit, or logout.",
                            )
                            .changed();
                        if changed {
                            if pinned {
                                match crate::display_refresh::RefreshPin::pin() {
                                    Ok(pin) => self.refresh_pin = Some(pin),
                                    Err(e) => {
                                        self.error_message = Some(format!("Fixed refresh: {e}"));
                                    }
                                }
                            } else {
                                // Dropping the pin restores the prior mode.
                                self.refresh_pin = None;
                            }
                        }
                        if let Some(pin) = &self.refresh_pin {
                            ui.label(
                                egui::RichText::new(format!("pinned {:.0} Hz", pin.pinned_hz()))
                                    .weak(),
                            );
                        }
                    });
                    ui.end_row();
                }

                // Volume
                ui.label("Volume:");
                ui.horizontal(|ui| {
                    let old_vol = self.volume;
                    ui.add(
                        egui::Slider::new(&mut self.volume, 0.0..=1.0)
                            .text("")
                            .fixed_decimals(0)
                            .custom_formatter(|v, _| format!("{}%", (v * 100.0) as u32)),
                    );
                    if (self.volume - old_vol).abs() > 0.001 {
                        self.volume_shared.store(self.volume);
                    }
                });
                ui.end_row();
            });
    }
}

/// Sidecar artifact path for a loaded audio file: `<file>.tsanalysis.json`.
fn sidecar_path(audio_path: &std::path::Path) -> PathBuf {
    let mut os = audio_path.as_os_str().to_os_string();
    os.push(".tsanalysis.json");
    PathBuf::from(os)
}

/// Loads a matching sidecar artifact or analyzes the track on a background
/// thread, storing the result in shared state for the next processor rebuild.
///
/// The result is discarded if another file was loaded in the meantime
/// (`generation` mismatch). Sidecar writes are best-effort: a read-only
/// volume must not break analysis.
fn spawn_pre_analysis(
    state: SharedStateHandle,
    samples: Arc<Vec<f32>>,
    num_channels: usize,
    sample_rate: u32,
    sidecar: PathBuf,
    generation: u64,
) {
    std::thread::spawn(move || {
        let analysis_signal = timestretch::downmix_to_mid(&samples, num_channels);

        let artifact = match timestretch::read_preanalysis_json(&sidecar) {
            Ok(cached) if cached.matches_source(&analysis_signal, sample_rate) => {
                log::info!("Pre-analysis: using cached sidecar {}", sidecar.display());
                cached
            }
            _ => {
                let start = std::time::Instant::now();
                let fresh = timestretch::analyze_for_dj(&analysis_signal, sample_rate);
                log::info!(
                    "Pre-analysis: {:.1} BPM, confidence {:.2}, {} beats, {} onsets ({:.2}s)",
                    fresh.bpm,
                    fresh.confidence,
                    fresh.beat_positions.len(),
                    fresh.transient_onsets.len(),
                    start.elapsed().as_secs_f64()
                );
                if let Err(e) = timestretch::write_preanalysis_json(&sidecar, &fresh) {
                    log::warn!(
                        "Pre-analysis: could not cache sidecar {}: {e}",
                        sidecar.display()
                    );
                }
                fresh
            }
        };

        let mut st = state.lock().unwrap();
        if st.analysis_generation == generation {
            st.pre_analysis = Some(Arc::new(artifact));
        } else {
            log::info!("Pre-analysis: discarding stale result (newer file loaded)");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const SR: f64 = 44100.0;
    const BUFFER: usize = 512;
    /// UI paint interval (~30 fps).
    const TICK_SECS: f64 = 0.033;

    /// Raw playhead as the deck publishes it: quantized to whole audio
    /// buffers of elapsed wall-clock playback.
    fn raw_at(t_secs: f64) -> usize {
        ((t_secs * SR) as usize / BUFFER) * BUFFER
    }

    /// Run `n` UI ticks of steady playback starting at tick `start`,
    /// returning the painted position after each tick.
    fn run_steady(sm: &mut PlayheadSmoother, t0: Instant, start: usize, n: usize) -> Vec<f64> {
        (start..start + n)
            .map(|i| {
                let t = i as f64 * TICK_SECS;
                sm.tick(raw_at(t), SR, t0 + Duration::from_secs_f64(t))
            })
            .collect()
    }

    #[test]
    fn steady_playback_scrolls_evenly() {
        let t0 = Instant::now();
        let mut sm = PlayheadSmoother::new(t0);
        // Warm up past the initial convergence, then measure.
        run_steady(&mut sm, t0, 1, 30);
        let painted = run_steady(&mut sm, t0, 31, 70);
        let deltas: Vec<f64> = painted.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
        let worst = deltas.iter().map(|d| (d - mean).abs()).fold(0.0, f64::max);
        // The raw value stair-steps by whole buffers, so its per-tick deltas
        // vary by ±35%; the smoothed playhead must advance at wall-clock
        // speed with the wobble filtered well below that.
        assert!((mean - SR * TICK_SECS).abs() < SR * 0.005, "mean {mean}");
        assert!(worst < mean * 0.15, "worst deviation {worst}, mean {mean}");
    }

    #[test]
    fn forward_seek_snaps_instead_of_gliding() {
        let t0 = Instant::now();
        let mut sm = PlayheadSmoother::new(t0);
        run_steady(&mut sm, t0, 1, 30);
        let target = raw_at(30.0 * TICK_SECS) + (5.0 * SR) as usize;
        let painted = sm.tick(target, SR, t0 + Duration::from_secs_f64(31.0 * TICK_SECS));
        assert!((painted - target as f64).abs() < 1.0, "painted {painted}");
    }

    #[test]
    fn loop_wrap_snaps_backward() {
        let t0 = Instant::now();
        let mut sm = PlayheadSmoother::new(t0);
        run_steady(&mut sm, t0, 1, 30);
        // A loop wrap moves the raw playhead backward by the loop length —
        // beyond the snap threshold, so the display jumps with it.
        let target = raw_at(30.0 * TICK_SECS).saturating_sub((0.5 * SR) as usize);
        let painted = sm.tick(target, SR, t0 + Duration::from_secs_f64(31.0 * TICK_SECS));
        assert!(
            (painted - target as f64).abs() < SR * TICK_SECS * 2.0,
            "painted {painted}, target {target}"
        );
    }

    #[test]
    fn splice_backward_step_never_moves_display_backward() {
        let t0 = Instant::now();
        let mut sm = PlayheadSmoother::new(t0);
        run_steady(&mut sm, t0, 1, 30);
        // The keylock engine re-publishes a slightly earlier source
        // position around a splice. The painted playhead must keep moving
        // forward at near-nominal speed — a backward blip on screen is the
        // "twitch right and snap back" artifact.
        let shift = (0.03 * SR) as usize;
        let mut last = sm.tick(
            raw_at(30.0 * TICK_SECS).saturating_sub(shift),
            SR,
            t0 + Duration::from_secs_f64(30.0 * TICK_SECS),
        );
        for i in 31..80 {
            let t = i as f64 * TICK_SECS;
            let painted = sm.tick(
                raw_at(t).saturating_sub(shift),
                SR,
                t0 + Duration::from_secs_f64(t),
            );
            let delta = painted - last;
            assert!(delta > 0.0, "tick {i}: display moved backward ({delta})");
            assert!(
                (delta - SR * TICK_SECS).abs()
                    < SR * TICK_SECS * PLAYHEAD_NOISE_MAX_DEVIATION * 1.5,
                "tick {i}: delta {delta} deviates from nominal {}",
                SR * TICK_SECS
            );
            last = painted;
        }
    }

    #[test]
    fn splice_sized_jump_bleeds_off_without_lurch() {
        let t0 = Instant::now();
        let mut sm = PlayheadSmoother::new(t0);
        run_steady(&mut sm, t0, 1, 30);
        // The keylock engine re-anchors its consumption by a splice-sized
        // 40 ms at a beat; the painted scroll rate must stay within the
        // noise-band velocity cap instead of reproducing the lurch.
        let jump = (0.04 * SR) as usize;
        let mut last = sm.tick(
            raw_at(30.0 * TICK_SECS) + jump,
            SR,
            t0 + Duration::from_secs_f64(30.0 * TICK_SECS),
        );
        let mut worst_dev = 0.0f64;
        for i in 31..80 {
            let t = i as f64 * TICK_SECS;
            let painted = sm.tick(raw_at(t) + jump, SR, t0 + Duration::from_secs_f64(t));
            worst_dev = worst_dev.max((painted - last - SR * TICK_SECS).abs());
            last = painted;
        }
        assert!(
            worst_dev < SR * TICK_SECS * PLAYHEAD_NOISE_MAX_DEVIATION * 1.5,
            "worst deviation {worst_dev} frames"
        );
    }

    #[test]
    fn backward_glide_tracks_without_snapping() {
        let t0 = Instant::now();
        let mut sm = PlayheadSmoother::new(t0);
        // A scrub throw moving backward at 2x, published per output block.
        let rate = -2.0 * SR;
        let start = 8.0 * SR;
        sm.reset(start as usize, t0);
        let mut last = start;
        for i in 1..40 {
            let t = i as f64 * TICK_SECS;
            let raw = ((start + rate * t) as usize / BUFFER) * BUFFER;
            let painted = sm.tick(raw, rate, t0 + Duration::from_secs_f64(t));
            let delta = painted - last;
            last = painted;
            if i > 5 {
                assert!(
                    (delta - rate * TICK_SECS).abs() < rate.abs() * TICK_SECS * 0.15,
                    "tick {i}: delta {delta}, nominal {}",
                    rate * TICK_SECS
                );
            }
        }
    }

    #[test]
    fn stalled_source_freezes_the_display() {
        let t0 = Instant::now();
        let mut sm = PlayheadSmoother::new(t0);
        run_steady(&mut sm, t0, 1, 30);
        let frozen = raw_at(30.0 * TICK_SECS);
        // Underrun/EOF drain: the raw value stops moving. The display may
        // coast a couple of buffers past it but must then hold still.
        let painted: Vec<f64> = (31..61)
            .map(|i| {
                let t = i as f64 * TICK_SECS;
                sm.tick(frozen, SR, t0 + Duration::from_secs_f64(t))
            })
            .collect();
        let last = *painted.last().unwrap();
        assert!(
            last - frozen as f64 <= SR * 0.25,
            "coasted too far: {} past the stall",
            last - frozen as f64
        );
        let late_drift = painted[29] - painted[24];
        assert!(late_drift < 1.0, "still moving during stall: {late_drift}");
    }
}

use eframe::egui;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, TryRecvError};
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
/// The Wide chain's true tempo-rate floor: its PV head clamps the ratio
/// to [0.5, 2.0] (`wide_pv_head.rs`), so requests below rate 0.5 pin
/// silently at -50%. The wide-fader brake takes over from here down.
const WIDE_MIN_TEMPO_RATE: f64 = 0.5;
/// Tempo fader width in points (3x egui's ~100pt default) for fine control.
const TEMPO_SLIDER_WIDTH: f32 = 300.0;
/// Auto-loop length ladder: `2^exp` beats. 1/8 beat is still thousands of
/// frames at DJ tempos — far more than the deck feed needs per wrap.
const LOOP_EXP_MIN: i32 = -3;
/// Ladder ceiling: 32 beats (8 bars).
const LOOP_EXP_MAX: i32 = 5;

/// How a fader target maps onto the engine and the wide-fader brake.
struct TempoMapping {
    /// Engine stretch ratio (output/input length), clamped to the active
    /// range's honest span — sub-floor targets pin at the ceiling.
    ratio: f64,
    /// Post-engine brake factor `b` in `[0, 1]`; 1.0 above the Wide
    /// chain's -50% floor.
    brake: f64,
    /// BPM that actually plays: `detected · (1/ratio) · brake`. Equals the
    /// fader value throughout the braked zone, so the fader never snaps.
    effective_bpm: f64,
}

/// Splits a target BPM into the engine's stretch ratio and the Wide
/// range's sub-floor brake factor. The wide chain can't stretch below
/// rate 0.5 (-50%, its PV head's ratio clamp); a CDJ-3000's WIDE fader
/// reaches -100% by stopping the platter, which the deck reproduces by
/// pinning the engine at that floor and braking its output (see
/// `brake.rs`). Standard range never brakes.
fn tempo_mapping(range: DeckRange, detected_bpm: f64, target_bpm: f64) -> TempoMapping {
    let max_ratio = match range {
        DeckRange::Standard => MAX_VARISPEED_RATIO,
        // Requests past the wide floor would pin silently inside the
        // chain — cap the ratio where the chain's honesty ends and let
        // the brake carry the rest.
        DeckRange::Wide => 1.0 / WIDE_MIN_TEMPO_RATE,
    };
    let ratio = (detected_bpm / target_bpm.max(1e-6)).clamp(MIN_STRETCH_RATIO, max_ratio);
    let brake = if range == DeckRange::Wide {
        let desired_rate = target_bpm.max(0.0) / detected_bpm.max(1e-6);
        (desired_rate / WIDE_MIN_TEMPO_RATE).clamp(0.0, 1.0)
    } else {
        1.0
    };
    TempoMapping {
        ratio,
        brake,
        effective_bpm: detected_bpm * (1.0 / ratio) * brake,
    }
}

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

/// Repaint pacing while the deck is playing, in milliseconds.
/// `TIMESTRETCH_REPAINT_MS` overrides it; `0` presents every vsync
/// (the old behavior, which makes ProMotion hunt and skip slots).
const PLAYING_REPAINT_MS: u64 = 33;

/// The playing-state repaint interval: the default pacing unless
/// overridden by `TIMESTRETCH_REPAINT_MS` (read once); `None` = uncapped.
fn playing_repaint_interval() -> Option<std::time::Duration> {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let ms = *MS.get_or_init(|| {
        std::env::var("TIMESTRETCH_REPAINT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(PLAYING_REPAINT_MS)
    });
    (ms > 0).then(|| std::time::Duration::from_millis(ms))
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

/// Messages from the background load worker, in order: one `Track`, then
/// (after a successful decode) one `Analysis` when the pre-analysis
/// artifact lands.
enum LoadMsg {
    Track(Result<LoadedTrack, String>),
    Analysis(Arc<timestretch::PreAnalysisArtifact>),
}

/// Everything `install_track` needs once decode + peaks complete.
struct LoadedTrack {
    path: PathBuf,
    /// Interleaved samples, as decoded.
    samples: Arc<Vec<f32>>,
    sample_rate: u32,
    num_frames: usize,
    peaks: BandPeaks,
}

/// An in-flight background load, replaced wholesale on each `load_file`:
/// dropping the old receiver makes the stale worker's sends fail silently.
struct LoadInProgress {
    rx: mpsc::Receiver<LoadMsg>,
    /// True once `Track` has been installed (spinner off; may still be
    /// awaiting `Analysis` on a pre-analysis cache miss).
    track_received: bool,
}

/// Bundled inputs for [`run_load_worker`].
struct LoadRequest {
    path: PathBuf,
    state: SharedStateHandle,
    /// `analysis_generation` at spawn; results are discarded on mismatch.
    generation: u64,
    tx: mpsc::Sender<LoadMsg>,
    /// For waking the UI when results land (the app may be idle-paused).
    ctx: egui::Context,
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
    /// Background decode + analysis in flight; polled each `update`.
    pending_load: Option<LoadInProgress>,
    /// Context handle for background threads to request repaints — also
    /// covers the argv initial-file load spawned before the first frame.
    egui_ctx: egui::Context,

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

    /// Wide-fader brake factor `b` in `[0, 1]` (1.0 = no brake), shared
    /// lock-free with the audio callback and the deck thread. Below the
    /// engine's -75% floor the Wide range's fader drives this instead of
    /// the stretch ratio; see `brake.rs`.
    brake_shared: Arc<AtomicRate>,

    // UI state
    stretch_ratio: f64,
    /// UI-side copy of the published brake factor.
    brake: f64,
    /// Target playback BPM the tempo fader binds to (0.0 until a BPM is
    /// detected). Derived view of `stretch_ratio`, kept in sync.
    target_bpm: f64,
    volume: f32,
    preset: PresetChoice,
    deck_engine: DeckEngine,
    deck_range: DeckRange,
    target_bpm_text: String,
    /// Auto-loop length as a ladder exponent (`2^exp` beats).
    loop_beats_exp: i32,

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

    /// Sets the playback tempo to `target_bpm` (for a track at `detected_bpm`)
    /// and syncs every derived view — stretch ratio, brake factor, the fader's
    /// `target_bpm`, and the BPM text box — to the value that actually plays
    /// after the engine's ratio clamp. Single write point for the tempo
    /// control.
    fn apply_target_bpm(&mut self, detected_bpm: f64, target_bpm: f64) {
        let m = tempo_mapping(self.deck_range, detected_bpm, target_bpm);
        self.stretch_ratio = m.ratio;
        self.brake = m.brake;
        self.brake_shared.store(m.brake);
        self.target_bpm = m.effective_bpm;
        self.target_bpm_text = format!("{:.1}", m.effective_bpm);
        let mut st = self.state.lock().unwrap();
        st.stretch_ratio = m.ratio;
        st.target_bpm = m.effective_bpm;
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
            pending_load: None,
            egui_ctx: cc.egui_ctx.clone(),
            band_peaks: None,
            overview_texture: None,
            zoom_span: ZoomSpan::default(),
            zoom_tiles: ZoomedTiles::default(),
            beat_grid: None,
            grid_marks: GridMarks::empty(),
            brake_shared: Arc::new(AtomicRate::new(1.0)),
            stretch_ratio: 1.0,
            brake: 1.0,
            target_bpm: 0.0,
            volume: 0.8,
            preset: PresetChoice::DjBeatmatch,
            deck_engine: DeckEngine::Keylock,
            deck_range: DeckRange::Standard,
            target_bpm_text: String::new(),
            loop_beats_exp: 2,
            error_message: None,
        };
        if let Some(path) = initial_file {
            app.load_file(path);
        }
        app
    }

    /// Kicks off a background load: decode, peaks (cached in a `.tspeaks`
    /// sidecar), and pre-analysis all run on a worker thread, so the UI
    /// stays responsive however long the track is. Results arrive through
    /// [`Self::poll_pending_load`].
    fn load_file(&mut self, path: PathBuf) {
        // Stop any existing playback
        self.stop_playback();

        self.error_message = None;
        self.file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        log::info!("Loading file: {}", path.display());

        // Clear the old track everywhere before the worker starts — the
        // deck must not stay interactable against audio that's going away.
        self.file_path = None;
        self.source_audio = None;
        self.band_peaks = None;
        self.overview_texture = None;
        self.zoom_tiles.clear();
        self.grid_marks = GridMarks::empty();
        self.beat_grid = None;
        // A fresh track starts at its own tempo (fader centered) and
        // unity ratio, regardless of any prior track's settings.
        self.stretch_ratio = 1.0;
        self.brake = 1.0;
        self.brake_shared.store(1.0);
        self.target_bpm = 0.0;
        self.target_bpm_text.clear();
        self.position.store(0);

        // The generation bump happens HERE, at spawn — not again at
        // install, or every worker's artifact would look stale.
        let generation = {
            let mut st = self.state.lock().unwrap();
            st.total_frames = 0;
            st.position_frames = 0;
            st.detected_bpm = 0.0;
            st.target_bpm = 0.0;
            st.transport = Transport::Stopped;
            st.stretch_ratio = 1.0;
            st.pre_analysis = None;
            st.loop_region = None;
            st.loop_in = None;
            st.analysis_generation += 1;
            st.analysis_generation
        };

        let (tx, rx) = mpsc::channel();
        // Replacing the slot drops any previous receiver: a superseded
        // worker's sends fail silently and its artifact store is rejected
        // by the generation guard.
        self.pending_load = Some(LoadInProgress {
            rx,
            track_received: false,
        });
        let req = LoadRequest {
            path,
            state: self.state.clone(),
            generation,
            tx,
            ctx: self.egui_ctx.clone(),
        };
        std::thread::spawn(move || run_load_worker(req));
    }

    /// Drains the load worker's channel: installs the decoded track, then
    /// the analysis when it lands. Called at the top of every `update`.
    fn poll_pending_load(&mut self) {
        loop {
            let msg = match &self.pending_load {
                Some(pending) => pending.rx.try_recv(),
                None => return,
            };
            match msg {
                Ok(LoadMsg::Track(Ok(track))) => {
                    self.install_track(track);
                    if let Some(pending) = &mut self.pending_load {
                        pending.track_received = true;
                    }
                }
                Ok(LoadMsg::Track(Err(msg))) => {
                    log::error!("{msg}");
                    self.error_message = Some(msg);
                    self.pending_load = None;
                }
                Ok(LoadMsg::Analysis(artifact)) => {
                    self.install_analysis(&artifact);
                    self.pending_load = None;
                }
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    // Worker gone without a final message: panic safety net.
                    let had_track = self
                        .pending_load
                        .as_ref()
                        .is_some_and(|pending| pending.track_received);
                    self.pending_load = None;
                    if !had_track {
                        self.error_message = Some("Load thread exited unexpectedly".to_string());
                    }
                    return;
                }
            }
        }
    }

    /// Installs a completed decode + peaks result — everything the old
    /// synchronous load did except the beat grid, which arrives separately
    /// via [`LoadMsg::Analysis`].
    fn install_track(&mut self, track: LoadedTrack) {
        self.band_peaks = Some(track.peaks);
        self.overview_texture = None;
        self.zoom_tiles.clear();

        {
            let mut st = self.state.lock().unwrap();
            st.sample_rate = track.sample_rate;
            st.total_frames = track.num_frames;
            st.position_frames = 0;
            st.transport = Transport::Stopped;
            st.stretch_ratio = self.stretch_ratio;
            st.preset = self.preset;
        }

        self.source_audio = Some(track.samples);
        self.file_path = Some(track.path);
        self.output_sample_rate = track.sample_rate;
        self.position.store(0);
    }

    /// Installs the pre-analysis artifact as the UI's beat grid + BPM —
    /// the artifact is the same analysis the engine consumes, so the fader
    /// and the splice guidance now agree by construction.
    fn install_analysis(&mut self, artifact: &timestretch::PreAnalysisArtifact) {
        let grid = grid_from_artifact(artifact);
        let bpm = grid.bpm;
        log::info!(
            "Detected BPM: {bpm:.1} ({} beats, {} segments, confidence {:.2})",
            grid.beats.len(),
            grid.segments.len(),
            grid.confidence
        );
        self.grid_marks = GridMarks::from_grid(&grid);
        self.beat_grid = Some(grid);
        if bpm > 0.0 {
            // Honor whatever ratio is set by the time analysis lands
            // (unity on a fresh load): the fader binds to effective BPM.
            let effective = bpm / self.stretch_ratio.max(MIN_STRETCH_RATIO);
            self.target_bpm = effective;
            self.target_bpm_text = format!("{effective:.1}");
            let mut st = self.state.lock().unwrap();
            st.detected_bpm = bpm;
            st.target_bpm = effective;
        } else {
            self.target_bpm = 0.0;
            self.target_bpm_text.clear();
            let mut st = self.state.lock().unwrap();
            st.detected_bpm = 0.0;
            st.target_bpm = 0.0;
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
        // The tempo range picks the profile (Standard = primary keylock
        // chain, Wide = wide-range Master Tempo chain); the Tape/Keylock
        // deck mode stays a live engine parameter (delay-matched
        // crossfade) INSIDE whichever profile is running — switching deck
        // mode mid-play is instant, switching range is a seek-priced
        // rebuild handled by the range selector.
        let profile = self.state.lock().unwrap().deck_range.profile();
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
            self.brake_shared.clone(),
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
            self.brake_shared.clone(),
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
        // Install any background load results before painting from them.
        self.poll_pending_load();
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
                // The brake scales the audible rate below the Wide floor;
                // at a full stop the rate is 0 and the display pins.
                Some(sample_rate as f64 / self.stretch_ratio.max(MIN_STRETCH_RATIO) * self.brake)
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
        // Repaint on a steady timer while playing (a scrub glide animates
        // the playhead the same way even while paused, so it keeps the
        // repaint loop alive too). Steadiness beats rate here: presenting
        // every vsync makes ProMotion hunt for a refresh rate, and the
        // hunting transitions skip slots (~2/s) that read as scroll
        // twitches; a constant sub-max cadence lets the panel settle —
        // glass-measured 0 missed slots over 40 s at 33 ms, with wakes
        // arriving at <1 ms spread. The playhead smoother extrapolates
        // per measured dt, so the paced cadence stays spatially exact.
        if transport == Transport::Playing || scrub_phase != ScrubPhase::Idle {
            match playing_repaint_interval() {
                Some(interval) => ctx.request_repaint_after(interval),
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

            let decoding = self
                .pending_load
                .as_ref()
                .is_some_and(|pending| !pending.track_received);
            if decoding {
                ui.separator();
                ui.spinner();
                ui.label(format!("Loading {}…", self.file_name));
            } else if !self.file_name.is_empty() {
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
                    if self.pending_load.is_some() {
                        // Track installed; analysis still running.
                        ui.spinner();
                    }
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
                        // While the wide-fader brake is engaged the deck's
                        // audible rate is floor·b (= b/ratio), not 1.0 —
                        // ease the glide into the braked speed so the
                        // crossfade back to the (braked) engine doesn't
                        // lurch.
                        let play_rate = if self.brake < 1.0 {
                            self.brake / self.stretch_ratio.max(MIN_STRETCH_RATIO)
                        } else {
                            1.0
                        };
                        self.scrub.release(if playing { play_rate } else { 0.0 });
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
                        // 0.0 means "uninitialized" only while no brake is
                        // engaged — at a full WIDE stop the fader honestly
                        // reads 0.0 BPM and must not snap back.
                        if self.target_bpm <= 0.0 && self.brake >= 1.0 {
                            self.target_bpm = detected;
                        }
                        // The WIDE range reaches a CDJ-style full stop:
                        // below the wide chain's -50% floor the engine
                        // pins and the brake resampler takes the rest
                        // (see brake.rs).
                        let min_bpm = if self.deck_range == DeckRange::Wide {
                            0.0
                        } else {
                            detected / self.max_stretch_ratio()
                        };
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
                        let pct = ((1.0 / self.stretch_ratio) * self.brake - 1.0) * 100.0;
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
                    match (self.deck_engine, self.deck_range) {
                        (DeckEngine::Tape, _) => {
                            ui.label(egui::RichText::new("tape: pitch follows tempo").weak())
                                .on_hover_text(
                                    "Tape mode: pitch follows the fader (delay-matched \
                                     varispeed at the active range's constant pipeline \
                                     delay, so switching live is seamless).",
                                );
                        }
                        (DeckEngine::Keylock, DeckRange::Standard) => {
                            ui.label(egui::RichText::new("keylock: two-band").weak())
                                .on_hover_text(
                                    "Keylock mode: both bands pitch-corrected at the \
                                     delay-matched transposition — the bass corrector \
                                     engages beyond ~±1–2% deviation; mild nudges keep \
                                     pitch-follow bass (~13 ms pipeline delay).",
                                );
                        }
                        (DeckEngine::Keylock, DeckRange::Wide) => {
                            ui.label(egui::RichText::new("keylock: full-spectrum").weak())
                                .on_hover_text(
                                    "Wide-range Master Tempo: the full spectrum is \
                                     pitch-corrected across the whole tempo range \
                                     (0 ms pipeline delay — the analysis window is \
                                     source-side lookahead).",
                                );
                        }
                    }
                });
                ui.end_row();

                // Tempo range: Standard (primary keylock chain, ±20% full
                // keylock, ~13 ms) vs Wide (CDJ-style wide-range Master
                // Tempo, 0 ms — source-side lookahead). A range change
                // rebuilds the engine and restores the playhead via
                // warm-start seek.
                ui.label("Range:");
                ui.horizontal(|ui| {
                    let old_range = self.deck_range;
                    egui::ComboBox::from_id_salt("deck_range_combo")
                        .selected_text(self.deck_range.label())
                        .show_ui(ui, |ui| {
                            for &range in DeckRange::ALL {
                                ui.selectable_value(&mut self.deck_range, range, range.label());
                            }
                        });
                    if old_range.rebuild_needed(self.deck_range) {
                        // Leaving Wide while braked below the floor: the
                        // brake zone doesn't exist in Standard, so release
                        // it and let the ratio clamp snap the fader to the
                        // -75% floor before the rebuild.
                        if self.brake < 1.0 {
                            let detected = self.state.lock().unwrap().detected_bpm;
                            if detected > 0.0 {
                                self.apply_target_bpm(
                                    detected,
                                    self.target_bpm.max(detected / MAX_VARISPEED_RATIO),
                                );
                            } else {
                                self.brake = 1.0;
                                self.brake_shared.store(1.0);
                            }
                        }
                        let (transport, latency_secs) = {
                            let mut st = self.state.lock().unwrap();
                            st.deck_range = self.deck_range;
                            (st.transport, st.reported_latency_secs)
                        };
                        let _ = latency_secs;
                        if transport != Transport::Stopped {
                            // Seek-priced rebuild: keep the playhead (NOT
                            // stop_playback — that zeroes the position).
                            let pos = self.position.load();
                            self.stop_processing_thread();
                            self.start_playback();
                            self.request_seek(pos);
                        }
                    }
                    let latency_ms = {
                        let st = self.state.lock().unwrap();
                        st.reported_latency_secs.map(|secs| secs * 1_000.0)
                    };
                    // Show whatever the engine has reported — 0.0 ms is the
                    // Wide range's honest number (source-side lookahead),
                    // not "no info"; before the first build there is no
                    // figure to show.
                    if let Some(latency_ms) = latency_ms {
                        ui.label(egui::RichText::new(format!("{latency_ms:.1} ms")).weak())
                            .on_hover_text(
                                "Constant pipeline delay reported by the active engine \
                                 (host-compensated; each range has its own honest \
                                 figure).",
                            );
                    }
                });
                ui.end_row();

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

/// Legacy JSON artifact sidecar (`<file>.tsanalysis.json`): read only to
/// migrate old analyses into the `.tsa` container, deleted once superseded.
fn legacy_json_path(audio_path: &std::path::Path) -> PathBuf {
    let mut os = audio_path.as_os_str().to_os_string();
    os.push(".tsanalysis.json");
    PathBuf::from(os)
}

/// Legacy peaks sidecar (`<file>.tspeaks`): never read anymore — peaks
/// recompute in milliseconds — deleted once superseded by the container.
fn legacy_peaks_path(audio_path: &std::path::Path) -> PathBuf {
    let mut os = audio_path.as_os_str().to_os_string();
    os.push(".tspeaks");
    PathBuf::from(os)
}

/// The background load: decode, then the single `.tsa` analysis container
/// — peaks ship to the UI immediately, pre-analysis follows. One decode,
/// one downmix, one hash, shared by everything. Container writes are
/// best-effort: a read-only volume must not break loading. A valid legacy
/// `.tsanalysis.json` artifact is absorbed into the container (skipping
/// the slow re-analysis), and both legacy sidecars are deleted once the
/// on-disk container supersedes them.
///
/// Results are dropped if another file was loaded in the meantime (the
/// receiver is gone and the `generation` guard rejects the artifact).
fn run_load_worker(req: LoadRequest) {
    let LoadRequest {
        path,
        state,
        generation,
        tx,
        ctx,
    } = req;

    let decoded = match decoder::decode_file(&path) {
        Ok(d) => d,
        Err(e) => {
            let _ = tx.send(LoadMsg::Track(Err(format!("Failed to load: {e}"))));
            ctx.request_repaint();
            return;
        }
    };
    log::info!(
        "Decoded: {} frames, {} Hz, {} ch",
        decoded.num_frames,
        decoded.sample_rate,
        decoded.channels
    );
    let sample_rate = decoded.sample_rate;
    let num_frames = decoded.num_frames;
    let num_channels = (decoded.channels as usize).max(1);

    // One mono downmix + one hash: the container identity, the peaks
    // input, and the pre-analysis input.
    let mono = timestretch::downmix_to_mid(&decoded.samples, num_channels);
    let content_hash = timestretch::hash_samples(&mono);

    let tsa_path = timestretch::analysis_file_path(&path);
    let on_disk =
        timestretch::read_analysis_file_validated(&tsa_path, sample_rate, mono.len(), content_hash);
    // Whether the container on disk already supersedes both legacy
    // sidecars; kept true across best-effort rewrites of the same content.
    let mut persisted_complete = on_disk
        .as_ref()
        .is_some_and(|af| af.artifact.is_some() && af.peaks.is_some());
    let mut analysis = match on_disk {
        Some(af) => {
            log::info!("Analysis: using cached container {}", tsa_path.display());
            af
        }
        None => timestretch::AnalysisFile::for_source(&mono, sample_rate),
    };
    let mut dirty = false;

    // Legacy migration: absorb a still-valid JSON artifact so the slow
    // re-analysis is skipped. The legacy `.tspeaks` cache is deliberately
    // NOT read — recomputing peaks costs milliseconds, keeping its parser
    // alive costs a hundred lines.
    if analysis.artifact.is_none() {
        #[allow(deprecated)]
        let legacy = timestretch::read_preanalysis_json(&legacy_json_path(&path));
        if let Ok(legacy) = legacy
            && legacy.matches_source(&mono, sample_rate)
        {
            log::info!(
                "Pre-analysis: migrating legacy JSON sidecar into {}",
                tsa_path.display()
            );
            analysis.artifact = Some(legacy);
            dirty = true;
        }
    }

    if analysis.peaks.is_none() {
        let start = std::time::Instant::now();
        // Computed from the hashed mono signal (not the interleaved
        // stereo): identical bucket count, per-quantization-identical
        // values, half the filter work — and the persisted peaks derive
        // from exactly the signal that keys them.
        analysis.peaks = Some(BandPeaks::compute(&mono, 1, sample_rate));
        log::info!("Peaks: computed in {:.2}s", start.elapsed().as_secs_f64());
        dirty = true;
    }

    // Write #1, before the Track send: peaks (and any migrated artifact)
    // survive a crash during the slow analysis below.
    if dirty {
        match timestretch::write_analysis_file(&tsa_path, &analysis) {
            Ok(()) => {
                persisted_complete = analysis.artifact.is_some();
                dirty = false;
            }
            Err(e) => log::warn!("Analysis: could not write {}: {e}", tsa_path.display()),
        }
    }

    let track = LoadedTrack {
        path: path.clone(),
        samples: Arc::new(decoded.samples),
        sample_rate,
        num_frames,
        peaks: analysis.peaks.clone().expect("peaks were just ensured"),
    };
    if tx.send(LoadMsg::Track(Ok(track))).is_err() {
        // A newer load replaced this one; skip the expensive analysis too.
        return;
    }
    ctx.request_repaint();

    // Pre-analysis: beat grid + BPM for the UI, splice guidance for the
    // engine. Skip early when a newer load has already started.
    if state.lock().unwrap().analysis_generation != generation {
        return;
    }
    let artifact = match analysis.artifact.clone() {
        Some(cached) => cached,
        None => {
            let start = std::time::Instant::now();
            let fresh = timestretch::analyze_for_dj(&mono, sample_rate);
            log::info!(
                "Pre-analysis: {:.1} BPM, confidence {:.2}, {} beats, {} onsets ({:.2}s)",
                fresh.bpm,
                fresh.confidence,
                fresh.beat_positions.len(),
                fresh.transient_onsets.len(),
                start.elapsed().as_secs_f64()
            );
            analysis.artifact = Some(fresh.clone());
            dirty = true;
            fresh
        }
    };

    // Write #2: the container now carries both chunks. No read-modify-
    // write — this worker holds the whole file.
    if dirty {
        match timestretch::write_analysis_file(&tsa_path, &analysis) {
            Ok(()) => persisted_complete = true,
            Err(e) => log::warn!("Analysis: could not write {}: {e}", tsa_path.display()),
        }
    }

    // Once the on-disk container supersedes both legacy sidecars — it
    // holds a valid artifact AND peaks for exactly this audio — delete
    // them so tracks converge to the single `.tsa` file. Only these two
    // sibling paths are ever touched (Halo's `.halo.*` variants are not).
    if persisted_complete {
        for legacy in [legacy_json_path(&path), legacy_peaks_path(&path)] {
            match std::fs::remove_file(&legacy) {
                Ok(()) => log::info!("Removed superseded legacy sidecar {}", legacy.display()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => log::warn!("Could not remove legacy sidecar {}: {e}", legacy.display()),
            }
        }
    }

    let artifact = Arc::new(artifact);
    {
        // Store worker-side (not only via the channel) so `start_playback`
        // sees the artifact even if `update` hasn't polled — e.g. with the
        // window minimized.
        let mut st = state.lock().unwrap();
        if st.analysis_generation != generation {
            log::info!("Pre-analysis: discarding stale result (newer file loaded)");
            return;
        }
        st.pre_analysis = Some(artifact.clone());
    }
    let _ = tx.send(LoadMsg::Analysis(artifact));
    ctx.request_repaint();
}

/// Rebuilds the [`timestretch::BeatGrid`] view of a pre-analysis artifact:
/// beats, downbeats, and segments are stored verbatim in the artifact, so
/// this is a field-by-field copy. Keeps beat jumps and auto-loops working
/// from cached analysis without a duplicate detection pass.
fn grid_from_artifact(artifact: &timestretch::PreAnalysisArtifact) -> timestretch::BeatGrid {
    let mut grid = timestretch::BeatGrid::empty(artifact.sample_rate);
    grid.beats = if artifact.beat_positions_fractional.is_empty() {
        // Pre-fractional sidecars (can't occur at the current minimum
        // compatible version, but free to guard).
        artifact.beat_positions.iter().map(|&p| p as f64).collect()
    } else {
        artifact.beat_positions_fractional.clone()
    };
    grid.downbeats = artifact.downbeat_beat_indices.clone();
    grid.segments = artifact.tempo_segments.clone();
    grid.tempo_candidates = artifact.tempo_candidates.clone();
    grid.bpm = artifact.bpm;
    grid.confidence = artifact.confidence;
    grid
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn grid_from_artifact_copies_beats_downbeats_segments() {
        let artifact = timestretch::PreAnalysisArtifact {
            sample_rate: 44_100,
            bpm: 123.4,
            confidence: 0.9,
            beat_positions_fractional: vec![100.5, 200.25, 300.0, 400.75],
            downbeat_beat_indices: vec![0, 2],
            tempo_segments: vec![timestretch::TempoSegment {
                start_beat: 0,
                bpm: 123.4,
            }],
            ..Default::default()
        };
        let grid = grid_from_artifact(&artifact);
        assert_eq!(grid.beats, artifact.beat_positions_fractional);
        assert_eq!(grid.downbeats, artifact.downbeat_beat_indices);
        assert_eq!(grid.segments.len(), 1);
        assert_eq!(grid.bpm, 123.4);
        assert_eq!(grid.sample_rate, 44_100);
        let marks = GridMarks::from_grid(&grid);
        assert!(marks.is_usable());
        assert_eq!(marks.downbeat_count(), 2);
    }

    #[test]
    fn grid_from_artifact_falls_back_to_integer_positions() {
        let artifact = timestretch::PreAnalysisArtifact {
            sample_rate: 44_100,
            bpm: 120.0,
            beat_positions: vec![100, 200, 300],
            ..Default::default()
        };
        let grid = grid_from_artifact(&artifact);
        assert_eq!(grid.beats, vec![100.0, 200.0, 300.0]);
    }

    #[test]
    fn grid_from_artifact_empty_artifact_yields_unusable_grid() {
        let grid = grid_from_artifact(&timestretch::PreAnalysisArtifact::default());
        assert!(grid.beats.is_empty());
        assert!(!GridMarks::from_grid(&grid).is_usable());
    }

    /// Unique temp dir per test.
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tsload_test_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Hand-writes a minimal mono 16-bit PCM WAV: a steady 120 BPM kick-ish
    /// pulse train so beat analysis has something to find.
    fn write_test_wav(path: &std::path::Path, secs: f64) {
        let sr = 44_100u32;
        let n = (secs * sr as f64) as usize;
        let mut pcm = Vec::with_capacity(n * 2);
        let beat_period = (sr as f64 * 0.5) as usize; // 120 BPM
        for i in 0..n {
            let since_beat = i % beat_period;
            // 60 Hz burst with a fast decay after each beat onset.
            let env = (-(since_beat as f64) / (sr as f64 * 0.05)).exp();
            let s =
                (0.8 * env * (std::f64::consts::TAU * 60.0 * i as f64 / sr as f64).sin()) as f32;
            pcm.extend_from_slice(&((s * i16::MAX as f32) as i16).to_le_bytes());
        }
        let data_len = pcm.len() as u32;
        let mut wav = Vec::with_capacity(44 + pcm.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&sr.to_le_bytes());
        wav.extend_from_slice(&(sr * 2).to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.extend_from_slice(&pcm);
        std::fs::write(path, wav).unwrap();
    }

    fn worker_request(
        wav: &std::path::Path,
        state: &SharedStateHandle,
        generation: u64,
    ) -> (LoadRequest, mpsc::Receiver<LoadMsg>) {
        let (tx, rx) = mpsc::channel();
        (
            LoadRequest {
                path: wav.to_path_buf(),
                state: state.clone(),
                generation,
                tx,
                ctx: egui::Context::default(),
            },
            rx,
        )
    }

    #[test]
    fn load_worker_writes_single_tsa() {
        let dir = temp_dir("worker");
        let wav = dir.join("track.wav");
        write_test_wav(&wav, 4.0);
        let state: SharedStateHandle = Arc::new(Mutex::new(SharedState::new()));
        state.lock().unwrap().analysis_generation = 7;

        let (req, rx) = worker_request(&wav, &state, 7);
        run_load_worker(req);

        let first = rx.recv().expect("worker must send a Track message");
        let LoadMsg::Track(Ok(track)) = first else {
            panic!("expected Track(Ok), got an error or Analysis first");
        };
        assert_eq!(track.sample_rate, 44_100);
        assert_eq!(track.num_frames, 4 * 44_100);
        let LoadMsg::Analysis(artifact) = rx.recv().expect("worker must send Analysis") else {
            panic!("expected Analysis after Track");
        };
        assert!(artifact.bpm > 0.0, "pulse train should yield a BPM");
        assert!(state.lock().unwrap().pre_analysis.is_some());
        // One sidecar, suffix-append convention — and no legacy files.
        assert!(dir.join("track.wav.tsa").exists());
        assert!(!dir.join("track.wav.tspeaks").exists());
        assert!(!dir.join("track.wav.tsanalysis.json").exists());

        // Second run against the same file: pure container hit; results
        // equivalent.
        let (req2, rx2) = worker_request(&wav, &state, 7);
        run_load_worker(req2);
        let LoadMsg::Track(Ok(track2)) = rx2.recv().unwrap() else {
            panic!("cache-hit run must still yield Track(Ok)");
        };
        assert_eq!(track2.num_frames, track.num_frames);
        let base = track.peaks.level(0);
        let base2 = track2.peaks.level(0);
        assert_eq!(base.num_buckets(), base2.num_buckets());
        // The cached pyramid is quantized; the fresh one wasn't. Values
        // must agree within one quantization step.
        for band in 0..waveform::NUM_BANDS {
            for (a, b) in base.pos[band].iter().zip(&base2.pos[band]) {
                assert!((a - b).abs() <= 1.0 / 255.0 + f32::EPSILON);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Run one worker and drain its channel; panics on decode errors.
    fn run_worker_to_completion(wav: &std::path::Path, state: &SharedStateHandle, generation: u64) {
        let (req, rx) = worker_request(wav, state, generation);
        run_load_worker(req);
        while let Ok(msg) = rx.recv() {
            if let LoadMsg::Track(Err(e)) = msg {
                panic!("worker failed: {e}");
            }
        }
    }

    /// Analyze the test WAV's mono signal the same way the worker does.
    fn analyzed_artifact(wav: &std::path::Path) -> timestretch::PreAnalysisArtifact {
        let decoded = decoder::decode_file(wav).unwrap();
        let mono =
            timestretch::downmix_to_mid(&decoded.samples, (decoded.channels as usize).max(1));
        timestretch::analyze_for_dj(&mono, decoded.sample_rate)
    }

    #[test]
    fn load_worker_migrates_legacy_json() {
        let dir = temp_dir("migrate");
        let wav = dir.join("track.wav");
        write_test_wav(&wav, 4.0);
        // Pre-seed a valid legacy JSON artifact and a garbage .tspeaks
        // (its contents are never read — only superseded and deleted). The
        // distinctive confidence value marks this exact artifact: a fresh
        // re-analysis would never reproduce it, so its presence in the
        // container proves absorption. (Exact float equality is off the
        // table — serde_json's default f64 parsing can be one ulp off.)
        let mut real = analyzed_artifact(&wav);
        real.confidence = 0.4242;
        #[allow(deprecated)]
        timestretch::write_preanalysis_json(&legacy_json_path(&wav), &real).unwrap();
        std::fs::write(legacy_peaks_path(&wav), b"garbage bytes").unwrap();

        let state: SharedStateHandle = Arc::new(Mutex::new(SharedState::new()));
        run_worker_to_completion(&wav, &state, 0);

        // The marked artifact was absorbed (not re-analyzed), the container
        // is complete, the legacy files are gone.
        let tsa = timestretch::read_analysis_file(&dir.join("track.wav.tsa")).unwrap();
        let migrated = tsa.artifact.expect("artifact chunk present");
        assert!(
            (migrated.confidence - 0.4242).abs() < 1e-6,
            "marker confidence proves absorption, got {}",
            migrated.confidence
        );
        assert!((migrated.bpm - real.bpm).abs() < 1e-9);
        assert_eq!(migrated.beat_positions, real.beat_positions);
        assert!(tsa.peaks.is_some(), "peaks chunk present");
        assert!(!legacy_json_path(&wav).exists(), "legacy JSON deleted");
        assert!(!legacy_peaks_path(&wav).exists(), "legacy .tspeaks deleted");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_worker_deletes_partial_legacy_files() {
        // Each legacy file alone: absorbed/superseded, then deleted.
        for (tag, json, tspeaks) in [("json_only", true, false), ("tspeaks_only", false, true)] {
            let dir = temp_dir(tag);
            let wav = dir.join("track.wav");
            write_test_wav(&wav, 2.0);
            if json {
                let real = analyzed_artifact(&wav);
                #[allow(deprecated)]
                timestretch::write_preanalysis_json(&legacy_json_path(&wav), &real).unwrap();
            }
            if tspeaks {
                std::fs::write(legacy_peaks_path(&wav), b"garbage").unwrap();
            }
            let state: SharedStateHandle = Arc::new(Mutex::new(SharedState::new()));
            run_worker_to_completion(&wav, &state, 0);
            assert!(dir.join("track.wav.tsa").exists(), "{tag}: .tsa written");
            assert!(!legacy_json_path(&wav).exists(), "{tag}: JSON gone");
            assert!(!legacy_peaks_path(&wav).exists(), "{tag}: .tspeaks gone");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn load_worker_stale_legacy_json_not_absorbed() {
        let dir = temp_dir("stale_json");
        let wav = dir.join("track.wav");
        write_test_wav(&wav, 2.0);
        // A legacy artifact whose content hash mismatches this audio.
        let stale = timestretch::PreAnalysisArtifact {
            version: timestretch::PREANALYSIS_VERSION,
            sample_rate: 44_100,
            bpm: 99.9,
            source_len_samples: 12_345,
            content_hash: 0xDEAD_BEEF,
            ..Default::default()
        };
        #[allow(deprecated)]
        timestretch::write_preanalysis_json(&legacy_json_path(&wav), &stale).unwrap();

        let state: SharedStateHandle = Arc::new(Mutex::new(SharedState::new()));
        run_worker_to_completion(&wav, &state, 0);

        // Fresh analysis ran (not the stale 99.9 BPM), and the stale JSON
        // was still deleted — superseded by the fresh ARTF chunk.
        let tsa = timestretch::read_analysis_file(&dir.join("track.wav.tsa")).unwrap();
        let artifact = tsa.artifact.expect("fresh artifact present");
        assert_ne!(artifact.bpm, 99.9, "stale artifact must not be absorbed");
        assert!(!legacy_json_path(&wav).exists(), "stale JSON deleted");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_worker_decode_error_sends_err() {
        let dir = temp_dir("decode_err");
        let bogus = dir.join("not_audio.wav");
        std::fs::write(&bogus, b"this is not a wav file").unwrap();
        let state: SharedStateHandle = Arc::new(Mutex::new(SharedState::new()));
        let (req, rx) = worker_request(&bogus, &state, 1);
        run_load_worker(req);
        let LoadMsg::Track(Err(msg)) = rx.recv().unwrap() else {
            panic!("expected Track(Err) for garbage input");
        };
        assert!(msg.starts_with("Failed to load"));
        assert!(
            rx.recv().is_err(),
            "no Analysis after a failed decode; channel must just close"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_generation_skips_artifact_store() {
        let dir = temp_dir("stale");
        let wav = dir.join("track.wav");
        write_test_wav(&wav, 2.0);
        let state: SharedStateHandle = Arc::new(Mutex::new(SharedState::new()));
        // The worker was spawned at generation 3, but a newer load bumped
        // the shared state to 4 before analysis finished.
        state.lock().unwrap().analysis_generation = 4;
        let (req, rx) = worker_request(&wav, &state, 3);
        run_load_worker(req);
        let LoadMsg::Track(Ok(_)) = rx.recv().unwrap() else {
            panic!("decode + peaks still complete for a stale worker");
        };
        assert!(
            rx.recv().is_err(),
            "stale worker must skip analysis entirely"
        );
        assert!(state.lock().unwrap().pre_analysis.is_none());
        // Write #1 still persisted the peaks — but no artifact was
        // computed for the stale generation.
        let tsa = timestretch::read_analysis_file(&dir.join("track.wav.tsa")).unwrap();
        assert!(tsa.peaks.is_some(), "peaks persisted before the bail");
        assert!(tsa.artifact.is_none(), "no artifact for a stale worker");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deck_range_maps_to_profiles_and_labels() {
        assert_eq!(DeckRange::ALL.len(), 2);
        assert_eq!(
            DeckRange::Standard.profile(),
            timestretch::engine::EngineProfile::Keylock
        );
        assert_eq!(
            DeckRange::Wide.profile(),
            timestretch::engine::EngineProfile::WideKeylock
        );
        assert_eq!(DeckRange::Standard.label(), "Standard");
        assert_eq!(DeckRange::Wide.label(), "Wide");
    }

    #[test]
    fn tempo_mapping_is_continuous_at_the_wide_floor() {
        // Exactly at the wide chain's -50% floor: ratio pinned, brake
        // exactly 1.0 — the brake resampler engages with zero effect, so
        // the handoff between the direct and braked paths is seamless.
        let m = tempo_mapping(DeckRange::Wide, 128.0, 64.0);
        assert_eq!(m.ratio, 2.0);
        assert_eq!(m.brake, 1.0);
        assert!((m.effective_bpm - 64.0).abs() < 1e-9);
    }

    #[test]
    fn tempo_mapping_brakes_below_the_wide_floor() {
        // Halfway between -50% and -100%: engine stays pinned at the
        // floor, the brake takes the rest, and the effective BPM equals
        // the fader value (no snap).
        let m = tempo_mapping(DeckRange::Wide, 128.0, 32.0);
        assert_eq!(m.ratio, 2.0);
        assert!((m.brake - 0.5).abs() < 1e-9);
        assert!((m.effective_bpm - 32.0).abs() < 1e-9);

        let stop = tempo_mapping(DeckRange::Wide, 128.0, 0.0);
        assert_eq!(stop.ratio, 2.0);
        assert_eq!(stop.brake, 0.0);
        assert_eq!(stop.effective_bpm, 0.0);
    }

    #[test]
    fn tempo_mapping_never_brakes_in_standard_range() {
        for target in [0.0, 16.0, 32.0, 128.0, 256.0] {
            let m = tempo_mapping(DeckRange::Standard, 128.0, target);
            assert_eq!(m.brake, 1.0);
            assert!((0.25..=4.0).contains(&m.ratio));
        }
    }

    #[test]
    fn only_range_changes_require_a_rebuild() {
        assert!(DeckRange::Standard.rebuild_needed(DeckRange::Wide));
        assert!(DeckRange::Wide.rebuild_needed(DeckRange::Standard));
        assert!(!DeckRange::Standard.rebuild_needed(DeckRange::Standard));
        assert!(!DeckRange::Wide.rebuild_needed(DeckRange::Wide));
    }

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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use timestretch::PreAnalysisArtifact;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transport {
    Stopped,
    Playing,
    Paused,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PresetChoice {
    None,
    DjBeatmatch,
    HouseLoop,
    Halftime,
    Ambient,
    VocalChop,
}

impl PresetChoice {
    pub const ALL: &'static [PresetChoice] = &[
        PresetChoice::None,
        PresetChoice::DjBeatmatch,
        PresetChoice::HouseLoop,
        PresetChoice::Halftime,
        PresetChoice::Ambient,
        PresetChoice::VocalChop,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            PresetChoice::None => "None",
            PresetChoice::DjBeatmatch => "DJ Beatmatch",
            PresetChoice::HouseLoop => "House Loop",
            PresetChoice::Halftime => "Halftime",
            PresetChoice::Ambient => "Ambient",
            PresetChoice::VocalChop => "Vocal Chop",
        }
    }
}

/// Which engine mode drives the deck.
///
/// `Tape` runs the engine in tape mode (pitch follows tempo, zero pipeline
/// delay); `Keylock` preserves pitch while the tempo changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeckEngine {
    Tape,
    Keylock,
}

impl DeckEngine {
    pub const ALL: &'static [DeckEngine] = &[DeckEngine::Tape, DeckEngine::Keylock];

    pub fn label(&self) -> &'static str {
        match self {
            DeckEngine::Tape => "Tape",
            DeckEngine::Keylock => "Keylock",
        }
    }
}

/// State shared between UI, processing, and audio threads.
pub struct SharedState {
    pub transport: Transport,
    pub stretch_ratio: f64,
    pub volume: f32,
    pub preset: PresetChoice,
    /// Deck engine architecture. Takes effect at the next playback start
    /// (the two pipelines are wired completely differently).
    pub deck_engine: DeckEngine,

    /// Current playback position in source frames.
    pub position_frames: usize,
    /// Total source frames (per channel).
    pub total_frames: usize,
    pub sample_rate: u32,

    /// Detected BPM of the source audio.
    pub detected_bpm: f64,
    /// Target BPM entered by user (0.0 = use stretch ratio directly).
    pub target_bpm: f64,

    /// Set by UI to request a seek.
    pub seek_request: Option<usize>,

    /// Constant pipeline (content) delay reported by the active processor,
    /// in seconds. Published by the processing thread at every build; read
    /// by the UI next to the profile selector.
    pub reported_latency_secs: f64,
    /// Tempo control-to-audio latency reported by the active processor, in
    /// seconds (excludes the callback size). Near-zero on the varispeed-
    /// first path; equals the pipeline delay on the vocoder path.
    pub reported_control_latency_secs: f64,

    /// Offline pre-analysis of the loaded track (beat grid, onsets).
    ///
    /// Arrives asynchronously from the analysis thread after load and is
    /// consumed at the next processor rebuild (seek/preset/EOF); running
    /// processors are never hot-swapped.
    pub pre_analysis: Option<Arc<PreAnalysisArtifact>>,
    /// Bumped on every file load; a background analysis result is discarded
    /// unless its generation still matches (guards rapid successive loads).
    pub analysis_generation: u64,

    /// Active loop region in source frames, if looping is engaged. When set,
    /// the processing thread wraps `[start, end)` gaplessly via the library's
    /// warm-start machinery.
    pub loop_region: Option<(usize, usize)>,
    /// Loop-in point staged by the UI (source frame) before the loop-out is
    /// set to complete a region.
    pub loop_in: Option<usize>,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            transport: Transport::Stopped,
            stretch_ratio: 1.0,
            volume: 0.8,
            preset: PresetChoice::DjBeatmatch,
            deck_engine: DeckEngine::Keylock,
            position_frames: 0,
            total_frames: 0,
            sample_rate: 44100,
            detected_bpm: 0.0,
            target_bpm: 0.0,
            seek_request: None,
            reported_latency_secs: 0.0,
            reported_control_latency_secs: 0.0,
            pre_analysis: None,
            analysis_generation: 0,
            loop_region: None,
            loop_in: None,
        }
    }

    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.total_frames as f64 / self.sample_rate as f64
    }

    pub fn position_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.position_frames as f64 / self.sample_rate as f64
    }
}

/// Atomic position counter for lock-free updates from processing thread.
pub struct AtomicPosition {
    frames: AtomicU64,
}

impl AtomicPosition {
    pub fn new() -> Self {
        Self {
            frames: AtomicU64::new(0),
        }
    }

    pub fn store(&self, frames: usize) {
        self.frames.store(frames as u64, Ordering::Relaxed);
    }

    pub fn load(&self) -> usize {
        self.frames.load(Ordering::Relaxed) as usize
    }
}

/// Flag for signaling the processing thread to stop.
pub struct StopFlag {
    flag: AtomicBool,
}

impl StopFlag {
    pub fn new() -> Self {
        Self {
            flag: AtomicBool::new(false),
        }
    }

    pub fn set(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    pub fn is_set(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

pub type SharedStateHandle = Arc<Mutex<SharedState>>;

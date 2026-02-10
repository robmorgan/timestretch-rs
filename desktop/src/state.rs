use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use timestretch::EdmPreset;

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

    pub fn to_edm_preset(&self) -> Option<EdmPreset> {
        match self {
            PresetChoice::None => None,
            PresetChoice::DjBeatmatch => Some(EdmPreset::DjBeatmatch),
            PresetChoice::HouseLoop => Some(EdmPreset::HouseLoop),
            PresetChoice::Halftime => Some(EdmPreset::Halftime),
            PresetChoice::Ambient => Some(EdmPreset::Ambient),
            PresetChoice::VocalChop => Some(EdmPreset::VocalChop),
        }
    }
}

/// State shared between UI, processing, and audio threads.
pub struct SharedState {
    pub transport: Transport,
    pub stretch_ratio: f64,
    pub pitch_semitones: f32,
    pub volume: f32,
    pub preset: PresetChoice,

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
    /// Set by UI when pitch changes (requires re-processing).
    pub pitch_changed: bool,
    /// Set by UI when preset changes (requires rebuilding processor).
    pub preset_changed: bool,
    /// Set by processing thread when it finishes loading pitch-shifted audio.
    pub pitch_processing: bool,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            transport: Transport::Stopped,
            stretch_ratio: 1.0,
            pitch_semitones: 0.0,
            volume: 0.8,
            preset: PresetChoice::None,
            position_frames: 0,
            total_frames: 0,
            sample_rate: 44100,
            detected_bpm: 0.0,
            target_bpm: 0.0,
            seek_request: None,
            pitch_changed: false,
            preset_changed: false,
            pitch_processing: false,
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

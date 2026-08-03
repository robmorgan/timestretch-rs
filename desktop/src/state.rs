use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
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
    pub preset: PresetChoice,
    /// Deck engine mode. Forwarded live to the engine's keylock toggle by
    /// the deck thread (the engine always runs the keylock chain; Tape is
    /// the delay-matched varispeed bypass), so switching mid-play is
    /// instant — no rebuild, constant pipeline latency.
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
}

/// Lock-free output volume (f32 bits): the realtime audio callback must
/// never take the shared-state mutex — blocking on the UI thread mid-
/// layout risks output underruns, and the contention can push a UI frame
/// past its present deadline.
pub struct AtomicVolume {
    bits: AtomicU32,
}

impl AtomicVolume {
    pub fn new(volume: f32) -> Self {
        Self {
            bits: AtomicU32::new(volume.to_bits()),
        }
    }

    pub fn store(&self, volume: f32) {
        self.bits.store(volume.to_bits(), Ordering::Relaxed);
    }

    pub fn load(&self) -> f32 {
        f32::from_bits(self.bits.load(Ordering::Relaxed))
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

/// Scrub gesture lifecycle. `Active` while the pointer holds the waveform;
/// `Settling` after release while the voice's momentum eases into the
/// settle-rate target (CDJ vinyl release).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrubPhase {
    Idle,
    Active,
    Settling,
}

/// Shared scrub state machine: the UI publishes the pointer-implied source
/// position while the zoomed waveform is dragged; the audio callback chases
/// it with a raw varispeed reader (bypassing the engine), then owns the
/// post-release momentum glide. The deck thread yields the playhead while
/// any phase is engaged and additionally stops feeding while `Active`.
pub struct ScrubState {
    /// `ScrubPhase` as u8 (0/1/2).
    phase: AtomicU8,
    /// Pointer-target source frame as `f64` bits (valid while `Active`).
    target_frame: AtomicU64,
    /// Rate the release glide eases toward, as `f64` bits: 1.0 resumes
    /// playback speed, 0.0 spins down to rest.
    settle_rate_target: AtomicU64,
    /// Voice read position as `f64` bits, published by the audio callback
    /// every rendered block while engaged; the UI displays it during the
    /// glide and uses it as the re-grab base.
    voice_frame: AtomicU64,
    /// Voice rate as `f64` bits (source frames per output frame, sign =
    /// direction), published next to `voice_frame`; the UI's playhead
    /// smoother extrapolates the glide display with it.
    voice_rate: AtomicU64,
    /// Predicted settle landing frame as `f64` bits, published by the
    /// callback when a glide starts.
    landing: AtomicU64,
    /// Bumped with each published landing; the UI consumes each sequence
    /// number exactly once to fire the engine warm-start seek.
    landing_seq: AtomicU64,
}

impl ScrubState {
    pub fn new() -> Self {
        Self {
            phase: AtomicU8::new(ScrubPhase::Idle as u8),
            target_frame: AtomicU64::new(0.0f64.to_bits()),
            settle_rate_target: AtomicU64::new(0.0f64.to_bits()),
            voice_frame: AtomicU64::new(0.0f64.to_bits()),
            voice_rate: AtomicU64::new(0.0f64.to_bits()),
            landing: AtomicU64::new(0.0f64.to_bits()),
            landing_seq: AtomicU64::new(0),
        }
    }

    pub fn phase(&self) -> ScrubPhase {
        match self.phase.load(Ordering::Acquire) {
            1 => ScrubPhase::Active,
            2 => ScrubPhase::Settling,
            _ => ScrubPhase::Idle,
        }
    }

    /// Engage the scrub at `frame` (the playhead where the drag started, or
    /// the gliding voice position on a mid-settle re-grab). The target is
    /// published before the phase so the audio callback never sees a stale
    /// target on engage.
    pub fn begin(&self, frame: f64) {
        self.target_frame.store(frame.to_bits(), Ordering::Relaxed);
        self.voice_frame.store(frame.to_bits(), Ordering::Relaxed);
        self.phase
            .store(ScrubPhase::Active as u8, Ordering::Release);
    }

    pub fn update_target(&self, frame: f64) {
        self.target_frame.store(frame.to_bits(), Ordering::Relaxed);
    }

    /// Release the drag into a momentum glide easing toward `rate_target`.
    pub fn release(&self, rate_target: f64) {
        self.settle_rate_target
            .store(rate_target.to_bits(), Ordering::Relaxed);
        self.phase
            .store(ScrubPhase::Settling as u8, Ordering::Release);
    }

    /// Abort the gesture without a glide (no audio stream to render it).
    pub fn cancel(&self) {
        self.phase.store(ScrubPhase::Idle as u8, Ordering::Release);
    }

    /// Callback-side: the glide reached its landing; hand back to the
    /// engine. CAS so a simultaneous re-grab (`begin` on the UI thread)
    /// wins over the completion.
    pub fn finish_settle(&self) {
        let _ = self.phase.compare_exchange(
            ScrubPhase::Settling as u8,
            ScrubPhase::Idle as u8,
            Ordering::AcqRel,
            Ordering::Relaxed,
        );
    }

    pub fn target(&self) -> f64 {
        f64::from_bits(self.target_frame.load(Ordering::Relaxed))
    }

    pub fn settle_rate_target(&self) -> f64 {
        f64::from_bits(self.settle_rate_target.load(Ordering::Relaxed))
    }

    pub fn publish_voice_frame(&self, frame: f64) {
        self.voice_frame.store(frame.to_bits(), Ordering::Relaxed);
    }

    pub fn voice_frame(&self) -> f64 {
        f64::from_bits(self.voice_frame.load(Ordering::Relaxed))
    }

    pub fn publish_voice_rate(&self, rate: f64) {
        self.voice_rate.store(rate.to_bits(), Ordering::Relaxed);
    }

    /// Voice rate in source frames per output frame (sign = direction).
    pub fn voice_rate(&self) -> f64 {
        f64::from_bits(self.voice_rate.load(Ordering::Relaxed))
    }

    /// Callback-side: publish the predicted glide landing. The frame is
    /// stored before the sequence bump so a consumer that sees the new
    /// sequence reads the matching landing.
    pub fn publish_landing(&self, frame: f64) {
        self.landing.store(frame.to_bits(), Ordering::Relaxed);
        self.landing_seq.fetch_add(1, Ordering::Release);
    }

    /// `(sequence, landing frame)` of the most recent glide, for the UI to
    /// consume once per sequence.
    pub fn landing(&self) -> (u64, f64) {
        let seq = self.landing_seq.load(Ordering::Acquire);
        (seq, f64::from_bits(self.landing.load(Ordering::Relaxed)))
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

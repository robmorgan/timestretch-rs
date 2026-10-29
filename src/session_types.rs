//! Validated configuration and allocation-free session protocol.
use timestretch_core::Numerics;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionError {
    InvalidFormat,
    InvalidParameter,
    UnsupportedBackend,
    UnsupportedOperation,
    Capacity,
    NonFiniteInput,
    StaleSource,
    InvalidReceipt,
    InvalidState,
    Kernel(timestretch_core::StretchError),
}
impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SessionError {}
impl From<timestretch_core::StretchError> for SessionError {
    fn from(e: timestretch_core::StretchError) -> Self {
        Self::Kernel(e)
    }
}
pub type SessionResult<T> = Result<T, SessionError>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    #[default]
    Auto,
    Portable,
    Accelerate,
}
impl Backend {
    pub fn resolve(self) -> SessionResult<Self> {
        let resolved = match self {
            Self::Auto if Numerics::Accelerate.available() => Self::Accelerate,
            Self::Auto => Self::Portable,
            explicit => explicit,
        };
        if resolved == Self::Accelerate && !Numerics::Accelerate.available() {
            Err(SessionError::UnsupportedBackend)
        } else {
            Ok(resolved)
        }
    }
    pub(crate) fn numerics(self) -> SessionResult<Numerics> {
        Ok(match self.resolve()? {
            Self::Accelerate => Numerics::Accelerate,
            _ => Numerics::Portable,
        })
    }
}

macro_rules! ratio {
    ($name:ident, $min:expr, $max:expr, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name(f64);
        impl $name {
            pub const MIN: Self = Self($min);
            pub const MAX: Self = Self($max);
            pub const UNITY: Self = Self(1.0);
            pub fn new(value: f64) -> SessionResult<Self> {
                if value.is_finite() && (Self::MIN.0..=Self::MAX.0).contains(&value) {
                    Ok(Self(value))
                } else {
                    Err(SessionError::InvalidParameter)
                }
            }
            pub fn get(self) -> f64 {
                self.0
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::UNITY
            }
        }
    };
}
ratio!(
    TempoRate,
    1.0 / 32.0,
    4.0,
    "Source time / output time, from 1/32 to 4. Zero speed is a transport action."
);
ratio!(
    PitchRatio,
    0.25,
    4.0,
    "Frequency multiplier, from 0.25 to 4 (−24 to +24 semitones)."
);
ratio!(
    DurationRatio,
    0.25,
    32.0,
    "Output time / source time, from 0.25 to 32. Inverse of TempoRate."
);
impl TempoRate {
    pub fn from_bpm(source: f64, target: f64) -> SessionResult<Self> {
        if !source.is_finite() || source <= 0.0 || !target.is_finite() || target <= 0.0 {
            return Err(SessionError::InvalidParameter);
        }
        Self::new(target / source)
    }
    pub fn duration(self) -> DurationRatio {
        DurationRatio(1.0 / self.0)
    }
}
impl DurationRatio {
    pub fn tempo(self) -> TempoRate {
        TempoRate(1.0 / self.0)
    }
}
impl PitchRatio {
    pub fn from_semitones(semitones: f64) -> SessionResult<Self> {
        Self::new(2.0f64.powf(semitones / 12.0))
    }

    /// Inclusive tempo bounds at this pitch, including the combined-factor limit.
    pub fn tempo_bounds(self) -> (TempoRate, TempoRate) {
        (
            TempoRate(TempoRate::MIN.0.max(self.0 / 32.0)),
            TempoRate(TempoRate::MAX.0.min(self.0 / 0.25)),
        )
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RatioConstraint {
    #[default]
    PreservePitch,
    PreserveDuration,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Parameters {
    pub tempo: TempoRate,
    pub pitch: PitchRatio,
    pub constraint: RatioConstraint,
}
impl Parameters {
    /// The supported combined factor is `pitch / tempo` in 0.25..=32.
    /// Individual ratio bounds do not imply every combination is supported.
    pub fn validate(self) -> SessionResult<()> {
        if !(0.25..=32.0).contains(&(self.pitch.get() / self.tempo.get())) {
            Err(SessionError::InvalidParameter)
        } else {
            Ok(())
        }
    }

    pub fn keylock(tempo: TempoRate) -> Self {
        Self {
            tempo,
            ..Self::default()
        }
    }
    /// Link pitch to tempo. Rates outside the pitch domain are rejected.
    pub fn linked(tempo: TempoRate) -> SessionResult<Self> {
        Ok(Self {
            tempo,
            pitch: PitchRatio::new(tempo.get())?,
            ..Self::default()
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AppliedParameters {
    pub requested: Parameters,
    pub achieved_stretch: f64,
    pub achieved_pitch: f64,
    pub command_id: u64,
    pub requested_at: u64,
    pub applied_at: u64,
}
impl AppliedParameters {
    pub fn tempo(self) -> f64 {
        1.0 / self.achieved_stretch
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RatePolicy {
    NativeOnly,
    #[default]
    AllowConversion,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SourceAccess {
    #[default]
    Sequential,
    Seekable,
}

#[derive(Clone, Debug)]
pub struct SessionConfig {
    pub source_sample_rate: u32,
    pub output_sample_rate: u32,
    pub channels: usize,
    pub backend: Backend,
    pub rate_policy: RatePolicy,
    pub source_access: SourceAccess,
    pub source_frames: Option<u64>,
    pub parameters: Parameters,
    /// Maximum collected but unacknowledged output plus pending output.
    pub max_inflight_frames: usize,
    pub max_commands: usize,
}
impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            source_sample_rate: 44100,
            output_sample_rate: 44100,
            channels: 2,
            backend: Backend::Auto,
            rate_policy: RatePolicy::AllowConversion,
            source_access: SourceAccess::Sequential,
            source_frames: None,
            parameters: Parameters::default(),
            max_inflight_frames: 131072,
            max_commands: 64,
        }
    }
}
impl SessionConfig {
    pub(crate) fn validate(&self) -> SessionResult<u32> {
        if !(1..=2).contains(&self.channels)
            || !(32000..=192000).contains(&self.source_sample_rate)
            || !(32000..=192000).contains(&self.output_sample_rate)
            || !(65536..=1048576).contains(&self.max_inflight_frames)
            || !(1..=4096).contains(&self.max_commands)
            || self.source_frames.is_some_and(|n| n > (1u64 << 48))
        {
            return Err(SessionError::InvalidFormat);
        }
        let native = match self.source_sample_rate {
            44100 | 48000 | 96000 => self.source_sample_rate,
            _ => 48000,
        };
        if self.rate_policy == RatePolicy::NativeOnly
            && (native != self.source_sample_rate || native != self.output_sample_rate)
        {
            return Err(SessionError::InvalidFormat);
        }
        self.parameters.validate()?;
        self.backend.resolve()?;
        Ok(native)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Epoch(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceRequest {
    pub epoch: Epoch,
    pub start_frame: u64,
    pub frames: usize,
}
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OutputStamp {
    pub epoch: Epoch,
    pub start_frame: u64,
    pub frames: usize,
}
impl OutputStamp {
    pub fn consumed(self, frames: usize) -> SessionResult<OutputConsumption> {
        if frames > self.frames {
            return Err(SessionError::InvalidReceipt);
        }
        Ok(OutputConsumption {
            epoch: self.epoch,
            end_frame: self
                .start_frame
                .checked_add(frames as u64)
                .ok_or(SessionError::InvalidReceipt)?,
        })
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OutputConsumption {
    pub epoch: Epoch,
    pub end_frame: u64,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Transport {
    Stopped,
    Paused,
    #[default]
    Playing,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessStatus {
    Priming,
    Progress,
    NeedSource,
    NeedOutput,
    NeedAcknowledgement,
    Paused,
    Rendered,
    Finished,
    Fault,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessReport {
    pub status: ProcessStatus,
    pub work_done: usize,
    pub output_ready: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimingInfo {
    pub native_sample_rate: u32,
    pub source_sample_rate: u32,
    pub output_sample_rate: u32,
    pub startup_source_frames: usize,
    pub source_converter_lookahead_frames: usize,
    pub output_converter_lookahead_native_frames: usize,
    pub source_lookahead_frames: usize,
    pub seek_history_frames: usize,
    pub output_alignment_delay_frames: usize,
    pub control_quantum_frames: usize,
    pub pending_output_frames: usize,
    pub unacknowledged_output_frames: usize,
}
/// Heap-owned buffers and boxed states. Allocator overhead, host queues and
/// opaque FFT plans are separate; `None` does not mean zero plan memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnedMemory {
    pub adapter_bytes: usize,
    pub native_buffer_bytes: usize,
    pub fft_plan_bytes: Option<usize>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SessionCounters {
    pub supplied_frames: u64,
    pub rendered_frames: u64,
    pub collected_frames: u64,
    pub consumed_frames: u64,
    pub stale_receipts: u64,
    pub late_commands: u64,
    pub rejected_commands: u64,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scheduled {
    pub id: u64,
    pub at: u64,
    pub parameters: Parameters,
}

/// Small reusable transition ramp. Hosts supply the last delivered frame;
/// generation never reads DSP buffers or allocates.
#[derive(Clone, Copy, Debug, Default)]
pub struct TransitionEnvelope {
    last: [f32; 2],
    remaining: usize,
}
impl TransitionEnvelope {
    pub fn begin(&mut self, last: [f32; 2]) {
        self.last = last;
        self.remaining = 64;
    }
    pub fn apply(&mut self, output: &mut [f32], channels: usize) {
        if !(1..=2).contains(&channels) {
            return;
        }
        for frame in output.chunks_exact_mut(channels) {
            if self.remaining == 0 {
                break;
            }
            let old = self.remaining as f32 / 64.0;
            for (ch, sample) in frame.iter_mut().enumerate() {
                *sample = self.last[ch] * old + *sample * (1.0 - old);
            }
            self.remaining -= 1;
        }
    }
}

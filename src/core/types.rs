/// A single audio sample (32-bit float, range -1.0 to 1.0).
pub type Sample = f32;

/// A stereo frame containing left and right samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub left: Sample,
    pub right: Sample,
}

impl Frame {
    /// Create a new stereo frame.
    pub fn new(left: Sample, right: Sample) -> Self {
        Self { left, right }
    }

    /// Create a mono frame (same value in both channels).
    pub fn mono(value: Sample) -> Self {
        Self {
            left: value,
            right: value,
        }
    }
}

/// Buffer holding audio samples in interleaved format.
///
/// For mono audio, samples are stored sequentially: `[s0, s1, s2, ...]`
/// For stereo audio, samples are interleaved: `[L0, R0, L1, R1, ...]`
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Raw interleaved sample data.
    pub data: Vec<Sample>,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl AudioBuffer {
    /// Create a new audio buffer.
    ///
    /// # Errors
    /// Returns `StretchError::InvalidChannels` if channels is 0 or greater than 2.
    /// Returns `StretchError::InvalidSampleRate` if sample_rate is 0.
    pub fn new(data: Vec<Sample>, channels: u16, sample_rate: u32) -> Result<Self, StretchError> {
        if channels == 0 || channels > 2 {
            return Err(StretchError::InvalidChannels(channels));
        }
        if sample_rate == 0 {
            return Err(StretchError::InvalidSampleRate(sample_rate));
        }
        Ok(Self {
            data,
            channels,
            sample_rate,
        })
    }

    /// Number of frames in the buffer (total samples / channels).
    pub fn num_frames(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.data.len() / self.channels as usize
    }

    /// Duration of the audio in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.num_frames() as f64 / self.sample_rate as f64
    }

    /// Returns true if the buffer contains no samples.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a single channel's data as a new vector.
    pub fn channel_data(&self, channel: u16) -> Vec<Sample> {
        if channel >= self.channels {
            return Vec::new();
        }
        let ch = channel as usize;
        let num_ch = self.channels as usize;
        self.data
            .iter()
            .skip(ch)
            .step_by(num_ch)
            .copied()
            .collect()
    }

    /// Create an `AudioBuffer` from separate channel vectors.
    ///
    /// # Errors
    /// Returns error if channels have different lengths or invalid parameters.
    pub fn from_channels(
        channels_data: &[Vec<Sample>],
        sample_rate: u32,
    ) -> Result<Self, StretchError> {
        if channels_data.is_empty() || channels_data.len() > 2 {
            return Err(StretchError::InvalidChannels(channels_data.len() as u16));
        }
        let num_frames = channels_data[0].len();
        for ch in channels_data {
            if ch.len() != num_frames {
                return Err(StretchError::InvalidInput(
                    "All channels must have the same number of samples".to_string(),
                ));
            }
        }
        let num_channels = channels_data.len() as u16;
        let mut data = Vec::with_capacity(num_frames * num_channels as usize);
        for i in 0..num_frames {
            for ch in channels_data {
                data.push(ch[i]);
            }
        }
        AudioBuffer::new(data, num_channels, sample_rate)
    }
}

/// EDM-optimized presets for time stretching.
///
/// Each preset configures the algorithm parameters for a specific use case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdmPreset {
    /// Small tempo adjustments for DJ mixing (±1–8%). Prioritizes transparency.
    DjBeatmatch,
    /// General purpose for house/techno loops. Balanced quality.
    HouseLoop,
    /// Halftime effect — stretch to 2x. Preserves kick punch.
    Halftime,
    /// Extreme stretch (2x–4x) for ambient transitions and build-ups.
    Ambient,
    /// Optimized for vocal chops and one-shots.
    VocalChop,
}

/// Parameters controlling the time stretch operation.
#[derive(Debug, Clone)]
pub struct StretchParams {
    /// Stretch ratio: >1.0 = slower (longer), <1.0 = faster (shorter).
    pub stretch_ratio: f64,
    /// Sample rate in Hz (default: 44100).
    pub sample_rate: u32,
    /// Number of audio channels (default: 1).
    pub channels: u16,
    /// Optional EDM preset to use.
    pub preset: Option<EdmPreset>,
    /// FFT size for phase vocoder (default: 4096).
    pub fft_size: usize,
    /// Hop size for phase vocoder (default: fft_size / 4).
    pub hop_size: Option<usize>,
    /// Transient detection sensitivity (0.0 to 1.0, default: 0.5).
    pub transient_sensitivity: f32,
    /// Whether to use the hybrid algorithm (default: true).
    pub use_hybrid: bool,
}

impl StretchParams {
    /// Create new stretch parameters with the given ratio.
    ///
    /// # Errors
    /// Returns `StretchError::InvalidStretchRatio` if ratio is not positive and finite.
    pub fn new(stretch_ratio: f64) -> Result<Self, StretchError> {
        if !stretch_ratio.is_finite() || stretch_ratio <= 0.0 {
            return Err(StretchError::InvalidStretchRatio(stretch_ratio));
        }
        Ok(Self {
            stretch_ratio,
            sample_rate: 44100,
            channels: 1,
            preset: None,
            fft_size: 4096,
            hop_size: None,
            transient_sensitivity: 0.5,
            use_hybrid: true,
        })
    }

    /// Set the EDM preset.
    pub fn with_preset(mut self, preset: EdmPreset) -> Self {
        self.preset = Some(preset);
        self.apply_preset();
        self
    }

    /// Set the sample rate.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the number of channels.
    pub fn with_channels(mut self, channels: u16) -> Self {
        self.channels = channels;
        self
    }

    /// Set the FFT size.
    pub fn with_fft_size(mut self, fft_size: usize) -> Self {
        self.fft_size = fft_size;
        self
    }

    /// Set transient detection sensitivity.
    pub fn with_transient_sensitivity(mut self, sensitivity: f32) -> Self {
        self.transient_sensitivity = sensitivity.clamp(0.0, 1.0);
        self
    }

    /// Get the effective hop size.
    pub fn effective_hop_size(&self) -> usize {
        self.hop_size.unwrap_or(self.fft_size / 4)
    }

    /// Apply preset-specific parameter tuning.
    fn apply_preset(&mut self) {
        match self.preset {
            Some(EdmPreset::DjBeatmatch) => {
                self.fft_size = 4096;
                self.transient_sensitivity = 0.6;
                self.use_hybrid = true;
            }
            Some(EdmPreset::HouseLoop) => {
                self.fft_size = 4096;
                self.transient_sensitivity = 0.5;
                self.use_hybrid = true;
            }
            Some(EdmPreset::Halftime) => {
                self.fft_size = 4096;
                self.transient_sensitivity = 0.7;
                self.use_hybrid = true;
            }
            Some(EdmPreset::Ambient) => {
                self.fft_size = 8192;
                self.transient_sensitivity = 0.3;
                self.use_hybrid = false;
            }
            Some(EdmPreset::VocalChop) => {
                self.fft_size = 2048;
                self.transient_sensitivity = 0.8;
                self.use_hybrid = true;
            }
            None => {}
        }
    }

    /// Validate all parameters.
    pub fn validate(&self) -> Result<(), StretchError> {
        if !self.stretch_ratio.is_finite() || self.stretch_ratio <= 0.0 {
            return Err(StretchError::InvalidStretchRatio(self.stretch_ratio));
        }
        if self.channels == 0 || self.channels > 2 {
            return Err(StretchError::InvalidChannels(self.channels));
        }
        if self.sample_rate == 0 {
            return Err(StretchError::InvalidSampleRate(self.sample_rate));
        }
        if self.fft_size == 0 || !self.fft_size.is_power_of_two() {
            return Err(StretchError::InvalidFftSize(self.fft_size));
        }
        Ok(())
    }
}

/// Errors that can occur during time stretching.
#[derive(Debug, Clone)]
pub enum StretchError {
    /// Stretch ratio must be positive and finite.
    InvalidStretchRatio(f64),
    /// Channel count must be 1 or 2.
    InvalidChannels(u16),
    /// Sample rate must be positive.
    InvalidSampleRate(u32),
    /// FFT size must be a power of two and non-zero.
    InvalidFftSize(usize),
    /// Invalid input data.
    InvalidInput(String),
    /// I/O error.
    IoError(String),
}

impl std::fmt::Display for StretchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StretchError::InvalidStretchRatio(r) => {
                write!(f, "Invalid stretch ratio: {r}. Must be positive and finite.")
            }
            StretchError::InvalidChannels(c) => {
                write!(f, "Invalid channel count: {c}. Must be 1 or 2.")
            }
            StretchError::InvalidSampleRate(sr) => {
                write!(f, "Invalid sample rate: {sr}. Must be greater than 0.")
            }
            StretchError::InvalidFftSize(s) => {
                write!(f, "Invalid FFT size: {s}. Must be a non-zero power of two.")
            }
            StretchError::InvalidInput(msg) => {
                write!(f, "Invalid input: {msg}")
            }
            StretchError::IoError(msg) => {
                write!(f, "I/O error: {msg}")
            }
        }
    }
}

impl std::error::Error for StretchError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_new() {
        let frame = Frame::new(0.5, -0.5);
        assert_eq!(frame.left, 0.5);
        assert_eq!(frame.right, -0.5);
    }

    #[test]
    fn test_frame_mono() {
        let frame = Frame::mono(0.3);
        assert_eq!(frame.left, 0.3);
        assert_eq!(frame.right, 0.3);
    }

    #[test]
    fn test_audio_buffer_mono() {
        let buf = AudioBuffer::new(vec![0.1, 0.2, 0.3], 1, 44100).unwrap();
        assert_eq!(buf.num_frames(), 3);
        assert!((buf.duration_secs() - 3.0 / 44100.0).abs() < 1e-10);
    }

    #[test]
    fn test_audio_buffer_stereo() {
        let buf = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 2, 44100).unwrap();
        assert_eq!(buf.num_frames(), 2);
    }

    #[test]
    fn test_audio_buffer_invalid_channels() {
        assert!(AudioBuffer::new(vec![0.1], 0, 44100).is_err());
        assert!(AudioBuffer::new(vec![0.1], 3, 44100).is_err());
    }

    #[test]
    fn test_audio_buffer_invalid_sample_rate() {
        assert!(AudioBuffer::new(vec![0.1], 1, 0).is_err());
    }

    #[test]
    fn test_audio_buffer_channel_data() {
        let buf = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 2, 44100).unwrap();
        let left = buf.channel_data(0);
        let right = buf.channel_data(1);
        assert_eq!(left, vec![0.1, 0.3, 0.5]);
        assert_eq!(right, vec![0.2, 0.4, 0.6]);
    }

    #[test]
    fn test_audio_buffer_channel_data_out_of_range() {
        let buf = AudioBuffer::new(vec![0.1, 0.2], 1, 44100).unwrap();
        assert!(buf.channel_data(1).is_empty());
    }

    #[test]
    fn test_audio_buffer_from_channels() {
        let left = vec![0.1, 0.3, 0.5];
        let right = vec![0.2, 0.4, 0.6];
        let buf = AudioBuffer::from_channels(&[left, right], 44100).unwrap();
        assert_eq!(buf.channels, 2);
        assert_eq!(buf.data, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    }

    #[test]
    fn test_audio_buffer_from_channels_mismatched() {
        let left = vec![0.1, 0.3];
        let right = vec![0.2, 0.4, 0.6];
        assert!(AudioBuffer::from_channels(&[left, right], 44100).is_err());
    }

    #[test]
    fn test_audio_buffer_empty() {
        let buf = AudioBuffer::new(vec![], 1, 44100).unwrap();
        assert!(buf.is_empty());
        assert_eq!(buf.num_frames(), 0);
    }

    #[test]
    fn test_stretch_params_new() {
        let params = StretchParams::new(1.5).unwrap();
        assert_eq!(params.stretch_ratio, 1.5);
        assert_eq!(params.sample_rate, 44100);
        assert_eq!(params.channels, 1);
        assert_eq!(params.fft_size, 4096);
    }

    #[test]
    fn test_stretch_params_invalid_ratio() {
        assert!(StretchParams::new(0.0).is_err());
        assert!(StretchParams::new(-1.0).is_err());
        assert!(StretchParams::new(f64::NAN).is_err());
        assert!(StretchParams::new(f64::INFINITY).is_err());
    }

    #[test]
    fn test_stretch_params_builder() {
        let params = StretchParams::new(2.0)
            .unwrap()
            .with_sample_rate(48000)
            .with_channels(2)
            .with_fft_size(8192)
            .with_preset(EdmPreset::Ambient);
        assert_eq!(params.sample_rate, 48000);
        assert_eq!(params.channels, 2);
        // Preset overrides fft_size
        assert_eq!(params.fft_size, 8192);
    }

    #[test]
    fn test_stretch_params_preset_values() {
        let params = StretchParams::new(1.0).unwrap().with_preset(EdmPreset::DjBeatmatch);
        assert_eq!(params.fft_size, 4096);
        assert!(params.use_hybrid);

        let params = StretchParams::new(1.0).unwrap().with_preset(EdmPreset::Ambient);
        assert_eq!(params.fft_size, 8192);
        assert!(!params.use_hybrid);
    }

    #[test]
    fn test_stretch_params_validate() {
        let params = StretchParams::new(1.0).unwrap();
        assert!(params.validate().is_ok());

        let mut params = StretchParams::new(1.0).unwrap();
        params.fft_size = 1000; // Not a power of two
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_stretch_error_display() {
        let err = StretchError::InvalidStretchRatio(0.0);
        assert!(err.to_string().contains("0"));

        let err = StretchError::InvalidChannels(3);
        assert!(err.to_string().contains("3"));
    }

    #[test]
    fn test_edm_preset_equality() {
        assert_eq!(EdmPreset::DjBeatmatch, EdmPreset::DjBeatmatch);
        assert_ne!(EdmPreset::DjBeatmatch, EdmPreset::Ambient);
    }

    #[test]
    fn test_transient_sensitivity_clamped() {
        let params = StretchParams::new(1.0).unwrap().with_transient_sensitivity(2.0);
        assert_eq!(params.transient_sensitivity, 1.0);

        let params = StretchParams::new(1.0).unwrap().with_transient_sensitivity(-1.0);
        assert_eq!(params.transient_sensitivity, 0.0);
    }

    #[test]
    fn test_effective_hop_size() {
        let params = StretchParams::new(1.0).unwrap();
        assert_eq!(params.effective_hop_size(), 1024); // 4096 / 4

        let mut params = StretchParams::new(1.0).unwrap();
        params.hop_size = Some(512);
        assert_eq!(params.effective_hop_size(), 512);
    }
}

//! Core types shared across the crate: samples, buffers, parameters, and errors.

/// A single audio sample (32-bit float, range -1.0 to 1.0).
pub type Sample = f32;

/// Number of audio channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channels {
    Mono,
    Stereo,
}

impl Channels {
    /// Returns the number of channels as a usize.
    #[inline]
    pub fn count(self) -> usize {
        match self {
            Channels::Mono => 1,
            Channels::Stereo => 2,
        }
    }
}

/// An audio buffer holding interleaved sample data.
///
/// For stereo audio, samples are interleaved as `[L0, R0, L1, R1, ...]`.
///
/// # Example
///
/// ```
/// use timestretch::{AudioBuffer, Channels};
///
/// let buf = AudioBuffer::from_mono(vec![0.0; 44100], 44100);
/// assert_eq!(buf.num_frames(), 44100);
/// assert!((buf.duration_secs() - 1.0).abs() < 1e-10);
///
/// let stereo = AudioBuffer::from_stereo(vec![0.0; 88200], 44100);
/// assert_eq!(stereo.num_frames(), 44100);
/// assert_eq!(stereo.channels, Channels::Stereo);
/// ```
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Interleaved sample data.
    pub data: Vec<Sample>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel layout.
    pub channels: Channels,
}

impl AudioBuffer {
    /// Creates a new audio buffer.
    pub fn new(data: Vec<Sample>, sample_rate: u32, channels: Channels) -> Self {
        Self {
            data,
            sample_rate,
            channels,
        }
    }

    /// Returns the number of frames (samples per channel).
    #[inline]
    pub fn num_frames(&self) -> usize {
        if self.data.is_empty() {
            return 0;
        }
        self.data.len() / self.channels.count()
    }

    /// Returns the duration in seconds.
    #[inline]
    pub fn duration_secs(&self) -> f64 {
        self.num_frames() as f64 / self.sample_rate as f64
    }

    /// Extracts a single channel from interleaved data.
    pub fn channel(&self, ch: usize) -> Vec<Sample> {
        let nc = self.channels.count();
        assert!(ch < nc, "channel index out of range");
        self.data.iter().skip(ch).step_by(nc).copied().collect()
    }

    /// Creates a mono buffer from a single channel of data.
    pub fn from_mono(data: Vec<Sample>, sample_rate: u32) -> Self {
        Self {
            data,
            sample_rate,
            channels: Channels::Mono,
        }
    }

    /// Creates a stereo buffer from interleaved L/R data.
    pub fn from_stereo(data: Vec<Sample>, sample_rate: u32) -> Self {
        Self {
            data,
            sample_rate,
            channels: Channels::Stereo,
        }
    }

    /// Returns `true` if the buffer contains no samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns `true` if this is a mono buffer.
    #[inline]
    pub fn is_mono(&self) -> bool {
        self.channels == Channels::Mono
    }

    /// Returns `true` if this is a stereo buffer.
    #[inline]
    pub fn is_stereo(&self) -> bool {
        self.channels == Channels::Stereo
    }

    /// Returns the left channel samples. For mono, returns all samples.
    pub fn left(&self) -> Vec<Sample> {
        self.channel(0)
    }

    /// Returns the right channel samples. For mono, returns all samples.
    pub fn right(&self) -> Vec<Sample> {
        if self.channels == Channels::Mono {
            self.data.clone()
        } else {
            self.channel(1)
        }
    }

    /// Mixes all channels down to a mono buffer by averaging.
    pub fn mix_to_mono(&self) -> Self {
        if self.channels == Channels::Mono {
            return self.clone();
        }
        let nc = self.channels.count();
        let num_frames = self.num_frames();
        let mut mono = Vec::with_capacity(num_frames);
        let inv = 1.0 / nc as f32;
        for i in 0..num_frames {
            let sum: f32 = (0..nc).map(|ch| self.data[i * nc + ch]).sum();
            mono.push(sum * inv);
        }
        Self {
            data: mono,
            sample_rate: self.sample_rate,
            channels: Channels::Mono,
        }
    }

    /// Converts a mono buffer to stereo by duplicating the signal to both channels.
    ///
    /// If the buffer is already stereo, returns a clone.
    pub fn to_stereo(&self) -> Self {
        if self.channels == Channels::Stereo {
            return self.clone();
        }
        let mut stereo = Vec::with_capacity(self.data.len() * 2);
        for &s in &self.data {
            stereo.push(s);
            stereo.push(s);
        }
        Self {
            data: stereo,
            sample_rate: self.sample_rate,
            channels: Channels::Stereo,
        }
    }

    /// Returns the total number of samples (frames * channels).
    #[inline]
    pub fn total_samples(&self) -> usize {
        self.data.len()
    }

    /// Interleaves separate channel vectors into a single buffer.
    pub fn from_channels(channels_data: &[Vec<Sample>], sample_rate: u32) -> Self {
        let nc = channels_data.len();
        let channels = if nc == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        };
        let num_frames = channels_data.iter().map(|c| c.len()).min().unwrap_or(0);
        let mut data = Vec::with_capacity(num_frames * nc);
        for i in 0..num_frames {
            for ch in channels_data {
                data.push(ch[i]);
            }
        }
        Self {
            data,
            sample_rate,
            channels,
        }
    }
}

/// EDM-specific presets for time stretching.
///
/// Each preset tunes the FFT size, hop size, transient sensitivity, and
/// WSOLA search range for a particular use case. Apply via
/// [`StretchParams::with_preset`].
///
/// # Example
///
/// ```
/// use timestretch::{StretchParams, EdmPreset};
///
/// // DJ beatmatching: minimal artifacts at small ratio changes
/// let dj = StretchParams::new(128.0 / 126.0)
///     .with_preset(EdmPreset::DjBeatmatch);
///
/// // Halftime: stretch to 2x for bass music
/// let half = StretchParams::new(2.0)
///     .with_preset(EdmPreset::Halftime);
/// ```
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

/// Internal configuration values for an EDM preset.
struct PresetConfig {
    fft_size: usize,
    hop_size: usize,
    transient_sensitivity: f32,
    wsola_search_ms: f64,
}

impl EdmPreset {
    /// Returns a human-readable description of this preset's characteristics.
    pub fn description(self) -> &'static str {
        match self {
            EdmPreset::DjBeatmatch => {
                "Small tempo adjustments (±1-8%). Maximum transparency, minimal artifacts."
            }
            EdmPreset::HouseLoop => {
                "General purpose for house/techno loops. Balanced quality and performance."
            }
            EdmPreset::Halftime => {
                "Halftime effect (2x stretch). Preserves kick punch and transient clarity."
            }
            EdmPreset::Ambient => {
                "Extreme stretch (2x-4x) for ambient transitions. Smooth, artifact-free."
            }
            EdmPreset::VocalChop => {
                "Optimized for vocal chops and one-shots. Preserves formant character."
            }
        }
    }

    /// Returns the internal parameter configuration for this preset.
    fn config(self) -> PresetConfig {
        match self {
            EdmPreset::DjBeatmatch => PresetConfig {
                fft_size: 4096,
                hop_size: 1024,
                transient_sensitivity: 0.3,
                wsola_search_ms: WSOLA_SEARCH_MS_SMALL,
            },
            EdmPreset::HouseLoop => PresetConfig {
                fft_size: 4096,
                hop_size: 1024,
                transient_sensitivity: 0.5,
                wsola_search_ms: WSOLA_SEARCH_MS_MEDIUM,
            },
            EdmPreset::Halftime => PresetConfig {
                fft_size: 4096,
                hop_size: 512,
                transient_sensitivity: 0.7,
                wsola_search_ms: WSOLA_SEARCH_MS_LARGE,
            },
            EdmPreset::Ambient => PresetConfig {
                fft_size: 8192,
                hop_size: 2048,
                transient_sensitivity: 0.2,
                wsola_search_ms: WSOLA_SEARCH_MS_LARGE,
            },
            EdmPreset::VocalChop => PresetConfig {
                fft_size: 2048,
                hop_size: 512,
                transient_sensitivity: 0.6,
                wsola_search_ms: WSOLA_SEARCH_MS_MEDIUM,
            },
        }
    }
}

/// Parameters for the time stretching algorithm.
///
/// Use the builder methods to configure. A ratio >1.0 makes audio longer
/// (slower tempo), <1.0 makes it shorter (faster tempo).
///
/// # Example
///
/// ```
/// use timestretch::{StretchParams, EdmPreset};
///
/// let params = StretchParams::new(1.5)
///     .with_sample_rate(44100)
///     .with_channels(2)
///     .with_preset(EdmPreset::HouseLoop);
///
/// assert_eq!(params.stretch_ratio, 1.5);
/// assert_eq!(params.sample_rate, 44100);
/// assert_eq!(params.output_length(44100), 66150);
/// ```
#[derive(Debug, Clone)]
pub struct StretchParams {
    /// Stretch ratio (>1.0 = slower/longer, <1.0 = faster/shorter).
    pub stretch_ratio: f64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: Channels,
    /// FFT size for phase vocoder.
    pub fft_size: usize,
    /// Hop size (analysis step).
    pub hop_size: usize,
    /// Optional EDM preset.
    pub preset: Option<EdmPreset>,
    /// Transient detection sensitivity (0.0 to 1.0).
    pub transient_sensitivity: f32,
    /// Sub-bass phase lock cutoff frequency in Hz.
    pub sub_bass_cutoff: f32,
    /// WSOLA segment size in samples.
    pub wsola_segment_size: usize,
    /// WSOLA search range in samples.
    pub wsola_search_range: usize,
    /// Whether to use beat-grid-aware segmentation.
    ///
    /// When enabled, the hybrid stretcher detects the beat grid and aligns
    /// segment boundaries to beat positions. This preserves the rhythmic
    /// groove in 4/4 EDM tracks. Enabled by default for EDM presets.
    pub beat_aware: bool,
    /// Whether to split sub-bass into a separate band for independent processing.
    ///
    /// When enabled, audio below `sub_bass_cutoff` Hz is extracted and processed
    /// exclusively through the phase vocoder with rigid phase locking, even during
    /// transient segments. The remaining audio goes through the normal hybrid
    /// algorithm (WSOLA for transients, PV for tonal). The two bands are summed
    /// at the end. This prevents WSOLA from smearing sub-bass during kick drums.
    /// Enabled by default for EDM presets.
    pub band_split: bool,
}

/// Converts a duration in milliseconds to samples at the given sample rate.
#[inline]
fn ms_to_samples(ms: f64, sample_rate: u32) -> usize {
    (sample_rate as f64 * ms / 1000.0) as usize
}

/// Default sample rate (44.1 kHz CD quality).
const DEFAULT_SAMPLE_RATE: u32 = 44100;
/// Default FFT size for phase vocoder (good frequency resolution for bass).
const DEFAULT_FFT_SIZE: usize = 4096;
/// Default hop size (FFT/4 = 75% overlap).
const DEFAULT_HOP_SIZE: usize = 1024;
/// Default transient detection sensitivity (0.0–1.0).
const DEFAULT_TRANSIENT_SENSITIVITY: f32 = 0.5;
/// Default sub-bass phase lock cutoff in Hz.
const DEFAULT_SUB_BASS_CUTOFF: f32 = 120.0;

/// Default WSOLA segment duration (~20ms) for transient-friendly segmentation.
const WSOLA_SEGMENT_MS: f64 = 20.0;
/// Default WSOLA search range (~10ms) for small stretch ratios.
const WSOLA_SEARCH_MS_SMALL: f64 = 10.0;
/// Medium WSOLA search range (~15ms) for moderate stretching.
const WSOLA_SEARCH_MS_MEDIUM: f64 = 15.0;
/// Large WSOLA search range (~30ms) for extreme stretch ratios.
const WSOLA_SEARCH_MS_LARGE: f64 = 30.0;

impl StretchParams {
    /// Creates new stretch params with the given ratio.
    pub fn new(stretch_ratio: f64) -> Self {
        Self {
            stretch_ratio,
            sample_rate: DEFAULT_SAMPLE_RATE,
            channels: Channels::Stereo,
            fft_size: DEFAULT_FFT_SIZE,
            hop_size: DEFAULT_HOP_SIZE,
            preset: None,
            transient_sensitivity: DEFAULT_TRANSIENT_SENSITIVITY,
            sub_bass_cutoff: DEFAULT_SUB_BASS_CUTOFF,
            wsola_segment_size: ms_to_samples(WSOLA_SEGMENT_MS, DEFAULT_SAMPLE_RATE),
            wsola_search_range: ms_to_samples(WSOLA_SEARCH_MS_SMALL, DEFAULT_SAMPLE_RATE),
            beat_aware: false,
            band_split: false,
        }
    }

    /// Computes the expected output length for a given input length.
    #[inline]
    pub fn output_length(&self, input_len: usize) -> usize {
        (input_len as f64 * self.stretch_ratio).round() as usize
    }

    /// Sets the sample rate.
    ///
    /// Note: this also recalculates WSOLA segment size (~20ms) and search range
    /// (~10ms) for the new sample rate. Call `with_wsola_segment_size()` or
    /// `with_wsola_search_range()` after this method to override those values.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        // Adjust WSOLA params for sample rate
        self.wsola_segment_size = ms_to_samples(WSOLA_SEGMENT_MS, sample_rate);
        self.wsola_search_range = ms_to_samples(WSOLA_SEARCH_MS_SMALL, sample_rate);
        self
    }

    /// Sets the number of channels.
    pub fn with_channels(mut self, channels: u32) -> Self {
        self.channels = if channels == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        };
        self
    }

    /// Sets the EDM preset, overriding FFT size, hop size, transient sensitivity,
    /// and WSOLA search range. Call this before other builder methods if you want
    /// to customize individual parameters after applying a preset.
    pub fn with_preset(mut self, preset: EdmPreset) -> Self {
        self.preset = Some(preset);
        self.beat_aware = true;
        self.band_split = true;
        let cfg = preset.config();
        self.fft_size = cfg.fft_size;
        self.hop_size = cfg.hop_size;
        self.transient_sensitivity = cfg.transient_sensitivity;
        self.wsola_search_range = ms_to_samples(cfg.wsola_search_ms, self.sample_rate);
        self
    }

    /// Sets the FFT size.
    pub fn with_fft_size(mut self, fft_size: usize) -> Self {
        self.fft_size = fft_size;
        self.hop_size = fft_size / 4;
        self
    }

    /// Sets the hop size (analysis step) directly, overriding the default `fft_size / 4`.
    pub fn with_hop_size(mut self, hop_size: usize) -> Self {
        self.hop_size = hop_size;
        self
    }

    /// Sets transient sensitivity.
    pub fn with_transient_sensitivity(mut self, sensitivity: f32) -> Self {
        self.transient_sensitivity = sensitivity;
        self
    }

    /// Sets the sub-bass phase lock cutoff frequency in Hz.
    pub fn with_sub_bass_cutoff(mut self, cutoff_hz: f32) -> Self {
        self.sub_bass_cutoff = cutoff_hz;
        self
    }

    /// Sets the WSOLA segment size in samples.
    pub fn with_wsola_segment_size(mut self, size: usize) -> Self {
        self.wsola_segment_size = size;
        self
    }

    /// Sets the WSOLA search range in samples.
    pub fn with_wsola_search_range(mut self, range: usize) -> Self {
        self.wsola_search_range = range;
        self
    }

    /// Enables or disables beat-aware segmentation.
    ///
    /// When enabled, the hybrid stretcher detects the beat grid and aligns
    /// segment boundaries to beat positions for better rhythmic preservation.
    pub fn with_beat_aware(mut self, enabled: bool) -> Self {
        self.beat_aware = enabled;
        self
    }

    /// Enables or disables sub-bass band-split processing.
    ///
    /// When enabled, audio below `sub_bass_cutoff` Hz is separated and processed
    /// exclusively through the phase vocoder, preventing WSOLA from smearing
    /// bass during kick transients.
    pub fn with_band_split(mut self, enabled: bool) -> Self {
        self.band_split = enabled;
        self
    }

    /// Creates stretch params from source and target BPM values.
    ///
    /// This is a convenience method for DJ workflows where you know the
    /// current and desired tempo. The stretch ratio is computed as
    /// `source_bpm / target_bpm` (stretching audio from a faster track
    /// makes it longer, from a slower track makes it shorter).
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::{StretchParams, EdmPreset};
    ///
    /// // Match a 126 BPM track to 128 BPM
    /// let params = StretchParams::from_tempo(126.0, 128.0)
    ///     .with_preset(EdmPreset::DjBeatmatch);
    /// assert!((params.stretch_ratio - 126.0 / 128.0).abs() < 1e-10);
    /// ```
    pub fn from_tempo(source_bpm: f64, target_bpm: f64) -> Self {
        Self::new(source_bpm / target_bpm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channels_count() {
        assert_eq!(Channels::Mono.count(), 1);
        assert_eq!(Channels::Stereo.count(), 2);
    }

    #[test]
    fn test_audio_buffer_num_frames() {
        let buf = AudioBuffer::from_mono(vec![0.0; 100], 44100);
        assert_eq!(buf.num_frames(), 100);

        let buf = AudioBuffer::from_stereo(vec![0.0; 200], 44100);
        assert_eq!(buf.num_frames(), 100);
    }

    #[test]
    fn test_audio_buffer_duration() {
        let buf = AudioBuffer::from_mono(vec![0.0; 44100], 44100);
        assert!((buf.duration_secs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_audio_buffer_channel_extraction() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buf = AudioBuffer::from_stereo(data, 44100);
        assert_eq!(buf.channel(0), vec![1.0, 3.0, 5.0]);
        assert_eq!(buf.channel(1), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_audio_buffer_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        assert_eq!(buf.num_frames(), 0);
        assert_eq!(buf.duration_secs(), 0.0);
    }

    #[test]
    fn test_stretch_params_builder() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(48000)
            .with_channels(2)
            .with_preset(EdmPreset::HouseLoop);
        assert_eq!(params.sample_rate, 48000);
        assert_eq!(params.channels, Channels::Stereo);
        assert_eq!(params.preset, Some(EdmPreset::HouseLoop));
        assert_eq!(params.fft_size, 4096);
    }

    #[test]
    fn test_stretch_params_builder_advanced() {
        let params = StretchParams::new(1.5)
            .with_sub_bass_cutoff(100.0)
            .with_wsola_segment_size(512)
            .with_wsola_search_range(256);
        assert!((params.sub_bass_cutoff - 100.0).abs() < f32::EPSILON);
        assert_eq!(params.wsola_segment_size, 512);
        assert_eq!(params.wsola_search_range, 256);
    }

    #[test]
    fn test_from_tempo() {
        // 126 BPM → 128 BPM: ratio should be 126/128
        let params = StretchParams::from_tempo(126.0, 128.0);
        assert!((params.stretch_ratio - 126.0 / 128.0).abs() < 1e-10);

        // Same BPM → ratio 1.0
        let params = StretchParams::from_tempo(120.0, 120.0);
        assert!((params.stretch_ratio - 1.0).abs() < 1e-10);

        // 120 BPM → 90 BPM: ratio should be 120/90 ≈ 1.333
        let params = StretchParams::from_tempo(120.0, 90.0);
        assert!((params.stretch_ratio - 120.0 / 90.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_channels() {
        let left = vec![1.0, 2.0, 3.0];
        let right = vec![4.0, 5.0, 6.0];
        let buf = AudioBuffer::from_channels(&[left, right], 44100);
        assert_eq!(buf.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(buf.channels, Channels::Stereo);
    }

    #[test]
    fn test_audio_buffer_is_empty() {
        let empty = AudioBuffer::from_mono(vec![], 44100);
        assert!(empty.is_empty());

        let non_empty = AudioBuffer::from_mono(vec![0.5], 44100);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_audio_buffer_is_mono_stereo() {
        let mono = AudioBuffer::from_mono(vec![0.0; 100], 44100);
        assert!(mono.is_mono());
        assert!(!mono.is_stereo());

        let stereo = AudioBuffer::from_stereo(vec![0.0; 200], 44100);
        assert!(stereo.is_stereo());
        assert!(!stereo.is_mono());
    }

    #[test]
    fn test_audio_buffer_left_right_stereo() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buf = AudioBuffer::from_stereo(data, 44100);
        assert_eq!(buf.left(), vec![1.0, 3.0, 5.0]);
        assert_eq!(buf.right(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_audio_buffer_left_right_mono() {
        let data = vec![1.0, 2.0, 3.0];
        let buf = AudioBuffer::from_mono(data, 44100);
        assert_eq!(buf.left(), vec![1.0, 2.0, 3.0]);
        assert_eq!(buf.right(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_audio_buffer_mix_to_mono() {
        let data = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0];
        let buf = AudioBuffer::from_stereo(data, 44100);
        let mono = buf.mix_to_mono();
        assert!(mono.is_mono());
        assert_eq!(mono.num_frames(), 3);
        assert!((mono.data[0] - 0.5).abs() < 1e-6);
        assert!((mono.data[1] - 0.5).abs() < 1e-6);
        assert!((mono.data[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_mix_to_mono_identity() {
        let data = vec![1.0, 2.0, 3.0];
        let buf = AudioBuffer::from_mono(data.clone(), 44100);
        let mono = buf.mix_to_mono();
        assert_eq!(mono.data, data);
    }

    #[test]
    fn test_audio_buffer_to_stereo() {
        let data = vec![1.0, 2.0, 3.0];
        let mono = AudioBuffer::from_mono(data, 44100);
        let stereo = mono.to_stereo();
        assert!(stereo.is_stereo());
        assert_eq!(stereo.num_frames(), 3);
        assert_eq!(stereo.data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_audio_buffer_to_stereo_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let stereo = AudioBuffer::from_stereo(data.clone(), 44100);
        let result = stereo.to_stereo();
        assert_eq!(result.data, data);
    }

    #[test]
    fn test_audio_buffer_mono_stereo_roundtrip() {
        let data = vec![1.0, 2.0, 3.0];
        let mono = AudioBuffer::from_mono(data.clone(), 44100);
        let stereo = mono.to_stereo();
        let back = stereo.mix_to_mono();
        assert!(back.is_mono());
        assert_eq!(back.data, data);
    }

    #[test]
    fn test_audio_buffer_total_samples() {
        let mono = AudioBuffer::from_mono(vec![0.0; 100], 44100);
        assert_eq!(mono.total_samples(), 100);

        let stereo = AudioBuffer::from_stereo(vec![0.0; 200], 44100);
        assert_eq!(stereo.total_samples(), 200);
        assert_eq!(stereo.num_frames(), 100);
    }
}

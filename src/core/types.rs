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

/// Parameters for the time stretching algorithm.
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
}

impl StretchParams {
    /// Creates new stretch params with the given ratio.
    pub fn new(stretch_ratio: f64) -> Self {
        Self {
            stretch_ratio,
            sample_rate: 44100,
            channels: Channels::Stereo,
            fft_size: 4096,
            hop_size: 1024,
            preset: None,
            transient_sensitivity: 0.5,
            sub_bass_cutoff: 120.0,
            wsola_segment_size: 882, // ~20ms at 44100
            wsola_search_range: 441, // ~10ms at 44100
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
        self.wsola_segment_size = (sample_rate as f64 * 0.02) as usize;
        self.wsola_search_range = (sample_rate as f64 * 0.01) as usize;
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
        match preset {
            EdmPreset::DjBeatmatch => {
                self.fft_size = 4096;
                self.hop_size = 1024;
                self.transient_sensitivity = 0.3;
                self.wsola_search_range = (self.sample_rate as f64 * 0.01) as usize;
            }
            EdmPreset::HouseLoop => {
                self.fft_size = 4096;
                self.hop_size = 1024;
                self.transient_sensitivity = 0.5;
                self.wsola_search_range = (self.sample_rate as f64 * 0.015) as usize;
            }
            EdmPreset::Halftime => {
                self.fft_size = 4096;
                self.hop_size = 512;
                self.transient_sensitivity = 0.7;
                self.wsola_search_range = (self.sample_rate as f64 * 0.03) as usize;
            }
            EdmPreset::Ambient => {
                self.fft_size = 8192;
                self.hop_size = 2048;
                self.transient_sensitivity = 0.2;
                self.wsola_search_range = (self.sample_rate as f64 * 0.03) as usize;
            }
            EdmPreset::VocalChop => {
                self.fft_size = 2048;
                self.hop_size = 512;
                self.transient_sensitivity = 0.6;
                self.wsola_search_range = (self.sample_rate as f64 * 0.015) as usize;
            }
        }
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
    fn test_from_channels() {
        let left = vec![1.0, 2.0, 3.0];
        let right = vec![4.0, 5.0, 6.0];
        let buf = AudioBuffer::from_channels(&[left, right], 44100);
        assert_eq!(buf.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(buf.channels, Channels::Stereo);
    }
}

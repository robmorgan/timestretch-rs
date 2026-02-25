//! Core types shared across the crate: samples, buffers, parameters, and errors.

use crate::core::window::WindowType;
use crate::stretch::phase_locking::PhaseLockingMode;
use crate::stretch::stereo::StereoMode;

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

    /// Creates a `Channels` from a numeric count.
    ///
    /// Returns `Mono` for 1, `Stereo` for 2, and `None` for other values.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::Channels;
    ///
    /// assert_eq!(Channels::from_count(1), Some(Channels::Mono));
    /// assert_eq!(Channels::from_count(2), Some(Channels::Stereo));
    /// assert_eq!(Channels::from_count(5), None);
    /// ```
    pub fn from_count(count: usize) -> Option<Self> {
        match count {
            1 => Some(Channels::Mono),
            2 => Some(Channels::Stereo),
            _ => None,
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

impl std::fmt::Display for AudioBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AudioBuffer({} frames, {}Hz, {:?}, {:.3}s)",
            self.num_frames(),
            self.sample_rate,
            self.channels,
            self.duration_secs()
        )
    }
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

    /// Consumes the buffer and returns the raw sample data.
    ///
    /// This is the named equivalent of `Vec::<f32>::from(buffer)`.
    #[inline]
    pub fn into_data(self) -> Vec<Sample> {
        self.data
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
        let inv = 1.0 / nc as f32;
        let mono = self
            .data
            .chunks_exact(nc)
            .map(|frame| frame.iter().sum::<f32>() * inv)
            .collect();
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
        let stereo = self.data.iter().flat_map(|&s| [s, s]).collect();
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

    /// Extracts a sub-region of the buffer by frame range.
    ///
    /// Returns a new buffer containing frames from `start_frame` to
    /// `start_frame + num_frames` (or the end if fewer frames remain).
    ///
    /// # Panics
    ///
    /// Panics if `start_frame` is beyond the buffer length.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::from_mono(vec![0.0, 0.1, 0.2, 0.3, 0.4], 44100);
    /// let sub = buf.slice(1, 3);
    /// assert_eq!(sub.data, vec![0.1, 0.2, 0.3]);
    /// ```
    pub fn slice(&self, start_frame: usize, num_frames: usize) -> Self {
        let total_frames = self.num_frames();
        assert!(
            start_frame <= total_frames,
            "start_frame {} exceeds buffer length {}",
            start_frame,
            total_frames
        );
        let end_frame = (start_frame + num_frames).min(total_frames);
        let nc = self.channels.count();
        let start_idx = start_frame * nc;
        let end_idx = end_frame * nc;
        Self {
            data: self.data[start_idx..end_idx].to_vec(),
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Concatenates multiple buffers into a single buffer.
    ///
    /// All buffers must have the same sample rate and channel layout.
    ///
    /// # Panics
    ///
    /// Panics if the buffers have mismatched sample rates or channel layouts.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let a = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
    /// let b = AudioBuffer::from_mono(vec![3.0, 4.0], 44100);
    /// let combined = AudioBuffer::concatenate(&[&a, &b]);
    /// assert_eq!(combined.data, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn concatenate(buffers: &[&AudioBuffer]) -> Self {
        if buffers.is_empty() {
            return Self {
                data: vec![],
                sample_rate: DEFAULT_SAMPLE_RATE,
                channels: Channels::Mono,
            };
        }
        let sample_rate = buffers[0].sample_rate;
        let channels = buffers[0].channels;
        let total_len: usize = buffers.iter().map(|b| b.data.len()).sum();
        let mut data = Vec::with_capacity(total_len);
        for buf in buffers {
            assert_eq!(
                buf.sample_rate, sample_rate,
                "sample rate mismatch: {} vs {}",
                buf.sample_rate, sample_rate
            );
            assert_eq!(
                buf.channels, channels,
                "channel layout mismatch: {:?} vs {:?}",
                buf.channels, channels
            );
            data.extend_from_slice(&buf.data);
        }
        Self {
            data,
            sample_rate,
            channels,
        }
    }

    /// Normalizes the buffer so the peak amplitude equals `target_peak`.
    ///
    /// If the buffer is silent (all zeros), returns a clone unchanged.
    /// `target_peak` is typically 1.0 for full-scale normalization.
    pub fn normalize(&self, target_peak: f32) -> Self {
        let current_peak = self.data.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if current_peak == 0.0 {
            return self.clone();
        }
        let gain = target_peak / current_peak;
        Self {
            data: self.data.iter().map(|s| s * gain).collect(),
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Applies a gain in decibels to the entire buffer.
    ///
    /// Positive values amplify, negative values attenuate.
    /// For example, `gain_db = -6.0` halves the amplitude.
    pub fn apply_gain(&self, gain_db: f32) -> Self {
        let linear = 10.0f32.powf(gain_db / 20.0);
        Self {
            data: self.data.iter().map(|s| s * linear).collect(),
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Removes leading and trailing silence from the buffer.
    ///
    /// Samples with absolute value below `threshold` are considered silence.
    /// Returns a new buffer with silence trimmed from both ends.
    pub fn trim_silence(&self, threshold: f32) -> Self {
        let nc = self.channels.count();
        let total_frames = self.num_frames();
        if total_frames == 0 {
            return self.clone();
        }

        // Find first non-silent frame
        let first = (0..total_frames)
            .find(|&frame| (0..nc).any(|ch| self.data[frame * nc + ch].abs() >= threshold));
        let first = match first {
            Some(f) => f,
            None => return Self::new(vec![], self.sample_rate, self.channels),
        };

        // Find last non-silent frame
        let last = (0..total_frames)
            .rev()
            .find(|&frame| (0..nc).any(|ch| self.data[frame * nc + ch].abs() >= threshold))
            .unwrap(); // safe: first exists so last must too

        self.slice(first, last - first + 1)
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

impl PartialEq for AudioBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.sample_rate == other.sample_rate
            && self.channels == other.channels
            && self.data == other.data
    }
}

/// Provides direct access to the underlying sample slice.
impl AsRef<[Sample]> for AudioBuffer {
    #[inline]
    fn as_ref(&self) -> &[Sample] {
        &self.data
    }
}

/// Provides mutable access to the underlying sample slice for in-place processing.
impl AsMut<[Sample]> for AudioBuffer {
    #[inline]
    fn as_mut(&mut self) -> &mut [Sample] {
        &mut self.data
    }
}

/// Extracts the underlying sample data, consuming the buffer.
///
/// This is useful for passing audio data to APIs that expect raw `Vec<f32>`.
impl From<AudioBuffer> for Vec<Sample> {
    #[inline]
    fn from(buffer: AudioBuffer) -> Self {
        buffer.data
    }
}

/// An iterator over frames of an [`AudioBuffer`].
///
/// Each item is a slice of samples for one frame: a single `&[f32]` for mono,
/// or `&[f32; 2]` (as a slice) for stereo.
#[derive(Debug)]
pub struct FrameIter<'a> {
    data: &'a [Sample],
    channels: usize,
    pos: usize,
}

impl<'a> Iterator for FrameIter<'a> {
    type Item = &'a [Sample];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + self.channels > self.data.len() {
            return None;
        }
        let frame = &self.data[self.pos..self.pos + self.channels];
        self.pos += self.channels;
        Some(frame)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.data.len() - self.pos) / self.channels;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for FrameIter<'a> {}

impl<'a> IntoIterator for &'a AudioBuffer {
    type Item = &'a [Sample];
    type IntoIter = FrameIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.frames()
    }
}

impl AudioBuffer {
    /// Returns an iterator over the frames of this buffer.
    ///
    /// Each frame is a slice of samples: 1 sample for mono, 2 for stereo.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::from_stereo(vec![1.0, 2.0, 3.0, 4.0], 44100);
    /// let frames: Vec<&[f32]> = buf.frames().collect();
    /// assert_eq!(frames.len(), 2);
    /// assert_eq!(frames[0], &[1.0, 2.0]);
    /// assert_eq!(frames[1], &[3.0, 4.0]);
    /// ```
    pub fn frames(&self) -> FrameIter<'_> {
        FrameIter {
            data: &self.data,
            channels: self.channels.count(),
            pos: 0,
        }
    }

    /// Returns the peak absolute amplitude in the buffer.
    ///
    /// Returns 0.0 for an empty buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::from_mono(vec![0.25, -0.8, 0.5], 44100);
    /// assert!((buf.peak() - 0.8).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn peak(&self) -> f32 {
        self.data.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
    }

    /// Returns the root mean square (RMS) amplitude of the buffer.
    ///
    /// Returns 0.0 for an empty buffer. Computed in `f64` for precision.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::from_mono(vec![1.0, -1.0], 44100);
    /// assert!((buf.rms() - 1.0).abs() < 1e-6);
    /// ```
    pub fn rms(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = self.data.iter().map(|&s| (s as f64) * (s as f64)).sum();
        (sum_sq / self.data.len() as f64).sqrt() as f32
    }

    /// Applies a linear fade-in over the given number of frames.
    ///
    /// Gain ramps from 0.0 at frame 0 to 1.0 at `duration_frames`.
    /// Frames beyond `duration_frames` are unmodified.
    /// If `duration_frames` exceeds the buffer length, the entire buffer
    /// is faded.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::from_mono(vec![1.0; 100], 44100);
    /// let faded = buf.fade_in(50);
    /// assert!((faded.data[0] - 0.0).abs() < 1e-6);     // start: silence
    /// assert!((faded.data[99] - 1.0).abs() < 1e-6);     // end: full volume
    /// ```
    pub fn fade_in(&self, duration_frames: usize) -> Self {
        let nc = self.channels.count();
        let total_frames = self.num_frames();
        let fade_frames = duration_frames.min(total_frames);
        let mut data = self.data.clone();
        for frame in 0..fade_frames {
            let gain = frame as f32 / fade_frames as f32;
            for ch in 0..nc {
                data[frame * nc + ch] *= gain;
            }
        }
        Self {
            data,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Applies a linear fade-out over the given number of frames.
    ///
    /// Gain ramps from 1.0 to 0.0 over the last `duration_frames` frames.
    /// Frames before the fade region are unmodified.
    /// If `duration_frames` exceeds the buffer length, the entire buffer
    /// is faded.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::from_mono(vec![1.0; 100], 44100);
    /// let faded = buf.fade_out(50);
    /// assert!((faded.data[0] - 1.0).abs() < 1e-6);      // start: full volume
    /// assert!(faded.data[99].abs() < 0.05);               // end: near silence
    /// ```
    pub fn fade_out(&self, duration_frames: usize) -> Self {
        let nc = self.channels.count();
        let total_frames = self.num_frames();
        let fade_frames = duration_frames.min(total_frames);
        let fade_start = total_frames - fade_frames;
        let mut data = self.data.clone();
        for frame in fade_start..total_frames {
            let pos_in_fade = frame - fade_start;
            let gain = 1.0 - pos_in_fade as f32 / fade_frames as f32;
            for ch in 0..nc {
                data[frame * nc + ch] *= gain;
            }
        }
        Self {
            data,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Resamples the buffer to a different sample rate using cubic interpolation.
    ///
    /// Each channel is resampled independently. The output buffer has the same
    /// duration but a different number of frames matching the new sample rate.
    /// Returns a clone if the target rate equals the current rate.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::from_mono(vec![0.0; 44100], 44100); // 1 second
    /// let resampled = buf.resample(48000);
    /// assert_eq!(resampled.sample_rate, 48000);
    /// assert_eq!(resampled.num_frames(), 48000); // still 1 second
    /// ```
    pub fn resample(&self, target_sample_rate: u32) -> Self {
        if target_sample_rate == self.sample_rate || self.data.is_empty() {
            return Self {
                data: self.data.clone(),
                sample_rate: target_sample_rate,
                channels: self.channels,
            };
        }

        let nc = self.channels.count();
        let src_frames = self.num_frames();
        let target_frames = (src_frames as f64 * target_sample_rate as f64
            / self.sample_rate as f64)
            .round() as usize;

        if target_frames == 0 {
            return Self {
                data: vec![],
                sample_rate: target_sample_rate,
                channels: self.channels,
            };
        }

        // Resample each channel independently using cubic interpolation
        let mut output = Vec::with_capacity(target_frames * nc);
        for ch in 0..nc {
            let channel_data: Vec<f32> = self.data.iter().skip(ch).step_by(nc).copied().collect();
            let resampled = crate::core::resample::resample_cubic(&channel_data, target_frames);
            // Interleave: store in scratch, will interleave below
            if ch == 0 {
                output.resize(target_frames * nc, 0.0);
            }
            for (i, &s) in resampled.iter().enumerate() {
                output[i * nc + ch] = s;
            }
        }

        Self {
            data: output,
            sample_rate: target_sample_rate,
            channels: self.channels,
        }
    }

    /// Creates a new buffer by crossfading the end of this buffer into the
    /// start of another buffer.
    ///
    /// The crossfade uses a raised-cosine curve for smooth transitions,
    /// which is the standard technique for gapless DJ transitions. Both
    /// buffers must have the same sample rate and channel layout.
    ///
    /// The resulting buffer has length `self.num_frames() + other.num_frames() - crossfade_frames`.
    /// If `crossfade_frames` exceeds either buffer's length, it is clamped.
    ///
    /// # Panics
    ///
    /// Panics if the buffers have different sample rates or channel layouts.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let a = AudioBuffer::from_mono(vec![1.0; 1000], 44100);
    /// let b = AudioBuffer::from_mono(vec![0.5; 1000], 44100);
    /// let mixed = a.crossfade_into(&b, 100);
    /// assert_eq!(mixed.num_frames(), 1900); // 1000 + 1000 - 100
    /// ```
    pub fn crossfade_into(&self, other: &AudioBuffer, crossfade_frames: usize) -> Self {
        assert_eq!(
            self.sample_rate, other.sample_rate,
            "sample rate mismatch: {} vs {}",
            self.sample_rate, other.sample_rate
        );
        assert_eq!(
            self.channels, other.channels,
            "channel layout mismatch: {:?} vs {:?}",
            self.channels, other.channels
        );

        let nc = self.channels.count();
        let self_frames = self.num_frames();
        let other_frames = other.num_frames();
        let fade_frames = crossfade_frames.min(self_frames).min(other_frames);

        if fade_frames == 0 {
            // No overlap — just concatenate
            let mut data = self.data.clone();
            data.extend_from_slice(&other.data);
            return Self {
                data,
                sample_rate: self.sample_rate,
                channels: self.channels,
            };
        }

        let non_fade_self = self_frames - fade_frames;
        let total_frames = self_frames + other_frames - fade_frames;
        let mut data = Vec::with_capacity(total_frames * nc);

        // Copy non-overlapping part of self
        data.extend_from_slice(&self.data[..non_fade_self * nc]);

        // Crossfade region: raised-cosine blend
        for i in 0..fade_frames {
            let t = i as f32 / fade_frames as f32;
            let fade_out = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
            let fade_in = 1.0 - fade_out;

            let self_frame = non_fade_self + i;
            let other_frame = i;

            for ch in 0..nc {
                let s = self.data[self_frame * nc + ch] * fade_out
                    + other.data[other_frame * nc + ch] * fade_in;
                data.push(s);
            }
        }

        // Copy remaining part of other
        if fade_frames < other_frames {
            data.extend_from_slice(&other.data[fade_frames * nc..]);
        }

        Self {
            data,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Reverses the audio in this buffer, frame by frame.
    ///
    /// For stereo audio, each frame's channels stay in order (L, R) — only
    /// the frame ordering is reversed. This is useful for creative DJ effects
    /// like reverse cymbal builds and tape-stop simulations.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
    /// let rev = buf.reverse();
    /// assert_eq!(rev.data, vec![3.0, 2.0, 1.0]);
    /// ```
    pub fn reverse(&self) -> Self {
        let nc = self.channels.count();
        let num_frames = self.num_frames();
        let mut data = Vec::with_capacity(self.data.len());
        for frame in (0..num_frames).rev() {
            let start = frame * nc;
            data.extend_from_slice(&self.data[start..start + nc]);
        }
        Self {
            data,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Returns the number of audio channels (1 for mono, 2 for stereo).
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let mono = AudioBuffer::from_mono(vec![0.0; 100], 44100);
    /// assert_eq!(mono.channel_count(), 1);
    ///
    /// let stereo = AudioBuffer::from_stereo(vec![0.0; 200], 44100);
    /// assert_eq!(stereo.channel_count(), 2);
    /// ```
    #[inline]
    pub fn channel_count(&self) -> usize {
        self.channels.count()
    }

    /// Splits the buffer into two at the given frame position.
    ///
    /// Returns `(left, right)` where `left` contains frames `[0, frame)` and
    /// `right` contains frames `[frame, end)`. If `frame` exceeds the number
    /// of frames, `right` will be empty.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0, 4.0], 44100);
    /// let (left, right) = buf.split_at(2);
    /// assert_eq!(left.data, vec![1.0, 2.0]);
    /// assert_eq!(right.data, vec![3.0, 4.0]);
    /// ```
    pub fn split_at(&self, frame: usize) -> (Self, Self) {
        let nc = self.channels.count();
        let total_frames = self.num_frames();
        let split = frame.min(total_frames);
        let left = Self {
            data: self.data[..split * nc].to_vec(),
            sample_rate: self.sample_rate,
            channels: self.channels,
        };
        let right = Self {
            data: self.data[split * nc..].to_vec(),
            sample_rate: self.sample_rate,
            channels: self.channels,
        };
        (left, right)
    }

    /// Repeats (loops) the buffer `count` times.
    ///
    /// Returns a new buffer containing the audio data repeated `count` times.
    /// Returns an empty buffer if `count` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
    /// let looped = buf.repeat(3);
    /// assert_eq!(looped.data, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    /// ```
    pub fn repeat(&self, count: usize) -> Self {
        if count == 0 || self.data.is_empty() {
            return Self {
                data: vec![],
                sample_rate: self.sample_rate,
                channels: self.channels,
            };
        }
        let mut data = Vec::with_capacity(self.data.len() * count);
        for _ in 0..count {
            data.extend_from_slice(&self.data);
        }
        Self {
            data,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Mixes (sums) this buffer with another, sample by sample.
    ///
    /// Both buffers must have the same sample rate and channel layout.
    /// The output length equals the longer of the two inputs; the shorter
    /// input is zero-padded. No clipping is applied — use
    /// [`normalize`](Self::normalize) or [`apply_gain`](Self::apply_gain)
    /// if the result exceeds ±1.0.
    ///
    /// # Panics
    ///
    /// Panics if the buffers have different sample rates or channel layouts.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let a = AudioBuffer::from_mono(vec![0.5, 0.5], 44100);
    /// let b = AudioBuffer::from_mono(vec![0.3, 0.3], 44100);
    /// let mixed = a.mix(&b);
    /// assert!((mixed.data[0] - 0.8).abs() < 1e-6);
    /// ```
    pub fn mix(&self, other: &AudioBuffer) -> Self {
        assert_eq!(
            self.sample_rate, other.sample_rate,
            "sample rate mismatch: {} vs {}",
            self.sample_rate, other.sample_rate
        );
        assert_eq!(
            self.channels, other.channels,
            "channel layout mismatch: {:?} vs {:?}",
            self.channels, other.channels
        );

        let len = self.data.len().max(other.data.len());
        let mut data = vec![0.0f32; len];
        for (i, d) in data.iter_mut().enumerate() {
            let a = if i < self.data.len() {
                self.data[i]
            } else {
                0.0
            };
            let b = if i < other.data.len() {
                other.data[i]
            } else {
                0.0
            };
            *d = a + b;
        }
        Self {
            data,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Creates a silent (all-zeros) mono buffer of the given duration.
    ///
    /// Useful for creating gaps, padding, or test signals.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::silence(44100, 1.0); // 1 second of silence
    /// assert_eq!(buf.num_frames(), 44100);
    /// assert!(buf.peak() < 1e-10);
    /// ```
    pub fn silence(sample_rate: u32, duration_secs: f64) -> Self {
        let num_samples = (sample_rate as f64 * duration_secs).round() as usize;
        Self {
            data: vec![0.0; num_samples],
            sample_rate,
            channels: Channels::Mono,
        }
    }

    /// Creates a mono buffer containing a sine tone at the given frequency.
    ///
    /// Generates `duration_secs` of a sine wave at `freq_hz` with amplitude
    /// `amplitude` (should be in the range 0.0–1.0). Useful for generating
    /// test signals.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::tone(440.0, 44100, 1.0, 0.8);
    /// assert_eq!(buf.num_frames(), 44100);
    /// assert!(buf.peak() <= 0.8 + 1e-6);
    /// ```
    pub fn tone(freq_hz: f64, sample_rate: u32, duration_secs: f64, amplitude: f32) -> Self {
        let num_samples = (sample_rate as f64 * duration_secs).round() as usize;
        let data: Vec<f32> = (0..num_samples)
            .map(|i| {
                amplitude
                    * (2.0 * std::f64::consts::PI * freq_hz * i as f64 / sample_rate as f64).sin()
                        as f32
            })
            .collect();
        Self {
            data,
            sample_rate,
            channels: Channels::Mono,
        }
    }

    /// Converts a mono buffer to stereo with a panning position.
    ///
    /// `pan` ranges from -1.0 (hard left) through 0.0 (center) to 1.0 (hard
    /// right). Uses constant-power panning (sine/cosine law) to maintain
    /// perceived loudness across the stereo field.
    ///
    /// For stereo input, this is a no-op and returns a clone.
    ///
    /// # Panics
    ///
    /// Panics if `pan` is outside the range `[-1.0, 1.0]`.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let mono = AudioBuffer::from_mono(vec![1.0, 1.0], 44100);
    /// let stereo = mono.pan(0.0); // center
    /// assert!(stereo.is_stereo());
    /// // Both channels should have equal amplitude at center pan
    /// let (l, r) = (stereo.data[0], stereo.data[1]);
    /// assert!((l - r).abs() < 1e-6);
    /// ```
    pub fn pan(&self, pan: f32) -> Self {
        assert!(
            (-1.0..=1.0).contains(&pan),
            "pan must be in [-1.0, 1.0], got {}",
            pan
        );

        if self.channels == Channels::Stereo {
            return self.clone();
        }

        // Constant-power panning: L = cos(angle), R = sin(angle)
        // where angle goes from 0 (hard left) to PI/2 (hard right)
        let angle = (pan + 1.0) * 0.5 * std::f32::consts::FRAC_PI_2;
        let gain_l = angle.cos();
        let gain_r = angle.sin();

        let mut data = Vec::with_capacity(self.data.len() * 2);
        for &s in &self.data {
            data.push(s * gain_l);
            data.push(s * gain_r);
        }

        Self {
            data,
            sample_rate: self.sample_rate,
            channels: Channels::Stereo,
        }
    }

    /// Applies a gain envelope defined by time-value breakpoints.
    ///
    /// `breakpoints` is a slice of `(time_secs, gain_linear)` pairs, sorted
    /// by time. Gain is linearly interpolated between breakpoints. Samples
    /// before the first breakpoint use the first gain value; samples after
    /// the last use the last gain value.
    ///
    /// This is useful for volume automation, ducking, and creative effects.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::from_mono(vec![1.0; 44100], 44100);
    /// // Fade from 1.0 to 0.0 over 1 second
    /// let faded = buf.with_gain_envelope(&[(0.0, 1.0), (1.0, 0.0)]);
    /// assert!(faded.data[0] > 0.99);
    /// assert!(faded.data[44099].abs() < 0.01);
    /// ```
    pub fn with_gain_envelope(&self, breakpoints: &[(f64, f32)]) -> Self {
        if breakpoints.is_empty() || self.data.is_empty() {
            return self.clone();
        }

        let nc = self.channels.count();
        let num_frames = self.num_frames();
        let mut data = self.data.clone();

        for frame in 0..num_frames {
            let time = frame as f64 / self.sample_rate as f64;
            let gain = interpolate_breakpoints(breakpoints, time);
            for ch in 0..nc {
                data[frame * nc + ch] *= gain;
            }
        }

        Self {
            data,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Removes DC offset from the audio by subtracting the mean of each channel.
    ///
    /// DC offset is a common problem in recorded audio where the waveform is
    /// shifted away from zero. This method centers each channel independently.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::from_mono(vec![1.5, 1.6, 1.4, 1.5], 44100);
    /// let centered = buf.remove_dc();
    /// let mean: f32 = centered.data.iter().sum::<f32>() / centered.data.len() as f32;
    /// assert!(mean.abs() < 1e-6);
    /// ```
    pub fn remove_dc(&self) -> Self {
        if self.data.is_empty() {
            return self.clone();
        }

        let nc = self.channels.count();
        let num_frames = self.num_frames();
        let mut data = self.data.clone();

        for ch in 0..nc {
            // Compute mean for this channel using f64 for precision
            let sum: f64 = self
                .data
                .iter()
                .skip(ch)
                .step_by(nc)
                .map(|&s| s as f64)
                .sum();
            let mean = (sum / num_frames as f64) as f32;

            // Subtract mean from this channel
            for frame in 0..num_frames {
                data[frame * nc + ch] -= mean;
            }
        }

        Self {
            data,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Applies a window function to the buffer, sample by sample.
    ///
    /// The window is scaled to the buffer length (each channel is windowed
    /// identically). This is useful for spectral analysis, granular synthesis,
    /// and creating smooth fade shapes.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::{AudioBuffer, WindowType};
    ///
    /// let buf = AudioBuffer::from_mono(vec![1.0; 100], 44100);
    /// let windowed = buf.apply_window(WindowType::Hann);
    /// // First and last samples should be near zero (Hann window)
    /// assert!(windowed.data[0].abs() < 0.01);
    /// assert!(windowed.data[99].abs() < 0.01);
    /// // Middle should be close to 1.0
    /// assert!((windowed.data[50] - 1.0).abs() < 0.1);
    /// ```
    pub fn apply_window(&self, window_type: crate::core::window::WindowType) -> Self {
        if self.data.is_empty() {
            return self.clone();
        }

        let num_frames = self.num_frames();
        let window = crate::core::window::generate_window(window_type, num_frames);
        let nc = self.channels.count();
        let mut data = self.data.clone();

        for frame in 0..num_frames {
            for ch in 0..nc {
                data[frame * nc + ch] *= window[frame];
            }
        }

        Self {
            data,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }
}

/// Linearly interpolates a value from sorted `(time, value)` breakpoints.
fn interpolate_breakpoints(breakpoints: &[(f64, f32)], time: f64) -> f32 {
    if breakpoints.len() == 1 || time <= breakpoints[0].0 {
        return breakpoints[0].1;
    }
    if time >= breakpoints[breakpoints.len() - 1].0 {
        return breakpoints[breakpoints.len() - 1].1;
    }

    // Find the segment containing `time`
    for i in 1..breakpoints.len() {
        if time <= breakpoints[i].0 {
            let (t0, v0) = breakpoints[i - 1];
            let (t1, v1) = breakpoints[i];
            let dt = t1 - t0;
            if dt <= 0.0 {
                return v1;
            }
            let frac = ((time - t0) / dt) as f32;
            return v0 + frac * (v1 - v0);
        }
    }

    breakpoints[breakpoints.len() - 1].1
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
    wsola_segment_ms: f64,
    transient_region_ms: f64,
    window_type: WindowType,
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
                hop_size: 4096 / 4, // 1024: 75% overlap, good COLA with Hann windows
                transient_sensitivity: 0.3, // low sensitivity for clean ±5% DJ use
                wsola_search_ms: WSOLA_SEARCH_MS_SMALL,
                wsola_segment_ms: 50.0, // more context for cross-correlation at small ratios
                transient_region_ms: 30.0, // kick attack+early decay is 30-50ms
                window_type: WindowType::Hann, // matched analysis/synthesis windows
            },
            EdmPreset::HouseLoop => PresetConfig {
                fft_size: 4096,
                hop_size: 4096 / 4, // 1024: 75% overlap, good COLA with Hann windows
                transient_sensitivity: 0.5,
                wsola_search_ms: WSOLA_SEARCH_MS_MEDIUM,
                wsola_segment_ms: 40.0,
                transient_region_ms: 30.0, // wider region for better kick handling
                window_type: WindowType::Hann, // matched analysis/synthesis windows
            },
            EdmPreset::Halftime => PresetConfig {
                fft_size: 4096,
                hop_size: 4096 / 4, // 1024: 75% overlap, good COLA with Hann windows
                transient_sensitivity: 0.7,
                wsola_search_ms: WSOLA_SEARCH_MS_LARGE,
                wsola_segment_ms: 30.0,
                transient_region_ms: 20.0, // 20ms input = 40ms at 2x in output
                window_type: WindowType::Hann, // matched analysis/synthesis windows
            },
            EdmPreset::Ambient => PresetConfig {
                fft_size: 8192,
                hop_size: 8192 / 2, // 4096: 50% overlap eliminates aliasing artifacts
                transient_sensitivity: 0.2,
                wsola_search_ms: WSOLA_SEARCH_MS_LARGE,
                wsola_segment_ms: 20.0,
                transient_region_ms: 10.0, // ambient has few transients
                window_type: WindowType::BlackmanHarris, // better sidelobe suppression for long FFT
            },
            EdmPreset::VocalChop => PresetConfig {
                fft_size: 4096,
                hop_size: 4096 / 4, // 1024: 75% overlap with Hann for better formant resolution
                transient_sensitivity: 0.6, // phase-based detection reduces false positives
                wsola_search_ms: WSOLA_SEARCH_MS_MEDIUM,
                wsola_segment_ms: 25.0,
                transient_region_ms: 20.0,
                window_type: WindowType::Hann, // matched analysis/synthesis windows
            },
        }
    }
}

impl std::fmt::Display for EdmPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdmPreset::DjBeatmatch => write!(f, "DjBeatmatch"),
            EdmPreset::HouseLoop => write!(f, "HouseLoop"),
            EdmPreset::Halftime => write!(f, "Halftime"),
            EdmPreset::Ambient => write!(f, "Ambient"),
            EdmPreset::VocalChop => write!(f, "VocalChop"),
        }
    }
}

/// Crossfade mode for segment transitions in the hybrid stretcher.
///
/// Controls how the crossfade duration between algorithm segments is determined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossfadeMode {
    /// Fixed crossfade duration in seconds (e.g., 0.005 for 5ms).
    Fixed(f64),
    /// Adaptive crossfade that varies by segment type transition:
    /// - Tonal→Transient: 3ms (fast transition to preserve onset)
    /// - Transient→Tonal: 8ms (smooth recovery)
    /// - Tonal→Tonal: 5ms (default)
    /// - Transient→Transient: 2ms (minimal blending)
    Adaptive,
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
    /// Window function for the phase vocoder.
    ///
    /// Different windows trade off frequency resolution, sidelobe suppression,
    /// and transient smearing. Defaults to Hann, which is a good general choice.
    /// Blackman-Harris offers better sidelobe suppression for tonal content;
    /// Kaiser provides a tunable trade-off via its beta parameter.
    pub window_type: WindowType,
    /// Whether to normalize output RMS to match input RMS.
    ///
    /// When enabled, the output amplitude is scaled so that its RMS energy
    /// matches the input. This prevents level changes during time stretching,
    /// which is important for DJ workflows and consistent loudness.
    pub normalize: bool,
    /// Phase locking algorithm for the phase vocoder.
    ///
    /// - [`PhaseLockingMode::Identity`] — simple nearest-peak locking (fast, may ring)
    /// - [`PhaseLockingMode::RegionOfInfluence`] — influence zones with deviation clamping (better quality)
    pub phase_locking_mode: PhaseLockingMode,
    /// Stereo processing mode.
    ///
    /// - [`StereoMode::MidSide`] (default) — preserves stereo image via mid/side encoding
    /// - [`StereoMode::Independent`] — processes L/R independently (legacy behavior)
    pub stereo_mode: StereoMode,
    /// Whether to apply spectral envelope preservation.
    ///
    /// When enabled, the spectral envelope (formant structure) of the input
    /// is preserved in the output, preventing unnatural timbre shifts.
    /// Most audible on vocals and synth pads.
    pub envelope_preservation: bool,
    /// Cepstral order for spectral envelope extraction.
    ///
    /// Controls the smoothness of the envelope: lower = smoother, higher = more detail.
    /// Typical values: 30-50 for vocals, 20-30 for general music.
    pub envelope_order: usize,
    /// Whether to use multi-resolution FFT processing.
    ///
    /// When enabled, tonal segments are split into three frequency bands using
    /// Linkwitz-Riley crossover filters at 200 Hz and 4000 Hz. Each band is
    /// processed with an FFT size optimized for its frequency range:
    /// - Sub-bass (0-200 Hz): `fft_size * 4` for precise frequency resolution
    /// - Mid (200-4000 Hz): `fft_size` (the configured value)
    /// - High (4000+ Hz): `fft_size / 4` for sharp temporal resolution
    ///
    /// This mirrors the approach used by professional time-stretching algorithms
    /// like Ableton's Complex Pro, and is particularly beneficial for material
    /// with both sustained bass and transient-rich high frequencies.
    /// Enabled by default for [`EdmPreset::DjBeatmatch`] and [`EdmPreset::Ambient`].
    pub multi_resolution: bool,
    /// Duration in seconds of the transient region around each detected onset.
    ///
    /// Controls how much audio around a transient is processed with WSOLA
    /// instead of the phase vocoder. Larger values protect more of the
    /// transient's attack and early decay from PV smearing.
    /// Default: 0.010 (10ms). DjBeatmatch uses 0.030 (30ms).
    pub transient_region_secs: f64,
    /// Whether to use elastic beat distribution for non-uniform stretching.
    ///
    /// When enabled, beats are anchored to a target beat grid while sustain
    /// regions absorb the excess time. This preserves rhythmic feel by keeping
    /// beats at their musically correct positions. The [`elastic_anchor`](Self::elastic_anchor)
    /// field controls how strongly transients are anchored to ratio 1.0 vs
    /// the global ratio. Automatically disabled when `stretch_ratio == 1.0`.
    /// Enabled by default for DjBeatmatch and HouseLoop presets.
    pub elastic_timing: bool,
    /// How strongly elastic timing anchors transient segments to ratio 1.0.
    ///
    /// Range: 0.0 to 1.0.
    /// - 0.0: transients get the global stretch ratio (no anchoring, beats move
    ///   to the target tempo). Best for DJ beatmatching.
    /// - 1.0: transients stay close to ratio 1.0 (maximum anchoring, beats stay
    ///   near the original tempo). Best for creative effects like halftime.
    ///
    /// Only takes effect when `elastic_timing` is enabled. Default: 0.0.
    pub elastic_anchor: f64,
    /// Whether to use HPSS (Harmonic-Percussive Source Separation) pre-processing.
    ///
    /// When enabled, tonal segments are separated into harmonic and percussive
    /// components. The harmonic part is PV-stretched (excellent for sustained tones)
    /// while the percussive part is WSOLA-stretched (preserves transient detail).
    /// The two stretched components are summed. This improves quality when harmonic
    /// and percussive content overlap (e.g., kick drum over a sustained chord).
    pub hpss_enabled: bool,
    /// Crossfade mode for segment transitions.
    ///
    /// Controls how crossfade durations are computed at segment boundaries.
    /// `Fixed` uses a constant duration; `Adaptive` scales based on the
    /// type of segments being joined (transient→tonal, tonal→transient, etc.).
    pub crossfade_mode: CrossfadeMode,
    /// Optional BPM for beat-grid-aware stretching.
    ///
    /// When set, transient positions are snapped to the nearest beat subdivision,
    /// improving rhythmic accuracy for tempo-matched content.
    pub bpm: Option<f64>,
    /// Whether to use dynamic WSOLA search range based on stretch ratio.
    ///
    /// When enabled, the effective search range scales with the stretch ratio,
    /// giving the WSOLA algorithm more room to find good matches at extreme
    /// ratios. The range is capped at 40ms to prevent excessive latency.
    pub dynamic_wsola_search: bool,
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
/// Default hop size (FFT/4 = 75% overlap for Hann window COLA compliance).
///
/// 75% overlap (hop = FFT/4) provides good constant-overlap-add (COLA) behavior
/// with Hann windows while halving the CPU cost compared to 87.5% overlap.
const DEFAULT_HOP_SIZE: usize = DEFAULT_FFT_SIZE / 4;
/// Default transient detection sensitivity (0.0–1.0).
const DEFAULT_TRANSIENT_SENSITIVITY: f32 = 0.5;
/// Default sub-bass phase lock cutoff in Hz.
const DEFAULT_SUB_BASS_CUTOFF: f32 = 120.0;

/// Default transient region duration in seconds (~10ms around each onset).
const DEFAULT_TRANSIENT_REGION_SECS: f64 = 0.010;
/// Default WSOLA segment duration (~20ms) for transient-friendly segmentation.
const WSOLA_SEGMENT_MS: f64 = 20.0;
/// Default WSOLA search range (~10ms) for small stretch ratios.
const WSOLA_SEARCH_MS_SMALL: f64 = 10.0;
/// Medium WSOLA search range (~15ms) for moderate stretching.
const WSOLA_SEARCH_MS_MEDIUM: f64 = 15.0;
/// Large WSOLA search range (~30ms) for extreme stretch ratios.
const WSOLA_SEARCH_MS_LARGE: f64 = 30.0;

impl Default for StretchParams {
    /// Returns default parameters with ratio 1.0 (identity), stereo, 44100 Hz.
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl std::fmt::Display for StretchParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StretchParams(ratio={:.4}, {}Hz, {:?}",
            self.stretch_ratio, self.sample_rate, self.channels
        )?;
        if let Some(preset) = &self.preset {
            write!(f, ", preset={}", preset)?;
        }
        write!(f, ", fft={}, hop={})", self.fft_size, self.hop_size)
    }
}

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
            window_type: WindowType::BlackmanHarris,
            normalize: false,
            phase_locking_mode: PhaseLockingMode::RegionOfInfluence,
            stereo_mode: StereoMode::MidSide,
            envelope_preservation: false,
            envelope_order: 40,
            multi_resolution: false,
            transient_region_secs: DEFAULT_TRANSIENT_REGION_SECS,
            elastic_timing: false,
            elastic_anchor: 0.0,
            hpss_enabled: false,
            crossfade_mode: CrossfadeMode::Fixed(0.012),
            bpm: None,
            dynamic_wsola_search: false,
        }
    }

    /// Computes the expected output length for a given input length.
    #[inline]
    pub fn output_length(&self, input_len: usize) -> usize {
        (input_len as f64 * self.stretch_ratio).round() as usize
    }

    /// Sets the sample rate.
    ///
    /// Recalculates WSOLA segment size and search range for the new sample rate,
    /// using the preset's values if a preset was set, or defaults otherwise.
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        // Recalculate WSOLA params from preset if one was set, otherwise use defaults
        if let Some(preset) = self.preset {
            let cfg = preset.config();
            self.wsola_segment_size = ms_to_samples(cfg.wsola_segment_ms, sample_rate);
            self.wsola_search_range = ms_to_samples(cfg.wsola_search_ms, sample_rate);
        } else {
            self.wsola_segment_size = ms_to_samples(WSOLA_SEGMENT_MS, sample_rate);
            self.wsola_search_range = ms_to_samples(WSOLA_SEARCH_MS_SMALL, sample_rate);
        }
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
    /// WSOLA params, and transient region size. Call this before other builder
    /// methods if you want to customize individual parameters after applying a preset.
    pub fn with_preset(mut self, preset: EdmPreset) -> Self {
        self.preset = Some(preset);
        self.beat_aware = true;
        self.band_split = true;
        let cfg = preset.config();
        self.fft_size = cfg.fft_size;
        self.hop_size = cfg.hop_size;
        self.transient_sensitivity = cfg.transient_sensitivity;
        self.wsola_search_range = ms_to_samples(cfg.wsola_search_ms, self.sample_rate);
        self.wsola_segment_size = ms_to_samples(cfg.wsola_segment_ms, self.sample_rate);
        self.transient_region_secs = cfg.transient_region_ms / 1000.0;
        self.window_type = cfg.window_type;
        // Enable envelope preservation for presets where timbre matters
        self.envelope_preservation = matches!(
            preset,
            EdmPreset::DjBeatmatch
                | EdmPreset::HouseLoop
                | EdmPreset::Halftime
                | EdmPreset::Ambient
                | EdmPreset::VocalChop
        );
        // Enable multi-resolution FFT for transparency-critical and extreme-stretch presets.
        // DjBeatmatch needs it for transparent tempo changes; Ambient benefits from the
        // large sub-bass FFT at extreme stretch ratios (2x-4x).
        self.multi_resolution = matches!(preset, EdmPreset::DjBeatmatch | EdmPreset::Ambient);
        // Enable elastic beat distribution for rhythm-critical presets.
        // DjBeatmatch and HouseLoop use low anchor so beats land at the target BPM;
        // creative presets like Halftime would use a high anchor to keep beats
        // near the original tempo (but elastic_timing is off for those).
        self.elastic_timing = matches!(preset, EdmPreset::DjBeatmatch | EdmPreset::HouseLoop);
        self.elastic_anchor = match preset {
            EdmPreset::DjBeatmatch => 0.0,
            EdmPreset::HouseLoop => 0.1,
            _ => 0.0,
        };
        // Enable HPSS for presets where harmonic/percussive overlap matters
        self.hpss_enabled = matches!(preset, EdmPreset::DjBeatmatch | EdmPreset::HouseLoop);
        // Use adaptive crossfade for all presets
        self.crossfade_mode = CrossfadeMode::Adaptive;
        // Enable dynamic WSOLA search range for presets where stretch ratio varies
        self.dynamic_wsola_search = matches!(
            preset,
            EdmPreset::DjBeatmatch | EdmPreset::HouseLoop | EdmPreset::Halftime
        );
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

    /// Sets the window function for the phase vocoder.
    ///
    /// - [`WindowType::Hann`] (default) — good general-purpose choice
    /// - [`WindowType::BlackmanHarris`] — better sidelobe suppression for tonal content
    /// - `WindowType::Kaiser(beta)` — tunable: higher beta means narrower mainlobe
    pub fn with_window_type(mut self, window_type: WindowType) -> Self {
        self.window_type = window_type;
        self
    }

    /// Enables or disables output RMS normalization.
    ///
    /// When enabled, the output is scaled so its RMS matches the input RMS.
    /// This prevents level changes during time stretching. Useful for DJ
    /// workflows and consistent loudness across different stretch ratios.
    pub fn with_normalize(mut self, enabled: bool) -> Self {
        self.normalize = enabled;
        self
    }

    /// Enables or disables spectral envelope preservation.
    ///
    /// When enabled, the formant structure of the input is preserved,
    /// preventing unnatural timbre shifts on vocals and synth pads.
    pub fn with_envelope_preservation(mut self, enabled: bool) -> Self {
        self.envelope_preservation = enabled;
        self
    }

    /// Sets the cepstral order for spectral envelope extraction.
    ///
    /// Lower values produce a smoother envelope (less detail),
    /// higher values preserve more spectral detail. Default: 40.
    pub fn with_envelope_order(mut self, order: usize) -> Self {
        self.envelope_order = order;
        self
    }

    /// Enables or disables multi-resolution FFT processing.
    ///
    /// When enabled, tonal segments are split into three frequency bands
    /// (sub-bass/mid/high) using Linkwitz-Riley crossover filters. Each
    /// band uses an FFT size optimized for its frequency range, improving
    /// quality for material with both sustained bass and transient-rich
    /// high frequencies.
    pub fn with_multi_resolution(mut self, enabled: bool) -> Self {
        self.multi_resolution = enabled;
        self
    }

    /// Sets the transient region duration in seconds.
    ///
    /// Controls how much audio around each detected onset is processed with
    /// WSOLA instead of the phase vocoder. Larger values protect more of the
    /// transient's attack and early decay from PV smearing.
    pub fn with_transient_region_secs(mut self, secs: f64) -> Self {
        self.transient_region_secs = secs;
        self
    }

    /// Enables or disables elastic beat distribution.
    ///
    /// When enabled and beats are detected, stretch ratios are distributed
    /// non-uniformly: transient segments stay close to ratio 1.0 while tonal
    /// segments absorb the excess stretch. Use [`with_elastic_anchor`](Self::with_elastic_anchor)
    /// to control how strongly transients are anchored to 1.0.
    pub fn with_elastic_timing(mut self, enabled: bool) -> Self {
        self.elastic_timing = enabled;
        self
    }

    /// Sets how strongly elastic timing anchors transient segments to ratio 1.0.
    ///
    /// - 0.0: transients use the global ratio (beats at target tempo, best for DJ)
    /// - 1.0: transients stay near ratio 1.0 (beats near original tempo, creative effects)
    ///
    /// Only takes effect when elastic timing is enabled.
    pub fn with_elastic_anchor(mut self, anchor: f64) -> Self {
        self.elastic_anchor = anchor.clamp(0.0, 1.0);
        self
    }

    /// Enables or disables HPSS (Harmonic-Percussive Source Separation).
    ///
    /// When enabled, tonal segments are separated into harmonic and percussive
    /// components before stretching. Harmonics use PV, percussive uses WSOLA.
    pub fn with_hpss(mut self, enabled: bool) -> Self {
        self.hpss_enabled = enabled;
        self
    }

    /// Sets the crossfade mode for segment transitions.
    ///
    /// - `CrossfadeMode::Fixed(secs)` — constant crossfade duration
    /// - [`CrossfadeMode::Adaptive`] — varies by segment type transition
    pub fn with_crossfade_mode(mut self, mode: CrossfadeMode) -> Self {
        self.crossfade_mode = mode;
        self
    }

    /// Set the BPM for beat-grid-aware stretching.
    ///
    /// When set, transient positions are snapped to the nearest beat subdivision,
    /// improving rhythmic accuracy for tempo-matched content.
    pub fn with_bpm(mut self, bpm: f64) -> Self {
        self.bpm = Some(bpm);
        self
    }

    /// Enables or disables dynamic WSOLA search range.
    ///
    /// When enabled, the effective search range scales with the stretch ratio,
    /// giving the WSOLA algorithm more room to find good matches at extreme
    /// ratios. The range is capped at 40ms to prevent excessive latency.
    pub fn with_dynamic_wsola_search(mut self, enabled: bool) -> Self {
        self.dynamic_wsola_search = enabled;
        self
    }

    /// Compute the effective WSOLA search range in seconds.
    ///
    /// When `dynamic_wsola_search` is true, the base search range (in ms)
    /// is scaled by the stretch ratio deviation from unity, capped at 40ms.
    /// Otherwise returns the static search range converted to seconds.
    pub fn effective_wsola_search_ms(&self) -> f64 {
        let base_ms = self.wsola_search_range as f64 * 1000.0 / self.sample_rate as f64;
        if self.dynamic_wsola_search {
            let ratio_factor = self.stretch_ratio.max(1.0 / self.stretch_ratio);
            (base_ms * ratio_factor).min(40.0) // cap at 40ms
        } else {
            base_ms
        }
    }

    /// Compute the effective WSOLA search range in samples.
    ///
    /// When `dynamic_wsola_search` is true, the base search range is scaled by
    /// the stretch ratio deviation from unity, capped at 40ms worth of samples.
    /// Otherwise returns the static `wsola_search_range`.
    pub fn effective_wsola_search_range(&self) -> usize {
        if self.dynamic_wsola_search {
            let ratio_factor = self.stretch_ratio.max(1.0 / self.stretch_ratio);
            let base = self.wsola_search_range as f64;
            let max_samples = 0.040 * self.sample_rate as f64; // 40ms cap
            (base * ratio_factor).min(max_samples) as usize
        } else {
            self.wsola_search_range
        }
    }

    /// Sets the stereo processing mode.
    ///
    /// - [`StereoMode::MidSide`] (default) — better stereo coherence
    /// - [`StereoMode::Independent`] — legacy L/R independent processing
    pub fn with_stereo_mode(mut self, mode: StereoMode) -> Self {
        self.stereo_mode = mode;
        self
    }

    /// Sets the phase locking algorithm for the phase vocoder.
    ///
    /// - [`PhaseLockingMode::Identity`] — fast but may produce ringing
    /// - [`PhaseLockingMode::RegionOfInfluence`] — better quality with deviation clamping
    pub fn with_phase_locking_mode(mut self, mode: PhaseLockingMode) -> Self {
        self.phase_locking_mode = mode;
        self
    }

    /// Sets the stretch ratio, overriding the value passed to [`new()`](Self::new).
    ///
    /// Useful for adjusting the ratio after applying a preset or other
    /// builder methods.
    pub fn with_stretch_ratio(mut self, ratio: f64) -> Self {
        self.stretch_ratio = ratio;
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
    fn test_channels_from_count() {
        assert_eq!(Channels::from_count(1), Some(Channels::Mono));
        assert_eq!(Channels::from_count(2), Some(Channels::Stereo));
        assert_eq!(Channels::from_count(0), None);
        assert_eq!(Channels::from_count(3), None);
        assert_eq!(Channels::from_count(6), None);
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
    fn test_preset_then_sample_rate_preserves_wsola_params() {
        // Calling with_preset then with_sample_rate should use preset's WSOLA config
        let params = StretchParams::new(1.078)
            .with_preset(EdmPreset::DjBeatmatch)
            .with_sample_rate(44100)
            .with_channels(2);
        // DjBeatmatch: wsola_segment_ms=50, wsola_search_ms=10
        let expected_segment = ms_to_samples(50.0, 44100);
        let expected_search = ms_to_samples(10.0, 44100);
        assert_eq!(params.wsola_segment_size, expected_segment);
        assert_eq!(params.wsola_search_range, expected_search);
    }

    #[test]
    fn test_preset_transient_region() {
        let dj = StretchParams::new(1.0).with_preset(EdmPreset::DjBeatmatch);
        assert!((dj.transient_region_secs - 0.030).abs() < 1e-6);

        let ambient = StretchParams::new(1.0).with_preset(EdmPreset::Ambient);
        assert!((ambient.transient_region_secs - 0.010).abs() < 1e-6);

        // Halftime: 20ms input region (will be 40ms in 2x output)
        let halftime = StretchParams::new(1.0).with_preset(EdmPreset::Halftime);
        assert!((halftime.transient_region_secs - 0.020).abs() < 1e-6);
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

    #[test]
    fn test_audio_buffer_slice_mono() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let buf = AudioBuffer::from_mono(data, 44100);

        let sliced = buf.slice(2, 5);
        assert_eq!(sliced.num_frames(), 5);
        assert_eq!(sliced.data, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(sliced.sample_rate, 44100);
        assert!(sliced.is_mono());
    }

    #[test]
    fn test_audio_buffer_slice_stereo() {
        // Stereo: [L0,R0, L1,R1, L2,R2, L3,R3]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let buf = AudioBuffer::from_stereo(data, 44100);

        let sliced = buf.slice(1, 2);
        assert_eq!(sliced.num_frames(), 2);
        assert_eq!(sliced.data, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_audio_buffer_slice_clamp() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        // Request more frames than available — should clamp
        let sliced = buf.slice(1, 100);
        assert_eq!(sliced.num_frames(), 2);
        assert_eq!(sliced.data, vec![2.0, 3.0]);
    }

    #[test]
    fn test_audio_buffer_slice_empty() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        let sliced = buf.slice(1, 0);
        assert_eq!(sliced.num_frames(), 0);
        assert!(sliced.is_empty());
    }

    #[test]
    fn test_audio_buffer_concatenate() {
        let a = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
        let b = AudioBuffer::from_mono(vec![3.0, 4.0, 5.0], 44100);
        let c = AudioBuffer::concatenate(&[&a, &b]);
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(c.num_frames(), 5);
        assert_eq!(c.sample_rate, 44100);
    }

    #[test]
    fn test_audio_buffer_concatenate_stereo() {
        let a = AudioBuffer::from_stereo(vec![1.0, 2.0, 3.0, 4.0], 48000);
        let b = AudioBuffer::from_stereo(vec![5.0, 6.0], 48000);
        let c = AudioBuffer::concatenate(&[&a, &b]);
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(c.num_frames(), 3);
        assert_eq!(c.sample_rate, 48000);
        assert!(c.is_stereo());
    }

    #[test]
    fn test_audio_buffer_concatenate_empty() {
        let result = AudioBuffer::concatenate(&[]);
        assert!(result.is_empty());
    }

    #[test]
    #[should_panic(expected = "sample rate mismatch")]
    fn test_audio_buffer_concatenate_mismatched_rate() {
        let a = AudioBuffer::from_mono(vec![1.0], 44100);
        let b = AudioBuffer::from_mono(vec![2.0], 48000);
        AudioBuffer::concatenate(&[&a, &b]);
    }

    #[test]
    #[should_panic(expected = "channel layout mismatch")]
    fn test_audio_buffer_concatenate_mismatched_channels() {
        let a = AudioBuffer::from_mono(vec![1.0], 44100);
        let b = AudioBuffer::from_stereo(vec![2.0, 3.0], 44100);
        AudioBuffer::concatenate(&[&a, &b]);
    }

    #[test]
    fn test_audio_buffer_normalize() {
        let buf = AudioBuffer::from_mono(vec![0.25, -0.5, 0.1], 44100);
        let norm = buf.normalize(1.0);
        // Peak was 0.5, so gain = 2.0
        assert!((norm.data[0] - 0.5).abs() < 1e-6);
        assert!((norm.data[1] - (-1.0)).abs() < 1e-6);
        assert!((norm.data[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_normalize_silence() {
        let buf = AudioBuffer::from_mono(vec![0.0, 0.0, 0.0], 44100);
        let norm = buf.normalize(1.0);
        assert_eq!(norm.data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_audio_buffer_normalize_half_scale() {
        let buf = AudioBuffer::from_mono(vec![0.5, -1.0, 0.25], 44100);
        let norm = buf.normalize(0.5);
        // Peak was 1.0, gain = 0.5
        assert!((norm.data[0] - 0.25).abs() < 1e-6);
        assert!((norm.data[1] - (-0.5)).abs() < 1e-6);
        assert!((norm.data[2] - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_apply_gain() {
        let buf = AudioBuffer::from_mono(vec![0.5, -0.5], 44100);
        // -6 dB ≈ 0.5 linear
        let quieter = buf.apply_gain(-6.0206);
        assert!((quieter.data[0] - 0.25).abs() < 0.01);
        assert!((quieter.data[1] - (-0.25)).abs() < 0.01);
    }

    #[test]
    fn test_audio_buffer_apply_gain_zero() {
        let buf = AudioBuffer::from_mono(vec![0.5, -0.5], 44100);
        let same = buf.apply_gain(0.0);
        assert!((same.data[0] - 0.5).abs() < 1e-6);
        assert!((same.data[1] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_trim_silence() {
        let data = vec![0.0, 0.0, 0.5, 0.8, -0.3, 0.0, 0.0];
        let buf = AudioBuffer::from_mono(data, 44100);
        let trimmed = buf.trim_silence(0.01);
        assert_eq!(trimmed.data, vec![0.5, 0.8, -0.3]);
    }

    #[test]
    fn test_audio_buffer_trim_silence_stereo() {
        // [0,0, 0,0, L,R, L,R, 0,0]
        let data = vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.2, 0.8, 0.0, 0.0];
        let buf = AudioBuffer::from_stereo(data, 44100);
        let trimmed = buf.trim_silence(0.01);
        assert_eq!(trimmed.num_frames(), 2);
        assert_eq!(trimmed.data, vec![0.5, 0.3, 0.2, 0.8]);
    }

    #[test]
    fn test_audio_buffer_trim_silence_all_silent() {
        let buf = AudioBuffer::from_mono(vec![0.0, 0.0, 0.0], 44100);
        let trimmed = buf.trim_silence(0.01);
        assert!(trimmed.is_empty());
    }

    #[test]
    fn test_audio_buffer_trim_silence_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let trimmed = buf.trim_silence(0.01);
        assert!(trimmed.is_empty());
    }

    #[test]
    fn test_audio_buffer_trim_silence_no_trim_needed() {
        let data = vec![0.5, 0.8, -0.3];
        let buf = AudioBuffer::from_mono(data.clone(), 44100);
        let trimmed = buf.trim_silence(0.01);
        assert_eq!(trimmed.data, data);
    }

    #[test]
    fn test_stretch_params_default() {
        let params = StretchParams::default();
        assert!((params.stretch_ratio - 1.0).abs() < 1e-10);
        assert_eq!(params.sample_rate, 44100);
        assert_eq!(params.channels, Channels::Stereo);
        assert_eq!(params.fft_size, 4096);
    }

    #[test]
    fn test_stretch_params_display() {
        let params = StretchParams::new(1.5)
            .with_sample_rate(48000)
            .with_preset(EdmPreset::DjBeatmatch);
        let s = format!("{}", params);
        assert!(s.contains("1.5000"));
        assert!(s.contains("48000"));
        assert!(s.contains("DjBeatmatch"));
    }

    #[test]
    fn test_stretch_params_display_no_preset() {
        let params = StretchParams::new(1.0);
        let s = format!("{}", params);
        assert!(s.contains("1.0000"));
        assert!(!s.contains("preset="));
    }

    #[test]
    fn test_edm_preset_display() {
        assert_eq!(format!("{}", EdmPreset::DjBeatmatch), "DjBeatmatch");
        assert_eq!(format!("{}", EdmPreset::HouseLoop), "HouseLoop");
        assert_eq!(format!("{}", EdmPreset::Halftime), "Halftime");
        assert_eq!(format!("{}", EdmPreset::Ambient), "Ambient");
        assert_eq!(format!("{}", EdmPreset::VocalChop), "VocalChop");
    }

    #[test]
    fn test_audio_buffer_display() {
        let buf = AudioBuffer::from_mono(vec![0.0; 44100], 44100);
        let s = format!("{}", buf);
        assert!(s.contains("44100 frames"));
        assert!(s.contains("44100Hz"));
        assert!(s.contains("Mono"));
        assert!(s.contains("1.000s"));
    }

    #[test]
    fn test_audio_buffer_partial_eq() {
        let a = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        let b = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        assert_eq!(a, b);
    }

    #[test]
    fn test_audio_buffer_partial_eq_different_data() {
        let a = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        let b = AudioBuffer::from_mono(vec![1.0, 2.0, 4.0], 44100);
        assert_ne!(a, b);
    }

    #[test]
    fn test_audio_buffer_partial_eq_different_rate() {
        let a = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
        let b = AudioBuffer::from_mono(vec![1.0, 2.0], 48000);
        assert_ne!(a, b);
    }

    #[test]
    fn test_audio_buffer_partial_eq_different_channels() {
        let a = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
        let b = AudioBuffer::from_stereo(vec![1.0, 2.0], 44100);
        assert_ne!(a, b);
    }

    #[test]
    fn test_audio_buffer_as_ref() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        let slice: &[f32] = buf.as_ref();
        assert_eq!(slice, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_audio_buffer_frames_mono() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        let frames: Vec<&[f32]> = buf.frames().collect();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0], &[1.0]);
        assert_eq!(frames[1], &[2.0]);
        assert_eq!(frames[2], &[3.0]);
    }

    #[test]
    fn test_audio_buffer_frames_stereo() {
        let buf = AudioBuffer::from_stereo(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 44100);
        let frames: Vec<&[f32]> = buf.frames().collect();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0], &[1.0, 2.0]);
        assert_eq!(frames[1], &[3.0, 4.0]);
        assert_eq!(frames[2], &[5.0, 6.0]);
    }

    #[test]
    fn test_audio_buffer_frames_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let frames: Vec<&[f32]> = buf.frames().collect();
        assert_eq!(frames.len(), 0);
    }

    #[test]
    fn test_audio_buffer_frames_exact_size() {
        let buf = AudioBuffer::from_stereo(vec![1.0, 2.0, 3.0, 4.0], 44100);
        let iter = buf.frames();
        assert_eq!(iter.len(), 2);
    }

    #[test]
    fn test_audio_buffer_into_iterator() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        let mut count = 0;
        for frame in &buf {
            assert_eq!(frame.len(), 1);
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_audio_buffer_into_iterator_stereo() {
        let buf = AudioBuffer::from_stereo(vec![1.0, 2.0, 3.0, 4.0], 44100);
        let frames: Vec<&[f32]> = (&buf).into_iter().collect();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], &[1.0, 2.0]);
    }

    #[test]
    fn test_audio_buffer_peak() {
        let buf = AudioBuffer::from_mono(vec![0.25, -0.8, 0.5], 44100);
        assert!((buf.peak() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_peak_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        assert!((buf.peak() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_rms() {
        // RMS of [1.0, -1.0] = sqrt((1 + 1) / 2) = 1.0
        let buf = AudioBuffer::from_mono(vec![1.0, -1.0], 44100);
        assert!((buf.rms() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_rms_sine() {
        // RMS of a sine wave is 1/sqrt(2) ≈ 0.7071
        let n = 44100;
        let data: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / n as f32).sin())
            .collect();
        let buf = AudioBuffer::from_mono(data, 44100);
        let expected = 1.0 / 2.0f32.sqrt();
        assert!(
            (buf.rms() - expected).abs() < 0.01,
            "Expected RMS ~{}, got {}",
            expected,
            buf.rms()
        );
    }

    #[test]
    fn test_audio_buffer_rms_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        assert!((buf.rms() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_fade_in() {
        let buf = AudioBuffer::from_mono(vec![1.0, 1.0, 1.0, 1.0], 44100);
        let faded = buf.fade_in(4);
        // Frame 0: gain=0/4=0.0, frame 1: 1/4=0.25, frame 2: 2/4=0.5, frame 3: 3/4=0.75
        assert!((faded.data[0] - 0.0).abs() < 1e-6);
        assert!((faded.data[1] - 0.25).abs() < 1e-6);
        assert!((faded.data[2] - 0.5).abs() < 1e-6);
        assert!((faded.data[3] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_fade_in_partial() {
        let buf = AudioBuffer::from_mono(vec![1.0, 1.0, 1.0, 1.0], 44100);
        let faded = buf.fade_in(2);
        // Only first 2 frames affected: gain=0.0, 0.5. Rest unchanged.
        assert!((faded.data[0] - 0.0).abs() < 1e-6);
        assert!((faded.data[1] - 0.5).abs() < 1e-6);
        assert!((faded.data[2] - 1.0).abs() < 1e-6);
        assert!((faded.data[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_fade_out() {
        let buf = AudioBuffer::from_mono(vec![1.0, 1.0, 1.0, 1.0], 44100);
        let faded = buf.fade_out(4);
        // Frame 0: gain=1.0, frame 1: 0.75, frame 2: 0.5, frame 3: 0.25
        assert!((faded.data[0] - 1.0).abs() < 1e-6);
        assert!((faded.data[1] - 0.75).abs() < 1e-6);
        assert!((faded.data[2] - 0.5).abs() < 1e-6);
        assert!((faded.data[3] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_fade_out_partial() {
        let buf = AudioBuffer::from_mono(vec![1.0, 1.0, 1.0, 1.0], 44100);
        let faded = buf.fade_out(2);
        // First 2 frames unmodified, last 2 faded
        assert!((faded.data[0] - 1.0).abs() < 1e-6);
        assert!((faded.data[1] - 1.0).abs() < 1e-6);
        assert!((faded.data[2] - 1.0).abs() < 1e-6); // gain = 1 - 0/2 = 1.0
        assert!((faded.data[3] - 0.5).abs() < 1e-6); // gain = 1 - 1/2 = 0.5
    }

    #[test]
    fn test_audio_buffer_fade_stereo() {
        let buf = AudioBuffer::from_stereo(vec![1.0, 1.0, 1.0, 1.0], 44100);
        let faded = buf.fade_in(2);
        // Frame 0 (both channels): gain = 0.0
        assert!((faded.data[0] - 0.0).abs() < 1e-6);
        assert!((faded.data[1] - 0.0).abs() < 1e-6);
        // Frame 1 (both channels): gain = 0.5
        assert!((faded.data[2] - 0.5).abs() < 1e-6);
        assert!((faded.data[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_fade_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let faded_in = buf.fade_in(100);
        assert!(faded_in.is_empty());
        let faded_out = buf.fade_out(100);
        assert!(faded_out.is_empty());
    }

    #[test]
    fn test_audio_buffer_fade_longer_than_buffer() {
        // Fade duration exceeds buffer length — should fade the whole thing
        let buf = AudioBuffer::from_mono(vec![1.0, 1.0], 44100);
        let faded = buf.fade_in(100);
        assert!((faded.data[0] - 0.0).abs() < 1e-6);
        assert!((faded.data[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_with_window_type() {
        let params = StretchParams::new(1.5).with_window_type(WindowType::BlackmanHarris);
        assert_eq!(params.window_type, WindowType::BlackmanHarris);

        let params = StretchParams::new(1.0).with_window_type(WindowType::Kaiser(800));
        assert_eq!(params.window_type, WindowType::Kaiser(800));
    }

    #[test]
    fn test_window_type_default_is_blackman_harris() {
        let params = StretchParams::new(1.0);
        assert_eq!(params.window_type, WindowType::BlackmanHarris);
    }

    #[test]
    fn test_preset_sets_window_type() {
        // Ambient preset should use Blackman-Harris (better sidelobe suppression)
        let params = StretchParams::new(2.0).with_preset(EdmPreset::Ambient);
        assert_eq!(params.window_type, WindowType::BlackmanHarris);

        // DjBeatmatch uses Hann (matched analysis/synthesis, good COLA at 75%)
        let params = StretchParams::new(1.0).with_preset(EdmPreset::DjBeatmatch);
        assert_eq!(params.window_type, WindowType::Hann);

        // HouseLoop uses Hann
        let params = StretchParams::new(1.5).with_preset(EdmPreset::HouseLoop);
        assert_eq!(params.window_type, WindowType::Hann);

        // VocalChop uses Hann
        let params = StretchParams::new(1.0).with_preset(EdmPreset::VocalChop);
        assert_eq!(params.window_type, WindowType::Hann);

        // Halftime uses Hann
        let params = StretchParams::new(2.0).with_preset(EdmPreset::Halftime);
        assert_eq!(params.window_type, WindowType::Hann);
    }

    #[test]
    fn test_preset_window_can_be_overridden() {
        let params = StretchParams::new(2.0)
            .with_preset(EdmPreset::Ambient)
            .with_window_type(WindowType::Hann);
        assert_eq!(params.window_type, WindowType::Hann);
    }

    #[test]
    fn test_with_normalize() {
        let params = StretchParams::new(1.5).with_normalize(true);
        assert!(params.normalize);

        let params = StretchParams::new(1.5);
        assert!(!params.normalize);
    }

    #[test]
    fn test_with_stretch_ratio() {
        let params = StretchParams::new(1.0)
            .with_preset(EdmPreset::DjBeatmatch)
            .with_stretch_ratio(0.984375);
        assert!((params.stretch_ratio - 0.984375).abs() < 1e-10);
        assert_eq!(params.preset, Some(EdmPreset::DjBeatmatch));
    }

    #[test]
    fn test_from_audio_buffer_to_vec() {
        let data = vec![1.0, 2.0, 3.0];
        let buf = AudioBuffer::from_mono(data.clone(), 44100);
        let extracted: Vec<f32> = buf.into();
        assert_eq!(extracted, data);
    }

    #[test]
    fn test_from_audio_buffer_to_vec_stereo() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buf = AudioBuffer::from_stereo(data.clone(), 48000);
        let extracted: Vec<f32> = Vec::from(buf);
        assert_eq!(extracted, data);
    }

    #[test]
    fn test_audio_buffer_debug() {
        let buf = AudioBuffer::from_mono(vec![0.0; 100], 44100);
        let debug_str = format!("{:?}", buf);
        assert!(debug_str.contains("AudioBuffer"));
        assert!(debug_str.contains("sample_rate: 44100"));
    }

    #[test]
    fn test_stretch_params_debug() {
        let params = StretchParams::new(1.5);
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("StretchParams"));
        assert!(debug_str.contains("stretch_ratio"));
    }

    #[test]
    fn test_frame_iter_debug() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
        let iter = buf.frames();
        let debug_str = format!("{:?}", iter);
        assert!(debug_str.contains("FrameIter"));
    }

    // --- resample tests ---

    #[test]
    fn test_resample_same_rate() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        let resampled = buf.resample(44100);
        assert_eq!(resampled.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(resampled.sample_rate, 44100);
    }

    #[test]
    fn test_resample_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let resampled = buf.resample(48000);
        assert!(resampled.is_empty());
        assert_eq!(resampled.sample_rate, 48000);
    }

    #[test]
    fn test_resample_mono_upsample() {
        // 1 second at 44100 → 48000
        let n = 44100;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let buf = AudioBuffer::from_mono(input, 44100);
        let resampled = buf.resample(48000);

        assert_eq!(resampled.sample_rate, 48000);
        assert_eq!(resampled.num_frames(), 48000);
        assert!(resampled.is_mono());

        // Duration should be preserved (~1 second)
        assert!((resampled.duration_secs() - 1.0).abs() < 0.01);

        // Output should be bounded
        assert!(resampled.data.iter().all(|s| s.abs() <= 1.1));
    }

    #[test]
    fn test_resample_mono_downsample() {
        // 1 second at 48000 → 44100
        let n = 48000;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();
        let buf = AudioBuffer::from_mono(input, 48000);
        let resampled = buf.resample(44100);

        assert_eq!(resampled.sample_rate, 44100);
        assert_eq!(resampled.num_frames(), 44100);
        assert!((resampled.duration_secs() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_resample_stereo() {
        // Stereo: each channel is a different frequency
        let n = 44100;
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / 44100.0;
            data.push((2.0 * std::f32::consts::PI * 440.0 * t).sin()); // left: 440 Hz
            data.push((2.0 * std::f32::consts::PI * 880.0 * t).sin()); // right: 880 Hz
        }
        let buf = AudioBuffer::from_stereo(data, 44100);
        let resampled = buf.resample(48000);

        assert_eq!(resampled.sample_rate, 48000);
        assert_eq!(resampled.num_frames(), 48000);
        assert!(resampled.is_stereo());
        assert_eq!(resampled.total_samples(), 48000 * 2);
    }

    #[test]
    fn test_resample_preserves_dc() {
        // A constant signal should remain constant after resampling
        let buf = AudioBuffer::from_mono(vec![0.5; 1000], 44100);
        let resampled = buf.resample(48000);
        // Middle samples should be very close to 0.5
        let mid = resampled.num_frames() / 2;
        assert!(
            (resampled.data[mid] - 0.5).abs() < 0.01,
            "DC signal should be preserved, got {}",
            resampled.data[mid]
        );
    }

    // --- crossfade_into tests ---

    #[test]
    fn test_crossfade_into_basic() {
        let a = AudioBuffer::from_mono(vec![1.0; 1000], 44100);
        let b = AudioBuffer::from_mono(vec![0.5; 1000], 44100);
        let mixed = a.crossfade_into(&b, 100);

        assert_eq!(mixed.num_frames(), 1900); // 1000 + 1000 - 100
        assert_eq!(mixed.sample_rate, 44100);
        assert!(mixed.is_mono());
    }

    #[test]
    fn test_crossfade_into_zero_overlap() {
        let a = AudioBuffer::from_mono(vec![1.0; 10], 44100);
        let b = AudioBuffer::from_mono(vec![0.5; 10], 44100);
        let mixed = a.crossfade_into(&b, 0);

        assert_eq!(mixed.num_frames(), 20);
        // First 10 should be 1.0, last 10 should be 0.5
        assert!((mixed.data[0] - 1.0).abs() < 1e-6);
        assert!((mixed.data[10] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_into_midpoint() {
        // At the midpoint of the crossfade, values should be ~average
        let a = AudioBuffer::from_mono(vec![1.0; 100], 44100);
        let b = AudioBuffer::from_mono(vec![0.0; 100], 44100);
        let mixed = a.crossfade_into(&b, 50);

        // Midpoint of crossfade is at frame 75 (non_fade=50, fade starts there, mid at +25)
        let mid_idx = 50 + 25;
        assert!(
            (mixed.data[mid_idx] - 0.5).abs() < 0.05,
            "Crossfade midpoint should be ~0.5, got {}",
            mixed.data[mid_idx]
        );
    }

    #[test]
    fn test_crossfade_into_stereo() {
        let a = AudioBuffer::from_stereo(vec![1.0; 200], 44100);
        let b = AudioBuffer::from_stereo(vec![0.5; 200], 44100);
        let mixed = a.crossfade_into(&b, 10);

        // 100 frames each, 10 frame overlap → 190 frames
        assert_eq!(mixed.num_frames(), 190);
        assert!(mixed.is_stereo());
    }

    #[test]
    fn test_crossfade_into_clamps_to_shorter() {
        let a = AudioBuffer::from_mono(vec![1.0; 5], 44100);
        let b = AudioBuffer::from_mono(vec![0.5; 100], 44100);
        let mixed = a.crossfade_into(&b, 50);

        // Crossfade clamped to min(50, 5, 100) = 5
        assert_eq!(mixed.num_frames(), 100); // 5 + 100 - 5 = 100
    }

    #[test]
    #[should_panic(expected = "sample rate mismatch")]
    fn test_crossfade_into_mismatched_rate() {
        let a = AudioBuffer::from_mono(vec![1.0; 10], 44100);
        let b = AudioBuffer::from_mono(vec![0.5; 10], 48000);
        a.crossfade_into(&b, 5);
    }

    #[test]
    #[should_panic(expected = "channel layout mismatch")]
    fn test_crossfade_into_mismatched_channels() {
        let a = AudioBuffer::from_mono(vec![1.0; 10], 44100);
        let b = AudioBuffer::from_stereo(vec![0.5; 20], 44100);
        a.crossfade_into(&b, 5);
    }

    #[test]
    fn test_crossfade_into_energy_conservation() {
        // Crossfading equal-amplitude signals should keep amplitude in range
        let a = AudioBuffer::from_mono(vec![0.7; 1000], 44100);
        let b = AudioBuffer::from_mono(vec![0.7; 1000], 44100);
        let mixed = a.crossfade_into(&b, 200);

        // All samples should be close to 0.7 (no boosting)
        assert!(
            mixed.data.iter().all(|&s| (s - 0.7).abs() < 0.01),
            "Equal-amplitude crossfade should preserve level"
        );
    }

    // --- reverse tests ---

    #[test]
    fn test_reverse_mono() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0, 4.0], 44100);
        let rev = buf.reverse();
        assert_eq!(rev.data, vec![4.0, 3.0, 2.0, 1.0]);
        assert_eq!(rev.sample_rate, 44100);
        assert!(rev.is_mono());
    }

    #[test]
    fn test_reverse_stereo() {
        // Stereo: frames are [L1, R1, L2, R2, L3, R3]
        let buf = AudioBuffer::from_stereo(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 44100);
        let rev = buf.reverse();
        // Reversed frame order: [L3, R3, L2, R2, L1, R1]
        assert_eq!(rev.data, vec![5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
        assert!(rev.is_stereo());
    }

    #[test]
    fn test_reverse_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let rev = buf.reverse();
        assert!(rev.is_empty());
    }

    #[test]
    fn test_reverse_double_is_identity() {
        let data = vec![0.1, 0.5, -0.3, 0.8, 0.0];
        let buf = AudioBuffer::from_mono(data.clone(), 44100);
        let double_rev = buf.reverse().reverse();
        assert_eq!(double_rev.data, data);
    }

    // --- channel_count tests ---

    #[test]
    fn test_channel_count_mono() {
        let buf = AudioBuffer::from_mono(vec![0.0; 10], 44100);
        assert_eq!(buf.channel_count(), 1);
    }

    #[test]
    fn test_channel_count_stereo() {
        let buf = AudioBuffer::from_stereo(vec![0.0; 20], 44100);
        assert_eq!(buf.channel_count(), 2);
    }

    #[test]
    fn test_audio_buffer_as_mut() {
        let mut buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100);
        let slice: &mut [f32] = buf.as_mut();
        slice[0] = 10.0;
        slice[2] = 30.0;
        assert_eq!(buf.data[0], 10.0);
        assert_eq!(buf.data[1], 2.0);
        assert_eq!(buf.data[2], 30.0);
    }

    #[test]
    fn test_audio_buffer_into_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let buf = AudioBuffer::from_stereo(data.clone(), 44100);
        assert_eq!(buf.into_data(), data);
    }

    #[test]
    fn test_audio_buffer_into_data_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        assert!(buf.into_data().is_empty());
    }

    // --- split_at tests ---

    #[test]
    fn test_split_at_mono() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0, 4.0], 44100);
        let (left, right) = buf.split_at(2);
        assert_eq!(left.data, vec![1.0, 2.0]);
        assert_eq!(right.data, vec![3.0, 4.0]);
        assert_eq!(left.sample_rate, 44100);
        assert_eq!(right.sample_rate, 44100);
    }

    #[test]
    fn test_split_at_stereo() {
        let buf = AudioBuffer::from_stereo(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 44100);
        let (left, right) = buf.split_at(1);
        assert_eq!(left.data, vec![1.0, 2.0]);
        assert_eq!(right.data, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_split_at_zero() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
        let (left, right) = buf.split_at(0);
        assert!(left.is_empty());
        assert_eq!(right.data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_split_at_end() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
        let (left, right) = buf.split_at(100); // beyond end
        assert_eq!(left.data, vec![1.0, 2.0]);
        assert!(right.is_empty());
    }

    // --- repeat tests ---

    #[test]
    fn test_repeat_mono() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
        let looped = buf.repeat(3);
        assert_eq!(looped.data, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        assert_eq!(looped.num_frames(), 6);
    }

    #[test]
    fn test_repeat_stereo() {
        let buf = AudioBuffer::from_stereo(vec![1.0, 2.0, 3.0, 4.0], 44100);
        let looped = buf.repeat(2);
        assert_eq!(looped.data, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(looped.num_frames(), 4);
    }

    #[test]
    fn test_repeat_zero() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
        let looped = buf.repeat(0);
        assert!(looped.is_empty());
    }

    #[test]
    fn test_repeat_one() {
        let buf = AudioBuffer::from_mono(vec![1.0, 2.0], 44100);
        let looped = buf.repeat(1);
        assert_eq!(looped.data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_repeat_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let looped = buf.repeat(5);
        assert!(looped.is_empty());
    }

    // --- mix tests ---

    #[test]
    fn test_mix_basic() {
        let a = AudioBuffer::from_mono(vec![0.5, 0.5], 44100);
        let b = AudioBuffer::from_mono(vec![0.3, 0.3], 44100);
        let mixed = a.mix(&b);
        assert!((mixed.data[0] - 0.8).abs() < 1e-6);
        assert!((mixed.data[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_mix_different_lengths() {
        let a = AudioBuffer::from_mono(vec![1.0, 1.0, 1.0], 44100);
        let b = AudioBuffer::from_mono(vec![0.5], 44100);
        let mixed = a.mix(&b);
        assert_eq!(mixed.num_frames(), 3);
        assert!((mixed.data[0] - 1.5).abs() < 1e-6);
        assert!((mixed.data[1] - 1.0).abs() < 1e-6);
        assert!((mixed.data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mix_stereo() {
        let a = AudioBuffer::from_stereo(vec![0.5, 0.3, 0.5, 0.3], 44100);
        let b = AudioBuffer::from_stereo(vec![0.1, 0.2, 0.1, 0.2], 44100);
        let mixed = a.mix(&b);
        assert!((mixed.data[0] - 0.6).abs() < 1e-6); // L
        assert!((mixed.data[1] - 0.5).abs() < 1e-6); // R
    }

    #[test]
    #[should_panic(expected = "sample rate mismatch")]
    fn test_mix_mismatched_rate() {
        let a = AudioBuffer::from_mono(vec![0.5], 44100);
        let b = AudioBuffer::from_mono(vec![0.5], 48000);
        a.mix(&b);
    }

    #[test]
    #[should_panic(expected = "channel layout mismatch")]
    fn test_mix_mismatched_channels() {
        let a = AudioBuffer::from_mono(vec![0.5], 44100);
        let b = AudioBuffer::from_stereo(vec![0.5, 0.5], 44100);
        a.mix(&b);
    }

    #[test]
    fn test_mix_empty() {
        let a = AudioBuffer::from_mono(vec![1.0], 44100);
        let b = AudioBuffer::from_mono(vec![], 44100);
        let mixed = a.mix(&b);
        assert_eq!(mixed.data, vec![1.0]);
    }

    // --- silence() tests ---

    #[test]
    fn test_silence_basic() {
        let buf = AudioBuffer::silence(44100, 1.0);
        assert_eq!(buf.num_frames(), 44100);
        assert_eq!(buf.sample_rate, 44100);
        assert_eq!(buf.channels, Channels::Mono);
        assert!(buf.peak() < 1e-10);
    }

    #[test]
    fn test_silence_zero_duration() {
        let buf = AudioBuffer::silence(44100, 0.0);
        assert_eq!(buf.num_frames(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_silence_48khz() {
        let buf = AudioBuffer::silence(48000, 0.5);
        assert_eq!(buf.num_frames(), 24000);
        assert_eq!(buf.sample_rate, 48000);
    }

    // --- tone() tests ---

    #[test]
    fn test_tone_basic() {
        let buf = AudioBuffer::tone(440.0, 44100, 1.0, 0.8);
        assert_eq!(buf.num_frames(), 44100);
        assert_eq!(buf.sample_rate, 44100);
        assert_eq!(buf.channels, Channels::Mono);
        assert!(buf.peak() <= 0.8 + 1e-4);
        assert!(buf.peak() > 0.7); // should be close to 0.8
    }

    #[test]
    fn test_tone_zero_amplitude() {
        let buf = AudioBuffer::tone(440.0, 44100, 1.0, 0.0);
        assert!(buf.peak() < 1e-10);
    }

    #[test]
    fn test_tone_frequency() {
        // A 1 Hz tone at 100 samples/sec should complete exactly 1 cycle
        let buf = AudioBuffer::tone(1.0, 100, 1.0, 1.0);
        assert_eq!(buf.num_frames(), 100);
        // At sample 0, sin(0) = 0; at sample 25, sin(PI/2) ≈ 1
        assert!(buf.data[0].abs() < 0.01);
        assert!((buf.data[25] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_tone_zero_duration() {
        let buf = AudioBuffer::tone(440.0, 44100, 0.0, 1.0);
        assert!(buf.is_empty());
    }

    // --- pan() tests ---

    #[test]
    fn test_pan_center() {
        let mono = AudioBuffer::from_mono(vec![1.0, 1.0, 1.0], 44100);
        let stereo = mono.pan(0.0);
        assert!(stereo.is_stereo());
        assert_eq!(stereo.num_frames(), 3);
        // Center: both channels should be equal (cos(PI/4) = sin(PI/4))
        for frame in stereo.frames() {
            assert!(
                (frame[0] - frame[1]).abs() < 1e-6,
                "Center pan should give equal L/R"
            );
        }
    }

    #[test]
    fn test_pan_hard_left() {
        let mono = AudioBuffer::from_mono(vec![1.0], 44100);
        let stereo = mono.pan(-1.0);
        // Hard left: angle = 0, cos(0)=1, sin(0)=0
        assert!((stereo.data[0] - 1.0).abs() < 1e-6);
        assert!(stereo.data[1].abs() < 1e-6);
    }

    #[test]
    fn test_pan_hard_right() {
        let mono = AudioBuffer::from_mono(vec![1.0], 44100);
        let stereo = mono.pan(1.0);
        // Hard right: angle = PI/2, cos(PI/2)≈0, sin(PI/2)=1
        assert!(stereo.data[0].abs() < 1e-6);
        assert!((stereo.data[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pan_stereo_noop() {
        let stereo = AudioBuffer::from_stereo(vec![0.5, 0.7, 0.5, 0.7], 44100);
        let panned = stereo.pan(0.5);
        assert_eq!(panned.data, stereo.data);
    }

    #[test]
    fn test_pan_constant_power() {
        // Constant power: L^2 + R^2 should be approximately constant across pan positions
        let mono = AudioBuffer::from_mono(vec![1.0], 44100);
        let power_center = {
            let s = mono.pan(0.0);
            s.data[0] * s.data[0] + s.data[1] * s.data[1]
        };
        let power_left = {
            let s = mono.pan(-1.0);
            s.data[0] * s.data[0] + s.data[1] * s.data[1]
        };
        let power_right = {
            let s = mono.pan(1.0);
            s.data[0] * s.data[0] + s.data[1] * s.data[1]
        };
        assert!(
            (power_center - power_left).abs() < 0.01,
            "Constant power violated: center={}, left={}",
            power_center,
            power_left
        );
        assert!(
            (power_center - power_right).abs() < 0.01,
            "Constant power violated: center={}, right={}",
            power_center,
            power_right
        );
    }

    #[test]
    fn test_pan_empty() {
        let mono = AudioBuffer::from_mono(vec![], 44100);
        let stereo = mono.pan(0.0);
        assert!(stereo.is_stereo());
        assert!(stereo.is_empty());
    }

    #[test]
    #[should_panic(expected = "pan must be in")]
    fn test_pan_out_of_range() {
        let mono = AudioBuffer::from_mono(vec![1.0], 44100);
        mono.pan(1.5);
    }

    // --- with_gain_envelope() tests ---

    #[test]
    fn test_gain_envelope_fade_out() {
        let buf = AudioBuffer::from_mono(vec![1.0; 100], 100); // 1 sec at 100 Hz
        let faded = buf.with_gain_envelope(&[(0.0, 1.0), (1.0, 0.0)]);
        // First sample should be ~1.0, last should be ~0.0
        assert!((faded.data[0] - 1.0).abs() < 0.02);
        assert!(faded.data[99].abs() < 0.02);
        // Middle should be ~0.5
        assert!((faded.data[50] - 0.5).abs() < 0.02);
    }

    #[test]
    fn test_gain_envelope_fade_in() {
        let buf = AudioBuffer::from_mono(vec![1.0; 100], 100);
        let faded = buf.with_gain_envelope(&[(0.0, 0.0), (1.0, 1.0)]);
        assert!(faded.data[0].abs() < 0.02);
        assert!((faded.data[99] - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_gain_envelope_constant() {
        let buf = AudioBuffer::from_mono(vec![1.0; 100], 100);
        let scaled = buf.with_gain_envelope(&[(0.0, 0.5)]);
        for &s in &scaled.data {
            assert!((s - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gain_envelope_empty_breakpoints() {
        let buf = AudioBuffer::from_mono(vec![1.0; 100], 100);
        let same = buf.with_gain_envelope(&[]);
        assert_eq!(same.data, buf.data);
    }

    #[test]
    fn test_gain_envelope_empty_buffer() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let same = buf.with_gain_envelope(&[(0.0, 1.0), (1.0, 0.0)]);
        assert!(same.is_empty());
    }

    #[test]
    fn test_gain_envelope_stereo() {
        let buf = AudioBuffer::from_stereo(vec![1.0, 0.5, 1.0, 0.5], 2); // 2 frames at 2 Hz
        let faded = buf.with_gain_envelope(&[(0.0, 1.0), (1.0, 0.0)]);
        // Frame 0 (time=0.0): gain=1.0
        assert!((faded.data[0] - 1.0).abs() < 1e-6); // L
        assert!((faded.data[1] - 0.5).abs() < 1e-6); // R
                                                     // Frame 1 (time=0.5): gain=0.5
        assert!((faded.data[2] - 0.5).abs() < 1e-6); // L
        assert!((faded.data[3] - 0.25).abs() < 1e-6); // R
    }

    #[test]
    fn test_gain_envelope_multi_segment() {
        let buf = AudioBuffer::from_mono(vec![1.0; 100], 100);
        // V-shape: full -> silent -> full
        let shaped = buf.with_gain_envelope(&[(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)]);
        assert!((shaped.data[0] - 1.0).abs() < 0.02);
        assert!(shaped.data[50].abs() < 0.02);
        assert!((shaped.data[99] - 1.0).abs() < 0.02);
    }

    // --- remove_dc() tests ---

    #[test]
    fn test_remove_dc_basic() {
        let buf = AudioBuffer::from_mono(vec![1.5, 1.6, 1.4, 1.5], 44100);
        let centered = buf.remove_dc();
        let mean: f64 =
            centered.data.iter().map(|&s| s as f64).sum::<f64>() / centered.data.len() as f64;
        assert!(mean.abs() < 1e-5, "DC should be removed, got mean={}", mean);
    }

    #[test]
    fn test_remove_dc_zero_mean() {
        // Already centered signal should be unchanged
        let data = vec![1.0, -1.0, 1.0, -1.0];
        let buf = AudioBuffer::from_mono(data.clone(), 44100);
        let centered = buf.remove_dc();
        for (a, b) in centered.data.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_remove_dc_stereo() {
        // L channel has DC offset of 1.0, R channel has DC offset of -0.5
        let buf = AudioBuffer::from_stereo(vec![1.0, -0.5, 1.0, -0.5, 1.0, -0.5], 44100);
        let centered = buf.remove_dc();
        // Both channels should now be centered
        let left: Vec<f32> = centered.data.iter().step_by(2).copied().collect();
        let right: Vec<f32> = centered.data.iter().skip(1).step_by(2).copied().collect();
        let l_mean: f64 = left.iter().map(|&s| s as f64).sum::<f64>() / left.len() as f64;
        let r_mean: f64 = right.iter().map(|&s| s as f64).sum::<f64>() / right.len() as f64;
        assert!(
            l_mean.abs() < 1e-5,
            "L DC should be removed, got {}",
            l_mean
        );
        assert!(
            r_mean.abs() < 1e-5,
            "R DC should be removed, got {}",
            r_mean
        );
    }

    #[test]
    fn test_remove_dc_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let centered = buf.remove_dc();
        assert!(centered.is_empty());
    }

    // --- apply_window() tests ---

    #[test]
    fn test_apply_window_hann() {
        let buf = AudioBuffer::from_mono(vec![1.0; 100], 44100);
        let windowed = buf.apply_window(crate::core::window::WindowType::Hann);
        // First and last should be near zero
        assert!(windowed.data[0].abs() < 0.01);
        assert!(windowed.data[99].abs() < 0.01);
        // Middle should be close to 1.0
        assert!((windowed.data[50] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_apply_window_stereo() {
        let buf = AudioBuffer::from_stereo(vec![1.0, 0.5, 1.0, 0.5, 1.0, 0.5], 44100);
        let windowed = buf.apply_window(crate::core::window::WindowType::Hann);
        // Both channels should be windowed identically (proportionally)
        for frame in windowed.frames() {
            if frame[0].abs() > 1e-6 {
                let ratio = frame[1] / frame[0];
                assert!((ratio - 0.5).abs() < 0.01, "L/R ratio should be preserved");
            }
        }
    }

    #[test]
    fn test_apply_window_empty() {
        let buf = AudioBuffer::from_mono(vec![], 44100);
        let windowed = buf.apply_window(crate::core::window::WindowType::Hann);
        assert!(windowed.is_empty());
    }

    #[test]
    fn test_apply_window_preserves_metadata() {
        let buf = AudioBuffer::from_mono(vec![1.0; 100], 48000);
        let windowed = buf.apply_window(crate::core::window::WindowType::BlackmanHarris);
        assert_eq!(windowed.sample_rate, 48000);
        assert_eq!(windowed.channels, Channels::Mono);
        assert_eq!(windowed.num_frames(), 100);
    }

    // --- BPM field tests ---

    #[test]
    fn test_with_bpm() {
        let params = StretchParams::new(1.0).with_bpm(128.0);
        assert_eq!(params.bpm, Some(128.0));
    }

    #[test]
    fn test_bpm_default_is_none() {
        let params = StretchParams::new(1.0);
        assert_eq!(params.bpm, None);
    }

    #[test]
    fn test_bpm_none_after_preset() {
        let params = StretchParams::new(1.0).with_preset(EdmPreset::DjBeatmatch);
        assert_eq!(params.bpm, None);
    }

    #[test]
    fn test_bpm_preserved_after_preset() {
        // BPM set before preset should be overwritten (preset doesn't set BPM)
        let params = StretchParams::new(1.0)
            .with_bpm(140.0)
            .with_preset(EdmPreset::HouseLoop);
        // Preset does not reset bpm, so it should be preserved
        assert_eq!(params.bpm, Some(140.0));
    }

    #[test]
    fn test_from_tempo_bpm_is_none() {
        let params = StretchParams::from_tempo(126.0, 128.0);
        assert_eq!(params.bpm, None);
    }

    // --- Dynamic WSOLA search tests ---

    #[test]
    fn test_dynamic_wsola_search_default_is_false() {
        let params = StretchParams::new(1.0);
        assert!(!params.dynamic_wsola_search);
    }

    #[test]
    fn test_with_dynamic_wsola_search() {
        let params = StretchParams::new(1.0).with_dynamic_wsola_search(true);
        assert!(params.dynamic_wsola_search);
    }

    #[test]
    fn test_preset_enables_dynamic_wsola_search() {
        // DjBeatmatch, HouseLoop, Halftime should enable it
        let dj = StretchParams::new(1.0).with_preset(EdmPreset::DjBeatmatch);
        assert!(dj.dynamic_wsola_search);

        let house = StretchParams::new(1.0).with_preset(EdmPreset::HouseLoop);
        assert!(house.dynamic_wsola_search);

        let halftime = StretchParams::new(2.0).with_preset(EdmPreset::Halftime);
        assert!(halftime.dynamic_wsola_search);

        // Ambient and VocalChop should not enable it
        let ambient = StretchParams::new(3.0).with_preset(EdmPreset::Ambient);
        assert!(!ambient.dynamic_wsola_search);

        let vocal = StretchParams::new(1.0).with_preset(EdmPreset::VocalChop);
        assert!(!vocal.dynamic_wsola_search);
    }

    #[test]
    fn test_effective_wsola_search_ms_static() {
        // With dynamic disabled, should return the static value
        let params = StretchParams::new(2.0)
            .with_sample_rate(44100)
            .with_wsola_search_range(ms_to_samples(10.0, 44100));
        let ms = params.effective_wsola_search_ms();
        assert!((ms - 10.0).abs() < 0.5, "Expected ~10ms, got {}", ms);
    }

    #[test]
    fn test_effective_wsola_search_ms_dynamic_scales() {
        // With dynamic enabled at 2x ratio, search should double
        let params = StretchParams::new(2.0)
            .with_sample_rate(44100)
            .with_wsola_search_range(ms_to_samples(10.0, 44100))
            .with_dynamic_wsola_search(true);
        let ms = params.effective_wsola_search_ms();
        assert!((ms - 20.0).abs() < 0.5, "Expected ~20ms, got {}", ms);
    }

    #[test]
    fn test_effective_wsola_search_ms_dynamic_caps_at_40() {
        // At extreme stretch ratio (10x), should cap at 40ms
        let params = StretchParams::new(10.0)
            .with_sample_rate(44100)
            .with_wsola_search_range(ms_to_samples(10.0, 44100))
            .with_dynamic_wsola_search(true);
        let ms = params.effective_wsola_search_ms();
        assert!((ms - 40.0).abs() < 0.5, "Expected cap at 40ms, got {}", ms);
    }

    #[test]
    fn test_effective_wsola_search_ms_dynamic_compress() {
        // With ratio < 1, should use 1/ratio as factor
        let params = StretchParams::new(0.5)
            .with_sample_rate(44100)
            .with_wsola_search_range(ms_to_samples(10.0, 44100))
            .with_dynamic_wsola_search(true);
        let ms = params.effective_wsola_search_ms();
        assert!((ms - 20.0).abs() < 0.5, "Expected ~20ms, got {}", ms);
    }

    #[test]
    fn test_effective_wsola_search_range_static() {
        // With dynamic disabled, should return the raw value
        let params = StretchParams::new(2.0)
            .with_sample_rate(44100)
            .with_wsola_search_range(100);
        assert_eq!(params.effective_wsola_search_range(), 100);
    }

    #[test]
    fn test_effective_wsola_search_range_dynamic_scales() {
        // At ratio=2.0, base=441 (10ms) → 882 (20ms)
        let params = StretchParams::new(2.0)
            .with_sample_rate(44100)
            .with_wsola_search_range(441)
            .with_dynamic_wsola_search(true);
        assert_eq!(params.effective_wsola_search_range(), 882);
    }

    #[test]
    fn test_effective_wsola_search_range_dynamic_symmetric() {
        // ratio=0.5 should use 1/0.5 = 2.0 as factor, same as ratio=2.0
        let params = StretchParams::new(0.5)
            .with_sample_rate(44100)
            .with_wsola_search_range(441)
            .with_dynamic_wsola_search(true);
        assert_eq!(params.effective_wsola_search_range(), 882);
    }

    #[test]
    fn test_effective_wsola_search_range_dynamic_caps_at_40ms() {
        // base=1323 (30ms) at ratio=2.0 → 2646 (60ms) but capped at 1764 (40ms)
        let params = StretchParams::new(2.0)
            .with_sample_rate(44100)
            .with_wsola_search_range(1323)
            .with_dynamic_wsola_search(true);
        let max_samples = (0.040 * 44100.0) as usize; // 1764
        assert_eq!(params.effective_wsola_search_range(), max_samples);
    }

    // --- Preset tuning tests ---

    #[test]
    fn test_dj_beatmatch_tuning() {
        let params = StretchParams::new(1.02).with_preset(EdmPreset::DjBeatmatch);
        assert_eq!(params.fft_size, 4096);
        assert_eq!(params.hop_size, 1024); // 75% overlap
        assert!((params.transient_sensitivity - 0.3).abs() < f32::EPSILON);
        assert_eq!(params.window_type, WindowType::Hann);
        assert!(params.dynamic_wsola_search);
    }

    #[test]
    fn test_house_loop_tuning() {
        let params = StretchParams::new(1.5).with_preset(EdmPreset::HouseLoop);
        assert_eq!(params.fft_size, 4096);
        assert_eq!(params.hop_size, 1024); // 75% overlap
        assert!((params.transient_region_secs - 0.030).abs() < 1e-6); // 30ms
        assert_eq!(params.window_type, WindowType::Hann);
        assert!(params.dynamic_wsola_search);
    }

    #[test]
    fn test_halftime_tuning() {
        let params = StretchParams::new(2.0).with_preset(EdmPreset::Halftime);
        assert_eq!(params.fft_size, 4096);
        assert_eq!(params.hop_size, 1024); // 75% overlap
        assert!((params.transient_region_secs - 0.020).abs() < 1e-6); // 20ms
        assert_eq!(params.window_type, WindowType::Hann);
        assert!(params.dynamic_wsola_search);
        assert_eq!(params.crossfade_mode, CrossfadeMode::Adaptive);
    }

    #[test]
    fn test_ambient_tuning() {
        let params = StretchParams::new(3.0).with_preset(EdmPreset::Ambient);
        assert_eq!(params.fft_size, 8192);
        assert_eq!(params.hop_size, 4096); // 50% overlap
        assert_eq!(params.window_type, WindowType::BlackmanHarris);
        assert_eq!(params.crossfade_mode, CrossfadeMode::Adaptive);
        assert!(!params.dynamic_wsola_search);
    }

    #[test]
    fn test_vocal_chop_tuning() {
        let params = StretchParams::new(1.0).with_preset(EdmPreset::VocalChop);
        assert_eq!(params.fft_size, 4096); // increased from 2048
        assert_eq!(params.hop_size, 1024);
        assert!((params.transient_sensitivity - 0.6).abs() < f32::EPSILON);
        assert_eq!(params.window_type, WindowType::Hann);
    }

    #[test]
    fn test_default_hop_size_is_fft_div_4() {
        let params = StretchParams::new(1.0);
        assert_eq!(params.hop_size, params.fft_size / 4);
    }
}

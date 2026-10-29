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
/// let buf = AudioBuffer::from_mono(vec![0.0; 44100], 44100).unwrap();
/// assert_eq!(buf.num_frames(), 44100);
/// assert!((buf.duration_secs() - 1.0).abs() < 1e-10);
///
/// let stereo = AudioBuffer::from_stereo(vec![0.0; 88200], 44100).unwrap();
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
    /// Validate frame geometry, sample rate and finite samples before owning external audio.
    pub fn try_new(
        data: Vec<Sample>,
        sample_rate: u32,
        channels: Channels,
    ) -> Result<Self, crate::StretchError> {
        if sample_rate == 0 || sample_rate > 768000 || data.len() % channels.count() != 0 {
            return Err(crate::StretchError::InvalidFormat(
                "invalid buffer geometry or rate".into(),
            ));
        }
        if data.iter().any(|s| !s.is_finite()) {
            return Err(crate::StretchError::NonFiniteInput);
        }
        Ok(Self {
            data,
            sample_rate,
            channels,
        })
    }

    /// Creates a validated buffer. Invalid rate, interleave or nonfinite samples are errors.
    pub fn new(
        data: Vec<Sample>,
        sample_rate: u32,
        channels: Channels,
    ) -> Result<Self, crate::StretchError> {
        Self::try_new(data, sample_rate, channels)
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
        if ch >= nc {
            return Vec::new();
        }
        self.data.iter().skip(ch).step_by(nc).copied().collect()
    }

    /// Consumes the buffer and returns the raw sample data.
    ///
    /// This is the named equivalent of `Vec::<f32>::from(buffer)`.
    #[inline]
    pub fn into_data(self) -> Vec<Sample> {
        self.data
    }

    /// Creates a validated mono buffer.
    pub fn from_mono(data: Vec<Sample>, sample_rate: u32) -> Result<Self, crate::StretchError> {
        Self::new(data, sample_rate, Channels::Mono)
    }
    /// Creates a validated interleaved stereo buffer.
    pub fn from_stereo(data: Vec<Sample>, sample_rate: u32) -> Result<Self, crate::StretchError> {
        Self::new(data, sample_rate, Channels::Stereo)
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
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let buf = AudioBuffer::from_mono(vec![0.0, 0.1, 0.2, 0.3, 0.4], 44100).unwrap();
    /// let sub = buf.slice(1, 3);
    /// assert_eq!(sub.data, vec![0.1, 0.2, 0.3]);
    /// ```
    pub fn slice(&self, start_frame: usize, num_frames: usize) -> Self {
        let total_frames = self.num_frames();
        if start_frame >= total_frames {
            return Self {
                data: Vec::new(),
                sample_rate: self.sample_rate,
                channels: self.channels,
            };
        }
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
    /// All buffers should have the same sample rate and channel layout.
    /// If they do not match, returns an empty buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let a = AudioBuffer::from_mono(vec![1.0, 2.0], 44100).unwrap();
    /// let b = AudioBuffer::from_mono(vec![3.0, 4.0], 44100).unwrap();
    /// let combined = AudioBuffer::concatenate(&[&a, &b]);
    /// assert_eq!(combined.data, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn concatenate(buffers: &[&AudioBuffer]) -> Self {
        if buffers.is_empty() {
            return Self {
                data: vec![],
                sample_rate: 44100,
                channels: Channels::Mono,
            };
        }
        let sample_rate = buffers[0].sample_rate;
        let channels = buffers[0].channels;
        let total_len: usize = buffers.iter().map(|b| b.data.len()).sum();
        let mut data = Vec::with_capacity(total_len);
        for buf in buffers {
            if buf.sample_rate != sample_rate || buf.channels != channels {
                return Self {
                    data: Vec::new(),
                    sample_rate,
                    channels,
                };
            }
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
            None => {
                return Self {
                    data: vec![],
                    sample_rate: self.sample_rate,
                    channels: self.channels,
                };
            }
        };

        // Find last non-silent frame
        let last = (0..total_frames)
            .rev()
            .find(|&frame| (0..nc).any(|ch| self.data[frame * nc + ch].abs() >= threshold))
            .unwrap_or(first);

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
    /// let buf = AudioBuffer::from_stereo(vec![1.0, 2.0, 3.0, 4.0], 44100).unwrap();
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
    /// let buf = AudioBuffer::from_mono(vec![0.25, -0.8, 0.5], 44100).unwrap();
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
    /// let buf = AudioBuffer::from_mono(vec![1.0, -1.0], 44100).unwrap();
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
    /// let buf = AudioBuffer::from_mono(vec![1.0; 100], 44100).unwrap();
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
    /// let buf = AudioBuffer::from_mono(vec![1.0; 100], 44100).unwrap();
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
    /// let buf = AudioBuffer::from_mono(vec![0.0; 44100], 44100).unwrap(); // 1 second
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

        // Resample each channel independently with the band-limited
        // windowed-sinc converter (ROADMAP Stage 17): cubic had no
        // anti-aliasing at all, so 48 -> 44.1 kHz conversions folded
        // ultrasonic content into the audible band.
        let mut output = Vec::with_capacity(target_frames * nc);
        for ch in 0..nc {
            let channel_data: Vec<f32> = self.data.iter().skip(ch).step_by(nc).copied().collect();
            let resampled =
                crate::support::resample::resample_sinc_default(&channel_data, target_frames);
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
    /// If the buffers have different sample rates or channel layouts, returns
    /// a clone of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let a = AudioBuffer::from_mono(vec![1.0; 1000], 44100).unwrap();
    /// let b = AudioBuffer::from_mono(vec![0.5; 1000], 44100).unwrap();
    /// let mixed = a.crossfade_into(&b, 100);
    /// assert_eq!(mixed.num_frames(), 1900); // 1000 + 1000 - 100
    /// ```
    pub fn crossfade_into(&self, other: &AudioBuffer, crossfade_frames: usize) -> Self {
        if self.sample_rate != other.sample_rate || self.channels != other.channels {
            return self.clone();
        }

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
    /// let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0], 44100).unwrap();
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
    /// let mono = AudioBuffer::from_mono(vec![0.0; 100], 44100).unwrap();
    /// assert_eq!(mono.channel_count(), 1);
    ///
    /// let stereo = AudioBuffer::from_stereo(vec![0.0; 200], 44100).unwrap();
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
    /// let buf = AudioBuffer::from_mono(vec![1.0, 2.0, 3.0, 4.0], 44100).unwrap();
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
    /// let buf = AudioBuffer::from_mono(vec![1.0, 2.0], 44100).unwrap();
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
    /// If the buffers have different sample rates or channel layouts, returns
    /// a clone of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let a = AudioBuffer::from_mono(vec![0.5, 0.5], 44100).unwrap();
    /// let b = AudioBuffer::from_mono(vec![0.3, 0.3], 44100).unwrap();
    /// let mixed = a.mix(&b);
    /// assert!((mixed.data[0] - 0.8).abs() < 1e-6);
    /// ```
    pub fn mix(&self, other: &AudioBuffer) -> Self {
        if self.sample_rate != other.sample_rate || self.channels != other.channels {
            return self.clone();
        }

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
    /// # Example
    ///
    /// ```
    /// use timestretch::AudioBuffer;
    ///
    /// let mono = AudioBuffer::from_mono(vec![1.0, 1.0], 44100).unwrap();
    /// let stereo = mono.pan(0.0); // center
    /// assert!(stereo.is_stereo());
    /// // Both channels should have equal amplitude at center pan
    /// let (l, r) = (stereo.data[0], stereo.data[1]);
    /// assert!((l - r).abs() < 1e-6);
    /// ```
    pub fn pan(&self, pan: f32) -> Self {
        assert!(
            (-1.0..=1.0).contains(&pan),
            "pan must be in [-1.0, 1.0], got {pan}"
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
    /// let buf = AudioBuffer::from_mono(vec![1.0; 44100], 44100).unwrap();
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
    /// let buf = AudioBuffer::from_mono(vec![1.5, 1.6, 1.4, 1.5], 44100).unwrap();
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
    /// let buf = AudioBuffer::from_mono(vec![1.0; 100], 44100).unwrap();
    /// let windowed = buf.apply_window(WindowType::Hann);
    /// // First and last samples should be near zero (Hann window)
    /// assert!(windowed.data[0].abs() < 0.01);
    /// assert!(windowed.data[99].abs() < 0.01);
    /// // Middle should be close to 1.0
    /// assert!((windowed.data[50] - 1.0).abs() < 0.1);
    /// ```
    pub fn apply_window(&self, window_type: crate::support::window::WindowType) -> Self {
        if self.data.is_empty() {
            return self.clone();
        }

        let num_frames = self.num_frames();
        let window = crate::support::window::generate_window(window_type, num_frames);
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

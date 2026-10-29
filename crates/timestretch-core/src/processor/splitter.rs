use super::fft::PackedFft;
use crate::StretchError;

/// Stateful default (unity-envelope) splitter. A is full-band and B high-pass;
/// both have an initial half-frame delay. They overlap in spectrum.
pub struct Splitter {
    size: usize,
    hop: usize,
    channels: usize,
    capacity: usize,
    buffered: usize,
    input: Vec<Vec<f32>>,
    a: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    pub(super) window: Vec<f32>,
    spectrum: Vec<f32>,
    fft: PackedFft,
    cutoff: usize,
}
impl Splitter {
    /// Allocate plans and all sample storage. `capacity` includes buffered input.
    pub fn new(
        size: usize,
        sample_rate: u32,
        channels: usize,
        capacity: usize,
    ) -> Result<Self, StretchError> {
        Self::new_with_numerics(
            size,
            sample_rate,
            channels,
            capacity,
            crate::Numerics::Portable,
        )
    }
    pub fn new_with_numerics(
        size: usize,
        sample_rate: u32,
        channels: usize,
        capacity: usize,
        numerics: crate::Numerics,
    ) -> Result<Self, StretchError> {
        if !numerics.available() {
            return Err(super::invalid("numerical backend unavailable"));
        }
        super::check_channels(channels)?;
        super::check_capacity(capacity)?;
        if !size.is_power_of_two()
            || !(256..=8192).contains(&size)
            || capacity < size
            || !(8000..=192000).contains(&sample_rate)
        {
            return Err(super::invalid("invalid splitter geometry"));
        }
        let window = (0..size)
            .map(|i| {
                (std::f32::consts::PI * i as f32 * (1.0 / size as f32)).sin()
                    * (1.0 / size as f32).sqrt()
            })
            .collect();
        Ok(Self {
            size,
            hop: size / 2,
            channels,
            capacity,
            buffered: size / 2,
            input: vec![vec![0.0; capacity + size / 2]; channels],
            a: vec![vec![0.0; size]; channels],
            b: vec![vec![0.0; size]; channels],
            window,
            spectrum: vec![0.0; size],
            fft: PackedFft::new(size, numerics),
            cutoff: (size as f32 * 1500.0 / sample_rate as f32).floor() as usize,
        })
    }
    /// Initial delay of this component only, in source samples.
    pub fn delay_frames(&self) -> usize {
        self.hop
    }
    /// Predict output count without advancing the splitter.
    pub fn output_frames(&self, frames: usize) -> Result<usize, StretchError> {
        if frames > self.capacity - self.buffered {
            return Err(super::invalid("splitter input exceeds remaining capacity"));
        }
        let total = self.buffered + frames;
        Ok(if total >= self.size {
            (total - self.size) / self.hop * self.hop + self.hop
        } else {
            0
        })
    }
    pub(super) fn clear_storage(&mut self) {
        for a in self.input.iter_mut().chain(&mut self.a).chain(&mut self.b) {
            a.fill(0.);
        }
    }
    /// Restore the half-frame input count, clear the leading half-frame of input
    /// and A/B buffers, and clear the spectrum. Other sample storage persists.
    pub fn reset(&mut self) {
        self.buffered = self.hop;
        for ch in 0..self.channels {
            self.input[ch][..self.hop].fill(0.0);
            self.a[ch][..self.hop].fill(0.0);
            self.b[ch][..self.hop].fill(0.0);
        }
        self.spectrum.fill(0.0);
    }
    /// Process planar input into equally sized A/B blocks. No allocation.
    pub fn process(
        &mut self,
        input: &[&[f32]],
        out_a: &mut [&mut [f32]],
        out_b: &mut [&mut [f32]],
    ) -> Result<usize, StretchError> {
        #[cfg(feature = "qa")]
        let _profile = crate::processor::qa::region(1);
        let frames = super::input_frames(input, self.channels)?;
        if frames > self.capacity - self.buffered {
            return Err(StretchError::BufferOverflow {
                buffer: "splitter",
                requested: frames,
                available: self.capacity - self.buffered,
            });
        }
        let total = self.buffered + frames;
        let produced = if total >= self.size {
            (total - self.size) / self.hop * self.hop + self.hop
        } else {
            0
        };
        super::output_capacity(out_a, self.channels, produced)?;
        super::output_capacity(out_b, self.channels, produced)?;
        for (ch, source) in input.iter().enumerate() {
            self.input[ch][self.buffered..total].copy_from_slice(source);
        }
        self.buffered = total;
        let mut written = 0;
        while self.buffered >= self.size {
            for ch in 0..self.channels {
                for i in 0..self.size {
                    self.spectrum[i] = self.input[ch][i] * self.window[i];
                }
                self.fft.forward(&mut self.spectrum);
                for i in 0..self.size {
                    let mut value = (self.input[ch][i] * self.size as f32) * self.window[i];
                    value *= self.window[i];
                    self.a[ch][i] += value;
                }
                let rise = (self.size >> 6).min(16) >> 1;
                let start = self.cutoff.saturating_sub(rise);
                self.spectrum[..start].fill(0.0);
                self.spectrum[self.hop..self.hop + start].fill(0.0);
                for j in 0..rise {
                    let k = (self.cutoff as isize - rise as isize + j as isize).max(0) as usize;
                    let gain = j as f32 / rise as f32;
                    self.spectrum[k] *= gain;
                    self.spectrum[k + self.hop] *= gain;
                }
                self.fft.inverse(&mut self.spectrum);
                for i in 0..self.size {
                    self.spectrum[i] *= self.window[i];
                    self.b[ch][i] += self.spectrum[i];
                }
                out_a[ch][written..written + self.hop].copy_from_slice(&self.a[ch][..self.hop]);
                out_b[ch][written..written + self.hop].copy_from_slice(&self.b[ch][..self.hop]);
                self.a[ch].copy_within(self.hop.., 0);
                self.b[ch].copy_within(self.hop.., 0);
                self.a[ch][self.hop..].fill(0.0);
                self.b[ch][self.hop..].fill(0.0);
            }
            self.buffered -= self.hop;
            for channel in &mut self.input {
                channel.copy_within(self.hop..self.hop + self.buffered, 0);
                channel[self.buffered..self.buffered + self.hop].fill(0.0);
            }
            written += self.hop;
        }
        Ok(written)
    }
}

use crate::memory::HeapBytes;
impl HeapBytes for Splitter {
    fn heap_bytes(&self) -> usize {
        self.input.heap_bytes()
            + self.a.heap_bytes()
            + self.b.heap_bytes()
            + self.window.heap_bytes()
            + self.spectrum.heap_bytes()
            + self.fft.heap_bytes()
    }
}

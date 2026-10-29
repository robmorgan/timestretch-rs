//! Owned staging for borrowed planar input, with power-of-two rings for retention.
use crate::StretchError;
pub(super) type Planar = [Vec<f32>; 2];
pub(super) fn planes(n: usize) -> Planar {
    [vec![0.0; n], vec![0.0; n]]
}
pub(super) fn views(p: &Planar, n: usize) -> [&[f32]; 2] {
    [&p[0][..n], &p[1][..n]]
}
pub(super) fn outputs(p: &mut Planar) -> [&mut [f32]; 2] {
    let [a, b] = p;
    [a.as_mut_slice(), b.as_mut_slice()]
}
pub(super) struct Source {
    channels: usize,
    capacity: usize,
    pending: Planar,
    ring: Planar,
    pub index: usize,
    pub count: usize,
    pub read: usize,
    pub write: usize,
}
impl Source {
    pub fn new(channels: usize, capacity: usize, max_input: usize) -> Self {
        Self {
            channels,
            capacity,
            pending: planes(max_input),
            ring: planes((capacity + 1).next_power_of_two()),
            index: 0,
            count: 0,
            read: 0,
            write: 0,
        }
    }
    fn buffered(&self) -> usize {
        (self.write + self.ring[0].len() - self.read) % self.ring[0].len()
    }
    pub fn available(&self) -> usize {
        self.buffered() + self.count - self.index
    }
    pub fn release(&mut self) -> Result<(), StretchError> {
        if self.count == 0 {
            return Ok(());
        }
        let n = self.count - self.index;
        if self.buffered() + n > self.capacity {
            return Err(super::invalid("scheduler queue overflow"));
        }
        let mask = self.ring[0].len() - 1;
        for ch in 0..self.channels {
            for i in 0..n {
                self.ring[ch][(self.write + i) & mask] = self.pending[ch][self.index + i];
            }
        }
        self.write = (self.write + n) & mask;
        self.index = 0;
        self.count = 0;
        Ok(())
    }
    pub fn set(&mut self, input: &[&[f32]], n: usize) -> Result<(), StretchError> {
        self.release()?;
        if n > self.pending[0].len() || input.iter().take(self.channels).any(|x| x.len() < n) {
            return Err(super::invalid("scheduler input exceeds storage"));
        }
        for (ch, source) in input.iter().enumerate().take(self.channels) {
            self.pending[ch][..n].copy_from_slice(&source[..n]);
        }
        self.count = n;
        Ok(())
    }
    pub fn take(&mut self, out: &mut [&mut [f32]], n: usize) -> bool {
        if self.available() < n {
            return false;
        }
        let buffered = self.buffered().min(n);
        let mask = self.ring[0].len() - 1;
        for (ch, dst) in out.iter_mut().enumerate().take(self.channels) {
            for (i, v) in dst.iter_mut().enumerate().take(buffered) {
                *v = self.ring[ch][(self.read + i) & mask];
            }
            dst[buffered..n]
                .copy_from_slice(&self.pending[ch][self.index..self.index + n - buffered]);
        }
        self.advance(n);
        true
    }
    pub fn advance(&mut self, n: usize) {
        let buffered = self.buffered().min(n);
        self.read = (self.read + buffered) & (self.ring[0].len() - 1);
        self.index += n - buffered;
    }
    pub fn reset(&mut self) {
        self.index = 0;
        self.count = 0;
        self.read = 0;
        self.write = 0;
        for ch in &mut self.ring {
            ch.fill(0.0);
        }
    }
}

use crate::memory::HeapBytes;
impl HeapBytes for Source {
    fn heap_bytes(&self) -> usize {
        self.pending.heap_bytes() + self.ring.heap_bytes()
    }
}

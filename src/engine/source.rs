//! The pull engine's source-supply contract.
//!
//! A pull engine demands a ratio-dependent, variable amount of source per
//! output block, so the host cannot feed it synchronously the way the old
//! push API worked. Instead the host owns a [`SourceProducer`] and keeps a
//! lock-free ring topped up; the processor consumes whatever each block
//! needs. The contract is explicit:
//!
//! - **Occupancy guarantee:** before each callback the host should keep at
//!   least [`SourceProducer::demand_hint`] frames buffered; the ring is
//!   sized (at construction) for several callbacks of slack.
//! - **Underrun policy:** if the ring runs dry mid-block the engine emits
//!   silence for the shortfall, counts it (see
//!   [`EngineController::underrun_frames`](crate::engine::EngineController::underrun_frames)),
//!   and resumes seamlessly when data returns. It never blocks and never
//!   errors in the audio path.
//! - **Position bookkeeping:** frames are numbered from the first frame ever
//!   pushed; the crate-internal `TimelineMap` maps the engine's output
//!   timeline back to fractional source frames (the `RatioMapFifo`
//!   mechanism proven in the old varispeed control path).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use crate::core::resample::STREAM_SINC_MAX_HALF_TAPS;

/// Lock-free SPSC ring of interleaved f32 samples.
///
/// Samples are stored as `u32` bit patterns so the buffer can be shared
/// safely without `unsafe`. Cursors are monotonic u64 sample counts;
/// `% capacity` yields the slot index.
#[derive(Debug)]
pub(crate) struct SourceRing {
    slots: Box<[AtomicU32]>,
    /// Consumer cursor: total samples popped.
    head: AtomicU64,
    /// Producer cursor: total samples pushed.
    tail: AtomicU64,
    channels: usize,
}

impl SourceRing {
    pub(crate) fn new(capacity_frames: usize, channels: usize) -> Arc<Self> {
        let capacity = capacity_frames * channels;
        let slots: Vec<AtomicU32> = (0..capacity).map(|_| AtomicU32::new(0)).collect();
        Arc::new(Self {
            slots: slots.into_boxed_slice(),
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            channels,
        })
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Total ring capacity in samples (consumer-side drain bookkeeping).
    #[inline]
    pub(crate) fn capacity_samples(&self) -> usize {
        self.slots.len()
    }

    /// Samples currently buffered (consumer/producer agree eventually; each
    /// side sees a conservative value for its own operation).
    #[inline]
    pub(crate) fn occupied(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        tail.saturating_sub(head) as usize
    }

    /// Producer side: pushes as many samples as fit, returns count pushed.
    pub(crate) fn push_slice(&self, input: &[f32]) -> usize {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        let free = self.capacity() - tail.saturating_sub(head) as usize;
        let count = input.len().min(free);
        for (i, &sample) in input.iter().take(count).enumerate() {
            let idx = ((tail + i as u64) % self.capacity() as u64) as usize;
            self.slots[idx].store(sample.to_bits(), Ordering::Relaxed);
        }
        self.tail.store(tail + count as u64, Ordering::Release);
        count
    }

    /// Producer-side view of free space in samples.
    #[inline]
    pub(crate) fn free(&self) -> usize {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        self.capacity() - tail.saturating_sub(head) as usize
    }

    /// Consumer side: pops up to `out.len()` samples, returns count popped.
    pub(crate) fn pop_slice(&self, out: &mut [f32]) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let count = out.len().min(tail.saturating_sub(head) as usize);
        for (i, sample) in out.iter_mut().take(count).enumerate() {
            let idx = ((head + i as u64) % self.capacity() as u64) as usize;
            *sample = f32::from_bits(self.slots[idx].load(Ordering::Relaxed));
        }
        self.head.store(head + count as u64, Ordering::Release);
        count
    }

    #[inline]
    pub(crate) fn channels(&self) -> usize {
        self.channels
    }
}

/// Host-side handle for feeding source audio into the engine.
///
/// Single producer: exactly one thread may call [`push`](Self::push) /
/// [`finish`](Self::finish) at a time.
#[derive(Debug)]
pub struct SourceProducer {
    ring: Arc<SourceRing>,
    pushed_frames: u64,
}

impl SourceProducer {
    pub(crate) fn new(ring: Arc<SourceRing>) -> Self {
        Self {
            ring,
            pushed_frames: 0,
        }
    }

    /// Pushes interleaved source samples; returns the number of *frames*
    /// accepted (may be less than offered when the ring is near capacity —
    /// retry the remainder later). Frames are never split: the push is
    /// clamped up front to whole frames that fit, and since this is the only
    /// producer the ring cannot shrink between the check and the write.
    pub fn push(&mut self, interleaved: &[f32]) -> usize {
        let channels = self.ring.channels();
        debug_assert_eq!(interleaved.len() % channels, 0);
        let free_frames = self.ring.free() / channels;
        let frames = (interleaved.len() / channels).min(free_frames);
        let pushed = self.ring.push_slice(&interleaved[..frames * channels]);
        debug_assert_eq!(pushed, frames * channels);
        self.pushed_frames += frames as u64;
        frames
    }

    /// Frames currently buffered and not yet consumed by the engine.
    pub fn occupied_frames(&self) -> usize {
        self.ring.occupied() / self.ring.channels()
    }

    /// Free space in the ring, in frames.
    pub fn free_frames(&self) -> usize {
        self.ring.slots.len() / self.ring.channels() - self.occupied_frames()
    }

    /// Total frames pushed since construction.
    pub fn pushed_frames(&self) -> u64 {
        self.pushed_frames
    }

    /// Minimum frames the host should keep buffered to guarantee the next
    /// callback of `out_frames` frames renders without underrun at any
    /// supported tempo rate up to `max_rate`.
    pub fn demand_hint(&self, out_frames: usize, max_rate: f64) -> usize {
        let rate = max_rate.clamp(
            super::control::MIN_TEMPO_RATE,
            super::control::MAX_TEMPO_RATE,
        );
        (out_frames as f64 * rate).ceil() as usize + STREAM_SINC_MAX_HALF_TAPS + 2
    }

    /// Signals end of stream by pushing enough silence to flush the
    /// resampler lookahead, releasing every output frame that covers real
    /// source audio. Returns `true` once all padding is accepted (call again
    /// if the ring was too full).
    pub fn finish(&mut self) -> bool {
        let channels = self.ring.channels();
        let padding = [0.0f32; 64];
        let mut remaining_frames = STREAM_SINC_MAX_HALF_TAPS + 2;
        while remaining_frames > 0 {
            let batch_frames = remaining_frames.min(padding.len() / channels);
            let accepted = self.push(&padding[..batch_frames * channels]);
            if accepted == 0 {
                return false;
            }
            remaining_frames -= accepted;
        }
        true
    }
}

/// One output↔source timeline checkpoint recorded per varispeed feed chunk.
#[derive(Debug, Clone, Copy, Default)]
struct TimelineEntry {
    /// Engine-output frames emitted by the varispeed stage at the checkpoint.
    output_abs: u64,
    /// Fractional source frames consumed at the checkpoint.
    source_pos: f64,
    /// Tempo rate in effect at the checkpoint (the chunk's retarget value —
    /// the resampler's intra-chunk ramp lands on it at chunk end).
    rate: f64,
}

/// Fixed-size FIFO mapping the engine's output timeline back to source
/// frames (port of the old control path's `RatioMapFifo`).
///
/// The varispeed step is piecewise-linear per fed chunk, so one checkpoint
/// per chunk plus linear interpolation reconstructs the source position of
/// any in-flight output frame to sub-chunk accuracy. The buffer never grows:
/// pushes beyond capacity merge into the newest entry (bounded extra
/// interpolation error over one span), and eviction always retains one
/// anchor at or behind the consumption cursor.
#[derive(Debug)]
pub(crate) struct TimelineMap {
    entries: Box<[TimelineEntry]>,
    head: usize,
    len: usize,
}

impl TimelineMap {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: vec![TimelineEntry::default(); capacity.max(2)].into_boxed_slice(),
            head: 0,
            len: 0,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    #[inline]
    fn at(&self, i: usize) -> TimelineEntry {
        self.entries[(self.head + i) % self.entries.len()]
    }

    /// Appends a checkpoint, deduping zero-advance chunks and merging into
    /// the newest entry when full.
    pub(crate) fn push(&mut self, output_abs: u64, source_pos: f64, rate: f64) {
        if self.len > 0 {
            let last_idx = (self.head + self.len - 1) % self.entries.len();
            let last = self.entries[last_idx];
            if output_abs <= last.output_abs || self.len == self.entries.len() {
                self.entries[last_idx] = TimelineEntry {
                    output_abs: output_abs.max(last.output_abs),
                    source_pos: source_pos.max(last.source_pos),
                    rate,
                };
                return;
            }
        }
        let idx = (self.head + self.len) % self.entries.len();
        self.entries[idx] = TimelineEntry {
            output_abs,
            source_pos,
            rate,
        };
        self.len += 1;
    }

    /// Drops checkpoints strictly behind `consumed`, always retaining one
    /// anchor entry at or behind it.
    pub(crate) fn evict_before(&mut self, consumed: u64) {
        while self.len >= 2 && self.at(1).output_abs <= consumed {
            self.head = (self.head + 1) % self.entries.len();
            self.len -= 1;
        }
    }

    /// Maps an output-timeline frame count to fractional source frames.
    /// Clamps outside the checkpointed range; `None` before any checkpoint.
    pub(crate) fn map_to_source(&self, output: f64) -> Option<f64> {
        if self.len == 0 {
            return None;
        }
        let first = self.at(0);
        if output <= first.output_abs as f64 {
            return Some(first.source_pos);
        }
        for i in 1..self.len {
            let a = self.at(i - 1);
            let b = self.at(i);
            if output <= b.output_abs as f64 {
                let span = (b.output_abs - a.output_abs) as f64;
                if span <= 0.0 {
                    return Some(b.source_pos);
                }
                let t = (output - a.output_abs as f64) / span;
                return Some(a.source_pos + t * (b.source_pos - a.source_pos));
            }
        }
        Some(self.at(self.len - 1).source_pos)
    }

    /// Instantaneous embedded tempo rate at an output-timeline frame count,
    /// interpolating the per-checkpoint retarget values (this resolves the
    /// resampler's intra-chunk ramp itself). Clamps outside the range.
    /// Drives the keylock corrector's delay-matched transposition.
    pub(crate) fn rate_at(&self, output: f64) -> Option<f64> {
        if self.len == 0 {
            return None;
        }
        let first = self.at(0);
        if output <= first.output_abs as f64 {
            return Some(first.rate);
        }
        for i in 1..self.len {
            let a = self.at(i - 1);
            let b = self.at(i);
            if output <= b.output_abs as f64 {
                let span = (b.output_abs - a.output_abs) as f64;
                if span <= 0.0 {
                    return Some(b.rate);
                }
                let t = (output - a.output_abs as f64) / span;
                return Some(a.rate + t * (b.rate - a.rate));
            }
        }
        Some(self.at(self.len - 1).rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_push_pop_wraps() {
        let ring = SourceRing::new(4, 2); // 8 samples
        assert_eq!(ring.push_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 6);
        let mut out = [0.0f32; 4];
        assert_eq!(ring.pop_slice(&mut out), 4);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ring.push_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]), 6);
        let mut out8 = [0.0f32; 8];
        assert_eq!(ring.pop_slice(&mut out8), 8);
        assert_eq!(out8, [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn ring_bounded_capacity() {
        let ring = SourceRing::new(2, 1);
        assert_eq!(ring.push_slice(&[1.0, 2.0, 3.0]), 2);
        assert_eq!(ring.occupied(), 2);
    }

    #[test]
    fn producer_never_splits_frames() {
        let ring = SourceRing::new(3, 2); // 6 samples capacity
        let mut producer = SourceProducer::new(Arc::clone(&ring));
        // Odd sample occupancy: only 5 samples free — a naive push of the
        // remaining space would split a stereo frame. The producer clamps to
        // 2 whole frames instead.
        assert_eq!(ring.push_slice(&[0.5]), 1);
        let frames = producer.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(frames, 2);
        assert_eq!(ring.occupied(), 5);
        let mut out = [0.0f32; 5];
        assert_eq!(ring.pop_slice(&mut out), 5);
        assert_eq!(out, [0.5, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ring_cross_thread_transfers_all_samples() {
        let ring = SourceRing::new(64, 1);
        let producer_ring = Arc::clone(&ring);
        let total = 100_000usize;
        let producer = std::thread::spawn(move || {
            let mut sent = 0usize;
            while sent < total {
                let batch: Vec<f32> = (sent..(sent + 37).min(total)).map(|i| i as f32).collect();
                let mut offset = 0;
                while offset < batch.len() {
                    let pushed = producer_ring.push_slice(&batch[offset..]);
                    offset += pushed;
                    if pushed == 0 {
                        std::thread::yield_now();
                    }
                }
                sent += batch.len();
            }
        });
        let mut received = 0usize;
        let mut buf = [0.0f32; 41];
        while received < total {
            let n = ring.pop_slice(&mut buf);
            for &sample in &buf[..n] {
                assert_eq!(sample, received as f32, "out-of-order sample");
                received += 1;
            }
            if n == 0 {
                std::thread::yield_now();
            }
        }
        producer.join().unwrap();
    }

    #[test]
    fn timeline_map_interpolates_and_clamps() {
        let mut map = TimelineMap::with_capacity(4);
        assert_eq!(map.map_to_source(0.0), None);
        map.push(0, 0.0, 1.0);
        map.push(100, 110.0, 1.2);
        map.push(200, 230.0, 1.2);
        assert_eq!(map.map_to_source(0.0), Some(0.0));
        assert!((map.map_to_source(50.0).unwrap() - 55.0).abs() < 1e-9);
        assert!((map.map_to_source(150.0).unwrap() - 170.0).abs() < 1e-9);
        // Clamps past the newest checkpoint.
        assert!((map.map_to_source(500.0).unwrap() - 230.0).abs() < 1e-9);
        // Rate interpolation resolves the ramp.
        assert!((map.rate_at(50.0).unwrap() - 1.1).abs() < 1e-9);
    }

    #[test]
    fn timeline_map_merges_when_full_and_keeps_anchor() {
        let mut map = TimelineMap::with_capacity(2);
        map.push(0, 0.0, 1.0);
        map.push(10, 10.0, 1.0);
        map.push(20, 20.0, 1.0); // merges into newest
        assert!((map.map_to_source(20.0).unwrap() - 20.0).abs() < 1e-9);
        map.evict_before(25);
        // One anchor must remain even when everything is behind the cursor.
        assert!(map.map_to_source(25.0).is_some());
    }
}

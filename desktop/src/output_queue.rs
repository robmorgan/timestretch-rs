//! Desktop-owned SPSC device queue. It transports the library's epochs and
//! cumulative receipts; it does not calculate source positions or DSP demand.
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
};
use timestretch::{Epoch, OutputConsumption, OutputStamp};

struct Queue {
    samples: Box<[AtomicU32]>,
    capacity: u64,
    channels: usize,
    head: AtomicU64,
    tail: AtomicU64,
    generation: AtomicU64,
    epoch: AtomicU64,
    boundary: AtomicU64,
    receipt_generation: AtomicU64,
    receipt_epoch: AtomicU64,
    receipt_end: AtomicU64,
    ended: AtomicBool,
    underruns: AtomicU64,
}
/// Host-held lifetime guard: the audio thread never owns the final queue Arc.
pub struct QueueLifetime {
    _queue: Arc<Queue>,
}
pub struct Producer {
    queue: Arc<Queue>,
    epoch: Epoch,
    epoch_boundary: u64,
    next_frame: u64,
}
pub struct Consumer {
    queue: Arc<Queue>,
    epoch: Epoch,
    read_frames: u64,
    consumed_frames: u64,
}

pub fn channel(capacity_frames: usize) -> (Producer, Consumer) {
    channel_with_channels(capacity_frames, 2)
}
pub fn channel_with_channels(capacity_frames: usize, channels: usize) -> (Producer, Consumer) {
    assert!((1..=2).contains(&channels));
    assert!(capacity_frames >= 2048);
    let queue = Arc::new(Queue {
        samples: (0..capacity_frames * channels)
            .map(|_| AtomicU32::new(0))
            .collect(),
        channels,
        capacity: capacity_frames as u64,
        head: AtomicU64::new(0),
        tail: AtomicU64::new(0),
        generation: AtomicU64::new(0),
        epoch: AtomicU64::new(1),
        boundary: AtomicU64::new(0),
        receipt_generation: AtomicU64::new(0),
        receipt_epoch: AtomicU64::new(1),
        receipt_end: AtomicU64::new(0),
        ended: AtomicBool::new(false),
        underruns: AtomicU64::new(0),
    });
    (
        Producer {
            queue: queue.clone(),
            epoch: Epoch(1),
            epoch_boundary: 0,
            next_frame: 0,
        },
        Consumer {
            queue,
            epoch: Epoch(1),
            read_frames: 0,
            consumed_frames: 0,
        },
    )
}
impl Producer {
    pub fn free_frames(&self) -> usize {
        // Stale frames still occupy storage until the consumer acknowledges
        // the O(1) boundary. They cannot be overwritten early.
        self.queue.capacity as usize
            - (self.queue.tail.load(Ordering::Acquire) - self.queue.head.load(Ordering::Acquire))
                as usize
    }
    pub fn occupied_frames(&self) -> usize {
        // Only the current epoch can satisfy the worker's priming reserve.
        (self.queue.tail.load(Ordering::Acquire)
            - self
                .queue
                .head
                .load(Ordering::Acquire)
                .max(self.epoch_boundary)) as usize
    }
    pub fn publish_epoch(&mut self, epoch: Epoch) {
        self.queue.generation.fetch_add(1, Ordering::AcqRel);
        self.epoch_boundary = self.queue.tail.load(Ordering::Relaxed);
        self.queue
            .boundary
            .store(self.epoch_boundary, Ordering::Relaxed);
        self.queue.epoch.store(epoch.0, Ordering::Relaxed);
        self.queue.ended.store(false, Ordering::Relaxed);
        self.queue.generation.fetch_add(1, Ordering::Release);
        self.epoch = epoch;
        self.next_frame = 0;
    }
    pub fn push(&mut self, samples: &[f32], stamp: OutputStamp) -> bool {
        if stamp.epoch != self.epoch
            || stamp.start_frame != self.next_frame
            || samples.len() != stamp.frames * self.queue.channels
            || stamp.frames > self.free_frames()
        {
            return false;
        }
        let tail = self.queue.tail.load(Ordering::Relaxed);
        for (i, frame) in samples.chunks_exact(self.queue.channels).enumerate() {
            let at = ((tail + i as u64) % self.queue.capacity) as usize * self.queue.channels;
            for (ch, &x) in frame.iter().enumerate() {
                self.queue.samples[at + ch].store(x.to_bits(), Ordering::Relaxed);
            }
        }
        self.next_frame += stamp.frames as u64;
        self.queue
            .tail
            .store(tail + stamp.frames as u64, Ordering::Release);
        true
    }
    pub fn receipt(&self) -> Option<OutputConsumption> {
        let a = self.queue.receipt_generation.load(Ordering::Acquire);
        if a & 1 != 0 {
            return None;
        }
        let epoch = Epoch(self.queue.receipt_epoch.load(Ordering::Relaxed));
        let end_frame = self.queue.receipt_end.load(Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Acquire);
        let b = self.queue.receipt_generation.load(Ordering::Relaxed);
        (a == b).then_some(OutputConsumption { epoch, end_frame })
    }
    pub fn finish(&self) {
        self.queue.ended.store(true, Ordering::Release);
    }
    pub fn underrun_frames(&self) -> u64 {
        self.queue.underruns.load(Ordering::Relaxed)
    }
}
impl Consumer {
    pub fn keep_alive(&self) -> QueueLifetime {
        QueueLifetime {
            _queue: self.queue.clone(),
        }
    }
    /// O(1) boundary handoff, including while paused. Never scans a stale backlog.
    pub fn sync_epoch(&mut self) -> bool {
        let a = self.queue.generation.load(Ordering::Acquire);
        if a & 1 != 0 {
            return false;
        }
        let epoch = Epoch(self.queue.epoch.load(Ordering::Relaxed));
        let boundary = self.queue.boundary.load(Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Acquire);
        let b = self.queue.generation.load(Ordering::Relaxed);
        if a != b || epoch == self.epoch {
            return false;
        }
        self.queue.head.store(boundary, Ordering::Release);
        self.epoch = epoch;
        self.read_frames = 0;
        self.consumed_frames = 0;
        self.publish_receipt();
        true
    }
    /// Copy available frames; the host acknowledges only what its output voice consumes.
    pub fn read(&mut self, out: &mut [f32]) -> usize {
        let generation = self.queue.generation.load(Ordering::Acquire);
        if generation & 1 != 0 || self.queue.epoch.load(Ordering::Relaxed) != self.epoch.0 {
            out.fill(0.0);
            return 0;
        }
        let head = self.queue.head.load(Ordering::Relaxed);
        let tail = self.queue.tail.load(Ordering::Acquire);
        let n = (out.len() / self.queue.channels).min((tail - head) as usize);
        for (i, frame) in out[..n * self.queue.channels]
            .chunks_exact_mut(self.queue.channels)
            .enumerate()
        {
            let at = ((head + i as u64) % self.queue.capacity) as usize * self.queue.channels;
            for (ch, x) in frame.iter_mut().enumerate() {
                *x = f32::from_bits(self.queue.samples[at + ch].load(Ordering::Relaxed));
            }
        }
        out[n * self.queue.channels..].fill(0.0);
        // Publication may race the copy. Commit a read only if its complete
        // sample span belonged to the epoch observed before the copy. The
        // producer cannot reuse these slots until head advances below.
        std::sync::atomic::fence(Ordering::Acquire);
        if self.queue.generation.load(Ordering::Relaxed) != generation {
            out.fill(0.0);
            return 0;
        }
        self.queue.head.store(head + n as u64, Ordering::Release);
        self.read_frames += n as u64;
        if !self.queue.ended.load(Ordering::Acquire) {
            self.queue.underruns.fetch_add(
                (out.len() / self.queue.channels - n) as u64,
                Ordering::Relaxed,
            );
        }
        n
    }
    pub fn acknowledge_frames(&mut self, count: usize) {
        self.consumed_frames = (self.consumed_frames + count as u64).min(self.read_frames);
        self.publish_receipt();
    }
    fn publish_receipt(&self) {
        self.queue.receipt_generation.fetch_add(1, Ordering::AcqRel);
        self.queue
            .receipt_epoch
            .store(self.epoch.0, Ordering::Relaxed);
        self.queue
            .receipt_end
            .store(self.consumed_frames, Ordering::Relaxed);
        self.queue
            .receipt_generation
            .fetch_add(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stale_epoch_storage_never_counts_as_primed_media() {
        let (mut producer, mut consumer) = channel(2048);
        assert!(producer.push(
            &[1.0; 2048],
            OutputStamp {
                epoch: Epoch(1),
                start_frame: 0,
                frames: 1024,
            }
        ));
        producer.publish_epoch(Epoch(2));
        assert_eq!(producer.occupied_frames(), 0);
        assert_eq!(producer.free_frames(), 1024);
        assert!(producer.push(
            &[2.0; 1024],
            OutputStamp {
                epoch: Epoch(2),
                start_frame: 0,
                frames: 512,
            }
        ));
        assert_eq!(producer.occupied_frames(), 512);
        assert_eq!(producer.free_frames(), 512);
        assert!(consumer.sync_epoch());
        assert_eq!(producer.occupied_frames(), 512);
        assert_eq!(producer.free_frames(), 1536);
        let mut samples = [0.0; 1024];
        assert_eq!(consumer.read(&mut samples), 512);
        assert_eq!(samples, [2.0; 1024]);
    }
    #[test]
    fn concurrent_epoch_publication_never_mixes_or_duplicates_frames() {
        let (mut producer, mut consumer) = channel(2048);
        let done = Arc::new(AtomicBool::new(false));
        let end = done.clone();
        let writer = std::thread::spawn(move || {
            for epoch in 2..2002 {
                producer.publish_epoch(Epoch(epoch));
                for start in (0..512).step_by(32) {
                    let samples: Vec<_> = (start..start + 32)
                        .flat_map(|i| [epoch as f32, i as f32])
                        .collect();
                    let stamp = OutputStamp {
                        epoch: Epoch(epoch),
                        start_frame: start,
                        frames: 32,
                    };
                    while !producer.push(&samples, stamp) {
                        std::thread::yield_now();
                    }
                    std::thread::yield_now();
                }
            }
            end.store(true, Ordering::Release);
        });
        let mut samples = [0.0; 126];
        let mut expected = 0;
        while !done.load(Ordering::Acquire)
            || consumer.queue.head.load(Ordering::Acquire)
                < consumer.queue.tail.load(Ordering::Acquire)
        {
            if consumer.sync_epoch() {
                expected = 0;
            }
            let n = consumer.read(&mut samples);
            for frame in samples[..n * 2].chunks_exact(2) {
                assert_eq!(frame[0], consumer.epoch.0 as f32);
                assert_eq!(frame[1], expected as f32);
                expected += 1;
            }
            consumer.acknowledge_frames(n);
            std::thread::yield_now();
        }
        writer.join().unwrap();
    }
    #[test]
    fn partial_consumption_and_epoch_handoff() {
        let (mut producer, mut consumer) = channel(2048);
        assert!(producer.push(
            &[1.0; 200],
            OutputStamp {
                epoch: Epoch(1),
                start_frame: 0,
                frames: 100
            }
        ));
        let mut out = [0.0; 80];
        assert_eq!(consumer.read(&mut out), 40);
        consumer.acknowledge_frames(17);
        assert_eq!(producer.receipt().unwrap().end_frame, 17);
        producer.publish_epoch(Epoch(2));
        assert!(consumer.sync_epoch());
        assert_eq!(producer.occupied_frames(), 0);
        assert_eq!(
            producer.receipt().unwrap(),
            OutputConsumption {
                epoch: Epoch(2),
                end_frame: 0
            }
        );
        assert!(producer.push(
            &[2.0; 80],
            OutputStamp {
                epoch: Epoch(2),
                start_frame: 0,
                frames: 40
            }
        ));
        consumer.read(&mut out);
        assert_eq!(out, [2.0; 80]);
    }
    #[test]
    fn concurrent_wrap_preserves_samples_and_cumulative_receipts() {
        let (mut producer, mut consumer) = channel(2048);
        let writer = std::thread::spawn(move || {
            for block in 0..2000 {
                while producer.free_frames() < 37 {
                    std::thread::yield_now();
                }
                let samples = [block as f32; 74];
                assert!(producer.push(
                    &samples,
                    OutputStamp {
                        epoch: Epoch(1),
                        start_frame: block * 37,
                        frames: 37
                    }
                ));
            }
            producer
        });
        let mut frames = 0;
        let mut out = [0.0; 26];
        while frames < 74000 {
            let n = consumer.read(&mut out);
            for i in 0..n {
                assert_eq!(out[i * 2], ((frames + i) / 37) as f32);
            }
            frames += n;
            consumer.acknowledge_frames(n);
            if n == 0 {
                std::thread::yield_now();
            }
        }
        let producer = writer.join().unwrap();
        assert_eq!(producer.receipt().unwrap().end_frame, 74000);
    }
}

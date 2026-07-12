//! Fixed per-channel delay line.
//!
//! The keylock chain routes the (un-keylocked) low band through this pure
//! delay, matched sample-exactly to the high-band corrector's constant
//! latency, so the two bands re-sum in time.

/// Per-channel fixed circular delay.
#[derive(Debug)]
pub(crate) struct FixedDelay {
    /// One circular buffer per channel, each `delay_frames` long.
    lines: Vec<Vec<f32>>,
    /// Shared write cursor (all channels advance in lockstep).
    pos: usize,
    delay_frames: usize,
}

impl FixedDelay {
    pub(crate) fn new(delay_frames: usize, channels: usize) -> Self {
        Self {
            lines: (0..channels)
                .map(|_| vec![0.0; delay_frames.max(1)])
                .collect(),
            pos: 0,
            delay_frames,
        }
    }

    /// Delays one channel's samples in place by `delay_frames`.
    ///
    /// Channels must be processed in order 0..channels every block: the
    /// shared cursor advances when the last channel is processed.
    pub(crate) fn process_channel(&mut self, ch: usize, io: &mut [f32]) {
        if self.delay_frames == 0 {
            return;
        }
        let line = &mut self.lines[ch];
        let len = line.len();
        let mut pos = self.pos;
        for sample in io.iter_mut() {
            std::mem::swap(&mut line[pos], sample);
            pos += 1;
            if pos == len {
                pos = 0;
            }
        }
        if ch + 1 == self.lines.len() {
            self.pos = pos;
        }
    }

    pub(crate) fn latency_frames(&self) -> usize {
        self.delay_frames
    }

    pub(crate) fn reset(&mut self) {
        for line in &mut self.lines {
            line.fill(0.0);
        }
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delays_by_exact_frame_count() {
        let mut delay = FixedDelay::new(5, 1);
        let mut first: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        delay.process_channel(0, &mut first);
        assert_eq!(first, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        let mut second: Vec<f32> = (9..=12).map(|i| i as f32).collect();
        delay.process_channel(0, &mut second);
        assert_eq!(second, [4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn channels_delay_independently_in_lockstep() {
        let mut delay = FixedDelay::new(2, 2);
        let mut left = vec![1.0, 2.0, 3.0];
        let mut right = vec![-1.0, -2.0, -3.0];
        delay.process_channel(0, &mut left);
        delay.process_channel(1, &mut right);
        assert_eq!(left, [0.0, 0.0, 1.0]);
        assert_eq!(right, [0.0, 0.0, -1.0]);
        let mut left2 = vec![4.0];
        let mut right2 = vec![-4.0];
        delay.process_channel(0, &mut left2);
        delay.process_channel(1, &mut right2);
        assert_eq!(left2, [2.0]);
        assert_eq!(right2, [-2.0]);
    }

    #[test]
    fn zero_delay_is_identity() {
        let mut delay = FixedDelay::new(0, 1);
        let mut io = vec![1.0, 2.0];
        delay.process_channel(0, &mut io);
        assert_eq!(io, [1.0, 2.0]);
    }
}

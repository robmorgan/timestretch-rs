//! Literal reference-bin selection; sentinel storage is owned, not borrowed.
pub(super) struct Selector {
    pub references: Vec<usize>,
    pub map: Vec<usize>,
    pub work: Vec<i32>,
    pub count: usize,
}
impl Selector {
    pub fn new(n: usize) -> Self {
        Self {
            references: vec![0; n],
            map: vec![0; n],
            work: vec![0; n],
            count: 0,
        }
    }
    pub fn reset(&mut self) {
        self.references.fill(0);
        self.map.fill(0);
        self.work.fill(0);
        self.count = 0;
    }
    pub fn process(
        &mut self,
        hop: usize,
        limit: usize,
        infinite: bool,
        phase: &[f32],
        magnitude: &[f32],
        saved: &[i32],
    ) {
        let n = phase.len();
        let low = (n >> 8).max(1);
        let half = n / 2;
        self.work[0] = 1;
        self.count = 0;
        if infinite {
            for k in 0..limit {
                self.references[k] = k;
                self.map[k] = k;
            }
            self.count = limit;
            return;
        }
        let scale = n as f32 / (hop as f32 * std::f32::consts::TAU);
        for (k, &value) in phase.iter().enumerate().take(limit) {
            let mapped = value.mul_add(scale, 0.5).floor() as i32;
            self.map[k] = if mapped <= 0 {
                k
            } else {
                (mapped as usize).min(half - 1)
            };
        }
        let mut count = 0;
        for k in 1..low {
            self.references[count] = k;
            count += 1;
        }
        for (k, &state) in saved.iter().enumerate().take(2 * low).skip(low) {
            self.work[k] = self.map[k] as i32 - k as i32;
            if (self.work[k] <= 0 && self.work[k - 1] >= 0) || (state & !1) == 2 {
                self.references[count] = k;
                count += 1;
            }
        }
        for (k, &state) in saved.iter().enumerate().take(limit).skip(2 * low) {
            self.work[k] = self.map[k] as i32 - k as i32;
            if (self.work[k] <= 0 && self.work[k - 1] > 0) || (state & !1) == 2 {
                self.references[count] = k;
                count += 1;
            }
        }
        if count > 0 {
            let mut cursor = 0;
            if self.references[0] < (n >> 6) {
                loop {
                    self.work[self.count] = self.references[cursor] as i32;
                    self.count += 1;
                    cursor += 1;
                    if !(self.references[cursor] < (n >> 6) && cursor < count) {
                        break;
                    }
                }
            }
            let first = if self.count == 0 { 1 } else { cursor };
            if first == 1 {
                if magnitude[self.references[2]] <= magnitude[self.references[1]]
                    || magnitude[self.references[1]] < magnitude[self.references[0]]
                {
                    self.work[self.count] = self.references[1] as i32;
                    self.count += 1;
                }
                cursor = 2;
            }
            let mut candidate = self.references[cursor];
            let mut boundary = candidate.max(1) as i32;
            loop {
                let k = boundary as usize;
                boundary -= 1;
                if !(self.map[k].abs_diff(candidate)
                    < self.map[k].abs_diff(self.references[cursor - 1])
                    && k != 0)
                {
                    break;
                }
            }
            let mut left = (boundary + 1) as usize;
            while candidate < (n >> 3) && cursor < count {
                let peak = candidate.max(1);
                let next = self.references[cursor + 1];
                let mut probe = peak;
                let mut stop;
                loop {
                    stop = probe;
                    if !(self.map[probe].abs_diff(candidate) < self.map[probe].abs_diff(next)
                        && probe < half - 1)
                    {
                        break;
                    }
                    stop = probe + 1;
                    let again = probe < limit - 2;
                    probe = stop;
                    if !again {
                        break;
                    }
                }
                let right = (stop - 1).min(limit - 2);
                if magnitude[left + 1] <= magnitude[peak] && magnitude[right] <= magnitude[peak] {
                    self.work[self.count] = peak as i32;
                    self.count += 1;
                }
                cursor += 1;
                candidate = self.references[cursor];
                left = right;
            }
            cursor = cursor.max(1);
            candidate = self.references[cursor];
            while candidate < limit && cursor < count {
                // The second history lookup really is the list index.
                if (magnitude[self.references[cursor + 1]] <= magnitude[candidate]
                    && magnitude[self.references[cursor - 1]] <= magnitude[candidate])
                    || saved[candidate] == 2
                    || saved[cursor] == 3
                {
                    self.work[self.count] = candidate as i32;
                    self.count += 1;
                }
                cursor += 1;
                candidate = self.references[cursor];
            }
            self.work[self.count.max(1) - 1] = (limit - 1) as i32;
            for k in 0..self.count {
                self.references[k] = self.work[k] as usize;
            }
        }
        if self.count < 1 {
            self.count = count.max(1);
            self.references[self.count - 1] = limit - 1;
        }
        let mut search = 1;
        for k in 0..limit {
            for r in search..=self.count {
                let prior = self.references[r - 1];
                let next = self.references[r];
                if self.map[k].abs_diff(prior) <= self.map[k].abs_diff(next)
                    || next == 0
                    || r == self.count
                {
                    self.map[k] = prior;
                    search = r.max(4) - 3;
                    break;
                }
            }
        }
    }
}

use crate::memory::HeapBytes;
impl HeapBytes for Selector {
    fn heap_bytes(&self) -> usize {
        self.references.heap_bytes() + self.map.heap_bytes() + self.work.heap_bytes()
    }
}

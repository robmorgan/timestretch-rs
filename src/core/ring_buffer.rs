//! Fixed-capacity ring buffer for real-time audio paths.

/// Fixed-capacity ring buffer.
///
/// The buffer never allocates after construction and never shifts memory.
/// All operations are O(n) in the number of elements copied, with deterministic
/// upper bounds.
#[derive(Debug, Clone)]
pub struct RingBuffer<T>
where
    T: Copy + Default,
{
    data: Vec<T>,
    head: usize,
    tail: usize,
    len: usize,
}

impl<T> RingBuffer<T>
where
    T: Copy + Default,
{
    /// Creates a ring buffer with fixed capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: vec![T::default(); cap],
            head: 0,
            tail: 0,
            len: 0,
        }
    }

    /// Returns the number of elements currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the fixed capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Returns available free space.
    #[inline]
    pub fn available(&self) -> usize {
        self.capacity().saturating_sub(self.len)
    }

    /// Returns true when no elements are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clears the ring buffer.
    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.len = 0;
    }

    /// Pushes one element. Returns `false` if the buffer is full.
    #[inline]
    pub fn push(&mut self, value: T) -> bool {
        if self.len == self.capacity() || self.capacity() == 0 {
            return false;
        }
        self.data[self.tail] = value;
        self.tail = (self.tail + 1) % self.capacity();
        self.len += 1;
        true
    }

    /// Pops one element from the front.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 || self.capacity() == 0 {
            return None;
        }
        let value = self.data[self.head];
        self.head = (self.head + 1) % self.capacity();
        self.len -= 1;
        if self.len == 0 {
            self.head = 0;
            self.tail = 0;
        }
        Some(value)
    }

    /// Discards up to `n` elements from the front.
    ///
    /// Returns the number of elements discarded.
    pub fn discard(&mut self, n: usize) -> usize {
        let to_drop = n.min(self.len);
        if to_drop == 0 || self.capacity() == 0 {
            return 0;
        }
        self.head = (self.head + to_drop) % self.capacity();
        self.len -= to_drop;
        if self.len == 0 {
            self.head = 0;
            self.tail = 0;
        }
        to_drop
    }

    /// Copies elements from the front into `out` without removing them.
    ///
    /// Returns the number of copied elements.
    pub fn peek_slice(&self, out: &mut [T]) -> usize {
        let to_copy = out.len().min(self.len);
        if to_copy == 0 || self.capacity() == 0 {
            return 0;
        }

        let first = to_copy.min(self.capacity() - self.head);
        out[..first].copy_from_slice(&self.data[self.head..self.head + first]);
        let second = to_copy - first;
        if second > 0 {
            out[first..first + second].copy_from_slice(&self.data[..second]);
        }
        to_copy
    }

    /// Pushes as many items as fit from `input`.
    ///
    /// Returns the number of items pushed.
    pub fn push_slice(&mut self, input: &[T]) -> usize {
        if input.is_empty() || self.capacity() == 0 || self.available() == 0 {
            return 0;
        }
        let to_push = input.len().min(self.available());
        let first = to_push.min(self.capacity() - self.tail);
        self.data[self.tail..self.tail + first].copy_from_slice(&input[..first]);
        self.tail = (self.tail + first) % self.capacity();

        let second = to_push - first;
        if second > 0 {
            self.data[..second].copy_from_slice(&input[first..first + second]);
            self.tail = second;
        }

        self.len += to_push;
        to_push
    }

    /// Pops as many items as available into `output`.
    ///
    /// Returns the number of items popped.
    pub fn pop_slice(&mut self, output: &mut [T]) -> usize {
        if output.is_empty() || self.capacity() == 0 || self.len == 0 {
            return 0;
        }
        let to_pop = output.len().min(self.len);
        let first = to_pop.min(self.capacity() - self.head);
        output[..first].copy_from_slice(&self.data[self.head..self.head + first]);
        self.head = (self.head + first) % self.capacity();

        let second = to_pop - first;
        if second > 0 {
            output[first..first + second].copy_from_slice(&self.data[..second]);
            self.head = second;
        }

        self.len -= to_pop;
        if self.len == 0 {
            self.head = 0;
            self.tail = 0;
        }
        to_pop
    }
}

#[cfg(test)]
mod tests {
    use super::RingBuffer;

    #[test]
    fn push_pop_wrap() {
        let mut rb = RingBuffer::<i32>::with_capacity(4);
        assert_eq!(rb.push_slice(&[1, 2, 3]), 3);
        let mut out = [0; 2];
        assert_eq!(rb.pop_slice(&mut out), 2);
        assert_eq!(out, [1, 2]);
        assert_eq!(rb.push_slice(&[4, 5, 6]), 3);
        let mut out2 = [0; 4];
        assert_eq!(rb.pop_slice(&mut out2), 4);
        assert_eq!(out2, [3, 4, 5, 6]);
    }

    #[test]
    fn bounded_capacity() {
        let mut rb = RingBuffer::<f32>::with_capacity(2);
        assert_eq!(rb.push_slice(&[1.0, 2.0, 3.0]), 2);
        assert_eq!(rb.len(), 2);
        assert_eq!(rb.available(), 0);
    }
}

use crate::StretchError;

/// Pitch-mark record. `estimated_period` is metadata; synthesis uses
/// differences between adjacent absolute `position` values.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PitchMark {
    /// Absolute source sample position.
    pub position: i64,
    /// Mean squared energy.
    pub energy: f32,
    /// Transient strength.
    pub transient: f32,
    /// Tonality metadata, not a calibrated confidence.
    pub tonality: f32,
    /// Onset metadata.
    pub onset: f32,
    /// Analysis metadata, not the period consumed by synthesis.
    pub estimated_period: f32,
}

/// Fixed-capacity mark list. Forward search retains the earlier mark on ties.
pub struct PitchMarks {
    records: Vec<PitchMark>,
    limit: usize,
}

impl PitchMarks {
    /// Allocate storage and seed a mark at position zero.
    pub fn new(capacity: usize) -> Result<Self, StretchError> {
        super::check_capacity(capacity)?;
        let mut result = Self {
            records: Vec::with_capacity(capacity),
            limit: capacity,
        };
        result.reset();
        Ok(result)
    }
    /// Forget marks and restore the initial zero mark without allocating.
    pub fn reset(&mut self) {
        self.records.clear();
        self.records.push(PitchMark::default());
    }
    /// Current records.
    pub fn records(&self) -> &[PitchMark] {
        &self.records
    }
    /// Remaining preallocated slots.
    pub fn remaining_capacity(&self) -> usize {
        self.limit - self.records.len()
    }
    /// Append a mark. Equal positions are permitted.
    pub fn push(&mut self, mark: PitchMark) -> Result<(), StretchError> {
        if self.records.len() == self.limit {
            return Err(StretchError::BufferOverflow {
                buffer: "pitch marks",
                requested: self.records.len() + 1,
                available: self.limit,
            });
        }
        if mark.position < self.last_position()
            || [
                mark.energy,
                mark.transient,
                mark.tonality,
                mark.onset,
                mark.estimated_period,
            ]
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err(super::invalid(
                "mark positions must be ordered and metadata finite",
            ));
        }
        self.records.push(mark);
        Ok(())
    }
    /// Keep the last `keep` records; caller must adjust synthesis indices too.
    pub fn retain_last(&mut self, keep: usize) -> Result<usize, StretchError> {
        if keep == 0 || keep > self.records.len() {
            return Err(super::invalid("invalid mark retention count"));
        }
        let discarded = self.records.len() - keep;
        self.records.copy_within(discarded.., 0);
        self.records.truncate(keep);
        Ok(discarded)
    }
    pub(crate) fn last_position(&self) -> i64 {
        self.records.last().map_or(-1, |r| r.position)
    }
    pub(crate) fn position(&self, index: usize) -> Option<i64> {
        self.records.get(index).map(|m| m.position)
    }
    pub(crate) fn period(&self, index: usize) -> Option<i32> {
        self.records
            .get(index + 1)
            .zip(self.records.get(index))
            .and_then(|(a, b)| i32::try_from(a.position - b.position).ok())
    }
    pub(crate) fn closest(&self, start: usize, position: i64) -> Option<usize> {
        if self.records.len() < 2 || start >= self.records.len() - 1 {
            return None;
        }
        let mut best = start;
        while best < self.records.len() - 1
            && self.records[best + 1].position.abs_diff(position)
                < self.records[best].position.abs_diff(position)
        {
            best += 1;
        }
        Some(best)
    }
    pub(crate) fn transient(&self, index: usize) -> f32 {
        self.records.get(index).map_or(-1.0, |r| r.transient)
    }
}

use crate::memory::HeapBytes;
impl HeapBytes for PitchMarks {
    fn heap_bytes(&self) -> usize {
        self.records.heap_bytes()
    }
}
impl HeapBytes for PitchMark {
    fn heap_bytes(&self) -> usize {
        0
    }
}

use super::PitchMarks;
use crate::StretchError;

/// Observable state of ordinary grain synthesis.
#[derive(Clone, Debug)]
pub struct SynthesisState {
    /// Input-time cursor, advanced with fused double arithmetic.
    pub input_time: f64,
    /// Buffered logical input samples.
    pub buffered_input: i32,
    /// Total samples received since reset.
    pub total_input: i64,
    /// Next grain's output center relative to the pending output block.
    pub next_center: i32,
    /// Finished output boundary.
    pub ready_boundary: i32,
    /// Previous output period and last input interval.
    pub previous_output_period: i32,
    /// Stored interval used by lookahead, not its simulated local period.
    pub input_period: i32,
    /// Last selected absolute mark position.
    pub selected_mark_position: i64,
    /// Selected index, retained even when a grain cannot yet be emitted.
    pub selected_mark_index: i32,
    /// Last selected mark index.
    pub last_mark_index: usize,
    /// Consecutive repeated selections.
    pub repeat_count: i32,
    /// Current lookup-table index scale.
    pub window_index_scale: f32,
}
impl Default for SynthesisState {
    fn default() -> Self {
        Self {
            input_time: 0.0,
            buffered_input: 0,
            total_input: 0,
            next_center: 0,
            ready_boundary: 0,
            previous_output_period: 0,
            input_period: 0,
            selected_mark_position: 0,
            selected_mark_index: 0,
            last_mark_index: 0,
            repeat_count: 1,
            window_index_scale: 1.0,
        }
    }
}

/// Ordinary grain synthesis: pitch fixed at unity, no transient-copy or sinc
/// branches. Signed logical indices are translated into owned padded storage.
pub struct GrainSynthesis {
    channels: usize,
    capacity: usize,
    pad: usize,
    requested: usize,
    input: Vec<Vec<f32>>,
    output: Vec<Vec<f32>>,
    table: Vec<f32>,
    stretch: f32,
    inv_stretch: f32,
    max_inv_stretch: f32,
    state: SynthesisState,
}
impl GrainSynthesis {
    /// Allocate fixed input/output buffers with `leading_pad` logical history.
    pub fn new(channels: usize, capacity: usize, leading_pad: usize) -> Result<Self, StretchError> {
        super::check_channels(channels)?;
        super::check_capacity(capacity)?;
        if leading_pad < 2 || leading_pad > capacity {
            return Err(super::invalid("invalid grain leading padding"));
        }
        let table = super::tables::flank();
        Ok(Self {
            channels,
            capacity,
            pad: leading_pad,
            requested: 0,
            input: vec![vec![0.0; capacity + leading_pad]; channels],
            output: vec![vec![0.0; capacity + leading_pad]; channels],
            table,
            stretch: 1.0,
            inv_stretch: 1.0,
            max_inv_stretch: 1.0,
            state: SynthesisState::default(),
        })
    }
    /// Current arithmetic/buffer state.
    pub fn state(&self) -> &SynthesisState {
        &self.state
    }
    /// Remaining input capacity before more output must be rendered.
    pub fn available_input_frames(&self) -> usize {
        self.capacity - self.state.buffered_input as usize
    }
    /// Captured vector-provider single-precision sin-cubed flank table.
    pub fn flank_table(&self) -> &[f32] {
        &self.table
    }
    /// Supply a validated 16,384-entry table to isolate grain arithmetic from
    /// platform sine-provider rounding. Copies once during construction.
    pub fn with_flank_table(
        channels: usize,
        capacity: usize,
        leading_pad: usize,
        table: &[f32],
    ) -> Result<Self, StretchError> {
        if table.len() != 16384
            || table
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(super::invalid("invalid ordinary flank table"));
        }
        let mut result = Self::new(channels, capacity, leading_pad)?;
        result.table.copy_from_slice(table);
        Ok(result)
    }
    pub(super) fn clear_storage(&mut self) {
        for a in self.input.iter_mut().chain(&mut self.output) {
            a.fill(0.);
        }
        self.requested = 0;
        self.state.window_index_scale = 1.;
    }
    /// Reset counters and stretch factors while retaining sample storage and
    /// the window-index scale.
    pub fn reset(&mut self) {
        let scale = self.state.window_index_scale;
        self.state = SynthesisState::default();
        self.state.window_index_scale = scale;
        self.stretch = 1.0;
        self.inv_stretch = 1.0;
        self.max_inv_stretch = 1.0;
    }
    /// Set the stretch duration factor without changing synthesis pitch.
    pub fn set_stretch(&mut self, stretch: f32) -> Result<(), StretchError> {
        if !stretch.is_finite() || !(0.1..=64.0).contains(&stretch) {
            return Err(StretchError::InvalidRatio("grain stretch must be 0.1..=64"));
        }
        self.stretch = stretch;
        self.inv_stretch = 1.0 / stretch;
        self.max_inv_stretch = self.inv_stretch.max(1.0);
        Ok(())
    }
    /// Synchronize the input cursor using the next-center offset scaled by the
    /// inverse stretch factor.
    pub fn set_play_time(&mut self, time: f64) -> Result<(), StretchError> {
        let next = f64::from(self.inv_stretch * self.state.next_center as f32) + time;
        if !next.is_finite() || next < -(1_i64 << 48) as f64 || next > (1_i64 << 48) as f64 {
            return Err(super::invalid(
                "grain source time is outside the supported range",
            ));
        }
        if time < self.state.input_time {
            self.state.last_mark_index = 0;
        }
        self.state.input_time = next;
        Ok(())
    }
    /// Adjust indices after the caller discards old records from the mark list.
    pub fn discard_mark_prefix(&mut self, count: usize) -> Result<(), StretchError> {
        self.state.selected_mark_index = self
            .state
            .selected_mark_index
            .checked_sub(
                i32::try_from(count).map_err(|_| super::invalid("mark prefix exceeds i32"))?,
            )
            .ok_or_else(|| super::invalid("selected mark index overflow"))?;
        self.state.last_mark_index = self
            .state
            .last_mark_index
            .checked_sub(count)
            .ok_or_else(|| super::invalid("cannot discard marks still referenced by synthesis"))?;
        Ok(())
    }
    /// Set requested output length and simulate required source lookahead.
    /// `None` means more marks are needed. The simulated time advance uses the
    /// stored `input_period`, not the simulated local period.
    pub fn frames_needed(
        &mut self,
        frames: usize,
        marks: &PitchMarks,
    ) -> Result<Option<usize>, StretchError> {
        if frames > self.capacity {
            return Err(super::invalid("grain output exceeds capacity"));
        }
        self.requested = frames;
        if self.state.ready_boundary >= frames as i32 {
            return Ok(Some(0));
        }
        let mut last = self.state.last_mark_index;
        let mut advance = self.state.next_center - self.state.ready_boundary;
        let mut ready = self.state.ready_boundary;
        let mut time = self.state.input_time;
        let mut required = 0i64;
        let mut steps = 0;
        while ready < frames as i32 {
            steps += 1;
            if steps > self.capacity + 2 {
                return Err(super::invalid("grain lookahead failed to advance"));
            }
            let Some(mut index) = marks.closest(last, time as i64) else {
                return Ok(None);
            };
            if index == last && marks.transient(index) > 0.7 {
                index += 1;
            }
            let Some(period) = marks.period(index) else {
                return Ok(None);
            };
            let Some(mark) = marks.position(index) else {
                return Ok(None);
            };
            required = required.max(
                mark - self.state.total_input
                    + i64::from((period as f32 * self.max_inv_stretch) as i32),
            );
            let output_advance = advance;
            advance = period.max(2);
            time += f64::from((self.inv_stretch * self.state.input_period as f32).floor());
            ready = ready
                .checked_add(output_advance)
                .ok_or_else(|| super::invalid("grain lookahead overflow"))?;
            last = index;
        }
        Ok(Some(required.max(0) as usize))
    }
    /// Append input without rendering. Uses the same retained input storage as
    /// `process`, making a feed/render host independent of file/callback pacing.
    pub fn push(&mut self, input: &[&[f32]]) -> Result<(), StretchError> {
        let frames = super::input_frames(input, self.channels)?;
        let available = self.capacity - self.state.buffered_input as usize;
        if frames > available {
            return Err(StretchError::BufferOverflow {
                buffer: "grain synthesis",
                requested: frames,
                available,
            });
        }
        let total = self
            .state
            .total_input
            .checked_add(frames as i64)
            .ok_or_else(|| super::invalid("grain absolute input position overflow"))?;
        let start = self.pad + self.state.buffered_input as usize;
        for (ch, source) in input.iter().enumerate() {
            self.input[ch][start..start + frames].copy_from_slice(source);
        }
        self.state.buffered_input += frames as i32;
        self.state.total_input = total;
        Ok(())
    }
    /// Render the requested frames into caller-owned buffers. `false` means
    /// source/mark lookahead is unavailable; successful calls allocate nothing.
    pub fn process(
        &mut self,
        output: &mut [&mut [f32]],
        marks: &PitchMarks,
    ) -> Result<bool, StretchError> {
        #[cfg(feature = "qa")]
        let _profile = crate::processor::qa::region(3);
        super::output_capacity(output, self.channels, self.requested)?;
        if self.state.buffered_input <= 0 && self.state.ready_boundary < self.requested as i32 {
            return Ok(false);
        }
        let mut steps = 0;
        if self.state.buffered_input > 0 {
            while self.state.ready_boundary < self.requested as i32 {
                steps += 1;
                if steps > self.capacity + 2 {
                    return Err(super::invalid("grain synthesis failed to advance"));
                }
                let index =
                    match marks.closest(self.state.last_mark_index, self.state.input_time as i64) {
                        Some(index) => index,
                        None if self.state.repeat_count == 1 => self.state.last_mark_index,
                        None => {
                            self.state.selected_mark_index = -1;
                            return Ok(false);
                        }
                    };
                self.state.selected_mark_index = index as i32;
                let period = marks
                    .period(index)
                    .unwrap_or(self.state.previous_output_period);
                let Some(mark) = marks.position(index) else {
                    return Ok(false);
                };
                if period < 0 {
                    return Ok(false);
                }
                self.grain(self.state.previous_output_period, period, mark)?;
                self.state.input_period = period;
                self.state.selected_mark_position = mark;
                self.state.repeat_count = if index == self.state.last_mark_index {
                    self.state
                        .repeat_count
                        .checked_add(1)
                        .ok_or_else(|| super::invalid("grain repetition overflow"))?
                } else {
                    1
                };
                self.state.last_mark_index = index;
                self.state.previous_output_period = period;
                self.state.input_time =
                    f64::from(period).mul_add(f64::from(self.inv_stretch), self.state.input_time);
            }
            let discard = i64::from(self.state.buffered_input)
                - self.state.total_input
                - (i64::from(self.state.previous_output_period)
                    * i64::from(self.state.repeat_count)
                    + self.pad as i64)
                + self.state.selected_mark_position;
            if discard > 0 {
                let discard = discard as usize;
                if discard > self.state.buffered_input as usize {
                    return Err(super::invalid("grain discard exceeds buffered input"));
                }
                self.state.buffered_input -= discard as i32;
                for ch in &mut self.input {
                    ch.copy_within(
                        self.pad + discard..self.pad + discard + self.state.buffered_input as usize,
                        self.pad,
                    );
                }
            }
        }
        self.state.next_center -= self.requested as i32;
        self.state.ready_boundary -= self.requested as i32;
        if self.state.next_center < 0 {
            return Err(super::invalid("negative pending grain output"));
        }
        for (ch, destination) in output.iter_mut().enumerate() {
            destination[..self.requested]
                .copy_from_slice(&self.output[ch][self.pad..self.pad + self.requested]);
            let keep = self.state.next_center as usize;
            if self.requested + keep > self.capacity {
                return Err(super::invalid("pending grain output exceeds capacity"));
            }
            self.output[ch].copy_within(
                self.pad + self.requested..self.pad + self.requested + keep,
                self.pad,
            );
        }
        Ok(true)
    }
    fn grain(&mut self, left: i32, period: i32, mark: i64) -> Result<(), StretchError> {
        let center = self.state.next_center;
        let source = i64::from(self.state.buffered_input) + mark - self.state.total_input;
        let source_start = self.pad as i64 + source - i64::from(left);
        let destination_start = self.pad as i64 + i64::from(center) - i64::from(left);
        let source_center = self.pad as i64 + source;
        let output_center = self.pad as i64 + i64::from(center);
        let reserve = (period as f32).mul_add(1.0, 1.5).floor() as usize;
        let size = (self.pad + self.capacity) as i64;
        if source_start < 0
            || destination_start < 0
            || source_center + i64::from(period) > size
            || output_center + reserve as i64 > size
            || source_center + i64::from(period)
                > self.pad as i64 + i64::from(self.state.buffered_input)
        {
            return Err(super::invalid("grain requires unavailable padded samples"));
        }
        let left = left as usize;
        let period = period as usize;
        self.state.window_index_scale = if left < 2 {
            self.table.len() as f32
        } else {
            (self.table.len() - 1) as f32 / (left - 1) as f32
        };
        for ch in 0..self.channels {
            for j in 0..left {
                let index = (f64::from(self.state.window_index_scale * j as f32) + 0.5) as usize;
                let destination = destination_start as usize + j;
                self.output[ch][destination] = self.input[ch][source_start as usize + j]
                    .mul_add(self.table[index], self.output[ch][destination]);
            }
        }
        self.state.ready_boundary = center;
        self.state.window_index_scale = if period < 2 {
            self.table.len() as f32
        } else {
            (self.table.len() - 1) as f32 / (period - 1) as f32
        };
        for ch in 0..self.channels {
            for j in 0..period {
                let index = (f64::from(self.state.window_index_scale * j as f32) + 0.5) as usize;
                self.output[ch][output_center as usize + j] =
                    (1.0 - self.table[index]) * self.input[ch][source_center as usize + j];
            }
            self.output[ch][output_center as usize + period..output_center as usize + reserve]
                .fill(0.0);
        }
        self.state.next_center = center
            .checked_add(period as i32)
            .ok_or_else(|| super::invalid("grain center overflow"))?;
        Ok(())
    }
}

use crate::memory::HeapBytes;
impl HeapBytes for GrainSynthesis {
    fn heap_bytes(&self) -> usize {
        self.input.heap_bytes() + self.output.heap_bytes() + self.table.heap_bytes()
    }
}

#[cfg(test)]
mod long_stream_tests {
    use super::*;
    #[test]
    fn absolute_grain_coordinates_cross_32_bits_without_audio_changes() {
        let input: Vec<_> = (0..2048).map(|i| (i as f32 * 0.03).sin() * 0.2).collect();
        let run = |origin| {
            let mut synth = GrainSynthesis::new(1, 4096, 1024).unwrap();
            synth.state.total_input = origin;
            synth.push(&[&input]).unwrap();
            synth.grain(128, 128, origin + 1024).unwrap();
            synth.output[0].clone()
        };
        assert_eq!(run(0), run(i64::from(i32::MAX) - 512));
        assert_eq!(run(0), run(1_i64 << 40));
    }
}

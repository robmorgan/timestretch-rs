use super::{PitchMark, PitchMarks, fft::PackedFft};
use crate::StretchError;

/// Active waveform-correlation analysis counters and filter history.
#[derive(Clone, Debug)]
pub struct AnalysisState {
    /// Number of queued mono samples.
    pub buffered_input: usize,
    /// Absolute position of the last emitted mark.
    pub last_position: i64,
    /// Current interval and preceding interval.
    pub candidate_interval: i32,
    /// Previous accepted interval.
    pub previous_interval: i32,
    /// Correlation lag retained between analysis windows.
    pub predicted_lag: i32,
    /// Number of analysis iterations since reset.
    pub analysis_counter: i64,
    /// Previous FIR and IIR absolute maxima.
    pub previous_maxima: [f32; 2],
    /// Persistent FIR sample and IIR output histories.
    pub filter_history: [f32; 2],
    /// Current and prior transient suppression flags.
    pub transients: [bool; 2],
}

/// Pitch-mark analysis using local waveform correlation. Marks align waveform
/// segments for grain synthesis; no separate musical pitch or F0 estimator runs.
pub struct MarkAnalysis {
    channels: usize,
    base: usize,
    capacity: usize,
    correlation_window: usize,
    half: usize,
    radius: usize,
    mono: Vec<f32>,
    work: Vec<f32>,
    left: Vec<f32>,
    right: Vec<f32>,
    correlation: Vec<f32>,
    fft: PackedFft,
    state: AnalysisState,
}
impl MarkAnalysis {
    /// Allocate waveform-correlation geometry with FIR/IIR coefficients
    /// (-0.5/-0.9). Sample-rate-to-window selection belongs to the host.
    pub fn new(base_window: usize, channels: usize, capacity: usize) -> Result<Self, StretchError> {
        Self::new_with_numerics(base_window, channels, capacity, crate::Numerics::Portable)
    }
    pub fn new_with_numerics(
        base_window: usize,
        channels: usize,
        capacity: usize,
        numerics: crate::Numerics,
    ) -> Result<Self, StretchError> {
        if !numerics.available() {
            return Err(super::invalid("numerical backend unavailable"));
        }
        super::check_channels(channels)?;
        super::check_capacity(capacity)?;
        if !base_window.is_power_of_two()
            || !(256..=2048).contains(&base_window)
            || capacity < 2 * base_window
        {
            return Err(super::invalid("invalid analysis geometry"));
        }
        let mut analysis = Self {
            channels,
            base: base_window,
            capacity,
            correlation_window: base_window,
            half: base_window / 2,
            radius: base_window / 2,
            mono: vec![0.0; capacity],
            work: vec![0.0; capacity],
            left: vec![0.0; 2 * base_window],
            right: vec![0.0; 2 * base_window],
            correlation: vec![0.0; 2 * base_window],
            fft: PackedFft::new(2 * base_window, numerics),
            state: AnalysisState {
                buffered_input: 0,
                last_position: 0,
                candidate_interval: 0,
                previous_interval: base_window as i32 / 2,
                predicted_lag: 0,
                analysis_counter: 0,
                previous_maxima: [0.0; 2],
                filter_history: [0.0; 2],
                transients: [false; 2],
            },
        };
        analysis.reset();
        Ok(analysis)
    }
    /// Inspect counters and retained filter state.
    pub fn state(&self) -> &AnalysisState {
        &self.state
    }
    pub(super) fn clear_storage(&mut self) {
        for a in [
            &mut self.mono,
            &mut self.work,
            &mut self.left,
            &mut self.right,
            &mut self.correlation,
        ] {
            a.fill(0.);
        }
        self.state.filter_history = [0.; 2];
    }
    /// Reset counters and suppression while retaining sample/filter storage.
    /// Reset the shared mark list separately.
    pub fn reset(&mut self) {
        self.correlation_window = self.base;
        self.half = self.base / 2;
        self.radius = self.half;
        let history = self.state.filter_history;
        self.state = AnalysisState {
            buffered_input: 0,
            last_position: 0,
            candidate_interval: 0,
            previous_interval: self.half as i32,
            predicted_lag: 0,
            analysis_counter: 0,
            previous_maxima: [0.0; 2],
            filter_history: history,
            transients: [false; 2],
        };
    }
    /// Sum channels (without averaging), generate marks and retain incomplete
    /// windows. No allocation. Returns the number of newly appended marks.
    pub fn process(
        &mut self,
        input: &[&[f32]],
        marks: &mut PitchMarks,
    ) -> Result<usize, StretchError> {
        #[cfg(feature = "qa")]
        let _profile = crate::processor::qa::region(2);
        let count = super::input_frames(input, self.channels)?;
        let available = self.capacity - self.state.buffered_input;
        if count > available {
            return Err(StretchError::BufferOverflow {
                buffer: "mark analysis",
                requested: count,
                available,
            });
        }
        // Smaller candidates fall back to the preceding accepted interval.
        let total = self.state.buffered_input + count;
        let needed = if total >= 2 * self.base {
            (total - 2 * self.base) / (self.base / 16) + 1
        } else {
            0
        };
        if marks.remaining_capacity() < needed {
            return Err(StretchError::BufferOverflow {
                buffer: "pitch marks",
                requested: needed,
                available: marks.remaining_capacity(),
            });
        }
        for i in 0..count {
            let mut sum = 0.0;
            for ch in input {
                sum += ch[i];
            }
            self.mono[self.state.buffered_input + i] = sum;
        }
        self.state.buffered_input += count;
        let before = marks.records().len();
        while self.state.buffered_input >= 2 * self.base {
            self.state.analysis_counter = self
                .state
                .analysis_counter
                .checked_add(1)
                .ok_or_else(|| super::invalid("analysis counter overflow"))?;
            if self.state.buffered_input < self.correlation_window {
                self.state.analysis_counter -= 1;
                break;
            }
            let mut candidate = self
                .state
                .last_position
                .checked_add(self.half as i64)
                .ok_or_else(|| super::invalid("absolute mark position overflow"))?;
            let maximum = self.mono[..self.correlation_window]
                .iter()
                .map(|x| x.abs())
                .fold(-f32::MAX, f32::max);
            let mut lag = self.state.predicted_lag;
            if maximum < 1e-24 {
                lag = 0;
            } else {
                self.correlate();
                let center = self.correlation.len() as i32 / 2;
                let original_start = center + self.state.predicted_lag - self.radius as i32;
                let mut first = original_start;
                let mut last = center + self.state.predicted_lag + self.radius as i32;
                if last > center + self.half as i32 - 1 {
                    last = center + self.half as i32 - 2;
                    first = last - 2 * self.radius as i32;
                }
                if original_start < center - (self.half as i32 - 1) {
                    first = center - self.half as i32;
                    last = first + 2 * self.radius as i32;
                }
                if first < 0 || last >= self.correlation.len() as i32 {
                    return Err(super::invalid("correlation search exceeds its buffer"));
                }
                let mut best = 0.0;
                let mut best_index = None;
                for i in first..=last {
                    let score = (self.correlation[i as usize] * 10000.0).floor();
                    if score > best {
                        best = score;
                        best_index = Some(i);
                    }
                }
                if let Some(index) = best_index {
                    lag = index - center;
                }
            }
            self.state.predicted_lag = lag;
            candidate = candidate
                .checked_sub(i64::from(lag))
                .ok_or_else(|| super::invalid("absolute mark position overflow"))?;
            let mut interval = i32::try_from(candidate - self.state.last_position)
                .map_err(|_| super::invalid("pitch mark interval overflow"))?;
            if candidate - marks.last_position() < (self.base as i64 / 4) >> 2 {
                interval = self.state.previous_interval;
                candidate = self
                    .state
                    .last_position
                    .checked_add(i64::from(interval))
                    .ok_or_else(|| super::invalid("absolute mark position overflow"))?;
                if self.base > 1023 {
                    self.correlation_window = self.base;
                    self.half = self.base / 2;
                    self.radius = self.half;
                    self.state.predicted_lag = self.half as i32 - interval;
                }
            }
            self.state.candidate_interval = interval;
            if interval <= 0 {
                return Err(super::invalid(
                    "analysis produced a nonpositive mark interval",
                ));
            }
            if self.state.buffered_input < interval as usize {
                break;
            }
            let interval = interval as usize;
            let mut fir_maximum = -f32::MAX;
            for i in 0..interval {
                let value = 0.5f32.mul_add(self.mono[i], -0.5 * self.state.filter_history[0]);
                self.state.filter_history[0] = self.mono[i];
                self.work[i] = value;
                fir_maximum = fir_maximum.max(value.abs());
            }
            let suppressed = self.state.transients[0] || self.state.transients[1];
            let fir_rise = rise_score(fir_maximum, self.state.previous_maxima[0], 0.01, suppressed);
            self.state.previous_maxima[0] = fir_maximum;
            let mut iir_maximum = -f32::MAX;
            let feedforward = 1.0 - (-0.9f32).abs();
            for i in 0..interval {
                self.state.filter_history[1] =
                    feedforward.mul_add(self.mono[i], 0.9 * self.state.filter_history[1]) + 1e-24;
                self.work[i] = self.state.filter_history[1];
                iir_maximum = iir_maximum.max(self.work[i].abs());
            }
            let iir_rise = rise_score(iir_maximum, self.state.previous_maxima[1], 0.1, suppressed);
            self.state.previous_maxima[1] = iir_maximum;
            let transient = fir_rise.max(iir_rise);
            let tonality = if f64::from(iir_maximum) < 0.02 {
                0.0
            } else {
                12.0
            };
            let mut energy = 0.0;
            for &v in &self.mono[..interval] {
                energy = v.mul_add(v, energy);
            }
            energy /= interval as f32;
            self.state.buffered_input -= interval;
            self.mono
                .copy_within(interval..interval + self.state.buffered_input, 0);
            self.state.last_position = candidate;
            marks.push(PitchMark {
                position: candidate,
                energy,
                transient,
                tonality,
                onset: 0.0,
                estimated_period: (self.half as i32 - self.state.predicted_lag) as f32,
            })?;
            self.state.previous_interval = interval as i32;
            self.state.transients[1] = self.state.transients[0];
            self.state.transients[0] = transient > 0.7;
        }
        Ok(marks.records().len() - before)
    }
    fn correlate(&mut self) {
        let n = self.left.len();
        let h = n / 2;
        let count = self.correlation_window;
        self.left.fill(0.0);
        self.right.fill(0.0);
        self.left[..count].copy_from_slice(&self.mono[..count]);
        self.right[..count].copy_from_slice(&self.mono[self.half..self.half + count]);
        self.fft.forward(&mut self.left);
        self.fft.forward(&mut self.right);
        self.left[0] *= self.right[0];
        self.left[h] *= self.right[h];
        for k in 1..h {
            let ar = self.left[k];
            let ai = self.left[h + k];
            let br = self.right[k];
            let bi = self.right[h + k];
            self.left[k] = ai.mul_add(bi, ar * br);
            self.left[h + k] = (-ar).mul_add(bi, ai * br);
        }
        for v in &mut self.left {
            *v *= 1.0 / n as f32;
        }
        self.fft.inverse(&mut self.left);
        let tail = count - 1;
        let total = count * 2 - 1;
        self.correlation[1..1 + tail].copy_from_slice(&self.left[n - tail..]);
        self.correlation[1 + tail..1 + total].copy_from_slice(&self.left[..total - tail]);
        for v in &mut self.correlation {
            *v *= n as f32;
        }
        let shift = h - count;
        if shift == 0 {
            self.correlation[0] = 0.0;
        } else {
            self.correlation.copy_within(1..1 + total, 1 + shift);
            self.correlation[..shift + 1].fill(0.0);
            self.correlation[h + count..].fill(0.0);
        }
    }
}

fn rise_score(current: f32, previous: f32, threshold: f64, suppressed: bool) -> f32 {
    if !(previous < current && f64::from(current) > threshold) || suppressed {
        return 0.0;
    }
    let numerator = f64::from(current - previous).mul_add(
        (f64::from(current) + 1e-31) / (f64::from(previous) + 1e-31),
        1e-31,
    );
    (numerator / (f64::from(current + previous) + 1e-31)).tanh() as f32
}

use crate::memory::HeapBytes;
impl HeapBytes for MarkAnalysis {
    fn heap_bytes(&self) -> usize {
        self.mono.heap_bytes()
            + self.work.heap_bytes()
            + self.left.heap_bytes()
            + self.right.heap_bytes()
            + self.correlation.heap_bytes()
            + self.fft.heap_bytes()
    }
}

#[cfg(test)]
mod long_stream_tests {
    use super::*;
    #[test]
    fn absolute_marks_cross_the_32_bit_boundary_without_changing_analysis() {
        let input: Vec<_> = (0..4096).map(|i| (i as f32 * 0.031).sin() * 0.2).collect();
        let run = |origin| {
            let mut analysis = MarkAnalysis::new(1024, 1, 8192).unwrap();
            let mut marks = PitchMarks::new(128).unwrap();
            if origin != 0 {
                marks
                    .push(PitchMark {
                        position: origin,
                        ..Default::default()
                    })
                    .unwrap();
                marks.retain_last(1).unwrap();
            }
            analysis.state.last_position = origin;
            analysis.process(&[&input], &mut marks).unwrap();
            marks
                .records()
                .iter()
                .map(|m| PitchMark {
                    position: m.position - origin,
                    ..*m
                })
                .collect::<Vec<_>>()
        };
        let expected = run(0);
        for origin in [i64::from(i32::MAX) - 512, 1_i64 << 40] {
            assert_eq!(run(origin), expected);
        }
    }
}

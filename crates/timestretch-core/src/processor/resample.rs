use crate::StretchError;

/// Four-point interpolation methods for final resampling.
#[derive(Clone, Copy, Debug)]
pub enum Interpolation {
    /// Cubic Hermite interpolation used by the streaming scheduler.
    Hermite,
    /// Cubic polynomial interpolation using four adjacent samples.
    Cubic,
}

/// Persistent observable resampler state, including interpolation scratch scalars.
#[derive(Clone, Debug, Default)]
pub struct ResamplerState {
    /// Four interpolation coefficients.
    pub coefficients: [f32; 4],
    /// Fraction and its square/cube.
    pub fractions: [f32; 3],
    /// Last channel's left and right slopes.
    pub slopes: [f32; 2],
    /// Integer input position relative to the next block.
    pub input_index: usize,
    /// Double-precision fractional source position relative to the next block.
    pub source_position: f64,
}

/// Fixed-capacity final resampling component. Three history samples and the
/// fractional cursor persist across arbitrary block boundaries, including zero
/// frames. At unity, the interpolation origin produces a two-sample delay.
pub struct Resampler {
    channels: usize,
    capacity: usize,
    interpolation: Interpolation,
    input: Vec<Vec<f32>>,
    state: ResamplerState,
}
impl Resampler {
    /// Allocate all history/input storage.
    pub fn new(
        channels: usize,
        capacity: usize,
        interpolation: Interpolation,
    ) -> Result<Self, StretchError> {
        super::check_channels(channels)?;
        super::check_capacity(capacity)?;
        Ok(Self {
            channels,
            capacity,
            interpolation,
            input: vec![vec![0.0; capacity + 4]; channels],
            state: ResamplerState::default(),
        })
    }
    /// Current arithmetic state.
    pub fn state(&self) -> &ResamplerState {
        &self.state
    }
    /// Reset the arithmetic state and input cells 1..5, retaining other storage.
    pub fn reset(&mut self) {
        self.state = ResamplerState::default();
        for ch in &mut self.input {
            ch[1..5].fill(0.0);
        }
    }
    /// Number of output frames this call will emit, without changing state.
    pub fn output_frames(&self, frames: usize, ratio: f32) -> Result<usize, StretchError> {
        if !ratio.is_finite() || !(0.1..=10.0).contains(&ratio) {
            return Err(StretchError::InvalidRatio(
                "resampling ratio must be 0.1..=10",
            ));
        }
        if frames > self.capacity {
            return Err(super::invalid("resampler input exceeds capacity"));
        }
        let mut position = self.state.source_position;
        let mut index = self.state.input_index;
        let mut count = 0;
        while index < frames {
            count += 1;
            position += f64::from(ratio);
            index = position as usize;
        }
        Ok(count)
    }
    /// Render into caller-owned planar buffers. Returns output frames; validates
    /// capacities before mutating state. No allocations or locks during calls.
    pub fn process(
        &mut self,
        input: &[&[f32]],
        output: &mut [&mut [f32]],
        ratio: f32,
    ) -> Result<usize, StretchError> {
        #[cfg(feature = "qa")]
        let _profile = crate::processor::qa::region(5);
        let frames = super::input_frames(input, self.channels)?;
        let needed = self.output_frames(frames, ratio)?;
        super::output_capacity(output, self.channels, needed)?;
        for (ch, source) in input.iter().enumerate() {
            self.input[ch][4..4 + frames].copy_from_slice(source);
        }
        let state = &mut self.state;
        let mut written = 0;
        let mut index = state.input_index;
        while index < frames {
            let fraction = state.fractions[0];
            let squared = fraction * fraction;
            state.fractions[1] = squared;
            if matches!(self.interpolation, Interpolation::Hermite) {
                let cubed = fraction * squared;
                state.fractions[2] = cubed;
                state.coefficients[0] = cubed.mul_add(2.0, squared * (-3.0)) + 1.0;
                state.coefficients[1] = fraction + squared.mul_add(-2.0, cubed);
                state.coefficients[2] = cubed - squared;
                state.coefficients[3] = cubed.mul_add(-2.0, squared * 3.0);
                for (ch, destination) in output.iter_mut().enumerate() {
                    let x = &self.input[ch][index + 1..index + 5];
                    let left = (((x[1] - x[0]) + x[2]) - x[1]) * 0.5;
                    let right = (((x[2] - x[1]) + x[3]) - x[2]) * 0.5;
                    state.slopes = [left, right];
                    let mut y = left * state.coefficients[1];
                    y = state.coefficients[0].mul_add(x[1], y);
                    y = state.coefficients[2].mul_add(right, y);
                    destination[written] = state.coefficients[3].mul_add(x[2], y);
                }
            } else {
                for (ch, destination) in output.iter_mut().enumerate() {
                    let x = &self.input[ch][index + 1..index + 5];
                    state.coefficients[0] = ((x[3] - x[2]) - x[0]) + x[1];
                    state.coefficients[1] = (x[0] - x[1]) - state.coefficients[0];
                    state.coefficients[2] = x[2] - x[0];
                    state.coefficients[3] = x[1];
                    let mut y = (state.coefficients[0] * fraction)
                        .mul_add(squared, state.coefficients[1] * squared);
                    y = state.coefficients[2].mul_add(fraction, y);
                    destination[written] = state.coefficients[3] + y;
                }
            }
            written += 1;
            state.source_position += f64::from(ratio);
            index = state.source_position as usize;
            state.fractions[0] = (state.source_position - index as f64) as f32;
        }
        state.source_position -= frames as f64;
        for ch in &mut self.input {
            ch[1] = ch[frames + 1];
            ch[2] = ch[frames + 2];
            ch[3] = ch[frames + 3];
        }
        state.input_index = state.source_position as usize;
        Ok(written)
    }
}

use crate::memory::HeapBytes;
impl HeapBytes for Resampler {
    fn heap_bytes(&self) -> usize {
        self.input.heap_bytes()
    }
}

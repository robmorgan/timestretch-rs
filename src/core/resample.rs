//! Sample-rate conversion via linear, cubic, and windowed-sinc interpolation.

use std::sync::Arc;

use crate::error::StretchError;

/// Linear interpolation resampling.
///
/// Resamples a mono audio signal by the given ratio (output_len / input_len).
/// Used for pitch correction after time stretching.
pub fn resample_linear(input: &[f32], output_len: usize) -> Vec<f32> {
    if input.is_empty() || output_len == 0 {
        return vec![];
    }
    if input.len() == 1 {
        return vec![input[0]; output_len];
    }

    let ratio = (input.len() - 1) as f64 / (output_len.max(1) - 1).max(1) as f64;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let pos = i as f64 * ratio;
        let idx = pos as usize;
        let frac = (pos - idx as f64) as f32;

        if idx + 1 < input.len() {
            output.push(input[idx] * (1.0 - frac) + input[idx + 1] * frac);
        } else {
            output.push(input[input.len() - 1]);
        }
    }

    output
}

/// Cubic interpolation resampling.
///
/// Uses 4-point Hermite interpolation for better quality than linear.
pub fn resample_cubic(input: &[f32], output_len: usize) -> Vec<f32> {
    if input.is_empty() || output_len == 0 {
        return vec![];
    }
    if input.len() < 4 {
        return resample_linear(input, output_len);
    }

    let ratio = (input.len() - 1) as f64 / (output_len.max(1) - 1).max(1) as f64;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let pos = i as f64 * ratio;
        let idx = pos as usize;
        let frac = (pos - idx as f64) as f32;

        // Get 4 surrounding samples with boundary clamping
        let s0 = input[idx.saturating_sub(1)];
        let s1 = input[idx];
        let s2 = input[(idx + 1).min(input.len() - 1)];
        let s3 = input[(idx + 2).min(input.len() - 1)];

        // Hermite interpolation
        let c0 = s1;
        let c1 = 0.5 * (s2 - s0);
        let c2 = s0 - 2.5 * s1 + 2.0 * s2 - 0.5 * s3;
        let c3 = 0.5 * (s3 - s0) + 1.5 * (s1 - s2);

        output.push(((c3 * frac + c2) * frac + c1) * frac + c0);
    }

    output
}

/// Default number of sinc lobes for high-quality resampling.
const DEFAULT_SINC_LOBES: usize = 8;

/// Windowed-sinc resampling for high-quality sample-rate conversion.
///
/// Uses a sinc interpolation kernel windowed with a Kaiser-Bessel window.
/// `lobes` controls the quality: more lobes = sharper cutoff but slower.
/// Typical values: 4 (fast), 8 (balanced), 16 (high quality).
///
/// Falls back to cubic interpolation for very short inputs (< 2 * lobes).
pub fn resample_sinc(input: &[f32], output_len: usize, lobes: usize) -> Vec<f32> {
    if input.is_empty() || output_len == 0 {
        return vec![];
    }
    let lobes = lobes.max(1);
    if input.len() < 2 * lobes {
        return resample_cubic(input, output_len);
    }

    let ratio = (input.len() - 1) as f64 / (output_len.max(1) - 1).max(1) as f64;
    let mut output = Vec::with_capacity(output_len);

    // Pre-compute Kaiser window for the sinc kernel.
    // Beta = 6.0 gives ~60 dB stopband attenuation, good for audio.
    let beta = 6.0f64;
    let bessel_beta = bessel_i0(beta);

    for i in 0..output_len {
        let pos = i as f64 * ratio;
        let center = pos as isize;
        let frac = pos - center as f64;

        let mut sample = 0.0f64;
        let mut weight_sum = 0.0f64;

        // Convolve with the windowed sinc kernel
        let start = -(lobes as isize) + 1;
        let end = lobes as isize + 1;
        for j in start..end {
            let idx = center + j;
            if idx < 0 || idx >= input.len() as isize {
                continue;
            }

            let x = frac - j as f64;
            let sinc_val = if x.abs() < 1e-10 {
                1.0
            } else {
                let pi_x = std::f64::consts::PI * x;
                pi_x.sin() / pi_x
            };

            // Kaiser window
            let t = (j as f64 - frac) / lobes as f64;
            let window = if t.abs() <= 1.0 {
                bessel_i0(beta * (1.0 - t * t).max(0.0).sqrt()) / bessel_beta
            } else {
                0.0
            };

            let w = sinc_val * window;
            sample += input[idx as usize] as f64 * w;
            weight_sum += w;
        }

        // Normalize to preserve DC gain
        if weight_sum.abs() > 1e-10 {
            sample /= weight_sum;
        }

        output.push(sample as f32);
    }

    output
}

/// Windowed-sinc resampling with default quality (8 lobes).
pub fn resample_sinc_default(input: &[f32], output_len: usize) -> Vec<f32> {
    resample_sinc(input, output_len, DEFAULT_SINC_LOBES)
}

/// Half-width of the streaming sinc kernel in zero-crossings at unity step.
///
/// The full kernel spans `2 * STREAM_SINC_HALF_TAPS` input samples when the
/// resampling step is <= 1 (pitch down / unity). 16 zero-crossings keep the
/// transition band narrow enough that pitch-up folding stays confined to the
/// top of the audible band at DJ-typical steps.
pub const STREAM_SINC_HALF_TAPS: usize = 16;

/// Table entries per zero-crossing. Linear interpolation between adjacent
/// entries yields a continuously variable fractional phase, equivalent to a
/// polyphase bank with inter-branch interpolation.
const STREAM_SINC_PHASES: usize = 512;

/// Kaiser window beta for the streaming sinc prototype (~-90 dB stopband).
const STREAM_SINC_KAISER_BETA: f64 = 9.0;

/// Maximum half-width of the dilated kernel when anti-aliasing pitch-up.
pub const STREAM_SINC_MAX_HALF_TAPS: usize = 80;

/// Maximum step for which the kernel cutoff is dilated. Steps beyond this
/// (more than +2 octaves of pitch-up) partially alias.
pub(crate) const STREAM_SINC_MAX_STEP: f64 = 4.0;

/// Extra cutoff scaling applied when downsampling so the filter's *stopband
/// edge* (not its -6 dB point) lands at the fold frequency. Without this,
/// half the transition band folds back with only mild attenuation.
const STREAM_SINC_CUTOFF_SCALE: f64 = 0.85;

/// The cutoff scaling ramps in between step 1.0 and this step, keeping the
/// unity kernel an exact delta (bit-transparent passthrough) while folding at
/// near-unity steps only touches content above ~20 kHz.
const STREAM_SINC_CUTOFF_RAMP_END: f64 = 1.1;

/// History retained across calls: enough for the widest kernel on both sides
/// of the cursor plus slack for cursor drift near block boundaries.
const STREAM_SINC_HISTORY: usize = 192;

/// Immutable Kaiser-windowed sinc prototype shared across channels.
///
/// Stores one side of the symmetric kernel sampled at `STREAM_SINC_PHASES`
/// points per zero-crossing; lookups linearly interpolate between entries.
#[derive(Debug)]
pub struct SincInterpTable {
    taps: Vec<f32>,
}

impl SincInterpTable {
    /// Builds the default streaming prototype (8 half-taps, Kaiser beta 8).
    pub fn new_stream_default() -> Arc<Self> {
        let entries = STREAM_SINC_HALF_TAPS * STREAM_SINC_PHASES;
        // One guard entry past the end keeps the lerp in `weight` in bounds.
        let mut taps = vec![0.0f32; entries + 2];
        let bessel_beta = bessel_i0(STREAM_SINC_KAISER_BETA);
        for (i, tap) in taps.iter_mut().enumerate().take(entries + 1) {
            let u = i as f64 / STREAM_SINC_PHASES as f64;
            let sinc_val = if u < 1e-12 {
                1.0
            } else {
                let pi_u = std::f64::consts::PI * u;
                pi_u.sin() / pi_u
            };
            let t = u / STREAM_SINC_HALF_TAPS as f64;
            let window = if t <= 1.0 {
                bessel_i0(STREAM_SINC_KAISER_BETA * (1.0 - t * t).max(0.0).sqrt()) / bessel_beta
            } else {
                0.0
            };
            *tap = (sinc_val * window) as f32;
        }
        Arc::new(Self { taps })
    }

    /// Kernel weight at absolute offset `u_abs` (in zero-crossings).
    /// Crate-visible so random-access interpolators (the SOLA corrector's
    /// elastic ring reads) can share the prototype.
    #[inline]
    pub(crate) fn weight(&self, u_abs: f64) -> f32 {
        if u_abs >= STREAM_SINC_HALF_TAPS as f64 {
            return 0.0;
        }
        let x = u_abs * STREAM_SINC_PHASES as f64;
        let i = x as usize;
        let frac = (x - i as f64) as f32;
        let a = self.taps[i];
        let b = self.taps[i + 1];
        a + (b - a) * frac
    }
}

/// Stateful bounded-latency windowed-sinc resampler for realtime pitch.
///
/// Consumes source samples at a (possibly time-varying) `step` per output
/// sample. For `step > 1` (pitch up / downsampling) the kernel cutoff is
/// dilated by `1/step` so the output stays anti-aliased up to
/// `STREAM_SINC_MAX_STEP`.
///
/// The cursor is sample-aligned with the input (no phase delay); outputs lag
/// input availability by up to `STREAM_SINC_HALF_TAPS` samples of lookahead
/// (`STREAM_SINC_MAX_HALF_TAPS` when the kernel is dilated), which
/// [`flush_into`](Self::flush_into) releases at end of stream.
///
/// All state is fixed-size: no allocations after construction.
#[derive(Debug, Clone)]
pub struct StreamingSincResampler {
    table: Arc<SincInterpTable>,
    history: [f32; STREAM_SINC_HISTORY],
    /// Fractional source cursor in the local frame (history starts at 0).
    src_pos: f64,
    /// Step at the end of the previous block; ramp anchor for the next one.
    prev_step: f64,
    /// Total input samples fed since the last reset.
    fed_total: u64,
    has_started: bool,
}

impl StreamingSincResampler {
    /// Creates a resampler sharing the given prototype table.
    pub fn new(table: Arc<SincInterpTable>) -> Self {
        Self {
            table,
            history: [0.0; STREAM_SINC_HISTORY],
            src_pos: STREAM_SINC_HISTORY as f64,
            prev_step: 1.0,
            fed_total: 0,
            has_started: false,
        }
    }

    /// Clears all state back to stream start.
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.src_pos = STREAM_SINC_HISTORY as f64;
        self.prev_step = 1.0;
        self.fed_total = 0;
        self.has_started = false;
    }

    /// Fractional source position (in input samples fed since the last
    /// reset) of the *next* output sample this resampler will emit.
    ///
    /// After each block the cursor is rebased so local index
    /// `STREAM_SINC_HISTORY` is the first not-yet-fed sample; subtracting
    /// that origin and adding the fed total yields an exact global cursor.
    /// Starts at 0.0 and advances by exactly `step` per emitted sample,
    /// making it suitable for driving source-position bookkeeping when the
    /// resampler sits ahead of downstream processing.
    pub fn next_output_source_pos(&self) -> f64 {
        self.fed_total as f64 + self.src_pos - STREAM_SINC_HISTORY as f64
    }

    /// Returns whether any input has been consumed since the last reset.
    pub fn is_engaged(&self) -> bool {
        self.has_started
    }

    /// Lookahead of the current kernel in samples (8 at step <= 1).
    pub fn group_delay_samples(&self) -> usize {
        STREAM_SINC_HALF_TAPS
    }

    /// Half-span of the kernel at the most recent step: the number of input
    /// samples held back as lookahead. [`STREAM_SINC_HALF_TAPS`] at unity or
    /// pitch-down, up to [`STREAM_SINC_MAX_HALF_TAPS`] when pitching up.
    pub fn current_half_span(&self) -> usize {
        Self::half_span_for_step(self.prev_step)
    }

    /// Normalized kernel cutoff for a given step (1.0 = input Nyquist).
    ///
    /// For step <= 1 the kernel is a pure interpolator (delta at unity). For
    /// step > 1 the cutoff shrinks to `1/step`, additionally scaled by
    /// `STREAM_SINC_CUTOFF_SCALE` (ramped in over
    /// `1.0..STREAM_SINC_CUTOFF_RAMP_END`) so the stopband edge sits at the
    /// fold frequency.
    #[inline]
    fn cutoff_for_step(step: f64) -> f64 {
        if step <= 1.0 {
            return 1.0;
        }
        let s = step.min(STREAM_SINC_MAX_STEP);
        let t = ((s - 1.0) / (STREAM_SINC_CUTOFF_RAMP_END - 1.0)).min(1.0);
        let g = 1.0 - (1.0 - STREAM_SINC_CUTOFF_SCALE) * t;
        g / s
    }

    /// Half-span of the (possibly dilated) kernel for a given step.
    #[inline]
    fn half_span_for_step(step: f64) -> usize {
        ((STREAM_SINC_HALF_TAPS as f64 / Self::cutoff_for_step(step)).ceil() as usize)
            .min(STREAM_SINC_MAX_HALF_TAPS)
    }

    /// Resamples `input`, ramping the step linearly from the previous block's
    /// final step to `step` across this block (parameterized by input
    /// consumption) to avoid zipper artifacts under pitch sweeps.
    ///
    /// Clears `output` first and never grows it beyond its capacity; a full
    /// buffer yields `StretchError::BufferOverflow` like the linear path.
    pub fn process_into(
        &mut self,
        input: &[f32],
        step: f64,
        output: &mut Vec<f32>,
    ) -> Result<(), StretchError> {
        self.process_into_capped(input, step, output, usize::MAX)
    }

    /// [`process_into`](Self::process_into) with an emission cap: at most
    /// `max_out` samples are emitted this call; any further outputs the fed
    /// input covers stay pending (the cursor holds) and emit on the next
    /// call — at that call's step. This is how the engine lands timestamped
    /// tempo retargets on an exact output sample even when the boundary
    /// falls inside one input sample's multiple emissions.
    pub fn process_into_capped(
        &mut self,
        input: &[f32],
        step: f64,
        output: &mut Vec<f32>,
        max_out: usize,
    ) -> Result<(), StretchError> {
        output.clear();
        if input.is_empty() {
            return Ok(());
        }

        let step_end = step;
        let step_begin = if self.has_started {
            self.prev_step
        } else {
            step_end
        };
        self.has_started = true;

        let h = STREAM_SINC_HISTORY;
        let total = h + input.len();
        // Widest kernel needed anywhere in this block gates emission so every
        // output has its full right half-kernel available.
        let half_span_max = Self::half_span_for_step(step_begin.max(step_end));
        let inv_input_len = 1.0 / input.len() as f64;

        let mut pos = self.src_pos;
        let start_pos = pos;
        while pos + half_span_max as f64 + 1.0 <= total as f64 && output.len() < max_out {
            if output.len() == output.capacity() {
                return Err(StretchError::BufferOverflow {
                    buffer: "stream_pitch_resample_output",
                    requested: output.len().saturating_add(1),
                    available: output.capacity(),
                });
            }

            let t = ((pos - start_pos) * inv_input_len).clamp(0.0, 1.0);
            let step_now = step_begin + (step_end - step_begin) * t;
            let cutoff = Self::cutoff_for_step(step_now);
            let half_span = Self::half_span_for_step(step_now);

            let center = pos.floor() as usize;
            let frac = pos - center as f64;
            let lo = center.saturating_sub(half_span);
            let hi = (center + half_span + 1).min(total);

            // Two passes so the compiler can vectorize the second: fill the
            // kernel weights for this output sample into a fixed stack
            // buffer (the table lookup does not vectorize), then take plain
            // dot products over the two contiguous source regions.
            let mut weights = [0.0f32; 2 * STREAM_SINC_MAX_HALF_TAPS + 1];
            let n_taps = hi - lo;
            let mut weight_sum = 0.0f64;
            for (k, w) in weights[..n_taps].iter_mut().enumerate() {
                let u = (((lo + k) as f64 - center as f64) - frac) * cutoff;
                let val = self.table.weight(u.abs());
                *w = val;
                weight_sum += val as f64;
            }
            let mut acc = 0.0f64;
            if lo < h {
                let h_end = hi.min(h);
                acc += dot_f32_f64(&self.history[lo..h_end], &weights[..h_end - lo]);
            }
            if hi > h {
                let in_lo = lo.max(h);
                acc += dot_f32_f64(&input[in_lo - h..hi - h], &weights[in_lo - lo..n_taps]);
            }
            if weight_sum.abs() > 1e-12 {
                acc /= weight_sum;
            }
            output.push(acc as f32);
            pos += step_now;
        }

        // Retain the last `h` samples of (history ++ input) and rebase the
        // cursor into the new local frame.
        let n = input.len();
        if n >= h {
            self.history.copy_from_slice(&input[n - h..]);
        } else {
            self.history.copy_within(n.., 0);
            self.history[h - n..].copy_from_slice(input);
        }
        self.src_pos = pos - n as f64;
        self.prev_step = step_end;
        self.fed_total += n as u64;
        Ok(())
    }

    /// Drains the lookahead tail by feeding just enough zeros to release all
    /// outputs covering real input, then resets.
    pub fn flush_into(&mut self, step: f64, output: &mut Vec<f32>) -> Result<(), StretchError> {
        if !self.has_started {
            output.clear();
            return Ok(());
        }
        let zeros = [0.0f32; STREAM_SINC_MAX_HALF_TAPS + 2];
        let drain = Self::half_span_for_step(step.max(self.prev_step)) + 2;
        self.process_into(&zeros[..drain], step, output)?;
        self.reset();
        Ok(())
    }
}

/// Dot product of equal-length slices, accumulated in f64 across four
/// independent lanes so the compiler can keep it in vector registers.
/// Shared by the streaming resampler and the SOLA corrector's ring reads.
#[inline]
pub(crate) fn dot_f32_f64(samples: &[f32], weights: &[f32]) -> f64 {
    debug_assert_eq!(samples.len(), weights.len());
    let mut acc = [0.0f64; 4];
    let mut s_chunks = samples.chunks_exact(4);
    let mut w_chunks = weights.chunks_exact(4);
    for (s, w) in (&mut s_chunks).zip(&mut w_chunks) {
        acc[0] += s[0] as f64 * w[0] as f64;
        acc[1] += s[1] as f64 * w[1] as f64;
        acc[2] += s[2] as f64 * w[2] as f64;
        acc[3] += s[3] as f64 * w[3] as f64;
    }
    for (s, w) in s_chunks.remainder().iter().zip(w_chunks.remainder()) {
        acc[0] += *s as f64 * *w as f64;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// Modified Bessel function of the first kind, order zero.
/// Approximated using the power series expansion.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half_x = x * 0.5;

    for k in 1..=25 {
        term *= (half_x / k as f64) * (half_x / k as f64);
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_linear_identity() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let output = resample_linear(&input, 100);
        assert_eq!(output.len(), 100);
        for i in 0..100 {
            assert!((output[i] - input[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_resample_linear_upsample() {
        let input = vec![0.0, 1.0];
        let output = resample_linear(&input, 5);
        assert_eq!(output.len(), 5);
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[4] - 1.0).abs() < 1e-6);
        // Monotonically increasing
        for i in 1..5 {
            assert!(output[i] >= output[i - 1]);
        }
    }

    #[test]
    fn test_resample_linear_downsample() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32) / 99.0).collect();
        let output = resample_linear(&input, 50);
        assert_eq!(output.len(), 50);
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[49] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_resample_empty() {
        assert!(resample_linear(&[], 10).is_empty());
        assert!(resample_linear(&[1.0, 2.0], 0).is_empty());
        assert!(resample_cubic(&[], 10).is_empty());
    }

    #[test]
    fn test_resample_cubic_identity() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let output = resample_cubic(&input, 100);
        assert_eq!(output.len(), 100);
        for i in 0..100 {
            assert!(
                (output[i] - input[i]).abs() < 1e-4,
                "mismatch at {}: {} vs {}",
                i,
                output[i],
                input[i]
            );
        }
    }

    #[test]
    fn test_resample_cubic_smooth() {
        // Cubic should produce smoother output than linear for a sine wave
        let input: Vec<f32> = (0..100)
            .map(|i| (i as f32 * std::f32::consts::PI * 2.0 / 100.0).sin())
            .collect();
        let output = resample_cubic(&input, 200);
        assert_eq!(output.len(), 200);
        // Check output is bounded
        for &s in &output {
            assert!((-1.1..=1.1).contains(&s));
        }
    }

    // --- windowed-sinc resampling tests ---

    #[test]
    fn test_resample_sinc_identity() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let output = resample_sinc_default(&input, 100);
        assert_eq!(output.len(), 100);
        for i in 0..100 {
            assert!(
                (output[i] - input[i]).abs() < 1e-3,
                "mismatch at {}: {} vs {}",
                i,
                output[i],
                input[i]
            );
        }
    }

    #[test]
    fn test_resample_sinc_upsample_sine() {
        // Sinc should accurately upsample a sine wave
        let sample_rate = 100.0;
        let freq = 5.0; // Well below Nyquist
        let input: Vec<f32> = (0..100)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let output = resample_sinc_default(&input, 200);
        assert_eq!(output.len(), 200);

        // Verify the output matches the expected sine at the upsampled rate
        let new_rate = 200.0;
        let mut max_error = 0.0f32;
        // Skip edges where the sinc kernel is truncated
        for (i, &out_val) in output.iter().enumerate().take(180).skip(20) {
            let expected = (2.0 * std::f32::consts::PI * freq * i as f32 / new_rate).sin();
            let err = (out_val - expected).abs();
            max_error = max_error.max(err);
        }
        assert!(
            max_error < 0.15,
            "Sinc upsample max error {:.4} should be < 0.15",
            max_error
        );
    }

    #[test]
    fn test_resample_sinc_downsample() {
        let input: Vec<f32> = (0..200).map(|i| (i as f32) / 199.0).collect();
        let output = resample_sinc_default(&input, 50);
        assert_eq!(output.len(), 50);
        // Endpoints should be preserved
        assert!((output[0] - 0.0).abs() < 0.05);
        assert!((output[49] - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_resample_sinc_empty() {
        assert!(resample_sinc(&[], 10, 8).is_empty());
        assert!(resample_sinc(&[1.0], 0, 8).is_empty());
    }

    #[test]
    fn test_resample_sinc_short_input_fallback() {
        // Input shorter than 2 * lobes should fall back to cubic
        let input = vec![0.0, 0.5, 1.0];
        let output = resample_sinc(&input, 6, 8);
        assert_eq!(output.len(), 6);
        // Should produce valid output via cubic fallback
        assert!(output.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_resample_sinc_better_than_cubic_for_sine() {
        // Sinc should have lower interpolation error than cubic for a sine
        let freq = 10.0;
        let sample_rate = 100.0;
        let input: Vec<f32> = (0..100)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let sinc_out = resample_sinc_default(&input, 200);
        let cubic_out = resample_cubic(&input, 200);

        let new_rate = 200.0;
        let mut sinc_err = 0.0f32;
        let mut cubic_err = 0.0f32;
        for i in 20..180 {
            let expected = (2.0 * std::f32::consts::PI * freq * i as f32 / new_rate).sin();
            sinc_err += (sinc_out[i] - expected).abs();
            cubic_err += (cubic_out[i] - expected).abs();
        }

        assert!(
            sinc_err <= cubic_err,
            "Sinc error ({:.4}) should be <= cubic error ({:.4})",
            sinc_err,
            cubic_err
        );
    }

    // --- streaming sinc resampler tests ---

    fn sine(freq: f32, sample_rate: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    /// Goertzel power of `signal` at `freq`.
    fn goertzel_power(signal: &[f32], freq: f64, sample_rate: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI * freq / sample_rate;
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &x in signal {
            let s0 = x as f64 + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (signal.len() as f64 / 2.0).powi(2)
    }

    fn stream_all(
        resampler: &mut StreamingSincResampler,
        input: &[f32],
        step: f64,
        chunk: usize,
    ) -> Vec<f32> {
        let mut out = Vec::new();
        let mut buf: Vec<f32> = Vec::with_capacity(input.len() * 4 + 256);
        for block in input.chunks(chunk) {
            resampler.process_into(block, step, &mut buf).unwrap();
            out.extend_from_slice(&buf);
        }
        out
    }

    #[test]
    fn test_streaming_sinc_unity_identity() {
        let table = SincInterpTable::new_stream_default();
        let mut rs = StreamingSincResampler::new(table);
        let input = sine(440.0, 44100.0, 4096);
        let mut out = Vec::with_capacity(8192);
        rs.process_into(&input, 1.0, &mut out).unwrap();

        // Sample-aligned passthrough: output[i] == input[i]; delivery lags by
        // the kernel lookahead.
        assert!(out.len() >= input.len() - STREAM_SINC_HALF_TAPS - 2);
        assert!(out.len() <= input.len());
        for (i, (&o, &x)) in out.iter().zip(input.iter()).enumerate() {
            assert!(
                (o - x).abs() < 1e-4,
                "unity mismatch at {}: {} vs {}",
                i,
                o,
                x
            );
        }
    }

    #[test]
    fn test_streaming_sinc_block_split_invariance() {
        // Constant exactly-representable step: chunking must not change output.
        let table = SincInterpTable::new_stream_default();
        let input = sine(997.0, 44100.0, 8192);
        let step = 1.25;

        let mut whole = StreamingSincResampler::new(Arc::clone(&table));
        let mut out_whole = Vec::with_capacity(16384);
        whole.process_into(&input, step, &mut out_whole).unwrap();

        // Pseudo-random chunk sizes in [64, 1024).
        let mut chunked = StreamingSincResampler::new(table);
        let mut out_chunked: Vec<f32> = Vec::new();
        let mut buf = Vec::with_capacity(16384);
        let mut offset = 0usize;
        let mut state = 0x2545F491u64;
        while offset < input.len() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let size = (64 + (state >> 33) % 960) as usize;
            let end = (offset + size).min(input.len());
            chunked
                .process_into(&input[offset..end], step, &mut buf)
                .unwrap();
            out_chunked.extend_from_slice(&buf);
            offset = end;
        }

        assert_eq!(out_whole.len(), out_chunked.len());
        for (i, (&a, &b)) in out_whole.iter().zip(out_chunked.iter()).enumerate() {
            assert!(a == b, "split divergence at {}: {} vs {}", i, a, b);
        }
    }

    #[test]
    fn test_streaming_sinc_antialiases_pitch_up() {
        // 18 kHz at step 1.3 folds to 44100 - 23400 = 20700 Hz if unfiltered.
        let sample_rate = 44100.0;
        let step = 1.3;
        let input = sine(18000.0, sample_rate as f32, 16384);

        let table = SincInterpTable::new_stream_default();
        let mut rs = StreamingSincResampler::new(table);
        let sinc_out = stream_all(&mut rs, &input, step, 512);

        // Naive linear resampling at the same step (the legacy quality level).
        let mut linear_out = Vec::new();
        let mut pos = 0.0f64;
        while pos + 1.0 < input.len() as f64 {
            let i = pos as usize;
            let frac = (pos - i as f64) as f32;
            linear_out.push(input[i] * (1.0 - frac) + input[i + 1] * frac);
            pos += step;
        }

        let alias_freq = sample_rate - 18000.0 * step;
        let skip = 256;
        let sinc_alias = goertzel_power(&sinc_out[skip..], alias_freq, sample_rate);
        let linear_alias = goertzel_power(&linear_out[skip..], alias_freq, sample_rate);
        let ratio_db = 10.0 * (linear_alias / sinc_alias.max(1e-30)).log10();
        assert!(
            ratio_db > 40.0,
            "sinc alias rejection only {:.1} dB better than linear (sinc {:.3e}, linear {:.3e})",
            ratio_db,
            sinc_alias,
            linear_alias
        );
    }

    #[test]
    fn test_streaming_sinc_ramped_step_smooth() {
        // Sweep step 1.0 -> 1.12 across blocks; no sample-to-sample jumps
        // beyond what a 1 kHz sine's slew allows.
        let sample_rate = 44100.0f32;
        let input = sine(1000.0, sample_rate, 16384);
        let table = SincInterpTable::new_stream_default();
        let mut rs = StreamingSincResampler::new(table);

        let mut out: Vec<f32> = Vec::new();
        let mut buf = Vec::with_capacity(32768);
        let blocks: Vec<&[f32]> = input.chunks(512).collect();
        let n_blocks = blocks.len();
        for (bi, block) in blocks.into_iter().enumerate() {
            let step = 1.0 + 0.12 * (bi as f64 / (n_blocks - 1) as f64);
            rs.process_into(block, step, &mut buf).unwrap();
            out.extend_from_slice(&buf);
        }

        // Max slew of a 1 kHz sine at <=1.12x pitch, with margin.
        let max_slew = 2.0 * std::f32::consts::PI * 1000.0 * 1.12 / sample_rate * 1.5;
        for w in out.windows(2) {
            let d = (w[1] - w[0]).abs();
            assert!(d <= max_slew, "discontinuity: |Δ| = {} > {}", d, max_slew);
        }
    }

    #[test]
    fn test_streaming_sinc_flush_releases_tail() {
        let table = SincInterpTable::new_stream_default();
        let mut rs = StreamingSincResampler::new(table);
        let step = 1.3;
        let input = sine(440.0, 44100.0, 4096);

        let mut out = stream_all(&mut rs, &input, step, 300);
        let mut tail = Vec::with_capacity(256);
        rs.flush_into(step, &mut tail).unwrap();
        out.extend_from_slice(&tail);

        let expected = (input.len() as f64 / step) as usize;
        let slack = (STREAM_SINC_MAX_HALF_TAPS as f64 / step) as usize + 4;
        assert!(
            out.len() >= expected - slack && out.len() <= expected + slack,
            "flush count {} not within {} of expected {}",
            out.len(),
            slack,
            expected
        );
        assert!(!rs.is_engaged(), "flush must reset engagement");
    }

    #[test]
    fn test_streaming_sinc_source_pos_unity_tracks_emission() {
        // At unity step the cursor advances exactly one source sample per
        // emitted output, starting from 0.
        let table = SincInterpTable::new_stream_default();
        let mut rs = StreamingSincResampler::new(table);
        assert_eq!(rs.next_output_source_pos(), 0.0);

        let input = sine(440.0, 44100.0, 2048);
        let out = stream_all(&mut rs, &input, 1.0, 256);
        assert!(
            (rs.next_output_source_pos() - out.len() as f64).abs() < 1e-9,
            "unity source pos {} != emitted {}",
            rs.next_output_source_pos(),
            out.len()
        );
    }

    #[test]
    fn test_streaming_sinc_source_pos_constant_step() {
        // With a constant step from stream start (no ramp), the cursor
        // advances exactly `step` per emitted sample regardless of chunking.
        let table = SincInterpTable::new_stream_default();
        let mut rs = StreamingSincResampler::new(table);
        let step = 1.25;
        let input = sine(997.0, 44100.0, 8192);
        let out = stream_all(&mut rs, &input, step, 300);
        let expected = out.len() as f64 * step;
        assert!(
            (rs.next_output_source_pos() - expected).abs() < 1e-6,
            "source pos {} != emitted*step {}",
            rs.next_output_source_pos(),
            expected
        );
        assert!(rs.next_output_source_pos() <= input.len() as f64);
    }

    #[test]
    fn test_streaming_sinc_source_pos_step_change_and_reset() {
        // Across a retarget the cursor stays monotonic, bounded by the fed
        // total, and equals the per-sample step integral; flush resets to 0.
        let table = SincInterpTable::new_stream_default();
        let mut rs = StreamingSincResampler::new(table);
        let input = sine(440.0, 44100.0, 8192);
        let mut buf = Vec::with_capacity(16384);

        let mut emitted = 0usize;
        let mut prev_pos = 0.0f64;
        let mut integral = 0.0f64;
        let mut prev_step = 1.0f64;
        for (bi, block) in input.chunks(512).enumerate() {
            let step = if bi < 8 { 1.0 } else { 0.8 };
            rs.process_into(block, step, &mut buf).unwrap();
            // Integrate the documented per-block linear ramp prev -> step.
            let begin = if bi == 0 { step } else { prev_step };
            let pos_now = rs.next_output_source_pos();
            assert!(pos_now >= prev_pos, "cursor went backwards at block {}", bi);
            integral += pos_now - prev_pos;
            // Advance must stay within the block's step range.
            let (lo, hi) = (begin.min(step), begin.max(step));
            let advance = pos_now - prev_pos;
            assert!(
                advance >= lo * buf.len() as f64 - 1e-6 && advance <= hi * buf.len() as f64 + 1e-6,
                "block {} advance {} outside [{}, {}] x {} outputs",
                bi,
                advance,
                lo,
                hi,
                buf.len()
            );
            prev_pos = pos_now;
            prev_step = step;
            emitted += buf.len();
        }
        assert!(emitted > 0);
        assert!(integral <= input.len() as f64);

        let mut tail = Vec::with_capacity(256);
        rs.flush_into(0.8, &mut tail).unwrap();
        assert_eq!(rs.next_output_source_pos(), 0.0, "flush must reset cursor");
    }

    #[test]
    fn test_streaming_sinc_dc_preserved_dilated() {
        // Constant input at step 2.0 exercises the dilated kernel; weight-sum
        // normalization must keep DC gain exact.
        let table = SincInterpTable::new_stream_default();
        let mut rs = StreamingSincResampler::new(table);
        let input = vec![0.7f32; 4096];
        let out = stream_all(&mut rs, &input, 2.0, 512);
        assert!(!out.is_empty());
        // Skip the ramp-in over zero-initialized history.
        for (i, &o) in out.iter().enumerate().skip(STREAM_SINC_MAX_HALF_TAPS) {
            assert!((o - 0.7).abs() < 1e-4, "DC drift at {}: {} vs 0.7", i, o);
        }
    }

    #[test]
    fn test_streaming_sinc_buffer_overflow_reported() {
        let table = SincInterpTable::new_stream_default();
        let mut rs = StreamingSincResampler::new(table);
        let input = vec![0.1f32; 1024];
        let mut out = Vec::with_capacity(8);
        let err = rs.process_into(&input, 1.0, &mut out).unwrap_err();
        assert!(matches!(
            err,
            StretchError::BufferOverflow {
                buffer: "stream_pitch_resample_output",
                ..
            }
        ));
    }

    #[test]
    fn test_bessel_i0_known_values() {
        // I0(0) = 1.0
        assert!((super::bessel_i0(0.0) - 1.0).abs() < 1e-10);
        // I0(1) ≈ 1.2660658777...
        assert!((super::bessel_i0(1.0) - 1.2660658777).abs() < 1e-6);
        // I0(3) ≈ 4.880792585...
        assert!((super::bessel_i0(3.0) - 4.880792585).abs() < 1e-4);
    }
}

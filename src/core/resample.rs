//! Sample-rate conversion via linear, cubic, and windowed-sinc interpolation.

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
        for i in 20..180 {
            let expected =
                (2.0 * std::f32::consts::PI * freq * i as f32 / new_rate).sin();
            let err = (output[i] - expected).abs();
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
            let expected =
                (2.0 * std::f32::consts::PI * freq * i as f32 / new_rate).sin();
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

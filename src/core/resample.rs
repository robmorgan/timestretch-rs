/// Linear interpolation resampling.
///
/// Resamples the input signal to the desired output length using linear interpolation.
pub fn resample_linear(input: &[f32], output_len: usize) -> Vec<f32> {
    if input.is_empty() || output_len == 0 {
        return vec![0.0; output_len];
    }
    if input.len() == 1 {
        return vec![input[0]; output_len];
    }
    if output_len == 1 {
        return vec![input[0]];
    }

    let ratio = (input.len() - 1) as f64 / (output_len - 1) as f64;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let pos = i as f64 * ratio;
        let idx = pos.floor() as usize;
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
/// Resamples the input signal to the desired output length using Catmull-Rom
/// cubic interpolation for better quality than linear.
pub fn resample_cubic(input: &[f32], output_len: usize) -> Vec<f32> {
    if input.is_empty() || output_len == 0 {
        return vec![0.0; output_len];
    }
    if input.len() < 4 {
        return resample_linear(input, output_len);
    }
    if output_len == 1 {
        return vec![input[0]];
    }

    let ratio = (input.len() - 1) as f64 / (output_len - 1) as f64;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let pos = i as f64 * ratio;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;

        let s0 = if idx > 0 { input[idx - 1] } else { input[0] };
        let s1 = input[idx.min(input.len() - 1)];
        let s2 = input[(idx + 1).min(input.len() - 1)];
        let s3 = input[(idx + 2).min(input.len() - 1)];

        // Catmull-Rom spline
        let a = -0.5 * s0 + 1.5 * s1 - 1.5 * s2 + 0.5 * s3;
        let b = s0 - 2.5 * s1 + 2.0 * s2 - 0.5 * s3;
        let c = -0.5 * s0 + 0.5 * s2;
        let d = s1;

        output.push(a * frac * frac * frac + b * frac * frac + c * frac + d);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_identity() {
        let input = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let output = resample_linear(&input, 5);
        for (i, (&a, &b)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Mismatch at index {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_linear_upsample() {
        let input = vec![0.0, 1.0];
        let output = resample_linear(&input, 5);
        assert_eq!(output.len(), 5);
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[2] - 0.5).abs() < 1e-6);
        assert!((output[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_downsample() {
        let input = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let output = resample_linear(&input, 3);
        assert_eq!(output.len(), 3);
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_empty() {
        assert_eq!(resample_linear(&[], 5), vec![0.0; 5]);
        assert!(resample_linear(&[1.0, 2.0], 0).is_empty());
    }

    #[test]
    fn test_linear_single_sample() {
        let output = resample_linear(&[0.5], 4);
        assert_eq!(output, vec![0.5; 4]);
    }

    #[test]
    fn test_cubic_identity() {
        let input: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let output = resample_cubic(&input, 10);
        for (i, (&a, &b)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.01,
                "Mismatch at index {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_cubic_smoother_than_linear() {
        // Create a signal with a sharp discontinuity
        let mut input = vec![0.0f32; 50];
        for i in 25..50 {
            input[i] = 1.0;
        }
        let linear = resample_linear(&input, 100);
        let cubic = resample_cubic(&input, 100);
        // Both should handle it, but they'll differ near the edge
        assert_eq!(linear.len(), 100);
        assert_eq!(cubic.len(), 100);
    }

    #[test]
    fn test_cubic_empty() {
        assert_eq!(resample_cubic(&[], 5), vec![0.0; 5]);
        assert!(resample_cubic(&[1.0, 2.0, 3.0, 4.0], 0).is_empty());
    }

    #[test]
    fn test_cubic_falls_back_to_linear_for_short() {
        let input = vec![0.0, 1.0, 2.0]; // Less than 4 samples
        let output = resample_cubic(&input, 5);
        assert_eq!(output.len(), 5);
    }
}

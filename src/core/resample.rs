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
            assert!(s >= -1.1 && s <= 1.1);
        }
    }
}

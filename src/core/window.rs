use std::f64::consts::PI;

/// Apply a Hann window to the given buffer in-place.
///
/// The Hann window is defined as: w(n) = 0.5 * (1 - cos(2π * n / (N-1)))
/// where N is the window length.
#[inline]
pub fn apply_hann_window(buffer: &mut [f32]) {
    let len = buffer.len();
    if len <= 1 {
        return;
    }
    let n_minus_1 = (len - 1) as f64;
    for (i, sample) in buffer.iter_mut().enumerate() {
        let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / n_minus_1).cos());
        *sample *= w as f32;
    }
}

/// Generate a Hann window of the given length.
pub fn hann_window(len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    if len == 1 {
        return vec![1.0];
    }
    let n_minus_1 = (len - 1) as f64;
    (0..len)
        .map(|i| (0.5 * (1.0 - (2.0 * PI * i as f64 / n_minus_1).cos())) as f32)
        .collect()
}

/// Apply a Blackman-Harris window to the given buffer in-place.
///
/// 4-term Blackman-Harris window with excellent sidelobe suppression (-92 dB).
#[inline]
pub fn apply_blackman_harris_window(buffer: &mut [f32]) {
    let len = buffer.len();
    if len <= 1 {
        return;
    }
    let n_minus_1 = (len - 1) as f64;
    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;
    for (i, sample) in buffer.iter_mut().enumerate() {
        let x = i as f64 / n_minus_1;
        let w = a0 - a1 * (2.0 * PI * x).cos() + a2 * (4.0 * PI * x).cos()
            - a3 * (6.0 * PI * x).cos();
        *sample *= w as f32;
    }
}

/// Generate a Blackman-Harris window of the given length.
pub fn blackman_harris_window(len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    if len == 1 {
        return vec![1.0];
    }
    let n_minus_1 = (len - 1) as f64;
    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;
    (0..len)
        .map(|i| {
            let x = i as f64 / n_minus_1;
            (a0 - a1 * (2.0 * PI * x).cos() + a2 * (4.0 * PI * x).cos()
                - a3 * (6.0 * PI * x).cos()) as f32
        })
        .collect()
}

/// Apply a Kaiser-Bessel window to the given buffer in-place.
///
/// Uses an approximation of the zeroth-order modified Bessel function I0.
/// The `beta` parameter controls the sidelobe level (typical: 5.0–12.0).
#[inline]
pub fn apply_kaiser_window(buffer: &mut [f32], beta: f64) {
    let len = buffer.len();
    if len <= 1 {
        return;
    }
    let n_minus_1 = (len - 1) as f64;
    let denom = bessel_i0(beta);
    for (i, sample) in buffer.iter_mut().enumerate() {
        let x = 2.0 * i as f64 / n_minus_1 - 1.0;
        let arg = beta * (1.0 - x * x).max(0.0).sqrt();
        let w = bessel_i0(arg) / denom;
        *sample *= w as f32;
    }
}

/// Generate a Kaiser window of the given length.
pub fn kaiser_window(len: usize, beta: f64) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }
    if len == 1 {
        return vec![1.0];
    }
    let n_minus_1 = (len - 1) as f64;
    let denom = bessel_i0(beta);
    (0..len)
        .map(|i| {
            let x = 2.0 * i as f64 / n_minus_1 - 1.0;
            let arg = beta * (1.0 - x * x).max(0.0).sqrt();
            (bessel_i0(arg) / denom) as f32
        })
        .collect()
}

/// Approximation of the zeroth-order modified Bessel function of the first kind I0(x).
///
/// Uses the series expansion with sufficient terms for audio accuracy.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half = x / 2.0;
    for k in 1..=25 {
        term *= (x_half / k as f64) * (x_half / k as f64);
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window_symmetry() {
        let w = hann_window(256);
        for i in 0..128 {
            assert!(
                (w[i] - w[255 - i]).abs() < 1e-6,
                "Hann window not symmetric at index {i}"
            );
        }
    }

    #[test]
    fn test_hann_window_endpoints() {
        let w = hann_window(256);
        assert!(w[0].abs() < 1e-6, "Hann window should start at 0");
        assert!(
            w[255].abs() < 1e-6,
            "Hann window should end at 0 (periodic)"
        );
    }

    #[test]
    fn test_hann_window_peak() {
        let w = hann_window(256);
        assert!(
            (w[127] - 1.0).abs() < 0.01,
            "Hann window peak should be near 1.0"
        );
    }

    #[test]
    fn test_hann_window_empty() {
        assert!(hann_window(0).is_empty());
    }

    #[test]
    fn test_hann_window_single() {
        let w = hann_window(1);
        assert_eq!(w, vec![1.0]);
    }

    #[test]
    fn test_blackman_harris_symmetry() {
        let w = blackman_harris_window(512);
        for i in 0..256 {
            assert!(
                (w[i] - w[511 - i]).abs() < 1e-6,
                "Blackman-Harris window not symmetric at index {i}"
            );
        }
    }

    #[test]
    fn test_blackman_harris_lower_sidelobes_than_hann() {
        // Blackman-Harris has lower sidelobes, so its endpoints should be closer to 0
        let bh = blackman_harris_window(256);
        let hann = hann_window(256);
        // At index 1, BH should be smaller than Hann
        assert!(bh[1] < hann[1], "BH should have tighter sidelobes");
    }

    #[test]
    fn test_kaiser_window_symmetry() {
        let w = kaiser_window(256, 8.0);
        for i in 0..128 {
            assert!(
                (w[i] - w[255 - i]).abs() < 1e-6,
                "Kaiser window not symmetric at index {i}"
            );
        }
    }

    #[test]
    fn test_kaiser_window_beta_effect() {
        let w_low = kaiser_window(256, 2.0);
        let w_high = kaiser_window(256, 12.0);
        // Higher beta = narrower main lobe = more tapered at edges
        assert!(
            w_high[1] < w_low[1],
            "Higher beta should produce smaller edge values"
        );
    }

    #[test]
    fn test_kaiser_window_peak() {
        let w = kaiser_window(256, 8.0);
        let max_val = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (max_val - 1.0).abs() < 1e-4,
            "Kaiser peak should be 1.0, got {max_val}"
        );
    }

    #[test]
    fn test_apply_hann_window_modifies_buffer() {
        let mut buf = vec![1.0f32; 256];
        apply_hann_window(&mut buf);
        // First element should be ~0 (windowed)
        assert!(buf[0].abs() < 1e-6);
        // Middle should be ~1.0
        assert!((buf[127] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_window_preserves_zeros() {
        let mut buf = vec![0.0f32; 256];
        apply_hann_window(&mut buf);
        for sample in &buf {
            assert_eq!(*sample, 0.0);
        }
    }

    #[test]
    fn test_bessel_i0_known_values() {
        // I0(0) = 1.0
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-10);
        // I0(1) ≈ 1.2660658...
        assert!((bessel_i0(1.0) - 1.2660658).abs() < 1e-5);
    }

    #[test]
    fn test_all_windows_non_negative() {
        for w in &[hann_window(256), blackman_harris_window(256), kaiser_window(256, 8.0)] {
            for (i, val) in w.iter().enumerate() {
                assert!(
                    *val >= -1e-6,
                    "Window value at {i} is negative: {val}"
                );
            }
        }
    }
}

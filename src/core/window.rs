//! Window functions for spectral analysis.
//!
//! Provides Hann, Blackman-Harris, and Kaiser-Bessel windows used by the
//! phase vocoder and transient detector.

use std::f64::consts::PI;

/// Blackman-Harris window coefficients (4-term).
const BH_A0: f64 = 0.35875;
const BH_A1: f64 = 0.48829;
const BH_A2: f64 = 0.14128;
const BH_A3: f64 = 0.01168;

/// Window function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    Hann,
    BlackmanHarris,
    Kaiser(u32), // beta parameter scaled by 100 (e.g., 800 = 8.0)
}

/// Generates a window function of the specified type and size.
pub fn generate_window(window_type: WindowType, size: usize) -> Vec<f32> {
    match window_type {
        WindowType::Hann => hann_window(size),
        WindowType::BlackmanHarris => blackman_harris_window(size),
        WindowType::Kaiser(beta_100) => kaiser_window(size, beta_100 as f64 / 100.0),
    }
}

/// Returns `Some(trivial_window)` for degenerate sizes (0 or 1), or `None`
/// to indicate the caller should compute the full window.
#[inline]
fn trivial_window(size: usize) -> Option<Vec<f32>> {
    match size {
        0 => Some(vec![]),
        1 => Some(vec![1.0]),
        _ => None,
    }
}

/// Generates a Hann window.
#[inline]
fn hann_window(size: usize) -> Vec<f32> {
    if let Some(w) = trivial_window(size) {
        return w;
    }
    let n = size as f64;
    (0..size)
        .map(|i| {
            let x = (2.0 * PI * i as f64) / (n - 1.0);
            (0.5 * (1.0 - x.cos())) as f32
        })
        .collect()
}

/// Generates a Blackman-Harris window.
#[inline]
fn blackman_harris_window(size: usize) -> Vec<f32> {
    if let Some(w) = trivial_window(size) {
        return w;
    }
    let n = size as f64;
    (0..size)
        .map(|i| {
            let x = i as f64 / (n - 1.0);
            let w = BH_A0 - BH_A1 * (2.0 * PI * x).cos() + BH_A2 * (4.0 * PI * x).cos()
                - BH_A3 * (6.0 * PI * x).cos();
            w as f32
        })
        .collect()
}

/// Generates a Kaiser window using the zeroth-order modified Bessel function.
#[inline]
fn kaiser_window(size: usize, beta: f64) -> Vec<f32> {
    if let Some(w) = trivial_window(size) {
        return w;
    }
    let n = size as f64;
    let denom = bessel_i0(beta);
    (0..size)
        .map(|i| {
            let x = 2.0 * i as f64 / (n - 1.0) - 1.0;
            let arg = beta * (1.0 - x * x).max(0.0).sqrt();
            (bessel_i0(arg) / denom) as f32
        })
        .collect()
}

/// Maximum number of series terms for Bessel I0 convergence.
const BESSEL_MAX_TERMS: usize = 30;
/// Relative convergence threshold for Bessel I0 series.
const BESSEL_CONVERGENCE: f64 = 1e-15;

/// Zeroth-order modified Bessel function of the first kind.
/// Computed via series expansion.
#[inline]
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half = x / 2.0;
    for k in 1..BESSEL_MAX_TERMS {
        term *= (x_half / k as f64) * (x_half / k as f64);
        sum += term;
        if term < BESSEL_CONVERGENCE * sum {
            break;
        }
    }
    sum
}

/// Applies a window function to a slice in-place.
#[inline]
pub fn apply_window(data: &mut [f32], window: &[f32]) {
    for (sample, &w) in data.iter_mut().zip(window.iter()) {
        *sample *= w;
    }
}

/// Applies a window function and returns a new vector.
#[inline]
pub fn apply_window_copy(data: &[f32], window: &[f32]) -> Vec<f32> {
    data.iter()
        .zip(window.iter())
        .map(|(&d, &w)| d * w)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window_properties() {
        let w = hann_window(1024);
        assert_eq!(w.len(), 1024);
        // First and last should be near zero
        assert!(w[0].abs() < 1e-6);
        assert!(w[1023].abs() < 1e-6);
        // Middle should be near 1.0
        assert!((w[512] - 1.0).abs() < 0.01);
        // Symmetric
        for i in 0..512 {
            assert!((w[i] - w[1023 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_blackman_harris_properties() {
        let w = blackman_harris_window(1024);
        assert_eq!(w.len(), 1024);
        // Should have good sidelobe suppression (first/last very small)
        assert!(w[0] < 0.01);
        assert!(w[1023] < 0.01);
        // Symmetric
        for i in 0..512 {
            assert!((w[i] - w[1023 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_kaiser_window_properties() {
        let w = kaiser_window(1024, 8.0);
        assert_eq!(w.len(), 1024);
        // Middle should be peak
        let mid = w[512];
        for &v in &w {
            assert!(v <= mid + 1e-6);
        }
        // Symmetric
        for i in 0..512 {
            assert!((w[i] - w[1023 - i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_empty_window() {
        assert!(hann_window(0).is_empty());
        assert!(blackman_harris_window(0).is_empty());
        assert!(kaiser_window(0, 8.0).is_empty());
    }

    #[test]
    fn test_single_sample_window() {
        assert_eq!(hann_window(1), vec![1.0]);
        assert_eq!(blackman_harris_window(1), vec![1.0]);
        assert_eq!(kaiser_window(1, 8.0), vec![1.0]);
    }

    #[test]
    fn test_apply_window() {
        let window = vec![0.5, 1.0, 0.5];
        let mut data = vec![2.0, 3.0, 4.0];
        apply_window(&mut data, &window);
        assert_eq!(data, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn test_generate_window_dispatch() {
        let h = generate_window(WindowType::Hann, 256);
        assert_eq!(h.len(), 256);
        let bh = generate_window(WindowType::BlackmanHarris, 256);
        assert_eq!(bh.len(), 256);
        let k = generate_window(WindowType::Kaiser(800), 256);
        assert_eq!(k.len(), 256);
    }
}

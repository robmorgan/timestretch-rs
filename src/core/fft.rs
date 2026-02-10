//! FFT-related constants and utilities shared across the crate.

use rustfft::num_complex::Complex;

/// Zero-valued complex number, used for FFT buffer initialization.
pub const COMPLEX_ZERO: Complex<f32> = Complex::new(0.0, 0.0);

/// Minimum window sum (as a fraction of max) to prevent amplification
/// in low-overlap regions during overlap-add normalization.
pub const WINDOW_SUM_FLOOR_RATIO: f32 = 0.1;

/// Absolute floor for window sum normalization to prevent division by zero.
pub const WINDOW_SUM_EPSILON: f32 = 1e-6;

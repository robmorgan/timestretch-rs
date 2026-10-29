//! FFT-related constants and utilities shared across the crate.

use rustfft::num_complex::Complex;

/// Zero-valued complex number, used for FFT buffer initialization.
pub const COMPLEX_ZERO: Complex<f32> = Complex::new(0.0, 0.0);

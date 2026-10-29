use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

/// Split-packed convention: DC at 0, Nyquist at N/2, real(k) at k
/// and imag(k) at N/2+k. Forward is conventional DFT; inverse is unnormalized.
pub(crate) struct RustFft {
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    work: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}
impl RustFft {
    pub fn new(size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(size);
        let inverse = planner.plan_fft_inverse(size);
        let scratch = vec![
            Complex::default();
            forward
                .get_inplace_scratch_len()
                .max(inverse.get_inplace_scratch_len())
        ];
        Self {
            forward,
            inverse,
            work: vec![Complex::default(); size],
            scratch,
        }
    }
    pub fn forward(&mut self, data: &mut [f32]) {
        self.pack(data);
        self.forward_core();
        self.unpack_spectrum(data);
    }
    pub(super) fn pack(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.work.len());
        for (v, &x) in self.work.iter_mut().zip(data.iter()) {
            *v = Complex::new(x, 0.0);
        }
    }
    pub(super) fn forward_core(&mut self) {
        self.forward
            .process_with_scratch(&mut self.work, &mut self.scratch);
    }
    pub(super) fn unpack_spectrum(&self, data: &mut [f32]) {
        let h = self.work.len() / 2;
        data[0] = self.work[0].re;
        data[h] = self.work[h].re;
        for k in 1..h {
            data[k] = self.work[k].re;
            data[h + k] = self.work[k].im;
        }
    }
    pub fn inverse(&mut self, data: &mut [f32]) {
        self.pack_spectrum(data);
        self.inverse_core();
        self.unpack(data);
    }
    pub(super) fn pack_spectrum(&mut self, data: &[f32]) {
        let n = self.work.len();
        assert_eq!(data.len(), n);
        let h = n / 2;
        self.work[0] = Complex::new(data[0], 0.0);
        self.work[h] = Complex::new(data[h], 0.0);
        for k in 1..h {
            self.work[k] = Complex::new(data[k], data[h + k]);
            self.work[n - k] = self.work[k].conj();
        }
    }
    pub(super) fn inverse_core(&mut self) {
        self.inverse
            .process_with_scratch(&mut self.work, &mut self.scratch);
    }
    pub(super) fn unpack(&self, data: &mut [f32]) {
        for (x, v) in data.iter_mut().zip(&self.work) {
            *x = v.re;
        }
    }
}

/// Backend is captured at construction; evaluation FFTs do not use this adapter.
pub(crate) enum PackedFft {
    Rust(RustFft),
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    Apple(timestretch_accelerate::RealFft),
}
impl PackedFft {
    pub fn new(size: usize, _numerics: crate::Numerics) -> Self {
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        if _numerics.apple_fft() {
            return Self::Apple(
                timestretch_accelerate::RealFft::new(size)
                    .expect("valid FFT geometry and available system plan"),
            );
        }
        Self::Rust(RustFft::new(size))
    }
    pub fn forward(&mut self, data: &mut [f32]) {
        match self {
            Self::Rust(f) => f.forward(data),
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            Self::Apple(f) => f.forward(data),
        }
    }
    pub fn inverse(&mut self, data: &mut [f32]) {
        match self {
            Self::Rust(f) => f.inverse(data),
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            Self::Apple(f) => f.inverse(data),
        }
    }
}

use crate::memory::HeapBytes;
impl HeapBytes for RustFft {
    fn heap_bytes(&self) -> usize {
        self.work.heap_bytes() + self.scratch.heap_bytes()
    }
}
impl HeapBytes for PackedFft {
    fn heap_bytes(&self) -> usize {
        match self {
            Self::Rust(f) => f.heap_bytes(),
            #[cfg(all(feature = "accelerate", target_os = "macos"))]
            Self::Apple(f) => f.owned_buffer_bytes(),
        }
    }
}

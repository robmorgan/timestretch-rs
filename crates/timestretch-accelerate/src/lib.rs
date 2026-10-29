//! Safe, owned adapters for Apple's packed real FFT and vector math.
//! The main timestretch crate retains forbid(unsafe_code). No global plans,
//! lazy initialization or allocation in processing calls. Only macOS builds.
#![cfg(target_os = "macos")]
#![deny(unsafe_op_in_unsafe_fn)]
use std::{ffi::c_void, ptr::NonNull};
#[repr(C)]
struct Split {
    re: *mut f32,
    im: *mut f32,
}
#[repr(C)]
struct Complex {
    re: f32,
    im: f32,
}
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vDSP_create_fftsetup(log: usize, radix: i32) -> *mut c_void;
    fn vDSP_destroy_fftsetup(plan: *mut c_void);
    fn vDSP_fft_zrip(
        plan: *mut c_void,
        data: *const Split,
        stride: isize,
        log: usize,
        direction: i32,
    );
    fn vDSP_ctoz(
        input: *const Complex,
        stride: isize,
        out: *const Split,
        out_stride: isize,
        n: usize,
    );
    fn vDSP_ztoc(
        input: *const Split,
        stride: isize,
        out: *mut Complex,
        out_stride: isize,
        n: usize,
    );
    fn vDSP_vsmul(
        input: *const f32,
        stride: isize,
        gain: *const f32,
        out: *mut f32,
        out_stride: isize,
        n: usize,
    );
    fn vDSP_vdiv(
        denominator: *const f32,
        denominator_stride: isize,
        numerator: *const f32,
        numerator_stride: isize,
        out: *mut f32,
        out_stride: isize,
        n: usize,
    );
    fn vDSP_zvphas(input: *const Split, stride: isize, out: *mut f32, out_stride: isize, n: usize);
    fn vDSP_zvabs(input: *const Split, stride: isize, out: *mut f32, out_stride: isize, n: usize);
    fn vvsincosf(s: *mut f32, c: *mut f32, input: *const f32, n: *const i32);
}
// Match the reference FFT workspace alignment. Accelerate may choose different
// arithmetic kernels for differently aligned split arrays.
#[repr(C, align(64))]
#[derive(Clone)]
struct AlignedChunk([f32; 16]);
struct AlignedFloats {
    chunks: Vec<AlignedChunk>,
    len: usize,
}
impl AlignedFloats {
    fn new(len: usize) -> Self {
        Self {
            chunks: vec![AlignedChunk([0.; 16]); len.div_ceil(16)],
            len,
        }
    }
}
impl std::ops::Deref for AlignedFloats {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        // SAFETY: repr(C) chunks contain exactly 16 contiguous initialized f32s,
        // size and alignment are both 64. Logical length never exceeds storage.
        unsafe { std::slice::from_raw_parts(self.chunks.as_ptr().cast(), self.len) }
    }
}
impl std::ops::DerefMut for AlignedFloats {
    fn deref_mut(&mut self) -> &mut [f32] {
        // SAFETY: same layout as Deref, with exclusive access to the allocation.
        unsafe { std::slice::from_raw_parts_mut(self.chunks.as_mut_ptr().cast(), self.len) }
    }
}
/// One independently owned, movable FFT plan and its preallocated storage.
pub struct RealFft {
    plan: NonNull<c_void>,
    log: usize,
    re: AlignedFloats,
    im: AlignedFloats,
}
// SAFETY: the plan and work vectors are exclusively owned. All execution needs
// &mut self; transferring the owner cannot race any use of this plan.
unsafe impl Send for RealFft {}
impl RealFft {
    /// Aligned workspaces only; the framework plan's allocation is opaque.
    pub fn owned_buffer_bytes(&self) -> usize {
        (self.re.chunks.capacity() + self.im.chunks.capacity())
            * std::mem::size_of::<AlignedChunk>()
    }

    pub fn new(n: usize) -> Result<Self, &'static str> {
        if !n.is_power_of_two() || !(4..=1 << 20).contains(&n) {
            return Err("FFT length must be a power of two in 4..=1048576");
        }
        let log = n.trailing_zeros() as usize;
        // SAFETY: valid radix-2 size. Null is handled, and Drop releases exactly once.
        let plan = NonNull::new(unsafe { vDSP_create_fftsetup(log, 0) })
            .ok_or("Accelerate FFT setup failed")?;
        Ok(Self {
            plan,
            log,
            re: AlignedFloats::new(n / 2),
            im: AlignedFloats::new(n / 2),
        })
    }
    fn split(&mut self) -> Split {
        Split {
            re: self.re.as_mut_ptr(),
            im: self.im.as_mut_ptr(),
        }
    }
    /// Conventional real forward transform: DC at 0, Nyquist at n/2.
    pub fn forward(&mut self, data: &mut [f32]) {
        self.pack(data);
        self.transform(false);
        self.unpack_spectrum(data);
    }
    /// Unnormalized inverse: inverse(forward(x)) = n*x.
    pub fn inverse(&mut self, data: &mut [f32]) {
        self.pack_spectrum(data);
        self.transform(true);
        self.unpack(data);
    }
    // These steps are public to measure packing/scaling separately from execution.
    pub fn pack(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.re.len() * 2);
        let split = self.split();
        // SAFETY: n/2 Complex pairs fit data; stride 2 is vDSP's interleaved
        // convention. Both split vectors have n/2 distinct writable elements.
        unsafe {
            vDSP_ctoz(data.as_ptr().cast(), 2, &split, 1, self.re.len());
        }
    }
    pub fn pack_spectrum(&mut self, data: &[f32]) {
        assert_eq!(data.len(), self.re.len() * 2);
        let h = self.re.len();
        self.re.copy_from_slice(&data[..h]);
        self.im.copy_from_slice(&data[h..]);
    }
    pub fn transform(&mut self, inverse: bool) {
        let split = self.split();
        // SAFETY: plan size matches live split vectors; &mut self excludes races.
        unsafe {
            vDSP_fft_zrip(
                self.plan.as_ptr(),
                &split,
                1,
                self.log,
                if inverse { -1 } else { 1 },
            );
        }
    }
    pub fn unpack_spectrum(&self, data: &mut [f32]) {
        assert_eq!(data.len(), self.re.len() * 2);
        let h = self.re.len();
        data[..h].copy_from_slice(&self.re);
        data[h..].copy_from_slice(&self.im);
        // SAFETY: exact alias is supported; stride 1 and count span the output.
        unsafe {
            vDSP_vsmul(data.as_ptr(), 1, &0.5, data.as_mut_ptr(), 1, data.len());
        }
    }
    pub fn unpack(&mut self, data: &mut [f32]) {
        assert_eq!(data.len(), self.re.len() * 2);
        let split = self.split();
        // SAFETY: n/2 complex slots fit n f32 values, with no alias to split.
        unsafe {
            vDSP_ztoc(&split, 1, data.as_mut_ptr().cast(), 2, self.re.len());
        }
    }
}
impl Drop for RealFft {
    fn drop(&mut self) {
        // SAFETY: unique non-null owned plan.
        unsafe {
            vDSP_destroy_fftsetup(self.plan.as_ptr());
        }
    }
}
/// Phase/magnitude of split-packed input; Nyquist is not DC's imaginary part.
pub fn polar(packed: &mut [f32], phase: &mut [f32], magnitude: &mut [f32]) {
    assert!(packed.len() >= 2 && packed.len() % 2 == 0);
    let h = packed.len() / 2;
    assert_eq!(phase.len(), h);
    assert_eq!(magnitude.len(), h);
    let nyquist = packed[h];
    packed[h] = 0.;
    let (re, im) = packed.split_at_mut(h);
    let split = Split {
        re: re.as_mut_ptr(),
        im: im.as_mut_ptr(),
    };
    // SAFETY: split and output slices are disjoint valid h-element arrays. The
    // temporary Nyquist zero is restored before returning; neither call unwinds.
    unsafe {
        vDSP_zvphas(&split, 1, phase.as_mut_ptr(), 1, h);
        vDSP_zvabs(&split, 1, magnitude.as_mut_ptr(), 1, h);
    }
    packed[h] = nyquist;
}
pub fn sin_cos(input: &[f32], sin: &mut [f32], cos: &mut [f32]) {
    assert_eq!(input.len(), sin.len());
    assert_eq!(input.len(), cos.len());
    let n = i32::try_from(input.len()).expect("vForce length exceeds i32");
    if n == 0 {
        return;
    }
    // SAFETY: checked matching lengths and disjoint borrowed output slices.
    unsafe {
        vvsincosf(sin.as_mut_ptr(), cos.as_mut_ptr(), input.as_ptr(), &n);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packing_scaling_and_round_trip() {
        for n in [1024, 2048, 4096, 8192] {
            let mut p = RealFft::new(n).unwrap();
            for kind in 0..3 {
                let mut x: Vec<f32> = (0..n)
                    .map(|i| match kind {
                        0 => 1.,
                        1 => {
                            if i % 2 == 0 {
                                1.
                            } else {
                                -1.
                            }
                        }
                        _ => {
                            if i == 0 {
                                1.
                            } else {
                                0.
                            }
                        }
                    })
                    .collect();
                let original = x.clone();
                p.forward(&mut x);
                if kind == 0 {
                    assert_eq!(x[0], n as f32);
                    assert_eq!(x[n / 2], 0.);
                }
                if kind == 1 {
                    assert_eq!(x[0], 0.);
                    assert_eq!(x[n / 2], n as f32);
                }
                p.inverse(&mut x);
                for (x, y) in x.iter().zip(original) {
                    assert!((x / n as f32 - y).abs() < 2e-6);
                }
            }
        }
    }
    #[test]
    fn polar_restores_nyquist() {
        let mut x = [-2., 3., 77., 4.];
        let mut phase = [0.; 2];
        let mut mag = [0.; 2];
        polar(&mut x, &mut phase, &mut mag);
        assert_eq!(x[2], 77.);
        assert_eq!(mag, [2., 5.]);
        assert!((phase[1] - 4f32.atan2(3.)).abs() < 3e-7);
    }
    #[test]
    fn fft_workspace_alignment_and_input_offsets_are_stable() {
        assert_eq!(std::mem::size_of::<AlignedChunk>(), 64);
        for n in [4, 16, 1024, 2048, 4096, 8192] {
            let mut fft = RealFft::new(n).unwrap();
            assert_eq!(fft.re.as_ptr() as usize % 64, 0);
            assert_eq!(fft.im.as_ptr() as usize % 64, 0);
            let source: Vec<_> = (0..n).map(|i| (i as f32 * 0.031).sin()).collect();
            let mut expected = source.clone();
            fft.forward(&mut expected);
            for offset in 0..16 {
                let mut buffer = vec![0.; n + offset];
                let data = &mut buffer[offset..];
                data.copy_from_slice(&source);
                fft.forward(data);
                assert!(
                    data.iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "size {n}, offset {offset}"
                );
            }
        }
    }
    #[test]
    fn rejects_bad_sizes() {
        for n in [0, 1, 3, 17, 1 << 21] {
            assert!(RealFft::new(n).is_err());
        }
    }
}

/// Optional QA allocation observer from tools/backend_allocations.c. Absent
/// during ordinary measurements. The injected observer counts the calling
/// thread's C allocator calls, including framework-internal allocations.
#[derive(Clone, Copy)]
pub struct AllocationProbe {
    begin: unsafe extern "C" fn(),
    end: unsafe extern "C" fn() -> usize,
}
impl AllocationProbe {
    pub fn load() -> Option<Self> {
        unsafe extern "C" {
            fn dlsym(handle: *mut c_void, name: *const std::ffi::c_char) -> *mut c_void;
        }
        // SAFETY: macOS RTLD_DEFAULT is -2; the optional symbols have the exact
        // ABI declared in our QA interposer, which stays loaded for the process.
        unsafe {
            let begin = dlsym(
                (-2isize) as *mut c_void,
                c"timestretch_qa_alloc_begin".as_ptr(),
            );
            let end = dlsym(
                (-2isize) as *mut c_void,
                c"timestretch_qa_alloc_end".as_ptr(),
            );
            if begin.is_null() || end.is_null() {
                return None;
            }
            Some(Self {
                begin: std::mem::transmute::<*mut c_void, unsafe extern "C" fn()>(begin),
                end: std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> usize>(end),
            })
        }
    }
    pub fn begin(self) {
        // SAFETY: resolved typed function, thread-local state.
        unsafe { (self.begin)() }
    }
    pub fn end(self) -> usize {
        // SAFETY: resolved typed function, thread-local state.
        unsafe { (self.end)() }
    }
}

/// Reference vector division boundary, used for overlap normalization.
pub fn reciprocal(denominator: &[f32], out: &mut [f32]) {
    assert_eq!(denominator.len(), out.len());
    out.fill(1.);
    if out.is_empty() {
        return;
    }
    // SAFETY: equal-length disjoint inputs; exact numerator/output alias is
    // supported by vDSP_vdiv. Its denominator is the first API argument.
    unsafe {
        vDSP_vdiv(
            denominator.as_ptr(),
            1,
            out.as_ptr(),
            1,
            out.as_mut_ptr(),
            1,
            out.len(),
        );
    }
}

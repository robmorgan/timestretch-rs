//! Internal buffer accounting shared by the facade. Opaque FFT plans are excluded.
#[doc(hidden)]
pub trait HeapBytes {
    fn heap_bytes(&self) -> usize;
}
macro_rules! scalar { ($($t:ty),*) => { $(impl HeapBytes for $t { fn heap_bytes(&self)->usize {0} })* }; }
scalar!(f32, f64, i32, usize, rustfft::num_complex::Complex<f32>);
impl<T: HeapBytes> HeapBytes for Vec<T> {
    fn heap_bytes(&self) -> usize {
        self.capacity() * std::mem::size_of::<T>()
            + self.iter().map(HeapBytes::heap_bytes).sum::<usize>()
    }
}
impl<T: HeapBytes, const N: usize> HeapBytes for [T; N] {
    fn heap_bytes(&self) -> usize {
        self.iter().map(HeapBytes::heap_bytes).sum()
    }
}
impl<T: HeapBytes> HeapBytes for Box<T> {
    fn heap_bytes(&self) -> usize {
        std::mem::size_of::<T>() + (**self).heap_bytes()
    }
}

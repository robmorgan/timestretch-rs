//! Allocation-free kernel faults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StretchError {
    InvalidFormat(&'static str),
    InvalidRatio(&'static str),
    NonFiniteInput,
    BufferOverflow {
        buffer: &'static str,
        requested: usize,
        available: usize,
    },
    InvalidState(&'static str),
}
impl std::fmt::Display for StretchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for StretchError {}

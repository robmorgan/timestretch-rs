//! Error types for the timestretch crate.

use std::fmt;

/// Errors that can occur during time stretching.
///
/// Implements [`std::error::Error`] and [`Display`](std::fmt::Display) for
/// easy integration with error-handling crates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StretchError {
    /// Invalid audio format or parameters.
    InvalidFormat(String),
    /// Invalid stretch ratio.
    InvalidRatio(String),
    /// I/O error.
    IoError(String),
    /// Input too short for the given parameters.
    InputTooShort { provided: usize, minimum: usize },
    /// BPM detection failed.
    BpmDetectionFailed(String),
}

impl fmt::Display for StretchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StretchError::InvalidFormat(msg) => write!(f, "invalid format: {}", msg),
            StretchError::InvalidRatio(msg) => write!(f, "invalid stretch ratio: {}", msg),
            StretchError::IoError(msg) => write!(f, "I/O error: {}", msg),
            StretchError::InputTooShort { provided, minimum } => {
                write!(
                    f,
                    "input too short: {} samples provided, {} required",
                    provided, minimum
                )
            }
            StretchError::BpmDetectionFailed(msg) => {
                write!(f, "BPM detection failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for StretchError {}

impl From<std::io::Error> for StretchError {
    fn from(err: std::io::Error) -> Self {
        StretchError::IoError(err.to_string())
    }
}

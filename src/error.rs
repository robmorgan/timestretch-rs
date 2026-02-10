//! Error types for the timestretch crate.

use std::fmt;

/// Errors that can occur during time stretching.
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
        }
    }
}

impl std::error::Error for StretchError {}

impl From<std::io::Error> for StretchError {
    fn from(err: std::io::Error) -> Self {
        StretchError::IoError(err.to_string())
    }
}

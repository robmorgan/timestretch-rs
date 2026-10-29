//! Numerical provider chosen explicitly at construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Numerics {
    #[default]
    Portable,
    Accelerate,
    #[cfg(feature = "qa")]
    AppleFft,
    #[cfg(feature = "qa")]
    AppleMath,
}
impl Numerics {
    pub fn available(self) -> bool {
        self == Self::Portable || cfg!(all(feature = "accelerate", target_os = "macos"))
    }
    #[cfg(any(feature = "qa", all(feature = "accelerate", target_os = "macos")))]
    pub(crate) fn apple_fft(self) -> bool {
        match self {
            Self::Accelerate => true,
            #[cfg(feature = "qa")]
            Self::AppleFft => true,
            _ => false,
        }
    }
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    pub(crate) fn apple_math(self) -> bool {
        match self {
            Self::Accelerate => true,
            #[cfg(feature = "qa")]
            Self::AppleMath => true,
            _ => false,
        }
    }
}

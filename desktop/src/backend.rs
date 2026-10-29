//! Display labels for the library's production backend selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackendChoice {
    #[default]
    Auto,
    Rust,
    #[cfg(target_os = "macos")]
    Apple,
}
impl BackendChoice {
    pub const ALL: &'static [Self] = &[
        Self::Auto,
        Self::Rust,
        #[cfg(target_os = "macos")]
        Self::Apple,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Automatic",
            Self::Rust => "Portable Rust",
            #[cfg(target_os = "macos")]
            Self::Apple => "Apple Accelerate",
        }
    }
    pub fn backend(self) -> timestretch::Backend {
        match self {
            Self::Auto => timestretch::Backend::Auto,
            Self::Rust => timestretch::Backend::Portable,
            #[cfg(target_os = "macos")]
            Self::Apple => timestretch::Backend::Accelerate,
        }
    }
}

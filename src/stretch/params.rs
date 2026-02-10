//! Internal algorithm parameters derived from user-facing [`StretchParams`].

use crate::core::types::StretchParams;

/// Minimum valid stretch/pitch ratio.
pub const RATIO_MIN: f64 = 0.01;
/// Maximum valid stretch/pitch ratio.
pub const RATIO_MAX: f64 = 100.0;
/// Minimum valid FFT size.
const FFT_SIZE_MIN: usize = 256;
/// Valid sample rate range.
const SAMPLE_RATE_MIN: u32 = 8000;
const SAMPLE_RATE_MAX: u32 = 192000;

/// Validates stretch parameters and returns an error message if invalid.
pub fn validate_params(params: &StretchParams) -> Result<(), String> {
    if !(RATIO_MIN..=RATIO_MAX).contains(&params.stretch_ratio) {
        return Err(format!(
            "Stretch ratio must be between {} and {}, got {}",
            RATIO_MIN, RATIO_MAX, params.stretch_ratio
        ));
    }
    if params.fft_size < FFT_SIZE_MIN {
        return Err(format!(
            "FFT size too small: {} (min {})",
            params.fft_size, FFT_SIZE_MIN
        ));
    }
    if !params.fft_size.is_power_of_two() {
        return Err(format!(
            "FFT size must be a power of two, got {}",
            params.fft_size
        ));
    }
    if params.hop_size == 0 || params.hop_size > params.fft_size {
        return Err(format!(
            "Hop size {} must be between 1 and FFT size {}",
            params.hop_size, params.fft_size
        ));
    }
    if !(SAMPLE_RATE_MIN..=SAMPLE_RATE_MAX).contains(&params.sample_rate) {
        return Err(format!(
            "Sample rate {} out of range ({}-{})",
            params.sample_rate, SAMPLE_RATE_MIN, SAMPLE_RATE_MAX
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::EdmPreset;

    #[test]
    fn test_validate_params_valid() {
        let params = StretchParams::new(1.0);
        assert!(validate_params(&params).is_ok());
    }

    #[test]
    fn test_validate_params_bad_ratio() {
        let mut params = StretchParams::new(0.0);
        assert!(validate_params(&params).is_err());

        params.stretch_ratio = -1.0;
        assert!(validate_params(&params).is_err());

        params.stretch_ratio = 200.0;
        assert!(validate_params(&params).is_err());
    }

    #[test]
    fn test_validate_params_bad_fft() {
        let mut params = StretchParams::new(1.0);
        params.fft_size = 100; // Not power of two
        assert!(validate_params(&params).is_err());

        params.fft_size = 128;
        assert!(validate_params(&params).is_err()); // Too small

        params.fft_size = 256;
        params.hop_size = 0;
        assert!(validate_params(&params).is_err());
    }

    #[test]
    fn test_preset_descriptions() {
        // Just verify all presets have descriptions
        let _ = EdmPreset::DjBeatmatch.description();
        let _ = EdmPreset::HouseLoop.description();
        let _ = EdmPreset::Halftime.description();
        let _ = EdmPreset::Ambient.description();
        let _ = EdmPreset::VocalChop.description();
    }
}

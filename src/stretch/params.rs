//! Internal algorithm parameters derived from user-facing [`StretchParams`].

use crate::core::types::{EdmPreset, StretchParams};

/// Returns a description of the given preset's characteristics.
pub fn preset_description(preset: EdmPreset) -> &'static str {
    match preset {
        EdmPreset::DjBeatmatch => {
            "Small tempo adjustments (±1-8%). Maximum transparency, minimal artifacts."
        }
        EdmPreset::HouseLoop => {
            "General purpose for house/techno loops. Balanced quality and performance."
        }
        EdmPreset::Halftime => {
            "Halftime effect (2x stretch). Preserves kick punch and transient clarity."
        }
        EdmPreset::Ambient => {
            "Extreme stretch (2x-4x) for ambient transitions. Smooth, artifact-free."
        }
        EdmPreset::VocalChop => {
            "Optimized for vocal chops and one-shots. Preserves formant character."
        }
    }
}

/// Validates stretch parameters and returns an error message if invalid.
pub fn validate_params(params: &StretchParams) -> Result<(), String> {
    if params.stretch_ratio <= 0.0 {
        return Err(format!(
            "Stretch ratio must be positive, got {}",
            params.stretch_ratio
        ));
    }
    if params.stretch_ratio > 100.0 {
        return Err(format!(
            "Stretch ratio too large: {} (max 100.0)",
            params.stretch_ratio
        ));
    }
    if params.stretch_ratio < 0.01 {
        return Err(format!(
            "Stretch ratio too small: {} (min 0.01)",
            params.stretch_ratio
        ));
    }
    if params.fft_size < 256 {
        return Err(format!("FFT size too small: {} (min 256)", params.fft_size));
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
    if params.sample_rate < 8000 || params.sample_rate > 192000 {
        return Err(format!(
            "Sample rate {} out of range (8000-192000)",
            params.sample_rate
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let _ = preset_description(EdmPreset::DjBeatmatch);
        let _ = preset_description(EdmPreset::HouseLoop);
        let _ = preset_description(EdmPreset::Halftime);
        let _ = preset_description(EdmPreset::Ambient);
        let _ = preset_description(EdmPreset::VocalChop);
    }
}

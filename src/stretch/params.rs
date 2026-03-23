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
    if !(0.0..=1.0).contains(&params.transient_lookahead_threshold_relax) {
        return Err(format!(
            "Transient lookahead threshold relax must be in [0,1], got {}",
            params.transient_lookahead_threshold_relax
        ));
    }
    if !(0.0..=1.0).contains(&params.transient_lookahead_peak_retain_ratio) {
        return Err(format!(
            "Transient lookahead peak retain ratio must be in [0,1], got {}",
            params.transient_lookahead_peak_retain_ratio
        ));
    }
    if !params.transient_strong_spike_bypass_multiplier.is_finite()
        || params.transient_strong_spike_bypass_multiplier < 1.0
    {
        return Err(format!(
            "Transient strong-spike bypass multiplier must be finite and >=1, got {}",
            params.transient_strong_spike_bypass_multiplier
        ));
    }
    if params.transient_lookahead_frames > 16 {
        return Err(format!(
            "Transient lookahead frames must be <= 16, got {}",
            params.transient_lookahead_frames
        ));
    }
    let policy = params.transient_threshold_policy.sanitized();
    if policy != params.transient_threshold_policy {
        return Err(format!(
            "Transient threshold policy is invalid: {:?}",
            params.transient_threshold_policy
        ));
    }
    if !params.residual_mix.is_finite() || !(0.0..=1.5).contains(&params.residual_mix) {
        return Err(format!(
            "Residual mix must be finite and in [0,1.5], got {}",
            params.residual_mix
        ));
    }
    if !params.envelope_strength.is_finite() || !(0.0..=2.0).contains(&params.envelope_strength) {
        return Err(format!(
            "Envelope strength must be finite and in [0,2], got {}",
            params.envelope_strength
        ));
    }
    if params.envelope_order == 0 {
        return Err("Envelope order must be >= 1".to_string());
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
    fn test_validate_params_bad_transient_lookahead() {
        let mut params = StretchParams::new(1.0);
        params.transient_lookahead_threshold_relax = 1.2;
        assert!(validate_params(&params).is_err());

        params = StretchParams::new(1.0);
        params.transient_lookahead_peak_retain_ratio = -0.1;
        assert!(validate_params(&params).is_err());

        params = StretchParams::new(1.0);
        params.transient_strong_spike_bypass_multiplier = 0.9;
        assert!(validate_params(&params).is_err());

        params = StretchParams::new(1.0);
        params.transient_lookahead_frames = 17;
        assert!(validate_params(&params).is_err());

        params = StretchParams::new(1.0);
        params.transient_threshold_policy.median_window_frames = 0;
        assert!(validate_params(&params).is_err());

        params = StretchParams::new(1.0);
        params.residual_mix = f32::INFINITY;
        assert!(validate_params(&params).is_err());

        params = StretchParams::new(1.0);
        params.envelope_strength = -0.1;
        assert!(validate_params(&params).is_err());

        params = StretchParams::new(1.0);
        params.envelope_order = 0;
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

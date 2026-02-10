use crate::core::types::EdmPreset;

/// Internal algorithm parameters derived from user-facing StretchParams.
#[derive(Debug, Clone)]
pub struct AlgorithmParams {
    pub fft_size: usize,
    pub hop_size: usize,
    pub stretch_ratio: f64,
    pub sample_rate: u32,
    /// WSOLA segment size in samples.
    pub wsola_segment_size: usize,
    /// WSOLA overlap as fraction of segment size (0.0 to 1.0).
    pub wsola_overlap: f32,
    /// WSOLA search range in samples.
    pub wsola_search_range: usize,
    /// Sub-bass cutoff frequency for phase locking.
    pub sub_bass_cutoff: f32,
    /// Crossfade length in samples for hybrid transitions.
    pub crossfade_len: usize,
    /// Transient detection sensitivity.
    pub transient_sensitivity: f32,
    /// Whether to use hybrid algorithm.
    pub use_hybrid: bool,
}

impl AlgorithmParams {
    /// Create algorithm parameters from a preset and stretch ratio.
    pub fn from_preset(
        preset: Option<EdmPreset>,
        stretch_ratio: f64,
        sample_rate: u32,
        fft_size: usize,
        hop_size: usize,
        transient_sensitivity: f32,
        use_hybrid: bool,
    ) -> Self {
        let base = Self {
            fft_size,
            hop_size,
            stretch_ratio,
            sample_rate,
            wsola_segment_size: (0.020 * sample_rate as f64) as usize, // ~20ms
            wsola_overlap: 0.5,
            wsola_search_range: (0.010 * sample_rate as f64) as usize, // ~10ms
            sub_bass_cutoff: 120.0,
            crossfade_len: (0.005 * sample_rate as f64) as usize, // 5ms
            transient_sensitivity,
            use_hybrid,
        };

        match preset {
            Some(EdmPreset::DjBeatmatch) => Self {
                wsola_search_range: (0.010 * sample_rate as f64) as usize,
                ..base
            },
            Some(EdmPreset::HouseLoop) => base,
            Some(EdmPreset::Halftime) => Self {
                wsola_search_range: (0.020 * sample_rate as f64) as usize,
                wsola_segment_size: (0.025 * sample_rate as f64) as usize,
                ..base
            },
            Some(EdmPreset::Ambient) => Self {
                wsola_search_range: (0.030 * sample_rate as f64) as usize,
                crossfade_len: (0.010 * sample_rate as f64) as usize,
                use_hybrid: false,
                ..base
            },
            Some(EdmPreset::VocalChop) => Self {
                wsola_segment_size: (0.015 * sample_rate as f64) as usize,
                wsola_search_range: (0.015 * sample_rate as f64) as usize,
                ..base
            },
            None => base,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_params_defaults() {
        let params = AlgorithmParams::from_preset(None, 1.0, 44100, 4096, 1024, 0.5, true);
        assert_eq!(params.fft_size, 4096);
        assert_eq!(params.hop_size, 1024);
        assert_eq!(params.sample_rate, 44100);
        assert!(params.use_hybrid);
    }

    #[test]
    fn test_algorithm_params_ambient_no_hybrid() {
        let params =
            AlgorithmParams::from_preset(Some(EdmPreset::Ambient), 2.0, 44100, 8192, 2048, 0.3, true);
        assert!(!params.use_hybrid);
    }

    #[test]
    fn test_algorithm_params_segment_sizes() {
        let params = AlgorithmParams::from_preset(None, 1.0, 48000, 4096, 1024, 0.5, true);
        // 20ms at 48kHz = 960 samples
        assert_eq!(params.wsola_segment_size, 960);
        // 5ms at 48kHz = 240 samples
        assert_eq!(params.crossfade_len, 240);
    }
}

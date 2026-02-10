use timestretch::{EdmPreset, StreamProcessor, StretchParams};
use wasm_bindgen::prelude::*;

/// WASM wrapper around the timestretch StreamProcessor for real-time stereo processing.
#[wasm_bindgen]
pub struct TimeStretchNode {
    processor: StreamProcessor,
}

#[wasm_bindgen]
impl TimeStretchNode {
    /// Create a new stereo streaming processor.
    ///
    /// - `sample_rate`: audio sample rate (e.g. 44100)
    /// - `stretch_ratio`: initial stretch ratio (1.0 = no change)
    /// - `preset`: optional EDM preset name ("DjBeatmatch", "HouseLoop", "Halftime", "Ambient", "VocalChop")
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: u32, stretch_ratio: f64, preset: Option<String>) -> Self {
        let mut params = StretchParams::new(stretch_ratio)
            .with_sample_rate(sample_rate)
            .with_channels(2); // always stereo

        if let Some(ref name) = preset {
            if let Some(p) = parse_preset(name) {
                params = params.with_preset(p);
            }
        }

        Self {
            processor: StreamProcessor::new(params),
        }
    }

    /// Feed stereo interleaved input samples and get stretched output.
    /// Input/output format: [L0, R0, L1, R1, ...]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        self.processor.process(input).unwrap_or_default()
    }

    /// Flush remaining samples from the processor.
    pub fn flush(&mut self) -> Vec<f32> {
        self.processor.flush().unwrap_or_default()
    }

    /// Smoothly change the stretch ratio (interpolated over time).
    pub fn set_stretch_ratio(&mut self, ratio: f64) {
        self.processor.set_stretch_ratio(ratio);
    }

    /// Reset all internal state (call when seeking or changing presets).
    pub fn reset(&mut self) {
        self.processor.reset();
    }

    /// Minimum number of input samples before output is produced.
    pub fn latency_samples(&self) -> usize {
        self.processor.latency_samples()
    }
}

/// Detect BPM from mono audio samples.
#[wasm_bindgen]
pub fn detect_bpm(samples: &[f32], sample_rate: u32) -> f64 {
    timestretch::detect_bpm(samples, sample_rate)
}

/// Return a JSON array of available preset names.
#[wasm_bindgen]
pub fn list_presets() -> String {
    r#"["DjBeatmatch","HouseLoop","Halftime","Ambient","VocalChop"]"#.to_string()
}

fn parse_preset(name: &str) -> Option<EdmPreset> {
    match name {
        "DjBeatmatch" => Some(EdmPreset::DjBeatmatch),
        "HouseLoop" => Some(EdmPreset::HouseLoop),
        "Halftime" => Some(EdmPreset::Halftime),
        "Ambient" => Some(EdmPreset::Ambient),
        "VocalChop" => Some(EdmPreset::VocalChop),
        _ => None,
    }
}

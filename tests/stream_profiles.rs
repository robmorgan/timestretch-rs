//! Library streaming profiles: parameter tables, constructor wiring, and
//! ordering semantics.

use timestretch::{
    EdmPreset, QualityMode, StreamProcessor, StreamProfile, StretchParams, WindowType,
};

const SR: u32 = 44_100;

#[test]
fn profile_parameter_tables() {
    let cases = [
        (
            StreamProfile::Live,
            1024usize,
            256usize,
            QualityMode::LowLatency,
        ),
        (StreamProfile::Club, 2048, 512, QualityMode::Balanced),
        (StreamProfile::Quality, 4096, 1024, QualityMode::Balanced),
    ];

    for (profile, fft, hop, quality_mode) in cases {
        let params = StretchParams::new(1.02)
            .with_sample_rate(SR)
            .with_channels(2)
            .with_stream_profile(profile);

        assert_eq!(params.fft_size, fft, "{profile}: fft");
        assert_eq!(params.hop_size, hop, "{profile}: hop");
        assert_eq!(params.quality_mode, quality_mode, "{profile}: quality mode");

        // Every profile carries the full DJ tuning bundle.
        assert_eq!(
            params.preset,
            Some(EdmPreset::DjBeatmatch),
            "{profile}: preset bundle"
        );
        assert_eq!(
            params.transient_sensitivity, 0.3,
            "{profile}: DJ transient sensitivity"
        );
        assert!(params.beat_aware, "{profile}: beat aware");
        assert_eq!(params.window_type, WindowType::Hann, "{profile}: window");

        // The profile must not touch caller-owned fields.
        assert_eq!(params.stretch_ratio, 1.02, "{profile}: ratio untouched");
        assert_eq!(params.sample_rate, SR, "{profile}: sample rate untouched");
        assert_eq!(params.channels.count(), 2, "{profile}: channels untouched");
    }
}

#[test]
fn profile_labels_and_order() {
    assert_eq!(StreamProfile::ALL.len(), 3);
    assert_eq!(StreamProfile::Live.label(), "Live");
    assert_eq!(StreamProfile::Club.label(), "Club");
    assert_eq!(StreamProfile::Quality.label(), "Quality");
    assert_eq!(format!("{}", StreamProfile::Club), "Club");
}

#[test]
fn profile_latency_ladder() {
    let latency = |profile: StreamProfile| {
        StreamProcessor::new(
            StretchParams::new(1.02)
                .with_sample_rate(SR)
                .with_channels(1)
                .with_stream_profile(profile),
        )
        .latency_samples()
    };

    assert_eq!(latency(StreamProfile::Live), 1536); // 34.8 ms
    assert_eq!(latency(StreamProfile::Club), 3072); // 69.7 ms
    assert_eq!(latency(StreamProfile::Quality), 6144); // 139.3 ms
}

#[test]
fn low_latency_tempo_ctor_uses_live_profile() {
    let processor =
        StreamProcessor::try_from_tempo_low_latency(126.0, 128.0, SR, 2).expect("valid tempo pair");
    let params = processor.params();
    assert_eq!(params.fft_size, 1024);
    assert_eq!(params.hop_size, 256);
    assert_eq!(params.quality_mode, QualityMode::LowLatency);
    // The DJ bundle is no longer silently dropped.
    assert_eq!(params.preset, Some(EdmPreset::DjBeatmatch));
    assert_eq!(params.transient_sensitivity, 0.3);
    assert!(processor.latency_secs() * 1000.0 < 40.0);
}

#[test]
fn tempo_ctor_with_profile_selects_configuration() {
    for &profile in StreamProfile::ALL {
        let processor = StreamProcessor::try_from_tempo_with_profile(126.0, 128.0, SR, 2, profile)
            .expect("valid tempo pair");
        assert_eq!(
            processor.params().preset,
            Some(EdmPreset::DjBeatmatch),
            "{profile}: bundle"
        );
        assert!(
            (processor.target_stretch_ratio() - 126.0 / 128.0).abs() < 1e-9,
            "{profile}: ratio from tempo"
        );
    }
}

/// Documented footgun: `with_preset` after `with_stream_profile` reverts the
/// profile's FFT configuration. The profile must be applied last.
#[test]
fn preset_after_profile_reverts_fft_documented_ordering() {
    let params = StretchParams::new(1.0)
        .with_sample_rate(SR)
        .with_channels(1)
        .with_stream_profile(StreamProfile::Live)
        .with_preset(EdmPreset::DjBeatmatch);
    assert_eq!(
        params.fft_size, 4096,
        "preset applied after the profile reverts fft (documented ordering)"
    );
}

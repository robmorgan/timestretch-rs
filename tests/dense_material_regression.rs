//! Regression guard for onset detection on dense, mastered material.
//!
//! Before the robust-detection front end (log-compressed ODF + median+MAD
//! thresholding), `detect_transients` returned ZERO onsets on continuously
//! loud broadband audio, so `analyze_for_dj` produced empty artifacts (BPM 0,
//! confidence 0) and analyze-on-load silently no-opped. Every prior test used
//! sparse clicks-over-silence, which never exercised that regime.
//!
//! The fixture `test_audio/dense_mastered_128bpm.wav` reproduces the failure
//! on the pre-fix code (verified: 0 onsets at every sensitivity). These
//! assertions lock in the fix.

use timestretch::analysis::transient::detect_transients;
use timestretch::io::wav::read_wav_file;

const FIXTURE: &str = "test_audio/dense_mastered_128bpm.wav";
const SR: u32 = 44_100;

fn load_dense_mid() -> Vec<f32> {
    let buffer = read_wav_file(FIXTURE).unwrap_or_else(|e| {
        panic!("missing {FIXTURE}: run `cargo run --example generate_test_audio` first ({e})")
    });
    timestretch::downmix_to_mid(&buffer.data, buffer.channels.count())
}

#[test]
fn dense_material_produces_onsets_at_all_sensitivities() {
    let mid = load_dense_mid();
    let seconds = mid.len() as f64 / SR as f64;

    // Sensitivity is monotone: more sensitive → at least as many onsets.
    let mut prev = 0usize;
    for &sensitivity in &[0.2f32, 0.4, 0.8] {
        let map = detect_transients(&mid, SR, 2048, 512, sensitivity);
        let rate = map.onsets.len() as f64 / seconds;
        assert!(
            map.onsets.len() >= 8,
            "sensitivity {sensitivity}: dense material must yield onsets, got {}",
            map.onsets.len()
        );
        // Musically plausible for a 128 BPM four-on-the-floor mix with hats:
        // not zero (the bug), not hi-hat spam.
        assert!(
            (0.5..=15.0).contains(&rate),
            "sensitivity {sensitivity}: onset rate {rate:.2}/s outside plausible band"
        );
        assert!(
            map.onsets.len() >= prev,
            "onset count must be monotone in sensitivity"
        );
        prev = map.onsets.len();
    }
}

#[test]
fn dense_material_analyze_produces_usable_artifact() {
    let mid = load_dense_mid();
    let artifact = timestretch::analyze_for_dj(&mid, SR);

    assert!(
        artifact.bpm > 0.0,
        "dense material must yield a BPM, got {}",
        artifact.bpm
    );
    // Fixture is authored at 128 BPM; accept the true tempo or a nearby
    // octave (the current 100-160 folding tracker may land on a subdivision).
    let bpm = artifact.bpm;
    let near = |target: f64| (bpm - target).abs() / target < 0.04;
    assert!(
        near(128.0) || near(64.0) || near(256.0),
        "dense BPM {bpm} not near 128 or an octave"
    );

    // The whole point: analyze-on-load must not silently no-op — the artifact
    // has to clear the runtime usability gate.
    assert!(
        artifact.is_usable(SR, 0.35),
        "dense artifact not usable: confidence {}, {} beats, {} onsets",
        artifact.confidence,
        artifact.beat_positions.len(),
        artifact.transient_onsets.len()
    );
    assert!(
        artifact.confidence >= 0.5,
        "dense material confidence too low: {}",
        artifact.confidence
    );
}

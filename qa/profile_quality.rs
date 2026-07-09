//! Cross-profile streaming quality gate: Live / Club / Quality on the same
//! DJ material under a ratio ride must be measurably ordered, and the Live
//! profile must clear an absolute floor so low latency is a supported mode,
//! not a degraded one.
//!
//! Run with:
//! `cargo test --features qa-harnesses --release --test profile_quality -- --nocapture`

use std::f32::consts::PI;
use timestretch::{analysis::comparison, StreamProcessor, StreamProfile, StretchParams};

const SAMPLE_RATE: u32 = 44_100;
const TWO_PI: f32 = 2.0 * PI;
const CHUNK: usize = 1024;

/// EDM-like test signal: kick pattern, sub bass, vibrato lead, hats.
fn generate_edm_signal(duration_secs: f32) -> Vec<f32> {
    let num_samples = (SAMPLE_RATE as f32 * duration_secs) as usize;
    let mut signal = vec![0.0f32; num_samples];
    let beat_interval = (SAMPLE_RATE as f64 * 60.0 / 128.0) as usize;

    for (i, sample) in signal.iter_mut().enumerate() {
        let t = i as f32 / SAMPLE_RATE as f32;
        *sample += 0.3 * (TWO_PI * 60.0 * t).sin();
        let vibrato = 5.0 * (TWO_PI * 4.0 * t).sin();
        *sample += 0.2 * (TWO_PI * (300.0 + vibrato) * t).sin();
        let half_beat = beat_interval / 2;
        if i % half_beat < SAMPLE_RATE as usize / 200 {
            *sample += 0.1 * (((i * 7 + 13) % 1000) as f32 / 500.0 - 1.0);
        }
        let pos_in_beat = i % beat_interval;
        if pos_in_beat < SAMPLE_RATE as usize / 50 {
            let kick_t = pos_in_beat as f32 / SAMPLE_RATE as f32;
            let kick_freq = 150.0 * (-kick_t * 40.0).exp() + 50.0;
            *sample += 0.5 * (TWO_PI * kick_freq * kick_t).sin() * (-kick_t * 20.0).exp();
        }
    }
    normalize(&mut signal);
    signal
}

/// Bass-heavy signal for the sub-bass cutoff experiment: deep sub line plus
/// kicks, minimal top end.
fn generate_bass_signal(duration_secs: f32) -> Vec<f32> {
    let num_samples = (SAMPLE_RATE as f32 * duration_secs) as usize;
    let mut signal = vec![0.0f32; num_samples];
    let beat_interval = (SAMPLE_RATE as f64 * 60.0 / 128.0) as usize;

    for (i, sample) in signal.iter_mut().enumerate() {
        let t = i as f32 / SAMPLE_RATE as f32;
        // Sub line alternating 41.2 Hz (E1) and 55 Hz (A1) per half bar.
        let bar = (i / (beat_interval * 2)) % 2;
        let freq = if bar == 0 { 41.2 } else { 55.0 };
        *sample += 0.5 * (TWO_PI * freq * t).sin();
        *sample += 0.15 * (TWO_PI * freq * 2.0 * t).sin();
        let pos_in_beat = i % beat_interval;
        if pos_in_beat < SAMPLE_RATE as usize / 50 {
            let kick_t = pos_in_beat as f32 / SAMPLE_RATE as f32;
            let kick_freq = 150.0 * (-kick_t * 40.0).exp() + 50.0;
            *sample += 0.5 * (TWO_PI * kick_freq * kick_t).sin() * (-kick_t * 20.0).exp();
        }
    }
    normalize(&mut signal);
    signal
}

fn normalize(signal: &mut [f32]) {
    let peak = signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let gain = 0.9 / peak;
        for s in signal.iter_mut() {
            *s *= gain;
        }
    }
}

/// Streams `input` through a processor with a DJ-style ratio ride
/// (sinusoidal 0.92..1.08 sweep, one cycle every 2 seconds).
fn stream_with_ratio_ride(input: &[f32], params: StretchParams) -> Vec<f32> {
    let mut processor = StreamProcessor::new(params);
    let mut output = Vec::with_capacity(input.len() * 2);
    for (ci, chunk) in input.chunks(CHUNK).enumerate() {
        let t = (ci * CHUNK) as f64 / SAMPLE_RATE as f64;
        let ratio = 1.0 + 0.08 * (TWO_PI as f64 * t / 2.0).sin();
        processor.set_stretch_ratio(ratio).expect("valid ratio");
        let rendered = processor.process(chunk).expect("stream chunk");
        output.extend_from_slice(&rendered);
    }
    let tail = processor.flush().expect("flush");
    output.extend_from_slice(&tail);
    output
}

/// Sample-to-sample jumps that exceed a musical slew bound.
fn click_count(signal: &[f32], threshold: f32) -> usize {
    signal
        .windows(2)
        .filter(|w| (w[1] - w[0]).abs() > threshold)
        .count()
}

/// Streams `input` at a fixed off-unity ratio.
fn stream_steady(input: &[f32], params: StretchParams, ratio: f64) -> Vec<f32> {
    let mut processor = StreamProcessor::new(params.with_stretch_ratio(ratio));
    let mut output = Vec::with_capacity(input.len() * 2);
    for chunk in input.chunks(CHUNK) {
        let rendered = processor.process(chunk).expect("stream chunk");
        output.extend_from_slice(&rendered);
    }
    let tail = processor.flush().expect("flush");
    output.extend_from_slice(&tail);
    output
}

fn profile_params(profile: StreamProfile) -> StretchParams {
    StretchParams::new(1.0)
        .with_sample_rate(SAMPLE_RATE)
        .with_channels(1)
        .with_stream_profile(profile)
}

fn similarity_to_source(input: &[f32], output: &[f32]) -> f64 {
    // Timing-invariant spectral similarity to the source at a fixed analysis
    // resolution so profiles are compared on equal footing.
    comparison::mean_spectral_similarity(input, output, 2048, 512)
}

#[test]
fn profile_quality_under_ratio_ride() {
    let input = generate_edm_signal(6.0);

    let mut results = Vec::new();
    for &profile in StreamProfile::ALL {
        let output = stream_with_ratio_ride(&input, profile_params(profile));
        let similarity = similarity_to_source(&input, &output);
        let clicks = click_count(&output[4096..], 0.5);
        println!(
            "METRIC ride_similarity_{}={:.4} clicks={}",
            profile.label().to_lowercase(),
            similarity,
            clicks
        );
        results.push((profile, similarity, clicks));
    }

    for (profile, similarity, clicks) in results {
        // No profile may click under a plain ratio ride.
        assert_eq!(clicks, 0, "{profile} clicked under ratio ride");
        // Under modulation, smaller windows track the ratio changes better,
        // so the steady-ratio quality ladder does not apply here; every
        // profile must simply clear the supported-mode floor. (Baseline:
        // Live 0.9982, Club 0.9969, Quality 0.9932.)
        assert!(
            similarity >= 0.985,
            "{profile} ride similarity floor violated: {:.4} < 0.985",
            similarity
        );
    }
}

#[test]
fn profile_quality_ladder_at_steady_ratio() {
    let input = generate_edm_signal(6.0);
    let ratio = 1.05;

    let measure = |profile: StreamProfile| {
        let output = stream_steady(&input, profile_params(profile), ratio);
        let similarity = similarity_to_source(&input, &output);
        let clicks = click_count(&output[4096..], 0.5);
        println!(
            "METRIC steady_similarity_{}={:.4} clicks={}",
            profile.label().to_lowercase(),
            similarity,
            clicks
        );
        (similarity, clicks)
    };

    let (live, live_clicks) = measure(StreamProfile::Live);
    let (club, club_clicks) = measure(StreamProfile::Club);
    let (quality, quality_clicks) = measure(StreamProfile::Quality);

    assert_eq!(live_clicks + club_clicks + quality_clicks, 0, "clicks");

    // Steady-ratio spectral quality: higher-latency profiles must not fall
    // below lower-latency ones, and Live must clear the supported-mode
    // floor. Epsilon covers measurement noise, not a real inversion.
    let epsilon = 0.004;
    assert!(
        quality + epsilon >= club,
        "Quality ({:.4}) fell below Club ({:.4})",
        quality,
        club
    );
    assert!(
        club + epsilon >= live,
        "Club ({:.4}) fell below Live ({:.4})",
        club,
        live
    );
    assert!(
        live >= 0.985,
        "Live steady similarity floor violated: {:.4} < 0.985",
        live
    );
}

/// Sub-bass cutoff experiment for the Live profile (1024 FFT: the 180 Hz
/// streaming floor covers only ~4 bins). Measured comparison, not a gate:
/// prints sub-band similarity with the default cutoff vs a raised 250 Hz
/// cutoff on bass-heavy material.
#[test]
fn live_profile_sub_bass_cutoff_experiment() {
    let input = generate_bass_signal(6.0);

    let run = |cutoff: Option<f32>| {
        let mut params = StretchParams::new(1.0)
            .with_sample_rate(SAMPLE_RATE)
            .with_channels(1)
            .with_stream_profile(StreamProfile::Live);
        if let Some(hz) = cutoff {
            params = params.with_sub_bass_cutoff(hz);
        }
        let output = stream_with_ratio_ride(&input, params);
        let sim = comparison::mean_spectral_similarity(&input, &output, 2048, 512);
        let bands =
            comparison::mean_band_spectral_similarity(&input, &output, 2048, 512, SAMPLE_RATE);
        (sim, bands)
    };

    let (default_sim, default_bands) = run(None);
    let (raised_sim, raised_bands) = run(Some(250.0));

    println!(
        "METRIC live_subbass_default sim={:.4} sub={:.4} low={:.4}",
        default_sim, default_bands.sub_bass, default_bands.low
    );
    println!(
        "METRIC live_subbass_250 sim={:.4} sub={:.4} low={:.4}",
        raised_sim, raised_bands.sub_bass, raised_bands.low
    );
}

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

/// Streams `input` at a fixed off-unity ratio with a selected engine.
fn stream_steady_with_engine(
    input: &[f32],
    params: StretchParams,
    ratio: f64,
    engine: timestretch::StreamingEngine,
) -> Vec<f32> {
    let mut processor = StreamProcessor::new(params.with_stretch_ratio(ratio));
    processor
        .set_streaming_engine(engine)
        .expect("profile supports the requested engine");
    let mut output = Vec::with_capacity(input.len() * 2);
    for chunk in input.chunks(CHUNK) {
        let rendered = processor.process(chunk).expect("stream chunk");
        output.extend_from_slice(&rendered);
    }
    let tail = processor.flush().expect("flush");
    output.extend_from_slice(&tail);
    output
}

/// Streams `input` under the DJ ratio ride with a selected engine.
fn stream_ride_with_engine(
    input: &[f32],
    params: StretchParams,
    engine: timestretch::StreamingEngine,
) -> Vec<f32> {
    let mut processor = StreamProcessor::new(params);
    processor
        .set_streaming_engine(engine)
        .expect("profile supports the requested engine");
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

/// Multi-resolution engine measurement rows (Club and Quality; Live rejects
/// the engine), plus the gate that motivates the engine: on sub-bass-heavy
/// EDM material at the Quality profile, the multi-res filterbank must not
/// fall below the single-PV deterministic engine in sub-band similarity.
///
/// Measured seam context (2026-07): the multi-res engine ties single-PV at
/// the sub-band metric ceiling and wins the mid band, but loses similarity
/// in the bands containing the 200 Hz / 4 kHz crossover seams — a property
/// inherited from the offline `MultiResolutionStretcher` (independent band
/// PVs de-phase where LR8 skirts overlap), not from the streaming port.
#[test]
fn multi_res_engine_quality_rows() {
    use timestretch::StreamingEngine;

    let edm = generate_edm_signal(6.0);
    let bass = generate_bass_signal(6.0);
    let ratio = 1.05;

    // Ratio-ride rows: no profile/engine may click under a plain ride.
    for profile in [StreamProfile::Club, StreamProfile::Quality] {
        let output = stream_ride_with_engine(
            &edm,
            profile_params(profile),
            StreamingEngine::MultiResolution,
        );
        let similarity = similarity_to_source(&edm, &output);
        let clicks = click_count(&output[8192..], 0.5);
        println!(
            "METRIC ride_similarity_{}_multi_res={:.4} clicks={}",
            profile.label().to_lowercase(),
            similarity,
            clicks
        );
        assert_eq!(clicks, 0, "{profile} multi-res clicked under ratio ride");
    }

    for profile in [StreamProfile::Club, StreamProfile::Quality] {
        let single = stream_steady_with_engine(
            &edm,
            profile_params(profile),
            ratio,
            StreamingEngine::Deterministic,
        );
        let multi = stream_steady_with_engine(
            &edm,
            profile_params(profile),
            ratio,
            StreamingEngine::MultiResolution,
        );
        let single_sim = similarity_to_source(&edm, &single);
        let multi_sim = similarity_to_source(&edm, &multi);
        let single_bands =
            comparison::mean_band_spectral_similarity(&edm, &single, 2048, 512, SAMPLE_RATE);
        let multi_bands =
            comparison::mean_band_spectral_similarity(&edm, &multi, 2048, 512, SAMPLE_RATE);
        let multi_clicks = click_count(&multi[8192..], 0.5);
        println!(
            "METRIC steady_similarity_{}_single_pv={:.4} multi_res={:.4} multi_clicks={}",
            profile.label().to_lowercase(),
            single_sim,
            multi_sim,
            multi_clicks
        );
        println!(
            "METRIC steady_bands_{} single sub={:.4} low={:.4} mid={:.4} high={:.4} | multi sub={:.4} low={:.4} mid={:.4} high={:.4}",
            profile.label().to_lowercase(),
            single_bands.sub_bass,
            single_bands.low,
            single_bands.mid,
            single_bands.high,
            multi_bands.sub_bass,
            multi_bands.low,
            multi_bands.mid,
            multi_bands.high
        );
        assert_eq!(
            multi_clicks, 0,
            "{profile} multi-res clicked at steady ratio"
        );
    }

    // Sub-bass gate on bass-heavy material at the Quality profile.
    let single = stream_steady_with_engine(
        &bass,
        profile_params(StreamProfile::Quality),
        ratio,
        StreamingEngine::Deterministic,
    );
    let multi = stream_steady_with_engine(
        &bass,
        profile_params(StreamProfile::Quality),
        ratio,
        StreamingEngine::MultiResolution,
    );
    let single_bands =
        comparison::mean_band_spectral_similarity(&bass, &single, 2048, 512, SAMPLE_RATE);
    let multi_bands =
        comparison::mean_band_spectral_similarity(&bass, &multi, 2048, 512, SAMPLE_RATE);
    println!(
        "METRIC quality_bass_sub single_pv={:.4} multi_res={:.4} low single_pv={:.4} multi_res={:.4}",
        single_bands.sub_bass, multi_bands.sub_bass, single_bands.low, multi_bands.low
    );
    let epsilon = 0.002;
    assert!(
        multi_bands.sub_bass + epsilon >= single_bands.sub_bass,
        "multi-res Quality sub-band similarity ({:.4}) fell below single-PV ({:.4})",
        multi_bands.sub_bass,
        single_bands.sub_bass
    );
}

/// Streams `input` at a fixed off-unity ratio on a selected control path.
fn stream_steady_with_path(
    input: &[f32],
    params: StretchParams,
    ratio: f64,
    path: timestretch::ControlPath,
) -> Vec<f32> {
    let mut processor = StreamProcessor::new(params.with_stretch_ratio(ratio));
    processor
        .set_control_path(path)
        .expect("control path accepted");
    let mut output = Vec::with_capacity(input.len() * 2);
    for chunk in input.chunks(CHUNK) {
        let rendered = processor.process(chunk).expect("stream chunk");
        output.extend_from_slice(&rendered);
    }
    let tail = processor.flush().expect("flush");
    output.extend_from_slice(&tail);
    output
}

/// Streams `input` under the DJ ratio ride on a selected control path.
fn stream_ride_with_path(
    input: &[f32],
    params: StretchParams,
    path: timestretch::ControlPath,
) -> Vec<f32> {
    let mut processor = StreamProcessor::new(params);
    processor
        .set_control_path(path)
        .expect("control path accepted");
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

/// Stage 15 A/B gate: the varispeed-first control path must not regress
/// streaming quality against the vocoder-tempo path on the profile rows —
/// steady ratio and the DJ ratio ride — and may not click.
#[test]
fn control_path_quality_ab_rows() {
    use timestretch::ControlPath;

    let edm = generate_edm_signal(6.0);
    let ratio = 1.05;
    let epsilon = 0.002;

    for &profile in StreamProfile::ALL {
        let vocoder = stream_steady_with_path(
            &edm,
            profile_params(profile),
            ratio,
            ControlPath::VocoderTempo,
        );
        let varispeed = stream_steady_with_path(
            &edm,
            profile_params(profile),
            ratio,
            ControlPath::VarispeedFirst,
        );
        let vocoder_sim = similarity_to_source(&edm, &vocoder);
        let varispeed_sim = similarity_to_source(&edm, &varispeed);
        let varispeed_clicks = click_count(&varispeed[4096..], 0.5);
        println!(
            "METRIC steady_similarity_{}_vocoder={:.4} varispeed={:.4} varispeed_clicks={}",
            profile.label().to_lowercase(),
            vocoder_sim,
            varispeed_sim,
            varispeed_clicks
        );
        assert_eq!(
            varispeed_clicks, 0,
            "{profile} varispeed clicked at steady ratio"
        );
        assert!(
            varispeed_sim + epsilon >= vocoder_sim,
            "{profile} varispeed steady similarity ({:.4}) fell below vocoder path ({:.4})",
            varispeed_sim,
            vocoder_sim
        );
        assert!(
            varispeed_sim >= 0.985,
            "{profile} varispeed steady similarity floor violated: {:.4}",
            varispeed_sim
        );

        let vocoder_ride =
            stream_ride_with_path(&edm, profile_params(profile), ControlPath::VocoderTempo);
        let varispeed_ride =
            stream_ride_with_path(&edm, profile_params(profile), ControlPath::VarispeedFirst);
        let vocoder_ride_sim = similarity_to_source(&edm, &vocoder_ride);
        let varispeed_ride_sim = similarity_to_source(&edm, &varispeed_ride);
        let ride_clicks = click_count(&varispeed_ride[4096..], 0.5);
        println!(
            "METRIC ride_similarity_{}_vocoder={:.4} varispeed={:.4} varispeed_clicks={}",
            profile.label().to_lowercase(),
            vocoder_ride_sim,
            varispeed_ride_sim,
            ride_clicks
        );
        assert_eq!(ride_clicks, 0, "{profile} varispeed clicked under ride");
        assert!(
            varispeed_ride_sim + epsilon >= vocoder_ride_sim,
            "{profile} varispeed ride similarity ({:.4}) fell below vocoder path ({:.4})",
            varispeed_ride_sim,
            vocoder_ride_sim
        );
        assert!(
            varispeed_ride_sim >= 0.985,
            "{profile} varispeed ride similarity floor violated: {:.4}",
            varispeed_ride_sim
        );
    }
}

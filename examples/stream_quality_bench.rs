//! Streaming (real-time engine) audio quality benchmark.
//!
//! Streams real music and synthetic fixtures through the pull-based engine
//! (Keylock profile, 256-frame callbacks) exactly the way an audio callback
//! would, then scores the output with reference-free quality metrics:
//!
//! - Timbre preservation: timing-invariant mean-spectrum cosine similarity
//!   between input and stretched output (slowdown + speedup).
//! - Transient preservation: F1 between input onsets mapped through the
//!   stretch ratio and onsets detected in the output.
//! - Pitch stability: cents deviation (p95/max) of a 440 Hz sine under the
//!   +/-8% / 2 s DJ tempo ride.
//! - Identity transparency: cross-correlation peak at rate 1.0.
//! - Click-freeness: discontinuity excess vs the input's own sample-to-sample
//!   slope statistics.
//!
//! Emits `METRIC name=value` lines for automated tracking.
//!
//! Run with: cargo run --release --example stream_quality_bench

use timestretch::analysis::comparison::{
    mean_band_spectral_similarity, mean_spectral_similarity, spectral_similarity,
};
use timestretch::analysis::preanalysis::downmix_to_mid;
use timestretch::analysis::transient::detect_transients;
use timestretch::engine::{Engine, EngineConfig, EngineProfile};
use timestretch::io::wav::read_wav_file;

const MUSIC_PATH: &str =
    "benchmarks/audio/bpm-corpus/12247392_Music Sounds Better With You_(Original Mix).wav";
const SAMPLE_RATE: u32 = 44_100;
const CALLBACK_FRAMES: usize = 256;

/// Segment of the track used for the music renders (skip intro, keep it fast).
const MUSIC_SKIP_SECS: usize = 10;
const MUSIC_SECS: usize = 40;

/// Stretch rates: moderate (124->115 / 124->132 BPM) and hard
/// (124->104 / 124->140 BPM) slowdowns and speedups.
const RATE_SLOW: f64 = 115.0 / 124.0;
const RATE_FAST: f64 = 132.0 / 124.0;
const RATE_SLOW2: f64 = 104.0 / 124.0;
const RATE_FAST2: f64 = 140.0 / 124.0;

const FFT_SIZE: usize = 4096;
const HOP_SIZE: usize = 1024;
const TRANSIENT_TOLERANCE_MS: f64 = 15.0;
const TRANSIENT_FFT: usize = 2048;
const TRANSIENT_HOP: usize = 256;
const TRANSIENT_SENSITIVITY: f32 = 0.45;

/// Output settle time excluded from all metrics (pipeline fill + filters).
const WARMUP_SECS: f64 = 0.5;

// ---------------------------------------------------------------------------
// Engine streaming driver (mirrors a real audio callback loop)
// ---------------------------------------------------------------------------

struct Render {
    /// Interleaved output samples (media-only; terminal shortfall trimmed).
    output: Vec<f32>,
    /// Mid-stream underrun frames (excludes expected end-of-stream shortfall).
    underrun_frames: u64,
    /// Wall-clock processing time in seconds (process() calls only).
    process_secs: f64,
    /// Reported pipeline latency in frames (constant, compensated in metrics).
    latency_frames: usize,
}

fn render_stream(input: &[f32], channels: usize, rate_at: &dyn Fn(f64) -> f64) -> Render {
    let handles = Engine::build(EngineConfig {
        sample_rate: SAMPLE_RATE,
        channels,
        profile: EngineProfile::Keylock,
        initial_tempo_rate: rate_at(0.0),
        max_block_frames: CALLBACK_FRAMES,
        source_capacity_frames: (CALLBACK_FRAMES * 16).max(32_768),
        pre_analysis: None,
    })
    .expect("engine builds");
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);
    let latency_frames = processor.pipeline_latency_frames();
    source.set_track_position(0);

    let mut feed_cursor = 0usize;
    let mut finished = false;
    let mut out = vec![0.0f32; CALLBACK_FRAMES * channels];
    let mut output: Vec<f32> = Vec::with_capacity(input.len() * 2 + 65_536);
    let mut process_secs = 0.0f64;

    let mut cb = 0usize;
    loop {
        let t = (cb * CALLBACK_FRAMES) as f64 / SAMPLE_RATE as f64;
        controller.set_tempo_rate(rate_at(t));

        while feed_cursor < input.len()
            && source.occupied_frames() < source.demand_hint(CALLBACK_FRAMES, 4.0)
        {
            let end = (feed_cursor + 8192 * channels).min(input.len());
            let accepted = source.push(&input[feed_cursor..end]);
            feed_cursor += accepted * channels;
            if accepted == 0 {
                break;
            }
        }
        if feed_cursor >= input.len() && !finished {
            finished = source.finish();
        }

        let underruns_before = controller.underrun_frames();
        let t0 = std::time::Instant::now();
        processor.process(&mut out);
        process_secs += t0.elapsed().as_secs_f64();
        if controller.underrun_frames() > underruns_before {
            let shortfall = controller.underrun_frames() - underruns_before;
            let missing = shortfall as usize * channels;
            output.extend_from_slice(&out[..out.len() - missing.min(out.len())]);
            return Render {
                output,
                underrun_frames: underruns_before,
                process_secs,
                latency_frames,
            };
        }
        output.extend_from_slice(&out);
        cb += 1;

        if cb > 4 * input.len() / (CALLBACK_FRAMES * channels) + 1024 {
            panic!("render did not terminate");
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// F1 between input onsets mapped through the stretch ratio and output onsets.
/// `latency` is the engine's reported constant pipeline delay in frames.
fn transient_f1(
    input_mid: &[f32],
    output_mid: &[f32],
    rate: f64,
    latency: usize,
) -> (f64, usize, usize) {
    let in_map = detect_transients(
        input_mid,
        SAMPLE_RATE,
        TRANSIENT_FFT,
        TRANSIENT_HOP,
        TRANSIENT_SENSITIVITY,
    );
    let out_map = detect_transients(
        output_mid,
        SAMPLE_RATE,
        TRANSIENT_FFT,
        TRANSIENT_HOP,
        TRANSIENT_SENSITIVITY,
    );
    let tol = (TRANSIENT_TOLERANCE_MS * SAMPLE_RATE as f64 / 1000.0) as i64;
    let warmup = (WARMUP_SECS * SAMPLE_RATE as f64) as i64;

    // Expected output positions of the input onsets (tempo rate > 1 means
    // faster playback, so output time = input time / rate), shifted by the
    // engine's constant reported latency.
    let expected: Vec<i64> = in_map
        .onsets
        .iter()
        .map(|&s| (s as f64 / rate) as i64 + latency as i64)
        .filter(|&s| s >= warmup && s < output_mid.len() as i64 - tol)
        .collect();
    let found: Vec<i64> = out_map
        .onsets
        .iter()
        .map(|&s| s as i64)
        .filter(|&s| s >= warmup)
        .collect();
    if expected.is_empty() || found.is_empty() {
        return (0.0, expected.len(), found.len());
    }

    let mut matched_exp = 0usize;
    for &e in &expected {
        if found.iter().any(|&f| (f - e).abs() <= tol) {
            matched_exp += 1;
        }
    }
    let mut matched_found = 0usize;
    for &f in &found {
        if expected.iter().any(|&e| (f - e).abs() <= tol) {
            matched_found += 1;
        }
    }
    let recall = matched_exp as f64 / expected.len() as f64;
    let precision = matched_found as f64 / found.len() as f64;
    let f1 = if recall + precision > 0.0 {
        2.0 * recall * precision / (recall + precision)
    } else {
        0.0
    };
    (f1, expected.len(), found.len())
}

/// Count of sample-to-sample jumps exceeding a threshold derived from the
/// input's own slope distribution (robust click detector), per million samples.
fn clicks_per_million(input: &[f32], output: &[f32]) -> f64 {
    let max_in_diff = input
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max);
    let threshold = (max_in_diff * 1.5).max(0.05);
    let warmup = (WARMUP_SECS * SAMPLE_RATE as f64) as usize;
    if output.len() <= warmup + 1 {
        return 0.0;
    }
    let region = &output[warmup..];
    let clicks = region
        .windows(2)
        .filter(|w| (w[1] - w[0]).abs() > threshold)
        .count();
    clicks as f64 * 1.0e6 / region.len() as f64
}

/// Instantaneous frequency from linearly-interpolated positive zero crossings.
fn zero_crossing_freq(window: &[f32]) -> Option<f64> {
    let (mut first, mut last, mut count) = (None, None, 0usize);
    for i in 1..window.len() {
        let (a, b) = (window[i - 1] as f64, window[i] as f64);
        if a <= 0.0 && b > 0.0 {
            let frac = if (b - a).abs() > 1e-12 {
                -a / (b - a)
            } else {
                0.0
            };
            let t = (i - 1) as f64 + frac;
            if first.is_none() {
                first = Some(t);
            }
            last = Some(t);
            count += 1;
        }
    }
    match (first, last) {
        (Some(f), Some(l)) if count >= 2 && l > f => {
            Some((count - 1) as f64 * SAMPLE_RATE as f64 / (l - f))
        }
        _ => None,
    }
}

/// p95 and max absolute cents deviation of a sine render vs the reference Hz.
fn cents_stats(output: &[f32], reference_hz: f64) -> (f64, f64) {
    let window = 4_410usize; // ~100 ms
    let hop = 1_102usize; // ~25 ms
    let warmup = (WARMUP_SECS * SAMPLE_RATE as f64) as usize;
    let mut cents: Vec<f64> = Vec::new();
    let mut pos = warmup;
    while pos + window <= output.len() {
        if let Some(f) = zero_crossing_freq(&output[pos..pos + window]) {
            cents.push((1200.0 * (f / reference_hz).log2()).abs());
        }
        pos += hop;
    }
    if cents.is_empty() {
        return (f64::INFINITY, f64::INFINITY);
    }
    cents.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = cents[((cents.len() - 1) as f64 * 0.95) as usize];
    let max = *cents.last().unwrap();
    (p95, max)
}

fn main() {
    // --- Fixtures ---
    let music = read_wav_file(MUSIC_PATH).unwrap_or_else(|e| {
        panic!("missing corpus file {MUSIC_PATH}: {e}");
    });
    assert_eq!(music.sample_rate, SAMPLE_RATE, "expected 44.1 kHz corpus");
    let ch = music.channels.count();
    let skip = MUSIC_SKIP_SECS * SAMPLE_RATE as usize * ch;
    let take = MUSIC_SECS * SAMPLE_RATE as usize * ch;
    let segment: Vec<f32> = music.data[skip..skip + take].to_vec();
    let segment_mid = downmix_to_mid(&segment, ch);

    let sine: Vec<f32> = (0..SAMPLE_RATE as usize * 10)
        .map(|i| {
            0.5 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SAMPLE_RATE as f64).sin() as f32
        })
        .collect();

    let ride = |t: f64| 1.0 + 0.08 * (2.0 * std::f64::consts::PI * 0.25 * t).sin();

    // --- Renders ---
    let slow = render_stream(&segment, ch, &|_| RATE_SLOW);
    let fast = render_stream(&segment, ch, &|_| RATE_FAST);
    let slow2 = render_stream(&segment, ch, &|_| RATE_SLOW2);
    let fast2 = render_stream(&segment, ch, &|_| RATE_FAST2);
    let identity = render_stream(&segment, ch, &|_| 1.0);
    let music_ride = render_stream(&segment, ch, &ride);
    let sine_ride = render_stream(&sine, 1, &ride);

    let slow_mid = downmix_to_mid(&slow.output, ch);
    let fast_mid = downmix_to_mid(&fast.output, ch);
    let slow2_mid = downmix_to_mid(&slow2.output, ch);
    let fast2_mid = downmix_to_mid(&fast2.output, ch);
    let identity_mid = downmix_to_mid(&identity.output, ch);
    let ride_mid = downmix_to_mid(&music_ride.output, ch);

    let warmup = (WARMUP_SECS * SAMPLE_RATE as f64) as usize;

    // --- Timbre: timing-invariant mean spectrum similarity ---
    let spec_slow = mean_spectral_similarity(&segment_mid, &slow_mid[warmup..], FFT_SIZE, HOP_SIZE);
    let spec_fast = mean_spectral_similarity(&segment_mid, &fast_mid[warmup..], FFT_SIZE, HOP_SIZE);
    let spec_slow2 =
        mean_spectral_similarity(&segment_mid, &slow2_mid[warmup..], FFT_SIZE, HOP_SIZE);
    let spec_fast2 =
        mean_spectral_similarity(&segment_mid, &fast2_mid[warmup..], FFT_SIZE, HOP_SIZE);

    // --- Transients ---
    let (tf1_slow, exp_slow, found_slow) =
        transient_f1(&segment_mid, &slow_mid, RATE_SLOW, slow.latency_frames);
    let (tf1_fast, exp_fast, found_fast) =
        transient_f1(&segment_mid, &fast_mid, RATE_FAST, fast.latency_frames);
    let (tf1_slow2, _, _) =
        transient_f1(&segment_mid, &slow2_mid, RATE_SLOW2, slow2.latency_frames);
    let (tf1_fast2, _, _) =
        transient_f1(&segment_mid, &fast2_mid, RATE_FAST2, fast2.latency_frames);

    // --- Identity transparency: latency-aligned frame-wise magnitude
    // spectral similarity (phase-blind: the LR8 crossover re-sums to
    // allpass, so waveform correlation would punish inaudible phase).
    let identity_corr = {
        let lat = identity.latency_frames;
        let n = (identity_mid.len() - lat).min(segment_mid.len()) - warmup;
        let a = &segment_mid[warmup..warmup + n];
        let b = &identity_mid[warmup + lat..warmup + lat + n];
        spectral_similarity(a, b, 2048, 512)
    };

    // --- Pitch stability ---
    let (p95_cents, max_cents) = cents_stats(&sine_ride.output, 440.0);

    // --- Clicks (worst across the music renders) ---
    let clicks = clicks_per_million(&segment_mid, &slow_mid)
        .max(clicks_per_million(&segment_mid, &fast_mid))
        .max(clicks_per_million(&segment_mid, &slow2_mid))
        .max(clicks_per_million(&segment_mid, &fast2_mid))
        .max(clicks_per_million(&segment_mid, &ride_mid));

    // --- Underruns across all renders (must stay 0) ---
    let underruns = slow.underrun_frames
        + fast.underrun_frames
        + slow2.underrun_frames
        + fast2.underrun_frames
        + identity.underrun_frames
        + music_ride.underrun_frames
        + sine_ride.underrun_frames;

    // --- Throughput (media seconds per process second, music renders) ---
    let media_secs = 5.0 * MUSIC_SECS as f64;
    let proc_secs = slow.process_secs
        + fast.process_secs
        + slow2.process_secs
        + fast2.process_secs
        + music_ride.process_secs;
    let realtime_x = media_secs / proc_secs;

    // --- Composite quality score (0-100, higher is better) ---
    let spec_score = 0.25 * (spec_slow + spec_fast + spec_slow2 + spec_fast2);
    let transient_score = 0.25 * (tf1_slow + tf1_fast + tf1_slow2 + tf1_fast2);
    let pitch_score = (-p95_cents / 10.0).exp();
    let click_score = (-clicks / 50.0).exp();
    let quality = 100.0
        * (0.30 * spec_score
            + 0.30 * transient_score
            + 0.20 * pitch_score
            + 0.10 * identity_corr
            + 0.10 * click_score)
        * if underruns > 0 { 0.5 } else { 1.0 };

    println!("--- streaming quality ---");
    println!(
        "spec slow={spec_slow:.4} fast={spec_fast:.4} slow2={spec_slow2:.4} fast2={spec_fast2:.4}"
    );
    // Diagnostic only (not scored): where the hard-slowdown loss lives.
    let bands = mean_band_spectral_similarity(
        &segment_mid,
        &slow2_mid[warmup..],
        FFT_SIZE,
        HOP_SIZE,
        SAMPLE_RATE,
    );
    println!(
        "slow2 bands: sub_bass={:.4} low={:.4} mid={:.4} high={:.4}",
        bands.sub_bass, bands.low, bands.mid, bands.high
    );
    println!(
        "transient_f1 slow={tf1_slow:.4} (exp {exp_slow} found {found_slow}) \
         fast={tf1_fast:.4} (exp {exp_fast} found {found_fast}) \
         slow2={tf1_slow2:.4} fast2={tf1_fast2:.4}"
    );
    println!("identity_corr={identity_corr:.4} (latency-aligned frame-wise spectral sim)");
    println!("pitch p95={p95_cents:.2}c max={max_cents:.2}c");
    println!("clicks/M={clicks:.1} underruns={underruns}");
    println!();
    println!("METRIC quality={quality:.3}");
    println!("METRIC spec_sim={:.4}", spec_score);
    println!("METRIC transient_f1={:.4}", transient_score);
    println!("METRIC pitch_p95_cents={p95_cents:.2}");
    println!("METRIC pitch_max_cents={max_cents:.2}");
    println!("METRIC identity_corr={identity_corr:.4}");
    println!("METRIC clicks_per_m={clicks:.1}");
    println!("METRIC underruns={underruns}");
    println!("METRIC realtime_x={realtime_x:.1}");
}

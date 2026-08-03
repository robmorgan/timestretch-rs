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

use timestretch::analysis::comparison::{mean_spectral_similarity, spectral_similarity};
use timestretch::analysis::preanalysis::downmix_to_mid;
use timestretch::analysis::transient::detect_transients;
use timestretch::engine::{Engine, EngineConfig, EngineProfile};
use timestretch::io::wav::read_wav_file;

const MUSIC_PATH: &str =
    "benchmarks/audio/bpm-corpus/12247392_Music Sounds Better With You_(Original Mix).wav";
/// Second scored track (different material: disco-house, 120 BPM) so the
/// score is not fit to a single track's splice timing.
const MUSIC_PATH_B: &str = "benchmarks/audio/bpm-corpus/14220825_Hot Stuff_(Original Mix).wav";
/// Third scored track (pop-dance, 116 BPM): broader material coverage.
const MUSIC_PATH_C: &str =
    "benchmarks/audio/bpm-corpus/15836669_Cold Heart_(PNAU Extended Mix).wav";
/// Fourth scored track (vocal-heavy funky house): vocals are a texture the
/// first three tracks under-represent.
const MUSIC_PATH_D: &str =
    "benchmarks/audio/bpm-corpus/15650709_Somebody To Love_(Extended Mix).wav";
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

/// Per-track stretch-quality scores over the four fixed-rate renders.
struct TrackScores {
    spec: [f64; 4],
    tf1: [f64; 4],
    clicks: f64,
    underruns: u64,
    media_secs: f64,
    process_secs: f64,
    /// Mid downmix of the segment (reused for identity/ride on track A).
    segment: Vec<f32>,
    segment_mid: Vec<f32>,
    channels: usize,
}

fn score_track(path: &str) -> TrackScores {
    let music = read_wav_file(path).unwrap_or_else(|e| {
        panic!("missing corpus file {path}: {e}");
    });
    assert_eq!(music.sample_rate, SAMPLE_RATE, "expected 44.1 kHz corpus");
    let ch = music.channels.count();
    let skip = MUSIC_SKIP_SECS * SAMPLE_RATE as usize * ch;
    let take = MUSIC_SECS * SAMPLE_RATE as usize * ch;
    let segment: Vec<f32> = music.data[skip..skip + take].to_vec();
    let segment_mid = downmix_to_mid(&segment, ch);
    let warmup = (WARMUP_SECS * SAMPLE_RATE as f64) as usize;

    let rates = [RATE_SLOW, RATE_FAST, RATE_SLOW2, RATE_FAST2];
    let mut spec = [0.0f64; 4];
    let mut tf1 = [0.0f64; 4];
    let mut clicks = 0.0f64;
    let mut underruns = 0u64;
    let mut process_secs = 0.0f64;
    for (i, &rate) in rates.iter().enumerate() {
        let render = render_stream(&segment, ch, &move |_| rate);
        let mid = downmix_to_mid(&render.output, ch);
        spec[i] = mean_spectral_similarity(&segment_mid, &mid[warmup..], FFT_SIZE, HOP_SIZE);
        let (f1, _, _) = transient_f1(&segment_mid, &mid, rate, render.latency_frames);
        tf1[i] = f1;
        clicks = clicks.max(clicks_per_million(&segment_mid, &mid));
        underruns += render.underrun_frames;
        process_secs += render.process_secs;
    }
    TrackScores {
        spec,
        tf1,
        clicks,
        underruns,
        media_secs: 4.0 * MUSIC_SECS as f64,
        process_secs,
        segment,
        segment_mid,
        channels: ch,
    }
}

fn main() {
    let warmup = (WARMUP_SECS * SAMPLE_RATE as f64) as usize;

    // --- Scored stretch renders on both tracks ---
    let a = score_track(MUSIC_PATH);
    let b = score_track(MUSIC_PATH_B);
    let c = score_track(MUSIC_PATH_C);
    let d = score_track(MUSIC_PATH_D);

    // --- Track A extras: identity, music ride, sine ride ---
    let ride = |t: f64| 1.0 + 0.08 * (2.0 * std::f64::consts::PI * 0.25 * t).sin();
    let ch = a.channels;
    let identity = render_stream(&a.segment, ch, &|_| 1.0);
    let music_ride = render_stream(&a.segment, ch, &ride);
    let sine: Vec<f32> = (0..SAMPLE_RATE as usize * 10)
        .map(|i| {
            0.5 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / SAMPLE_RATE as f64).sin() as f32
        })
        .collect();
    let sine_ride = render_stream(&sine, 1, &ride);

    let identity_mid = downmix_to_mid(&identity.output, ch);
    let ride_mid = downmix_to_mid(&music_ride.output, ch);

    // Identity transparency: latency-aligned frame-wise magnitude spectral
    // similarity (phase-blind: the LR8 crossover re-sums to allpass, so
    // waveform correlation would punish inaudible phase).
    let identity_corr = {
        let lat = identity.latency_frames;
        let n = (identity_mid.len() - lat).min(a.segment_mid.len()) - warmup;
        let sa = &a.segment_mid[warmup..warmup + n];
        let sb = &identity_mid[warmup + lat..warmup + lat + n];
        spectral_similarity(sa, sb, 2048, 512)
    };

    // Pitch stability.
    let (p95_cents, max_cents) = cents_stats(&sine_ride.output, 440.0);

    // Ride timbre preservation (monitored, not scored): timing-invariant
    // mean-spectrum similarity survives the time-varying rate map.
    let ride_spec =
        mean_spectral_similarity(&a.segment_mid, &ride_mid[warmup..], FFT_SIZE, HOP_SIZE);

    // Clicks (worst across every music render).
    let clicks = a
        .clicks
        .max(b.clicks)
        .max(c.clicks)
        .max(d.clicks)
        .max(clicks_per_million(&a.segment_mid, &ride_mid));

    let underruns = a.underruns
        + b.underruns
        + c.underruns
        + d.underruns
        + identity.underrun_frames
        + music_ride.underrun_frames
        + sine_ride.underrun_frames;

    let media_secs = a.media_secs + b.media_secs + c.media_secs + MUSIC_SECS as f64;
    let proc_secs = a.process_secs + b.process_secs + c.process_secs + music_ride.process_secs;
    let realtime_x = media_secs / proc_secs;

    // --- Composite quality score (0-100, higher is better) ---
    let mean4 = |x: &[f64; 4]| 0.25 * (x[0] + x[1] + x[2] + x[3]);
    let spec_score = (mean4(&a.spec) + mean4(&b.spec) + mean4(&c.spec) + mean4(&d.spec)) / 4.0;
    let transient_score = (mean4(&a.tf1) + mean4(&b.tf1) + mean4(&c.tf1) + mean4(&d.tf1)) / 4.0;
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
        "A spec slow={:.4} fast={:.4} slow2={:.4} fast2={:.4}",
        a.spec[0], a.spec[1], a.spec[2], a.spec[3]
    );
    println!(
        "A tf1  slow={:.4} fast={:.4} slow2={:.4} fast2={:.4}",
        a.tf1[0], a.tf1[1], a.tf1[2], a.tf1[3]
    );
    println!(
        "B spec slow={:.4} fast={:.4} slow2={:.4} fast2={:.4}",
        b.spec[0], b.spec[1], b.spec[2], b.spec[3]
    );
    println!(
        "B tf1  slow={:.4} fast={:.4} slow2={:.4} fast2={:.4}",
        b.tf1[0], b.tf1[1], b.tf1[2], b.tf1[3]
    );
    println!(
        "C spec slow={:.4} fast={:.4} slow2={:.4} fast2={:.4}",
        c.spec[0], c.spec[1], c.spec[2], c.spec[3]
    );
    println!(
        "C tf1  slow={:.4} fast={:.4} slow2={:.4} fast2={:.4}",
        c.tf1[0], c.tf1[1], c.tf1[2], c.tf1[3]
    );
    println!(
        "D spec slow={:.4} fast={:.4} slow2={:.4} fast2={:.4}",
        d.spec[0], d.spec[1], d.spec[2], d.spec[3]
    );
    println!(
        "D tf1  slow={:.4} fast={:.4} slow2={:.4} fast2={:.4}",
        d.tf1[0], d.tf1[1], d.tf1[2], d.tf1[3]
    );
    println!("identity_corr={identity_corr:.4} (latency-aligned frame-wise spectral sim)");
    println!("pitch p95={p95_cents:.2}c max={max_cents:.2}c");
    println!("clicks/M={clicks:.1} underruns={underruns}");
    println!();
    println!("METRIC quality={quality:.3}");
    println!("METRIC spec_sim={spec_score:.4}");
    println!("METRIC transient_f1={transient_score:.4}");
    println!("METRIC pitch_p95_cents={p95_cents:.2}");
    println!("METRIC pitch_max_cents={max_cents:.2}");
    println!("METRIC identity_corr={identity_corr:.4}");
    println!("METRIC clicks_per_m={clicks:.1}");
    println!("METRIC underruns={underruns}");
    println!("METRIC realtime_x={realtime_x:.1}");
    println!("METRIC ride_spec={ride_spec:.4}");
}

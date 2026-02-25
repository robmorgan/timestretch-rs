//! Audio quality benchmark comparing library output against a professional reference.
//!
//! Loads an original WAV, stretches it with the DjBeatmatch preset, writes
//! the result, and compares against an Ableton Live 11 Complex Pro reference
//! using spectral similarity, band-level similarity, cross-correlation, and
//! transient match scoring.
//!
//! Uses a windowed approach: compares multiple short segments independently
//! to avoid cumulative timing drift between different algorithms, then
//! averages the results.
//!
//! Run with: cargo run --release --example benchmark_quality
//! Self-test: cargo run --release --example benchmark_quality -- --self-test

use timestretch::analysis::comparison::{
    cross_correlation, mean_band_spectral_similarity, mean_spectral_similarity,
    spectral_similarity, transient_match_score_with_params, BandSimilarity,
};
use timestretch::io::wav::{read_wav_file, write_wav_file_16bit};
use timestretch::{EdmPreset, StretchParams};

const ORIGINAL_PATH: &str =
    "benchmarks/audio/12247392_Music Sounds Better With You_(Original Mix)_124bpm.wav";
const REFERENCE_PATH: &str =
    "benchmarks/audio/12247392_Music Sounds Better With You_(Original Mix)_115bpm.wav";
const OUTPUT_PATH: &str =
    "benchmarks/audio/12247392_Music Sounds Better With You_(Original Mix)_115bpm_library.wav";

const SOURCE_BPM: f64 = 124.0;
const TARGET_BPM: f64 = 115.0;

const FFT_SIZE: usize = 4096;
const HOP_SIZE: usize = 1024;
const TRANSIENT_TOLERANCE_MS: f64 = 15.0;

/// Transient detection parameters matching DjBeatmatch preset.
const TRANSIENT_FFT_SIZE: usize = 2048;
const TRANSIENT_HOP_SIZE: usize = 512;
const TRANSIENT_SENSITIVITY: f32 = 0.45;

/// Duration of each comparison window in seconds.
const SEGMENT_SECS: usize = 30;
/// How far into the track to start comparing (skip intro silence/ramp).
const SKIP_SECS: usize = 10;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let self_test = args.iter().any(|a| a == "--self-test");

    if self_test {
        run_self_test();
        return;
    }

    println!("=== Timestretch Quality Benchmark ===\n");

    // --- Load audio files ---
    println!("Loading original: {ORIGINAL_PATH}");
    let original = read_wav_file(ORIGINAL_PATH).expect("Failed to load original WAV");
    println!(
        "  {} samples, {}Hz, {:?}, {:.1}s",
        original.data.len(),
        original.sample_rate,
        original.channels,
        original.data.len() as f64
            / (original.sample_rate as f64 * original.channels.count() as f64)
    );

    println!("Loading reference: {REFERENCE_PATH}");
    let reference = read_wav_file(REFERENCE_PATH).expect("Failed to load reference WAV");
    println!(
        "  {} samples, {}Hz, {:?}, {:.1}s",
        reference.data.len(),
        reference.sample_rate,
        reference.channels,
        reference.data.len() as f64
            / (reference.sample_rate as f64 * reference.channels.count() as f64)
    );

    // --- Stretch the original ---
    // Compute the actual stretch ratio from the reference file's duration,
    // rather than from declared BPMs. This ensures we compare at the same
    // stretch ratio Ableton actually used.
    let orig_duration = original.data.len() as f64
        / (original.sample_rate as f64 * original.channels.count() as f64);
    let ref_duration_secs = reference.data.len() as f64
        / (reference.sample_rate as f64 * reference.channels.count() as f64);
    let ratio = ref_duration_secs / orig_duration;
    let effective_target_bpm = SOURCE_BPM / ratio;
    println!(
        "\nStretching at ratio {ratio:.4} (matching reference duration {ref_duration_secs:.1}s)"
    );
    println!(
        "  Declared: {SOURCE_BPM} -> {TARGET_BPM} BPM, Effective: {SOURCE_BPM} -> {effective_target_bpm:.1} BPM"
    );

    let params = StretchParams::new(ratio)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_sample_rate(original.sample_rate)
        .with_channels(original.channels.count() as u32);

    let start = std::time::Instant::now();
    let output = timestretch::stretch_buffer(&original, &params).expect("Stretch failed");
    let elapsed = start.elapsed();

    println!(
        "  Done in {:.2}s ({:.1}x realtime)",
        elapsed.as_secs_f64(),
        (original.data.len() as f64
            / (original.sample_rate as f64 * original.channels.count() as f64))
            / elapsed.as_secs_f64()
    );
    println!(
        "  Output: {} samples ({:.1}s)",
        output.data.len(),
        output.data.len() as f64 / (output.sample_rate as f64 * output.channels.count() as f64)
    );

    // --- Write output ---
    println!("\nWriting output: {OUTPUT_PATH}");
    write_wav_file_16bit(OUTPUT_PATH, &output).expect("Failed to write output WAV");

    // --- Resample reference to match output sample rate if needed ---
    let reference = if reference.sample_rate != output.sample_rate {
        println!(
            "\nResampling reference from {}Hz to {}Hz...",
            reference.sample_rate, output.sample_rate
        );
        let resampled = reference.resample(output.sample_rate);
        println!(
            "  Resampled: {} samples ({:.1}s)",
            resampled.data.len(),
            resampled.data.len() as f64
                / (resampled.sample_rate as f64 * resampled.channels.count() as f64)
        );
        resampled
    } else {
        reference
    };

    // --- Compare against reference ---
    let sample_rate = output.sample_rate;
    let out_duration =
        output.data.len() as f64 / (output.sample_rate as f64 * output.channels.count() as f64);
    let ref_duration = reference.data.len() as f64
        / (reference.sample_rate as f64 * reference.channels.count() as f64);

    if (ref_duration - out_duration).abs() > 1.0 {
        println!(
            "\n  NOTE: Reference duration ({:.1}s) differs from output ({:.1}s) by {:.1}s.",
            ref_duration,
            out_duration,
            (ref_duration - out_duration).abs()
        );
        println!("  Comparison will use the shorter duration.");
    }

    // Convert to mono for comparison
    let ref_mono = to_mono(&reference.data, reference.channels.count());
    let out_mono = to_mono(&output.data, output.channels.count());

    run_comparison(&ref_mono, &out_mono, sample_rate);
}

/// Runs the self-test: benchmarks library output against itself.
/// All scores should be ~1.0; if not, the comparison metrics have bugs.
fn run_self_test() {
    println!("=== Self-Test Mode ===");
    println!("Benchmarking library output against itself (scores should be ~1.0)\n");

    println!("Loading original: {ORIGINAL_PATH}");
    let original = read_wav_file(ORIGINAL_PATH).expect("Failed to load original WAV");

    let ratio = SOURCE_BPM / TARGET_BPM;
    println!("Stretching at ratio {ratio:.4}...");

    let params = StretchParams::new(ratio)
        .with_preset(EdmPreset::DjBeatmatch)
        .with_sample_rate(original.sample_rate)
        .with_channels(original.channels.count() as u32);

    let output = timestretch::stretch_buffer(&original, &params).expect("Stretch failed");
    let out_mono = to_mono(&output.data, output.channels.count());

    println!("Comparing output against itself...\n");
    run_comparison(&out_mono, &out_mono, output.sample_rate);
}

/// Runs the windowed comparison between reference and test signals.
///
/// Uses two complementary spectral metrics:
/// - Frame-by-frame: temporal alignment sensitive, captures fine detail
/// - Mean spectral shape: timing-invariant, captures overall frequency balance
///
/// The overall score uses the mean spectral shape since it's robust to the
/// non-uniform timing differences between different stretching algorithms.
fn run_comparison(ref_mono: &[f32], out_mono: &[f32], sample_rate: u32) {
    let seg_samples = SEGMENT_SECS * sample_rate as usize;
    let skip_samples = SKIP_SECS * sample_rate as usize;
    let usable_len = ref_mono.len().min(out_mono.len());

    if usable_len <= skip_samples + seg_samples {
        eprintln!("ERROR: Not enough audio for comparison after skipping {SKIP_SECS}s intro.");
        std::process::exit(1);
    }

    let num_segments = (usable_len - skip_samples) / seg_samples;
    println!(
        "Comparing {} segments of {}s each (skipping first {}s)...\n",
        num_segments, SEGMENT_SECS, SKIP_SECS
    );

    let mut mean_spec_sims = Vec::new();
    let mut frame_spec_sims = Vec::new();
    let mut band_sims: Vec<BandSimilarity> = Vec::new();
    let mut xcorr_peaks = Vec::new();
    let mut transient_matched = 0u32;
    let mut transient_ref_total = 0u32;
    let mut transient_test_total = 0u32;

    for seg_idx in 0..num_segments {
        let start = skip_samples + seg_idx * seg_samples;
        let end = start + seg_samples;
        if end > usable_len {
            break;
        }

        let ref_seg = &ref_mono[start..end];
        let out_seg = &out_mono[start..end];

        // Mean spectral shape (timing-invariant).
        let ms = mean_spectral_similarity(ref_seg, out_seg, FFT_SIZE, HOP_SIZE);
        // Frame-by-frame spectral (timing-sensitive, for diagnostic).
        let ss = spectral_similarity(ref_seg, out_seg, FFT_SIZE, HOP_SIZE);
        // Per-band mean spectral (timing-invariant).
        let bs = mean_band_spectral_similarity(ref_seg, out_seg, FFT_SIZE, HOP_SIZE, sample_rate);
        let tm = transient_match_score_with_params(
            ref_seg,
            out_seg,
            sample_rate,
            TRANSIENT_TOLERANCE_MS,
            TRANSIENT_FFT_SIZE,
            TRANSIENT_HOP_SIZE,
            TRANSIENT_SENSITIVITY,
        );
        let xc = cross_correlation(ref_seg, out_seg);

        println!(
            "  Seg {:>2}: mean={:.3} frame={:.3} sub={:.3} low={:.3} mid={:.3} hi={:.3} xcorr={:.3} trans={}/{}",
            seg_idx + 1,
            ms,
            ss,
            bs.sub_bass,
            bs.low,
            bs.mid,
            bs.high,
            xc.peak_value,
            tm.matched,
            tm.total_reference,
        );

        mean_spec_sims.push(ms);
        frame_spec_sims.push(ss);
        band_sims.push(bs);
        xcorr_peaks.push(xc.peak_value);
        transient_matched += tm.matched as u32;
        transient_ref_total += tm.total_reference as u32;
        transient_test_total += tm.total_test as u32;
    }

    // Average results
    let n = mean_spec_sims.len() as f64;
    let avg_mean_spec = mean_spec_sims.iter().sum::<f64>() / n;
    let avg_frame_spec = frame_spec_sims.iter().sum::<f64>() / n;
    let avg_sub = band_sims.iter().map(|b| b.sub_bass).sum::<f64>() / n;
    let avg_low = band_sims.iter().map(|b| b.low).sum::<f64>() / n;
    let avg_mid = band_sims.iter().map(|b| b.mid).sum::<f64>() / n;
    let avg_high = band_sims.iter().map(|b| b.high).sum::<f64>() / n;
    let avg_xcorr = xcorr_peaks.iter().sum::<f64>() / n;
    let avg_trans = if transient_ref_total > 0 {
        transient_matched as f64 / transient_ref_total as f64
    } else {
        0.0
    };

    // --- Print report ---
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║         QUALITY BENCHMARK REPORT             ║");
    println!(
        "║  ({} x {}s segments, {}Hz)               ║",
        mean_spec_sims.len(),
        SEGMENT_SECS,
        sample_rate
    );
    println!("╠══════════════════════════════════════════════╣");
    println!("║                                              ║");
    println!("║  Spectral Shape (timing-invariant):          ║");
    println!(
        "║    Mean Spectral:      {:>6.4}  {}  ║",
        avg_mean_spec,
        grade(avg_mean_spec)
    );
    println!("║                                              ║");
    println!("║  Spectral Detail (timing-sensitive):         ║");
    println!(
        "║    Frame-by-frame:     {:>6.4}  {}  ║",
        avg_frame_spec,
        grade(avg_frame_spec)
    );
    println!("║                                              ║");
    println!("║  Band Similarity:                            ║");
    println!(
        "║    Sub-bass:           {:>6.4}  {}  ║",
        avg_sub,
        grade(avg_sub)
    );
    println!(
        "║    Low:                {:>6.4}  {}  ║",
        avg_low,
        grade(avg_low)
    );
    println!(
        "║    Mid:                {:>6.4}  {}  ║",
        avg_mid,
        grade(avg_mid)
    );
    println!(
        "║    High:               {:>6.4}  {}  ║",
        avg_high,
        grade(avg_high)
    );
    println!("║                                              ║");
    println!(
        "║  Cross-Correlation:    {:>6.4}  {}  ║",
        avg_xcorr,
        grade(avg_xcorr)
    );
    println!("║                                              ║");
    println!(
        "║  Transient Match:      {:>6.4}  {}  ║",
        avg_trans,
        grade(avg_trans)
    );
    println!(
        "║    Matched: {:>3} / {:>3} ref, {:>3} test         ║",
        transient_matched, transient_ref_total, transient_test_total
    );
    println!("║                                              ║");
    println!("╚══════════════════════════════════════════════╝");

    // Overall score uses timing-invariant metrics since different algorithms
    // inherently produce different temporal placement of events.
    let overall = (avg_mean_spec + avg_sub + avg_low + avg_mid + avg_high) / 5.0;
    println!(
        "\nOverall score: {:.4} {} (spectral shape weighted)",
        overall,
        grade(overall)
    );
    println!(
        "Timing score:  {:.4} {} (cross-correlation, diagnostic only)",
        avg_xcorr,
        grade(avg_xcorr)
    );
}

/// Converts interleaved multi-channel audio to mono by averaging channels.
fn to_mono(data: &[f32], num_channels: usize) -> Vec<f32> {
    if num_channels == 1 {
        return data.to_vec();
    }
    let num_frames = data.len() / num_channels;
    let inv = 1.0 / num_channels as f32;
    (0..num_frames)
        .map(|f| {
            let start = f * num_channels;
            (0..num_channels).map(|ch| data[start + ch]).sum::<f32>() * inv
        })
        .collect()
}

/// Returns a letter grade for a similarity score.
fn grade(score: f64) -> &'static str {
    if score >= 0.95 {
        "[A+]"
    } else if score >= 0.90 {
        "[A] "
    } else if score >= 0.85 {
        "[B+]"
    } else if score >= 0.80 {
        "[B] "
    } else if score >= 0.70 {
        "[C] "
    } else if score >= 0.60 {
        "[D] "
    } else {
        "[F] "
    }
}

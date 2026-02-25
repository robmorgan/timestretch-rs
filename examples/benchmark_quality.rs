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
    band_spectral_similarity, cross_correlation, spectral_similarity,
    transient_match_score_with_params, BandSimilarity,
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
    let ratio = SOURCE_BPM / TARGET_BPM;
    println!("\nStretching at ratio {ratio:.4} ({SOURCE_BPM} -> {TARGET_BPM} BPM)...");

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
    let expected_duration = original.data.len() as f64
        / (original.sample_rate as f64 * original.channels.count() as f64)
        * ratio;
    let ref_duration = reference.data.len() as f64
        / (reference.sample_rate as f64 * reference.channels.count() as f64);

    if (ref_duration - expected_duration).abs() > 1.0 {
        println!(
            "\n  NOTE: Reference duration ({:.1}s) differs from expected ({:.1}s).",
            ref_duration, expected_duration
        );
        println!("  The Ableton reference may have been trimmed or exported differently.");
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
        "Comparing {} segments of {}s each (skipping first {}s)...",
        num_segments, SEGMENT_SECS, SKIP_SECS
    );

    let mut spec_sims = Vec::new();
    let mut band_sims: Vec<BandSimilarity> = Vec::new();
    let mut xcorr_peaks = Vec::new();
    let mut transient_matched = 0u32;
    let mut transient_ref_total = 0u32;
    let mut transient_test_total = 0u32;
    let mut drift_offsets: Vec<isize> = Vec::new();

    for seg_idx in 0..num_segments {
        let seg_start = skip_samples + seg_idx * seg_samples;
        let seg_end = seg_start + seg_samples;
        if seg_end > usable_len {
            break;
        }

        let ref_seg = &ref_mono[seg_start..seg_end];
        let out_seg = &out_mono[seg_start..seg_end];

        // Align this segment using cross-correlation
        let xcorr = cross_correlation(ref_seg, out_seg);
        drift_offsets.push(xcorr.peak_offset);

        // Apply alignment within this segment
        let (r, o) = if xcorr.peak_offset.unsigned_abs() < seg_samples / 2 {
            if xcorr.peak_offset > 0 {
                let off = xcorr.peak_offset as usize;
                let len = seg_samples - off;
                (&ref_seg[..len], &out_seg[off..off + len])
            } else if xcorr.peak_offset < 0 {
                let off = (-xcorr.peak_offset) as usize;
                let len = seg_samples - off;
                (&ref_seg[off..off + len], &out_seg[..len])
            } else {
                (ref_seg, out_seg)
            }
        } else {
            // Offset too large — skip alignment for this segment
            (ref_seg, out_seg)
        };

        let ss = spectral_similarity(r, o, FFT_SIZE, HOP_SIZE);
        let bs = band_spectral_similarity(r, o, FFT_SIZE, HOP_SIZE, sample_rate);
        let tm = transient_match_score_with_params(
            r,
            o,
            sample_rate,
            TRANSIENT_TOLERANCE_MS,
            TRANSIENT_FFT_SIZE,
            TRANSIENT_HOP_SIZE,
            TRANSIENT_SENSITIVITY,
        );

        // Cross-correlation on aligned segment (check alignment quality)
        let xc = cross_correlation(r, o);

        println!(
            "  Seg {:>2}: spec={:.3} sub={:.3} low={:.3} mid={:.3} hi={:.3} xcorr={:.3} trans={}/{} drift={:+}",
            seg_idx + 1,
            ss,
            bs.sub_bass,
            bs.low,
            bs.mid,
            bs.high,
            xc.peak_value,
            tm.matched,
            tm.total_reference,
            xcorr.peak_offset,
        );

        spec_sims.push(ss);
        band_sims.push(bs);
        xcorr_peaks.push(xc.peak_value);
        transient_matched += tm.matched as u32;
        transient_ref_total += tm.total_reference as u32;
        transient_test_total += tm.total_test as u32;
    }

    // Average results
    let n = spec_sims.len() as f64;
    let avg_spec = spec_sims.iter().sum::<f64>() / n;
    let avg_sub = band_sims.iter().map(|b| b.sub_bass).sum::<f64>() / n;
    let avg_low = band_sims.iter().map(|b| b.low).sum::<f64>() / n;
    let avg_mid = band_sims.iter().map(|b| b.mid).sum::<f64>() / n;
    let avg_high = band_sims.iter().map(|b| b.high).sum::<f64>() / n;
    let avg_band = band_sims.iter().map(|b| b.overall).sum::<f64>() / n;
    let avg_xcorr = xcorr_peaks.iter().sum::<f64>() / n;
    let avg_trans = if transient_ref_total > 0 {
        transient_matched as f64 / transient_ref_total as f64
    } else {
        0.0
    };

    // --- Drift diagnostics ---
    println!("\n--- Timing Drift Diagnostics ---");
    if drift_offsets.len() >= 2 {
        // Linear regression: slope in samples/segment
        let n_pts = drift_offsets.len() as f64;
        let sum_x: f64 = (0..drift_offsets.len()).map(|i| i as f64).sum();
        let sum_y: f64 = drift_offsets.iter().map(|&o| o as f64).sum();
        let sum_xy: f64 = drift_offsets
            .iter()
            .enumerate()
            .map(|(i, &o)| i as f64 * o as f64)
            .sum();
        let sum_x2: f64 = (0..drift_offsets.len()).map(|i| (i * i) as f64).sum();

        let slope = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x * sum_x);
        let samples_per_sec = slope / SEGMENT_SECS as f64;

        for (i, &offset) in drift_offsets.iter().enumerate() {
            let time_ms = offset as f64 * 1000.0 / sample_rate as f64;
            println!(
                "  Seg {:>2}: offset = {:>6} samples ({:>+.1}ms)",
                i + 1,
                offset,
                time_ms
            );
        }

        println!(
            "\n  Drift trend: {:.2} samples/segment ({:.2} samples/sec)",
            slope, samples_per_sec
        );
        if samples_per_sec.abs() > 1.0 {
            println!("  WARNING: Progressive drift detected. Low cross-correlation may be");
            println!("  caused by cumulative timing error, not transient quality issues.");
        } else {
            println!("  Drift is minimal (< 1 sample/sec). Timing is stable.");
        }
    }

    // --- Print report ---
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║         QUALITY BENCHMARK REPORT             ║");
    println!(
        "║  ({} x {}s segments, {}Hz)               ║",
        num_segments, SEGMENT_SECS, sample_rate
    );
    println!("╠══════════════════════════════════════════════╣");
    println!("║                                              ║");
    println!(
        "║  Spectral Similarity:  {:>6.4}  {}  ║",
        avg_spec,
        grade(avg_spec)
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
    println!(
        "║    Overall:            {:>6.4}  {}  ║",
        avg_band,
        grade(avg_band)
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

    // Summary
    let overall = (avg_spec + avg_band + avg_xcorr + avg_trans) / 4.0;
    println!("\nOverall score: {:.4} {}", overall, grade(overall));
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
